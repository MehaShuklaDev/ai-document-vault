"""Document chat endpoints: sessions, ask (JSON or SSE stream), history."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.ai import rag
from app.ai.providers import LLMError, Usage, get_embeddings, get_llm
from app.ai.usage import arecord_usage
from app.api.deps import chat_rate_limit, get_current_user, quota_check
from app.core.config import get_settings
from app.core.db import get_async_sessionmaker, get_db
from app.core.logging import get_logger
from app.models import ChatMessage, ChatSession, Document, DocumentStatus
from app.schemas import AskRequest, AskResponse, HistoryOut, MessageOut, SessionCreate, SessionOut
from app.services import cache

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
async def _get_session(db: AsyncSession, session_id: uuid.UUID, owner_id: str) -> ChatSession:
    s = (await db.execute(select(ChatSession).where(ChatSession.id == session_id, ChatSession.owner_id == owner_id))).scalar_one_or_none()
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    return s


async def _ready_documents(db: AsyncSession, owner_id: str, document_ids: list[uuid.UUID]) -> list[Document]:
    docs = (
        (await db.execute(select(Document).where(Document.id.in_(document_ids), Document.owner_id == owner_id, Document.deleted_at.is_(None)))).scalars().all()
    )
    return list(docs)


async def _history(db: AsyncSession, session_id: uuid.UUID, limit: int = 10) -> list[tuple[str, str]]:
    rows = (await db.execute(select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.seq.desc()).limit(limit))).scalars().all()
    return [(m.role, m.content) for m in reversed(rows) if m.role in ("user", "assistant")]


async def _next_seq(db: AsyncSession, session_id: uuid.UUID) -> int:
    return ((await db.execute(select(func.max(ChatMessage.seq)).where(ChatMessage.session_id == session_id))).scalar_one() or 0) + 1


def _sum_usage(usages: list[Usage | None]) -> dict:
    us = [u for u in usages if u]
    return {
        "input_tokens": sum(u.input_tokens for u in us),
        "output_tokens": sum(u.output_tokens for u in us),
        "cost_usd": round(sum(u.cost_usd for u in us), 8),
        "latency_ms": sum(u.latency_ms for u in us),
        "calls": [u.as_dict() for u in us],
    }


class _Prepared:
    def __init__(self, session: ChatSession, docs: list[Document], version_ids: list[uuid.UUID], names: dict, history: list[tuple[str, str]]):
        self.session, self.docs, self.version_ids, self.names, self.history = session, docs, version_ids, names, history


async def _prepare(db: AsyncSession, session_id: uuid.UUID, owner_id: str) -> _Prepared:
    session = await _get_session(db, session_id, owner_id)
    docs = await _ready_documents(db, owner_id, list(session.document_ids))
    ready = [d for d in docs if d.status == DocumentStatus.ready and d.current_version_id]
    if not ready:
        pending = [d for d in docs if d.status in (DocumentStatus.uploaded, DocumentStatus.processing)]
        msg = "documents are still processing; try again shortly" if pending else "no ready documents in this session"
        raise HTTPException(status.HTTP_409_CONFLICT, msg)
    return _Prepared(session, ready, [d.current_version_id for d in ready], {d.id: d.filename for d in ready}, await _history(db, session.id))


async def _persist_turn(
    db: AsyncSession,
    prep: _Prepared,
    question: str,
    standalone: str,
    answer_text: str,
    citations: list[dict],
    follow_ups: list[str],
    usages: list[Usage | None],
    cached: bool,
    owner_id: str,
) -> tuple[ChatMessage, ChatMessage]:
    seq = await _next_seq(db, prep.session.id)
    q_msg = ChatMessage(session_id=prep.session.id, seq=seq, role="user", content=question, standalone_question=standalone if standalone != question else None)
    usage = _sum_usage(usages)
    usage["cached"] = cached
    a_msg = ChatMessage(session_id=prep.session.id, seq=seq + 1, role="assistant", content=answer_text, citations=citations, follow_ups=follow_ups, usage=usage)
    db.add_all([q_msg, a_msg])
    for u in usages:
        if u:
            await arecord_usage(db, u, "chat", owner_id=owner_id, session_id=prep.session.id)
    prep.session.message_count += 2
    prep.session.last_message_at = datetime.now(UTC)
    if not prep.session.title:
        prep.session.title = question[:80]
    await db.flush()
    await db.refresh(q_msg)
    await db.refresh(a_msg)
    return q_msg, a_msg


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------
@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED, summary="Start a chat session over one or more documents")
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    docs = await _ready_documents(db, user, body.document_ids)
    found = {d.id for d in docs}
    missing = [str(i) for i in body.document_ids if i not in found]
    if missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"documents not found: {', '.join(missing)}")
    title = body.title or (docs[0].title or docs[0].filename if len(docs) == 1 else f"Chat over {len(docs)} documents")
    session = ChatSession(owner_id=user, title=title[:256], document_ids=list(dict.fromkeys(body.document_ids)), settings={})
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return SessionOut.model_validate(session)


@router.get("/sessions", response_model=list[SessionOut], summary="List chat sessions")
async def list_sessions(
    limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)
):
    rows = (
        (await db.execute(select(ChatSession).where(ChatSession.owner_id == user).order_by(ChatSession.updated_at.desc()).limit(limit).offset(offset)))
        .scalars()
        .all()
    )
    return [SessionOut.model_validate(r) for r in rows]


@router.get("/sessions/{session_id}", response_model=HistoryOut, summary="Chat history (multi-turn context)")
async def get_history(session_id: uuid.UUID, limit: int = Query(50, ge=1, le=500), db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    session = await _get_session(db, session_id, user)
    rows = (await db.execute(select(ChatMessage).where(ChatMessage.session_id == session.id).order_by(ChatMessage.seq.asc()).limit(limit))).scalars().all()
    return HistoryOut(session=SessionOut.model_validate(session), messages=[MessageOut.model_validate(m) for m in rows])


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    session = await _get_session(db, session_id, user)
    await db.delete(session)


# ---------------------------------------------------------------------------
# ask (JSON)
# ---------------------------------------------------------------------------
@router.post(
    "/sessions/{session_id}/ask",
    response_model=AskResponse,
    dependencies=[Depends(chat_rate_limit), Depends(quota_check)],
    summary="Ask a question (grounded answer with citations)",
)
async def ask(session_id: uuid.UUID, body: AskRequest, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    prep = await _prepare(db, session_id, user)
    usages: list[Usage | None] = []

    standalone, u = await rag.condense_question(prep.history, body.question)
    usages.append(u)

    key = cache.answer_key(user, [str(v) for v in prep.version_ids], standalone, get_llm().model)
    hit = await cache.aget_cached_answer(key) if body.use_cache else None
    if hit:
        q_msg, a_msg = await _persist_turn(db, prep, body.question, standalone, hit["answer"], hit["citations"], hit.get("follow_ups", []), usages, True, user)
        return AskResponse(session_id=prep.session.id, question=MessageOut.model_validate(q_msg), answer=MessageOut.model_validate(a_msg), cached=True)

    chunks, us = await rag.retrieve(db, owner_id=user, version_ids=prep.version_ids, question=standalone)
    usages.extend(us)
    if not chunks:
        q_msg, a_msg = await _persist_turn(db, prep, body.question, standalone, rag.NO_ANSWER, [], [], usages, False, user)
        return AskResponse(session_id=prep.session.id, question=MessageOut.model_validate(q_msg), answer=MessageOut.model_validate(a_msg), cached=False)

    citations, context, messages = rag.prepare_generation(chunks, prep.names, prep.history, standalone)
    try:
        answer_text, kept, u = await rag.generate(messages, citations)
    except LLMError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"AI provider error: {e}") from e
    usages.append(u)

    follow_ups: list[str] = []
    if body.suggest_follow_ups:
        follow_ups, u = await rag.suggest_follow_ups(standalone, answer_text, context)
        usages.append(u)

    cit_dicts = [c.as_dict() for c in kept]
    q_msg, a_msg = await _persist_turn(db, prep, body.question, standalone, answer_text, cit_dicts, follow_ups, usages, False, user)
    if body.use_cache:
        await cache.aset_cached_answer(key, {"answer": answer_text, "citations": cit_dicts, "follow_ups": follow_ups})
    return AskResponse(session_id=prep.session.id, question=MessageOut.model_validate(q_msg), answer=MessageOut.model_validate(a_msg), cached=False)


# ---------------------------------------------------------------------------
# ask (SSE stream)
# ---------------------------------------------------------------------------
@router.post(
    "/sessions/{session_id}/ask/stream", dependencies=[Depends(chat_rate_limit), Depends(quota_check)], summary="Ask a question, streaming tokens over SSE"
)
async def ask_stream(session_id: uuid.UUID, body: AskRequest, user: str = Depends(get_current_user)):
    """Server-Sent Events. Event sequence:
    `status` → `citations` (sources, before any token) → `token`* → `follow_ups` → `done` (persisted message ids + usage) | `error`.
    """

    async def gen():
        Session = get_async_sessionmaker()
        async with Session() as db:
            try:
                prep = await _prepare(db, session_id, user)
            except HTTPException as e:
                yield {"event": "error", "data": json.dumps({"detail": e.detail, "status": e.status_code})}
                return
            usages: list[Usage | None] = []
            yield {"event": "status", "data": json.dumps({"stage": "understanding"})}
            standalone, u = await rag.condense_question(prep.history, body.question)
            usages.append(u)
            yield {"event": "status", "data": json.dumps({"stage": "retrieving", "query": standalone})}
            chunks, us = await rag.retrieve(db, owner_id=user, version_ids=prep.version_ids, question=standalone)
            usages.extend(us)
            if not chunks:
                q_msg, a_msg = await _persist_turn(db, prep, body.question, standalone, rag.NO_ANSWER, [], [], usages, False, user)
                await db.commit()
                yield {"event": "token", "data": json.dumps({"t": rag.NO_ANSWER})}
                yield {"event": "done", "data": json.dumps({"message_id": str(a_msg.id), "citations": [], "usage": a_msg.usage})}
                return
            citations, context, messages = rag.prepare_generation(chunks, prep.names, prep.history, standalone)
            yield {"event": "citations", "data": json.dumps([c.as_dict() for c in citations])}
            yield {"event": "status", "data": json.dumps({"stage": "answering"})}
            buf: list[str] = []
            gen_usage: Usage | None = None
            try:
                async for item in rag.generate_stream(messages):
                    if isinstance(item, Usage):
                        gen_usage = item
                    else:
                        buf.append(item)
                        yield {"event": "token", "data": json.dumps({"t": item})}
            except Exception as e:
                log.exception("stream_generation_failed")
                yield {"event": "error", "data": json.dumps({"detail": f"AI provider error: {e}"})}
                return
            usages.append(gen_usage)
            raw = "".join(buf)
            answer_text, kept = rag.verify_citations(raw, citations)
            follow_ups: list[str] = []
            if body.suggest_follow_ups:
                follow_ups, u = await rag.suggest_follow_ups(standalone, answer_text, context)
                usages.append(u)
                yield {"event": "follow_ups", "data": json.dumps(follow_ups)}
            cit_dicts = [c.as_dict() for c in kept]
            q_msg, a_msg = await _persist_turn(db, prep, body.question, standalone, answer_text, cit_dicts, follow_ups, usages, False, user)
            await db.commit()
            yield {"event": "done", "data": json.dumps({"message_id": str(a_msg.id), "final_text": answer_text, "citations": cit_dicts, "usage": a_msg.usage})}

    return EventSourceResponse(gen(), ping=15)


@router.get("/providers", summary="Which AI providers/models are active")
async def providers():
    s = get_settings()
    return {
        "llm": {"provider": get_llm().name, "model": get_llm().model, "fast_model": get_llm(fast=True).model, "configured": s.llm_provider},
        "embeddings": {
            "provider": get_embeddings().name,
            "model": get_embeddings().model,
            "dimensions": get_embeddings().dimensions,
            "configured": s.embedding_provider,
        },
        "resolution": "LLM_PROVIDER=auto: OPENAI_API_KEY → openai · ANTHROPIC_API_KEY → anthropic · logged-in `claude` CLI → claude-cli · else startup error",
    }
