"""RAG orchestration for chat.

ask() pipeline:
  1. condense  – multi-turn: rewrite follow-up into a standalone query (fast model)
  2. cache     – identical (docs, question) answered recently? return it
  3. retrieve  – embed query, hybrid search over the session's document versions
  4. generate  – grounded answer with numbered citations (sync or streamed)
  5. verify    – keep only citations that reference real passages; drop hallucinated ones
  6. follow-up – 3 suggested next questions (fast model, best-effort)

Every LLM/embedding call yields a Usage that the caller persists.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings import aembed_query
from app.ai.prompts import chat_messages, condense_messages, followup_messages, format_context
from app.ai.providers import Usage, get_llm, parse_json_response
from app.ai.rerank import rerank
from app.ai.retrieval import RetrievedChunk, hybrid_search, trim_to_budget
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

_CITE = re.compile(r"\[(\d+)\]")
NO_ANSWER = "I couldn't find anything relevant in the selected documents. Try rephrasing, or check that the documents have finished processing."


@dataclass
class Citation:
    n: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_start: int | None
    page_end: int | None
    section: str | None
    snippet: str
    score: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "filename": self.filename,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "snippet": self.snippet,
            "score": round(self.score, 5),
        }


@dataclass
class Answer:
    text: str
    standalone_question: str
    citations: list[Citation]
    follow_ups: list[str]
    usages: list[Usage] = field(default_factory=list)
    cached: bool = False
    retrieved: list[RetrievedChunk] = field(default_factory=list)


def _label(c: RetrievedChunk, names: dict[uuid.UUID, str]) -> str:
    name = names.get(c.document_id, "document")
    if c.page_start and c.page_end and c.page_end != c.page_start:
        return f"{name}, pp.{c.page_start}-{c.page_end}"
    if c.page_start:
        return f"{name}, p.{c.page_start}"
    return name


def build_citations(chunks: list[RetrievedChunk], names: dict[uuid.UUID, str]) -> list[Citation]:
    return [
        Citation(
            n=i,
            chunk_id=c.id,
            document_id=c.document_id,
            filename=names.get(c.document_id, "document"),
            page_start=c.page_start,
            page_end=c.page_end,
            section=c.section,
            snippet=c.content[:240].strip(),
            score=c.score,
        )
        for i, c in enumerate(chunks, start=1)
    ]


def verify_citations(answer: str, citations: list[Citation]) -> tuple[str, list[Citation]]:
    """Remove citation markers that point at passages we never provided and
    return only the citations that were actually used, renumbered densely."""
    valid = {c.n for c in citations}
    used: list[int] = []
    for m in _CITE.finditer(answer):
        n = int(m.group(1))
        if n in valid and n not in used:
            used.append(n)
    remap = {old: new for new, old in enumerate(used, start=1)}

    def _sub(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[{remap[n]}]" if n in remap else ""

    cleaned = _CITE.sub(_sub, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    kept = []
    for c in citations:
        if c.n in remap:
            c.n = remap[c.n]
            kept.append(c)
    kept.sort(key=lambda c: c.n)
    return cleaned, kept


async def condense_question(history: list[tuple[str, str]], question: str) -> tuple[str, Usage | None]:
    if not history:
        return question, None
    try:
        c = await get_llm(fast=True).acomplete(condense_messages(history, question), temperature=0.0, max_tokens=200, json_mode=True)
        q = str(parse_json_response(c.text).get("standalone_question") or question).strip()
        return q or question, c.usage
    except Exception as e:  # best-effort; fall back to raw question
        log.warning("condense_failed", error=str(e))
        return question, None


async def suggest_follow_ups(question: str, answer: str, context: str) -> tuple[list[str], Usage | None]:
    try:
        c = await get_llm(fast=True).acomplete(followup_messages(question, answer, context), temperature=0.7, max_tokens=250, json_mode=True)
        qs = [str(q).strip() for q in parse_json_response(c.text).get("questions", []) if str(q).strip()]
        return qs[:3], c.usage
    except Exception as e:
        log.warning("followup_failed", error=str(e))
        return [], None


async def retrieve(db: AsyncSession, *, owner_id: str, version_ids: list[uuid.UUID], question: str) -> tuple[list[RetrievedChunk], list[Usage]]:
    """Embed → hybrid search → (optional) rerank → token budget. Returns chunks and
    every Usage incurred (embedding, and reranker if it called a model)."""
    s = get_settings()
    vec, usage = await aembed_query(question)
    usages = [usage]
    if s.rerank_enabled:
        candidates = await hybrid_search(
            db, owner_id=owner_id, version_ids=version_ids, query_text=question, query_vector=vec, top_k=s.rerank_candidates, trim=False
        )
        ranked, rr_usage = await rerank(question, candidates, s.retrieval_top_k)
        if rr_usage:
            usages.append(rr_usage)
        return trim_to_budget(ranked, s.context_token_budget), usages
    chunks = await hybrid_search(db, owner_id=owner_id, version_ids=version_ids, query_text=question, query_vector=vec)
    return chunks, usages


def prepare_generation(chunks: list[RetrievedChunk], names: dict[uuid.UUID, str], history: list[tuple[str, str]], question: str):
    citations = build_citations(chunks, names)
    context = format_context([(c.n, _label(ch, names), ch.content) for c, ch in zip(citations, chunks, strict=True)])
    messages = chat_messages(context, history[-8:], question)
    return citations, context, messages


async def generate(messages, citations: list[Citation]) -> tuple[str, list[Citation], Usage]:
    c = await get_llm().acomplete(messages, temperature=0.2, max_tokens=900)
    text, kept = verify_citations(c.text, citations)
    return text, kept, c.usage


async def generate_stream(messages) -> AsyncIterator[str | Usage]:
    async for item in get_llm().astream(messages, temperature=0.2, max_tokens=900):
        yield item
