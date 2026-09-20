"""Document management endpoints."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from app.ai.analysis import SummaryOptions, aanalyse_document, acompare_documents, build_excerpt
from app.ai.providers import LLMError
from app.ai.structured import CATEGORY_TO_SCHEMA, SCHEMAS, aextract_structured, schema_for_category
from app.ai.usage import arecord_usage
from app.api.deps import get_current_user, upload_rate_limit
from app.core.config import get_settings
from app.core.db import get_async_sessionmaker, get_db
from app.core.logging import get_logger
from app.models import Chunk, Document, DocumentInsight, DocumentStatus, DocumentVersion, JobStage, JobStatus, ProcessingJob
from app.schemas import (
    BatchUploadResult,
    CompareOut,
    CompareRequest,
    DocumentDetailOut,
    DocumentOut,
    DocumentPage,
    DocumentStatusOut,
    ExtractionOut,
    ExtractionRequest,
    InsightsOut,
    StageOut,
    SummaryRequest,
    UploadResult,
)
from app.services.extraction import sniff_mime
from app.services.storage import get_storage, sha256_bytes
from app.workers.tasks import enqueue_processing

log = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

PIPELINE_STAGES = [JobStage.extract, JobStage.chunk, JobStage.embed, JobStage.analyse]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
async def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Read at most ``limit`` bytes; reject oversize uploads before buffering everything."""
    chunks, total = [], 0
    while True:
        part = await file.read(1024 * 1024)
        if not part:
            break
        total += len(part)
        if total > limit:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"file exceeds {limit // (1024 * 1024)} MB limit")
        chunks.append(part)
    return b"".join(chunks)


async def _get_owned_document(db: AsyncSession, document_id: uuid.UUID, owner_id: str, *, with_versions: bool = False) -> Document:
    stmt = select(Document).where(Document.id == document_id, Document.owner_id == owner_id, Document.deleted_at.is_(None))
    if with_versions:
        stmt = stmt.options(selectinload(Document.versions))
    doc = (await db.execute(stmt)).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return doc


async def _ingest(db: AsyncSession, owner_id: str, file: UploadFile, replace_document_id: uuid.UUID | None) -> UploadResult:
    s = get_settings()
    data = await _read_limited(file, s.max_upload_bytes)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
    filename = (file.filename or "upload").strip()[:512]
    mime = sniff_mime(data, filename, file.content_type)
    if mime not in s.allowed_mime_types:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"unsupported file type '{mime}'; allowed: PDF, DOCX, TXT, MD")
    digest = sha256_bytes(data)

    # dedup: same bytes already owned by this user → return existing document
    existing = (
        await db.execute(
            select(Document)
            .join(DocumentVersion, DocumentVersion.document_id == Document.id)
            .where(Document.owner_id == owner_id, Document.deleted_at.is_(None), DocumentVersion.sha256 == digest)
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None and (replace_document_id is None or existing.id == replace_document_id):
        task_id = None
        if existing.status == DocumentStatus.failed:
            # same bytes, earlier attempt failed → give it another go instead of returning a dead record
            existing.status = DocumentStatus.uploaded
            existing.error = None
            await db.commit()
            task_id = _safe_enqueue(existing.id, existing.current_version_id)
            await db.commit()
        return UploadResult(document=DocumentOut.model_validate(existing), deduplicated=True, task_id=task_id)

    storage = get_storage()
    if not storage.exists(digest):
        storage.put(digest, data, mime)

    new_version = False
    if replace_document_id is not None:
        doc = await _get_owned_document(db, replace_document_id, owner_id)
        doc.version_count += 1
        doc.filename = filename
        doc.mime_type = mime
        version_no = doc.version_count
        new_version = True
    else:
        doc = Document(owner_id=owner_id, filename=filename, mime_type=mime, status=DocumentStatus.uploaded)
        db.add(doc)
        await db.flush()
        version_no = 1

    ver = DocumentVersion(document_id=doc.id, version=version_no, sha256=digest, storage_key=digest, size_bytes=len(data))
    db.add(ver)
    await db.flush()
    doc.current_version_id = ver.id
    doc.status = DocumentStatus.uploaded
    doc.error = None
    doc.processed_at = None
    db.add(ProcessingJob(document_id=doc.id, version_id=ver.id, stage=JobStage.pipeline, status=JobStatus.queued, meta={}))
    await db.commit()

    task_id = _safe_enqueue(doc.id, ver.id)
    if task_id is None:
        doc.status = DocumentStatus.failed
        doc.error = "could not enqueue processing (broker unavailable); call POST /documents/{id}/reprocess"
    else:
        log.info("document_enqueued", document_id=str(doc.id), version=version_no, bytes=len(data), mime=mime, task_id=task_id)
    await db.commit()
    await db.refresh(doc)
    return UploadResult(document=DocumentOut.model_validate(doc), deduplicated=False, new_version=new_version, task_id=task_id)


def _safe_enqueue(document_id: uuid.UUID, version_id: uuid.UUID | None) -> str | None:
    """Enqueue without letting a broker outage turn into a 500; caller records the failure."""
    if version_id is None:
        return None
    try:
        return enqueue_processing(document_id, version_id)
    except Exception as e:  # kombu.OperationalError etc.
        log.error("enqueue_failed", document_id=str(document_id), error=str(e))
        return None


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
@router.post(
    "",
    response_model=UploadResult,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(upload_rate_limit)],
    summary="Upload a document (async processing)",
)
async def upload_document(
    file: UploadFile = File(...),
    replace_document_id: uuid.UUID | None = Form(None, description="Upload as a new version of this existing document"),
    db: AsyncSession = Depends(get_db),
    user: str = Depends(get_current_user),
):
    """Accepts PDF, DOCX, TXT or MD. Returns **202** immediately; poll
    `GET /documents/{id}/status` or list documents until `status == ready`.

    Identical content (same SHA-256) is deduplicated and returns the existing document."""
    return await _ingest(db, user, file, replace_document_id)


@router.post(
    "/batch",
    response_model=BatchUploadResult,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(upload_rate_limit)],
    summary="Upload several documents at once",
)
async def upload_batch(files: list[UploadFile] = File(...), db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    if len(files) > 20:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "max 20 files per batch")
    results, errors = [], []
    for f in files:
        try:
            results.append(await _ingest(db, user, f, None))
        except HTTPException as e:
            await db.rollback()
            errors.append({"filename": f.filename or "?", "error": str(e.detail)})
    return BatchUploadResult(results=results, errors=errors)


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
@router.get("", response_model=DocumentPage, summary="List documents")
async def list_documents(
    status_: DocumentStatus | None = Query(None, alias="status"),
    category: str | None = None,
    tag: str | None = None,
    q: str | None = Query(None, description="Case-insensitive match on filename or AI title"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    user: str = Depends(get_current_user),
):
    base = select(Document).where(Document.owner_id == user, Document.deleted_at.is_(None))
    if status_:
        base = base.where(Document.status == status_)
    if category:
        base = base.where(Document.category == category)
    if tag:
        base = base.where(Document.tags.any(tag.lower()))
    if q:
        like = f"%{q}%"
        base = base.where(Document.filename.ilike(like) | Document.title.ilike(like))
    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    rows = (await db.execute(base.order_by(Document.created_at.desc()).limit(limit).offset(offset))).scalars().all()
    return DocumentPage(items=[DocumentOut.model_validate(r) for r in rows], total=total, limit=limit, offset=offset)


@router.post("/compare", response_model=CompareOut, summary="AI comparison of two documents")
async def compare_documents(body: CompareRequest, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    if body.document_id_a == body.document_id_b:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "pick two different documents")
    a = await _get_owned_document(db, body.document_id_a, user)
    b = await _get_owned_document(db, body.document_id_b, user)
    for d in (a, b):
        if d.status != DocumentStatus.ready:
            raise HTTPException(status.HTTP_409_CONFLICT, f"document {d.id} is not ready (status={d.status.value})")

    async def _brief(d: Document) -> str:
        ins = await _latest_insight(db, d)
        summary = ins.content.get("summary", "") if ins else ""
        pts = ins.content.get("key_points", []) if ins else []
        rows = (await db.execute(select(Chunk.content).where(Chunk.version_id == d.current_version_id).order_by(Chunk.chunk_index).limit(3))).scalars().all()
        return f"Title: {d.title}\nSummary: {summary}\nKey points: {'; '.join(pts)}\nExcerpt: {' '.join(rows)[:2500]}"

    try:
        cmp, usage = await acompare_documents(await _brief(a), await _brief(b))
    except LLMError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"AI provider error: {e}") from e
    await arecord_usage(db, usage, "compare", owner_id=user)
    return CompareOut(document_a=DocumentOut.model_validate(a), document_b=DocumentOut.model_validate(b), **cmp.model_dump(), usage=usage.as_dict())


@router.get("/{document_id}", response_model=DocumentDetailOut, summary="Get a document")
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    doc = await _get_owned_document(db, document_id, user, with_versions=True)
    return DocumentDetailOut.model_validate(doc)


async def _status_payload(db: AsyncSession, doc: Document) -> DocumentStatusOut:
    jobs = (
        (
            await db.execute(
                select(ProcessingJob)
                .where(ProcessingJob.document_id == doc.id, ProcessingJob.version_id == doc.current_version_id)
                .order_by(ProcessingJob.queued_at.asc(), ProcessingJob.started_at.asc().nulls_first())
            )
        )
        .scalars()
        .all()
    )
    latest: dict[JobStage, ProcessingJob] = {}
    for j in jobs:
        latest[j.stage] = j  # later rows overwrite → latest attempt per stage
    done = sum(1 for st in PIPELINE_STAGES if st in latest and latest[st].status == JobStatus.succeeded)
    progress = 1.0 if doc.status == DocumentStatus.ready else done / len(PIPELINE_STAGES)
    stages = [
        StageOut(
            stage=j.stage.value,
            status=j.status.value,
            attempt=j.attempt,
            started_at=j.started_at,
            finished_at=j.finished_at,
            duration_ms=j.duration_ms,
            error=j.error,
            meta=j.meta or {},
        )
        for j in (latest.get(st) for st in [JobStage.pipeline, *PIPELINE_STAGES, JobStage.structured])
        if j is not None
    ]
    return DocumentStatusOut(id=doc.id, status=doc.status, error=doc.error, progress=round(progress, 2), stages=stages)


@router.get("/{document_id}/status", response_model=DocumentStatusOut, summary="Processing status per stage")
async def get_status(document_id: uuid.UUID, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    doc = await _get_owned_document(db, document_id, user)
    return await _status_payload(db, doc)


@router.get("/{document_id}/status/stream", summary="Processing status pushed over SSE until ready/failed")
async def stream_status(document_id: uuid.UUID, user: str = Depends(get_current_user)):
    """Real-time alternative to polling: emits a `status` event whenever the stage list
    changes (checked every second), then `done` and closes. Times out after 10 minutes."""

    async def gen():
        Session = get_async_sessionmaker()
        last = None
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 600
        while loop.time() < deadline:
            async with Session() as db:
                try:
                    doc = await _get_owned_document(db, document_id, user)
                except HTTPException as e:
                    yield {"event": "error", "data": json.dumps({"detail": e.detail, "status": e.status_code})}
                    return
                payload = (await _status_payload(db, doc)).model_dump(mode="json")
            if payload != last:
                yield {"event": "status", "data": json.dumps(payload)}
                last = payload
            if payload["status"] in ("ready", "failed"):
                yield {"event": "done", "data": json.dumps({"status": payload["status"]})}
                return
            await asyncio.sleep(1.0)
        yield {"event": "error", "data": json.dumps({"detail": "timeout waiting for processing"})}

    return EventSourceResponse(gen(), ping=15)


async def _latest_extraction(db: AsyncSession, doc: Document, schema_name: str | None = None) -> DocumentInsight | None:
    stmt = select(DocumentInsight).where(
        DocumentInsight.document_id == doc.id, DocumentInsight.version_id == doc.current_version_id, DocumentInsight.kind == "extraction"
    )
    if schema_name:
        stmt = stmt.where(DocumentInsight.options_hash == schema_name)
    return (await db.execute(stmt.order_by(DocumentInsight.created_at.desc()).limit(1))).scalar_one_or_none()


def _extraction_out(doc: Document, ins: DocumentInsight, cached: bool) -> ExtractionOut:
    c = ins.content
    return ExtractionOut(
        document_id=doc.id,
        version_id=ins.version_id,
        schema_name=c.get("schema_name", ins.options.get("schema", "generic")),
        fields=c.get("fields", {}),
        confidence=c.get("confidence", 0.0),
        notes=c.get("notes", ""),
        model=ins.model,
        generated_at=ins.created_at,
        cached=cached,
    )


@router.get("/extraction/schemas", summary="Available structured-extraction schemas and category mapping")
async def list_schemas():
    return {"schemas": SCHEMAS, "category_mapping": CATEGORY_TO_SCHEMA}


@router.get("/{document_id}/extraction", response_model=ExtractionOut, summary="Structured fields extracted for the document's category")
async def get_extraction(document_id: uuid.UUID, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    doc = await _get_owned_document(db, document_id, user)
    if doc.status != DocumentStatus.ready:
        raise HTTPException(status.HTTP_409_CONFLICT, f"document not ready (status={doc.status.value})")
    ins = await _latest_extraction(db, doc)
    if ins is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no extraction yet; POST /extraction to run one")
    return _extraction_out(doc, ins, cached=True)


@router.post("/{document_id}/extraction", response_model=ExtractionOut, summary="Run structured extraction with a chosen schema")
async def run_extraction(document_id: uuid.UUID, body: ExtractionRequest, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    doc = await _get_owned_document(db, document_id, user)
    if doc.status != DocumentStatus.ready:
        raise HTTPException(status.HTTP_409_CONFLICT, f"document not ready (status={doc.status.value})")
    schema_name = body.schema_name or schema_for_category(doc.category)
    if schema_name not in SCHEMAS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown schema '{schema_name}'; available: {sorted(SCHEMAS)}")
    if not body.force:
        existing = await _latest_extraction(db, doc, schema_name)
        if existing:
            return _extraction_out(doc, existing, cached=True)
    rows = (await db.execute(select(Chunk.content, Chunk.token_count).where(Chunk.version_id == doc.current_version_id).order_by(Chunk.chunk_index))).all()
    excerpt = build_excerpt([r.content for r in rows], [r.token_count for r in rows])
    try:
        extraction, usage = await aextract_structured(schema_name, excerpt)
    except LLMError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"AI provider error: {e}") from e
    await arecord_usage(db, usage, "extraction", owner_id=user, document_id=doc.id)
    ins = DocumentInsight(
        document_id=doc.id,
        version_id=doc.current_version_id,
        kind="extraction",
        content=extraction.model_dump(),
        options={"schema": schema_name},
        options_hash=schema_name,
        model=usage.model,
    )
    db.add(ins)
    await db.flush()
    await db.refresh(ins)
    return _extraction_out(doc, ins, cached=False)


async def _latest_insight(db: AsyncSession, doc: Document, options_hash: str | None = None) -> DocumentInsight | None:
    stmt = select(DocumentInsight).where(
        DocumentInsight.document_id == doc.id, DocumentInsight.version_id == doc.current_version_id, DocumentInsight.kind == "analysis"
    )
    if options_hash:
        stmt = stmt.where(DocumentInsight.options_hash == options_hash)
    return (await db.execute(stmt.order_by(DocumentInsight.created_at.desc()).limit(1))).scalar_one_or_none()


def _insight_out(doc: Document, ins: DocumentInsight, cached: bool) -> InsightsOut:
    c = ins.content
    return InsightsOut(
        document_id=doc.id,
        version_id=ins.version_id,
        title=c.get("title", ""),
        summary=c.get("summary", ""),
        key_points=c.get("key_points", []),
        category=c.get("category", "other"),
        tags=c.get("tags", []),
        sentiment=c.get("sentiment", {"label": "neutral", "score": 0, "rationale": ""}),
        entities=c.get("entities", {}),
        language=c.get("language", "en"),
        suggested_questions=c.get("suggested_questions", []),
        options=ins.options,
        model=ins.model,
        generated_at=ins.created_at,
        cached=cached,
    )


@router.get("/{document_id}/insights", response_model=InsightsOut, summary="AI summary, key points, tags, sentiment, entities, suggested questions")
async def get_insights(document_id: uuid.UUID, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    doc = await _get_owned_document(db, document_id, user)
    if doc.status != DocumentStatus.ready:
        raise HTTPException(status.HTTP_409_CONFLICT, f"document not ready (status={doc.status.value}); poll /status")
    ins = await _latest_insight(db, doc)
    if ins is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no insights generated yet")
    return _insight_out(doc, ins, cached=True)


@router.post("/{document_id}/summary", response_model=InsightsOut, summary="Generate a customised summary (length / tone / focus)")
async def custom_summary(document_id: uuid.UUID, body: SummaryRequest, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    doc = await _get_owned_document(db, document_id, user)
    if doc.status != DocumentStatus.ready:
        raise HTTPException(status.HTTP_409_CONFLICT, f"document not ready (status={doc.status.value})")
    options = SummaryOptions(length=body.length, tone=body.tone, focus=body.focus)
    if not body.force:
        existing = await _latest_insight(db, doc, options.hash())
        if existing:
            return _insight_out(doc, existing, cached=True)
    rows = (await db.execute(select(Chunk.content, Chunk.token_count).where(Chunk.version_id == doc.current_version_id).order_by(Chunk.chunk_index))).all()
    excerpt = build_excerpt([r.content for r in rows], [r.token_count for r in rows])
    try:
        analysis, usage = await aanalyse_document(excerpt, options)
    except LLMError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"AI provider error: {e}") from e
    await arecord_usage(db, usage, "summary", owner_id=user, document_id=doc.id)
    ins = DocumentInsight(
        document_id=doc.id,
        version_id=doc.current_version_id,
        kind="analysis",
        content=analysis.model_dump(),
        options=options.model_dump(),
        options_hash=options.hash(),
        model=usage.model,
    )
    db.add(ins)
    await db.flush()
    await db.refresh(ins)
    return _insight_out(doc, ins, cached=False)


@router.post("/{document_id}/reprocess", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED, summary="Re-run the processing pipeline")
async def reprocess(
    document_id: uuid.UUID,
    force: bool = Query(False, description="Re-queue even if status is 'processing' (e.g. a worker died mid-run)"),
    db: AsyncSession = Depends(get_db),
    user: str = Depends(get_current_user),
):
    doc = await _get_owned_document(db, document_id, user)
    if doc.status == DocumentStatus.processing and not force:
        raise HTTPException(status.HTTP_409_CONFLICT, "already processing; pass ?force=true to override")
    doc.status = DocumentStatus.uploaded
    doc.error = None
    db.add(ProcessingJob(document_id=doc.id, version_id=doc.current_version_id, stage=JobStage.pipeline, status=JobStatus.queued, meta={"reprocess": True}))
    await db.commit()
    if _safe_enqueue(doc.id, doc.current_version_id) is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "task queue unavailable; try again later")
    await db.refresh(doc)
    return DocumentOut.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document (soft delete; vectors removed)")
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    doc = await _get_owned_document(db, document_id, user)
    doc.deleted_at = datetime.now(UTC)
    # remove searchable content immediately; metadata row stays for audit/metrics
    from sqlalchemy import delete as sa_delete

    await db.execute(sa_delete(Chunk).where(Chunk.document_id == doc.id))
    return None
