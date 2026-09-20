"""Document processing pipeline (Celery task).

Stages: extract → chunk → embed → analyse. Each stage writes a ``processing_jobs``
row with timing and outcome, so `/metrics/processing` can report per-stage p50/p95
and failure rates without log scraping.

Idempotency: the task operates on a *version*. Re-running it deletes that
version's chunks/insights first, so a retry after a mid-way crash cannot create
duplicates.

Errors: ``ExtractionError`` and validation errors are *permanent* (bad input) →
document marked failed, no retry. Infra/LLM transient errors → Celery autoretry
with exponential back-off, max 3 attempts.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.ai.analysis import SummaryOptions, analyse_document, build_excerpt
from app.ai.embeddings import embed_texts
from app.ai.providers import LLMError
from app.ai.structured import extract_structured, schema_for_category
from app.ai.usage import record_usage
from app.core.config import get_settings
from app.core.db import sync_session
from app.core.logging import get_logger
from app.models import Chunk, Document, DocumentInsight, DocumentStatus, DocumentVersion, JobStage, JobStatus, ProcessingJob
from app.services.chunking import chunk_pages
from app.services.extraction import ExtractionError, extract_pages
from app.services.storage import get_storage
from app.workers.celery_app import celery_app

log = get_logger(__name__)


class PermanentError(Exception):
    """Bad input; retrying will not help."""


def _now() -> datetime:
    return datetime.now(UTC)


class StageRecorder:
    """Context manager that records one processing_jobs row per stage."""

    def __init__(self, db: Session, document_id: uuid.UUID, version_id: uuid.UUID, stage: JobStage, attempt: int, task_id: str | None):
        self.db, self.stage = db, stage
        self.job = ProcessingJob(
            document_id=document_id,
            version_id=version_id,
            stage=stage,
            status=JobStatus.running,
            attempt=attempt,
            celery_task_id=task_id,
            started_at=_now(),
            meta={},
        )
        db.add(self.job)
        db.flush()
        self.t0 = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.job.finished_at = _now()
        self.job.duration_ms = int((time.perf_counter() - self.t0) * 1000)
        if exc is None:
            self.job.status = JobStatus.succeeded
        else:
            self.job.status = JobStatus.failed
            self.job.error = f"{type(exc).__name__}: {exc}"[:2000]
        # commit per stage: a later failure must not roll back this stage's record,
        # and partial progress (page_count, chunk_count) becomes visible to /status
        self.db.commit()
        return False  # propagate


def _load(db: Session, document_id: uuid.UUID, version_id: uuid.UUID) -> tuple[Document, DocumentVersion]:
    doc = db.get(Document, document_id)
    ver = db.get(DocumentVersion, version_id)
    if doc is None or ver is None or ver.document_id != doc.id:
        raise PermanentError("document/version not found")
    return doc, ver


def _claim_pipeline_job(db: Session, document_id: uuid.UUID, version_id: uuid.UUID, attempt: int, task_id: str | None) -> ProcessingJob:
    """Flip the 'queued' umbrella job created at upload time to 'running' (or create one on retry)."""
    job = (
        db.execute(
            select(ProcessingJob)
            .where(
                ProcessingJob.version_id == version_id,
                ProcessingJob.stage == JobStage.pipeline,
                ProcessingJob.status.in_([JobStatus.queued, JobStatus.running]),
            )
            .order_by(ProcessingJob.queued_at.desc())
        )
        .scalars()
        .first()
    )
    if job is None:
        job = ProcessingJob(document_id=document_id, version_id=version_id, stage=JobStage.pipeline, status=JobStatus.running, meta={})
        db.add(job)
    # anything else still open for this version was orphaned by a dead worker / force-reprocess
    for stale in db.execute(
        select(ProcessingJob).where(
            ProcessingJob.version_id == version_id, ProcessingJob.status.in_([JobStatus.queued, JobStatus.running]), ProcessingJob.id != job.id
        )
    ).scalars():
        stale.status = JobStatus.failed
        stale.finished_at = _now()
        stale.error = "superseded by a newer run"
    job.status = JobStatus.running
    job.attempt = attempt
    job.celery_task_id = task_id
    job.started_at = _now()
    return job


def _mark_failed(document_id: uuid.UUID, version_id: uuid.UUID | None, error: str, attempt: int, task_id: str | None) -> None:
    with sync_session() as db:
        doc = db.get(Document, document_id)
        if doc:
            doc.status = DocumentStatus.failed
            doc.error = error[:2000]
        job = _claim_pipeline_job(db, document_id, version_id, attempt, task_id) if version_id else None
        if job is None:
            job = ProcessingJob(
                document_id=document_id,
                version_id=version_id,
                stage=JobStage.pipeline,
                status=JobStatus.running,
                attempt=attempt,
                celery_task_id=task_id,
                started_at=_now(),
                meta={},
            )
            db.add(job)
        job.status = JobStatus.failed
        job.finished_at = _now()
        job.duration_ms = int((job.finished_at - job.started_at).total_seconds() * 1000) if job.started_at else 0
        job.error = error[:2000]


def run_pipeline(document_id: uuid.UUID, version_id: uuid.UUID, *, attempt: int = 1, task_id: str | None = None) -> dict:
    """Synchronous pipeline body; separated from the Celery wrapper so tests can call it directly."""
    s = get_settings()
    storage = get_storage()
    t_start = time.perf_counter()

    with sync_session() as db:
        doc, ver = _load(db, document_id, version_id)
        doc.status = DocumentStatus.processing
        doc.error = None
        # idempotency: wipe partial output from a previous attempt
        db.execute(delete(Chunk).where(Chunk.version_id == version_id))
        db.execute(delete(DocumentInsight).where(DocumentInsight.version_id == version_id))
        pipeline_job = _claim_pipeline_job(db, document_id, version_id, attempt, task_id)
        db.flush()

        # ---- 1. extract
        with StageRecorder(db, document_id, version_id, JobStage.extract, attempt, task_id) as rec:
            data = storage.get(ver.storage_key)
            try:
                pages = extract_pages(data, doc.mime_type, max_pages=s.max_pages)
            except ExtractionError as e:
                raise PermanentError(str(e)) from e
            rec.job.meta = {"pages": len(pages), "bytes": len(data)}
        doc.page_count = max(p.number for p in pages)

        # ---- 2. chunk
        with StageRecorder(db, document_id, version_id, JobStage.chunk, attempt, task_id) as rec:
            chunks = chunk_pages(
                pages, chunk_tokens=s.chunk_tokens, overlap_tokens=s.chunk_overlap_tokens, encoding=s.tokenizer_encoding, max_chunks=s.max_chunks_per_document
            )
            if not chunks:
                raise PermanentError("no text chunks produced")
            rec.job.meta = {"chunks": len(chunks), "tokens": sum(c.token_count for c in chunks)}
        doc.chunk_count = len(chunks)
        doc.token_count = sum(c.token_count for c in chunks)

        # ---- 3. embed (batched, cached)
        with StageRecorder(db, document_id, version_id, JobStage.embed, attempt, task_id) as rec:
            vectors, usages = embed_texts([c.content for c in chunks])
            for u in usages:
                record_usage(db, u, "embed", owner_id=doc.owner_id, document_id=doc.id)
            rows = [
                Chunk(
                    document_id=doc.id,
                    version_id=ver.id,
                    owner_id=doc.owner_id,
                    chunk_index=c.index,
                    content=c.content,
                    token_count=c.token_count,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    section=c.section,
                    embedding=v,
                )
                for c, v in zip(chunks, vectors, strict=True)
            ]
            db.bulk_save_objects(rows)
            rec.job.meta = {"vectors": len(rows), "cache_hits": sum(1 for u in usages if u.cached), "api_calls": sum(1 for u in usages if not u.cached)}

        # ---- 4. analyse (summary, tags, sentiment, questions)
        with StageRecorder(db, document_id, version_id, JobStage.analyse, attempt, task_id) as rec:
            excerpt = build_excerpt([c.content for c in chunks], [c.token_count for c in chunks])
            options = SummaryOptions()
            analysis, usage = analyse_document(excerpt, options)
            record_usage(db, usage, "analysis", owner_id=doc.owner_id, document_id=doc.id)
            db.add(
                DocumentInsight(
                    document_id=doc.id,
                    version_id=ver.id,
                    kind="analysis",
                    content=analysis.model_dump(),
                    options=options.model_dump(),
                    options_hash=options.hash(),
                    model=usage.model,
                )
            )
            doc.title = analysis.title or doc.filename
            doc.category = analysis.category
            doc.tags = analysis.tags
            doc.language = analysis.language
            rec.job.meta = {"model": usage.model, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}

        # ---- 5. structured extraction per category (optional, non-fatal)
        if s.structured_extraction_enabled:
            schema_name = schema_for_category(analysis.category)
            try:
                with StageRecorder(db, document_id, version_id, JobStage.structured, attempt, task_id) as rec:
                    extraction, usage = extract_structured(schema_name, excerpt)
                    record_usage(db, usage, "extraction", owner_id=doc.owner_id, document_id=doc.id)
                    db.add(
                        DocumentInsight(
                            document_id=doc.id,
                            version_id=ver.id,
                            kind="extraction",
                            content=extraction.model_dump(),
                            options={"schema": schema_name},
                            options_hash=schema_name,
                            model=usage.model,
                        )
                    )
                    rec.job.meta = {"schema": schema_name, "confidence": extraction.confidence, "fields": len(extraction.fields)}
            except LLMError as e:
                # a bad extraction must not fail an otherwise good document; the stage row records it
                log.warning("structured_extraction_failed", document_id=str(document_id), error=str(e))

        doc.status = DocumentStatus.ready
        doc.processed_at = _now()
        pipeline_job.status = JobStatus.succeeded
        pipeline_job.finished_at = _now()
        pipeline_job.duration_ms = int((time.perf_counter() - t_start) * 1000)
        pipeline_job.meta = {"pages": doc.page_count, "chunks": doc.chunk_count, "tokens": doc.token_count}

    log.info("document_processed", document_id=str(document_id), chunks=len(chunks), ms=int((time.perf_counter() - t_start) * 1000))
    return {"document_id": str(document_id), "chunks": len(chunks), "pages": doc.page_count}


@celery_app.task(
    bind=True,
    name="app.workers.tasks.process_document",
    autoretry_for=(LLMError, ConnectionError, TimeoutError, OSError),
    retry_backoff=5,
    retry_backoff_max=120,
    retry_jitter=True,
    max_retries=3,
)
def process_document(self, document_id: str, version_id: str) -> dict:
    doc_id, ver_id = uuid.UUID(document_id), uuid.UUID(version_id)
    attempt = self.request.retries + 1
    try:
        return run_pipeline(doc_id, ver_id, attempt=attempt, task_id=self.request.id)
    except PermanentError as e:
        _mark_failed(doc_id, ver_id, str(e), attempt, self.request.id)
        log.warning("document_failed_permanent", document_id=document_id, error=str(e))
        return {"document_id": document_id, "error": str(e)}
    except SoftTimeLimitExceeded:
        _mark_failed(doc_id, ver_id, "processing exceeded time limit", attempt, self.request.id)
        raise
    except Exception as e:
        if attempt > self.max_retries:
            _mark_failed(doc_id, ver_id, f"{type(e).__name__}: {e}", attempt, self.request.id)
        else:
            with sync_session() as db:
                d = db.get(Document, doc_id)
                if d:
                    d.error = f"attempt {attempt} failed: {type(e).__name__}: {e}"[:2000]
        raise


def enqueue_processing(document_id: uuid.UUID, version_id: uuid.UUID) -> str:
    res = process_document.apply_async(args=[str(document_id), str(version_id)])
    return res.id
