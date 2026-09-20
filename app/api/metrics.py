"""Monitoring endpoints: document statistics, processing metrics, AI cost, Prometheus."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import Float, case, cast, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.db import get_db
from app.models import AIUsage, ChatMessage, ChatSession, Document, DocumentVersion, JobStage, JobStatus, ProcessingJob
from app.schemas import CostByKey, CostMetrics, DocumentStats, ProcessingMetrics, StageMetrics

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/documents", response_model=DocumentStats, summary="Document statistics for the current user")
async def document_stats(db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    base = select(Document).where(Document.owner_id == user, Document.deleted_at.is_(None)).subquery()
    total = (await db.execute(select(func.count()).select_from(base))).scalar_one()
    by_status = dict((await db.execute(select(base.c.status, func.count()).group_by(base.c.status))).all())
    by_cat = dict((await db.execute(select(func.coalesce(base.c.category, "uncategorised"), func.count()).group_by(base.c.category))).all())
    by_mime = dict((await db.execute(select(base.c.mime_type, func.count()).group_by(base.c.mime_type))).all())
    sums = (
        await db.execute(
            select(
                func.coalesce(func.sum(base.c.page_count), 0), func.coalesce(func.sum(base.c.chunk_count), 0), func.coalesce(func.sum(base.c.token_count), 0)
            )
        )
    ).one()
    total_bytes = (
        await db.execute(
            select(func.coalesce(func.sum(DocumentVersion.size_bytes), 0))
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(Document.owner_id == user, Document.deleted_at.is_(None))
        )
    ).scalar_one()
    since = datetime.now(UTC) - timedelta(hours=24)
    recent = (await db.execute(select(func.count()).select_from(base).where(base.c.created_at >= since))).scalar_one()
    tags = (
        await db.execute(
            text(
                "SELECT t AS tag, COUNT(*) AS n FROM documents d, unnest(d.tags) AS t WHERE d.owner_id = :u AND d.deleted_at IS NULL GROUP BY t ORDER BY n DESC LIMIT 15"
            ),
            {"u": user},
        )
    ).all()
    return DocumentStats(
        total=total,
        by_status={(k.value if hasattr(k, "value") else str(k)): v for k, v in by_status.items()},
        by_category=by_cat,
        by_mime_type=by_mime,
        total_pages=int(sums[0]),
        total_chunks=int(sums[1]),
        total_tokens=int(sums[2]),
        total_bytes=int(total_bytes),
        uploads_last_24h=recent,
        top_tags=[{"tag": r.tag, "count": r.n} for r in tags],
    )


@router.get("/processing", response_model=ProcessingMetrics, summary="Pipeline throughput, latency percentiles and failures")
async def processing_metrics(hours: int = Query(24, ge=1, le=24 * 30), db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    since = datetime.now(UTC) - timedelta(hours=hours)
    owned = select(Document.id).where(Document.owner_id == user).subquery()
    j = ProcessingJob
    scope = (j.document_id.in_(select(owned.c.id)), j.queued_at >= since)

    counts = dict((await db.execute(select(j.status, func.count()).where(*scope, j.stage == JobStage.pipeline).group_by(j.status))).all())
    get = lambda st: int(counts.get(st, 0))  # noqa: E731
    finished = get(JobStatus.succeeded) + get(JobStatus.failed)
    failure_rate = round(get(JobStatus.failed) / finished, 4) if finished else 0.0

    stage_rows = (
        await db.execute(
            select(
                j.stage,
                func.count(),
                func.sum(case((j.status == JobStatus.succeeded, 1), else_=0)),
                func.sum(case((j.status == JobStatus.failed, 1), else_=0)),
                func.percentile_cont(0.5).within_group(cast(j.duration_ms, Float)),
                func.percentile_cont(0.95).within_group(cast(j.duration_ms, Float)),
                func.avg(cast(j.duration_ms, Float)),
            )
            .where(*scope, j.duration_ms.isnot(None))
            .group_by(j.stage)
        )
    ).all()
    order = [JobStage.pipeline, JobStage.extract, JobStage.chunk, JobStage.embed, JobStage.analyse]
    stages = sorted(
        [
            StageMetrics(stage=r[0].value, runs=int(r[1]), succeeded=int(r[2] or 0), failed=int(r[3] or 0), p50_ms=_r(r[4]), p95_ms=_r(r[5]), avg_ms=_r(r[6]))
            for r in stage_rows
        ],
        key=lambda s: order.index(JobStage(s.stage)) if JobStage(s.stage) in order else 99,
    )
    failures = (
        await db.execute(
            select(j.document_id, j.stage, j.error, j.finished_at, Document.filename)
            .join(Document, Document.id == j.document_id)
            .where(*scope, j.status == JobStatus.failed)
            .order_by(j.finished_at.desc())
            .limit(10)
        )
    ).all()
    return ProcessingMetrics(
        window_hours=hours,
        queued=get(JobStatus.queued),
        running=get(JobStatus.running),
        succeeded=get(JobStatus.succeeded),
        failed=get(JobStatus.failed),
        failure_rate=failure_rate,
        throughput_per_hour=round(get(JobStatus.succeeded) / hours, 3),
        stages=stages,
        recent_failures=[
            {"document_id": str(f[0]), "filename": f[4], "stage": f[1].value, "error": f[2], "at": f[3].isoformat() if f[3] else None} for f in failures
        ],
    )


def _r(v) -> float | None:
    return round(float(v), 1) if v is not None else None


@router.get("/costs", response_model=CostMetrics, summary="AI token usage and estimated cost")
async def cost_metrics(hours: int = Query(24 * 30, ge=1, le=24 * 365), db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)):
    since = datetime.now(UTC) - timedelta(hours=hours)
    u = AIUsage
    scope = (u.owner_id == user, u.created_at >= since)
    totals = (await db.execute(select(func.count(), func.coalesce(func.sum(u.cost_usd), 0), func.sum(case((u.cached, 1), else_=0))).where(*scope))).one()
    calls, cost, cached = int(totals[0]), float(totals[1]), int(totals[2] or 0)

    async def _group(col):
        rows = (
            await db.execute(
                select(
                    col,
                    func.count(),
                    func.coalesce(func.sum(u.input_tokens), 0),
                    func.coalesce(func.sum(u.output_tokens), 0),
                    func.coalesce(func.sum(u.cost_usd), 0),
                    func.avg(cast(u.latency_ms, Float)),
                )
                .where(*scope)
                .group_by(col)
                .order_by(func.sum(u.cost_usd).desc())
            )
        ).all()
        return [
            CostByKey(key=str(r[0]), calls=int(r[1]), input_tokens=int(r[2]), output_tokens=int(r[3]), cost_usd=round(float(r[4]), 6), avg_latency_ms=_r(r[5]))
            for r in rows
        ]

    chat_rows = (
        await db.execute(
            select(
                func.count(ChatMessage.id),
                func.avg(cast(ChatMessage.usage["latency_ms"].astext, Float)),
                func.sum(case((ChatMessage.usage["cached"].astext == "true", 1), else_=0)),
            )
            .join(ChatSession, ChatSession.id == ChatMessage.session_id)
            .where(ChatSession.owner_id == user, ChatMessage.role == "assistant", ChatMessage.created_at >= since)
        )
    ).one()
    return CostMetrics(
        window_hours=hours,
        total_cost_usd=round(cost, 6),
        total_calls=calls,
        cache_hit_rate=round(cached / calls, 4) if calls else 0.0,
        by_purpose=await _group(u.purpose),
        by_model=await _group(u.model),
        chat={"answers": int(chat_rows[0] or 0), "avg_latency_ms": _r(chat_rows[1]), "cached_answers": int(chat_rows[2] or 0)},
    )


@router.get("/prometheus", include_in_schema=False)
async def prometheus():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
