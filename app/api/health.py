"""Liveness / readiness."""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.ai.providers import get_embeddings, get_llm
from app.core.db import get_async_engine
from app.schemas import Health
from app.services.cache import redis_ping

router = APIRouter(prefix="/health", tags=["health"])
VERSION = "1.0.0"


@router.get("/live", summary="Process is up")
async def live():
    return {"status": "ok"}


@router.get("/ready", response_model=Health, summary="Dependencies reachable (DB, Redis, pgvector)")
async def ready(response: Response):
    checks = {"database": False, "pgvector": False, "redis": False}
    try:
        async with get_async_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
            checks["database"] = True
            ext = (await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))).first()
            checks["pgvector"] = ext is not None
    except Exception:
        pass
    checks["redis"] = await redis_ping()
    ok = all(checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return Health(
        status="ok" if ok else "degraded",
        checks=checks,
        version=VERSION,
        llm_provider=f"{get_llm().name}:{get_llm().model}",
        embedding_provider=f"{get_embeddings().name}:{get_embeddings().model}",
    )
