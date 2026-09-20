"""Test fixtures.

Unit tests need nothing. Integration tests (``@pytest.mark.integration``) need the
Postgres + Redis from ``docker compose up -d db redis`` and are skipped when the
database is unreachable. They run the Celery pipeline *inline* (no worker) by
calling ``run_pipeline`` directly.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("EMBEDDING_PROVIDER", "fake")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("STORAGE_LOCAL_PATH", "./data/test-blobs")


def _db_available() -> bool:
    try:
        from sqlalchemy import create_engine, text

        from app.core.config import get_settings

        eng = create_engine(get_settings().sync_database_url, connect_args={"connect_timeout": 2})
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


DB_AVAILABLE = _db_available()
integration = pytest.mark.skipif(not DB_AVAILABLE, reason="database not reachable; run `docker compose up -d db redis && alembic upgrade head`")


@pytest.fixture
def user_id() -> str:
    # unique tenant per test → no cross-test interference, no cleanup needed
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def _fresh_async_engine():
    """The async engine is cached per process; each pytest-asyncio test gets its own
    event loop, so dispose the pool after every test to avoid cross-loop connections."""
    # sse-starlette caches an asyncio.Event bound to the first loop that used it
    try:
        from sse_starlette.sse import AppStatus

        AppStatus.should_exit_event = None
    except Exception:
        pass
    yield
    from app.core import db as dbmod
    from app.services import cache

    await dbmod.get_async_engine().dispose()
    dbmod.get_async_engine.cache_clear()
    dbmod.get_async_sessionmaker.cache_clear()
    try:
        await cache.async_redis().aclose()
    except Exception:
        pass
    cache.async_redis.cache_clear()


@pytest.fixture
async def client(user_id):
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers={"X-User-Id": user_id}) as c:
        yield c


@pytest.fixture
def run_inline(monkeypatch):
    """Replace Celery enqueue with a synchronous in-process pipeline run."""
    from app.api import documents as documents_api
    from app.workers import tasks

    def _inline(document_id, version_id):
        task_id = f"inline-{uuid.uuid4().hex[:8]}"
        try:
            tasks.run_pipeline(document_id, version_id, attempt=1, task_id=task_id)
        except tasks.PermanentError as e:  # mirror the Celery wrapper's handling
            tasks._mark_failed(document_id, version_id, str(e), 1, task_id)
        return task_id

    monkeypatch.setattr(documents_api, "enqueue_processing", _inline)
    return _inline


SAMPLE_TXT = b"""ACME Corp Quarterly Report Q3 2025

EXECUTIVE SUMMARY
Revenue for the third quarter reached $12.4 million, an increase of 18% year over year. Operating margin improved to 21%. The board approved a dividend of $0.15 per share payable on November 30, 2025.

RISKS
Supply chain delays in the Singapore facility caused a two-week shipment backlog. Invoice INV-88213 from Delta Logistics totalling $48,900 remains disputed. Management expects resolution by December.

OUTLOOK
ACME expects Q4 revenue between $13.0 and $13.6 million. The new Berlin office opens in January 2026 with 40 engineers led by Maria Fontaine.
"""

SAMPLE_MD = b"""# Employee Handbook

## Vacation Policy
Full-time employees accrue 1.5 vacation days per month, capped at 24 days. Requests must be submitted two weeks in advance through the HR portal.

## Remote Work
Employees may work remotely up to three days per week with manager approval. Core hours are 10:00 to 15:00 local time.
"""
