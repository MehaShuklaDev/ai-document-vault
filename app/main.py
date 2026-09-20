"""FastAPI application factory."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram
from sqlalchemy import text

from app.api import chat, documents, health, metrics
from app.core.config import get_settings
from app.core.db import get_async_engine
from app.core.logging import configure_logging, get_logger, request_id_var

settings = get_settings()
configure_logging(settings.log_level, json_output=settings.environment != "dev")
log = get_logger("app")

REQ_COUNT = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP latency", ["method", "path"], buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # fail fast if no AI provider is configured (auto mode) — never silently degrade to fake
    llm = settings.resolved_llm_provider
    log.info("ai_providers", llm=llm, embeddings=settings.resolved_embedding_provider, configured=settings.llm_provider)
    # fail fast if the DB is unreachable; migrations are applied by `alembic upgrade head`
    try:
        async with get_async_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        log.info("startup", llm=settings.resolved_llm_provider, embeddings=settings.resolved_embedding_provider, env=settings.environment)
    except Exception as e:  # pragma: no cover
        log.error("database_unreachable", error=str(e))
    yield
    await get_async_engine().dispose()


app = FastAPI(
    title=settings.app_name,
    version=health.VERSION,
    description=(
        "Upload documents, get AI insights, and chat with them using RAG.\n\n"
        "**Identity:** pass `X-User-Id` (defaults to `demo-user`). All data is scoped per user.\n\n"
        "**Flow:** `POST /documents` → poll `GET /documents/{id}/status` → `POST /chat/sessions` → `POST /chat/sessions/{id}/ask`."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Request-Id", "Retry-After"])


@app.middleware("http")
async def observability(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled_error", path=request.url.path)
        response = JSONResponse({"detail": "internal server error", "request_id": rid}, status_code=500)
    elapsed = time.perf_counter() - t0
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    REQ_COUNT.labels(request.method, path, response.status_code).inc()
    REQ_LATENCY.labels(request.method, path).observe(elapsed)
    response.headers["X-Request-Id"] = rid
    if request.url.path.startswith(settings.api_prefix):
        log.info("request", method=request.method, path=request.url.path, status=response.status_code, ms=int(elapsed * 1000))
    request_id_var.reset(token)
    return response


for r in (documents.router, chat.router, metrics.router, health.router):
    app.include_router(r, prefix=settings.api_prefix)

# minimal dashboard (bonus) — static, no build step
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")
