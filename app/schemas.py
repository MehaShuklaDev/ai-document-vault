"""Pydantic request/response models (the public API contract)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ai.analysis import SummaryLength, SummaryTone
from app.models import DocumentStatus


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ documents
class DocumentVersionOut(ORMModel):
    id: uuid.UUID
    version: int
    sha256: str
    size_bytes: int
    created_at: datetime


class DocumentOut(ORMModel):
    id: uuid.UUID
    filename: str
    mime_type: str
    status: DocumentStatus
    title: str | None
    category: str | None
    tags: list[str] | None
    language: str | None
    page_count: int | None
    chunk_count: int | None
    token_count: int | None
    version_count: int
    current_version_id: uuid.UUID | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None


class DocumentDetailOut(DocumentOut):
    versions: list[DocumentVersionOut]


class UploadResult(BaseModel):
    document: DocumentOut
    deduplicated: bool = Field(description="True when identical content already existed; no new processing was queued.")
    new_version: bool = Field(False, description="True when this upload became a new version of an existing document.")
    task_id: str | None = None


class BatchUploadResult(BaseModel):
    results: list[UploadResult]
    errors: list[dict[str, str]]


class StageOut(ORMModel):
    stage: str
    status: str
    attempt: int
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: int | None
    error: str | None
    meta: dict[str, Any]


class DocumentStatusOut(BaseModel):
    id: uuid.UUID
    status: DocumentStatus
    error: str | None
    progress: float = Field(description="0..1 fraction of pipeline stages completed for the current version")
    stages: list[StageOut]


class SentimentOut(BaseModel):
    label: str
    score: float
    rationale: str


class InsightsOut(BaseModel):
    document_id: uuid.UUID
    version_id: uuid.UUID
    title: str
    summary: str
    key_points: list[str]
    category: str
    tags: list[str]
    sentiment: SentimentOut
    entities: dict[str, list[str]]
    language: str
    suggested_questions: list[str]
    options: dict[str, Any]
    model: str | None
    generated_at: datetime
    cached: bool = False


class SummaryRequest(BaseModel):
    length: SummaryLength = "medium"
    tone: SummaryTone = "neutral"
    focus: str | None = Field(None, max_length=200, description="Optional focus area, e.g. 'financial risks'")
    force: bool = Field(False, description="Regenerate even if a summary with the same options exists")


class ExtractionOut(BaseModel):
    document_id: uuid.UUID
    version_id: uuid.UUID
    schema_name: str
    fields: dict[str, Any]
    confidence: float
    notes: str
    model: str | None
    generated_at: datetime
    cached: bool = False


class ExtractionRequest(BaseModel):
    schema_name: str | None = Field(None, description="Override the category-derived schema, e.g. 'invoice', 'contract', 'generic'")
    force: bool = False


class CompareRequest(BaseModel):
    document_id_a: uuid.UUID
    document_id_b: uuid.UUID


class CompareOut(BaseModel):
    document_a: DocumentOut
    document_b: DocumentOut
    comparison: str
    similarities: list[str]
    differences: list[str]
    recommendation: str
    usage: dict[str, Any]


class Page(BaseModel):
    items: list[Any]
    total: int
    limit: int
    offset: int


class DocumentPage(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


# ----------------------------------------------------------------------- chat
class SessionCreate(BaseModel):
    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=20, description="Documents in scope; multi-document chat when >1")
    title: str | None = Field(None, max_length=256)


class SessionOut(ORMModel):
    id: uuid.UUID
    title: str | None
    document_ids: list[uuid.UUID]
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None


class CitationOut(BaseModel):
    n: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_start: int | None
    page_end: int | None
    section: str | None
    snippet: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    use_cache: bool = True
    suggest_follow_ups: bool = True


class MessageOut(ORMModel):
    id: uuid.UUID
    seq: int
    role: Literal["user", "assistant", "system"]
    content: str
    standalone_question: str | None
    citations: list[CitationOut] | None
    follow_ups: list[str] | None
    usage: dict[str, Any] | None
    created_at: datetime


class AskResponse(BaseModel):
    session_id: uuid.UUID
    question: MessageOut
    answer: MessageOut
    cached: bool


class HistoryOut(BaseModel):
    session: SessionOut
    messages: list[MessageOut]


# -------------------------------------------------------------------- metrics
class DocumentStats(BaseModel):
    total: int
    by_status: dict[str, int]
    by_category: dict[str, int]
    by_mime_type: dict[str, int]
    total_pages: int
    total_chunks: int
    total_tokens: int
    total_bytes: int
    uploads_last_24h: int
    top_tags: list[dict[str, Any]]


class StageMetrics(BaseModel):
    stage: str
    runs: int
    succeeded: int
    failed: int
    p50_ms: float | None
    p95_ms: float | None
    avg_ms: float | None


class ProcessingMetrics(BaseModel):
    window_hours: int
    queued: int
    running: int
    succeeded: int
    failed: int
    failure_rate: float
    throughput_per_hour: float
    stages: list[StageMetrics]
    recent_failures: list[dict[str, Any]]


class CostByKey(BaseModel):
    key: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    avg_latency_ms: float | None


class CostMetrics(BaseModel):
    window_hours: int
    total_cost_usd: float
    total_calls: int
    cache_hit_rate: float
    by_purpose: list[CostByKey]
    by_model: list[CostByKey]
    chat: dict[str, Any]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, bool]
    version: str
    llm_provider: str
    embedding_provider: str
