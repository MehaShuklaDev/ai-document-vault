"""Application settings.

Everything is driven by environment variables (see `.env.example`). Defaults are
chosen so that `docker compose up` works with zero configuration and *no* API
keys: the Fake AI providers are selected unless a real provider is requested.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderNotConfigured(RuntimeError):
    """Raised in LLM_PROVIDER=auto when neither an API key nor a logged-in Claude CLI is available."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- app ---
    app_name: str = "Vault Document AI"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    # --- persistence ---
    database_url: str = "postgresql+asyncpg://vault:vault@localhost:5433/vault"
    redis_url: str = "redis://localhost:6380/0"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # --- storage ---
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_path: str = "./data/blobs"
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"

    # --- upload limits ---
    max_upload_mb: int = 25
    max_pages: int = 500
    max_chunks_per_document: int = 4000
    allowed_mime_types: tuple[str, ...] = (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
    )

    # --- chunking ---
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 60
    tokenizer_encoding: str = "cl100k_base"

    # --- AI providers ---
    # "auto" (default): OPENAI_API_KEY → openai · ANTHROPIC_API_KEY → anthropic · `claude` CLI logged in → claude-cli · else fake
    llm_provider: Literal["auto", "openai", "anthropic", "claude-cli", "fake"] = "auto"
    # "auto": OPENAI_API_KEY → openai · else fake (hashed bag-of-words; hybrid FTS still retrieves well)
    embedding_provider: Literal["auto", "openai", "fake"] = "auto"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    claude_cli_path: str = "claude"  # for LLM_PROVIDER=claude-cli: path to the Claude Code binary
    llm_model: str | None = None  # provider default if None
    llm_fast_model: str | None = None  # cheap model for condense / tagging
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # --- retrieval ---
    retrieval_top_k: int = 8
    retrieval_vector_candidates: int = 20
    retrieval_fts_candidates: int = 20
    context_token_budget: int = 3000
    rrf_k: int = 60

    # --- reranking (feature flag) ---
    rerank_enabled: bool = False
    rerank_provider: Literal["cross_encoder", "llm", "lexical"] = "lexical"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    rerank_weight: float = 0.6  # blend: w*rerank + (1-w)*rrf

    # --- structured extraction per category ---
    structured_extraction_enabled: bool = True

    # --- caching ---
    embedding_cache_ttl_seconds: int = 60 * 60 * 24 * 7
    answer_cache_ttl_seconds: int = 60 * 60

    # --- rate limiting (sliding window per user) ---
    rate_limit_enabled: bool = True
    rate_limit_chat_per_minute: int = 30
    rate_limit_upload_per_minute: int = 20

    # --- quota (per user, rolling 24 h; 0 = disabled) ---
    quota_usd_per_user_per_day: float = 0.0
    quota_tokens_per_user_per_day: int = 0

    # --- celery ---
    celery_task_soft_time_limit: int = 600
    celery_task_time_limit: int = 900

    # --- identity (placeholder until real auth) ---
    default_user_id: str = "demo-user"

    # ---- provider resolution -------------------------------------------------
    def find_claude_cli(self) -> str | None:
        """Path to a usable Claude Code binary, or None."""
        import glob
        import os
        import shutil

        if self.claude_cli_path and os.path.sep in self.claude_cli_path:
            return self.claude_cli_path if os.path.exists(self.claude_cli_path) else None
        found = shutil.which(self.claude_cli_path or "claude")
        if found:
            return found
        # macOS desktop app bundles the CLI; pick the newest version present
        candidates = sorted(glob.glob(os.path.expanduser("~/Library/Application Support/Claude/claude-code/*/claude.app/Contents/MacOS/claude")))
        return candidates[-1] if candidates else None

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        if self.find_claude_cli():
            return "claude-cli"
        raise ProviderNotConfigured(
            "No LLM available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env, or install and log in to the "
            "Claude Code CLI (`npm i -g @anthropic-ai/claude-code && claude`), or set LLM_PROVIDER=fake explicitly for offline testing."
        )

    @property
    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider != "auto":
            return self.embedding_provider
        return "openai" if self.openai_api_key else "fake"

    @property
    def sync_database_url(self) -> str:
        """psycopg URL for Celery workers / Alembic."""
        return self.database_url.replace("+asyncpg", "+psycopg")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    llm_pricing_override: dict[str, tuple[float, float]] = Field(default_factory=dict)


@lru_cache
def get_settings() -> Settings:
    return Settings()
