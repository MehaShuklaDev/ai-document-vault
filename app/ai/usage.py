"""Persist every AI call into ``ai_usage`` (sync + async variants)."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.ai.providers import Usage
from app.models import AIUsage


def _row(usage: Usage, purpose: str, owner_id: str | None, document_id: uuid.UUID | None, session_id: uuid.UUID | None) -> AIUsage:
    return AIUsage(
        owner_id=owner_id,
        document_id=document_id,
        session_id=session_id,
        purpose=purpose,
        provider=usage.provider,
        model=usage.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        latency_ms=usage.latency_ms,
        cached=usage.cached,
    )


def record_usage(
    db: Session, usage: Usage, purpose: str, *, owner_id: str | None = None, document_id: uuid.UUID | None = None, session_id: uuid.UUID | None = None
) -> None:
    db.add(_row(usage, purpose, owner_id, document_id, session_id))


async def arecord_usage(
    db: AsyncSession, usage: Usage, purpose: str, *, owner_id: str | None = None, document_id: uuid.UUID | None = None, session_id: uuid.UUID | None = None
) -> None:
    db.add(_row(usage, purpose, owner_id, document_id, session_id))
