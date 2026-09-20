"""Shared FastAPI dependencies: identity and rate limiting."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import get_db
from app.models import AIUsage
from app.services.cache import check_rate_limit


async def get_current_user(x_user_id: str | None = Header(default=None, alias="X-User-Id")) -> str:
    """Identity placeholder. Swap for JWT/OIDC verification without touching routers.

    Every document, chunk and session is scoped to this id.
    """
    uid = (x_user_id or get_settings().default_user_id).strip()
    if not uid or len(uid) > 128:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid X-User-Id")
    return uid


def rate_limited(bucket: str, per_minute_attr: str) -> Callable:
    async def _dep(request: Request, user: str = Depends(get_current_user)) -> None:
        limit = getattr(get_settings(), per_minute_attr)
        allowed, remaining, retry_ms = await check_rate_limit(bucket, user, limit)
        request.state.rate_remaining = remaining
        if not allowed:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"rate limit exceeded for {bucket}: {limit}/min",
                headers={"Retry-After": str(max(1, retry_ms // 1000)), "X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"},
            )

    return _dep


chat_rate_limit = rate_limited("chat", "rate_limit_chat_per_minute")
upload_rate_limit = rate_limited("upload", "rate_limit_upload_per_minute")


async def quota_usage(db: AsyncSession, user: str) -> dict:
    """Rolling-24h spend and tokens for a user, plus the configured limits."""
    s = get_settings()
    since = datetime.now(UTC) - timedelta(hours=24)
    row = (
        await db.execute(
            select(func.coalesce(func.sum(AIUsage.cost_usd), 0), func.coalesce(func.sum(AIUsage.input_tokens + AIUsage.output_tokens), 0)).where(
                AIUsage.owner_id == user, AIUsage.created_at >= since
            )
        )
    ).one()
    used_usd, used_tokens = float(row[0]), int(row[1])
    return {
        "window_hours": 24,
        "used_usd": round(used_usd, 6),
        "limit_usd": s.quota_usd_per_user_per_day or None,
        "used_tokens": used_tokens,
        "limit_tokens": s.quota_tokens_per_user_per_day or None,
        "remaining_usd": round(s.quota_usd_per_user_per_day - used_usd, 6) if s.quota_usd_per_user_per_day else None,
        "remaining_tokens": max(0, s.quota_tokens_per_user_per_day - used_tokens) if s.quota_tokens_per_user_per_day else None,
        "exceeded": bool(
            (s.quota_usd_per_user_per_day and used_usd >= s.quota_usd_per_user_per_day)
            or (s.quota_tokens_per_user_per_day and used_tokens >= s.quota_tokens_per_user_per_day)
        ),
    }


async def quota_check(request: Request, db: AsyncSession = Depends(get_db), user: str = Depends(get_current_user)) -> None:
    """Block AI-spending endpoints once a user's rolling-24h USD or token quota is exhausted.

    Rate limiting bounds *burst*; quota bounds *total spend*. Both are enforced per user.
    Disabled (0) by default so the demo is frictionless; set QUOTA_USD_PER_USER_PER_DAY in prod.
    """
    s = get_settings()
    if not (s.quota_usd_per_user_per_day or s.quota_tokens_per_user_per_day):
        return
    q = await quota_usage(db, user)
    request.state.quota = q
    if q["exceeded"]:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"daily AI quota exhausted: ${q['used_usd']:.4f} of ${q['limit_usd']} · "
                f"{q['used_tokens']} of {q['limit_tokens'] or '∞'} tokens (rolling 24h)"
            ),
            headers={"Retry-After": "3600", "X-Quota-Used-USD": str(q["used_usd"]), "X-Quota-Limit-USD": str(q["limit_usd"] or "")},
        )
