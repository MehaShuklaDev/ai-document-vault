"""Shared FastAPI dependencies: identity and rate limiting."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, Header, HTTPException, Request, status

from app.core.config import get_settings
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
