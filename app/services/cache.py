"""Redis helpers: embedding cache, answer cache and a sliding-window rate limiter.

Both sync (worker) and async (API) clients are exposed. All operations degrade
gracefully: if Redis is unreachable the cache simply misses and the limiter
allows the request — availability over strictness.
"""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache

import redis
import redis.asyncio as aredis

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@lru_cache
def sync_redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, socket_timeout=2, socket_connect_timeout=2)


@lru_cache
def async_redis() -> aredis.Redis:
    return aredis.Redis.from_url(get_settings().redis_url, socket_timeout=2, socket_connect_timeout=2)


def _h(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------- embeddings
def embedding_key(model: str, text: str) -> str:
    return f"emb:{_h(model, text)}"


def get_cached_embeddings(model: str, texts: list[str]) -> list[list[float] | None]:
    if not texts:
        return []
    try:
        raw = sync_redis().mget([embedding_key(model, t) for t in texts])
        return [json.loads(r) if r else None for r in raw]
    except redis.RedisError as e:
        log.warning("embedding_cache_unavailable", error=str(e))
        return [None] * len(texts)


def set_cached_embeddings(model: str, texts: list[str], vectors: list[list[float]]) -> None:
    if not texts:
        return
    ttl = get_settings().embedding_cache_ttl_seconds
    try:
        pipe = sync_redis().pipeline(transaction=False)
        for t, v in zip(texts, vectors, strict=True):
            pipe.set(embedding_key(model, t), json.dumps(v), ex=ttl)
        pipe.execute()
    except redis.RedisError as e:
        log.warning("embedding_cache_write_failed", error=str(e))


async def aget_cached_embedding(model: str, text: str) -> list[float] | None:
    try:
        raw = await async_redis().get(embedding_key(model, text))
        return json.loads(raw) if raw else None
    except redis.RedisError:
        return None


async def aset_cached_embedding(model: str, text: str, vector: list[float]) -> None:
    try:
        await async_redis().set(embedding_key(model, text), json.dumps(vector), ex=get_settings().embedding_cache_ttl_seconds)
    except redis.RedisError:
        pass


# ------------------------------------------------------------------- answers
def answer_key(owner_id: str, version_ids: list[str], question: str, model: str) -> str:
    return f"ans:{_h(owner_id, model, ','.join(sorted(version_ids)), question.strip().lower())}"


async def aget_cached_answer(key: str) -> dict | None:
    try:
        raw = await async_redis().get(key)
        return json.loads(raw) if raw else None
    except redis.RedisError:
        return None


async def aset_cached_answer(key: str, payload: dict) -> None:
    try:
        await async_redis().set(key, json.dumps(payload, default=str), ex=get_settings().answer_cache_ttl_seconds)
    except redis.RedisError:
        pass


# -------------------------------------------------------------- rate limiting
_SLIDING_WINDOW_LUA = """
local key, now, window, limit = KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count < limit then
  redis.call('ZADD', key, now, now .. '-' .. math.random())
  redis.call('PEXPIRE', key, window)
  return {1, limit - count - 1, 0}
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
local retry = window - (now - tonumber(oldest[2]))
return {0, 0, retry}
"""


async def check_rate_limit(bucket: str, identity: str, limit_per_minute: int) -> tuple[bool, int, int]:
    """Returns (allowed, remaining, retry_after_ms)."""
    s = get_settings()
    if not s.rate_limit_enabled:
        return True, limit_per_minute, 0
    try:
        r = async_redis()
        res = await r.eval(_SLIDING_WINDOW_LUA, 1, f"rl:{bucket}:{identity}", int(time.time() * 1000), 60_000, limit_per_minute)
        return bool(res[0]), int(res[1]), int(res[2])
    except redis.RedisError as e:
        log.warning("rate_limiter_unavailable", error=str(e))
        return True, limit_per_minute, 0


async def redis_ping() -> bool:
    try:
        return bool(await async_redis().ping())
    except Exception:
        return False
