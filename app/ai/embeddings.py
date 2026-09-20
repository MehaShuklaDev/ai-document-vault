"""Embedding service: batching + Redis cache in front of the provider.

Batches of up to ``BATCH`` texts go to the provider in one request; texts already
cached are skipped and only the misses are billed. Cache hits are still reported
as a ``Usage`` with ``cached=True`` and zero tokens so metrics show hit rates.
"""

from __future__ import annotations

from app.ai.providers import Usage, get_embeddings
from app.services import cache

BATCH = 100


def embed_texts(texts: list[str]) -> tuple[list[list[float]], list[Usage]]:
    provider = get_embeddings()
    cached = cache.get_cached_embeddings(provider.model, texts)
    result: list[list[float] | None] = list(cached)
    usages: list[Usage] = []
    hits = sum(1 for c in cached if c is not None)
    if hits:
        usages.append(Usage(provider.name, provider.model, 0, 0, 0, cached=True))
    miss_idx = [i for i, v in enumerate(result) if v is None]
    for start in range(0, len(miss_idx), BATCH):
        idx = miss_idx[start : start + BATCH]
        batch = [texts[i] for i in idx]
        vecs, usage = provider.embed(batch)
        usages.append(usage)
        for i, v in zip(idx, vecs, strict=True):
            result[i] = v
        cache.set_cached_embeddings(provider.model, batch, vecs)
    return [v for v in result if v is not None], usages


async def aembed_query(text: str) -> tuple[list[float], Usage]:
    provider = get_embeddings()
    hit = await cache.aget_cached_embedding(provider.model, text)
    if hit is not None:
        return hit, Usage(provider.name, provider.model, 0, 0, 0, cached=True)
    vecs, usage = await provider.aembed([text])
    await cache.aset_cached_embedding(provider.model, text, vecs[0])
    return vecs[0], usage
