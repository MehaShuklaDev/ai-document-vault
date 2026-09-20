"""Second-stage reranking of retrieval candidates, behind a feature flag.

Hybrid search + RRF is a good *recall* stage; a reranker improves *precision* by
scoring each (question, passage) pair jointly instead of comparing independent
embeddings. Three implementations share one protocol:

* ``CrossEncoderReranker`` – sentence-transformers cross-encoder (e.g.
  ``cross-encoder/ms-marco-MiniLM-L-6-v2``). Local, ~20 ms for 20 pairs on CPU,
  no API cost. The dependency is optional and imported lazily.
* ``LLMReranker`` – asks the fast model to score passages 0–10 in one JSON call.
  No extra dependency; costs a few hundred tokens per question.
* ``LexicalReranker`` – deterministic word-overlap scorer used by the Fake
  provider path and as a safe fallback when a model cannot be loaded.

Enable with ``RERANK_ENABLED=true`` and choose ``RERANK_PROVIDER``. Retrieval then
fetches ``rerank_candidates`` (default 20) fused results, reranks, and keeps
``retrieval_top_k``. Scores are blended with the RRF score (``rerank_weight``) so a
passage that both stages like ranks first.
"""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Protocol

from app.ai.providers import Message, Usage, get_llm, parse_json_response
from app.ai.retrieval import RetrievedChunk
from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

_WORD = re.compile(r"[A-Za-z0-9]+")


class Reranker(Protocol):
    name: str

    async def score(self, question: str, passages: list[str]) -> tuple[list[float], Usage | None]: ...


class LexicalReranker:
    """Jaccard-ish overlap on informative tokens; in [0, 1]."""

    name = "lexical"

    async def score(self, question, passages):
        q = {w.lower() for w in _WORD.findall(question) if len(w) > 2}
        out = []
        for p in passages:
            pw = {w.lower() for w in _WORD.findall(p)}
            out.append(len(q & pw) / (len(q) or 1))
        return out, None


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # optional dependency

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    async def score(self, question, passages):
        import asyncio

        t0 = time.perf_counter()
        # CPU-bound → run off the event loop
        raw = await asyncio.to_thread(self._model.predict, [(question, p) for p in passages])
        scores = [float(s) for s in raw]
        lo, hi = min(scores), max(scores)
        norm = [(s - lo) / (hi - lo) if hi > lo else 0.5 for s in scores]
        return norm, Usage("local", self.model_name, 0, 0, int((time.perf_counter() - t0) * 1000))


_RERANK_SYSTEM = """You grade how well each passage answers a question.
Return a JSON object {"scores": [int, ...]} with one integer 0-10 per passage, in order.
10 = directly and completely answers; 5 = partially relevant; 0 = unrelated. Judge only relevance, not writing quality."""


class LLMReranker:
    name = "llm"

    async def score(self, question, passages):
        numbered = "\n\n".join(f"[{i}] {p[:800]}" for i, p in enumerate(passages, start=1))
        msgs = [Message("system", _RERANK_SYSTEM), Message("user", f"Question: {question}\n\nPassages:\n{numbered}")]
        c = await get_llm(fast=True).acomplete(msgs, temperature=0.0, max_tokens=200, json_mode=True)
        try:
            scores = [float(s) / 10.0 for s in parse_json_response(c.text).get("scores", [])]
        except Exception:
            scores = []
        if len(scores) != len(passages):  # malformed → neutral, keep RRF order
            scores = [0.5] * len(passages)
        return scores, c.usage


@lru_cache
def get_reranker() -> Reranker | None:
    s = get_settings()
    if not s.rerank_enabled:
        return None
    if s.rerank_provider == "cross_encoder":
        try:
            return CrossEncoderReranker(s.rerank_model)
        except Exception as e:  # missing dependency / no model download
            log.warning("cross_encoder_unavailable_falling_back_to_lexical", error=str(e)[:200])
            return LexicalReranker()
    if s.rerank_provider == "llm":
        return LexicalReranker() if s.resolved_llm_provider == "fake" else LLMReranker()
    return LexicalReranker()


async def rerank(question: str, chunks: list[RetrievedChunk], top_k: int) -> tuple[list[RetrievedChunk], Usage | None]:
    """Blend reranker score with the (normalised) RRF score and keep top_k."""
    rr = get_reranker()
    if rr is None or len(chunks) <= 1:
        return chunks[:top_k], None
    s = get_settings()
    scores, usage = await rr.score(question, [c.content for c in chunks])
    rrf = [c.score for c in chunks]
    lo, hi = min(rrf), max(rrf)
    rrf_norm = [(x - lo) / (hi - lo) if hi > lo else 0.5 for x in rrf]
    w = s.rerank_weight
    for c, r, base in zip(chunks, scores, rrf_norm, strict=True):
        c.rerank_score = r
        c.score = w * r + (1 - w) * base
    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
    if usage:
        usage.provider = usage.provider or rr.name
    return ranked, usage


def dumps_debug(chunks: list[RetrievedChunk]) -> str:  # pragma: no cover - debugging aid
    return json.dumps([{"idx": c.chunk_index, "score": round(c.score, 4), "rerank": getattr(c, "rerank_score", None)} for c in chunks])
