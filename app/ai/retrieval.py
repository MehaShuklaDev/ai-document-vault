"""Hybrid retrieval: pgvector cosine search ∪ PostgreSQL full-text search,
fused with Reciprocal Rank Fusion, trimmed to a token budget.

Why hybrid: dense vectors capture paraphrase ("cost" ≈ "price") but are weak on
exact tokens (invoice numbers, proper nouns, codes). BM25-style FTS is the
opposite. RRF needs no score calibration between the two, which is why it is
preferred over weighted score sums.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings


@dataclass
class RetrievedChunk:
    id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int
    page_start: int | None
    page_end: int | None
    section: str | None
    score: float  # fused RRF score
    vector_rank: int | None = None
    fts_rank: int | None = None
    rerank_score: float | None = None


_COLS = "c.id, c.document_id, c.version_id, c.chunk_index, c.content, c.token_count, c.page_start, c.page_end, c.section"

_VECTOR_SQL = text(
    f"""
    SELECT {_COLS}
    FROM chunks c
    WHERE c.owner_id = :owner_id AND c.version_id IN :version_ids AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> CAST(:qvec AS vector)
    LIMIT :k
    """
).bindparams(bindparam("version_ids", expanding=True, type_=PG_UUID(as_uuid=True)))

_FTS_SQL = text(
    f"""
    SELECT {_COLS}
    FROM chunks c
    WHERE c.owner_id = :owner_id AND c.version_id IN :version_ids
      AND c.content_tsv @@ websearch_to_tsquery('english', :q)
    ORDER BY ts_rank_cd(c.content_tsv, websearch_to_tsquery('english', :q)) DESC
    LIMIT :k
    """
).bindparams(bindparam("version_ids", expanding=True, type_=PG_UUID(as_uuid=True)))


def _row_to_chunk(row) -> RetrievedChunk:
    return RetrievedChunk(
        id=row.id,
        document_id=row.document_id,
        version_id=row.version_id,
        chunk_index=row.chunk_index,
        content=row.content,
        token_count=row.token_count,
        page_start=row.page_start,
        page_end=row.page_end,
        section=row.section,
        score=0.0,
    )


def rrf_fuse(ranked_lists: list[list[uuid.UUID]], k: int = 60) -> dict[uuid.UUID, float]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d))."""
    scores: dict[uuid.UUID, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def trim_to_budget(chunks: list[RetrievedChunk], budget: int) -> list[RetrievedChunk]:
    out, used = [], 0
    for c in chunks:
        if used + c.token_count > budget and out:
            break
        out.append(c)
        used += c.token_count
    return out


async def hybrid_search(
    db: AsyncSession,
    *,
    owner_id: str,
    version_ids: list[uuid.UUID],
    query_text: str,
    query_vector: list[float],
    top_k: int | None = None,
    token_budget: int | None = None,
    trim: bool = True,
) -> list[RetrievedChunk]:
    """Return fused top_k chunks. With ``trim=False`` the caller (e.g. a reranker)
    receives the full candidate list, untrimmed, and applies the budget itself."""
    s = get_settings()
    top_k = top_k or s.retrieval_top_k
    token_budget = token_budget or s.context_token_budget
    if not version_ids:
        return []

    qvec = "[" + ",".join(f"{x:.7f}" for x in query_vector) + "]"
    vec_rows = (await db.execute(_VECTOR_SQL, {"owner_id": owner_id, "version_ids": version_ids, "qvec": qvec, "k": s.retrieval_vector_candidates})).all()
    fts_rows = (await db.execute(_FTS_SQL, {"owner_id": owner_id, "version_ids": version_ids, "q": query_text, "k": s.retrieval_fts_candidates})).all()

    by_id: dict[uuid.UUID, RetrievedChunk] = {}
    for rank, r in enumerate(vec_rows, start=1):
        c = by_id.setdefault(r.id, _row_to_chunk(r))
        c.vector_rank = rank
    for rank, r in enumerate(fts_rows, start=1):
        c = by_id.setdefault(r.id, _row_to_chunk(r))
        c.fts_rank = rank

    fused = rrf_fuse([[r.id for r in vec_rows], [r.id for r in fts_rows]], k=s.rrf_k)
    for cid, sc in fused.items():
        by_id[cid].score = sc
    ranked = sorted(by_id.values(), key=lambda c: c.score, reverse=True)[:top_k]
    return trim_to_budget(ranked, token_budget) if trim else ranked
