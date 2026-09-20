"""Recursive, token-aware, structure-first chunker.

Algorithm
---------
1. Each page is split into *segments* on blank lines (paragraphs). Markdown-style
   headings become the ``section`` label for following segments.
2. Segments longer than the chunk budget are recursively split on sentence
   boundaries, then on raw token windows as a last resort.
3. Segments are greedily packed into chunks of at most ``chunk_tokens`` tokens.
   Packing may cross page boundaries (context is more valuable than page purity),
   and the resulting chunk records ``page_start``/``page_end`` for citations.
4. The last ``overlap_tokens`` tokens of a chunk are prepended to the next one so
   a fact that straddles a boundary is retrievable from at least one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import tiktoken

from app.core.logging import get_logger
from app.services.extraction import Page

log = get_logger(__name__)


class _ApproxEncoding:
    """Offline fallback when the tiktoken BPE file cannot be downloaded (air-gapped
    or TLS-intercepted networks). Tokens are words / whitespace / punctuation, which
    over-counts BPE by ~25% — conservative for chunk budgets. ``decode`` is exact."""

    _tok = re.compile(r"\w+|\s+|[^\w\s]")

    def encode(self, text: str, disallowed_special=()) -> list[str]:
        return self._tok.findall(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


@lru_cache
def _enc(name: str = "cl100k_base"):
    try:
        return tiktoken.get_encoding(name)
    except Exception as e:  # network / TLS failure fetching the BPE ranks
        log.warning("tiktoken_unavailable_using_approx_tokenizer", error=str(e)[:200])
        return _ApproxEncoding()


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    return len(_enc(encoding).encode(text, disallowed_special=()))


@dataclass
class Segment:
    text: str
    page: int
    section: str | None
    tokens: int


@dataclass
class ChunkOut:
    index: int
    content: str
    token_count: int
    page_start: int
    page_end: int
    section: str | None
    segments: list[Segment] = field(default_factory=list, repr=False)


_HEADING = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 ,&\-/]{4,60})$")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT.split(text)
    return [p for p in parts if p.strip()]


def _split_tokens(text: str, budget: int, encoding: str) -> list[str]:
    enc = _enc(encoding)
    ids = enc.encode(text, disallowed_special=())
    return [enc.decode(ids[i : i + budget]) for i in range(0, len(ids), budget)]


def _segments_from_page(page: Page, budget: int, encoding: str) -> list[Segment]:
    out: list[Segment] = []
    section = page.section
    for para in re.split(r"\n\s*\n", page.text):
        para = para.strip()
        if not para:
            continue
        if _HEADING.match(para) and len(para) < 120:
            section = para.lstrip("# ").strip()
        pieces = [para]
        if count_tokens(para, encoding) > budget:
            pieces = []
            for sent in _split_sentences(para):
                if count_tokens(sent, encoding) > budget:
                    pieces.extend(_split_tokens(sent, budget, encoding))
                else:
                    pieces.append(sent)
        for piece in pieces:
            out.append(Segment(text=piece, page=page.number, section=section, tokens=count_tokens(piece, encoding)))
    return out


def _tail_tokens(text: str, n: int, encoding: str) -> str:
    if n <= 0:
        return ""
    enc = _enc(encoding)
    ids = enc.encode(text, disallowed_special=())
    return enc.decode(ids[-n:]) if len(ids) > n else text


def chunk_pages(
    pages: list[Page],
    *,
    chunk_tokens: int = 400,
    overlap_tokens: int = 60,
    encoding: str = "cl100k_base",
    max_chunks: int | None = None,
) -> list[ChunkOut]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    segments: list[Segment] = []
    for p in pages:
        segments.extend(_segments_from_page(p, chunk_tokens, encoding))

    chunks: list[ChunkOut] = []
    cur: list[Segment] = []
    cur_tokens = 0
    carry = ""  # overlap text from previous chunk

    def flush():
        nonlocal cur, cur_tokens, carry
        if not cur:
            return
        body = "\n\n".join(s.text for s in cur)
        content = (carry + "\n\n" + body).strip() if carry else body
        chunks.append(
            ChunkOut(
                index=len(chunks),
                content=content,
                token_count=count_tokens(content, encoding),
                page_start=cur[0].page,
                page_end=cur[-1].page,
                section=cur[0].section,
                segments=cur,
            )
        )
        carry = _tail_tokens(body, overlap_tokens, encoding)
        cur, cur_tokens = [], 0

    budget = chunk_tokens - overlap_tokens
    for seg in segments:
        if cur and cur_tokens + seg.tokens > budget:
            flush()
            if max_chunks and len(chunks) >= max_chunks:
                break
        cur.append(seg)
        cur_tokens += seg.tokens
    if not (max_chunks and len(chunks) >= max_chunks):
        flush()
    return chunks
