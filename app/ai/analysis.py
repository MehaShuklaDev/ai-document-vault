"""Document-level AI analysis: summary, key points, category/tags, sentiment,
entities, suggested questions — one structured call.

Long documents do not fit in one prompt. We build a *representative excerpt*:
the first chunks (title, abstract, intro carry the most signal) plus evenly
spaced samples from the rest, within ``ANALYSIS_TOKEN_BUDGET``. This is cheaper
and faster than map-reduce summarisation and good enough for a summary +
metadata pass; the chat path always uses full-fidelity retrieval anyway.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.ai.prompts import analysis_messages, compare_messages
from app.ai.providers import LLMError, Usage, get_llm, parse_json_response

ANALYSIS_TOKEN_BUDGET = 6000
HEAD_CHUNKS = 4

SummaryLength = Literal["short", "medium", "long", "bullets"]
SummaryTone = Literal["neutral", "executive", "casual", "technical"]


class Sentiment(BaseModel):
    label: Literal["positive", "neutral", "negative", "mixed"] = "neutral"
    score: float = Field(0.0, ge=-1, le=1)
    rationale: str = ""


class Entities(BaseModel):
    people: list[str] = []
    organizations: list[str] = []
    dates: list[str] = []
    amounts: list[str] = []

    @field_validator("people", "organizations", "dates", "amounts", mode="before")
    @classmethod
    def _cap(cls, v):
        return list(dict.fromkeys(str(x) for x in (v or [])))[:10]


class DocumentAnalysis(BaseModel):
    title: str = ""
    summary: str = ""
    key_points: list[str] = []
    category: str = "other"
    tags: list[str] = []
    sentiment: Sentiment = Sentiment()
    entities: Entities = Entities()
    language: str = "en"
    suggested_questions: list[str] = []

    @field_validator("tags", mode="before")
    @classmethod
    def _norm_tags(cls, v):
        seen, out = set(), []
        for t in v or []:
            t = str(t).strip().lower()[:64]
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out[:8]

    @field_validator("category", mode="before")
    @classmethod
    def _norm_cat(cls, v):
        allowed = {
            "contract",
            "invoice",
            "report",
            "research",
            "policy",
            "manual",
            "correspondence",
            "presentation",
            "legal",
            "financial",
            "technical",
            "other",
        }
        v = str(v or "other").strip().lower()
        return v if v in allowed else "other"

    @field_validator("key_points", "suggested_questions", mode="before")
    @classmethod
    def _cap_list(cls, v):
        return [str(x).strip() for x in (v or []) if str(x).strip()][:8]


class SummaryOptions(BaseModel):
    length: SummaryLength = "medium"
    tone: SummaryTone = "neutral"
    focus: str | None = Field(None, max_length=200)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:16]


def build_excerpt(chunks: list[str], token_counts: list[int], budget: int = ANALYSIS_TOKEN_BUDGET) -> str:
    if not chunks:
        return ""
    picked: list[int] = []
    used = 0
    for i in range(min(HEAD_CHUNKS, len(chunks))):
        if used + token_counts[i] > budget:
            break
        picked.append(i)
        used += token_counts[i]
    rest = list(range(len(picked), len(chunks)))
    if rest:
        remaining = budget - used
        avg = max(1, sum(token_counts[i] for i in rest) // len(rest))
        n = max(0, remaining // avg)
        if n:
            step = max(1, len(rest) // n)
            for i in rest[::step]:
                if used + token_counts[i] > budget:
                    break
                picked.append(i)
                used += token_counts[i]
    parts = [chunks[i] for i in sorted(set(picked))]
    return "\n\n[...]\n\n".join(parts)


def _parse(text: str) -> DocumentAnalysis:
    try:
        return DocumentAnalysis.model_validate(parse_json_response(text))
    except (ValidationError, ValueError) as e:
        raise LLMError(f"Analysis response failed validation: {e}") from e


def analyse_document(excerpt: str, options: SummaryOptions) -> tuple[DocumentAnalysis, Usage]:
    llm = get_llm()
    c = llm.complete(
        analysis_messages(excerpt, length=options.length, tone=options.tone, focus=options.focus), temperature=0.1, max_tokens=1500, json_mode=True
    )
    return _parse(c.text), c.usage


async def aanalyse_document(excerpt: str, options: SummaryOptions) -> tuple[DocumentAnalysis, Usage]:
    llm = get_llm()
    c = await llm.acomplete(
        analysis_messages(excerpt, length=options.length, tone=options.tone, focus=options.focus), temperature=0.1, max_tokens=1500, json_mode=True
    )
    return _parse(c.text), c.usage


class Comparison(BaseModel):
    comparison: str = ""
    similarities: list[str] = []
    differences: list[str] = []
    recommendation: str = ""


async def acompare_documents(doc_a: str, doc_b: str) -> tuple[Comparison, Usage]:
    llm = get_llm()
    c = await llm.acomplete(compare_messages(doc_a, doc_b), temperature=0.2, max_tokens=900, json_mode=True)
    try:
        return Comparison.model_validate(parse_json_response(c.text)), c.usage
    except (ValidationError, ValueError) as e:
        raise LLMError(f"Comparison response failed validation: {e}") from e
