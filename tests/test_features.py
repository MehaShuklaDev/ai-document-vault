"""Tests for reranking, structured extraction and the SSE status stream."""

from __future__ import annotations

import json
import uuid

import pytest

from app.ai import rerank as rerank_mod
from app.ai.retrieval import RetrievedChunk
from app.ai.structured import SCHEMAS, StructuredExtraction, extract_structured, extraction_messages, schema_for_category
from tests.conftest import SAMPLE_MD, SAMPLE_TXT, integration

P = "/api/v1"


def _chunk(i: int, content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        chunk_index=i,
        content=content,
        token_count=50,
        page_start=1,
        page_end=1,
        section=None,
        score=score,
    )


# ------------------------------------------------------------------- reranker (unit)


@pytest.mark.asyncio
async def test_lexical_reranker_promotes_relevant_passage(monkeypatch):
    from app.core.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "rerank_enabled", True)
    monkeypatch.setattr(s, "rerank_weight", 0.8)
    rerank_mod.get_reranker.cache_clear()
    monkeypatch.setattr(rerank_mod, "get_reranker", lambda: rerank_mod.LexicalReranker())
    chunks = [
        _chunk(0, "The quarterly dividend was approved by the board.", score=0.03),  # RRF liked this most
        _chunk(1, "The Berlin office is led by Maria Fontaine with forty engineers.", score=0.02),
        _chunk(2, "Supply chain delays in Singapore caused a backlog.", score=0.01),
    ]
    ranked, usage = await rerank_mod.rerank("Who leads the Berlin office?", chunks, top_k=2)
    assert ranked[0].chunk_index == 1 and len(ranked) == 2
    assert ranked[0].rerank_score is not None and ranked[0].rerank_score > ranked[1].rerank_score
    assert usage is None  # lexical reranker costs nothing


@pytest.mark.asyncio
async def test_rerank_disabled_is_passthrough(monkeypatch):
    monkeypatch.setattr(rerank_mod, "get_reranker", lambda: None)
    chunks = [_chunk(0, "a", 0.3), _chunk(1, "b", 0.2), _chunk(2, "c", 0.1)]
    ranked, usage = await rerank_mod.rerank("q", chunks, top_k=2)
    assert [c.chunk_index for c in ranked] == [0, 1] and usage is None


@pytest.mark.asyncio
async def test_llm_reranker_parses_scores_and_falls_back_on_mismatch(monkeypatch):
    from app.ai.providers import Completion, Usage

    class _LLM:
        name, model = "fake", "fake-llm"

        def __init__(self, text):
            self.text = text

        async def acomplete(self, *a, **k):
            return Completion(self.text, Usage("fake", "fake-llm", 10, 5, 1))

    monkeypatch.setattr(rerank_mod, "get_llm", lambda fast=False: _LLM('{"scores": [2, 9]}'))
    scores, usage = await rerank_mod.LLMReranker().score("q", ["a", "b"])
    assert scores == [0.2, 0.9] and usage.input_tokens == 10
    monkeypatch.setattr(rerank_mod, "get_llm", lambda fast=False: _LLM("garbage"))
    scores, _ = await rerank_mod.LLMReranker().score("q", ["a", "b", "c"])
    assert scores == [0.5, 0.5, 0.5]


def test_cross_encoder_missing_dependency_falls_back(monkeypatch):
    from app.core.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "rerank_enabled", True)
    monkeypatch.setattr(s, "rerank_provider", "cross_encoder")
    rerank_mod.get_reranker.cache_clear()
    rr = rerank_mod.get_reranker()
    assert rr is not None and rr.name in ("lexical", "cross_encoder")
    rerank_mod.get_reranker.cache_clear()


# ------------------------------------------------------- structured extraction (unit)


def test_schema_mapping_and_prompt_shape():
    assert schema_for_category("invoice") == "invoice"
    assert schema_for_category("Contract") == "contract"
    assert schema_for_category("presentation") == "generic"
    assert schema_for_category(None) == "generic"
    msgs = extraction_messages("invoice", "Invoice INV-1 total $10")
    assert "EXTRACTION SCHEMA (invoice)" in msgs[1].content and '"total_amount"' in msgs[1].content
    assert "verbatim" in msgs[0].content.lower()
    assert set(SCHEMAS) >= {"invoice", "contract", "financial", "report", "policy", "research", "legal", "generic"}


def test_fake_extraction_fills_schema_from_text():
    ex, usage = extract_structured("report", SAMPLE_TXT.decode())
    assert ex.schema_name == "report" and set(ex.fields) == set(SCHEMAS["report"])
    assert "12.4" in ex.fields["revenue"]
    assert 0 < ex.confidence <= 1 and usage.model == "fake-llm"


def test_structured_extraction_normalises_bad_values():
    ex = StructuredExtraction.model_validate(
        {"schema_name": "invoice", "fields": {"total_amount": 12.5, "line_items": [{"a": 1}, "x", 3], "vendor_name": None}, "confidence": 0.7}
    )
    assert ex.fields["total_amount"] == "12.5" and ex.fields["vendor_name"] == "" and ex.fields["line_items"] == [{"a": 1}, "x"]


# ------------------------------------------------------------------ API integration

pytestmark_integration = [integration, pytest.mark.asyncio]


@integration
@pytest.mark.asyncio
async def test_pipeline_runs_structured_stage_and_endpoints(client, run_inline):
    r = await client.post(f"{P}/documents", files={"file": ("report.txt", SAMPLE_TXT, "text/plain")})
    doc_id = r.json()["document"]["id"]
    st = (await client.get(f"{P}/documents/{doc_id}/status")).json()
    stages = {s["stage"]: s for s in st["stages"]}
    assert stages["structured"]["status"] == "succeeded" and stages["structured"]["meta"]["schema"] == "report"

    r = await client.get(f"{P}/documents/{doc_id}/extraction")
    assert r.status_code == 200
    ex = r.json()
    assert ex["schema_name"] == "report" and "revenue" in ex["fields"] and ex["cached"] is True

    # explicit schema override, then cached on repeat
    r = await client.post(f"{P}/documents/{doc_id}/extraction", json={"schema_name": "invoice"})
    assert r.status_code == 200 and r.json()["schema_name"] == "invoice" and r.json()["cached"] is False
    r2 = await client.post(f"{P}/documents/{doc_id}/extraction", json={"schema_name": "invoice"})
    assert r2.json()["cached"] is True
    r = await client.post(f"{P}/documents/{doc_id}/extraction", json={"schema_name": "nope"})
    assert r.status_code == 400

    r = await client.get(f"{P}/documents/extraction/schemas")
    assert "invoice" in r.json()["schemas"] and r.json()["category_mapping"]["invoice"] == "invoice"

    costs = (await client.get(f"{P}/metrics/costs")).json()
    assert "extraction" in {p["key"] for p in costs["by_purpose"]}


@integration
@pytest.mark.asyncio
async def test_status_stream_sse(client, run_inline):
    r = await client.post(f"{P}/documents", files={"file": ("report.txt", SAMPLE_TXT, "text/plain")})
    doc_id = r.json()["document"]["id"]
    async with client.stream("GET", f"{P}/documents/{doc_id}/status/stream") as resp:
        assert resp.status_code == 200
        body = (await resp.aread()).decode()
    events = [line[7:] for line in body.splitlines() if line.startswith("event: ")]
    assert events[0] == "status" and events[-1] == "done"
    first = json.loads([line[6:] for line in body.splitlines() if line.startswith("data: ")][0])
    assert first["status"] == "ready" and first["progress"] == 1.0


@integration
@pytest.mark.asyncio
async def test_status_stream_unknown_document(client):
    async with client.stream("GET", f"{P}/documents/{uuid.uuid4()}/status/stream") as resp:
        body = (await resp.aread()).decode()
    assert "event: error" in body and "404" in body


@integration
@pytest.mark.asyncio
async def test_chat_with_reranker_enabled(client, run_inline, monkeypatch):
    from app.core.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "rerank_enabled", True)
    monkeypatch.setattr(s, "rerank_provider", "lexical")
    rerank_mod.get_reranker.cache_clear()
    try:
        a = await client.post(f"{P}/documents", files={"file": ("report.txt", SAMPLE_TXT, "text/plain")})
        b = await client.post(f"{P}/documents", files={"file": ("handbook.md", SAMPLE_MD, "text/markdown")})
        sid = (await client.post(f"{P}/chat/sessions", json={"document_ids": [a.json()["document"]["id"], b.json()["document"]["id"]]})).json()["id"]
        r = await client.post(f"{P}/chat/sessions/{sid}/ask", json={"question": "How many vacation days accrue per month?", "use_cache": False})
        assert r.status_code == 200, r.text
        ans = r.json()["answer"]
        assert "1.5" in ans["content"] and ans["citations"][0]["filename"] == "handbook.md"
    finally:
        rerank_mod.get_reranker.cache_clear()
