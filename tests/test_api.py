"""API integration tests: real Postgres (pgvector) + Redis, fake AI providers,
pipeline executed inline instead of via a Celery worker."""

from __future__ import annotations

import json
import uuid

import pytest

from tests.conftest import SAMPLE_MD, SAMPLE_TXT, integration

pytestmark = [integration, pytest.mark.asyncio]
P = "/api/v1"


async def _upload(client, name: str, data: bytes, mime: str = "text/plain", **form):
    r = await client.post(f"{P}/documents", files={"file": (name, data, mime)}, data=form)
    assert r.status_code == 202, r.text
    return r.json()


async def test_upload_process_insights_and_status(client, run_inline):
    up = await _upload(client, "report.txt", SAMPLE_TXT)
    doc = up["document"]
    assert up["deduplicated"] is False and doc["status"] in ("uploaded", "ready")

    r = await client.get(f"{P}/documents/{doc['id']}/status")
    st = r.json()
    assert st["status"] == "ready" and st["progress"] == 1.0
    stages = {s["stage"]: s for s in st["stages"]}
    assert {"pipeline", "extract", "chunk", "embed", "analyse"} <= set(stages)
    assert all(s["status"] == "succeeded" for s in stages.values())
    assert stages["chunk"]["meta"]["chunks"] >= 1

    r = await client.get(f"{P}/documents/{doc['id']}/insights")
    ins = r.json()
    assert r.status_code == 200 and ins["summary"] and ins["category"] and ins["suggested_questions"]

    r = await client.get(f"{P}/documents/{doc['id']}")
    d = r.json()
    assert d["status"] == "ready" and d["chunk_count"] >= 1 and d["title"] and len(d["versions"]) == 1


async def test_dedup_and_versioning(client, run_inline):
    a = await _upload(client, "report.txt", SAMPLE_TXT)
    b = await _upload(client, "report-copy.txt", SAMPLE_TXT)
    assert b["deduplicated"] is True and b["document"]["id"] == a["document"]["id"]

    v2 = await _upload(client, "report-v2.txt", SAMPLE_TXT + b"\nAddendum: revised guidance.", replace_document_id=a["document"]["id"])
    assert v2["new_version"] is True and v2["document"]["version_count"] == 2
    r = await client.get(f"{P}/documents/{a['document']['id']}")
    assert [v["version"] for v in r.json()["versions"]] == [1, 2]


async def test_upload_validation(client, run_inline):
    r = await client.post(f"{P}/documents", files={"file": ("x.bin", b"\x00\x01\x02\x03binary", "application/octet-stream")})
    assert r.status_code == 415
    r = await client.post(f"{P}/documents", files={"file": ("empty.txt", b"", "text/plain")})
    assert r.status_code == 400


async def test_tenant_isolation(client, run_inline):
    up = await _upload(client, "report.txt", SAMPLE_TXT)
    other = await client.get(f"{P}/documents/{up['document']['id']}", headers={"X-User-Id": "someone-else"})
    assert other.status_code == 404
    r = await client.get(f"{P}/documents", headers={"X-User-Id": "someone-else"})
    assert r.json()["total"] == 0


async def test_chat_multi_turn_citations_cache_and_history(client, run_inline):
    up = await _upload(client, "report.txt", SAMPLE_TXT)
    doc_id = up["document"]["id"]
    r = await client.post(f"{P}/chat/sessions", json={"document_ids": [doc_id]})
    assert r.status_code == 201
    sid = r.json()["id"]

    r = await client.post(f"{P}/chat/sessions/{sid}/ask", json={"question": "Who leads the Berlin office?"})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["cached"] is False
    ans = a["answer"]
    assert "Maria Fontaine" in ans["content"]
    assert ans["citations"] and ans["citations"][0]["filename"] == "report.txt" and ans["citations"][0]["page_start"] == 1
    assert all(f"[{c['n']}]" in ans["content"] for c in ans["citations"])
    assert len(ans["follow_ups"]) == 3
    assert ans["usage"]["calls"]

    # follow-up is condensed using history
    r = await client.post(f"{P}/chat/sessions/{sid}/ask", json={"question": "and what about the disputed invoice?"})
    b = r.json()
    assert b["question"]["standalone_question"] and "Berlin" in b["question"]["standalone_question"]
    assert "INV-88213" in b["answer"]["content"]

    # identical question → answer cache
    r = await client.post(f"{P}/chat/sessions/{sid}/ask", json={"question": "Who leads the Berlin office?"})
    assert r.json()["cached"] is True

    r = await client.get(f"{P}/chat/sessions/{sid}")
    h = r.json()
    assert h["session"]["message_count"] == 6 and [m["seq"] for m in h["messages"]] == [1, 2, 3, 4, 5, 6]


async def test_multi_document_chat(client, run_inline):
    a = await _upload(client, "report.txt", SAMPLE_TXT)
    b = await _upload(client, "handbook.md", SAMPLE_MD, "text/markdown")
    r = await client.post(f"{P}/chat/sessions", json={"document_ids": [a["document"]["id"], b["document"]["id"]]})
    sid = r.json()["id"]
    r = await client.post(f"{P}/chat/sessions/{sid}/ask", json={"question": "How many vacation days accrue per month?"})
    ans = r.json()["answer"]
    assert "1.5" in ans["content"]
    assert ans["citations"][0]["filename"] == "handbook.md"


async def test_chat_stream_sse(client, run_inline):
    up = await _upload(client, "report.txt", SAMPLE_TXT)
    r = await client.post(f"{P}/chat/sessions", json={"document_ids": [up["document"]["id"]]})
    sid = r.json()["id"]
    async with client.stream("POST", f"{P}/chat/sessions/{sid}/ask/stream", json={"question": "What is the Q4 revenue outlook?"}) as resp:
        assert resp.status_code == 200
        body = (await resp.aread()).decode()
    events = [line[7:] for line in body.splitlines() if line.startswith("event: ")]
    assert events[0] == "status" and "citations" in events and "token" in events and events[-1] == "done"
    done = json.loads([line[6:] for line in body.splitlines() if line.startswith("data: ")][-1])
    assert "13.0" in done["final_text"] and done["citations"]


async def test_chat_blocked_until_ready(client, monkeypatch):
    from app.api import documents as documents_api

    monkeypatch.setattr(documents_api, "enqueue_processing", lambda d, v: "noop")  # never processed
    up = await _upload(client, "report.txt", SAMPLE_TXT)
    r = await client.post(f"{P}/chat/sessions", json={"document_ids": [up["document"]["id"]]})
    sid = r.json()["id"]
    r = await client.post(f"{P}/chat/sessions/{sid}/ask", json={"question": "anything"})
    assert r.status_code == 409
    r = await client.get(f"{P}/documents/{up['document']['id']}/insights")
    assert r.status_code == 409


async def test_custom_summary_and_compare(client, run_inline):
    a = await _upload(client, "report.txt", SAMPLE_TXT)
    b = await _upload(client, "handbook.md", SAMPLE_MD, "text/markdown")
    r = await client.post(f"{P}/documents/{a['document']['id']}/summary", json={"length": "short", "tone": "executive", "focus": "risks"})
    assert r.status_code == 200 and r.json()["options"]["focus"] == "risks" and r.json()["cached"] is False
    r2 = await client.post(f"{P}/documents/{a['document']['id']}/summary", json={"length": "short", "tone": "executive", "focus": "risks"})
    assert r2.json()["cached"] is True
    r = await client.post(f"{P}/documents/compare", json={"document_id_a": a["document"]["id"], "document_id_b": b["document"]["id"]})
    assert r.status_code == 200 and r.json()["comparison"]


async def test_metrics_endpoints(client, run_inline):
    up = await _upload(client, "report.txt", SAMPLE_TXT)
    r = await client.post(f"{P}/chat/sessions", json={"document_ids": [up["document"]["id"]]})
    await client.post(f"{P}/chat/sessions/{r.json()['id']}/ask", json={"question": "What was Q3 revenue?"})

    r = await client.get(f"{P}/metrics/documents")
    m = r.json()
    assert m["total"] == 1 and m["by_status"] == {"ready": 1} and m["total_chunks"] >= 1 and m["top_tags"]

    r = await client.get(f"{P}/metrics/processing")
    m = r.json()
    assert m["succeeded"] == 1 and m["failed"] == 0 and {s["stage"] for s in m["stages"]} >= {"extract", "chunk", "embed", "analyse"}

    r = await client.get(f"{P}/metrics/costs")
    m = r.json()
    assert m["total_calls"] >= 3 and {p["key"] for p in m["by_purpose"]} >= {"embed", "analysis", "chat"} and m["chat"]["answers"] == 1


async def test_delete_document_removes_from_chat(client, run_inline):
    up = await _upload(client, "report.txt", SAMPLE_TXT)
    r = await client.delete(f"{P}/documents/{up['document']['id']}")
    assert r.status_code == 204
    r = await client.get(f"{P}/documents/{up['document']['id']}")
    assert r.status_code == 404
    r = await client.post(f"{P}/chat/sessions", json={"document_ids": [up["document"]["id"]]})
    assert r.status_code == 404


async def test_health(client):
    r = await client.get(f"{P}/health/ready")
    assert r.status_code == 200 and r.json()["checks"] == {"database": True, "pgvector": True, "redis": True}


async def test_failed_extraction_marks_document_failed(client, run_inline):
    import io

    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    w.write(buf)
    up = await _upload(client, "blank.pdf", buf.getvalue(), "application/pdf")
    # run_inline raises PermanentError inside run_pipeline; the API wrapper never sees it because
    # the real task catches it — emulate by checking that the stage row recorded the failure
    r = await client.get(f"{P}/documents/{up['document']['id']}/status")
    st = r.json()
    assert any(s["stage"] == "extract" and s["status"] == "failed" for s in st["stages"])


async def test_unknown_session_404(client):
    r = await client.get(f"{P}/chat/sessions/{uuid.uuid4()}")
    assert r.status_code == 404
