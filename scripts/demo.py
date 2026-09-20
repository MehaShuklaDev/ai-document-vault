"""End-to-end demo against a running API.

    python scripts/demo.py [--api http://localhost:8000] [--file path.pdf ...]

Uploads sample documents (or the ones you pass), waits for processing, prints
insights, opens a multi-document chat session, asks three questions (including a
follow-up that relies on conversation context), streams one answer over SSE, and
finishes with the metrics endpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

SAMPLES = {
    "acme-q3-report.txt": """ACME Corp Quarterly Report Q3 2025

EXECUTIVE SUMMARY
Revenue for the third quarter reached $12.4 million, an increase of 18% year over year. Operating margin improved to 21%. The board approved a dividend of $0.15 per share payable on November 30, 2025.

RISKS
Supply chain delays in the Singapore facility caused a two-week shipment backlog. Invoice INV-88213 from Delta Logistics totalling $48,900 remains disputed. Management expects resolution by December.

OUTLOOK
ACME expects Q4 revenue between $13.0 and $13.6 million. The new Berlin office opens in January 2026 with 40 engineers led by Maria Fontaine.
""",
    "employee-handbook.md": """# Employee Handbook

## Vacation Policy
Full-time employees accrue 1.5 vacation days per month, capped at 24 days. Requests must be submitted two weeks in advance through the HR portal.

## Remote Work
Employees may work remotely up to three days per week with manager approval. Core hours are 10:00 to 15:00 local time.

## Expenses
Expenses above $500 require pre-approval from a director. Receipts must be uploaded within 30 days.
""",
}


def hr(title: str) -> None:
    print(f"\n\033[1m── {title} " + "─" * max(0, 70 - len(title)) + "\033[0m")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--user", default="demo-user")
    ap.add_argument("--file", action="append", default=[], help="document(s) to upload instead of the samples")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.api + "/api/v1", headers={"X-User-Id": a.user}, timeout=60)

    hr("health")
    print(json.dumps(c.get("/health/ready").json(), indent=2))
    print(json.dumps(c.get("/chat/providers").json()))

    hr("upload")
    files = [(Path(f).name, Path(f).read_bytes()) for f in a.file] or [(n, t.encode()) for n, t in SAMPLES.items()]
    doc_ids = []
    for name, data in files:
        r = c.post("/documents", files={"file": (name, data)})
        r.raise_for_status()
        j = r.json()
        doc_ids.append(j["document"]["id"])
        print(f"  {name:28s} → {j['document']['id']}  dedup={j['deduplicated']} task={j.get('task_id')}")

    hr("processing (SSE /status/stream for the first document)")
    with c.stream("GET", f"/documents/{doc_ids[0]}/status/stream") as r:
        ev = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: ") and ev == "status":
                d = json.loads(line[6:])
                print(
                    f"  ⟳ {d['status']:10s} {int(d['progress'] * 100):3d}%  "
                    + " ".join(f"{s['stage']}:{s['status'][0]}" for s in d["stages"] if s["stage"] != "pipeline")
                )
            elif ev == "done":
                break

    hr("processing (polling /status for the rest)")
    t0 = time.time()
    pending = set(doc_ids)
    while pending and time.time() - t0 < 180:
        for did in list(pending):
            st = c.get(f"/documents/{did}/status").json()
            stages = " ".join(f"{s['stage']}:{s['status'][0]}" for s in st["stages"] if s["stage"] != "pipeline")
            print(f"  {did[:8]} {st['status']:10s} {int(st['progress']*100):3d}%  {stages}" + (f"  ERROR {st['error']}" if st["error"] else ""))
            if st["status"] in ("ready", "failed"):
                pending.discard(did)
        if pending:
            time.sleep(1.5)
    if pending:
        print("  still processing — is a worker running? (make worker)")
        return 1

    hr("insights")
    for did in doc_ids:
        i = c.get(f"/documents/{did}/insights").json()
        print(
            f"  [{i['category']}] {i['title']}\n    {i['summary'][:200]}\n    tags={i['tags']} sentiment={i['sentiment']['label']}\n    suggested: {i['suggested_questions'][:2]}"
        )

    hr("structured extraction (schema chosen from AI category)")
    for did in doc_ids:
        r = c.get(f"/documents/{did}/extraction")
        if r.status_code == 200:
            ex = r.json()
            filled = {k: v for k, v in ex["fields"].items() if v}
            print(f"  {did[:8]} schema={ex['schema_name']} confidence={ex['confidence']} filled={len(filled)}/{len(ex['fields'])}")
            for k, v in list(filled.items())[:5]:
                print(f"     {k}: {str(v)[:90]}")
        else:
            print(f"  {did[:8]} no extraction ({r.status_code})")

    hr("custom summary (short / executive / focus=risks)")
    s = c.post(f"/documents/{doc_ids[0]}/summary", json={"length": "short", "tone": "executive", "focus": "risks"}).json()
    print("  " + s["summary"][:300])

    hr("multi-document chat")
    sid = c.post("/chat/sessions", json={"document_ids": doc_ids}).json()["id"]
    print(f"  session {sid}")
    for q in [
        "What was the Q3 revenue and who leads the Berlin office?",
        "and what about the disputed invoice?",
        "How many vacation days do employees accrue per month?",
    ]:
        r = c.post(f"/chat/sessions/{sid}/ask", json={"question": q}).json()
        ans = r["answer"]
        print(f"\n  Q: {q}")
        if r["question"]["standalone_question"]:
            print(f"     (condensed → {r['question']['standalone_question']})")
        print(f"  A: {ans['content'][:400]}")
        for cit in ans["citations"]:
            print(f"     [{cit['n']}] {cit['filename']} p.{cit['page_start']}  “{cit['snippet'][:70]}…”")
        print(f"     follow-ups: {ans['follow_ups']}   cost=${ans['usage']['cost_usd']:.5f} cached={r['cached']}")

    hr("streaming (SSE)")
    with c.stream("POST", f"/chat/sessions/{sid}/ask/stream", json={"question": "What is the Q4 outlook?"}) as r:
        ev = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: ") and ev:
                d = json.loads(line[6:])
                if ev == "token":
                    sys.stdout.write(d["t"])
                    sys.stdout.flush()
                elif ev == "status":
                    print(f"  ⟳ {d['stage']}")
                elif ev == "citations":
                    print(f"  sources: {[x['filename'] for x in d]}\n  ", end="")
                elif ev == "done":
                    print(f"\n  ✓ persisted message {d['message_id']}")

    hr("metrics")
    print("  documents :", json.dumps(c.get("/metrics/documents").json())[:300])
    pm = c.get("/metrics/processing").json()
    print(
        f"  processing: succeeded={pm['succeeded']} failed={pm['failed']} failure_rate={pm['failure_rate']} "
        + " ".join(f"{s['stage']}(p50={s['p50_ms']}ms)" for s in pm["stages"])
    )
    cm = c.get("/metrics/costs").json()
    print(
        f"  costs     : total=${cm['total_cost_usd']} calls={cm['total_calls']} cache_hit_rate={cm['cache_hit_rate']} by_purpose={[(p['key'], p['calls']) for p in cm['by_purpose']]}"
    )
    print("\nOpen http://localhost:8000/ for the dashboard, http://localhost:8000/docs for the API.")
    print("Run `python scripts/eval.py` for the golden-set evaluation report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
