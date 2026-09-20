"""Evaluation harness for document chat.

Golden cases live in ``evals/golden/*.yaml``: each file bundles a small document
(inline text) and questions with what a correct answer must contain and which
source (file, page) a correct citation must point to. The runner:

1. uploads the documents through the real API (ASGI, in-process) and runs the
   pipeline inline — same code path as production minus the broker;
2. asks every question in a fresh session (and follow-ups in the same session
   to exercise condense-question);
3. scores each answer:

   * **answer_hit**    – all ``expect_any`` groups satisfied (each group = list of
     acceptable substrings, case-insensitive);
   * **citation_precision** – fraction of returned citations that point to an
     expected (file, page) source;
   * **citation_recall** – fraction of expected sources that appear in citations;
   * **abstained** – for ``expect_no_answer`` cases, the model said it could not answer;
   * latency and cost from the message usage;

4. aggregates and compares against thresholds (``evals/thresholds.yaml``), which
   differ for the Fake provider (deterministic, strict) and real providers
   (looser, nightly).

Run: ``python scripts/eval.py [--provider fake|real] [--json out.json]``.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

GOLDEN_DIR = Path(__file__).parent / "golden"
THRESHOLDS = Path(__file__).parent / "thresholds.yaml"
P = "/api/v1"


@dataclass
class CaseResult:
    suite: str
    question: str
    standalone: str | None
    answer: str
    answer_hit: bool
    missing: list[str]
    citation_precision: float | None
    citation_recall: float | None
    abstained: bool | None
    latency_ms: int
    cost_usd: float
    citations: list[dict] = field(default_factory=list)


@dataclass
class Summary:
    provider: str
    cases: int
    answer_accuracy: float
    citation_precision: float
    citation_recall: float
    abstention_accuracy: float | None
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    failures: list[dict]


def load_suites(paths: list[Path] | None = None) -> list[dict]:
    files = paths or sorted(GOLDEN_DIR.glob("*.yaml"))
    return [yaml.safe_load(p.read_text()) | {"_name": p.stem} for p in files]


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def score_answer(answer: str, expect_any: list[list[str]]) -> tuple[bool, list[str]]:
    a = _norm(answer)
    missing = [" | ".join(group) for group in expect_any if not any(_norm(x) in a for x in group)]
    return not missing, missing


def score_citations(citations: list[dict], expected: list[dict]) -> tuple[float | None, float | None]:
    if not expected:
        return None, None
    exp = {(e["file"], e.get("page")) for e in expected}
    got = [(c["filename"], c.get("page_start")) for c in citations]
    if not got:
        return 0.0, 0.0

    def match(g):
        return any(g[0] == f and (p is None or g[1] == p) for f, p in exp)

    precision = sum(1 for g in got if match(g)) / len(got)
    recall = sum(1 for f, p in exp if any(g[0] == f and (p is None or g[1] == p) for g in got)) / len(exp)
    return round(precision, 3), round(recall, 3)


NO_ANSWER_MARKERS = (
    "couldn't find",
    "could not find",
    "do not contain",
    "does not contain",
    "not enough information",
    "no information",
    "not mentioned",
    "isn't mentioned",
    "not specified",
)


async def run_suite(client, suite: dict, inline_pipeline, results: list[CaseResult]) -> None:
    name = suite["_name"]
    doc_ids: dict[str, str] = {}
    for d in suite["documents"]:
        r = await client.post(f"{P}/documents", files={"file": (d["file"], d["text"].encode(), d.get("mime", "text/plain"))})
        r.raise_for_status()
        j = r.json()["document"]
        doc_ids[d["file"]] = j["id"]
        # inline pipeline: the test/runner monkeypatches enqueue; nothing else to do
    # wait for readiness (inline → immediate; real worker → poll)
    for _ in range(120):
        statuses = [(await client.get(f"{P}/documents/{i}")).json()["status"] for i in doc_ids.values()]
        if all(s in ("ready", "failed") for s in statuses):
            break
        await asyncio.sleep(1)
    if any(s == "failed" for s in statuses):
        raise RuntimeError(f"{name}: document processing failed: {statuses}")

    for case in suite["cases"]:
        sid = (await client.post(f"{P}/chat/sessions", json={"document_ids": list(doc_ids.values())})).json()["id"]
        turns = case.get("turns") or [case]
        for turn in turns:
            t0 = time.perf_counter()
            r = await client.post(f"{P}/chat/sessions/{sid}/ask", json={"question": turn["question"], "use_cache": False})
            r.raise_for_status()
            body = r.json()
            ans = body["answer"]
            latency = int((time.perf_counter() - t0) * 1000)
            expect_any = turn.get("expect_any", [])
            hit, missing = score_answer(ans["content"], expect_any) if expect_any else (True, [])
            prec, rec = score_citations(ans.get("citations") or [], turn.get("expect_sources", []))
            abst = None
            if turn.get("expect_no_answer"):
                abst = any(m in ans["content"].lower() for m in NO_ANSWER_MARKERS)
                hit = abst
            results.append(
                CaseResult(
                    suite=name,
                    question=turn["question"],
                    standalone=body["question"].get("standalone_question"),
                    answer=ans["content"],
                    answer_hit=hit,
                    missing=missing,
                    citation_precision=prec,
                    citation_recall=rec,
                    abstained=abst,
                    latency_ms=latency,
                    cost_usd=float((ans.get("usage") or {}).get("cost_usd") or 0),
                    citations=ans.get("citations") or [],
                )
            )


def summarise(provider: str, results: list[CaseResult]) -> Summary:
    lat = sorted(r.latency_ms for r in results) or [0]
    precs = [r.citation_precision for r in results if r.citation_precision is not None]
    recs = [r.citation_recall for r in results if r.citation_recall is not None]
    absts = [r.abstained for r in results if r.abstained is not None]
    return Summary(
        provider=provider,
        cases=len(results),
        answer_accuracy=round(sum(r.answer_hit for r in results) / len(results), 3) if results else 0.0,
        citation_precision=round(statistics.mean(precs), 3) if precs else 0.0,
        citation_recall=round(statistics.mean(recs), 3) if recs else 0.0,
        abstention_accuracy=round(sum(absts) / len(absts), 3) if absts else None,
        p50_latency_ms=statistics.median(lat),
        p95_latency_ms=lat[min(len(lat) - 1, int(0.95 * (len(lat) - 1)))],
        total_cost_usd=round(sum(r.cost_usd for r in results), 6),
        failures=[{"suite": r.suite, "question": r.question, "missing": r.missing, "answer": r.answer[:200]} for r in results if not r.answer_hit],
    )


def check_thresholds(summary: Summary, provider_kind: str) -> list[str]:
    th = yaml.safe_load(THRESHOLDS.read_text())[provider_kind]
    problems = []
    for key, minimum in th.items():
        val = getattr(summary, key)
        if val is None:
            continue
        if val < minimum:
            problems.append(f"{key}={val} < {minimum}")
    return problems


async def run_all(provider_kind: str = "fake", user_id: str | None = None, suites: list[dict] | None = None) -> tuple[Summary, list[CaseResult]]:
    """Run every golden suite in-process. Requires the DB (and Redis for caches) to be up."""
    from httpx import ASGITransport, AsyncClient

    from app.api import documents as documents_api
    from app.main import app
    from app.workers import tasks

    def _inline(document_id, version_id):
        tid = f"eval-{uuid.uuid4().hex[:8]}"
        try:
            tasks.run_pipeline(document_id, version_id, attempt=1, task_id=tid)
        except tasks.PermanentError as e:
            tasks._mark_failed(document_id, version_id, str(e), 1, tid)
        return tid

    original = documents_api.enqueue_processing
    documents_api.enqueue_processing = _inline
    results: list[CaseResult] = []
    try:
        uid = user_id or f"eval-{uuid.uuid4().hex[:8]}"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://eval", headers={"X-User-Id": uid}, timeout=120) as client:
            for suite in suites or load_suites():
                await run_suite(client, suite, _inline, results)
    finally:
        documents_api.enqueue_processing = original
    from app.ai.providers import get_llm

    return summarise(f"{get_llm().name}:{get_llm().model}", results), results


def to_markdown(summary: Summary, results: list[CaseResult]) -> str:
    lines = [
        f"## Eval — {summary.provider}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| cases | {summary.cases} |",
        f"| answer accuracy | {summary.answer_accuracy} |",
        f"| citation precision | {summary.citation_precision} |",
        f"| citation recall | {summary.citation_recall} |",
        f"| abstention accuracy | {summary.abstention_accuracy} |",
        f"| p50 / p95 latency | {summary.p50_latency_ms} / {summary.p95_latency_ms} ms |",
        f"| total cost | ${summary.total_cost_usd} |",
        "",
        "| suite | question | hit | cit P | cit R | ms |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.suite} | {r.question[:60]} | {'✓' if r.answer_hit else '✗ ' + '; '.join(r.missing)[:60]} | {r.citation_precision} | {r.citation_recall} | {r.latency_ms} |"
        )
    return "\n".join(lines)


def dump_json(summary: Summary, results: list[CaseResult], path: str) -> None:
    Path(path).write_text(json.dumps({"summary": asdict(summary), "results": [asdict(r) for r in results]}, indent=2, default=str))


def provider_kind() -> str:
    from app.core.config import get_settings

    return "fake" if get_settings().resolved_llm_provider == "fake" else "real"


__all__: list[str] = [
    "run_all",
    "summarise",
    "check_thresholds",
    "to_markdown",
    "dump_json",
    "load_suites",
    "score_answer",
    "score_citations",
    "provider_kind",
    "Any",
]
