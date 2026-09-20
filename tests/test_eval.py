"""The evaluation harness itself: scoring functions (unit) and a full golden run
against the Fake provider with CI thresholds (integration)."""

from __future__ import annotations

import pytest

from evals.runner import check_thresholds, load_suites, run_all, score_answer, score_citations, summarise
from tests.conftest import integration


def test_score_answer_groups_are_any_of():
    hit, missing = score_answer("Revenue was $12.4 million.", [["$12.4 million", "12.4M"], ["million"]])
    assert hit and missing == []
    hit, missing = score_answer("Revenue grew.", [["$12.4 million", "12.4M"]])
    assert not hit and missing == ["$12.4 million | 12.4M"]


def test_score_citations_precision_recall():
    cits = [{"filename": "a.txt", "page_start": 1}, {"filename": "b.md", "page_start": 2}]
    p, r = score_citations(cits, [{"file": "a.txt", "page": 1}])
    assert p == 0.5 and r == 1.0
    p, r = score_citations(cits, [{"file": "a.txt"}, {"file": "c.pdf", "page": 3}])
    assert p == 0.5 and r == 0.5
    assert score_citations([], [{"file": "a.txt"}]) == (0.0, 0.0)
    assert score_citations(cits, []) == (None, None)


def test_golden_suites_are_well_formed():
    suites = load_suites()
    assert len(suites) >= 2
    for s in suites:
        assert s["documents"] and s["cases"]
        files = {d["file"] for d in s["documents"]}
        for c in s["cases"]:
            for turn in c.get("turns") or [c]:
                assert turn["question"]
                assert turn.get("expect_any") or turn.get("expect_no_answer")
                for src in turn.get("expect_sources", []):
                    assert src["file"] in files


def test_thresholds_apply_to_summary():
    from evals.runner import CaseResult

    good = [CaseResult("s", "q", None, "a", True, [], 1.0, 1.0, None, 10, 0.0)]
    assert check_thresholds(summarise("fake", good), "fake") == []
    bad = [CaseResult("s", "q", None, "a", False, ["x"], 0.0, 0.0, None, 10, 0.0)]
    problems = check_thresholds(summarise("fake", bad), "fake")
    assert any(p.startswith("answer_accuracy") for p in problems)


@integration
@pytest.mark.asyncio
async def test_golden_eval_meets_ci_thresholds():
    summary, results = await run_all("fake")
    assert summary.cases >= 10
    problems = check_thresholds(summary, "fake")
    assert not problems, f"{problems}\nfailures: {summary.failures}"
