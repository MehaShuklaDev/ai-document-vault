"""Run the golden evaluation suites and print a report.

    python scripts/eval.py                 # provider from env (default fake)
    python scripts/eval.py --json out.json --md report.md --strict

--strict exits non-zero when any threshold in evals/thresholds.yaml is missed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.runner import check_thresholds, dump_json, provider_kind, run_all, to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write full results to this file")
    ap.add_argument("--md", help="write a markdown report to this file")
    ap.add_argument("--strict", action="store_true", help="exit 1 if thresholds are missed")
    a = ap.parse_args()

    summary, results = asyncio.run(run_all(provider_kind()))
    md = to_markdown(summary, results)
    print(md)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        dump_json(summary, results, a.json)
    problems = check_thresholds(summary, provider_kind())
    if problems:
        print("\nTHRESHOLDS MISSED: " + "; ".join(problems))
        return 1 if a.strict else 0
    print("\nAll thresholds met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
