"""Offline demo: one gold case end to end, no keys, no services.

Creates a throwaway SQLite store + local object dir, runs the Helvion
vendor-diligence case through intake, the plan gate, execution, and
the report gate, and prints the composed draft.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .config import load_settings
from .corpus import GOLD_CASES
from .models import BriefIn
from .service import CaseService


def main() -> None:
    scratch = Path(tempfile.mkdtemp(prefix="deep-diligence-demo-"))
    settings = load_settings(
        sqlite_path=str(scratch / "diligence.db"),
        object_dir=str(scratch / "objects"),
    )
    service = CaseService(settings)
    gold = GOLD_CASES[0]
    print(f"== Demo case: {gold['key']} (offline stub model, generated corpus)\n")
    case = service.create_case(BriefIn(**gold["brief"]))
    case_id = case["case_id"]
    evidence = service.store.list_evidence(case_id)
    print(f"Case {case_id}: {len(evidence)} evidence documents ingested.")
    plan = service.run_planning(case_id)
    print(f"Plan v{plan['version']} generated — awaiting the analyst (plan gate).")
    service.decide_plan(case_id, analyst="Demo Analyst", approve=True)
    print("Plan approved. Executing workstreams...")
    report = service.run_execution(case_id)
    coverage = report["coverage"]
    print(f"Draft composed. Coverage: {coverage}. Refused claims: {report['refused_claims']}")
    service.decide_report(case_id, analyst="Demo Analyst", approve=True)
    case = service.store.get_case(case_id)
    print(f"Report approved. Final case status: {case['status']}\n")
    print(report["markdown"])
    print(f"(scratch data in {scratch})")


if __name__ == "__main__":
    main()
