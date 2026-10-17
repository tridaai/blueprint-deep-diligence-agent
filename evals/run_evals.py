#!/usr/bin/env python3
"""Gold case-pack gate for the Deep Diligence Agent (offline).

Runs each gold case end to end on the deterministic stub model over
the bundled corpus and grades:
  * finding recall against the planted gold facts (gate >= 0.90),
  * citation completeness on composed claims (gate == 100%),
  * budget compliance: the run completed inside its caps (gate 100%).

Exit code 0 only when every gate passes for every case.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_diligence.config import load_settings  # noqa: E402
from deep_diligence.corpus import GOLD_CASES  # noqa: E402
from deep_diligence.models import BriefIn  # noqa: E402
from deep_diligence.service import CaseService  # noqa: E402

RECALL_GATE = 0.90


def run_case(gold: dict) -> dict:
    scratch = Path(tempfile.mkdtemp(prefix="deep-diligence-eval-"))
    settings = load_settings(
        sqlite_path=str(scratch / "diligence.db"),
        object_dir=str(scratch / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
    )
    service = CaseService(settings)
    case = service.create_case(BriefIn(**gold["brief"]))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Eval Harness", approve=True)
    report = service.run_execution(case_id)
    case = service.store.get_case(case_id)
    found = {
        f["fact_ref"]
        for f in service.store.list_findings(case_id)
        if f.get("fact_ref")
    }
    gold_facts = set(gold["gold_facts"])
    recall = len(found & gold_facts) / len(gold_facts) if gold_facts else 1.0
    claims = report["claims"]
    cited = [c for c in claims if c["evidence_ids"]]
    citation = len(cited) / len(claims) if claims else 1.0
    usage = case["usage"].get("execution", {})
    budget = gold["brief"]["budget"]
    compliant = (
        case["status"] == "DRAFT_READY"
        and usage.get("steps", 10**9) <= budget["max_steps"]
        and usage.get("subagents", 10**9) <= budget["max_subagents"]
        and usage.get("tokens", 10**9) <= budget["max_tokens"]
    )
    return {
        "key": gold["key"],
        "recall": recall,
        "missing": sorted(gold_facts - found),
        "citation": citation,
        "budget_compliant": compliant,
        "usage": usage,
    }


def main() -> int:
    results = [run_case(gold) for gold in GOLD_CASES]
    all_pass = True
    for result in results:
        passed = (
            result["recall"] >= RECALL_GATE
            and result["citation"] == 1.0
            and result["budget_compliant"]
        )
        all_pass = all_pass and passed
        print(
            f"{result['key']}: recall {result['recall']:.3f} "
            f"(gate {RECALL_GATE}), citation {result['citation']:.3f} "
            f"(gate 1.000), budget {'OK' if result['budget_compliant'] else 'VIOLATED'} "
            f"-> {'PASS' if passed else 'FAIL'}"
        )
        if result["missing"]:
            print(f"  missing gold facts: {result['missing']}")
        print(f"  execution usage: {result['usage']}")
    print("EVALS:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
