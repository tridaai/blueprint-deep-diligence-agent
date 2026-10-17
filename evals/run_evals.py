#!/usr/bin/env python3
"""Gold case-pack gate for the Deep Diligence Agent (offline).

Runs each gold case end to end on the deterministic stub model over
the bundled corpus and grades:
  * finding recall against the planted gold facts (gate >= 0.90),
  * citation completeness on composed claims (gate == 100%),
  * citation correctness: every figure a claim states ($ amounts and
    percentages) must appear in the evidence it cites — a citation
    that points at a document saying something else is a failure,
    not a pass (gate == 100%),
  * budget compliance: the run completed inside its caps (gate 100%).

The citation-integrity pack (INTEGRITY_CASES) adds a corpus with a
near-duplicate document carrying a stale figure, and per-case
citation checks: named findings must cite named documents.

Exit code 0 only when every gate passes for every case.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_diligence.config import load_settings  # noqa: E402
from deep_diligence.corpus import GOLD_CASES, INTEGRITY_CASES  # noqa: E402
from deep_diligence.evidence import evidence_text  # noqa: E402
from deep_diligence.models import BriefIn  # noqa: E402
from deep_diligence.service import CaseService  # noqa: E402

RECALL_GATE = 0.90
FIGURE = re.compile(r"\$[\d,.]+(?:\.\d+)?[MBK]?|\d+(?:\.\d+)?%")


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
    # Citation correctness: a figure stated in a claim must be
    # supported by the evidence the claim actually cites.
    evidence_by_id = {
        ev["evidence_id"]: ev for ev in service.store.list_evidence(case_id)
    }
    figure_total = 0
    figure_violations: list[dict] = []
    for claim in claims:
        figures = FIGURE.findall(claim["claim"])
        if not figures:
            continue
        cited_text = "\n".join(
            evidence_text(service.objects, evidence_by_id[eid])
            for eid in claim["evidence_ids"]
            if eid in evidence_by_id
        )
        for figure in figures:
            figure_total += 1
            if figure not in cited_text:
                figure_violations.append(
                    {"claim_id": claim.get("claim_id"), "figure": figure}
                )
    correctness = (
        1.0 - len(figure_violations) / figure_total if figure_total else 1.0
    )
    # Per-case citation checks (integrity pack): a named finding must
    # cite a named document — e.g. the current figure must rest on
    # the current filing, not the stale near-duplicate.
    findings = service.store.list_findings(case_id)
    citation_checks = []
    for check in gold.get("citation_checks", []):
        matching = [f for f in findings if f.get("fact_ref") == check["fact_ref"]]
        ok = any(
            evidence_by_id[eid]["doc_slug"] == check["doc_slug"]
            for f in matching
            for eid in f["evidence_ids"]
            if eid in evidence_by_id
        )
        citation_checks.append({**check, "ok": ok})
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
        "correctness": correctness,
        "figure_violations": figure_violations,
        "citation_checks": citation_checks,
        "budget_compliant": compliant,
        "usage": usage,
    }


def main() -> int:
    results = [run_case(gold) for gold in (*GOLD_CASES, *INTEGRITY_CASES)]
    all_pass = True
    for result in results:
        checks_ok = all(c["ok"] for c in result["citation_checks"])
        passed = (
            result["recall"] >= RECALL_GATE
            and result["citation"] == 1.0
            and result["correctness"] == 1.0
            and checks_ok
            and result["budget_compliant"]
        )
        all_pass = all_pass and passed
        print(
            f"{result['key']}: recall {result['recall']:.3f} "
            f"(gate {RECALL_GATE}), citation {result['citation']:.3f} "
            f"(gate 1.000), correctness {result['correctness']:.3f} "
            f"(gate 1.000), budget {'OK' if result['budget_compliant'] else 'VIOLATED'} "
            f"-> {'PASS' if passed else 'FAIL'}"
        )
        if result["missing"]:
            print(f"  missing gold facts: {result['missing']}")
        if result["figure_violations"]:
            print(f"  figure violations: {result['figure_violations']}")
        for check in result["citation_checks"]:
            print(
                f"  citation check {check['fact_ref']} -> {check['doc_slug']}: "
                f"{'OK' if check['ok'] else 'FAILED'}"
            )
        print(f"  execution usage: {result['usage']}")
    print("EVALS:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
