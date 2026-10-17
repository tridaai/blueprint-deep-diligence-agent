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
citation checks: named findings must cite named documents. It also
carries poisoned findings — injected into the case record before
execution — which the independent verifier must quarantine with the
expected reasons. Across every pack, the final reports are re-checked
claim by claim: unsupported claims in final reports must be 0.

Every pack also plants contradictions between documents, and the
conflict gate grades them: each planted pair must yield exactly one
conflict record carrying both gold values, and the pack must yield
no other conflicts — facts from different periods (an FY2024 margin
against its FY2025 successor) are different keys and must never
conflict.

The budget case runs one gold brief with an execution step cap
sized to exhaust mid-run: the partial report must itemise coverage
with at least one NOT COVERED question, no cap may be violated, an
amendment below consumption must be refused, and a recorded
amendment plus re-run must reach DRAFT_READY with full recall —
the budget ledger showing grant, exhaustion, amendment, and
completion in order.

Exit code 0 only when every gate passes for every case.

Profiles: `--profile <id>` runs the same packs under a different
harness profile (profiles/<id>.json), with the offline stub model
standing in for the profile's target model — the profile is
configuration, the model is injected. The report header names the
profile and its version, so every result is attributable to the
profile that produced it. The default profile (offline-stub) is the
gate: its run decides the exit code. A run under any other profile
is reported for attribution and exits 0 when the run completes —
whether that profile may be promoted is a separate decision the
gate does not make here (an unloadable profile exits 2).
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_diligence.budgets import budget_view  # noqa: E402
from deep_diligence.config import load_settings  # noqa: E402
from deep_diligence.conflicts import extract_assertions  # noqa: E402
from deep_diligence.corpus import CORPUS, GOLD_CASES, INTEGRITY_CASES  # noqa: E402
from deep_diligence.evidence import evidence_text  # noqa: E402
from deep_diligence.models import BriefIn, BudgetIn  # noqa: E402
from deep_diligence.profiles import ProfileError, load_profile  # noqa: E402
from deep_diligence.service import CaseService  # noqa: E402
from deep_diligence.verifier import check_claim  # noqa: E402

DEFAULT_PROFILE = "offline-stub"

RECALL_GATE = 0.90
FIGURE = re.compile(r"\$[\d,.]+(?:\.\d+)?[MBK]?|\d+(?:\.\d+)?%")
FACT_TEXTS = {f.id: f.text for d in CORPUS for f in d.facts}


def expected_conflict(fact_a: str, fact_b: str) -> dict | None:
    """The gold disagreement between two planted facts: the
    (metric, period) both assert with different values, and the two
    values as stated. Derived from the fact texts with the same
    deterministic extraction the pipeline uses."""
    assertions_a = {
        (a["metric"], a["period"]): a for a in extract_assertions(FACT_TEXTS[fact_a])
    }
    assertions_b = {
        (a["metric"], a["period"]): a for a in extract_assertions(FACT_TEXTS[fact_b])
    }
    for key in assertions_a:
        if key in assertions_b and assertions_a[key]["value"] != assertions_b[key]["value"]:
            return {
                "metric": key[0],
                "period": key[1],
                "values": {
                    assertions_a[key]["value_text"],
                    assertions_b[key]["value_text"],
                },
            }
    return None


def run_case(gold: dict, profile_id: str = DEFAULT_PROFILE) -> dict:
    scratch = Path(tempfile.mkdtemp(prefix="deep-diligence-eval-"))
    settings = load_settings(
        sqlite_path=str(scratch / "diligence.db"),
        object_dir=str(scratch / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
        profile_id=profile_id,
    )
    service = CaseService(settings)
    case = service.create_case(BriefIn(**gold["brief"]))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Eval Harness", approve=True)
    # Poisoned findings (integrity pack): injected straight into the
    # case record, standing in for a sub-agent overstating its
    # evidence. Execution composes them like any other finding; the
    # verifier is what stands between them and the report.
    evidence_by_slug = {
        ev["doc_slug"]: ev["evidence_id"]
        for ev in service.store.list_evidence(case_id)
    }
    for poison in gold.get("poisoned_findings", []):
        service.store.add_finding_auto_id(
            case_id,
            {
                "question_id": poison["question_id"],
                "claim": poison["claim"],
                "evidence_ids": [evidence_by_slug[poison["doc_slug"]]],
                "fact_ref": None,
            },
        )
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
    # Verifier grading: every claim in the final report is re-checked
    # against the evidence it cites (unsupported claims in final
    # reports must be 0 across all packs), and any poisoned finding
    # must sit in quarantine with its expected reason, absent from
    # the report body.
    unsupported: list[dict] = []
    for claim in report["claims"]:
        texts = [
            evidence_text(service.objects, evidence_by_id[eid])
            for eid in claim["evidence_ids"]
            if eid in evidence_by_id
        ]
        failures = check_claim(claim["claim"], texts)
        if failures:
            unsupported.append(
                {"claim_id": claim.get("claim_id"), "failures": failures}
            )
    quarantine = service.store.list_quarantine(case_id)
    poison_results = []
    report_claim_texts = [c["claim"] for c in report["claims"]]
    for poison in gold.get("poisoned_findings", []):
        record = next(
            (q for q in quarantine if q["claim"] == poison["claim"]), None
        )
        poison_results.append(
            {
                **poison,
                "quarantined": record is not None,
                "reason_ok": bool(record)
                and record["reason"] == poison["expect_reason"],
                "in_report": poison["claim"] in report_claim_texts
                or poison["claim"] in report["markdown"],
            }
        )
    # Conflict grading: every planted contradiction must surface as
    # exactly one conflict record carrying both gold values, and the
    # pack must hold no other conflicts — period-distinct facts are
    # different keys and can never create one.
    conflict_records = service.store.list_conflicts(case_id)
    conflict_results = []
    for fact_a, fact_b in gold.get("contradictions", []):
        expected = expected_conflict(fact_a, fact_b)
        matching = [
            r
            for r in conflict_records
            if expected is not None
            and r["key"]["metric"] == expected["metric"]
            and r["key"]["period"] == expected["period"]
            and {s["value_text"] for s in r["sides"]} == expected["values"]
        ]
        conflict_results.append(
            {
                "pair": [fact_a, fact_b],
                "expected": expected,
                "records": len(matching),
                "ok": len(matching) == 1,
            }
        )
    conflicts_ok = all(c["ok"] for c in conflict_results) and len(
        conflict_records
    ) == len(gold.get("contradictions", []))
    verifier_report = service.store.get_verifier_report(case_id) or {}
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
        "profile": service.profile.record(),
        "recall": recall,
        "missing": sorted(gold_facts - found),
        "citation": citation,
        "correctness": correctness,
        "figure_violations": figure_violations,
        "citation_checks": citation_checks,
        "unsupported": unsupported,
        "poison_results": poison_results,
        "conflict_results": conflict_results,
        "conflicts_detected": len(conflict_records),
        "conflicts_ok": conflicts_ok,
        "verification": {
            "checked": verifier_report.get("checked"),
            "supported": verifier_report.get("supported"),
            "quarantined": verifier_report.get("quarantined"),
        },
        "budget_compliant": compliant,
        "usage": usage,
    }


def run_budget_case(profile_id: str = DEFAULT_PROFILE) -> dict:
    """The budget-hardening case: the Helvion gold brief with an
    execution step cap sized to exhaust mid-run (6 of the 28 steps a
    full run needs). Gates: the case lands in BUDGET_EXHAUSTED with
    a partial report whose coverage table names at least one
    NOT COVERED question; no cap is violated (the tripped cap's
    first crossing is exactly one past the limit, every other
    counter stays inside its cap); an amendment below consumption
    is refused; a recorded amendment returns the case to APPROVED
    and the re-run reaches DRAFT_READY with full recall, the ledger
    showing grant -> exhaustion -> amendment -> completion in
    order."""
    gold = GOLD_CASES[0]
    scratch = Path(tempfile.mkdtemp(prefix="deep-diligence-eval-budget-"))
    settings = load_settings(
        sqlite_path=str(scratch / "diligence.db"),
        object_dir=str(scratch / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
        profile_id=profile_id,
    )
    service = CaseService(settings)
    budget = {**gold["brief"]["budget"], "execution": {"max_steps": 6}}
    case = service.create_case(BriefIn(**{**gold["brief"], "budget": budget}))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Eval Harness", approve=True)
    report = service.run_execution(case_id)
    case = service.store.get_case(case_id)
    status_after_stop = case["status"]
    audit = service.store.list_audit(case_id)
    view = budget_view(case, audit)["view"]
    usage = case["usage"]["execution"]
    coverage = report.get("coverage_detail") or []
    not_covered = [r for r in coverage if r["status"] == "NOT COVERED"]
    exhaustions = view["exhaustions"]
    cap_discipline = (
        bool(exhaustions)
        and all(
            e["kind"] == "steps" and e["phase"] == "execution"
            for e in exhaustions
        )
        and min(e["observed"] for e in exhaustions) == 7
        and usage["subagents"] <= budget["max_subagents"]
        and usage["tokens"] <= budget["max_tokens"]
    )
    # An amendment below what execution already consumed is refused.
    consumed_steps = view["phases"]["execution"]["consumed"]["steps"]
    refused = False
    try:
        service.amend_budget(
            case_id,
            analyst="Eval Harness",
            reason="Deliberately below consumption",
            budget=BudgetIn(
                **{
                    **gold["brief"]["budget"],
                    "execution": {"max_steps": consumed_steps - 1},
                }
            ),
        )
    except ValueError:
        refused = True
    amended = service.amend_budget(
        case_id,
        analyst="Eval Harness",
        reason="The step cap under-scoped a three-workstream brief",
        budget=BudgetIn(
            **{**gold["brief"]["budget"], "execution": {"max_steps": 120}}
        ),
    )
    service.run_execution(case_id)
    case = service.store.get_case(case_id)
    # Ledger order, read after the re-execution: the first grant
    # precedes the exhaustion, the amendment follows it, and the
    # completing run's consumption checkpoint closes the sequence.
    events = [a["event"] for a in service.store.list_audit(case_id)]
    ledger_order = False
    if "budget_amended" in events:
        last_consumed = len(events) - 1 - events[::-1].index("budget_consumed")
        ledger_order = (
            events.index("budget_exceeded")
            < events.index("budget_amended")
            < last_consumed
        )
    found = {
        f["fact_ref"]
        for f in service.store.list_findings(case_id)
        if f.get("fact_ref")
    }
    gold_facts = set(gold["gold_facts"])
    final_usage = case["usage"]["execution"]
    within_amended = (
        final_usage["steps"] <= 120
        and final_usage["subagents"] <= budget["max_subagents"]
        and final_usage["tokens"] <= budget["max_tokens"]
    )
    return {
        "key": "budget-tight-helvion",
        "profile": service.profile.record(),
        "exhausted": report["partial"] is True,
        "status_after_stop": status_after_stop,
        "exceeded": (report.get("exceeded_kind"), report.get("exceeded_phase")),
        "coverage_rows": len(coverage),
        "not_covered": len(not_covered),
        "cap_discipline": cap_discipline,
        "amendment_refused_below_consumption": refused,
        "amended_status": amended["status"],
        "ledger_order": ledger_order,
        "final_status": case["status"],
        "recall_after_amendment": (
            len(found & gold_facts) / len(gold_facts) if gold_facts else 1.0
        ),
        "missing_after_amendment": sorted(gold_facts - found),
        "within_amended_caps": within_amended,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gold case-pack gate for the Deep Diligence Agent."
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=(
            "harness profile to run the packs under "
            f"(default {DEFAULT_PROFILE}, the gate). A run under any "
            "other profile is reported for attribution, not gated."
        ),
    )
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
    except ProfileError as exc:
        print(f"cannot run evals: {exc}", file=sys.stderr)
        return 2
    gated = args.profile == DEFAULT_PROFILE
    print(
        f"harness profile: {profile.id} v{profile.version} "
        f"(target {profile.target_model})"
        + ("" if gated else " — profile run, reported not gated")
    )
    results = [
        run_case(gold, profile_id=args.profile)
        for gold in (*GOLD_CASES, *INTEGRITY_CASES)
    ]
    all_pass = True
    total_unsupported = 0
    for result in results:
        checks_ok = all(c["ok"] for c in result["citation_checks"])
        poison_ok = all(
            p["quarantined"] and p["reason_ok"] and not p["in_report"]
            for p in result["poison_results"]
        )
        total_unsupported += len(result["unsupported"])
        passed = (
            result["recall"] >= RECALL_GATE
            and result["citation"] == 1.0
            and result["correctness"] == 1.0
            and checks_ok
            and poison_ok
            and result["conflicts_ok"]
            and not result["unsupported"]
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
        verification = result["verification"]
        print(
            f"  verifier: {verification['checked']} checked, "
            f"{verification['supported']} supported, "
            f"{verification['quarantined']} quarantined"
        )
        for poison in result["poison_results"]:
            print(
                f"  poisoned finding ({poison['expect_reason']}): "
                f"quarantined {'OK' if poison['quarantined'] and poison['reason_ok'] else 'FAILED'}"
                f", in final report {'YES — GATE FAILURE' if poison['in_report'] else 'no'}"
            )
        for conflict in result["conflict_results"]:
            expected = conflict["expected"] or {}
            values = " vs ".join(sorted(expected.get("values", [])))
            print(
                f"  conflict {'/'.join(conflict['pair'])} -> "
                f"{expected.get('metric')} {expected.get('period')}: "
                f"{values} — {conflict['records']} record(s) "
                f"{'OK' if conflict['ok'] else 'FAILED'}"
            )
        print(
            f"  conflicts detected: {result['conflicts_detected']} "
            f"(gate: exactly the planted contradictions, no others)"
        )
        print(
            f"  unsupported claims in final report: "
            f"{len(result['unsupported'])} (gate 0)"
        )
        if result["unsupported"]:
            print(f"  unsupported detail: {result['unsupported']}")
        print(f"  execution usage: {result['usage']}")
    budget_result = run_budget_case(profile_id=args.profile)
    budget_passed = (
        budget_result["status_after_stop"] == "BUDGET_EXHAUSTED"
        and budget_result["exhausted"]
        and budget_result["exceeded"] == ("steps", "execution")
        and budget_result["coverage_rows"] == 3
        and budget_result["not_covered"] >= 1
        and budget_result["cap_discipline"]
        and budget_result["amendment_refused_below_consumption"]
        and budget_result["amended_status"] == "APPROVED"
        and budget_result["ledger_order"]
        and budget_result["final_status"] == "DRAFT_READY"
        and budget_result["recall_after_amendment"] >= RECALL_GATE
        and not budget_result["missing_after_amendment"]
        and budget_result["within_amended_caps"]
    )
    all_pass = all_pass and budget_passed
    print(
        f"{budget_result['key']}: exhausted "
        f"{budget_result['status_after_stop']} on "
        f"{budget_result['exceeded'][0]} during "
        f"{budget_result['exceeded'][1]}, coverage rows "
        f"{budget_result['coverage_rows']} ({budget_result['not_covered']} "
        f"NOT COVERED), cap discipline "
        f"{'OK' if budget_result['cap_discipline'] else 'VIOLATED'}, "
        f"below-consumption amendment "
        f"{'refused' if budget_result['amendment_refused_below_consumption'] else 'NOT REFUSED'}, "
        f"amendment -> {budget_result['amended_status']}, ledger order "
        f"{'OK' if budget_result['ledger_order'] else 'WRONG'}, re-run "
        f"{budget_result['final_status']} with recall "
        f"{budget_result['recall_after_amendment']:.3f} "
        f"-> {'PASS' if budget_passed else 'FAIL'}"
    )
    if budget_result["missing_after_amendment"]:
        print(
            f"  missing gold facts after amendment: "
            f"{budget_result['missing_after_amendment']}"
        )
    print(
        f"unsupported claims in final reports across all packs: "
        f"{total_unsupported} (gate 0)"
    )
    print("EVALS:", "PASS" if all_pass else "FAIL")
    if not gated:
        print(
            f"profile run under {profile.id} v{profile.version}: grades "
            "above are reported for attribution; the gate is the "
            f"{DEFAULT_PROFILE} run."
        )
        return 0
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
