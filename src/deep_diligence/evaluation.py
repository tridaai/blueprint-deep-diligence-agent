"""Case-pack evaluation: the ruler every profile is graded with.

`evaluate_case` runs one gold case end to end on the deterministic
stub model over the bundled corpus and grades it the way the eval
gate (evals/run_evals.py) always has: finding recall against the
planted gold facts, citation completeness and correctness on the
composed claims, the verifier's quarantine of any poisoned findings,
the conflict gate on planted contradictions, and zero unsupported
claims in the final report. `evaluate_suite` runs every pack once
and returns per-case results plus the slim per-case metrics the
profile regression gate (promotion.py) consumes.

The suite lives here, inside the package, so the eval runner and
the promotion gate grade with exactly the same ruler — there is no
second, softer grading path for promotion decisions.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from .config import load_settings
from .conflicts import extract_assertions
from .corpus import ALL_CASES, CORPUS
from .evidence import evidence_text
from .models import BriefIn
from .profiles import load_profile
from .service import CaseService
from .verifier import check_claim

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def evaluate_case(gold: dict, profile_id: str = "offline-stub") -> dict:
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
        "split": gold["split"],
        "profile": service.profile.record(),
        "recall": recall,
        "found": len(found & gold_facts),
        "gold": len(gold_facts),
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


def evaluate_suite(profile_id: str) -> dict:
    """One full pass over every pack under one profile.

    Returns the profile record, the full per-case results, and the
    slim per-case metrics the promotion gate aggregates by subset.
    """
    profile = load_profile(profile_id)
    results = [evaluate_case(gold, profile_id=profile_id) for gold in ALL_CASES]
    return {
        "profile": profile.record(),
        "results": results,
        "cases": [
            {
                "key": r["key"],
                "split": r["split"],
                "recall": r["recall"],
                "found": r["found"],
                "gold": r["gold"],
                "citation": r["citation"],
                "correctness": r["correctness"],
                "unsupported": len(r["unsupported"]),
            }
            for r in results
        ],
    }
