"""Round 18: observability — the run trace, the failure
taxonomy, and the operations digest.

All three are reads over the records the shell already keeps:
these tests grade them off the same records the service wrote,
the way the eval packs grade. The trace must reconstruct a run
in causal order and store nothing; the taxonomy must give every
case exactly one label from its documented set, with the
deciding records named; the digest must fold a day to totals
that recompute by hand from the ledgers, and its stored
snapshots must not move when the day is regenerated.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.corpus import (
    CHALLENGE_CASES,
    DEPENDENCY_CASES,
    GOLD_CASES,
    INTEGRITY_CASES,
)
from deep_diligence.costs import fold_cost_ledger
from deep_diligence.models import BriefIn
from deep_diligence.observability import OUTCOME_LABELS
from deep_diligence.service import CaseService, GateError
from deep_diligence.store import utcnow

VERDANT = DEPENDENCY_CASES[0]["brief"]
KESTREL = CHALLENGE_CASES[0]
TODAY = utcnow()[:10]


def _full_run(service: CaseService, gold: dict) -> str:
    case = service.create_case(BriefIn(**gold["brief"]))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    return case_id


def _budget_case(service: CaseService) -> str:
    """The Helvion brief with an execution step cap sized to
    exhaust mid-run (the budget pack's own construction)."""
    gold = GOLD_CASES[0]
    budget = {**gold["brief"]["budget"], "execution": {"max_steps": 6}}
    case = service.create_case(
        BriefIn(**{**gold["brief"], "budget": budget})
    )
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    assert service.store.get_case(case_id)["status"] == "BUDGET_EXHAUSTED"
    return case_id


# -- the trace ------------------------------------------------------------------


def test_trace_reconstructs_a_full_run_in_causal_order(service):
    case_id = _full_run(service, GOLD_CASES[0])
    service.decide_report(case_id, analyst="Tester", approve=True)
    trace = service.case_trace(case_id)
    audit = service.store.list_audit(case_id)
    # Every audit entry becomes exactly one act, in order.
    assert [a["event"] for a in trace["acts"]] == [e["event"] for e in audit]
    assert [a["seq"] for a in trace["acts"]] == list(range(len(audit)))
    events = [a["event"] for a in trace["acts"]]
    # The full act sequence, in the order the run happened.
    key = [
        "case_created",
        "plan_generated",
        "plan_approved",
        "execution_started",
        "workstream_finished",
        "report_composed",
        "verification_completed",
        "report_approved",
    ]
    positions = [events.index(name) for name in key]
    assert positions == sorted(positions)
    kinds = {a["event"]: a["kind"] for a in trace["acts"]}
    assert kinds["plan_approved"] == "plan_gate"
    assert kinds["report_composed"] == "composition"
    assert kinds["verification_completed"] == "verification"
    assert kinds["report_approved"] == "report_gate"
    assert trace["outcome"]["label"] == "completed"
    assert trace["counts"]["findings"] == len(
        service.store.list_findings(case_id)
    )


def test_trace_runs_carry_grants_consumption_and_cost(service):
    case_id = _full_run(service, GOLD_CASES[0])
    trace = service.case_trace(case_id)
    audit = service.store.list_audit(case_id)
    grants = [e for e in audit if e["event"] == "budget_granted"]
    assert [r["phase"] for r in trace["runs"]] == [
        g["payload"]["phase"] for g in grants
    ]
    for run, grant in zip(trace["runs"], grants):
        assert run["caps"] == grant["payload"]["caps"]
        assert run["profile"] == grant["payload"]["profile"]
    consumed = {
        e["payload"]["phase"]: e["payload"]["usage"]
        for e in audit
        if e["event"] == "budget_consumed"
    }
    for run in trace["runs"]:
        assert run["consumed"] == consumed[run["phase"]]
    # The trace's cost is the cost ledger's own total, to the cent.
    ledger_total = fold_cost_ledger(audit)["totals"]["cost_usd"]
    assert Decimal(trace["cost"]["total_usd"]) == ledger_total
    assert Decimal(
        sum(Decimal(r["cost"]["total_usd"]) for r in trace["runs"])
    ) == ledger_total
    execution = next(r for r in trace["runs"] if r["phase"] == "execution")
    assert execution["attempt"] == 1
    assert execution["exceeded"] is None


def test_trace_and_outcome_are_pure_reads(service):
    case_id = _full_run(service, GOLD_CASES[0])
    before_audit = service.store.list_audit(case_id)
    before_case = service.store.get_case(case_id)
    first = service.case_trace(case_id)
    second = service.case_trace(case_id)
    service.case_outcome(case_id)
    assert first == second
    assert service.store.list_audit(case_id) == before_audit
    assert service.store.get_case(case_id) == before_case


def test_trace_timing_is_honest_about_what_was_never_recorded(service):
    case_id = _full_run(service, GOLD_CASES[0])
    trace = service.case_trace(case_id)
    assert trace["timing"]["per_act_durations_recorded"] is False
    assert "never recorded" in trace["timing"]["note"]
    assert all("duration" not in act for act in trace["acts"])
    # What WAS recorded — the runs' usage, wall seconds included —
    # rides the runs.
    assert all(
        r["consumed"] and "wall_seconds" in r["consumed"]
        for r in trace["runs"]
    )


# -- the taxonomy -----------------------------------------------------------------


def test_outcome_in_progress_then_completed(service, case):
    case_id = case["case_id"]
    assert service.case_outcome(case_id)["label"] == "in_progress"
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "in_progress"
    assert outcome["detail"]["status"] == "DRAFT_READY"
    service.decide_report(case_id, analyst="Tester", approve=True)
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "completed"
    assert outcome["detail"]["decided_by"] == "Tester"


def test_outcome_budget_exhausted_names_the_uncovered_questions(service):
    case_id = _budget_case(service)
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "budget_exhausted"
    report = service.store.get_report(case_id)
    uncovered = [
        r["question_id"]
        for r in report["coverage_detail"]
        if r["status"] == "NOT COVERED"
    ]
    assert uncovered  # the tight cap leaves at least one question bare
    assert outcome["detail"]["uncovered"] == uncovered
    assert outcome["detail"]["exceeded_kind"] == "steps"
    assert outcome["detail"]["exceeded_phase"] == "execution"
    # The trace tells the same story: the execution run tripped.
    trace = service.case_trace(case_id)
    execution = next(r for r in trace["runs"] if r["phase"] == "execution")
    assert execution["exceeded"]["kind"] == "steps"
    composed = [
        a for a in trace["acts"] if a["event"] == "report_composed"
    ]
    assert composed[-1]["detail"]["partial"] is True


def test_outcome_completed_partial_after_the_analyst_approves_it(service):
    case_id = _budget_case(service)
    # Answer anything the challenge pass holds open on the
    # partial draft, then approve the partial as it stands.
    evidence_id = service.store.list_evidence(case_id)[0]["evidence_id"]
    for challenge in service.store.list_challenges(case_id):
        if challenge["status"] == "OPEN" and (
            challenge["severity"] == "must_answer"
        ):
            service.decide_challenge(
                case_id,
                challenge["challenge_id"],
                analyst="Tester",
                action="rebut",
                note="The partial stands as composed; the figure "
                "is attributed to its source.",
                counter_evidence_ids=[evidence_id],
            )
    decided = service.decide_report(case_id, analyst="Tester", approve=True)
    assert decided["status"] == "FINAL"
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "completed_partial"
    report = service.store.get_report(case_id)
    assert outcome["detail"]["uncovered"] == [
        r["question_id"]
        for r in report["approved_coverage"]
        if r["status"] == "NOT COVERED"
    ]


def test_outcome_workstream_failed_names_the_branch(service, monkeypatch):
    case = service.create_case(BriefIn(**VERDANT))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    original = service._run_agent
    runs = {"count": 0}

    def flaky(case_, message, phase):
        if phase == "execution":
            runs["count"] += 1
            if runs["count"] == 2:
                raise RuntimeError("the market sub-agent died mid-run")
        return original(case_, message, phase)

    monkeypatch.setattr(service, "_run_agent", flaky)
    service.run_execution(case_id)
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "workstream_failed"
    failed = outcome["detail"]["failed"]
    assert [f["question_id"] for f in failed] == ["q2"]
    assert "RuntimeError" in failed[0]["reason"]


def test_outcome_challenge_blocked_then_clears(service):
    case_id = _full_run(service, KESTREL)
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "challenge_blocked"
    must = [
        c
        for c in service.store.list_challenges(case_id)
        if c["status"] == "OPEN" and c["severity"] == "must_answer"
    ]
    assert len(must) == 2
    assert outcome["detail"]["challenge_ids"] == [
        c["challenge_id"] for c in must
    ]
    filing_id = next(
        ev["evidence_id"]
        for ev in service.store.list_evidence(case_id)
        if ev["kind"] == "filing"
    )
    for challenge in must:
        service.decide_challenge(
            case_id,
            challenge["challenge_id"],
            analyst="Tester",
            action="rebut",
            note="Stands as reported, attributed to its source.",
            counter_evidence_ids=[filing_id],
        )
    # The advisory challenge still stands open and blocks nothing.
    assert service.case_outcome(case_id)["label"] == "in_progress"
    service.decide_report(case_id, analyst="Tester", approve=True)
    assert service.case_outcome(case_id)["label"] == "completed"


def test_outcome_claims_quarantined_names_the_reasons(service):
    gold = INTEGRITY_CASES[0]
    case = service.create_case(BriefIn(**gold["brief"]))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    evidence_by_slug = {
        ev["doc_slug"]: ev["evidence_id"]
        for ev in service.store.list_evidence(case_id)
    }
    for poison in gold["poisoned_findings"]:
        service.store.add_finding_auto_id(
            case_id,
            {
                "question_id": poison["question_id"],
                "claim": poison["claim"],
                "evidence_ids": [evidence_by_slug[poison["doc_slug"]]],
                "fact_ref": None,
            },
        )
    service.run_execution(case_id)
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "claims_quarantined"
    assert sorted(c["reason"] for c in outcome["detail"]["claims"]) == [
        "entity_mismatch",
        "figure_not_in_evidence",
    ]


def test_outcome_plan_rejected_and_report_rejected(service, case):
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(
        case_id, analyst="Tester", approve=False, note="Scope it tighter"
    )
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "plan_rejected"
    assert outcome["detail"]["decided_by"] == "Tester"
    assert outcome["detail"]["note"] == "Scope it tighter"

    other = _full_run(service, GOLD_CASES[1])
    service.decide_report(
        other, analyst="Tester", approve=False, note="Not yet"
    )
    outcome = service.case_outcome(other)
    assert outcome["label"] == "report_rejected"
    assert outcome["detail"]["note"] == "Not yet"


def test_outcome_stale_evidence_names_questions_and_documents(service):
    case_id = _full_run(service, DEPENDENCY_CASES[0])
    if service.store.get_source("Trade coverage") is None:
        service.register_source(
            "Research Desk",
            name="Trade coverage",
            kind="news",
            owner="Trade press publisher",
            usage_basis="public",
            notes="Trade-press coverage gathered by the research desk.",
        )
    result = service.add_case_evidence(
        case_id,
        "Research Desk",
        title="Verdant Robotics — update: Mule-6 platform deployments",
        kind="product",
        source="Trade coverage",
        doc_date="2026-03-15",
        text=(
            "# Verdant Robotics — update: Mule-6 platform deployments\n\n"
            "Verdant Robotics reports that platform deployment is "
            "accelerating: the Mule-6 platform now anchors the "
            "product roadmap, and platform customers are expanding "
            "their deployments across retail warehouses.\n\n"
            "## Key facts\n"
            "- [F-UPD-201] Verdant Robotics reports the Mule-6 "
            "platform now runs in 300 customer deployments.\n"
        ),
    )
    assert result["stale_questions"] == ["q1"]
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "stale_evidence"
    assert outcome["detail"]["question_ids"] == ["q1"]
    assert outcome["detail"]["evidence_ids"] == [
        result["evidence"]["evidence_id"]
    ]


def test_outcome_interrupted_names_the_attempt_and_the_split(service, case):
    case_id = case["case_id"]
    service.store.update_case(case_id, status="EXECUTING")
    service.store.save_checkpoint(
        case_id,
        {
            "case_id": case_id,
            "plan_version": 1,
            "attempt": 2,
            "waves": [["q-financials"], ["q-products"]],
            "workstreams": {
                "q-financials": {
                    "status": "completed",
                    "findings": 3,
                    "finding_ids": [],
                    "wave": 1,
                    "reason": None,
                },
                "q-products": {
                    "status": "pending",
                    "findings": 0,
                    "finding_ids": [],
                    "wave": 2,
                    "reason": None,
                },
            },
            "budget_snapshot": {},
            "complete": False,
            "outcome": None,
            "started_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "interrupted"
    assert outcome["detail"]["attempt"] == 2
    assert outcome["detail"]["completed_questions"] == ["q-financials"]
    assert outcome["detail"]["pending_questions"] == ["q-products"]


def test_outcome_evidence_drift(service, approved_case):
    case_id = approved_case["case_id"]
    service.run_execution(case_id)
    report = service.store.get_report(case_id)
    evidence_id = report["claims"][0]["evidence_ids"][0]
    ev = service.store.get_evidence(case_id, evidence_id)
    service.objects.put(ev["object_key"], b"# edited after the fact\n")
    with pytest.raises(GateError, match="EVIDENCE_DRIFT"):
        service.decide_report(case_id, analyst="Tester", approve=True)
    outcome = service.case_outcome(case_id)
    assert outcome["label"] == "evidence_drift"
    assert outcome["detail"]["evidence_ids"] == [evidence_id]


def test_taxonomy_is_total_over_the_states_a_desk_produces(service):
    """Every case the flows below produce classifies into the
    documented label set, and no case at a terminal state hides
    in the in-progress fallback."""
    seen: dict[str, str] = {}
    fresh = service.create_case(BriefIn(**GOLD_CASES[2]["brief"]))
    seen[fresh["case_id"]] = service.case_outcome(fresh["case_id"])["label"]
    rejected = service.create_case(BriefIn(**GOLD_CASES[1]["brief"]))
    service.run_planning(rejected["case_id"])
    service.decide_plan(rejected["case_id"], analyst="Tester", approve=False)
    seen[rejected["case_id"]] = service.case_outcome(
        rejected["case_id"]
    )["label"]
    final = _full_run(service, GOLD_CASES[0])
    service.decide_report(final, analyst="Tester", approve=True)
    seen[final] = service.case_outcome(final)["label"]
    exhausted = _budget_case(service)
    seen[exhausted] = service.case_outcome(exhausted)["label"]
    blocked = _full_run(service, KESTREL)
    seen[blocked] = service.case_outcome(blocked)["label"]
    for label in seen.values():
        assert label in OUTCOME_LABELS
    terminal = {
        service.store.get_case(cid)["status"]
        for cid in (final, exhausted, rejected["case_id"])
    }
    assert terminal == {"FINAL", "BUDGET_EXHAUSTED", "PLAN_REJECTED"}
    assert seen[final] == "completed"
    assert seen[exhausted] == "budget_exhausted"
    assert seen[rejected["case_id"]] == "plan_rejected"
    assert seen[blocked] == "challenge_blocked"
    assert seen[fresh["case_id"]] == "in_progress"


# -- the digest -------------------------------------------------------------------


def test_digest_absent_until_generated(service, settings):
    assert service.get_digest(TODAY) is None
    assert service.list_digests() == []
    client = TestClient(create_app(settings))
    assert client.get(f"/api/digests/{TODAY}").status_code == 404


def _digest_scenario(service: CaseService) -> dict[str, str]:
    final = _full_run(service, GOLD_CASES[0])
    service.decide_report(final, analyst="Tester", approve=True)
    exhausted = _budget_case(service)
    blocked = _full_run(service, KESTREL)
    return {"final": final, "exhausted": exhausted, "blocked": blocked}


def test_digest_folds_the_day_to_hand_computed_totals(service):
    ids = _digest_scenario(service)
    record = service.generate_digest(TODAY, "Tester")
    assert record["version"] == 1
    assert record["generated_by"] == "Tester"
    digest = record["digest"]
    assert digest["day"] == TODAY
    assert sorted(digest["cases"]["opened"]) == sorted(ids.values())
    assert digest["cases"]["opened_count"] == 3
    assert digest["cases"]["closed"] == [ids["final"]]
    assert digest["cases"]["by_status"] == {
        "BUDGET_EXHAUSTED": 1,
        "DRAFT_READY": 1,
        "FINAL": 1,
    }
    assert digest["gate_decisions"] == {
        "plan_approved": 3,
        "plan_rejected": 0,
        "report_approved": 1,
        "report_rejected": 0,
    }
    # The cost is the ledgers' own sum, recomputed case by case.
    expected_cost = sum(
        fold_cost_ledger(service.store.list_audit(cid))["totals"]["cost_usd"]
        for cid in ids.values()
    )
    assert Decimal(digest["cost"]["total_usd"]) == expected_cost
    assert digest["cost"]["runs_recorded"] == 6  # two runs per case
    assert sorted(
        row["case_id"] for row in digest["cost"]["by_case"]
    ) == sorted(ids.values())
    # Taxonomy over the day's cases, one label each.
    assert digest["taxonomy"] == {
        "budget_exhausted": 1,
        "challenge_blocked": 1,
        "completed": 1,
    }
    assert sum(digest["taxonomy"].values()) == digest["cases"]["opened_count"]
    # The partial still standing, with its uncovered questions.
    assert [p["case_id"] for p in digest["partial_reports"]] == [
        ids["exhausted"]
    ]
    report = service.store.get_report(ids["exhausted"])
    assert digest["partial_reports"][0]["uncovered"] == [
        r["question_id"]
        for r in report["coverage_detail"]
        if r["status"] == "NOT COVERED"
    ]
    assert digest["partial_reports"][0]["exceeded_kind"] == "steps"
    # Challenges: raised counts the records minted today across
    # the desk; the open must-answer list is Kestrel's two, aged.
    raised = sum(
        1
        for cid in ids.values()
        for c in service.store.list_challenges(cid)
        if c["created_at"][:10] == TODAY
    )
    assert digest["challenges"]["raised"] == raised
    assert digest["challenges"]["answered"] == 0
    open_ids = {
        c["challenge_id"] for c in digest["challenges"]["open_must_answer"]
    }
    expected_open = {
        c["challenge_id"]
        for c in service.store.list_challenges(ids["blocked"])
        if c["status"] == "OPEN" and c["severity"] == "must_answer"
    }
    assert open_ids == expected_open
    assert all(
        c["age_seconds"] >= 0 for c in digest["challenges"]["open_must_answer"]
    )


def test_digest_versions_stack_and_old_snapshots_do_not_move(service):
    first = service.create_case(BriefIn(**GOLD_CASES[0]["brief"]))
    v1 = service.generate_digest(TODAY, "Tester")
    assert v1["version"] == 1
    assert v1["digest"]["cases"]["opened_count"] == 1
    service.create_case(BriefIn(**GOLD_CASES[1]["brief"]))
    v2 = service.generate_digest(TODAY, "Tester")
    assert v2["version"] == 2
    assert v2["digest"]["cases"]["opened_count"] == 2
    stored = service.list_digests()
    assert [r["version"] for r in stored] == [1, 2]
    # The first snapshot still says what it said.
    assert stored[0]["digest"]["cases"]["opened_count"] == 1
    assert stored[0]["digest"]["cases"]["opened"] == [first["case_id"]]
    assert service.get_digest(TODAY)["version"] == 2


def test_digest_generation_writes_nothing_to_any_case(service):
    case_id = _full_run(service, GOLD_CASES[0])
    before = service.store.list_audit(case_id)
    before_case = service.store.get_case(case_id)
    service.generate_digest(TODAY, "Tester")
    assert service.store.list_audit(case_id) == before
    assert service.store.get_case(case_id) == before_case


def test_digest_day_must_be_a_real_iso_date(service):
    for bad in ("2026-13-40", "18-10-2026", "2026-1-1", "yesterday"):
        with pytest.raises(ValueError, match="ISO calendar date"):
            service.generate_digest(bad, "Tester")
        with pytest.raises(ValueError, match="ISO calendar date"):
            service.get_digest(bad)


# -- the API ----------------------------------------------------------------------


def test_observability_over_http(settings):
    client = TestClient(create_app(settings))
    created = client.post("/api/cases", json=GOLD_CASES[0]["brief"]).json()
    case_id = created["case_id"]
    client.post(f"/api/cases/{case_id}/plan")
    client.post(
        f"/api/cases/{case_id}/plan/decision",
        json={"analyst": "Tester", "approve": True, "note": ""},
    )
    client.post(f"/api/cases/{case_id}/execute")
    trace = client.get(f"/api/cases/{case_id}/trace")
    assert trace.status_code == 200
    body = trace.json()
    assert body["case_id"] == case_id
    assert body["outcome"]["label"] == "in_progress"
    assert [r["phase"] for r in body["runs"]] == ["planning", "execution"]
    assert body["acts"][0]["event"] == "case_created"
    outcome = client.get(f"/api/cases/{case_id}/outcome")
    assert outcome.status_code == 200
    assert outcome.json() == body["outcome"]
    assert client.get("/api/cases/case-nope/trace").status_code == 404
    assert client.get("/api/cases/case-nope/outcome").status_code == 404

    made = client.post(
        f"/api/digests/{TODAY}", json={"analyst": "Tester"}
    )
    assert made.status_code == 201
    assert made.json()["version"] == 1
    got = client.get(f"/api/digests/{TODAY}")
    assert got.status_code == 200
    assert got.json()["digest"]["cases"]["opened_count"] == 1
    listed = client.get("/api/digests")
    assert [r["day"] for r in listed.json()] == [TODAY]
    assert client.post("/api/digests/not-a-date", json={"analyst": "T"}).status_code == 422
    assert client.get("/api/digests/1999-01-01").status_code == 404
