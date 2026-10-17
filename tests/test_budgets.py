import pytest

from deep_diligence.budgets import budget_view, resolve_phase_budget
from deep_diligence.corpus import GOLD_CASES
from deep_diligence.models import BriefIn, BudgetIn
from deep_diligence.service import GateError


def _run_with_budget(service, gold_brief, **budget):
    brief = BriefIn(**{**gold_brief, "budget": {**gold_brief["budget"], **budget}})
    case = service.create_case(brief)
    service.run_planning(case["case_id"])
    service.decide_plan(case["case_id"], analyst="Tester", approve=True)
    report = service.run_execution(case["case_id"])
    return case["case_id"], report


def _run_with_execution_budget(service, gold_brief, **execution):
    brief = BriefIn(
        **{
            **gold_brief,
            "budget": {**gold_brief["budget"], "execution": execution},
        }
    )
    case = service.create_case(brief)
    service.run_planning(case["case_id"])
    service.decide_plan(case["case_id"], analyst="Tester", approve=True)
    report = service.run_execution(case["case_id"])
    return case["case_id"], report


def _exhausted_case(service, gold_brief):
    """A case whose execution stopped on a tight step cap."""
    case_id, report = _run_with_execution_budget(
        service, gold_brief, max_steps=6
    )
    assert service.store.get_case(case_id)["status"] == "BUDGET_EXHAUSTED"
    return case_id, report


def test_step_cap_produces_partial_report(service, gold_brief):
    case_id, report = _run_with_budget(service, gold_brief, max_steps=6)
    case = service.store.get_case(case_id)
    assert case["status"] == "BUDGET_EXHAUSTED"
    assert report["partial"] is True
    assert report["exceeded_kind"] == "steps"
    assert "Partial report" in report["markdown"]
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert "budget_exceeded" in events


def test_subagent_cap_produces_partial_report(service, gold_brief):
    case_id, report = _run_with_budget(service, gold_brief, max_subagents=1)
    assert service.store.get_case(case_id)["status"] == "BUDGET_EXHAUSTED"
    assert report["exceeded_kind"] == "subagents"


def test_token_cap_produces_partial_report(service, gold_brief):
    # Planning costs ~780 stub tokens; 1000 leaves the execution run
    # room to start but not to finish — the abort must land mid-run.
    case_id, report = _run_with_budget(service, gold_brief, max_tokens=1000)
    assert service.store.get_case(case_id)["status"] == "BUDGET_EXHAUSTED"
    assert report["exceeded_kind"] == "tokens"


def test_partial_report_still_needs_the_report_gate(service, gold_brief):
    case_id, _ = _run_with_budget(service, gold_brief, max_steps=6)
    service.decide_report(case_id, analyst="Tester", approve=True)
    assert service.store.get_case(case_id)["status"] == "FINAL"


# -- per-phase budgets --------------------------------------------------------


def test_phase_budgets_default_to_the_flat_values():
    flat = {
        "max_steps": 80,
        "max_subagents": 8,
        "max_tokens": 1000,
        "max_wall_seconds": 900,
    }
    for phase in ("planning", "execution"):
        assert resolve_phase_budget(flat, phase) == {
            "max_steps": 80,
            "max_tokens": 1000,
        }
    overridden = {**flat, "execution": {"max_steps": 6, "max_tokens": None}}
    assert resolve_phase_budget(overridden, "execution") == {
        "max_steps": 6,
        "max_tokens": 1000,
    }
    assert resolve_phase_budget(overridden, "planning") == {
        "max_steps": 80,
        "max_tokens": 1000,
    }


def test_flat_brief_grants_each_phase_the_flat_caps(service, gold_brief):
    case_id, _ = _run_with_budget(service, gold_brief)
    grants = [
        a["payload"]
        for a in service.store.list_audit(case_id)
        if a["event"] == "budget_granted"
    ]
    assert [g["phase"] for g in grants] == ["planning", "execution"]
    for grant in grants:
        assert grant["caps"] == {"max_steps": 120, "max_tokens": 400000}
        assert grant["case_caps"] == {"max_subagents": 6, "max_wall_seconds": 600}


def test_execution_phase_cap_trips_only_execution(service, gold_brief):
    case_id, report = _run_with_execution_budget(
        service, gold_brief, max_steps=6
    )
    assert service.store.get_case(case_id)["status"] == "BUDGET_EXHAUSTED"
    assert report["exceeded_kind"] == "steps"
    assert report["exceeded_phase"] == "execution"
    audit = service.store.list_audit(case_id)
    exceeded = [a for a in audit if a["event"] == "budget_exceeded"]
    assert exceeded and all(
        a["payload"]["phase"] == "execution" for a in exceeded
    )
    # Planning ran under the flat caps and completed: its grant and
    # its consumption checkpoint both precede the exhaustion.
    planning_grant = next(
        a for a in audit if a["event"] == "budget_granted"
    )
    assert planning_grant["payload"]["phase"] == "planning"
    assert planning_grant["payload"]["caps"]["max_steps"] == 120


def test_planning_phase_cap_trips_planning(service, gold_brief):
    brief = BriefIn(
        **{
            **gold_brief,
            "budget": {**gold_brief["budget"], "planning": {"max_steps": 2}},
        }
    )
    case = service.create_case(brief)
    with pytest.raises(GateError):
        service.run_planning(case["case_id"])
    audit = service.store.list_audit(case["case_id"])
    exceeded = [a for a in audit if a["event"] == "budget_exceeded"]
    assert exceeded and all(
        a["payload"]["phase"] == "planning" for a in exceeded
    )
    consumed = [
        a
        for a in audit
        if a["event"] == "budget_consumed"
        and a["payload"]["phase"] == "planning"
    ]
    assert len(consumed) == 1
    assert service.store.get_case(case["case_id"])["status"] == "INTAKE"


# -- the budget ledger ----------------------------------------------------------


def test_budget_view_is_folded_from_the_ledger(service, gold_brief):
    case_id, _ = _run_with_budget(service, gold_brief)
    case = service.store.get_case(case_id)
    view = budget_view(case, service.store.list_audit(case_id))["view"]
    planning = view["phases"]["planning"]
    execution = view["phases"]["execution"]
    assert planning["granted"] == {"max_steps": 120, "max_tokens": 400000}
    assert planning["consumed"]["steps"] == case["usage"]["planning"]["steps"]
    assert planning["consumed"]["tokens"] == case["usage"]["planning"]["tokens"]
    assert execution["consumed"]["steps"] == case["usage"]["execution"]["steps"]
    assert execution["consumed"]["subagents"] == 3
    assert view["case"]["granted"] == {"max_subagents": 6, "max_wall_seconds": 600}
    assert view["case"]["consumed"]["subagents"] == 3
    assert view["exhaustions"] == []
    assert view["amendments"] == []
    # The ledger sequence per phase: grant precedes consumption.
    events = [e["event"] for e in view["ledger"]]
    assert events == [
        "budget_granted",
        "budget_consumed",
        "budget_granted",
        "budget_consumed",
    ]


def test_exhaustion_is_a_ledger_entry_with_phase_and_usage(
    service, gold_brief
):
    case_id, _ = _exhausted_case(service, gold_brief)
    view = budget_view(
        service.store.get_case(case_id), service.store.list_audit(case_id)
    )["view"]
    assert view["exhaustions"], "the exhaustion must be in the ledger view"
    first = view["exhaustions"][0]
    assert first["phase"] == "execution"
    assert first["kind"] == "steps"
    assert first["limit"] == 6
    # The first crossing is exactly one past the cap; sibling
    # sub-agents already in flight may register further crossings,
    # never fewer.
    assert min(e["observed"] for e in view["exhaustions"]) == 7
    consumed = view["phases"]["execution"]["consumed"]
    assert consumed["steps"] >= 7
    assert consumed["subagents"] == 3


# -- the partial-report protocol -----------------------------------------------


def test_partial_report_itemises_coverage_per_question(service, gold_brief):
    # Cap 22 of the 28 steps a full run needs: some workstreams land
    # findings, at least one never starts recording.
    case_id, report = _run_with_execution_budget(
        service, gold_brief, max_steps=22
    )
    assert report["partial"] is True
    detail = report["coverage_detail"]
    assert [r["question_id"] for r in detail] == ["q1", "q2", "q3"]
    statuses = {r["question_id"]: r["status"] for r in detail}
    assert "PARTIAL" in statuses.values()
    assert "NOT COVERED" in statuses.values()
    for row in detail:
        if row["status"] == "PARTIAL":
            assert row["findings"] > 0
            assert row["question_id"] in report["interrupted_questions"]
        if row["status"] == "NOT COVERED":
            assert row["findings"] == 0
    markdown = report["markdown"]
    assert "Coverage by question" in markdown
    assert "NOT COVERED" in markdown
    assert "stopped on its steps budget during the execution phase" in markdown
    assert service.store.get_case(case_id)["status"] == "BUDGET_EXHAUSTED"


def test_partial_approval_records_the_coverage_it_approved(
    service, gold_brief
):
    case_id, report = _run_with_budget(service, gold_brief, max_steps=6)
    service.decide_report(case_id, analyst="Tester", approve=True)
    assert service.store.get_case(case_id)["status"] == "FINAL"
    stored = service.store.get_report(case_id)
    assert stored["approved_coverage"] == report["coverage_detail"]
    approvals = [
        a
        for a in service.store.list_audit(case_id)
        if a["event"] == "report_approved"
    ]
    assert len(approvals) == 1
    assert approvals[0]["payload"]["partial"] is True
    assert approvals[0]["payload"]["coverage"] == report["coverage_detail"]


# -- budget amendment ------------------------------------------------------------


def test_amendment_below_consumption_is_refused(service, gold_brief):
    case_id, _ = _exhausted_case(service, gold_brief)
    view = budget_view(
        service.store.get_case(case_id), service.store.list_audit(case_id)
    )["view"]
    consumed_steps = view["phases"]["execution"]["consumed"]["steps"]
    consumed_spawns = view["case"]["consumed"]["subagents"]
    assert consumed_steps > 6 and consumed_spawns == 3
    base = dict(gold_brief["budget"])
    with pytest.raises(ValueError, match="below"):
        service.amend_budget(
            case_id,
            analyst="Tester",
            reason="Trying to shave the cap back",
            budget=BudgetIn(
                **{**base, "execution": {"max_steps": consumed_steps - 1}}
            ),
        )
    with pytest.raises(ValueError, match="below"):
        service.amend_budget(
            case_id,
            analyst="Tester",
            reason="Trying to shave the spawn cap back",
            budget=BudgetIn(**{**base, "max_subagents": consumed_spawns - 1}),
        )
    # A refused amendment changes nothing.
    case = service.store.get_case(case_id)
    assert case["status"] == "BUDGET_EXHAUSTED"
    assert case["brief"]["budget"]["execution"]["max_steps"] == 6
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert "budget_amended" not in events


def test_amendment_needs_an_exhausted_case_and_a_reason(
    service, approved_case, gold_brief
):
    with pytest.raises(GateError):
        service.amend_budget(
            approved_case["case_id"],
            analyst="Tester",
            reason="More room",
            budget=BudgetIn(**gold_brief["budget"]),
        )


def test_amendment_reason_may_not_be_blank(service, gold_brief):
    case_id, _ = _exhausted_case(service, gold_brief)
    with pytest.raises(ValueError, match="reason"):
        service.amend_budget(
            case_id,
            analyst="Tester",
            reason="   ",
            budget=BudgetIn(**gold_brief["budget"]),
        )


def test_amendment_returns_case_to_approved_and_rerun_completes(
    service, gold_brief
):
    case_id, _ = _exhausted_case(service, gold_brief)
    amended = service.amend_budget(
        case_id,
        analyst="Tester",
        reason="The step cap under-scoped a three-workstream brief",
        budget=BudgetIn(
            **{**gold_brief["budget"], "execution": {"max_steps": 120}}
        ),
    )
    assert amended["status"] == "APPROVED"
    assert amended["brief"]["budget"]["execution"]["max_steps"] == 120
    audit = service.store.list_audit(case_id)
    amendment = next(a for a in audit if a["event"] == "budget_amended")
    assert amendment["payload"]["reason"].startswith("The step cap")
    assert amendment["payload"]["previous"]["execution"]["max_steps"] == 6
    assert amendment["payload"]["amended"]["execution"]["max_steps"] == 120
    # Ledger order: grant -> exhaustion -> amendment -> the
    # re-execution's consumption checkpoint.
    events = [a["event"] for a in audit]
    assert events.index("budget_exceeded") < events.index("budget_amended")

    report = service.run_execution(case_id)
    case = service.store.get_case(case_id)
    assert case["status"] == "DRAFT_READY"
    assert report["partial"] is False
    found = {
        f["fact_ref"]
        for f in service.store.list_findings(case_id)
        if f.get("fact_ref")
    }
    assert set(GOLD_CASES[0]["gold_facts"]) <= found
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert events.index("budget_amended") < len(events) - 1 - events[::-1].index(
        "budget_consumed"
    )
