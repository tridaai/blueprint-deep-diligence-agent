from deep_diligence.models import BriefIn


def _run_with_budget(service, gold_brief, **budget):
    brief = BriefIn(**{**gold_brief, "budget": {**gold_brief["budget"], **budget}})
    case = service.create_case(brief)
    service.run_planning(case["case_id"])
    service.decide_plan(case["case_id"], analyst="Tester", approve=True)
    report = service.run_execution(case["case_id"])
    return case["case_id"], report


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
