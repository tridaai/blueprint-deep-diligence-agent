import pytest

from deep_diligence.service import GateError


def test_execution_is_impossible_before_plan_approval(service, case):
    with pytest.raises(GateError):
        service.run_execution(case["case_id"])


def test_plan_gate_full_lifecycle(service, case):
    case_id = case["case_id"]
    plan = service.run_planning(case_id)
    assert plan["status"] == "PENDING"
    assert [ws["question_id"] for ws in plan["plan"]["workstreams"]] == ["q1", "q2", "q3"]
    assert service.store.get_case(case_id)["status"] == "PLAN_READY"
    # Approved plan required: execution still blocked while merely ready.
    with pytest.raises(GateError):
        service.run_execution(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    assert service.store.get_case(case_id)["status"] == "APPROVED"
    report = service.run_execution(case_id)
    assert report["status"] == "DRAFT"
    assert report["partial"] is False
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    service.decide_report(case_id, analyst="Tester", approve=True)
    assert service.store.get_case(case_id)["status"] == "FINAL"
    events = [a["event"] for a in service.store.list_audit(case_id)]
    for expected in ("plan_generated", "plan_approved", "execution_started",
                     "report_composed", "report_approved", "subagent_spawned"):
        assert expected in events


def test_plan_rejection_and_replan(service, case):
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=False, note="Scope q3 out")
    assert service.store.get_case(case_id)["status"] == "PLAN_REJECTED"
    plan2 = service.run_planning(case_id)
    assert plan2["version"] == 2
    assert service.store.get_case(case_id)["status"] == "PLAN_READY"


def test_report_rejection_and_reexecution(service, approved_case):
    case_id = approved_case["case_id"]
    service.run_execution(case_id)
    service.decide_report(case_id, analyst="Tester", approve=False)
    assert service.store.get_case(case_id)["status"] == "REPORT_REJECTED"
    service.run_execution(case_id)
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"


def test_decisions_require_the_right_state(service, case):
    with pytest.raises(GateError):
        service.decide_plan(case["case_id"], analyst="T", approve=True)
    with pytest.raises(GateError):
        service.decide_report(case["case_id"], analyst="T", approve=True)
