"""Round 8: parallel workstreams — dependencies, waves, handoffs,
failure isolation, and deterministic composition.

The schedule is the shell's, not the model's: these tests grade it
off the audit trail and the case record, the same evidence the eval
packs grade.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.composer import compose_report
from deep_diligence.config import load_settings
from deep_diligence.corpus import DEPENDENCY_CASES
from deep_diligence.models import BriefIn
from deep_diligence.schedule import plan_defects, waves
from deep_diligence.service import CaseService, GateError

VERDANT = DEPENDENCY_CASES[0]["brief"]


def _ws(qid, deps=()):
    return {
        "question_id": qid,
        "workstream": "financial-scan",
        "objective": "x",
        "keywords": [],
        "depends_on": list(deps),
    }


# -- the graph rules ----------------------------------------------------------


def test_plan_defects_names_an_unknown_dependency():
    defects = plan_defects([_ws("q1"), _ws("q2", ["q9"])])
    assert defects == [
        "workstream q2 depends on q9, "
        "which is not a workstream in this plan"
    ]


def test_plan_defects_names_a_self_dependency():
    defects = plan_defects([_ws("q1", ["q1"])])
    assert any("depends on itself" in d for d in defects)


def test_plan_defects_names_the_cycle_path():
    defects = plan_defects([_ws("q1", ["q2"]), _ws("q2", ["q1"])])
    assert any(
        d.startswith("dependency cycle:") and "q1" in d and "q2" in d
        for d in defects
    )


def test_a_plan_without_dependencies_is_clean_and_one_wave():
    plan = [_ws("q1"), _ws("q2"), _ws("q3")]
    assert plan_defects(plan) == []
    assert waves(plan) == [plan]


def test_waves_layer_a_chain_in_plan_order():
    plan = [_ws("q1"), _ws("q2", ["q1"]), _ws("q3")]
    layers = waves(plan)
    assert [[ws["question_id"] for ws in layer] for layer in layers] == [
        ["q1", "q3"],
        ["q2"],
    ]


def test_waves_handle_a_diamond():
    plan = [
        _ws("q1"),
        _ws("q2", ["q1"]),
        _ws("q3", ["q1"]),
        _ws("q4", ["q2", "q3"]),
    ]
    layers = waves(plan)
    assert [[ws["question_id"] for ws in layer] for layer in layers] == [
        ["q1"],
        ["q2", "q3"],
        ["q4"],
    ]


# -- the plan gate ------------------------------------------------------------


def _case_with_plan(service, plan_doc):
    case = service.create_case(BriefIn(**VERDANT))
    plan = service.store.save_plan(
        case["case_id"],
        {
            "case_id": case["case_id"],
            "status": "PENDING",
            "plan": plan_doc,
            "markdown": "",
            "decided_by": None,
            "decision_note": None,
        },
    )
    service.store.update_case(case["case_id"], status="PLAN_READY")
    return case, plan


def test_the_gate_refuses_a_plan_with_an_unknown_dependency(service):
    plan_doc = {
        "subject": "Verdant Robotics",
        "workstreams": [_ws("q1"), _ws("q2", ["q9"]), _ws("q3")],
    }
    case, _ = _case_with_plan(service, plan_doc)
    with pytest.raises(GateError, match="depends on q9"):
        service.decide_plan(case["case_id"], analyst="Tester", approve=True)
    # The refusal leaves the plan at the gate, undecided.
    assert service.store.get_case(case["case_id"])["status"] == "PLAN_READY"
    assert service.store.get_plan(case["case_id"])["status"] == "PENDING"


def test_the_gate_refuses_a_cyclic_plan(service):
    plan_doc = {
        "subject": "Verdant Robotics",
        "workstreams": [_ws("q1", ["q2"]), _ws("q2", ["q1"]), _ws("q3")],
    }
    case, _ = _case_with_plan(service, plan_doc)
    with pytest.raises(GateError, match="dependency cycle"):
        service.decide_plan(case["case_id"], analyst="Tester", approve=True)


def test_a_cyclic_brief_never_gets_an_approved_plan(service):
    brief = {
        **VERDANT,
        "questions": [
            {**VERDANT["questions"][0], "depends_on": ["q2"]},
            {**VERDANT["questions"][1], "depends_on": ["q1"]},
            VERDANT["questions"][2],
        ],
    }
    case = service.create_case(BriefIn(**brief))
    service.run_planning(case["case_id"])
    with pytest.raises(GateError, match="dependency cycle"):
        service.decide_plan(case["case_id"], analyst="Tester", approve=True)


def test_a_valid_dependency_plan_approves(service):
    case = service.create_case(BriefIn(**VERDANT))
    service.run_planning(case["case_id"])
    decided = service.decide_plan(case["case_id"], analyst="Tester", approve=True)
    assert decided["status"] == "APPROVED"
    plan = service.store.get_plan(case["case_id"])
    deps = {
        ws["question_id"]: ws.get("depends_on")
        for ws in plan["plan"]["workstreams"]
    }
    assert deps == {"q1": [], "q2": ["q1"], "q3": []}


# -- scheduled execution --------------------------------------------------------


def _run_verdant(service):
    case = service.create_case(BriefIn(**VERDANT))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    report = service.run_execution(case_id)
    return case_id, report


def _audit_index(audit, event, question_id):
    for i, entry in enumerate(audit):
        payload = entry.get("payload") or {}
        if entry["event"] == event and payload.get("question_id") == question_id:
            return i
    return None


def test_execution_follows_the_schedule(service):
    case_id, report = _run_verdant(service)
    audit = service.store.list_audit(case_id)
    started = next(
        e for e in audit if e["event"] == "execution_started"
    )
    assert started["payload"]["waves"] == [["q1", "q3"], ["q2"]]
    # The dependent started strictly after its prerequisite finished.
    assert _audit_index(audit, "workstream_finished", "q1") < _audit_index(
        audit, "workstream_started", "q2"
    )
    # And its handoff carried exactly q1's findings — no more, no less.
    handoff = next(
        e["payload"] for e in audit if e["event"] == "workstream_handoff"
    )
    q1_findings = {
        f["finding_id"]
        for f in service.store.list_findings(case_id)
        if f["question_id"] == "q1"
    }
    assert handoff["question_id"] == "q2"
    assert handoff["depends_on"] == ["q1"]
    assert set(handoff["finding_ids"]) == q1_findings
    assert q1_findings  # the handoff was not vacuous
    statuses = {s["question_id"]: s["status"] for s in report["workstreams"]}
    assert statuses == {"q1": "completed", "q2": "completed", "q3": "completed"}


def test_workstream_statuses_over_http(settings):
    client = TestClient(create_app(settings))
    created = client.post("/api/cases", json=VERDANT)
    case_id = created.json()["case_id"]
    client.post(f"/api/cases/{case_id}/plan")
    client.post(
        f"/api/cases/{case_id}/plan/decision",
        json={"analyst": "Tester", "approve": True, "note": ""},
    )
    client.post(f"/api/cases/{case_id}/execute")
    rows = client.get(f"/api/cases/{case_id}/workstreams").json()
    assert {r["question_id"]: r["status"] for r in rows} == {
        "q1": "completed",
        "q2": "completed",
        "q3": "completed",
    }


def test_a_failed_wave_is_isolated_and_named(service, monkeypatch):
    case = service.create_case(BriefIn(**VERDANT))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    original = service._run_agent
    execution_runs = {"count": 0}

    def flaky(case_, message, phase):
        if phase == "execution":
            execution_runs["count"] += 1
            if execution_runs["count"] == 2:
                raise RuntimeError("the market sub-agent died mid-run")
        return original(case_, message, phase)

    monkeypatch.setattr(service, "_run_agent", flaky)
    report = service.run_execution(case_id)
    # The case still drafts; the other branches completed.
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    statuses = {s["question_id"]: s for s in report["workstreams"]}
    assert statuses["q1"]["status"] == "completed"
    assert statuses["q3"]["status"] == "completed"
    assert statuses["q2"]["status"] == "failed"
    assert "RuntimeError" in statuses["q2"]["reason"]
    # The coverage table names the failure on the uncovered row,
    # in the record and in the rendered draft.
    row = next(
        r for r in report["coverage_detail"] if r["question_id"] == "q2"
    )
    assert row["status"] == "NOT COVERED"
    assert "market sub-agent died" in row["reason"]
    assert row["reason"] in report["markdown"]
    audit = service.store.list_audit(case_id)
    assert any(e["event"] == "workstream_run_failed" for e in audit)


def test_an_empty_branch_reads_not_covered_with_its_reason(service):
    variant = DEPENDENCY_CASES[1]
    case = service.create_case(BriefIn(**variant["brief"]))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    report = service.run_execution(case_id)
    statuses = {s["question_id"]: s["status"] for s in report["workstreams"]}
    assert statuses == {"q1": "completed", "q2": "completed", "q3": "empty"}
    row = next(
        r for r in report["coverage_detail"] if r["question_id"] == "q3"
    )
    assert row["status"] == "NOT COVERED"
    assert row["reason"] == "the workstream ran and recorded no findings"
    assert row["reason"] in report["markdown"]


# -- determinism ----------------------------------------------------------------


def test_repeated_executions_compose_the_same_report(tmp_path):
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    markdowns = []
    for name in ("run-a", "run-b"):
        settings = load_settings(
            sqlite_path=str(tmp_path / name / "diligence.db"),
            object_dir=str(tmp_path / name / "objects"),
            migrations_dir=str(repo_root / "migrations"),
        )
        service = CaseService(settings)
        case_id, report = _run_verdant(service)
        markdowns.append(report["markdown"])
    assert markdowns[0] == markdowns[1]


def test_composition_orders_claims_by_plan_not_by_recording(service, case):
    case_id = case["case_id"]
    evidence_id = service.store.list_evidence(case_id)[0]["evidence_id"]
    # Record findings in scrambled order across questions: q3
    # first, q1's facts out of order, q2 last.
    scrambled = [
        ("q3", "Claim three B.", "F-HEL-302"),
        ("q1", "Claim one B.", "F-HEL-102"),
        ("q3", "Claim three A.", "F-HEL-301"),
        ("q1", "Claim one A.", "F-HEL-101"),
        ("q2", "Claim two A.", "F-HEL-201"),
    ]
    for question_id, claim, fact_ref in scrambled:
        service.store.add_finding_auto_id(
            case_id,
            {
                "question_id": question_id,
                "claim": claim,
                "evidence_ids": [evidence_id],
                "fact_ref": fact_ref,
            },
        )
    case = service.store.get_case(case_id)
    report = compose_report(service.store, service.objects, case)
    assert [
        (c["question_id"], c["fact_ref"]) for c in report["claims"]
    ] == [
        ("q1", "F-HEL-101"),
        ("q1", "F-HEL-102"),
        ("q2", "F-HEL-201"),
        ("q3", "F-HEL-301"),
        ("q3", "F-HEL-302"),
    ]
    assert [c["claim_id"] for c in report["claims"]] == [
        "CLM-0001",
        "CLM-0002",
        "CLM-0003",
        "CLM-0004",
        "CLM-0005",
    ]
