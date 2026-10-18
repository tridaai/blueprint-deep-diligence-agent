"""Round 16: the challenge pass.

The pass is the desk's second opinion over the verified draft —
deterministic, computed from the stored records, recorded as
challenges the analyst must answer. These tests pin: the crafted
case records exactly the challenges its records support; the
gate refuses approval over an unanswered must-answer challenge
(and names it) while advisory challenges never block; each of
the three answers does what it says (rebuttal recorded with
counter-evidence from the case, revise returns the claim to
re-investigation, downgrade quarantines the claim out of the
report); the pass is idempotent by content; a re-composition
supersedes answered challenges instead of deleting them; and no
delivered format ever carries a challenge.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.challenges import sync_challenges
from deep_diligence.corpus import CHALLENGE_CASES
from deep_diligence.models import BriefIn
from deep_diligence.service import GateError

KESTREL = CHALLENGE_CASES[0]
ALDER = CHALLENGE_CASES[1]


def _run_case(service, brief) -> str:
    case = service.create_case(BriefIn(**brief))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    return case_id


def _kestrel(service) -> str:
    return _run_case(service, KESTREL["brief"])


def _open_must_answers(service, case_id) -> list[dict]:
    return [
        c
        for c in service.store.list_challenges(case_id)
        if c["severity"] == "must_answer" and c["status"] == "OPEN"
    ]


def _filing_id(service, case_id) -> str:
    return next(
        ev["evidence_id"]
        for ev in service.store.list_evidence(case_id)
        if ev["kind"] == "filing"
    )


def test_kestrel_records_exactly_the_crafted_challenges(service):
    case_id = _kestrel(service)
    raised = service.store.list_challenges(case_id)
    assert [(c["challenge_id"], c["rule"], c["severity"], c["claim_id"]) for c in raised] == [
        ("CHL-0001", "single_source_figure", "must_answer", "CLM-0005"),
        ("CHL-0002", "single_source_figure", "must_answer", "CLM-0006"),
        ("CHL-0003", "oldest_source_on_contested_metric", "advisory", "CLM-0009"),
    ]
    # The record names the evidence the rule read.
    revenue = raised[0]
    assert revenue["detail"]["assertions"] == [
        {"metric": "revenue", "period": "2025", "value_text": "$96M"}
    ]
    assert revenue["detail"]["evidence"]["kind"] == "news"
    assert revenue["detail"]["evidence"]["source_class"] == "secondary"
    stale = raised[2]
    assert stale["detail"]["conflict_id"] == "CNF-0001"
    assert stale["detail"]["oldest_doc_date"] == "2025-11-05"
    assert stale["detail"]["freshest_side_date"] == "2026-02-01"
    # The pass is an audited act.
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert "challenges_raised" in events


def test_clean_case_records_no_challenges(service):
    case_id = _run_case(service, ALDER["brief"])
    assert service.store.list_challenges(case_id) == []
    decided = service.decide_report(case_id, analyst="Tester", approve=True)
    assert decided["status"] == "FINAL"


def test_gate_refuses_approval_and_names_open_challenges(service):
    case_id = _kestrel(service)
    with pytest.raises(GateError) as excinfo:
        service.decide_report(case_id, analyst="Tester", approve=True)
    message = str(excinfo.value)
    assert "CHL-0001" in message and "CHL-0002" in message
    # The refusal changes nothing: the draft still stands at the gate.
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    # Rejection is the analyst's prerogative — the gate rule binds
    # approval only.
    decided = service.decide_report(
        case_id, analyst="Tester", approve=False, note="not yet"
    )
    assert decided["status"] == "REPORT_REJECTED"


def test_rebuttal_must_name_counter_evidence_from_the_case(service):
    case_id = _kestrel(service)
    challenge = _open_must_answers(service, case_id)[0]
    with pytest.raises(ValueError):
        service.decide_challenge(
            case_id, challenge["challenge_id"], "Tester", "rebut",
            note="  ", counter_evidence_ids=[_filing_id(service, case_id)],
        )
    with pytest.raises(ValueError):
        service.decide_challenge(
            case_id, challenge["challenge_id"], "Tester", "rebut",
            note="The figure stands as reported.",
        )
    with pytest.raises(ValueError):
        service.decide_challenge(
            case_id, challenge["challenge_id"], "Tester", "rebut",
            note="The figure stands as reported.",
            counter_evidence_ids=["EV-9999"],
        )
    # Nothing was recorded by the refused attempts.
    record = service.store.get_challenge(case_id, challenge["challenge_id"])
    assert record["status"] == "OPEN" and record["answers"] == []


def test_rebuttal_unblocks_approval_and_rides_the_case(service):
    case_id = _kestrel(service)
    filing = _filing_id(service, case_id)
    for challenge in _open_must_answers(service, case_id):
        service.decide_challenge(
            case_id, challenge["challenge_id"], "Tester", "rebut",
            note="Stands as reported, attributed to its source.",
            counter_evidence_ids=[filing],
        )
    decided = service.decide_report(case_id, analyst="Tester", approve=True)
    assert decided["status"] == "FINAL"
    for record in service.store.list_challenges(case_id):
        if record["severity"] != "must_answer":
            continue
        assert record["status"] == "ANSWERED"
        assert record["disposition"] == "rebutted"
        assert record["answers"][0]["counter_evidence_ids"] == [filing]
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert events.count("challenge_rebutted") == 2
    # An answered challenge cannot be answered again.
    answered = next(
        c for c in service.store.list_challenges(case_id)
        if c["severity"] == "must_answer"
    )
    with pytest.raises(GateError):
        service.decide_challenge(
            case_id, answered["challenge_id"], "Tester", "rebut",
            note="again", counter_evidence_ids=[filing],
        )


def test_revise_sends_the_claim_back_for_reinvestigation(service):
    case_id = _kestrel(service)
    challenge = _open_must_answers(service, case_id)[0]
    findings_before = service.store.list_findings(case_id)
    record = service.decide_challenge(
        case_id, challenge["challenge_id"], "Tester", "revise",
        note="Re-derive this from a primary source.",
    )
    assert record["status"] == "ANSWERED"
    assert record["disposition"] == "revised"
    # The reinvestigate path's effects: the finding leaves the
    # record and the case returns to APPROVED, ready to execute.
    findings = service.store.list_findings(case_id)
    assert len(findings) == len(findings_before) - 1
    assert all(f["finding_id"] != challenge["finding_id"] for f in findings)
    assert service.store.get_case(case_id)["status"] == "APPROVED"


def test_downgrade_quarantines_the_claim_and_rerenders(service):
    case_id = _kestrel(service)
    debt = next(
        c for c in _open_must_answers(service, case_id)
        if c["detail"]["assertions"][0]["metric"] == "net debt"
    )
    coverage_before = service.store.get_report(case_id)["coverage"]["q1"]
    record = service.decide_challenge(
        case_id, debt["challenge_id"], "Tester", "downgrade",
        note="The desk will not carry this figure on trade coverage.",
    )
    assert record["disposition"] == "downgraded"
    quarantine_id = record["answers"][0]["quarantine_id"]
    shelf = service.store.get_quarantine(case_id, quarantine_id)
    assert shelf["reason"] == "challenge_downgraded"
    assert shelf["status"] == "QUARANTINED"
    assert shelf["claim_id"] == debt["claim_id"]
    # The claim is out of the body and every derived view agrees.
    report = service.store.get_report(case_id)
    assert all(c["claim_id"] != debt["claim_id"] for c in report["claims"])
    assert "$18M" not in report["markdown"]
    assert report["coverage"]["q1"] == coverage_before - 1
    model_q1 = next(
        q for q in report["model"]["questions"] if q["question_id"] == "q1"
    )
    assert all(c["claim_id"] != debt["claim_id"] for c in model_q1["claims"])
    # The downgraded claim's shelf record can still be decided like
    # any other quarantine record.
    decided = service.decide_quarantine(
        case_id, quarantine_id, "Tester", "drop", "confirmed out"
    )
    assert decided["status"] == "DROPPED"


def test_advisory_challenges_never_block(service, gold_brief):
    case_id = _run_case(service, gold_brief)  # Helvion
    raised = service.store.list_challenges(case_id)
    assert raised, "the gold draft should carry advisory challenges"
    assert all(c["severity"] == "advisory" for c in raised)
    decided = service.decide_report(case_id, analyst="Tester", approve=True)
    assert decided["status"] == "FINAL"


def test_pass_is_idempotent_by_content(service):
    case_id = _kestrel(service)
    before = service.store.list_challenges(case_id)
    again = service.run_challenge_pass(case_id)
    assert again == before
    assert service.store.list_challenges(case_id) == before


def test_superseded_not_deleted_when_the_draft_moves_on(service):
    case_id = _kestrel(service)
    filing = _filing_id(service, case_id)
    stale = next(
        c for c in service.store.list_challenges(case_id)
        if c["rule"] == "oldest_source_on_contested_metric"
    )
    service.decide_challenge(
        case_id, stale["challenge_id"], "Tester", "rebut",
        note="The launch count is labelled with its date in the report.",
        counter_evidence_ids=[filing],
    )
    # A later draft in which the challenged claim no longer stands
    # (its finding left the case): the sync supersedes the answered
    # challenge — the objection and its answer stay on the case.
    case = service.store.get_case(case_id)
    report = service.store.get_report(case_id)
    moved = {
        **report,
        "claims": [
            c for c in report["claims"] if c["claim_id"] != stale["claim_id"]
        ],
    }
    outcome = sync_challenges(service.store, case, moved)
    record = service.store.get_challenge(case_id, stale["challenge_id"])
    assert record["status"] == "SUPERSEDED"
    assert record["answers"][0]["disposition"] == "rebutted"
    assert record["superseded_at"] is not None
    # Nothing was deleted: the shelf only grew or held.
    assert len(outcome["records"]) >= 3
    assert all(
        r["challenge_id"] for r in service.store.list_challenges(case_id)
    )


def test_challenges_never_reach_the_delivered_report(service):
    case_id = _kestrel(service)
    report = service.store.get_report(case_id)
    export = service.report_export(case_id)
    import json

    delivered = report["markdown"] + report["html"] + json.dumps(export)
    assert "CHL-" not in delivered
    assert "single_source_figure" not in delivered
    assert "must_answer" not in delivered


def test_challenge_decisions_over_http(settings):
    app = create_app(settings)
    client = TestClient(app)
    case_id = client.post("/api/cases", json=KESTREL["brief"]).json()["case_id"]
    client.post(f"/api/cases/{case_id}/plan")
    client.post(
        f"/api/cases/{case_id}/plan/decision",
        json={"analyst": "API Tester", "approve": True, "note": ""},
    )
    client.post(f"/api/cases/{case_id}/execute")
    body = client.get(f"/api/cases/{case_id}/challenges").json()
    assert len(body) == 3
    must = [c for c in body if c["severity"] == "must_answer"]
    assert len(must) == 2
    blocked = client.post(
        f"/api/cases/{case_id}/report/decision",
        json={"analyst": "API Tester", "approve": True, "note": ""},
    )
    assert blocked.status_code == 409
    assert "CHL-" in blocked.json()["detail"]
    filing = next(
        ev["evidence_id"]
        for ev in client.get(f"/api/cases/{case_id}/evidence").json()
        if ev["kind"] == "filing"
    )
    ok = client.post(
        f"/api/cases/{case_id}/challenges/{must[0]['challenge_id']}/decision",
        json={
            "analyst": "API Tester",
            "action": "rebut",
            "note": "Stands as reported, attributed to its source.",
            "counter_evidence_ids": [filing],
        },
    )
    assert ok.status_code == 200
    assert ok.json()["challenge"]["disposition"] == "rebutted"
    missing = client.post(
        f"/api/cases/{case_id}/challenges/CHL-9999/decision",
        json={"analyst": "API Tester", "action": "rebut", "note": "x",
              "counter_evidence_ids": [filing]},
    )
    assert missing.status_code == 404
    bad_action = client.post(
        f"/api/cases/{case_id}/challenges/{must[1]['challenge_id']}/decision",
        json={"analyst": "API Tester", "action": "dismiss", "note": ""},
    )
    assert bad_action.status_code == 422
