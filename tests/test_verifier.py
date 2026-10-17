"""Round 3: the independent verifier and the quarantine.

The verifier re-reads every composed claim against the evidence it
cites — fetched from the object store, not from the composer's or
the ledger's structures — and quarantines what the evidence does not
support. A quarantined claim can be sent back for re-investigation
or dropped; it can never be approved into the report.
"""

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.corpus import INTEGRITY_CASES
from deep_diligence.models import BriefIn
from deep_diligence.service import GateError
from deep_diligence.verifier import check_claim, verify_case

ASTER = INTEGRITY_CASES[0]
POISON = {p["expect_reason"]: p for p in ASTER["poisoned_findings"]}


def _aster_approved(service) -> str:
    case = service.create_case(BriefIn(**ASTER["brief"]))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    return case_id


def _inject_poison(service, case_id: str) -> dict[str, str]:
    by_slug = {
        ev["doc_slug"]: ev["evidence_id"]
        for ev in service.store.list_evidence(case_id)
    }
    finding_ids = {}
    for poison in ASTER["poisoned_findings"]:
        finding = service.store.add_finding_auto_id(
            case_id,
            {
                "question_id": poison["question_id"],
                "claim": poison["claim"],
                "evidence_ids": [by_slug[poison["doc_slug"]]],
                "fact_ref": None,
            },
        )
        finding_ids[poison["expect_reason"]] = finding["finding_id"]
    return finding_ids


def _poisoned_draft(service) -> str:
    case_id = _aster_approved(service)
    _inject_poison(service, case_id)
    service.run_execution(case_id)
    return case_id


def test_check_claim_catches_overstated_figure_and_foreign_entity():
    evidence = ["Revenue grew 11% year over year to $186M in FY2025."]
    failures = check_claim(
        "Revenue grew 11% year over year to $198M in FY2025.", evidence
    )
    assert [f["reason"] for f in failures] == ["figure_not_in_evidence"]
    assert "$198M" in failures[0]["detail"]

    failures = check_claim(
        "Aster Foundry supplies castings to Helvion Grid Services.",
        ["Aster Foundry operates 3 casting facilities in Ohio."],
    )
    assert [f["reason"] for f in failures] == ["entity_mismatch"]
    assert check_claim(evidence[0], evidence) == []


def test_verifier_quarantines_both_poisoned_findings(service):
    case_id = _poisoned_draft(service)
    result = service.store.get_verifier_report(case_id)
    assert result["checked"] == result["supported"] + 2
    assert result["quarantined"] == 2
    assert result["by_reason"] == {
        "figure_not_in_evidence": 1,
        "entity_mismatch": 1,
    }
    quarantine = service.store.list_quarantine(case_id)
    assert {q["reason"] for q in quarantine} == {
        "figure_not_in_evidence",
        "entity_mismatch",
    }
    assert all(q["status"] == "QUARANTINED" for q in quarantine)
    report = service.store.get_report(case_id)
    claim_texts = [c["claim"] for c in report["claims"]]
    for poison in ASTER["poisoned_findings"]:
        assert poison["claim"] not in claim_texts
        assert poison["claim"] not in report["markdown"]
    assert "$198M" not in report["markdown"]
    assert {q["claim_id"] for q in report["quarantined"]} == {
        q["claim_id"] for q in quarantine
    }
    # The ledger describes the report the analyst sees: supported only.
    assert len(service.store.list_ledger(case_id)) == len(report["claims"])
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert "verification_completed" in events
    assert "claims_quarantined" in events


def test_clean_case_verifies_fully_supported(service, approved_case):
    case_id = approved_case["case_id"]
    report = service.run_execution(case_id)
    result = service.store.get_verifier_report(case_id)
    assert result["checked"] == len(report["claims"]) > 0
    assert result["supported"] == result["checked"]
    assert result["quarantined"] == 0
    assert service.store.list_quarantine(case_id) == []
    assert report["verification"]["quarantined"] == 0
    assert "## Verification" in report["markdown"]


def test_quarantined_claim_never_reaches_the_final_report(service):
    case_id = _poisoned_draft(service)
    decided = service.decide_report(case_id, analyst="Tester", approve=True)
    assert decided["status"] == "FINAL"
    final = service.store.get_report(case_id)
    for poison in ASTER["poisoned_findings"]:
        assert poison["claim"] not in [c["claim"] for c in final["claims"]]
        assert poison["claim"] not in final["markdown"]


def test_send_back_returns_case_to_execution(service):
    case_id = _poisoned_draft(service)
    quarantine = service.store.list_quarantine(case_id)
    figure = next(
        q for q in quarantine if q["reason"] == "figure_not_in_evidence"
    )
    entity = next(q for q in quarantine if q["reason"] == "entity_mismatch")
    # Dropping the other claim first leaves the case at the gate.
    service.decide_quarantine(
        case_id, entity["quarantine_id"], "Tester", "drop", "Not our fact."
    )
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    record = service.decide_quarantine(
        case_id,
        figure["quarantine_id"],
        "Tester",
        "reinvestigate",
        "Re-check the revenue figure against the audited filing.",
    )
    assert record["status"] == "SENT_BACK"
    assert record["decided_by"] == "Tester"
    case = service.store.get_case(case_id)
    assert case["status"] == "APPROVED"
    # Both findings left the case record: a re-investigation must
    # re-derive claims from evidence, not resurrect these.
    remaining = {f["finding_id"] for f in service.store.list_findings(case_id)}
    assert figure["finding_id"] not in remaining
    assert entity["finding_id"] not in remaining
    # Re-execution composes a clean draft; nothing is quarantined.
    report = service.run_execution(case_id)
    assert service.store.list_quarantine(case_id) == []
    assert report["verification"]["quarantined"] == 0
    decided = service.decide_report(case_id, analyst="Tester", approve=True)
    assert decided["status"] == "FINAL"
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert "claim_sent_back" in events and "claim_dropped" in events


def test_drop_is_terminal_and_blocks_further_decisions(service):
    case_id = _poisoned_draft(service)
    record = service.store.list_quarantine(case_id)[0]
    decided = service.decide_quarantine(
        case_id, record["quarantine_id"], "Tester", "drop", "Drop it."
    )
    assert decided["status"] == "DROPPED"
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    with pytest.raises(GateError, match="already DROPPED"):
        service.decide_quarantine(
            case_id, record["quarantine_id"], "Tester", "drop", "Again."
        )
    with pytest.raises(KeyError):
        service.decide_quarantine(case_id, "Q-9999", "Tester", "drop", "")


def test_verifier_reads_the_object_store_not_the_ledger(service, approved_case):
    case_id = approved_case["case_id"]
    service.run_execution(case_id)
    # Wipe the citation ledger: the verifier's verdicts must not move,
    # because it never consults the ledger — it re-reads the report's
    # claims and fetches the evidence bytes itself.
    service.store.replace_ledger(case_id, [])
    result = verify_case(service.store, service.objects, case_id)
    assert result["quarantined"] == 0
    # Now change what the evidence actually says. The claim citing it
    # must fall, on the figure check, against the live bytes.
    report = service.store.get_report(case_id)
    claim = next(c for c in report["claims"] if "$412M" in c["claim"])
    ev = service.store.get_evidence(case_id, claim["evidence_ids"][0])
    original = service.objects.get(ev["object_key"])
    service.objects.put(
        ev["object_key"], original.replace(b"$412M", b"$400M")
    )
    result = verify_case(service.store, service.objects, case_id)
    assert result["quarantined"] == 1
    record = service.store.list_quarantine(case_id)[0]
    assert record["claim_id"] == claim["claim_id"]
    assert record["reason"] == "figure_not_in_evidence"


class _AlwaysUnsupported:
    def invoke(self, messages):
        class _Response:
            content = "UNSUPPORTED"

        return _Response()


def test_model_check_is_advisory_only(service, approved_case):
    case_id = approved_case["case_id"]
    service.run_execution(case_id)
    result = verify_case(
        service.store,
        service.objects,
        case_id,
        model=_AlwaysUnsupported(),
        model_name="fake-model",
    )
    # The model says every claim is unsupported; the deterministic
    # checks — the gate — find every claim supported, so nothing is
    # quarantined. The model verdict is on the record, as a signal.
    assert result["mode"] == "live-assisted"
    assert result["quarantined"] == 0
    assert all(
        c["model_check"] == "unsupported" for c in result["checks"]
    )
    assert all(c["verdict"] == "supported" for c in result["checks"])


def test_verification_and_quarantine_over_http(settings):
    app = create_app(settings)
    client = TestClient(app)
    service = app.state.service
    case_id = client.post("/api/cases", json=ASTER["brief"]).json()["case_id"]
    client.post(f"/api/cases/{case_id}/plan")
    client.post(
        f"/api/cases/{case_id}/plan/decision",
        json={"analyst": "API Tester", "approve": True, "note": ""},
    )
    _inject_poison(service, case_id)
    client.post(f"/api/cases/{case_id}/execute")

    verification = client.get(f"/api/cases/{case_id}/verification")
    assert verification.status_code == 200
    body = verification.json()
    assert body["quarantined"] == 2
    assert body["checked"] == body["supported"] + body["quarantined"]

    quarantine = client.get(f"/api/cases/{case_id}/quarantine").json()
    assert len(quarantine) == 2
    assert {q["reason"] for q in quarantine} == {
        "figure_not_in_evidence",
        "entity_mismatch",
    }

    missing = client.post(
        f"/api/cases/{case_id}/quarantine/Q-9999/decision",
        json={"analyst": "API Tester", "action": "drop", "note": ""},
    )
    assert missing.status_code == 404

    dropped = client.post(
        f"/api/cases/{case_id}/quarantine/{quarantine[0]['quarantine_id']}/decision",
        json={"analyst": "API Tester", "action": "drop", "note": "Drop it."},
    )
    assert dropped.status_code == 200
    assert dropped.json()["quarantine"]["status"] == "DROPPED"
    assert dropped.json()["case_status"] == "DRAFT_READY"

    sent_back = client.post(
        f"/api/cases/{case_id}/quarantine/{quarantine[1]['quarantine_id']}/decision",
        json={
            "analyst": "API Tester",
            "action": "reinvestigate",
            "note": "Re-derive this claim from the filing.",
        },
    )
    assert sent_back.status_code == 200
    assert sent_back.json()["quarantine"]["status"] == "SENT_BACK"
    assert sent_back.json()["case_status"] == "APPROVED"
