"""Round 2: the citation ledger as a stored artifact, and the
EVIDENCE_DRIFT gate that re-verifies it before a report can be final.
"""

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.corpus import GOLD_CASES
from deep_diligence.ledger import verify_ledger
from deep_diligence.objects import sha256_hex
from deep_diligence.service import GateError


def _draft(service, case_id):
    service.run_execution(case_id)
    return service.store.get_report(case_id)


def test_ledger_records_every_claim_with_citation_time_hashes(service, approved_case):
    case_id = approved_case["case_id"]
    report = _draft(service, case_id)
    entries = service.store.list_ledger(case_id)
    claims = {c["claim_id"]: c for c in report["claims"]}
    assert len(entries) == len(claims) == 13  # Helvion coverage 5+5+3
    for entry in entries:
        claim = claims[entry["claim_id"]]
        assert entry["finding_id"] == claim["finding_id"]
        assert entry["question_id"] == claim["question_id"]
        assert entry["claim"] == claim["claim"]
        assert entry["citations"], "every ledger entry must cite evidence"
        for citation in entry["citations"]:
            assert citation["evidence_id"] in claim["evidence_ids"]
            ev = service.store.get_evidence(case_id, citation["evidence_id"])
            live = sha256_hex(service.objects.get(ev["object_key"]))
            # The hash at citation time: the live bytes hashed at
            # composition, equal to the ingestion hash while nothing
            # has touched the evidence.
            assert citation["sha256"] == live
            assert citation["ingested_sha256"] == ev["sha256"] == live
    assert report["ledger"]["entries"] == 13
    assert report["ledger"]["citations"] == sum(
        len(e["citations"]) for e in entries
    )
    assert report["integrity"]["status"] == "OK"


def test_references_section_is_generated_from_the_ledger(service, approved_case):
    case_id = approved_case["case_id"]
    report = _draft(service, case_id)
    markdown = report["markdown"]
    assert "## References" in markdown
    assert "## Sources" not in markdown
    entries = service.store.list_ledger(case_id)
    cited = {}
    for entry in entries:
        for citation in entry["citations"]:
            cited.setdefault(citation["evidence_id"], citation)
    reference_lines = [
        line for line in markdown.splitlines() if " · sha256 " in line
    ]
    assert len(reference_lines) == len(cited) == 5  # all five Helvion docs cited
    for evidence_id, citation in cited.items():
        line = next(l for l in reference_lines if f"[{evidence_id}]" in l)
        assert citation["source"] in line
        assert f"sha256 {citation['sha256'][:12]}" in line
    assert "cited by CLM-0001" in markdown


def test_gate_blocks_approval_when_evidence_drifts(service, approved_case):
    case_id = approved_case["case_id"]
    report = _draft(service, case_id)
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    evidence_id = report["claims"][0]["evidence_ids"][0]
    ev = service.store.get_evidence(case_id, evidence_id)
    original = service.objects.get(ev["object_key"])

    # The evidence changes after the report cited it.
    service.objects.put(ev["object_key"], b"# edited after the fact\n")
    check = verify_ledger(service.store, service.objects, case_id)
    assert check["status"] == "EVIDENCE_DRIFT"
    assert {d["evidence_id"] for d in check["drift"]} == {evidence_id}
    assert check["drift"][0]["cited_sha256"] == ev["sha256"]
    assert check["drift"][0]["current_sha256"] != ev["sha256"]

    with pytest.raises(GateError, match="EVIDENCE_DRIFT"):
        service.decide_report(case_id, analyst="Tester", approve=True)
    # The case has not moved, and the flag is on the report record.
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    flagged = service.store.get_report(case_id)
    assert flagged["integrity"]["status"] == "EVIDENCE_DRIFT"
    assert flagged["integrity"]["drift"][0]["evidence_id"] == evidence_id
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert "evidence_drift_detected" in events

    # Restoring the cited bytes makes a fresh verification pass, and
    # the same approval then goes through.
    service.objects.put(ev["object_key"], original)
    decided = service.decide_report(case_id, analyst="Tester", approve=True)
    assert decided["status"] == "FINAL"
    assert service.store.get_report(case_id)["integrity"]["status"] == "OK"


def test_ledger_over_http_and_drift_returns_409(settings):
    app = create_app(settings)
    client = TestClient(app)
    brief = GOLD_CASES[2]["brief"]  # Selene
    case_id = client.post("/api/cases", json=brief).json()["case_id"]
    client.post(f"/api/cases/{case_id}/plan")
    client.post(
        f"/api/cases/{case_id}/plan/decision",
        json={"analyst": "API Tester", "approve": True, "note": ""},
    )
    client.post(f"/api/cases/{case_id}/execute")

    body = client.get(f"/api/cases/{case_id}/ledger").json()
    assert len(body["entries"]) == 10  # Selene coverage 4+3+3
    assert body["verification"]["status"] == "OK"
    assert body["verification"]["citations_checked"] == 10

    service = app.state.service
    ev = service.store.list_evidence(case_id)[0]
    service.objects.put(ev["object_key"], b"# quietly changed\n")
    drifted = client.get(f"/api/cases/{case_id}/ledger").json()
    assert drifted["verification"]["status"] == "EVIDENCE_DRIFT"
    blocked = client.post(
        f"/api/cases/{case_id}/report/decision",
        json={"analyst": "API Tester", "approve": True, "note": ""},
    )
    assert blocked.status_code == 409
    assert "EVIDENCE_DRIFT" in blocked.json()["detail"]
    report = client.get(f"/api/cases/{case_id}/report").json()
    assert report["integrity"]["status"] == "EVIDENCE_DRIFT"
