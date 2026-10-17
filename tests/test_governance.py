"""Round 10 — source governance.

The registry (sources are governed records; ingestion refuses
unregistered ones on both paths; briefs fail closed on entries
that name nothing), provenance, retention with legal hold,
deletion with a tombstone, and redaction with its claim flags.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deep_diligence import evidence as evidence_mod
from deep_diligence.api import create_app
from deep_diligence.config import load_settings
from deep_diligence.corpus import CORPUS, CorpusDoc, GOLD_CASES
from deep_diligence.evidence import ingest_case_evidence
from deep_diligence.ledger import verify_ledger
from deep_diligence.models import BriefIn
from deep_diligence.service import CaseService, GateError


@pytest.fixture()
def client(settings):
    return TestClient(create_app(settings))


def _draft_case(service: CaseService, gold_brief) -> str:
    case = service.create_case(BriefIn(**gold_brief))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    return case_id


# ---------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------


def test_registry_is_seeded_from_the_corpus(service):
    sources = service.store.list_sources()
    names = {s["name"] for s in sources}
    assert names == {doc.source for doc in CORPUS}
    for record in sources:
        assert record["kind"] in (
            "filing", "news", "internal dataset", "web",
        )
        assert record["usage_basis"] in (
            "public", "licensed", "client-provided",
        )
        assert record["owner"]
        assert record["registered_by"] == "system (corpus seed)"
        assert record["history"][0]["action"] == "registered"


def test_registration_is_a_recorded_act(service):
    record = service.register_source(
        "Desk Lead",
        name="Client data room",
        kind="internal dataset",
        owner="Client (deal team)",
        usage_basis="client-provided",
        notes="Documents the client places in the data room.",
    )
    assert record["registered_by"] == "Desk Lead"
    assert record["history"][0]["analyst"] == "Desk Lead"
    with pytest.raises(GateError, match="already registered"):
        service.register_source(
            "Desk Lead",
            name="Client data room",
            kind="internal dataset",
            owner="Client (deal team)",
            usage_basis="client-provided",
        )
    with pytest.raises(ValueError, match="unknown source kind"):
        service.register_source(
            "Desk Lead",
            name="Rumor mill",
            kind="gossip",
            owner="Unknown",
            usage_basis="public",
        )


def test_upload_with_an_unregistered_source_is_refused(service, case):
    with pytest.raises(ValueError, match="Unregistered Wire Service"):
        service.add_case_evidence(
            case["case_id"],
            "Tester",
            title="A wire story",
            kind="news",
            source="Unregistered Wire Service",
            doc_date="2026-03-01",
            text="Some text the desk cannot stand behind.",
        )
    # And once the source is registered, the same upload succeeds.
    service.register_source(
        "Desk Lead",
        name="Unregistered Wire Service",
        kind="news",
        owner="Wire service publisher",
        usage_basis="licensed",
    )
    result = service.add_case_evidence(
        case["case_id"],
        "Tester",
        title="A wire story",
        kind="news",
        source="Unregistered Wire Service",
        doc_date="2026-03-01",
        text="Some text the desk can now stand behind.",
    )
    assert result["evidence"]["source"] == "Unregistered Wire Service"


def test_corpus_ingestion_refuses_an_unregistered_source(service, monkeypatch):
    rogue = CorpusDoc(
        slug="rogue-wire-story",
        company="helvion-grid",
        company_name="Helvion Grid Systems",
        kind="news",
        title="A rogue wire story",
        source="Rogue Wire",
        date="2026-03-01",
        body="Unvetted.",
        facts=(),
    )
    monkeypatch.setattr(
        evidence_mod, "docs_for_subject", lambda ref: [rogue]
    )
    fake_case = {
        "case_id": "case-rogue",
        "subject_ref": "helvion-grid",
        "brief": {"allowed_sources": []},
    }
    with pytest.raises(ValueError, match="Rogue Wire"):
        ingest_case_evidence(service.store, service.objects, fake_case)


# ---------------------------------------------------------------------
# Brief source policy
# ---------------------------------------------------------------------


def test_brief_with_an_unknown_allowed_source_is_refused(service, gold_brief):
    brief = BriefIn(**{**gold_brief, "allowed_sources": ["no-such-source"]})
    with pytest.raises(ValueError, match="no-such-source"):
        service.create_case(brief)


def test_brief_allowed_sources_may_name_documents_kinds_and_sources(
    service, gold_brief
):
    by_kind = service.create_case(
        BriefIn(**{**gold_brief, "allowed_sources": ["filing"]})
    )
    kinds = {
        ev["kind"] for ev in service.store.list_evidence(by_kind["case_id"])
    }
    assert kinds == {"filing"}

    slug = "helvion-annual-report-fy2025"
    by_doc = service.create_case(
        BriefIn(**{**gold_brief, "allowed_sources": [slug]})
    )
    docs = service.store.list_evidence(by_doc["case_id"])
    assert [ev["doc_slug"] for ev in docs] == [slug]

    by_source = service.create_case(
        BriefIn(
            **{
                **gold_brief,
                "allowed_sources": ["Company filing (annual report)"],
            }
        )
    )
    sources = {
        ev["source"]
        for ev in service.store.list_evidence(by_source["case_id"])
    }
    assert sources == {"Company filing (annual report)"}


def test_brief_refusal_surfaces_as_422(client, gold_brief):
    brief = {**gold_brief, "allowed_sources": ["no-such-source"]}
    response = client.post("/api/cases", json=brief)
    assert response.status_code == 422
    assert "no-such-source" in response.json()["detail"]


def test_case_sources_show_the_governance_records(service, gold_brief):
    brief = BriefIn(**{**gold_brief, "allowed_sources": ["filing"]})
    case = service.create_case(brief)
    view = service.case_sources(case["case_id"])
    assert view["allowed"][0]["entry"] == "filing"
    assert view["allowed"][0]["resolves_to"] == "kind"
    admitted = {
        s["name"] for s in view["allowed"][0]["sources"] if s
    }
    assert admitted == {"Company filing (annual report)"}
    used = {s["name"] for s in view["used"]}
    assert used == admitted


# ---------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------


def test_provenance_is_complete_for_a_gold_case(service, gold_brief):
    case_id = _draft_case(service, gold_brief)
    provenance = service.case_provenance(case_id)
    evidence = service.store.list_evidence(case_id)
    assert len(provenance["items"]) == len(evidence)
    by_id = {item["evidence_id"]: item for item in provenance["items"]}
    for ev in evidence:
        item = by_id[ev["evidence_id"]]
        assert item["sha256"] == ev["sha256"]
        assert item["ingestion"] == "corpus_intake"
        assert item["source_registry"] is not None
        assert item["source_registry"]["name"] == ev["source"]
    # Citations line up with the findings and the ledger.
    cited = {
        item["evidence_id"]
        for item in provenance["items"]
        if item["cited_by_claims"]
    }
    ledger = service.store.list_ledger(case_id)
    ledger_cited = {
        c["evidence_id"] for e in ledger for c in e["citations"]
    }
    assert cited == ledger_cited
    for item in provenance["items"]:
        if item["cited_by_claims"]:
            assert item["cited_by_findings"]


# ---------------------------------------------------------------------
# Retention, legal hold, deletion
# ---------------------------------------------------------------------


@pytest.fixture()
def short_retention_service(tmp_path):
    settings = load_settings(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir="migrations",
        retention_days=30,
    )
    return CaseService(settings)


def _backdate(service: CaseService, case_id: str, when: str) -> None:
    service.store.update_case(case_id, created_at=when)


def test_sweep_lists_past_retention_and_deletes_nothing(
    short_retention_service, gold_brief
):
    service = short_retention_service
    old = service.create_case(BriefIn(**gold_brief))
    _backdate(service, old["case_id"], "2026-01-01T00:00:00+00:00")
    fresh = service.create_case(BriefIn(**GOLD_CASES[1]["brief"]))
    due = service.retention_sweep()
    assert [row["case_id"] for row in due] == [old["case_id"]]
    assert due[0]["retention_due_at"].startswith("2026-01-31")
    # The sweep is a report: both cases are still there.
    assert service.store.get_case(old["case_id"]) is not None
    assert service.store.get_case(fresh["case_id"]) is not None


def test_legal_hold_blocks_deletion_and_the_sweep(
    short_retention_service, gold_brief
):
    service = short_retention_service
    case = service.create_case(BriefIn(**gold_brief))
    _backdate(service, case["case_id"], "2026-01-01T00:00:00+00:00")
    held = service.set_legal_hold(
        case["case_id"], "Counsel", True, "Litigation anticipated"
    )
    assert held["retention_class"] == "legal_hold"
    assert service.retention_sweep() == []
    with pytest.raises(GateError, match="legal hold"):
        service.delete_case(case["case_id"], "Desk Lead", "Retention ran out")
    with pytest.raises(GateError, match="already under legal hold"):
        service.set_legal_hold(case["case_id"], "Counsel", True)
    cleared = service.set_legal_hold(case["case_id"], "Counsel", False)
    assert cleared["retention_class"] == "standard"
    history = cleared["retention_history"]
    assert [h["action"] for h in history] == ["set", "cleared"]
    events = [
        e["event"] for e in service.store.list_audit(case["case_id"])
    ]
    assert "legal_hold_set" in events
    assert "legal_hold_cleared" in events


def test_deletion_tombstones_and_removes_the_case(service, gold_brief):
    case = service.create_case(BriefIn(**gold_brief))
    case_id = case["case_id"]
    before = service.store.list_evidence(case_id)
    assert service.objects.list_keys(f"cases/{case_id}/")
    tombstone = service.delete_case(
        case_id, "Desk Lead", "Engagement closed; retention served"
    )
    assert tombstone["case_id"] == case_id
    assert tombstone["subject"] == case["subject"]
    assert tombstone["deleted_by"] == "Desk Lead"
    assert tombstone["reason"] == "Engagement closed; retention served"
    assert tombstone["evidence_count"] == len(before)
    assert tombstone["final_report_sha256"] is None
    assert service.store.get_case(case_id) is None
    assert service.store.get_tombstone(case_id) == tombstone
    assert service.objects.list_keys(f"cases/{case_id}/") == []
    with pytest.raises(ValueError, match="recorded reason"):
        other = service.create_case(BriefIn(**GOLD_CASES[1]["brief"]))
        service.delete_case(other["case_id"], "Desk Lead", "  ")


def test_deletion_records_the_final_reports_hash(service, gold_brief):
    case_id = _draft_case(service, gold_brief)
    service.decide_report(case_id, analyst="Tester", approve=True)
    from deep_diligence.objects import sha256_hex

    report = service.store.get_report(case_id)
    expected = sha256_hex(report["markdown"].encode("utf-8"))
    tombstone = service.delete_case(case_id, "Desk Lead", "Matter closed")
    assert tombstone["final_report_sha256"] == expected


def test_deletion_surfaces_over_the_api(client, gold_brief):
    created = client.post("/api/cases", json=gold_brief).json()
    case_id = created["case_id"]
    assert created["retention_class"] == "standard"
    held = client.post(
        f"/api/cases/{case_id}/retention/hold",
        json={"analyst": "Counsel", "action": "set", "reason": "Hold it"},
    )
    assert held.status_code == 200
    assert held.json()["retention_class"] == "legal_hold"
    refused = client.post(
        f"/api/cases/{case_id}/delete",
        json={"analyst": "Desk Lead", "reason": "Retention served"},
    )
    assert refused.status_code == 409
    client.post(
        f"/api/cases/{case_id}/retention/hold",
        json={"analyst": "Counsel", "action": "clear"},
    )
    deleted = client.post(
        f"/api/cases/{case_id}/delete",
        json={"analyst": "Desk Lead", "reason": "Retention served"},
    )
    assert deleted.status_code == 200
    assert client.get(f"/api/cases/{case_id}").status_code == 404
    tombs = client.get("/api/retention/tombstones").json()
    assert [t["case_id"] for t in tombs] == [case_id]


# ---------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------


def _cited_evidence(service: CaseService, case_id: str) -> tuple[str, str]:
    report = service.store.get_report(case_id)
    claim = report["claims"][0]
    return claim["claim_id"], claim["evidence_ids"][0]


def test_redaction_replaces_content_and_keeps_the_original_hash(
    service, gold_brief
):
    case_id = _draft_case(service, gold_brief)
    _, evidence_id = _cited_evidence(service, case_id)
    ev = service.store.get_evidence(case_id, evidence_id)
    original_sha = ev["sha256"]
    result = service.redact_evidence(
        case_id, evidence_id, "Counsel", "Contains personal data"
    )
    redaction = result["redaction"]
    assert redaction["original_sha256"] == original_sha
    assert redaction["analyst"] == "Counsel"
    content = service.objects.get(ev["object_key"]).decode("utf-8")
    assert content.startswith("[REDACTED")
    assert "personal data" in content
    stored = service.store.get_evidence(case_id, evidence_id)
    assert stored["redacted"] is True
    # The ledger does not call a lawful redaction drift.
    check = verify_ledger(service.store, service.objects, case_id)
    assert check["status"] == "OK"
    assert evidence_id in {
        r["evidence_id"] for r in check["redacted_citations"]
    }
    with pytest.raises(GateError, match="already redacted"):
        service.redact_evidence(
            case_id, evidence_id, "Counsel", "Again"
        )
    with pytest.raises(ValueError, match="recorded reason"):
        other = service.store.list_evidence(case_id)[-1]
        service.redact_evidence(case_id, other["evidence_id"], "Counsel", " ")


def test_redaction_flags_block_approval_until_acknowledged(
    service, gold_brief
):
    case_id = _draft_case(service, gold_brief)
    claim_id, evidence_id = _cited_evidence(service, case_id)
    result = service.redact_evidence(
        case_id, evidence_id, "Counsel", "Source withdrew the document"
    )
    flags = result["flags"]
    assert flags and all(f["status"] == "OPEN" for f in flags)
    assert claim_id in {f["claim_id"] for f in flags}
    with pytest.raises(GateError, match="redacted"):
        service.decide_report(case_id, analyst="Tester", approve=True)
    for flag in flags:
        service.decide_redaction_flag(
            case_id, flag["flag_id"], "Tester", "acknowledge",
            note="Claim stands on the record as cited",
        )
    decided = service.decide_report(case_id, analyst="Tester", approve=True)
    assert decided["status"] == "FINAL"
    events = [e["event"] for e in service.store.list_audit(case_id)]
    assert "evidence_redacted" in events
    assert "redaction_claim_acknowledged" in events


def test_redaction_flag_send_back_mirrors_the_quarantine_path(
    service, gold_brief
):
    case_id = _draft_case(service, gold_brief)
    _, evidence_id = _cited_evidence(service, case_id)
    result = service.redact_evidence(
        case_id, evidence_id, "Counsel", "Document was superseded"
    )
    flag = result["flags"][0]
    finding_id = flag["finding_id"]
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    decided = service.decide_redaction_flag(
        case_id, flag["flag_id"], "Tester", "send_back"
    )
    assert decided["status"] == "SENT_BACK"
    assert service.store.get_case(case_id)["status"] == "APPROVED"
    remaining = {
        f["finding_id"] for f in service.store.list_findings(case_id)
    }
    assert finding_id not in remaining


def test_the_verifier_treats_redacted_evidence_as_missing(
    service, gold_brief
):
    from deep_diligence.verifier import verify_case

    case_id = _draft_case(service, gold_brief)
    _, evidence_id = _cited_evidence(service, case_id)
    service.redact_evidence(
        case_id, evidence_id, "Counsel", "Withdrawn by the issuer"
    )
    # A fresh verification pass over the persisted report: the
    # claims that cite the redacted item cannot verify against a
    # marker — they read as evidence_missing and quarantine.
    report = verify_case(service.store, service.objects, case_id)
    redacted_checks = [
        check
        for check in report["checks"]
        if any(
            "redacted" in failure["detail"]
            for failure in check["failures"]
        )
    ]
    assert redacted_checks
    assert all(
        "evidence_missing" in check["reasons"] for check in redacted_checks
    )
    assert all(
        check["verdict"] == "quarantined" for check in redacted_checks
    )


def test_redaction_surfaces_over_the_api(client, gold_brief):
    case_id = client.post("/api/cases", json=gold_brief).json()["case_id"]
    client.post(f"/api/cases/{case_id}/plan")
    client.post(
        f"/api/cases/{case_id}/plan/decision",
        json={"analyst": "Tester", "approve": True},
    )
    client.post(f"/api/cases/{case_id}/execute")
    provenance = client.get(f"/api/cases/{case_id}/provenance").json()
    assert provenance["items"]
    assert all(
        item["source_registry"] for item in provenance["items"]
    )
    target = next(
        item for item in provenance["items"] if item["cited_by_claims"]
    )
    redacted = client.post(
        f"/api/cases/{case_id}/evidence/{target['evidence_id']}/redact",
        json={"analyst": "Counsel", "reason": "Personal data"},
    )
    assert redacted.status_code == 200
    assert redacted.json()["flags"]
    refused = client.post(
        f"/api/cases/{case_id}/report/decision",
        json={"analyst": "Tester", "approve": True},
    )
    assert refused.status_code == 409
    view = client.get(f"/api/cases/{case_id}/redactions").json()
    for flag in view["flags"]:
        if flag["status"] == "OPEN":
            done = client.post(
                f"/api/cases/{case_id}/redaction-flags/{flag['flag_id']}/decision",
                json={"analyst": "Tester", "action": "acknowledge"},
            )
            assert done.status_code == 200
    approved = client.post(
        f"/api/cases/{case_id}/report/decision",
        json={"analyst": "Tester", "approve": True},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "FINAL"


def test_source_registration_surfaces_over_the_api(client):
    sources = client.get("/api/sources").json()
    assert len(sources) == len({doc.source for doc in CORPUS})
    created = client.post(
        "/api/sources",
        json={
            "analyst": "Desk Lead",
            "name": "Client data room",
            "kind": "internal dataset",
            "owner": "Client (deal team)",
            "usage_basis": "client-provided",
        },
    )
    assert created.status_code == 201
    duplicate = client.post(
        "/api/sources",
        json={
            "analyst": "Desk Lead",
            "name": "Client data room",
            "kind": "internal dataset",
            "owner": "Client (deal team)",
            "usage_basis": "client-provided",
        },
    )
    assert duplicate.status_code == 409
