import pytest
from pydantic import ValidationError

from deep_diligence.corpus import docs_for_subject
from deep_diligence.models import BriefIn
from deep_diligence.service import CaseService


def test_case_created_with_scoped_evidence(service, case):
    assert case["status"] == "INTAKE"
    evidence = service.store.list_evidence(case["case_id"])
    assert len(evidence) == len(docs_for_subject("helvion-grid")) == 5
    assert evidence[0]["evidence_id"] == "EV-0001"
    assert all(len(ev["sha256"]) == 64 for ev in evidence)
    events = [a["event"] for a in service.store.list_audit(case["case_id"])]
    assert events[:2] == ["case_created", "evidence_ingested"]


def test_unknown_subject_refused(service, gold_brief):
    brief = BriefIn(**{**gold_brief, "subject_ref": "no-such-company"})
    with pytest.raises(ValueError, match="unknown subject_ref"):
        service.create_case(brief)


def test_allowed_sources_scope_the_corpus(service, gold_brief):
    brief = BriefIn(**{**gold_brief, "allowed_sources": ["filing"]})
    case = service.create_case(brief)
    kinds = {ev["kind"] for ev in service.store.list_evidence(case["case_id"])}
    assert kinds == {"filing"}


def test_brief_validation():
    with pytest.raises(ValidationError):
        BriefIn(
            subject="X Co", subject_ref="helvion-grid", analyst="A",
            questions=[
                {"id": "q1", "text": "What is the revenue trend here?", "workstream": "financial-scan"},
                {"id": "q1", "text": "What is the revenue trend here?", "workstream": "financial-scan"},
            ],
        )
    with pytest.raises(ValidationError):
        BriefIn(
            subject="X Co", subject_ref="helvion-grid", analyst="A",
            questions=[
                {"id": "q1", "text": "What is the revenue trend here?", "workstream": "legal-scan"},
            ],
        )
    with pytest.raises(ValidationError):
        BriefIn(
            subject="X Co", subject_ref="helvion-grid", analyst="A", questions=[],
        )
