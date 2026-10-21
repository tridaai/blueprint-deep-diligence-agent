"""Round 11: report composer depth.

The report is a structured model first, and three formats render
it: the markdown, a self-contained print-ready HTML page, and the
JSON export. The executive summary is composed after verification
from the supported claims alone, under a summary-integrity check
that drops any sentence stating a figure, date, or entity no
supported claim states. Every question section carries a support
grade from the one documented rule. The Risks and open items
section is generated from records. Analyst section notes annotate
the export and nothing else — they never touch the report body or
its hash.
"""

from __future__ import annotations

import copy
import html as html_lib
import re

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.config import load_settings
from deep_diligence.corpus import DEPENDENCY_CASES, GOLD_CASES, INTEGRITY_CASES
from deep_diligence.models import BriefIn
from deep_diligence.report_model import (
    build_summary,
    check_summary_sentences,
    question_grade,
)
from deep_diligence.service import CaseService, GateError

HELVION = GOLD_CASES[0]
ASTER = INTEGRITY_CASES[0]
VERDANT = DEPENDENCY_CASES[0]
FIGURE = re.compile(r"\$[\d,.]+(?:\.\d+)?[MBK]?|\d+(?:\.\d+)?%")


@pytest.fixture()
def settings(tmp_path):
    return load_settings(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir="migrations",
    )


@pytest.fixture()
def service(settings) -> CaseService:
    return CaseService(settings)


@pytest.fixture()
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


def _run_to_draft(service: CaseService, gold: dict) -> tuple[str, dict]:
    case = service.create_case(BriefIn(**copy.deepcopy(gold["brief"])))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Eval Harness", approve=True)
    report = service.run_execution(case_id)
    return case_id, report


def _aster_draft(service: CaseService) -> tuple[str, dict]:
    case = service.create_case(BriefIn(**ASTER["brief"]))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Eval Harness", approve=True)
    evidence_by_slug = {
        ev["doc_slug"]: ev["evidence_id"]
        for ev in service.store.list_evidence(case_id)
    }
    for poison in ASTER["poisoned_findings"]:
        service.store.add_finding_auto_id(
            case_id,
            {
                "question_id": poison["question_id"],
                "claim": poison["claim"],
                "evidence_ids": [evidence_by_slug[poison["doc_slug"]]],
                "fact_ref": None,
            },
        )
    return case_id, service.run_execution(case_id)


# ---------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------


def test_summary_is_composed_from_supported_claims_only(service):
    _, report = _run_to_draft(service, HELVION)
    summary = report["model"]["executive_summary"]
    supported = [c["claim"] for c in report["claims"]]
    # One headline per question that has supported claims, in brief
    # order, each a supported claim verbatim — the summary states
    # nothing the verified report does not.
    assert summary["sentences"]
    assert all(s in supported for s in summary["sentences"])
    assert summary["dropped"] == []
    counts = summary["counts"]
    assert counts["claims_supported"] == len(report["claims"]) == 13
    assert counts["conflicts_open"] == 1
    assert counts["questions_total"] == 3
    assert counts["questions_uncovered"] == 0
    markdown = report["markdown"]
    assert "## Executive summary" in markdown
    assert markdown.index("## Executive summary") < markdown.index("## q1")
    for sentence in summary["sentences"]:
        assert sentence in markdown


def test_summary_integrity_check_drops_and_records_a_failing_sentence():
    supported = [
        "Revenue grew 18% year over year to $412M in FY2025.",
        "Helvion Grid Systems operates from a plant in Bilbao.",
    ]
    fabricated = (
        "Zorblat Industries acquired the company for $999M in June 2031."
    )
    kept, dropped = check_summary_sentences(
        [supported[0], fabricated], supported
    )
    assert kept == [supported[0]]
    assert len(dropped) == 1
    assert dropped[0]["sentence"] == fabricated
    failures = " ".join(f["detail"] for f in dropped[0]["failures"])
    assert "$999M" in failures
    assert "Zorblat Industries" in failures


def test_summary_counts_uncovered_questions_from_coverage(service):
    case_id, report = _run_to_draft(service, DEPENDENCY_CASES[1])
    counts = report["model"]["executive_summary"]["counts"]
    assert counts["questions_uncovered"] >= 1
    # The verdict vocabulary of the counts line renders in markdown.
    assert (
        f"Questions not covered: {counts['questions_uncovered']}"
        in report["markdown"]
    )


# ---------------------------------------------------------------------
# Support grades
# ---------------------------------------------------------------------

_EVIDENCE = {
    "EV-F": {"kind": "filing"},
    "EV-N": {"kind": "news"},
    "EV-P": {"kind": "product"},
}


def _claim(claim_id: str, evidence_id: str) -> dict:
    return {
        "claim_id": claim_id,
        "finding_id": "FND-0001",
        "question_id": "q1",
        "claim": "text",
        "evidence_ids": [evidence_id],
    }


def test_question_grade_is_the_one_documented_rule():
    assert question_grade([], _EVIDENCE) == "UNANSWERED"
    # A single claim is THIN however strong its source.
    assert question_grade([_claim("CLM-1", "EV-F")], _EVIDENCE) == "THIN"
    # Two claims, one on a primary source: STRONG.
    assert (
        question_grade(
            [_claim("CLM-1", "EV-F"), _claim("CLM-2", "EV-N")], _EVIDENCE
        )
        == "STRONG"
    )
    # Two claims on secondary sources only: STANDARD.
    assert (
        question_grade(
            [_claim("CLM-1", "EV-N"), _claim("CLM-2", "EV-P")], _EVIDENCE
        )
        == "STANDARD"
    )
    # A claim the verifier's model check marked unsupported drags
    # the question to THIN however the rest stands.
    assert (
        question_grade(
            [_claim("CLM-1", "EV-F"), _claim("CLM-2", "EV-F")],
            _EVIDENCE,
            weak_claim_ids={"CLM-2"},
        )
        == "THIN"
    )


def test_grades_render_in_every_format(service):
    case_id, report = _run_to_draft(service, HELVION)
    model = report["model"]
    grades = {q["question_id"]: q["grade"] for q in model["questions"]}
    assert grades == {"q1": "STRONG", "q2": "STANDARD", "q3": "STANDARD"}
    for question in model["questions"]:
        assert (
            f"## {question['question_id']} — {question['text']} "
            f"· support: {question['grade']}" in report["markdown"]
        )
        assert (
            f'grade-{question["grade"]}' in report["html"]
            and f'>{question["grade"]}</span>' in report["html"]
        )
    exported = service.report_export(case_id)
    assert {
        q["question_id"]: q["grade"] for q in exported["questions"]
    } == grades


# ---------------------------------------------------------------------
# Risks and open items
# ---------------------------------------------------------------------


def test_open_items_are_generated_from_records(service):
    case_id, report = _aster_draft(service)
    items = report["model"]["risks_and_open_items"]
    assert items["empty"] is False
    # The planted Aster contradiction is open.
    assert len(items["conflicts"]) == 1
    assert items["conflicts"][0]["status"] == "OPEN"
    assert set(items["conflicts"][0]["values"]) == {"$171M", "$186M"}
    # Both poisoned claims sit in quarantine with their reasons.
    by_reason = {q["reason"] for q in items["quarantined"]}
    assert by_reason == {"figure_not_in_evidence", "entity_mismatch"}
    # Quarantined claims are referenced, never quoted: the poisoned
    # text and its figure reach no format of the report.
    for poison in ASTER["poisoned_findings"]:
        assert poison["claim"] not in report["markdown"]
        assert poison["claim"] not in report["html"]
    assert "$198M" not in report["markdown"]
    markdown = report["markdown"]
    assert "## Risks and open items" in markdown
    assert "### Conflicts" in markdown
    assert "### Quarantined claims" in markdown
    assert "figure_not_in_evidence" in markdown
    # A disposition renders when the analyst gives one.
    record = service.store.list_quarantine(case_id)[0]
    service.decide_quarantine(
        case_id, record["quarantine_id"], "Desk Lead", "drop", "not supported"
    )
    refreshed = service.store.get_report(case_id)
    dropped = [
        q
        for q in refreshed["model"]["risks_and_open_items"]["quarantined"]
        if q["claim_id"] == record["claim_id"]
    ]
    assert dropped[0]["status"] == "DROPPED"
    assert dropped[0]["decided_by"] == "Desk Lead"
    assert "DROPPED" in refreshed["markdown"]


def test_open_items_say_so_plainly_when_there_are_none(service):
    # Every gold pack plants at least one record for the section,
    # so exercise the empty branch through the finalize step itself:
    # a draft re-rendered with no conflict records and no other
    # governance records standing against it.
    from deep_diligence.composer import apply_conflicts

    case_id, report = _run_to_draft(service, HELVION)
    case = service.store.get_case(case_id)
    rendered = apply_conflicts(
        service.store, service.objects, case, report, []
    )
    items = rendered["model"]["risks_and_open_items"]
    assert items["empty"] is True
    assert "## Risks and open items" in rendered["markdown"]
    assert "None recorded" in rendered["markdown"]
    assert "None recorded" in rendered["html"]


def test_staleness_proceed_renders_in_open_items(service):
    case_id, _ = _run_to_draft(service, VERDANT)
    if service.store.get_source("Trade coverage") is None:
        service.register_source(
            "Research Desk",
            name="Trade coverage",
            kind="news",
            owner="Trade press publisher",
            usage_basis="public",
            notes="Trade-press coverage gathered by the research desk.",
        )
    service.add_case_evidence(
        case_id,
        "Research Desk",
        title="Verdant Robotics — update: Mule-6 platform deployments",
        source="Trade coverage",
        kind="news",
        doc_date="2026-03-15",
        text=(
            "# Verdant Robotics — update: Mule-6 platform deployments\n\n"
            "Verdant Robotics reports that platform deployment is "
            "accelerating: the Mule-6 platform now anchors the product "
            "roadmap, and platform customers are expanding their "
            "deployments across retail warehouses.\n\n"
            "## Key facts\n"
            "- [F-UPD-201] Verdant Robotics reports the Mule-6 platform "
            "now runs in 300 customer deployments.\n"
        ),
    )
    service.decide_staleness(
        case_id, "Desk Lead", "proceed", "immaterial to the thesis"
    )
    report = service.store.get_report(case_id)
    proceeds = report["model"]["risks_and_open_items"]["staleness_proceeds"]
    assert len(proceeds) == 1
    assert proceeds[0]["analyst"] == "Desk Lead"
    risks = report["markdown"].split("## Risks and open items")[1]
    assert "### Staleness proceeded on" in risks
    assert "immaterial to the thesis" in risks


def test_acknowledged_redaction_flag_renders_in_open_items(service):
    case_id, report = _run_to_draft(service, HELVION)
    claim = report["claims"][0]
    evidence_id = claim["evidence_ids"][0]
    service.redact_evidence(
        case_id, evidence_id, "Desk Lead", "contains personal data"
    )
    flags = service.store.list_redaction_flags(case_id)
    assert flags
    flag = next(f for f in flags if f["claim_id"] == claim["claim_id"])
    service.decide_redaction_flag(
        case_id, flag["flag_id"], "Desk Lead", "acknowledge", "reviewed"
    )
    refreshed = service.store.get_report(case_id)
    rendered = refreshed["model"]["risks_and_open_items"]["redaction_flags"]
    mine = [f for f in rendered if f["flag_id"] == flag["flag_id"]]
    assert mine[0]["status"] == "ACKNOWLEDGED"
    risks = refreshed["markdown"].split("## Risks and open items")[1]
    assert "### Redaction flags" in risks
    assert "ACKNOWLEDGED" in risks


# ---------------------------------------------------------------------
# Three formats, one model
# ---------------------------------------------------------------------


def test_formats_agree_on_claims_and_figures(service):
    case_id, report = _run_to_draft(service, HELVION)
    model = report["model"]
    exported = service.report_export(case_id)
    model_claims = {
        c["claim_id"]: c
        for q in model["questions"]
        for c in q["claims"]
    }
    export_claims = {
        c["claim_id"]: c
        for q in exported["questions"]
        for c in q["claims"]
    }
    assert set(model_claims) == set(export_claims)
    for claim_id, claim in model_claims.items():
        # The claim text and its citations are identical in the
        # export, and the text renders verbatim in the markdown and
        # (escaped) in the HTML.
        assert export_claims[claim_id]["claim"] == claim["claim"]
        assert (
            export_claims[claim_id]["evidence_ids"] == claim["evidence_ids"]
        )
        assert claim["claim"] in report["markdown"]
        assert html_lib.escape(claim["claim"], quote=True) in report["html"]
        for evidence_id in claim["evidence_ids"]:
            assert f"[{evidence_id}]" in report["markdown"]
    # The figures agree across formats: the set of $/% figures in
    # the markdown, in the HTML, and across the model's claims and
    # summary sentences is one set.
    model_text = "\n".join(
        [c["claim"] for c in model_claims.values()]
        + model["executive_summary"]["sentences"]
    )
    assert set(FIGURE.findall(report["markdown"])) == set(
        FIGURE.findall(model_text)
    )
    assert set(FIGURE.findall(report["html"])) == set(
        FIGURE.findall(model_text)
    )
    # References and conflicts ride in the export too, from the
    # same model.
    assert exported["references"] == model["references"]
    assert exported["conflicts"] == model["conflicts"]
    assert exported["coverage"] == model["coverage"]


def test_html_is_self_contained_and_house_styled(service):
    _, report = _run_to_draft(service, HELVION)
    page = report["html"]
    assert page.startswith("<!doctype html>")
    assert "<style>" in page
    assert "#4F46E5" in page and "#0A0A0A" in page
    # No external assets of any kind.
    assert "http://" not in page and "https://" not in page
    assert "<script" not in page and "<link" not in page
    assert "## Executive summary" not in page  # rendered, not raw markdown
    assert "<h2>Executive summary</h2>" in page


def test_report_formats_over_http(client: TestClient, service):
    # The app and this service share one store (same settings), so
    # a case built here is served there.
    case_id, _ = _run_to_draft(service, HELVION)
    page = client.get(f"/api/cases/{case_id}/report/html")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "<h2>Executive summary</h2>" in page.text
    exported = client.get(f"/api/cases/{case_id}/report/export")
    assert exported.status_code == 200
    body = exported.json()
    assert body["subject"] == HELVION["brief"]["subject"]
    assert body["annotations"] == {}
    assert {q["grade"] for q in body["questions"]} <= {
        "STRONG",
        "STANDARD",
        "THIN",
        "UNANSWERED",
    }
    # A case with no report yet has no formats to serve.
    fresh = client.post(
        "/api/cases", json=copy.deepcopy(HELVION["brief"])
    ).json()
    assert client.get(f"/api/cases/{fresh['case_id']}/report/html").status_code == 404
    assert (
        client.get(f"/api/cases/{fresh['case_id']}/report/export").status_code
        == 404
    )


# ---------------------------------------------------------------------
# Analyst section notes
# ---------------------------------------------------------------------


def test_section_notes_annotate_the_export_and_nothing_else(service):
    case_id, report = _run_to_draft(service, HELVION)
    before = service.store.get_report(case_id)
    note = service.add_report_section_note(
        case_id, "Desk Lead", "q1", "Check the FY2024 restatement with IR."
    )
    assert note["note_id"] == "SNOTE-0001"
    service.add_report_section_note(
        case_id, "Desk Lead", "executive_summary", "Reads well."
    )
    after = service.store.get_report(case_id)
    # The report record is untouched: same body, same renderings,
    # same integrity hash inputs.
    assert after["markdown"] == before["markdown"]
    assert after["html"] == before["html"]
    assert after["claims"] == before["claims"]
    assert after["model"] == before["model"]
    # The note text appears in no format's body.
    assert "restatement" not in after["markdown"]
    assert "restatement" not in after["html"]
    # It rides in the export's annotations field, under its section.
    exported = service.report_export(case_id)
    assert [n["note_id"] for n in exported["annotations"]["q1"]] == [
        "SNOTE-0001"
    ]
    assert exported["annotations"]["q1"][0]["text"].startswith("Check the")
    assert "executive_summary" in exported["annotations"]
    # Editing changes the annotation, never the report.
    edited = service.edit_report_section_note(
        case_id, "SNOTE-0001", "Desk Lead", "Confirmed with IR on the call."
    )
    assert edited["edited_at"] and edited["edited_by"] == "Desk Lead"
    assert service.store.get_report(case_id)["markdown"] == before["markdown"]
    exported = service.report_export(case_id)
    assert exported["annotations"]["q1"][0]["text"].startswith("Confirmed")
    events = [
        e["event"]
        for e in service.store.list_audit(case_id)
        if e["event"].startswith("report_section_note")
    ]
    assert events == [
        "report_section_note_added",
        "report_section_note_added",
        "report_section_note_edited",
    ]


def test_section_note_validation(service):
    case_id, _ = _run_to_draft(service, HELVION)
    with pytest.raises(ValueError, match="unknown report section"):
        service.add_report_section_note(case_id, "Desk Lead", "q99", "text")
    with pytest.raises(ValueError, match="needs text"):
        service.add_report_section_note(case_id, "Desk Lead", "q1", "  ")
    with pytest.raises(KeyError):
        service.edit_report_section_note(
            case_id, "SNOTE-9999", "Desk Lead", "text"
        )
    # A case with no report has nothing to annotate.
    fresh = service.create_case(BriefIn(**copy.deepcopy(HELVION["brief"])))
    with pytest.raises(GateError, match="no report"):
        service.add_report_section_note(
            fresh["case_id"], "Desk Lead", "q1", "too early"
        )


def test_section_notes_over_http(client: TestClient, service):
    case_id, _ = _run_to_draft(service, HELVION)
    created = client.post(
        f"/api/cases/{case_id}/report/notes",
        json={
            "analyst": "Desk Lead",
            "section": "risks_and_open_items",
            "text": "The conflict is the story here.",
        },
    )
    assert created.status_code == 201
    note_id = created.json()["note_id"]
    listed = client.get(f"/api/cases/{case_id}/report/notes")
    assert [n["note_id"] for n in listed.json()] == [note_id]
    edited = client.put(
        f"/api/cases/{case_id}/report/notes/{note_id}",
        json={"analyst": "Desk Lead", "text": "Updated view."},
    )
    assert edited.status_code == 200
    assert edited.json()["text"] == "Updated view."
    bad = client.post(
        f"/api/cases/{case_id}/report/notes",
        json={"analyst": "Desk Lead", "section": "nope", "text": "x"},
    )
    assert bad.status_code == 422
    missing = client.put(
        f"/api/cases/{case_id}/report/notes/SNOTE-9999",
        json={"analyst": "Desk Lead", "text": "x"},
    )
    assert missing.status_code == 404
    exported = client.get(f"/api/cases/{case_id}/report/export").json()
    assert (
        exported["annotations"]["risks_and_open_items"][0]["text"]
        == "Updated view."
    )


def test_approved_report_no_longer_reads_as_a_draft(service):
    case_id, report = _run_to_draft(service, HELVION)
    assert "Status: DRAFT" in report["markdown"]
    assert "Status: DRAFT" in report["html"]
    service.decide_report(case_id, "Desk Lead", True)
    final = service.store.get_report(case_id)
    for text in (final["markdown"], final["html"]):
        assert "Status: DRAFT" not in text
        assert "Status: FINAL — approved by Desk Lead on" in text
    assert final["model"]["status"] == "APPROVED"
    assert service.report_export(case_id)["status"] == "APPROVED"
    # The swap is the status line only; the body is untouched.
    assert final["markdown"].split("\n", 3)[3] == report["markdown"].split("\n", 3)[3]


def test_rejected_report_keeps_its_draft_status(service):
    case_id, _ = _run_to_draft(service, HELVION)
    service.decide_report(case_id, "Desk Lead", False, "redo")
    assert "Status: DRAFT" in service.store.get_report(case_id)["markdown"]
