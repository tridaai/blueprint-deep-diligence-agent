"""Round 4: conflict detection over supported findings.

Where two supported findings assert different values for the same
(entity, metric, period) key, a deterministic pass creates a stored
conflict record holding both sides — claims, evidence, values
verbatim, and each side's source class — and the report gains a
Conflicts section generated from the records. The analyst's only
move is to mark a conflict NOTED with a note; the record is never
deleted or resolved away. Facts from different periods are
different keys and never conflict; the same value from two sources
is corroboration, not a conflict.
"""

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.conflicts import detect_conflicts, extract_assertions
from deep_diligence.corpus import GOLD_CASES
from deep_diligence.models import BriefIn
from deep_diligence.service import GateError

HELVION = GOLD_CASES[0]["brief"]
COPPERLINE = GOLD_CASES[1]["brief"]
SELENE = GOLD_CASES[2]["brief"]


def _run_case(service, brief) -> str:
    case = service.create_case(BriefIn(**brief))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    return case_id


def _evidence_id(service, case_id: str, slug: str) -> str:
    return next(
        ev["evidence_id"]
        for ev in service.store.list_evidence(case_id)
        if ev["doc_slug"] == slug
    )


def _inject(service, case_id: str, question_id: str, claim: str, slug: str):
    return service.store.add_finding_auto_id(
        case_id,
        {
            "question_id": question_id,
            "claim": claim,
            "evidence_ids": [_evidence_id(service, case_id, slug)],
            "fact_ref": None,
        },
    )


# -- the deterministic extraction --------------------------------------


def test_extraction_pairs_each_margin_with_its_own_period():
    assertions = extract_assertions(
        "Gross margin was 31% in FY2025, up from 27% in FY2024."
    )
    assert {(a["metric"], a["period"], a["value_text"]) for a in assertions} == {
        ("gross margin", "2025", "31%"),
        ("gross margin", "2024", "27%"),
    }


def test_extraction_types_figures_and_ignores_codes_and_years():
    # The revenue level is the money figure, not the growth percent.
    assertions = extract_assertions(
        "Revenue grew 18% year over year to $412M in FY2025."
    )
    by_metric = {a["metric"]: a["value_text"] for a in assertions}
    assert by_metric["revenue"] == "$412M"
    assert by_metric["revenue growth"] == "18%"
    # Product codes and bare years are not figures.
    assert extract_assertions(
        "The AF-40 is Aster's flagship induction furnace product platform."
    ) == []
    assert extract_assertions(
        "Helvion's Poland factory expands its European market coverage from 2027."
    ) == []
    # One claim, two money metrics: each figure goes to its own.
    assertions = extract_assertions(
        "The company carried $41M of debt against $9M of cash at year end."
    )
    assert {(a["metric"], a["value_text"]) for a in assertions} == {
        ("debt", "$41M"),
        ("cash", "$9M"),
    }


def test_extraction_normalises_values_so_formats_agree():
    a = extract_assertions("Copperline employed 1,150 people at the end of FY2025.")
    b = extract_assertions("Copperline employed 1150 people at the end of FY2025.")
    assert a[0]["value"] == b[0]["value"]


# -- detection over a full run ------------------------------------------


def test_helvion_revenue_contradiction_becomes_one_conflict(service):
    case_id = _run_case(service, HELVION)
    conflicts = service.store.list_conflicts(case_id)
    assert len(conflicts) == 1
    record = conflicts[0]
    assert record["key"]["metric"] == "revenue"
    assert record["key"]["period"] == "2025"
    assert record["key"]["entity"] == "Helvion Grid Systems"
    assert record["status"] == "OPEN"
    sides = {s["value_text"]: s for s in record["sides"]}
    assert set(sides) == {"$412M", "$421M"}
    filing_side = sides["$412M"]["claims"][0]
    news_side = sides["$421M"]["claims"][0]
    assert filing_side["source_class"] == "primary"
    assert news_side["source_class"] == "secondary"
    # The news article is the later-dated source for the same period.
    assert news_side["more_recent"] is True
    assert filing_side["more_recent"] is False
    assert filing_side["evidence_ids"] and news_side["evidence_ids"]
    # The audit trail carries the detection.
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert "conflicts_detected" in events


def test_conflict_renders_in_the_report_with_both_sides(service):
    case_id = _run_case(service, HELVION)
    report = service.store.get_report(case_id)
    markdown = report["markdown"]
    assert "## Conflicts" in markdown
    section = markdown.split("## Conflicts")[1].split("## References")[0]
    assert "$412M" in section and "$421M" in section
    assert "primary source" in section and "secondary source" in section
    assert "more recent" in section
    assert "takes no position" in section
    assert len(report["conflicts"]) == 1
    assert report["conflicts"][0]["values"] == ["$412M", "$421M"]


def test_no_false_conflicts_from_period_distinct_facts(service):
    # Copperline's EBITDA margin is 4.1% in FY2025 and 5.0% in
    # FY2024 — different keys, stated side by side in one claim —
    # and its revenue is stated once. The only conflict in the case
    # is the planted headcount disagreement.
    case_id = _run_case(service, COPPERLINE)
    conflicts = service.store.list_conflicts(case_id)
    assert len(conflicts) == 1
    assert conflicts[0]["key"]["metric"] == "headcount"
    assert {s["value_text"] for s in conflicts[0]["sides"]} == {"1,150", "1,300"}


def test_same_value_from_two_sources_is_corroboration_not_conflict(service):
    case = service.create_case(BriefIn(**HELVION))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    # A second, differently worded claim restating the filing's
    # revenue figure, resting on the same filing: corroboration.
    _inject(
        service,
        case_id,
        "q1",
        "FY2025 revenue was $412M for Helvion Grid Systems.",
        "helvion-annual-report-fy2025",
    )
    service.run_execution(case_id)
    conflicts = service.store.list_conflicts(case_id)
    assert len(conflicts) == 1
    sides = {s["value_text"]: s for s in conflicts[0]["sides"]}
    assert len(sides["$412M"]["claims"]) == 2
    assert len(sides["$421M"]["claims"]) == 1


def test_period_distinct_injected_fact_creates_no_conflict(service):
    case = service.create_case(BriefIn(**HELVION))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    _inject(
        service,
        case_id,
        "q1",
        "Gross margin was 27% in FY2024.",
        "helvion-annual-report-fy2025",
    )
    service.run_execution(case_id)
    conflicts = service.store.list_conflicts(case_id)
    assert [c["key"]["metric"] for c in conflicts] == ["revenue"]


def test_selene_result_time_conflict_has_no_period(service):
    case_id = _run_case(service, SELENE)
    conflicts = service.store.list_conflicts(case_id)
    assert len(conflicts) == 1
    record = conflicts[0]
    assert record["key"]["metric"] == "result time"
    assert record["key"]["period"] is None
    assert {s["value_text"] for s in record["sides"]} == {"25", "41"}
    sides = {s["value_text"]: s for s in record["sides"]}
    # The independent field evaluation is the later-dated source.
    assert sides["41"]["claims"][0]["more_recent"] is True
    assert sides["25"]["claims"][0]["more_recent"] is False


def test_detect_conflicts_on_empty_claims_is_empty(service, case):
    assert detect_conflicts(service.store, case, []) == []


# -- the analyst's one move ----------------------------------------------


def test_analyst_marks_conflict_noted_and_it_stays(service):
    case_id = _run_case(service, HELVION)
    record = service.store.list_conflicts(case_id)[0]
    noted = service.decide_conflict(
        case_id,
        record["conflict_id"],
        analyst="Priya",
        note="Desk uses the audited filing for the pack; flagging the "
        "press figure for the data room Q&A.",
    )
    assert noted["status"] == "NOTED"
    assert noted["noted_by"] == "Priya"
    # Still listed, and the note renders into the report permanently.
    assert len(service.store.list_conflicts(case_id)) == 1
    markdown = service.store.get_report(case_id)["markdown"]
    assert "marked NOTED by Priya" in markdown
    assert "data room Q&A" in markdown
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert "conflict_noted" in events
    # There is no second move: an already-noted conflict refuses.
    with pytest.raises(GateError):
        service.decide_conflict(
            case_id, record["conflict_id"], analyst="Priya", note="again"
        )


def test_noting_requires_a_note_and_a_report_at_the_gate(service, case):
    with pytest.raises(GateError):
        service.decide_conflict(case["case_id"], "CNF-0001", analyst="P", note="x")
    case_id = _run_case(service, HELVION)
    record = service.store.list_conflicts(case_id)[0]
    with pytest.raises(ValueError):
        service.decide_conflict(
            case_id, record["conflict_id"], analyst="P", note="  "
        )
    with pytest.raises(KeyError):
        service.decide_conflict(case_id, "CNF-9999", analyst="P", note="x")


# -- API -------------------------------------------------------------------


def test_conflicts_api_lists_notes_and_refuses_a_second_move(settings, service):
    case_id = _run_case(service, HELVION)
    client = TestClient(create_app(settings))
    listed = client.get(f"/api/cases/{case_id}/conflicts")
    assert listed.status_code == 200
    records = listed.json()
    assert len(records) == 1
    conflict_id = records[0]["conflict_id"]
    decided = client.post(
        f"/api/cases/{case_id}/conflicts/{conflict_id}/decision",
        json={"analyst": "Priya", "note": "Seen; using the filing figure."},
    )
    assert decided.status_code == 200
    assert decided.json()["conflict"]["status"] == "NOTED"
    again = client.post(
        f"/api/cases/{case_id}/conflicts/{conflict_id}/decision",
        json={"analyst": "Priya", "note": "Second note"},
    )
    assert again.status_code == 409
    missing = client.post(
        f"/api/cases/{case_id}/conflicts/CNF-9999/decision",
        json={"analyst": "Priya", "note": "x"},
    )
    assert missing.status_code == 404
    no_note = client.post(
        f"/api/cases/{case_id}/conflicts/{conflict_id}/decision",
        json={"analyst": "Priya", "note": ""},
    )
    assert no_note.status_code == 422
    # No delete or resolve route exists for a conflict.
    assert client.request(
        "DELETE", f"/api/cases/{case_id}/conflicts/{conflict_id}"
    ).status_code in (404, 405)
    assert len(client.get(f"/api/cases/{case_id}/conflicts").json()) == 1
