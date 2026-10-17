"""Round 9 — long-case continuity.

A diligence case runs for days; the desk must survive restarts,
pauses, and new evidence. Four behaviours are pinned here:

- An execution killed between waves leaves a case in EXECUTING
  with a checkpoint that says where it stopped; detection records
  the interruption once, and resume — an explicit analyst action —
  finishes the case without re-running the completed workstreams,
  without duplicating a finding ID, and without re-spending the
  case-level budgets the killed run consumed. Files written to the
  case filesystem before the kill read back after the restart.
- Evidence arriving mid-case marks the workstreams it speaks to
  stale and blocks execution until an analyst decides: proceed
  (the decision lands in the report's limits) or re-plan (the
  marked workstreams re-run; the rest keep their findings under
  the evidence-unchanged rule; the superseded draft is preserved).
- A general plan revision is a separate, re-gated move that
  preserves the prior plan and report versions and re-executes in
  full — reuse is a staleness answer, not a default.
- Analyst notes are the desk's memory: they reach the planner's
  context on a re-plan, bounded and labelled, and they are never
  evidence — a note ID cannot be cited.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.harness import (
    MAX_NOTE_CHARS,
    MAX_PLANNER_NOTES,
    notes_section,
    plan_message,
)
from deep_diligence.corpus import DEPENDENCY_CASES
from deep_diligence.models import BriefIn
from deep_diligence.schedule import waves
from deep_diligence.service import CaseService, GateError
from deep_diligence.store import Store
from deep_diligence.tools import build_case_tools

VERDANT_CASE = DEPENDENCY_CASES[0]
GOLD_REFS = set(VERDANT_CASE["gold_facts"])


def _verdant_brief() -> BriefIn:
    import copy

    return BriefIn(**copy.deepcopy(VERDANT_CASE["brief"]))

Q1_FACT_COUNT = 4  # F-VER-101..104
Q3_FACT_COUNT = 3  # F-VER-301..303

NEW_ROAMING_DOC = """# Verdant Robotics — update: Mule-6 platform deployments

Verdant Robotics reports that platform deployment is accelerating:
the Mule-6 platform now anchors the product roadmap, and platform
customers are expanding their deployments across retail warehouses.

## Key facts
- [F-UPD-201] Verdant Robotics reports the Mule-6 platform now runs in 300 customer deployments.
"""


def _audit(service, case_id, event):
    return [
        e for e in service.store.list_audit(case_id) if e["event"] == event
    ]


@pytest.fixture()
def killed_case(service: CaseService) -> str:
    """A Verdant case whose process died between waves.

    Wave 1 (q1 + q3) runs for real through the service's own
    machinery — its findings, its budget ledger entries — and then
    the record is left exactly where a dead process leaves it:
    status EXECUTING, checkpoint written after wave 1 and never
    closed. A file written to the case filesystem before the kill
    must read back after the restart.
    """
    created = service.create_case(_verdant_brief())
    case_id = created["case_id"]
    service.run_planning(case_id)
    plan = service.store.get_plan(case_id)
    service.decide_plan(case_id, "Desk Lead", True)
    case = service.store.get_case(case_id)
    service._backend(case_id).write_text_direct(
        "/execution/partial-notes.md", "wave 1 done; q2 still owed"
    )
    plan_doc = plan.get("plan") or {}
    schedule = waves(plan_doc["workstreams"])
    service.store.update_case(case_id, status="EXECUTING")
    service.store.add_audit(
        case_id,
        "execution_started",
        {
            "plan_version": plan["version"],
            "waves": [[ws["question_id"] for ws in layer] for layer in schedule],
        },
    )
    checkpoint = service._open_checkpoint(case, plan, schedule, {}, 1, None)
    _usage, error, statuses, _interrupted = service._run_wave(
        case, plan_doc, schedule[0], 1
    )
    assert error is None
    service._refresh_checkpoint(case_id, checkpoint, statuses)
    return case_id


# ---------------------------------------------------------------------
# Interrupted cases: detection + resume
# ---------------------------------------------------------------------


def test_interrupted_case_is_detected_once(
    service: CaseService, killed_case: str
):
    found = service.interrupted_cases()
    assert [f["case"]["case_id"] for f in found] == [killed_case]
    entry = found[0]
    assert entry["completed_questions"] == ["q1", "q3"]
    assert entry["pending_questions"] == ["q2"]
    checkpoint = entry["checkpoint"]
    assert checkpoint["complete"] is False
    assert checkpoint["attempt"] == 1
    assert checkpoint["workstreams"]["q1"]["status"] == "completed"
    assert checkpoint["workstreams"]["q2"]["status"] == "pending"
    # The interruption is recorded once, not on every scan.
    service.interrupted_cases()
    detected = _audit(service, killed_case, "execution_interruption_detected")
    assert len(detected) == 1
    assert detected[0]["payload"]["completed_questions"] == ["q1", "q3"]


def test_a_completed_execution_is_not_interrupted(service: CaseService):
    created = service.create_case(_verdant_brief())
    case_id = created["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, "Desk Lead", True)
    service.run_execution(case_id)
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    assert service.interrupted_cases() == []
    checkpoint = service.store.get_checkpoint(case_id)
    assert checkpoint["complete"] is True
    assert checkpoint["outcome"] == "draft_ready"
    assert checkpoint["workstreams"]["q2"]["status"] == "completed"


def test_resume_completes_the_case_without_redoing_finished_work(
    service: CaseService, killed_case: str
):
    before = service.store.list_findings(killed_case)
    assert len(before) == Q1_FACT_COUNT + Q3_FACT_COUNT
    before_ids = {f["finding_id"] for f in before}

    report = service.resume_execution(killed_case, "Desk Lead")

    case = service.store.get_case(killed_case)
    assert case["status"] == "DRAFT_READY"
    assert report["case_id"] == killed_case
    findings = service.store.list_findings(killed_case)
    # Every gold fact is found exactly once: recall after resume is
    # complete and no finding was duplicated on the way.
    assert {f["fact_ref"] for f in findings} == GOLD_REFS
    assert len(findings) == len(GOLD_REFS)
    assert len({f["finding_id"] for f in findings}) == len(findings)
    # The pre-kill findings stand under their original IDs.
    assert before_ids <= {f["finding_id"] for f in findings}
    # The completed workstreams never ran again; only q2 started.
    starts = _audit(service, killed_case, "workstream_started")
    started = [s["payload"]["question_id"] for s in starts]
    assert started.count("q1") == 1
    assert started.count("q3") == 1
    assert started.count("q2") == 1
    # The resume itself is on the record, naming what it skipped.
    resumed = _audit(service, killed_case, "execution_resumed")
    assert len(resumed) == 1
    assert resumed[0]["payload"]["analyst"] == "Desk Lead"
    assert resumed[0]["payload"]["skipped_questions"] == ["q1", "q3"]
    assert resumed[0]["payload"]["budget_carried"] is not None
    checkpoint = service.store.get_checkpoint(killed_case)
    assert checkpoint["complete"] is True
    assert checkpoint["outcome"] == "draft_ready"
    assert service.interrupted_cases() == []


def test_resume_requires_an_interrupted_case(
    service: CaseService,
):
    created = service.create_case(_verdant_brief())
    with pytest.raises(GateError, match="only an interrupted case"):
        service.resume_execution(created["case_id"], "Desk Lead")


def test_budget_carries_over_resume(service: CaseService, killed_case: str):
    """The killed run spent 2 of the brief's 6 sub-agent spawns.
    A resume whose guard re-granted the case caps would report
    fewer; the ledger fold — and the report's budget view — must
    show all 3 spawns the case actually used."""
    brief = service.store.get_case(killed_case)["brief"]
    assert brief["budget"]["max_subagents"] == 6
    service.resume_execution(killed_case, "Desk Lead")
    from deep_diligence.budgets import budget_view

    view = budget_view(
        service.store.get_case(killed_case),
        service.store.list_audit(killed_case),
    )
    assert view["view"]["case"]["consumed"]["subagents"] == 3
    assert view["view"]["exhaustions"] == []
    checkpoint = service.store.get_checkpoint(killed_case)
    assert checkpoint["attempt"] == 1  # the same attempt, continued


def test_case_files_survive_the_kill(
    service: CaseService, settings, killed_case: str
):
    # A fresh service on the same store — the restart.
    restarted = CaseService(settings)
    backend = restarted._backend(killed_case)
    assert (
        backend.read_text_direct("/execution/partial-notes.md")
        == "wave 1 done; q2 still owed"
    )
    restarted.resume_execution(killed_case, "Desk Lead")
    # And the resumed run's own filesystem output is there too.
    assert backend.read_text_direct("/execution/notes.md")


# ---------------------------------------------------------------------
# Evidence arrival + staleness
# ---------------------------------------------------------------------


@pytest.fixture()
def executed_case(service: CaseService) -> str:
    created = service.create_case(_verdant_brief())
    case_id = created["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, "Desk Lead", True)
    service.run_execution(case_id)
    assert service.store.get_case(case_id)["status"] == "DRAFT_READY"
    return case_id


def _add_roaming_doc(service: CaseService, case_id: str) -> dict:
    return service.add_case_evidence(
        case_id,
        "Research Desk",
        title="Verdant Robotics — update: Mule-6 platform deployments",
        kind="product",
        source="Trade coverage",
        doc_date="2026-03-15",
        text=NEW_ROAMING_DOC,
    )


def test_evidence_arrival_marks_only_the_workstreams_it_touches(
    service: CaseService, executed_case: str
):
    result = _add_roaming_doc(service, executed_case)
    ev = result["evidence"]
    assert ev["origin"] == "analyst_upload"
    assert ev["sha256"]
    assert result["stale_questions"] == ["q1"]
    case = service.store.get_case(executed_case)
    assert [m["question_id"] for m in case["stale_marks"]] == ["q1"]
    assert case["stale_marks"][0]["evidence_id"] == ev["evidence_id"]
    added = _audit(service, executed_case, "evidence_added")
    assert added[-1]["payload"]["sha256"] == ev["sha256"]
    assert _audit(service, executed_case, "workstreams_marked_stale")


def test_new_evidence_on_a_final_case_is_refused(
    service: CaseService, executed_case: str
):
    service.decide_report(executed_case, "Desk Lead", True)
    assert service.store.get_case(executed_case)["status"] == "FINAL"
    with pytest.raises(GateError, match="FINAL"):
        _add_roaming_doc(service, executed_case)


def test_stale_marks_block_execution_until_decided(
    service: CaseService, executed_case: str
):
    # Back to an executable state with marks open: reject the
    # report (REPORT_REJECTED is executable) and try to run.
    service.decide_report(executed_case, "Desk Lead", False, "redo q1")
    _add_roaming_doc(service, executed_case)
    with pytest.raises(GateError, match="stale"):
        service.run_execution(executed_case)


def test_proceed_records_the_staleness_in_the_report_limits(
    service: CaseService, executed_case: str
):
    _add_roaming_doc(service, executed_case)
    case = service.decide_staleness(
        executed_case, "Desk Lead", "proceed", "immaterial to the thesis"
    )
    assert case["stale_marks"] == []
    assert case["staleness_decisions"][0]["action"] == "proceed"
    assert case["staleness_decisions"][0]["questions"] == ["q1"]
    report = service.store.get_report(executed_case)
    assert "marked q1 stale" in report["markdown"]
    assert "decision to proceed" in report["markdown"]
    assert "immaterial to the thesis" in report["markdown"]
    assert _audit(service, executed_case, "staleness_proceeded")


def test_replan_reruns_only_the_stale_workstream_and_reuses_the_rest(
    service: CaseService, settings, executed_case: str
):
    before = service.store.list_findings(executed_case)
    before_by_q = {}
    for f in before:
        before_by_q.setdefault(f["question_id"], set()).add(f["finding_id"])
    _add_roaming_doc(service, executed_case)
    case = service.decide_staleness(executed_case, "Desk Lead", "replan")
    assert case["status"] == "PLAN_REJECTED"
    assert case["reuse_findings"] is True
    assert case["must_rerun"] == ["q1"]

    # The prior plan and the superseded draft both survive.
    assert service.store.get_plan(executed_case, version=1) is not None
    snapshots = service.store.list_report_snapshots(executed_case)
    assert len(snapshots) == 1
    assert "Mule-6" in snapshots[0]["report"]["markdown"]

    service.run_planning(executed_case)
    assert service.store.get_plan(executed_case)["version"] == 2
    service.decide_plan(executed_case, "Desk Lead", True)
    service.run_execution(executed_case)
    assert service.store.get_case(executed_case)["status"] == "DRAFT_READY"

    findings = service.store.list_findings(executed_case)
    by_q = {}
    for f in findings:
        by_q.setdefault(f["question_id"], set()).add(f["finding_id"])
    # Untouched workstreams kept their findings, IDs and all.
    assert by_q["q2"] == before_by_q["q2"]
    assert by_q["q3"] == before_by_q["q3"]
    # q1 re-ran: it found the same facts again (the record dedups
    # by fact, so the IDs stand) plus the new document's fact.
    assert by_q["q1"] >= before_by_q["q1"]
    assert {f["fact_ref"] for f in findings if f["question_id"] == "q1"} == {
        "F-VER-101", "F-VER-102", "F-VER-103", "F-VER-104", "F-UPD-201",
    }
    starts = _audit(service, executed_case, "workstream_started")
    started = [s["payload"]["question_id"] for s in starts]
    assert started.count("q1") == 2  # original run + the re-plan run
    assert started.count("q2") == 1
    assert started.count("q3") == 1
    reused = _audit(service, executed_case, "findings_reused")
    assert reused and reused[-1]["payload"]["questions"] == ["q2", "q3"]
    # The new draft carries the decision in its limits.
    report = service.store.get_report(executed_case)
    assert "re-planned" in report["markdown"]
    # A second service on the same store sees the same record —
    # continuity is in the store, not in process memory.
    assert Store(settings).get_case(executed_case)["status"] == "DRAFT_READY"


def test_reuse_falls_back_when_cited_evidence_has_drifted(
    service: CaseService, executed_case: str
):
    """The reuse rule's second half: a finding is reused only while
    the evidence it cites is unchanged. Replace the bytes behind a
    q2 document and q2 re-runs too, even though the new evidence
    never touched it."""
    q2_doc = next(
        ev
        for ev in service.store.list_evidence(executed_case)
        if ev["doc_slug"] == "verdant-market-analysis"
    )
    _add_roaming_doc(service, executed_case)
    service.decide_staleness(executed_case, "Desk Lead", "replan")
    service.objects.put(q2_doc["object_key"], b"tampered bytes")
    service.run_planning(executed_case)
    service.decide_plan(executed_case, "Desk Lead", True)
    service.run_execution(executed_case)
    starts = _audit(service, executed_case, "workstream_started")
    started = [s["payload"]["question_id"] for s in starts]
    assert started.count("q2") == 2  # drifted evidence: re-run
    assert started.count("q3") == 1  # unchanged evidence: reused


def test_general_revision_replans_without_reuse(
    service: CaseService, executed_case: str
):
    case = service.replan_case(executed_case, "Desk Lead", "scope shifted")
    assert case["status"] == "PLAN_REJECTED"
    assert case["reuse_findings"] is False
    assert len(service.store.list_report_snapshots(executed_case)) == 1
    service.run_planning(executed_case)
    assert service.store.get_plan(executed_case)["version"] == 2
    service.decide_plan(executed_case, "Desk Lead", True)
    service.run_execution(executed_case)
    starts = _audit(service, executed_case, "workstream_started")
    started = [s["payload"]["question_id"] for s in starts]
    assert started.count("q1") == 2
    assert started.count("q2") == 2
    assert started.count("q3") == 2
    assert not _audit(service, executed_case, "findings_reused")
    assert _audit(service, executed_case, "plan_revision_requested")


# ---------------------------------------------------------------------
# Analyst notes
# ---------------------------------------------------------------------


def test_notes_are_stored_and_reach_the_replan_context(
    service: CaseService, executed_case: str, monkeypatch
):
    note = service.add_case_note(
        executed_case,
        "Desk Lead",
        "Watch the fleet attach rate — the client cares most about q2.",
    )
    assert note["note_id"] == "NOTE-0001"
    assert service.store.list_notes(executed_case)[0]["text"].startswith(
        "Watch the fleet attach rate"
    )
    assert _audit(service, executed_case, "case_note_added")

    # Capture the instruction the service actually hands the
    # planner on the re-plan, by wrapping the service's own runner.
    captured: dict = {}
    original = CaseService._run_agent

    def capturing(self, case, message, phase):
        captured[phase] = message
        return original(self, case, message, phase)

    monkeypatch.setattr(CaseService, "_run_agent", capturing)
    service.replan_case(executed_case, "Desk Lead")
    service.run_planning(executed_case)
    message = captured["planning"]
    assert "ANALYST NOTES" in message
    assert "Watch the fleet attach rate" in message
    assert "never evidence" in message


def test_notes_section_is_bounded_and_labelled():
    notes = [
        {
            "note_id": f"NOTE-{i:04d}",
            "analyst": "Desk Lead",
            "created_at": "2026-03-01T00:00:00+00:00",
            "text": "x" * (MAX_NOTE_CHARS + 200) if i == 12 else f"note {i}",
        }
        for i in range(1, 13)
    ]
    section = notes_section(notes)
    assert "never evidence" in section
    # The two oldest notes fall outside the bound; the newest ten
    # are carried. (Headers are bracketed — check with the bracket,
    # since NOTE-0001 is a prefix of NOTE-0010.)
    assert "[NOTE-0001 " not in section
    assert "[NOTE-0002 " not in section
    assert "[NOTE-0003 " in section
    assert "[NOTE-0012 " in section
    assert "..." in section  # the long note was truncated
    assert len(section) < (MAX_PLANNER_NOTES * (MAX_NOTE_CHARS + 80)) + 400
    # A brief with no notes plans exactly as before.
    case = {"brief": {"subject": "Verdant Robotics"}}
    assert "ANALYST NOTES" not in plan_message(case)
    assert "ANALYST NOTES" in plan_message(case, notes)


def test_a_note_can_never_be_cited(service: CaseService, executed_case: str):
    note = service.add_case_note(executed_case, "Desk Lead", "context only")
    store = service.store
    case = store.get_case(executed_case)
    tools = build_case_tools(store, service.objects, case)
    record_finding = next(t for t in tools if t.name == "record_finding")
    # Citation validation resolves IDs against the case's evidence
    # table; a note lives in its own table, so its ID is refused
    # exactly like any other non-evidence ID.
    result = record_finding.func(
        "q1",
        "A claim that tries to cite the analyst note.",
        [note["note_id"]],
        "F-FAKE-001",
    )
    assert result.startswith("ERROR")
    assert not any(
        f["fact_ref"] == "F-FAKE-001" for f in store.list_findings(executed_case)
    )


# ---------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------


def test_continuity_api(service: CaseService, killed_case: str):
    client = TestClient(create_app(service.settings, service))

    interrupted = client.get("/api/cases/interrupted")
    assert interrupted.status_code == 200
    rows = interrupted.json()
    assert [r["case_id"] for r in rows] == [killed_case]
    assert rows[0]["pending_questions"] == ["q2"]

    checkpoint = client.get(f"/api/cases/{killed_case}/checkpoint")
    assert checkpoint.status_code == 200
    assert checkpoint.json()["complete"] is False

    note = client.post(
        f"/api/cases/{killed_case}/notes",
        json={"analyst": "Desk Lead", "text": "Resume it when ready."},
    )
    assert note.status_code == 201
    assert note.json()["note_id"] == "NOTE-0001"
    notes = client.get(f"/api/cases/{killed_case}/notes")
    assert len(notes.json()) == 1

    resumed = client.post(
        f"/api/cases/{killed_case}/resume", json={"analyst": "Desk Lead"}
    )
    assert resumed.status_code == 200
    assert client.get(f"/api/cases/{killed_case}").json()["status"] == "DRAFT_READY"
    assert client.get("/api/cases/interrupted").json() == []

    # Resume again -> 409; the case is no longer interrupted.
    again = client.post(
        f"/api/cases/{killed_case}/resume", json={"analyst": "Desk Lead"}
    )
    assert again.status_code == 409

    added = client.post(
        f"/api/cases/{killed_case}/evidence",
        json={
            "analyst": "Research Desk",
            "title": "Verdant Robotics — update: Mule-6 platform deployments",
            "kind": "product",
            "source": "Trade coverage",
            "doc_date": "2026-03-15",
            "text": NEW_ROAMING_DOC,
        },
    )
    assert added.status_code == 201
    assert added.json()["stale_questions"] == ["q1"]
    case_view = client.get(f"/api/cases/{killed_case}").json()
    assert case_view["stale_marks"][0]["question_id"] == "q1"

    decided = client.post(
        f"/api/cases/{killed_case}/staleness/decision",
        json={"analyst": "Desk Lead", "action": "proceed", "note": "noted"},
    )
    assert decided.status_code == 200
    assert decided.json()["stale_marks"] == []
    assert decided.json()["staleness_decisions"][0]["action"] == "proceed"

    replanned = client.post(
        f"/api/cases/{killed_case}/replan",
        json={"analyst": "Desk Lead", "note": "one more pass"},
    )
    assert replanned.status_code == 200
    assert replanned.json()["status"] == "PLAN_REJECTED"
    versions = client.get(f"/api/cases/{killed_case}/report/versions")
    assert versions.status_code == 200
    assert len(versions.json()) == 1
    assert "Mule-6" in versions.json()[0]["markdown"]
