"""Round 12: the analyst workbench — queue, timeline, search, views.

The queue is derived from the case records, never stored, so the
tests plant cases in each state the queue reads and check the
derivation itself: which group each case lands in, how many
records stand behind an item, who has waited longest, and how old
the wait is — against timestamps planted on the records, read at
a fixed "now", so ages are exact, not approximate.

Assignment and priority are triage: recorded acts that order and
filter the queue and gate nothing. The last test pins the hard
rule — there is no bulk decision route anywhere in the API, and
every decision route is addressed to one case.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.config import load_settings
from deep_diligence.corpus import GOLD_CASES, INTEGRITY_CASES
from deep_diligence.governance import parse_ts
from deep_diligence.models import BriefIn
from deep_diligence.service import CaseService
from deep_diligence.workbench import ACTS, build_queue

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-10-20T12:00:00+00:00"


def _ts(days: float = 0, hours: float = 0) -> str:
    base = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    from datetime import timedelta

    return (base - timedelta(days=days, hours=hours)).isoformat(
        timespec="seconds"
    )


def _full_run(service: CaseService, brief: dict) -> str:
    case = service.create_case(BriefIn(**brief))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    return case_id


def _set_audit_ts(store, case_id: str, event: str, ts: str) -> None:
    store._execute(
        "UPDATE audit SET ts = ? WHERE case_id = ? AND event = ?",
        (ts, case_id, event),
    )


def _set_case_updated(store, case_id: str, ts: str) -> None:
    case = store.get_case(case_id)
    case["updated_at"] = ts
    store._execute(
        "UPDATE cases SET payload = ?, updated_at = ? WHERE case_id = ?",
        (json.dumps(case), ts, case_id),
    )


def _set_composed_at(store, case_id: str, ts: str) -> None:
    report = store.get_report(case_id)
    report["composed_at"] = ts
    store.save_report(case_id, report)


def _set_conflict_age(store, case_id: str, ts: str) -> None:
    for record in store.list_conflicts(case_id):
        record["created_at"] = ts
        store.update_conflict(case_id, record)


@pytest.fixture(scope="module")
def desk(tmp_path_factory):
    """One desk, planted: a case in every state the queue reads,
    with the waits backdated on the records themselves."""
    tmp = tmp_path_factory.mktemp("workbench")
    settings = load_settings(
        sqlite_path=str(tmp / "diligence.db"),
        object_dir=str(tmp / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
    )
    service = CaseService(settings)
    store = service.store
    planted: dict[str, str] = {}

    # A plan waiting at its gate for four days.
    plan_case = service.create_case(BriefIn(**GOLD_CASES[1]["brief"]))
    planted["plan"] = plan_case["case_id"]
    service.run_planning(planted["plan"])
    _set_audit_ts(store, planted["plan"], "plan_generated", _ts(days=4))

    # A finished draft (Helvion) with its planted conflict open.
    planted["draft"] = _full_run(service, GOLD_CASES[0]["brief"])
    _set_composed_at(store, planted["draft"], _ts(days=2))
    _set_conflict_age(store, planted["draft"], _ts(days=2))

    # The integrity case: two poisoned claims quarantined (five
    # days waiting), its conflict open, draft composed a day ago.
    aster = INTEGRITY_CASES[0]
    case = service.create_case(BriefIn(**aster["brief"]))
    planted["aster"] = case["case_id"]
    service.run_planning(planted["aster"])
    service.decide_plan(planted["aster"], analyst="Tester", approve=True)
    by_slug = {
        ev["doc_slug"]: ev["evidence_id"]
        for ev in store.list_evidence(planted["aster"])
    }
    for poison in aster["poisoned_findings"]:
        store.add_finding_auto_id(
            planted["aster"],
            {
                "question_id": poison["question_id"],
                "claim": poison["claim"],
                "evidence_ids": [by_slug[poison["doc_slug"]]],
                "fact_ref": None,
            },
        )
    service.run_execution(planted["aster"])
    for record in store.list_quarantine(planted["aster"]):
        record["created_at"] = _ts(days=5)
        store.update_quarantine(planted["aster"], record)
    _set_composed_at(store, planted["aster"], _ts(days=1))
    _set_conflict_age(store, planted["aster"], _ts(days=1))

    # A draft (Selene) with an open redaction flag: composed
    # three days ago, the flag raised twelve hours ago.
    planted["flag"] = _full_run(service, GOLD_CASES[2]["brief"])
    cited = store.list_ledger(planted["flag"])[0]["citations"][0]
    outcome = service.redact_evidence(
        planted["flag"], cited["evidence_id"], "Tester",
        "Names a private individual",
    )
    planted["flag_records"] = len(outcome["flags"])
    assert planted["flag_records"] >= 1
    for flag in store.list_redaction_flags(planted["flag"]):
        flag["created_at"] = _ts(hours=12)
        store.update_redaction_flag(planted["flag"], flag)
    _set_composed_at(store, planted["flag"], _ts(days=3))
    _set_conflict_age(store, planted["flag"], _ts(days=3))

    # A case a dead process left in EXECUTING six days ago.
    stranded = service.create_case(BriefIn(**GOLD_CASES[1]["brief"]))
    planted["interrupted"] = stranded["case_id"]
    service.run_planning(planted["interrupted"])
    service.decide_plan(planted["interrupted"], analyst="Tester", approve=True)
    store.update_case(planted["interrupted"], status="EXECUTING")
    _set_case_updated(store, planted["interrupted"], _ts(days=6))

    # A case carried all the way to FINAL: it owes nobody anything.
    planted["final"] = _full_run(service, GOLD_CASES[2]["brief"])
    service.decide_report(planted["final"], analyst="Tester", approve=True)
    assert store.get_case(planted["final"])["status"] == "FINAL"

    # Triage on the planted desk: one case owned, one urgent.
    service.assign_case(planted["draft"], "Desk Lead", "Priya Nair")
    service.set_priority(planted["flag"], "Desk Lead", "urgent")
    return service, planted


def _queue(service: CaseService) -> dict:
    return build_queue(
        service.store, service.interrupted_cases(), now=parse_ts(NOW)
    )


def _group(queue: dict, act: str) -> dict:
    return next(g for g in queue["groups"] if g["act"] == act)


def _items(queue: dict, act: str) -> list[dict]:
    return _group(queue, act)["items"]


# -- the queue ---------------------------------------------------------------


def test_queue_groups_cover_every_planted_state(desk):
    service, planted = desk
    queue = _queue(service)
    assert [g["act"] for g in queue["groups"]] == [a for a, _ in ACTS]
    assert _group(queue, "plan_decision")["count"] == 1
    assert _group(queue, "report_decision")["count"] == 3
    assert _group(queue, "quarantine_decision")["count"] == 1
    assert _group(queue, "conflict_note")["count"] == 3
    assert _group(queue, "redaction_flag_decision")["count"] == 1
    assert _group(queue, "interrupted_resume")["count"] == 1
    assert _group(queue, "budget_amendment")["count"] == 0
    assert queue["total"] == 10
    landed = {
        item["case_id"] for g in queue["groups"] for item in g["items"]
    }
    # The FINAL case appears in no group: it owes nobody anything.
    assert planted["final"] not in landed
    assert landed == {
        planted[k]
        for k in ("plan", "draft", "aster", "flag", "interrupted")
    }


def test_queue_items_name_their_records(desk):
    service, planted = desk
    queue = _queue(service)
    (aster_item,) = _items(queue, "quarantine_decision")
    assert aster_item["case_id"] == planted["aster"]
    assert aster_item["records"] == 2
    assert len(aster_item["record_ids"]) == 2
    (plan_item,) = _items(queue, "plan_decision")
    assert "plan v1" in plan_item["detail"]
    (flag_item,) = _items(queue, "redaction_flag_decision")
    assert flag_item["records"] == planted["flag_records"]
    (stranded,) = _items(queue, "interrupted_resume")
    assert stranded["case_id"] == planted["interrupted"]
    assert "still owed" in stranded["detail"]


def test_queue_orders_oldest_first_with_exact_ages(desk):
    service, planted = desk
    queue = _queue(service)
    drafts = _items(queue, "report_decision")
    assert [i["case_id"] for i in drafts] == [
        planted["flag"], planted["draft"], planted["aster"],
    ]
    assert [i["age_seconds"] for i in drafts] == [259200, 172800, 86400]
    group = _group(queue, "report_decision")
    assert group["oldest_waiting_since"] == _ts(days=3)
    assert group["oldest_age_seconds"] == 259200
    assert _items(queue, "plan_decision")[0]["age_seconds"] == 345600
    assert _items(queue, "quarantine_decision")[0]["age_seconds"] == 432000
    assert _items(queue, "redaction_flag_decision")[0]["age_seconds"] == 43200
    assert _items(queue, "interrupted_resume")[0]["age_seconds"] == 518400
    conflicts = _items(queue, "conflict_note")
    assert [i["case_id"] for i in conflicts] == [
        planted["flag"], planted["draft"], planted["aster"],
    ]


def test_queue_filters_by_assignee_priority_and_act(desk):
    service, planted = desk
    store = service.store
    by_owner = build_queue(
        store, [], filters={"assignee": "Priya Nair"}, now=parse_ts(NOW)
    )
    owned = {
        (i["case_id"], i["act"])
        for g in by_owner["groups"] for i in g["items"]
    }
    assert owned == {
        (planted["draft"], "report_decision"),
        (planted["draft"], "conflict_note"),
    }
    unassigned = build_queue(
        store, [], filters={"assignee": "unassigned"}, now=parse_ts(NOW)
    )
    assert all(
        i["assignee"] is None
        for g in unassigned["groups"] for i in g["items"]
    )
    urgent = build_queue(
        store, [], filters={"priority": "urgent"}, now=parse_ts(NOW)
    )
    urgent_cases = {
        i["case_id"] for g in urgent["groups"] for i in g["items"]
    }
    assert urgent_cases == {planted["flag"]}
    only_plans = build_queue(
        store, [], filters={"act": "plan_decision"}, now=parse_ts(NOW)
    )
    assert only_plans["total"] == 1
    assert _group(only_plans, "plan_decision")["count"] == 1
    assert _group(only_plans, "report_decision")["count"] == 0
    with pytest.raises(ValueError, match="unknown act"):
        build_queue(store, [], filters={"act": "everything"})
    with pytest.raises(ValueError, match="unknown workbench filter"):
        build_queue(store, [], filters={"owner": "Priya"})


def test_budget_exhausted_case_queues_for_amendment(service, gold_brief):
    brief = BriefIn(
        **{
            **gold_brief,
            "budget": {**gold_brief["budget"], "execution": {"max_steps": 6}},
        }
    )
    case = service.create_case(brief)
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    assert service.store.get_case(case_id)["status"] == "BUDGET_EXHAUSTED"
    queue = service.workbench_queue()
    (item,) = _items(queue, "budget_amendment")
    assert item["case_id"] == case_id
    assert "amended" in item["detail"]
    # The partial draft's decision is represented by the
    # amendment item — the case does not also queue as a plain
    # draft decision.
    assert all(
        i["case_id"] != case_id for i in _items(queue, "report_decision")
    )


# -- assignment + priority -----------------------------------------------------


def test_assignment_and_priority_are_recorded_acts(service, case):
    case_id = case["case_id"]
    updated = service.assign_case(case_id, "Desk Lead", "Priya Nair")
    assert updated["assignee"] == "Priya Nair"
    updated = service.assign_case(case_id, "Priya Nair", "Tom Okafor")
    assert updated["assignee"] == "Tom Okafor"
    events = [
        e for e in service.store.list_audit(case_id)
        if e["event"] == "case_assigned"
    ]
    assert [e["payload"]["previous"] for e in events] == [None, "Priya Nair"]
    assert events[-1]["payload"]["analyst"] == "Priya Nair"
    cleared = service.assign_case(case_id, "Desk Lead", "")
    assert cleared["assignee"] is None

    updated = service.set_priority(case_id, "Desk Lead", "urgent")
    assert updated["priority"] == "urgent"
    (event,) = [
        e for e in service.store.list_audit(case_id)
        if e["event"] == "case_priority_set"
    ]
    assert event["payload"] == {
        "analyst": "Desk Lead", "priority": "urgent", "previous": "normal",
    }
    with pytest.raises(ValueError, match="unknown priority"):
        service.set_priority(case_id, "Desk Lead", "whenever")


# -- the timeline ---------------------------------------------------------------


def test_timeline_merges_decisions_and_notes_in_order(service, case):
    case_id = case["case_id"]
    service.add_case_note(case_id, "Priya Nair", "Client wants the headcount discrepancy explained, not buried.")
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Desk Lead", approve=True, note="Scope is right")
    service.run_execution(case_id)
    service.add_report_section_note(case_id, "Tester", "q1", "Cross-check this against the filing.")
    timeline = service.case_timeline(case_id)
    events = [t["event"] for t in timeline]
    assert events[0] == "case_created"
    assert events.index("plan_generated") < events.index("plan_approved")
    assert events.index("plan_approved") < events.index("report_composed")
    stamps = [t["ts"] for t in timeline]
    assert stamps == sorted(stamps)
    (note_entry,) = [t for t in timeline if t["kind"] == "note"]
    assert note_entry["label"] == "Analyst note"
    assert note_entry["detail"].startswith("Client wants the headcount")
    assert note_entry["actor"] == "Priya Nair"
    (section_entry,) = [t for t in timeline if t["kind"] == "section_note"]
    assert "[q1]" in section_entry["detail"]
    (decision,) = [t for t in timeline if t["event"] == "plan_approved"]
    assert decision["label"] == "Plan approved"
    assert decision["actor"] == "Desk Lead"
    assert "Scope is right" in decision["detail"]


# -- saved views ------------------------------------------------------------------


def test_saved_views_round_trip(service):
    view = service.save_view(
        "Priya Nair", "My urgent plans",
        {"act": "plan_decision", "priority": "urgent"},
    )
    assert view["view_id"].startswith("VIEW-")
    assert view["filters"] == {"act": "plan_decision", "priority": "urgent"}
    assert service.list_views("Priya Nair") == [view]
    assert service.list_views("Tom Okafor") == []
    # Saving under the same name replaces the filters; it does
    # not stack a second view.
    replaced = service.save_view(
        "Priya Nair", "My urgent plans", {"assignee": "unassigned"}
    )
    assert replaced["view_id"] == view["view_id"]
    assert replaced["filters"] == {"assignee": "unassigned"}
    assert len(service.list_views()) == 1
    # The view's filters drive the queue unchanged.
    queue = service.workbench_queue(filters=replaced["filters"])
    assert queue["filters"] == {"assignee": "unassigned"}
    deleted = service.delete_view("Priya Nair", view["view_id"])
    assert deleted["view_id"] == view["view_id"]
    assert service.list_views() == []
    with pytest.raises(KeyError):
        service.delete_view("Priya Nair", view["view_id"])
    other = service.save_view("Tom Okafor", "His", {})
    with pytest.raises(KeyError):
        service.delete_view("Priya Nair", other["view_id"])
    with pytest.raises(ValueError, match="unknown workbench filter"):
        service.save_view("Priya Nair", "Bad", {"owner": "Priya"})


# -- search ------------------------------------------------------------------------


def test_search_names_the_field_each_hit_came_from(desk):
    service, planted = desk
    by_subject = service.search_cases("helvion")
    hit = next(r for r in by_subject if r["case_id"] == planted["draft"])
    assert hit["matches"][0]["field"] == "subject"

    case = service.store.get_case(planted["draft"])
    question = case["brief"]["questions"][0]
    token = next(
        w.strip(".,?").lower()
        for w in question["text"].split()
        if len(w.strip(".,?")) >= 6
    )
    by_question = service.search_cases(token)
    hit = next(r for r in by_question if r["case_id"] == planted["draft"])
    assert any(
        m["field"] == "question" and m["ref"] == question["id"]
        for m in hit["matches"]
    )

    finding = service.store.list_findings(planted["draft"])[0]
    token = next(
        w.strip(".,").lower()
        for w in finding["claim"].split()
        if len(w.strip(".,")) >= 6
    )
    by_finding = service.search_cases(token)
    hit = next(r for r in by_finding if r["case_id"] == planted["draft"])
    assert any(
        m["field"] == "finding" and m["ref"] == finding["finding_id"]
        for m in hit["matches"]
    )

    assert service.search_cases("zzz-no-such-text") == []
    with pytest.raises(ValueError, match="needs a query"):
        service.search_cases("   ")


# -- over HTTP -----------------------------------------------------------------------


@pytest.fixture()
def client(settings):
    return TestClient(create_app(settings))


def test_workbench_over_http(client):
    brief = GOLD_CASES[2]["brief"]
    case_id = client.post("/api/cases", json=brief).json()["case_id"]
    queue = client.get("/api/workbench/queue").json()
    assert [g["act"] for g in queue["groups"]] == [a for a, _ in ACTS]
    assert queue["total"] == 0
    client.post(f"/api/cases/{case_id}/plan")
    queue = client.get("/api/workbench/queue").json()
    assert queue["total"] == 1
    (item,) = queue["groups"][0]["items"]
    assert item["case_id"] == case_id and item["act"] == "plan_decision"

    assigned = client.post(
        f"/api/cases/{case_id}/assignment",
        json={"analyst": "Desk Lead", "assignee": "Priya Nair"},
    )
    assert assigned.status_code == 200
    assert assigned.json()["assignee"] == "Priya Nair"
    assert client.get(f"/api/cases/{case_id}").json()["priority"] == "normal"
    prioritised = client.post(
        f"/api/cases/{case_id}/priority",
        json={"analyst": "Desk Lead", "priority": "urgent"},
    )
    assert prioritised.json()["priority"] == "urgent"
    filtered = client.get(
        "/api/workbench/queue", params={"priority": "urgent"}
    ).json()
    assert filtered["total"] == 1
    bad = client.get("/api/workbench/queue", params={"act": "everything"})
    assert bad.status_code == 422

    timeline = client.get(f"/api/cases/{case_id}/timeline").json()
    assert timeline[0]["event"] == "case_created"
    assert any(t["event"] == "case_assigned" for t in timeline)

    hits = client.get("/api/workbench/search", params={"q": "selene"}).json()
    assert hits and hits[0]["matches"][0]["field"] == "subject"
    assert client.get("/api/workbench/search").status_code == 422

    saved = client.post(
        "/api/workbench/views",
        json={
            "analyst": "Priya Nair",
            "name": "Urgent plans",
            "filters": {"act": "plan_decision", "priority": "urgent"},
        },
    )
    assert saved.status_code == 201
    view_id = saved.json()["view_id"]
    views = client.get(
        "/api/workbench/views", params={"analyst": "Priya Nair"}
    ).json()
    assert [v["view_id"] for v in views] == [view_id]
    bad_view = client.post(
        "/api/workbench/views",
        json={"analyst": "Priya Nair", "name": "Bad", "filters": {"x": 1}},
    )
    assert bad_view.status_code == 422
    gone = client.request(
        "DELETE", f"/api/workbench/views/{view_id}",
        params={"analyst": "Priya Nair"},
    )
    assert gone.status_code == 200
    again = client.request(
        "DELETE", f"/api/workbench/views/{view_id}",
        params={"analyst": "Priya Nair"},
    )
    assert again.status_code == 404


def test_no_bulk_decision_route_exists(settings):
    """The hard rule, asserted against the route table itself:
    every decision route is addressed to one case (and, where the
    decision is on a record, to that record), and nothing in the
    API offers a bulk or queue-level decision."""
    app = create_app(settings)
    paths = [route.path for route in app.routes]
    assert not any("bulk" in path for path in paths)
    decision_paths = [p for p in paths if "decision" in p]
    assert decision_paths  # the per-case gates exist
    assert all("{case_id}" in p for p in decision_paths)
    workbench_paths = [p for p in paths if p.startswith("/api/workbench")]
    assert workbench_paths
    assert all(
        "decision" not in p and "approve" not in p and "reject" not in p
        for p in workbench_paths
    )
    client = TestClient(app)
    assert client.post("/api/workbench/decisions", json={}).status_code in (
        404, 405,
    )
    assert client.post("/api/workbench/queue/decide", json={}).status_code in (
        404, 405,
    )
