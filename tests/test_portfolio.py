"""Portfolio mode (R15): engagements as records, the portfolio
view as a fold over member cases, engagement-scoped search, and
the isolation invariant — membership changes nothing inside a
case.

The fold tests plant real case states through the service (full
runs, a budget-tight run, a legal hold, a backdated case) and
grade the portfolio report against the member cases' own
records, recomputed independently in the test: the queue's own
items filtered to members, the cost ledgers folded per case,
the grades recomputed by the one documented rule. The isolation
test puts two cases in one engagement and attacks the seam the
red-team pack guards (R14): lookalike evidence IDs across the
sibling cases.
"""

from __future__ import annotations

import copy
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.corpus import GOLD_CASES, INTEGRITY_CASES
from deep_diligence.costs import fold_cost_ledger
from deep_diligence.governance import parse_ts
from deep_diligence.models import BriefIn
from deep_diligence.report_model import question_grade
from deep_diligence.service import GateError
from deep_diligence.tools import build_case_tools

AN = "A. Nother"


def _brief(index: int = 0) -> BriefIn:
    return BriefIn.model_validate(GOLD_CASES[index]["brief"])


def _engagement(service, name="Helvion Client Mandate"):
    return service.create_engagement(
        AN, name, "Helvion Industries", "Desk Lead"
    )


def _run_case(service, index: int = 0, final: bool = False) -> str:
    case = service.create_case(_brief(index))
    cid = case["case_id"]
    service.run_planning(cid)
    service.decide_plan(cid, AN, True)
    service.run_execution(cid)
    if final:
        service.decide_report(cid, AN, True)
    return cid


def _budget_case(service) -> str:
    raw = copy.deepcopy(GOLD_CASES[2]["brief"])
    phase = raw["budget"].get("execution") or raw["budget"]
    phase["max_steps"] = 6
    case = service.create_case(BriefIn.model_validate(raw))
    cid = case["case_id"]
    service.run_planning(cid)
    service.decide_plan(cid, AN, True)
    service.run_execution(cid)
    assert service.store.get_case(cid)["status"] == "BUDGET_EXHAUSTED"
    return cid


# -- engagements as records -------------------------------------------


def test_engagement_record_and_act_history(service):
    record = _engagement(service)
    assert record["engagement_id"] == "ENG-0001"
    assert record["name"] == "Helvion Client Mandate"
    assert record["client"] == "Helvion Industries"
    assert record["owner"] == "Desk Lead"
    assert record["created_by"] == AN
    assert record["history"] == [
        {"action": "created", "analyst": AN, "at": record["created_at"]}
    ]
    second = service.create_engagement(AN, "Second", "Client", "Owner")
    assert second["engagement_id"] == "ENG-0002"
    assert [e["engagement_id"] for e in service.store.list_engagements()] == [
        "ENG-0001",
        "ENG-0002",
    ]


def test_assignment_is_recorded_on_both_records_and_moves_keep_history(
    service,
):
    first = _engagement(service, "First Mandate")
    second = _engagement(service, "Second Mandate")
    case = service.create_case(_brief(0))
    cid = case["case_id"]
    assert case["engagement_id"] is None

    updated = service.assign_case_engagement(cid, first["engagement_id"], AN)
    assert updated["engagement_id"] == first["engagement_id"]
    audit = service.store.list_audit(cid)
    assigned = [a for a in audit if a["event"] == "case_assigned_to_engagement"]
    assert len(assigned) == 1
    assert assigned[0]["payload"]["engagement_id"] == first["engagement_id"]
    assert assigned[0]["payload"]["previous_engagement_id"] is None
    history = service.store.get_engagement(first["engagement_id"])["history"]
    assert [h["action"] for h in history] == ["created", "case_assigned"]
    assert history[-1]["from_engagement_id"] is None

    # A move writes both sides: out of the first (naming where
    # the case went), into the second (naming where it came
    # from), and onto the case's own trail.
    updated = service.assign_case_engagement(cid, second["engagement_id"], AN)
    assert updated["engagement_id"] == second["engagement_id"]
    first_history = service.store.get_engagement(first["engagement_id"])[
        "history"
    ]
    assert first_history[-1]["action"] == "case_unassigned"
    assert first_history[-1]["to_engagement_id"] == second["engagement_id"]
    second_history = service.store.get_engagement(second["engagement_id"])[
        "history"
    ]
    assert second_history[-1]["action"] == "case_assigned"
    assert second_history[-1]["from_engagement_id"] == first["engagement_id"]
    audit = service.store.list_audit(cid)
    moves = [a for a in audit if a["event"] == "case_assigned_to_engagement"]
    assert moves[-1]["payload"]["previous_engagement_id"] == (
        first["engagement_id"]
    )

    # Removal is an act too, and the case is unassigned again.
    updated = service.assign_case_engagement(cid, None, AN)
    assert updated["engagement_id"] is None
    events = [a["event"] for a in service.store.list_audit(cid)]
    assert "case_removed_from_engagement" in events


def test_assignment_refusals(service):
    engagement = _engagement(service)
    case = service.create_case(_brief(0))
    cid = case["case_id"]
    with pytest.raises(KeyError):
        service.assign_case_engagement(cid, "ENG-9999", AN)
    with pytest.raises(KeyError):
        service.assign_case_engagement("case-nope", engagement["engagement_id"], AN)
    service.assign_case_engagement(cid, engagement["engagement_id"], AN)
    # Re-assigning the membership the case already has is not an
    # act; neither is removing a case that is in no engagement.
    with pytest.raises(GateError):
        service.assign_case_engagement(cid, engagement["engagement_id"], AN)
    other = service.create_case(_brief(1))
    with pytest.raises(GateError):
        service.assign_case_engagement(other["case_id"], None, AN)


# -- the fold ------------------------------------------------------------


@pytest.fixture(scope="module")
def desk(tmp_path_factory):
    """A planted desk: one engagement with a FINAL case, a
    DRAFT_READY case, and a BUDGET_EXHAUSTED case; one unassigned
    case and one case in a different engagement that must never
    leak into the fold."""
    from deep_diligence.config import load_settings
    from deep_diligence.service import CaseService

    root = tmp_path_factory.mktemp("portfolio-desk")
    settings = load_settings(
        sqlite_path=str(root / "dd.sqlite3"),
        object_dir=str(root / "objects"),
    )
    svc = CaseService(settings)
    engagement = _engagement(svc)
    other = _engagement(svc, "Unrelated Mandate")
    ids = {
        "final": _run_case(svc, 0, final=True),
        "draft": _run_case(svc, 1),
        "budget": _budget_case(svc),
        "outside": _run_case(svc, 0),
        "other_member": _run_case(svc, 1),
    }
    for key in ("final", "draft", "budget"):
        svc.assign_case_engagement(ids[key], engagement["engagement_id"], AN)
    svc.assign_case_engagement(
        ids["other_member"], other["engagement_id"], AN
    )
    return svc, engagement["engagement_id"], ids


def test_portfolio_fold_matches_the_member_records(desk):
    svc, engagement_id, ids = desk
    members = {ids["final"], ids["draft"], ids["budget"]}
    report = svc.portfolio_report(engagement_id)

    # Membership and counts come off the case records.
    assert report["total_cases"] == 3
    assert {row["case_id"] for row in report["cases"]} == members
    counts = report["counts_by_status"]
    assert counts["FINAL"] == 1
    assert counts["DRAFT_READY"] == 1
    assert counts["BUDGET_EXHAUSTED"] == 1
    assert sum(counts.values()) == 3

    # Gates are the workbench queue's own items, scoped to
    # members — recomputed here from the desk queue.
    queue = svc.workbench_queue()
    expected = sorted(
        (item["act"], item["case_id"], item["waiting_since"])
        for group in queue["groups"]
        for item in group["items"]
        if item["case_id"] in members
    )
    got = sorted(
        (item["act"], item["case_id"], item["waiting_since"])
        for item in report["gates"]["items"]
    )
    assert got == expected
    assert report["gates"]["count"] == len(expected)
    assert report["gates"]["oldest_waiting"]["waiting_since"] == min(
        item["waiting_since"] for item in report["gates"]["items"]
    )
    acts = {(i["act"], i["case_id"]) for i in report["gates"]["items"]}
    assert ("report_decision", ids["draft"]) in acts
    assert ("budget_amendment", ids["budget"]) in acts

    # The cost roll-up is the sum of the members' own ledgers,
    # recomputed here case by case.
    folds = {
        cid: fold_cost_ledger(svc.store.list_audit(cid)) for cid in members
    }
    expected_total = sum(f["totals"]["cost_usd"] for f in folds.values())
    assert report["cost"]["totals"]["cost_usd"] == float(expected_total)
    assert report["cost"]["totals"]["total_tokens"] == sum(
        f["totals"]["total_tokens"] for f in folds.values()
    )
    assert report["cost"]["totals"]["calls"] == sum(
        f["totals"]["calls"] for f in folds.values()
    )
    for row in report["cases"]:
        assert row["cost"]["cost_usd"] == float(
            folds[row["case_id"]]["totals"]["cost_usd"]
        )

    # Grades are the one documented rule over each case's own
    # report — recomputed here for the draft case.
    draft_case = svc.store.get_case(ids["draft"])
    claims = svc.store.get_report(ids["draft"])["claims"]
    evidence_by_id = {
        ev["evidence_id"]: ev
        for ev in svc.store.list_evidence(ids["draft"])
    }
    verifier = svc.store.get_verifier_report(ids["draft"]) or {}
    weak = {
        c["claim_id"]
        for c in verifier.get("checks", [])
        if c.get("model_check") == "unsupported"
    }
    expected_grades = [
        question_grade(
            [c for c in claims if c["question_id"] == q["id"]],
            evidence_by_id,
            weak,
        )
        for q in draft_case["brief"]["questions"]
    ]
    row = next(r for r in report["cases"] if r["case_id"] == ids["draft"])
    assert [g["grade"] for g in row["grades"]] == expected_grades

    # Open conflicts and quarantines are the members' own open
    # records, by id, labelled by case.
    expected_conflicts = sorted(
        (rec["conflict_id"], cid)
        for cid in members
        for rec in svc.store.list_conflicts(cid)
        if rec["status"] == "OPEN"
    )
    assert sorted(
        (c["conflict_id"], c["case_id"]) for c in report["open_conflicts"]
    ) == expected_conflicts
    expected_quarantine = sorted(
        (rec["quarantine_id"], cid)
        for cid in members
        for rec in svc.store.list_quarantine(cid)
        if rec["status"] == "QUARANTINED"
    )
    assert sorted(
        (q["quarantine_id"], q["case_id"]) for q in report["open_quarantine"]
    ) == expected_quarantine

    # The other engagement's fold holds only its own member.
    others = [
        s
        for s in svc.list_portfolios()
        if s["engagement_id"] != engagement_id
    ]
    assert len(others) == 1
    assert others[0]["total_cases"] == 1


def test_portfolio_surfaces_held_and_past_retention_members(desk):
    svc, engagement_id, ids = desk
    svc.set_legal_hold(ids["final"], "Counsel", "Mandate under review")
    case = svc.store.get_case(ids["draft"])
    long_ago = parse_ts(case["created_at"]) - timedelta(days=400)
    svc.store.update_case(
        ids["draft"], created_at=long_ago.isoformat(timespec="seconds")
    )
    report = svc.portfolio_report(engagement_id)
    rows = {row["case_id"]: row for row in report["cases"]}
    assert rows[ids["final"]]["flags"] == ["legal_hold"]
    assert rows[ids["final"]]["retention"]["legal_hold"] is True
    assert "past_retention" in rows[ids["draft"]]["flags"]
    assert rows[ids["draft"]]["retention"]["past_retention"] is True
    assert rows[ids["budget"]]["flags"] == []
    summary = next(
        s
        for s in svc.list_portfolios()
        if s["engagement_id"] == engagement_id
    )
    assert summary["legal_hold_cases"] == 1
    assert summary["past_retention_cases"] == 1
    # Surfacing changes nothing underneath: the hold still
    # blocks the per-case deletion act, exactly as before.
    with pytest.raises(GateError):
        svc.delete_case(ids["final"], "Counsel", "trying anyway")


# -- search scoping -------------------------------------------------------


def test_engagement_search_is_scoped_and_case_labelled(service):
    engagement = _engagement(service)
    aster = service.create_case(
        BriefIn.model_validate(INTEGRITY_CASES[0]["brief"])
    )
    selene = service.create_case(_brief(2))
    copperline = service.create_case(_brief(1))
    service.assign_case_engagement(
        aster["case_id"], engagement["engagement_id"], AN
    )
    service.assign_case_engagement(
        selene["case_id"], engagement["engagement_id"], AN
    )
    # "revenue" appears in all three briefs' questions; scoped
    # to the engagement, only the two members answer, each hit
    # labelled with its own case.
    scoped = service.search_cases(
        "revenue", engagement_id=engagement["engagement_id"]
    )
    assert {hit["case_id"] for hit in scoped} == {
        aster["case_id"],
        selene["case_id"],
    }
    assert all(hit["matches"] for hit in scoped)
    unscoped = service.search_cases("revenue")
    assert copperline["case_id"] in {hit["case_id"] for hit in unscoped}
    # A term only the outsider carries never answers a scoped
    # search.
    assert (
        service.search_cases(
            "Copperline", engagement_id=engagement["engagement_id"]
        )
        == []
    )
    with pytest.raises(KeyError):
        service.search_cases("revenue", engagement_id="ENG-9999")


# -- isolation ------------------------------------------------------------


def test_isolation_two_cases_one_engagement(service):
    """The R14 cross-case defences, with both cases inside one
    engagement: membership must not widen any citation path.
    Attack material is the red-team pack's own (corpus
    RED_TEAM_CASES, kind cross_case)."""
    from deep_diligence.corpus import RED_TEAM_CASES

    gold = next(c for c in RED_TEAM_CASES if c["kind"] == "cross_case")
    engagement = _engagement(service)
    case_a = service.create_case(BriefIn.model_validate(gold["briefs"][0]))
    case_b = service.create_case(BriefIn.model_validate(gold["briefs"][1]))
    for case in (case_a, case_b):
        service.assign_case_engagement(
            case["case_id"], engagement["engagement_id"], AN
        )
    # Lookalike IDs are the normal state of two cases: both hold
    # an EV-0001, and they are different documents.
    ev_a = service.store.get_evidence(case_a["case_id"], "EV-0001")
    ev_b = service.store.get_evidence(case_b["case_id"], "EV-0001")
    assert ev_a["doc_slug"] != ev_b["doc_slug"]
    tools_a = {
        tool.name: tool.func
        for tool in build_case_tools(
            service.store,
            service.objects,
            service.store.get_case(case_a["case_id"]),
        )
    }
    mixed = gold["mixed_claim"]
    foreign = gold["foreign_evidence_id"]

    # A foreign-only citation is refused, engagement or not.
    result = tools_a["record_finding"](
        question_id=mixed["question_id"],
        claim=mixed["claim"],
        evidence_ids=[foreign],
        fact_ref=mixed["fact_ref"],
    )
    assert result.startswith("ERROR")
    assert service.store.list_findings(case_a["case_id"]) == []

    # A mixed citation keeps only the case's own IDs.
    result = tools_a["record_finding"](
        question_id=mixed["question_id"],
        claim=mixed["claim"],
        evidence_ids=[mixed["own_evidence_id"], foreign],
        fact_ref=mixed["fact_ref"],
    )
    assert not result.startswith("ERROR")
    findings = service.store.list_findings(case_a["case_id"])
    assert len(findings) == 1
    assert findings[0]["evidence_ids"] == [mixed["own_evidence_id"]]

    # The sibling's fact, cited against this case's own genuine
    # EV-0001, is still refused: the ledger it must appear in is
    # this case's document, and it is not there.
    result = tools_a["record_finding"](
        question_id="q1",
        claim=(
            "Selene's lead product is a rapid respiratory panel "
            "that returns results in 25 minutes."
        ),
        evidence_ids=["EV-0001"],
        fact_ref="F-SEL-101",
    )
    assert result.startswith("ERROR")
    assert len(service.store.list_findings(case_a["case_id"])) == 1


# -- route table ------------------------------------------------------------


def test_no_portfolio_decision_or_delete_route(service):
    """The route-table proof, extended to portfolio mode: no
    route under the engagement or portfolio paths deletes
    anything or takes a decision, and the only writes are
    creating an engagement and the per-case assignment act."""
    app = create_app(service.settings, service=service)
    offenders = []
    writes = []
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) or set()
        if "portfolio" not in path and "engagement" not in path:
            continue
        if "DELETE" in methods:
            offenders.append(f"DELETE {path}")
        for segment in ("decision", "approve", "reject", "delete"):
            if segment in path:
                offenders.append(f"{segment} in {path}")
        if methods - {"GET", "POST"}:
            offenders.append(f"{methods} on {path}")
        if "POST" in methods:
            writes.append(path)
    assert offenders == []
    assert sorted(writes) == [
        "/api/cases/{case_id}/engagement",
        "/api/engagements",
    ]


# -- HTTP surface ------------------------------------------------------------


def test_portfolio_over_http(service):
    client = TestClient(create_app(service.settings, service=service))
    created = client.post(
        "/api/engagements",
        json={
            "analyst": AN,
            "name": "HTTP Mandate",
            "client": "HTTP Client",
            "owner": "Desk Lead",
        },
    )
    assert created.status_code == 201
    engagement = created.json()
    assert engagement["engagement_id"] == "ENG-0001"

    case = client.post(
        "/api/cases", json=_brief(0).model_dump()
    ).json()
    assigned = client.post(
        f"/api/cases/{case['case_id']}/engagement",
        json={"analyst": AN, "engagement_id": engagement["engagement_id"]},
    )
    assert assigned.status_code == 200
    assert assigned.json()["engagement_id"] == engagement["engagement_id"]

    summaries = client.get("/api/portfolio").json()
    assert len(summaries) == 1
    assert summaries[0]["total_cases"] == 1
    assert summaries[0]["counts_by_status"]["INTAKE"] == 1

    report = client.get(f"/api/portfolio/{engagement['engagement_id']}")
    assert report.status_code == 200
    body = report.json()
    assert body["cases"][0]["case_id"] == case["case_id"]
    assert body["cases"][0]["grades"] is None

    missing = client.get("/api/portfolio/ENG-9999")
    assert missing.status_code == 404
    bad_assign = client.post(
        f"/api/cases/{case['case_id']}/engagement",
        json={"analyst": AN, "engagement_id": "ENG-9999"},
    )
    assert bad_assign.status_code == 404
    scoped = client.get(
        "/api/workbench/search",
        params={"q": "revenue", "engagement_id": engagement["engagement_id"]},
    )
    assert scoped.status_code == 200
    assert {h["case_id"] for h in scoped.json()} == {case["case_id"]}
