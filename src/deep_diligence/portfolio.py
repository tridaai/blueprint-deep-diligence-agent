"""Portfolio mode: engagements, and the portfolio view over them.

A desk does not run one diligence at a time. It runs a client's
pipeline — several vendor and acquisition cases under one
mandate — and until this round the shell had no word for that
grouping: the workbench queue was the whole desk or one case at
a time, search spanned every case the desk had ever held, and
the cost totals were one desk-wide line.

An **engagement** is the grouping: a governed record (name,
client label, owner, created by/at) with its own act history.
Membership lives on the case records — a case carries at most
one `engagement_id` — and assigning, moving, or removing a case
is a recorded act on both records (see
service.assign_case_engagement). The engagement itself holds no
case state and no member list: everything the portfolio view
shows is folded, on each read, from the member cases' own
records, so it cannot drift from them.

Isolation is the invariant this round exists to protect.
Portfolio membership changes nothing inside a case:

* gates stay per case — the portfolio's gate list is the
  workbench queue's own items (workbench.build_queue), filtered
  to members, never a new decision surface. There is no
  portfolio-level decision or deletion route anywhere.
* evidence stays per case — a finding's citations resolve
  through the case-scoped store reads they always used, so an
  evidence ID names a document in its own case only, even when
  a sibling case in the same engagement holds a lookalike ID
  (every case mints EV-0001, EV-0002, ... of its own). A
  portfolio read is never a citation path: search scoped to an
  engagement returns case-scoped hits labelled by case, and
  nothing here reads another case's evidence on a case's behalf.
* budgets and costs stay per case — the cost roll-up sums the
  per-case cost ledgers (costs.fold_cost_ledger over each case's
  own audit trail); it prices nothing and merges no budgets.
* retention and legal hold stay per case (governance, R10) —
  the portfolio view surfaces held and past-retention members
  distinctly precisely so nobody mistakes the grouping for a
  bulk handle on them.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from .costs import fold_cost_ledger
from .governance import parse_ts, retention_due
from .models import STATUSES
from .report_model import question_grade
from .store import Store, utcnow


def member_cases(store: Store, engagement_id: str) -> list[dict]:
    """The engagement's cases, read off the case records
    themselves — membership lives there, so this is the only
    member list there is."""
    return [
        case
        for case in store.list_cases()
        if case.get("engagement_id") == engagement_id
    ]


def _case_grades(store: Store, case: dict) -> list[dict] | None:
    """Per-question support grades for a case with a report,
    computed by the one documented rule (report_model
    .question_grade) over the report's claims — the same grades
    the report model renders, recomputed from the records so
    the portfolio row and the report cannot disagree. None when
    the case has no report yet."""
    report = store.get_report(case["case_id"])
    if report is None:
        return None
    claims = report.get("claims") or []
    evidence_by_id = {
        ev["evidence_id"]: ev for ev in store.list_evidence(case["case_id"])
    }
    verifier_report = store.get_verifier_report(case["case_id"]) or {}
    weak_claim_ids = {
        check["claim_id"]
        for check in verifier_report.get("checks", [])
        if check.get("model_check") == "unsupported"
    }
    return [
        {
            "question_id": question["id"],
            "grade": question_grade(
                [c for c in claims if c["question_id"] == question["id"]],
                evidence_by_id,
                weak_claim_ids,
            ),
        }
        for question in case["brief"]["questions"]
    ]


def _retention_view(case: dict, retention_days: int, now: datetime) -> dict:
    """A member's retention standing, for the portfolio view:
    its class, whether a legal hold stands, whether standard
    retention has run out, and the flags that make held and
    past-retention members distinct in the report. Read-only —
    the acts themselves stay per case (governance, R10)."""
    retention_class = case.get("retention_class") or "standard"
    due = retention_due(case, retention_days)
    past = retention_class == "standard" and now >= due
    flags: list[str] = []
    if retention_class == "legal_hold":
        flags.append("legal_hold")
    if past:
        flags.append("past_retention")
    return {
        "retention_class": retention_class,
        "legal_hold": retention_class == "legal_hold",
        "past_retention": past,
        "retention_due_at": due.isoformat(timespec="seconds"),
        "flags": flags,
    }


def _case_cost(store: Store, case_id: str) -> tuple[dict, Decimal]:
    """One case's cost line for the roll-up: its own ledger,
    folded from its own audit trail. Returns the JSON-facing
    line and the exact Decimal total the roll-up sums."""
    fold = fold_cost_ledger(store.list_audit(case_id))
    totals = fold["totals"]
    line = {
        "case_id": case_id,
        "calls": totals["calls"],
        "total_tokens": totals["total_tokens"],
        "cost_usd": float(totals["cost_usd"]),
        "contains_estimates": fold["contains_estimates"],
        "has_unpriced": fold["has_unpriced"],
    }
    return line, totals["cost_usd"]


def build_portfolio(
    store: Store,
    engagement: dict,
    queue: dict,
    retention_days: int,
    currency: str = "USD",
    now: datetime | None = None,
) -> dict:
    """One engagement's portfolio report, folded from the member
    cases' records on every read.

    `queue` is the desk's workbench queue (workbench.build_queue):
    the gate list is that queue's items filtered to members, so
    the portfolio shows exactly the acts the desk queue shows —
    same groups, same waits — scoped to the mandate. Nothing in
    the report is stored, and nothing in it decides anything.
    """
    now = now or parse_ts(utcnow())
    engagement_id = engagement["engagement_id"]
    members = member_cases(store, engagement_id)
    member_ids = {case["case_id"] for case in members}

    counts = {status: 0 for status in STATUSES}
    for case in members:
        counts[case["status"]] = counts.get(case["status"], 0) + 1

    gate_items: list[dict] = []
    for group in queue["groups"]:
        for item in group["items"]:
            if item["case_id"] in member_ids:
                gate_items.append({**item, "label": group["label"]})
    gate_items.sort(key=lambda i: (i["waiting_since"], i["case_id"]))
    oldest = gate_items[0] if gate_items else None

    case_rows: list[dict] = []
    open_conflicts: list[dict] = []
    open_quarantine: list[dict] = []
    cost_lines: list[dict] = []
    total_cost = Decimal(0)
    total_tokens = 0
    total_calls = 0
    contains_estimates = False
    for case in members:
        case_id = case["case_id"]
        conflicts = [
            {
                "conflict_id": record["conflict_id"],
                "case_id": case_id,
                "subject": case["subject"],
                "key": record["key"],
            }
            for record in store.list_conflicts(case_id)
            if record["status"] == "OPEN"
        ]
        quarantine = [
            {
                "quarantine_id": record["quarantine_id"],
                "claim_id": record["claim_id"],
                "case_id": case_id,
                "subject": case["subject"],
                "reason": record["reason"],
            }
            for record in store.list_quarantine(case_id)
            if record["status"] == "QUARANTINED"
        ]
        open_conflicts.extend(conflicts)
        open_quarantine.extend(quarantine)
        cost_line, exact = _case_cost(store, case_id)
        cost_lines.append(cost_line)
        total_cost += exact
        total_tokens += cost_line["total_tokens"]
        total_calls += cost_line["calls"]
        contains_estimates = (
            contains_estimates or cost_line["contains_estimates"]
        )
        retention = _retention_view(case, retention_days, now)
        case_rows.append(
            {
                "case_id": case_id,
                "subject": case["subject"],
                "status": case["status"],
                "analyst": case["analyst"],
                "assignee": case.get("assignee"),
                "priority": case.get("priority") or "normal",
                "created_at": case["created_at"],
                "updated_at": case["updated_at"],
                "grades": _case_grades(store, case),
                "open_conflicts": [c["conflict_id"] for c in conflicts],
                "open_quarantine": [q["quarantine_id"] for q in quarantine],
                "cost": cost_line,
                "retention": retention,
                "flags": retention["flags"],
            }
        )

    return {
        "engagement": {
            "engagement_id": engagement_id,
            "name": engagement["name"],
            "client": engagement["client"],
            "owner": engagement["owner"],
            "created_by": engagement["created_by"],
            "created_at": engagement["created_at"],
        },
        "generated_at": utcnow(),
        "total_cases": len(members),
        "counts_by_status": counts,
        "cases": case_rows,
        "gates": {
            "count": len(gate_items),
            "items": gate_items,
            "oldest_waiting": oldest,
        },
        "open_conflicts": open_conflicts,
        "open_quarantine": open_quarantine,
        "cost": {
            "currency": currency,
            "cases": cost_lines,
            "totals": {
                "cases": len(cost_lines),
                "calls": total_calls,
                "total_tokens": total_tokens,
                "cost_usd": float(total_cost),
            },
            "contains_estimates": contains_estimates,
        },
    }


def summarize_portfolio(report: dict) -> dict:
    """The list-facing shape of a portfolio report: the
    engagement, the case counts, the open-gate count and oldest
    wait, the cost total, and how many members stand held or
    past retention — the figures a pipeline review scans, with
    the full fold one read away."""
    oldest = report["gates"]["oldest_waiting"]
    return {
        "engagement_id": report["engagement"]["engagement_id"],
        "name": report["engagement"]["name"],
        "client": report["engagement"]["client"],
        "owner": report["engagement"]["owner"],
        "total_cases": report["total_cases"],
        "counts_by_status": report["counts_by_status"],
        "gates_open": report["gates"]["count"],
        "oldest_waiting_since": oldest["waiting_since"] if oldest else None,
        "oldest_waiting_case_id": oldest["case_id"] if oldest else None,
        "cost_usd": report["cost"]["totals"]["cost_usd"],
        "legal_hold_cases": sum(
            1 for row in report["cases"] if row["retention"]["legal_hold"]
        ),
        "past_retention_cases": sum(
            1 for row in report["cases"] if row["retention"]["past_retention"]
        ),
    }
