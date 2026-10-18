"""Analyst workbench: the queue, the timeline, case search.

Until this round, everything an analyst might owe a case lived on
that case's own page: a plan waiting at its gate, a quarantined
claim on its draft, a conflict nobody had noted yet, a case a dead
process left mid-execution. Finding the work meant opening every
case. The workbench composes those existing surfaces into the
desk's working view:

* **The queue** — every case that needs an analyst act, grouped by
  the act needed. It is derived, in code, from the records the
  case already carries (its status, its plan and report, its
  quarantine, conflict, and redaction-flag records, and the
  interrupted-case scan). No queue state is stored anywhere: the
  queue is what the records say, recomputed on each read, so it
  cannot drift from the cases it describes.
* **The timeline** — one chronological view of a case: the audit
  trail (which already carries every gate decision and state
  change), with the analyst's notes folded in at their timestamps
  and enriched with their text. Rendered from records; nothing is
  re-recorded to feed it.
* **Search** — deterministic substring search over case subjects,
  brief questions, and finding claims, naming the field each hit
  came from.

Assignment and priority live on the case record as triage (see
service.assign_case / set_priority): they order and filter the
queue, and gate nothing. There is deliberately no bulk decision
anywhere in this module or its API — a gate decision is one case,
one analyst, made on the case itself, exactly as before.
"""

from __future__ import annotations

from datetime import datetime

from .governance import parse_ts
from .models import PRIORITIES
from .store import Store, utcnow

# The acts, in the order the desk works them: gates first, then
# the records a draft carries, then the recoveries.
ACTS: tuple[tuple[str, str], ...] = (
    ("plan_decision", "Plan awaiting decision"),
    ("report_decision", "Draft awaiting decision"),
    ("quarantine_decision", "Quarantined claims awaiting disposition"),
    ("conflict_note", "Conflicts not yet noted"),
    ("redaction_flag_decision", "Redaction flags open"),
    ("interrupted_resume", "Interrupted — awaiting resume"),
    ("budget_amendment", "Budget exhausted — awaiting amendment"),
)
ACT_LABELS = dict(ACTS)

FILTER_KEYS = ("act", "assignee", "priority")
UNASSIGNED = "unassigned"

# Statuses from which the record-level decisions (quarantine,
# conflict NOTED, redaction flags) can actually be taken — the
# same gates the decision endpoints enforce.
_AT_GATE = ("DRAFT_READY", "BUDGET_EXHAUSTED")


def validate_filters(filters: dict | None) -> dict:
    """A queue filter (or a saved view's filter set) may only use
    the workbench's own keys, with values the queue understands.
    Anything else is refused by name, so a saved view can never
    drift into meaning something the queue does not implement."""
    filters = dict(filters or {})
    for key in filters:
        if key not in FILTER_KEYS:
            raise ValueError(
                f"unknown workbench filter {key!r}; "
                f"filters are: {', '.join(FILTER_KEYS)}"
            )
    act = filters.get("act")
    if act is not None and act not in ACT_LABELS:
        raise ValueError(
            f"unknown act {act!r}; acts are: {', '.join(ACT_LABELS)}"
        )
    priority = filters.get("priority")
    if priority is not None and priority not in PRIORITIES:
        raise ValueError(
            f"unknown priority {priority!r}; "
            f"priorities are: {', '.join(PRIORITIES)}"
        )
    return {k: v for k, v in filters.items() if v is not None}


def _age_seconds(waiting_since: str, now: datetime) -> int:
    return max(0, int((now - parse_ts(waiting_since)).total_seconds()))


def _latest_audit_ts(audit: list[dict], event: str) -> str | None:
    hits = [entry["ts"] for entry in audit if entry["event"] == event]
    return hits[-1] if hits else None


def _item(case: dict, act: str, detail: str, waiting_since: str,
          now: datetime, records: int = 1,
          record_ids: list[str] | None = None) -> dict:
    return {
        "case_id": case["case_id"],
        "subject": case["subject"],
        "status": case["status"],
        "act": act,
        "detail": detail,
        "records": records,
        "record_ids": record_ids or [],
        "waiting_since": waiting_since,
        "age_seconds": _age_seconds(waiting_since, now),
        "assignee": case.get("assignee"),
        "priority": case.get("priority") or "normal",
        "analyst": case.get("analyst"),
    }


def _case_items(store: Store, case: dict, now: datetime) -> list[dict]:
    """Every act one case owes, derived from its records."""
    case_id = case["case_id"]
    status = case["status"]
    items: list[dict] = []
    audit: list[dict] | None = None

    def trail() -> list[dict]:
        nonlocal audit
        if audit is None:
            audit = store.list_audit(case_id)
        return audit

    if status == "PLAN_READY":
        plan = store.get_plan(case_id)
        version = plan["version"] if plan else "?"
        waiting = (
            _latest_audit_ts(trail(), "plan_generated")
            or case["updated_at"]
        )
        items.append(
            _item(case, "plan_decision",
                  f"plan v{version} awaiting an analyst decision",
                  waiting, now)
        )
    if status == "DRAFT_READY":
        report = store.get_report(case_id)
        waiting = (report or {}).get("composed_at") or case["updated_at"]
        items.append(
            _item(case, "report_decision",
                  "report draft awaiting an analyst decision",
                  waiting, now)
        )
    if status == "BUDGET_EXHAUSTED":
        waiting = (
            _latest_audit_ts(trail(), "budget_exceeded")
            or case["updated_at"]
        )
        items.append(
            _item(case, "budget_amendment",
                  "partial draft composed; the budget may be "
                  "amended for re-execution, or the draft decided "
                  "as it stands",
                  waiting, now)
        )
    if status in _AT_GATE:
        open_quarantine = [
            r for r in store.list_quarantine(case_id)
            if r["status"] == "QUARANTINED"
        ]
        if open_quarantine:
            items.append(
                _item(
                    case, "quarantine_decision",
                    f"{len(open_quarantine)} quarantined claim(s) "
                    "awaiting disposition (send back or drop)",
                    min(r["created_at"] for r in open_quarantine),
                    now,
                    records=len(open_quarantine),
                    record_ids=[r["quarantine_id"] for r in open_quarantine],
                )
            )
        open_conflicts = [
            r for r in store.list_conflicts(case_id)
            if r["status"] == "OPEN"
        ]
        if open_conflicts:
            items.append(
                _item(
                    case, "conflict_note",
                    f"{len(open_conflicts)} conflict(s) not yet "
                    "marked NOTED",
                    min(r["created_at"] for r in open_conflicts),
                    now,
                    records=len(open_conflicts),
                    record_ids=[r["conflict_id"] for r in open_conflicts],
                )
            )
        report = store.get_report(case_id)
        live_claims = {
            c["claim_id"] for c in (report or {}).get("claims", [])
        }
        open_flags = [
            f for f in store.list_redaction_flags(case_id)
            if f["status"] == "OPEN" and f["claim_id"] in live_claims
        ]
        if open_flags:
            items.append(
                _item(
                    case, "redaction_flag_decision",
                    f"{len(open_flags)} redaction flag(s) open; the "
                    "report cannot be approved until each is "
                    "acknowledged or its claim sent back",
                    min(f["created_at"] for f in open_flags),
                    now,
                    records=len(open_flags),
                    record_ids=[f["flag_id"] for f in open_flags],
                )
            )
    return items


def build_queue(store: Store, interrupted: list[dict],
                filters: dict | None = None,
                now: datetime | None = None) -> dict:
    """The queue: every case needing an analyst act, grouped by
    the act, oldest waiting first inside each group.

    `interrupted` is the service's interrupted-case scan (listing
    is the detection point there, so the scan stays the
    service's act; the queue composes its result). Filters narrow
    every group the same way; all seven groups are always
    present, empty ones with a count of zero, so the desk sees
    the whole shape of the work, not just the filtered slice.
    """
    filters = validate_filters(filters)
    now = now or parse_ts(utcnow())
    by_act: dict[str, list[dict]] = {act: [] for act, _ in ACTS}
    for case in store.list_cases():
        for item in _case_items(store, case, now):
            by_act[item["act"]].append(item)
    for entry in interrupted:
        case = entry["case"]
        checkpoint = entry.get("checkpoint") or {}
        waiting = checkpoint.get("updated_at") or case["updated_at"]
        pending = entry.get("pending_questions") or []
        item = _item(
            case, "interrupted_resume",
            "execution stopped mid-run; still owed: "
            + (", ".join(pending) if pending else "none recorded"),
            waiting, now,
        )
        by_act["interrupted_resume"].append(item)

    def wanted(item: dict) -> bool:
        if filters.get("act") and item["act"] != filters["act"]:
            return False
        if filters.get("priority") and item["priority"] != filters["priority"]:
            return False
        assignee = filters.get("assignee")
        if assignee == UNASSIGNED:
            return item["assignee"] is None
        if assignee:
            return item["assignee"] == assignee
        return True

    groups: list[dict] = []
    total = 0
    for act, label in ACTS:
        items = sorted(
            (i for i in by_act[act] if wanted(i)),
            key=lambda i: (i["waiting_since"], i["case_id"]),
        )
        total += len(items)
        groups.append(
            {
                "act": act,
                "label": label,
                "count": len(items),
                "oldest_waiting_since": (
                    items[0]["waiting_since"] if items else None
                ),
                "oldest_age_seconds": (
                    items[0]["age_seconds"] if items else None
                ),
                "items": items,
            }
        )
    return {
        "generated_at": utcnow(),
        "filters": filters,
        "total": total,
        "groups": groups,
    }


# -- timeline ---------------------------------------------------------------

_EVENT_LABELS = {
    "case_created": "Case created",
    "evidence_ingested": "Evidence ingested",
    "plan_generated": "Plan generated",
    "plan_approved": "Plan approved",
    "plan_rejected": "Plan sent back",
    "execution_started": "Execution started",
    "execution_resumed": "Execution resumed",
    "execution_interruption_detected": "Interruption detected",
    "report_composed": "Report composed",
    "report_approved": "Report approved",
    "report_rejected": "Report sent back",
    "verification_completed": "Verification completed",
    "claims_quarantined": "Claims quarantined",
    "claim_sent_back": "Quarantined claim sent back",
    "claim_dropped": "Quarantined claim dropped",
    "conflicts_detected": "Conflicts detected",
    "conflict_noted": "Conflict marked NOTED",
    "evidence_drift_detected": "Evidence drift detected",
    "budget_granted": "Budget granted",
    "budget_consumed": "Budget consumed",
    "cost_recorded": "Run cost recorded",
    "budget_exceeded": "Budget exceeded",
    "budget_amended": "Budget amended",
    "case_note_added": "Analyst note",
    "report_section_note_added": "Report section note",
    "report_section_note_edited": "Report section note edited",
    "evidence_added": "Evidence added mid-case",
    "staleness_marked": "Workstream marked stale",
    "staleness_decided": "Staleness decided",
    "plan_revision_requested": "Plan revision requested",
    "legal_hold_set": "Legal hold set",
    "legal_hold_cleared": "Legal hold cleared",
    "evidence_redacted": "Evidence redacted",
    "report_claims_flagged_redaction": "Report claims flagged (redaction)",
    "redaction_claim_acknowledged": "Redaction-flagged claim acknowledged",
    "redaction_claim_sent_back": "Redaction-flagged claim sent back",
    "source_registered": "Source registered",
    "case_assigned": "Case assigned",
    "case_priority_set": "Priority set",
    "case_assigned_to_engagement": "Case added to an engagement",
    "case_removed_from_engagement": "Case removed from its engagement",
    "case_deleted": "Case deleted",
    "subagent_spawned": "Sub-agent spawned",
    "workstream_started": "Workstream started",
    "workstream_finished": "Workstream finished",
    "workstream_handoff": "Workstream handoff",
    "workstream_run_failed": "Workstream run failed",
    "findings_reused": "Findings carried over (reuse rule)",
    "report_snapshotted": "Report snapshot preserved",
    "planning_budget_exceeded": "Planning stopped on its budget",
    "workstreams_marked_stale": "Workstreams marked stale",
}

_DETAIL_KEYS = (
    ("version", "plan v{value}"),
    ("claim_id", "claim {value}"),
    ("conflict_id", "{value}"),
    ("quarantine_id", "{value}"),
    ("flag_id", "{value}"),
    ("evidence_id", "{value}"),
    ("assignee", "assignee: {value}"),
    ("priority", "priority: {value}"),
    ("engagement_name", "engagement: {value}"),
    ("action", "{value}"),
    ("reason", "{value}"),
    ("note", "{value}"),
)


def _detail(payload: dict) -> str:
    parts: list[str] = []
    for key, template in _DETAIL_KEYS:
        value = payload.get(key)
        if value not in (None, ""):
            parts.append(template.format(value=value))
    return " · ".join(parts)


def build_timeline(store: Store, case_id: str) -> list[dict]:
    """One chronological view of a case, merged from its records:
    the audit trail (every gate decision and state change is
    already an audit event), with the case notes and report
    section notes folded in at their own events and enriched with
    their text, so the timeline reads as the case's story rather
    than as a raw event dump. Nothing is recorded to feed this
    view; it renders what the case already holds."""
    notes = {n["note_id"]: n for n in store.list_notes(case_id)}
    section_notes = {
        n["note_id"]: n for n in store.list_section_notes(case_id)
    }
    entries: list[dict] = []
    for seq, entry in enumerate(store.list_audit(case_id)):
        event = entry["event"]
        payload = entry.get("payload") or {}
        kind = "event"
        detail = _detail(payload)
        note = notes.get(payload.get("note_id") or "")
        section_note = section_notes.get(payload.get("note_id") or "")
        if event == "case_note_added" and note is not None:
            kind = "note"
            detail = note["text"]
        elif event.startswith("report_section_note") and section_note is not None:
            kind = "section_note"
            detail = f"[{section_note['section']}] {section_note['text']}"
        actor = (
            payload.get("analyst")
            or payload.get("decided_by")
            or payload.get("noted_by")
            or payload.get("registered_by")
        )
        entries.append(
            {
                "ts": entry["ts"],
                "seq": seq,
                "kind": kind,
                "event": event,
                "label": _EVENT_LABELS.get(
                    event, event.replace("_", " ").capitalize()
                ),
                "detail": detail,
                "actor": actor,
            }
        )
    entries.sort(key=lambda e: (e["ts"], e["seq"]))
    for entry in entries:
        del entry["seq"]
    return entries


# -- search -------------------------------------------------------------------

def _snippet(text: str, needle: str, width: int = 200) -> str:
    if len(text) <= width:
        return text
    at = text.lower().find(needle)
    start = max(0, at - 60)
    end = min(len(text), start + width)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end] + suffix


def search_cases(
    store: Store, query: str, case_ids: set[str] | None = None
) -> list[dict]:
    """Deterministic substring search over the desk's cases: the
    subject, the brief's questions, and the findings' claims. A
    hit names the field it came from (and the question or finding
    id), so the analyst lands on the text that matched, not just
    the case. Case-insensitive; ordering follows the case list
    (oldest case first), matches in field order.

    `case_ids` scopes the search to a set of cases — portfolio
    mode passes an engagement's members. Scoping narrows which
    cases are read; it never widens what a hit can draw on: each
    result is still one case's own records, labelled with that
    case's id, and a finding's citations still resolve only
    inside that case."""
    needle = (query or "").strip().lower()
    if not needle:
        raise ValueError("a search needs a query")
    results: list[dict] = []
    for case in store.list_cases():
        if case_ids is not None and case["case_id"] not in case_ids:
            continue
        matches: list[dict] = []
        if needle in case["subject"].lower():
            matches.append(
                {"field": "subject", "ref": case["case_id"],
                 "text": case["subject"]}
            )
        for question in case["brief"].get("questions", []):
            if needle in question["text"].lower():
                matches.append(
                    {"field": "question", "ref": question["id"],
                     "text": _snippet(question["text"], needle)}
                )
        for finding in store.list_findings(case["case_id"]):
            if needle in finding["claim"].lower():
                matches.append(
                    {"field": "finding", "ref": finding["finding_id"],
                     "text": _snippet(finding["claim"], needle)}
                )
        if matches:
            results.append(
                {
                    "case_id": case["case_id"],
                    "subject": case["subject"],
                    "status": case["status"],
                    "assignee": case.get("assignee"),
                    "priority": case.get("priority") or "normal",
                    "matches": matches,
                }
            )
    return results
