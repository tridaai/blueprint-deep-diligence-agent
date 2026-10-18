"""Observability: the run trace, the failure taxonomy, and the
operations digest.

Three reads over the records the shell already keeps — the case
record, the versioned plan, the audit trail with its budget and
cost ledgers, the execution checkpoint, the findings, the report,
the verifier's quarantine, the conflicts, and the challenges.
Nothing here writes to a case, counts into a counter, or asks the
harness anything: each read is a fold over stored records,
computed at the moment of asking, so it cannot drift from what
it summarises (the discipline the workbench queue, the portfolio
fold, and the cost ledger already follow). The one stored thing
in this round is the digest snapshot itself — a dated record the
desk generates as an act, kept versioned in the store.

- **The trace** (`build_trace`) reconstructs one case's run end
  to end: the phase runs folded from the budget ledger (the caps
  each run was granted, what it consumed, what it cost, the cap
  it tripped if it tripped one), and the act sequence — intake,
  planning, the plan gate, execution with its waves and
  workstream outcomes, composition, verification, the challenge
  pass and its dispositions, the report gate — in the audit
  trail's own order, which is causal order. Every audit entry
  becomes exactly one act; nothing is dropped or re-ordered.
  Timing honesty: the trail's stamps are second-resolution and
  the runs record usage (steps, tokens, spawns, wall seconds)
  per run — those figures ride the runs. No per-act or
  per-workstream durations were ever recorded, so the trace
  shows none rather than estimating them.
- **The taxonomy** (`classify_case_outcome`) names how a case's
  run ended, or what stands in its way: ONE function, a
  documented rule order, first match wins, and a fallback that
  makes the classification total — every case maps to exactly
  one label, so an outcome with no class is impossible by
  construction and the interesting failures (drift, exhaustion,
  a dead branch) can never hide inside a generic bucket.
- **The digest** (`fold_digest`) summarises one day across the
  desk: cases opened and closed, the gate decisions taken, the
  partial reports still standing with the questions they leave
  uncovered, the day's cost folded from the cost ledgers with
  Decimal arithmetic (recomputed from the recorded lines, never
  a stored counter), taxonomy counts over the day's cases, and
  the challenge position — what was raised, what was answered,
  and the must-answer challenges still open desk-wide, with
  their ages. The service stores the fold as a versioned record
  (migration 0013); the fold itself is pure.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal

from .costs import fold_cost_ledger
from .store import Store, utcnow

# ---------------------------------------------------------------- taxonomy

# Every label classify_case_outcome can return, in rule order.
# The eval pack asserts totality against this set: a case whose
# classification falls outside it, or a terminal case that lands
# in `in_progress`, is a gate failure.
OUTCOME_LABELS = (
    "evidence_drift",
    "budget_exhausted",
    "challenge_blocked",
    "interrupted",
    "workstream_failed",
    "claims_quarantined",
    "stale_evidence",
    "plan_rejected",
    "report_rejected",
    "completed_partial",
    "completed",
    "in_progress",
)


def _failed_workstreams(
    store: Store, case_id: str, report: dict | None, checkpoint: dict | None
) -> list[dict]:
    """The latest execution's failed workstreams, from the same
    records the case view reads: the report's workstream record
    once a report exists, else the checkpoint, else the audit
    trail's finish events."""
    if report and report.get("workstreams"):
        return [
            {
                "question_id": s["question_id"],
                "workstream": s.get("workstream"),
                "reason": s.get("reason"),
            }
            for s in report["workstreams"]
            if s.get("status") == "failed"
        ]
    if checkpoint and checkpoint.get("workstreams"):
        return [
            {
                "question_id": qid,
                "workstream": entry.get("workstream"),
                "reason": entry.get("reason"),
            }
            for qid, entry in checkpoint["workstreams"].items()
            if entry.get("status") == "failed"
        ]
    failed: dict[str, dict] = {}
    for entry in store.list_audit(case_id):
        payload = entry.get("payload") or {}
        if entry["event"] == "workstream_finished" and (
            payload.get("status") == "failed"
        ):
            failed[payload["question_id"]] = {
                "question_id": payload["question_id"],
                "workstream": payload.get("workstream"),
                "reason": payload.get("reason"),
            }
    return list(failed.values())


def _coverage_split(rows: list[dict] | None) -> dict:
    rows = rows or []
    return {
        "uncovered": [
            r["question_id"] for r in rows if r.get("status") == "NOT COVERED"
        ],
        "partial": [
            r["question_id"] for r in rows if r.get("status") == "PARTIAL"
        ],
    }


def classify_case_outcome(store: Store, case_id: str) -> dict:
    """The one taxonomy: how this case's run ended, or what
    stands in its way. Rules run in this order and the first
    match wins:

    1. `evidence_drift` — the report record is flagged
       EVIDENCE_DRIFT: cited evidence changed after citation
       and the report gate refused approval on it. The drifted
       evidence IDs are named.
    2. `budget_exhausted` — the execution stopped on a cap and
       the partial draft stands undecided (status
       BUDGET_EXHAUSTED). The cap, the phase, and the questions
       the coverage table leaves uncovered are named.
    3. `challenge_blocked` — an unanswered must-answer
       challenge stands open, so the report gate would refuse
       approval. The challenge IDs are named. (Advisory
       challenges never classify a case; they are counted on
       the case view and in the digest, and block nothing.)
    4. `interrupted` — the case sits in EXECUTING with a
       checkpoint that never closed: the process died mid-run
       and a resume is owed. The attempt and the checkpoint's
       completed/pending split are named where recorded.
    5. `workstream_failed` — the latest execution has a
       workstream that ended failed (its wave's run raised; the
       failure was isolated per the schedule, and the case
       drafted around it). The branches and their recorded
       errors are named.
    6. `claims_quarantined` — the current draft has claims
       standing in quarantine, awaiting an analyst disposition.
       The claims and the verifier's reasons are named.
    7. `stale_evidence` — evidence arrived mid-case and the
       workstreams it speaks to stand marked stale: execution
       is refused until an analyst proceeds or re-plans. The
       questions and the documents are named.
    8. `plan_rejected` — the analyst sent the plan back at the
       plan gate (status PLAN_REJECTED).
    9. `report_rejected` — the analyst sent the draft back at
       the report gate (status REPORT_REJECTED).
    10. `completed_partial` — FINAL over a partial draft: the
        analyst approved the coverage table the approval record
        carries. The uncovered questions are named.
    11. `completed` — FINAL: the report was approved.
    12. `in_progress` — anything else: the case stands at a
        gate or has not run yet, and the standing status is
        named. This fallback is what makes the taxonomy total.
    """
    case = store.get_case(case_id)
    if case is None:
        raise KeyError(case_id)
    status = case["status"]
    report = store.get_report(case_id)
    checkpoint = store.get_checkpoint(case_id)

    if report and (report.get("integrity") or {}).get("status") == (
        "EVIDENCE_DRIFT"
    ):
        drift = report["integrity"].get("drift") or []
        return {
            "label": "evidence_drift",
            "detail": {
                "evidence_ids": sorted({d["evidence_id"] for d in drift}),
                "checked_at": report["integrity"].get("checked_at"),
            },
        }
    if status == "BUDGET_EXHAUSTED":
        detail = {
            "exceeded_kind": (report or {}).get("exceeded_kind"),
            "exceeded_phase": (report or {}).get("exceeded_phase"),
        }
        detail.update(_coverage_split((report or {}).get("coverage_detail")))
        return {"label": "budget_exhausted", "detail": detail}
    open_must_answer = [
        c
        for c in store.list_challenges(case_id)
        if c["status"] == "OPEN" and c["severity"] == "must_answer"
    ]
    if open_must_answer and status != "FINAL":
        return {
            "label": "challenge_blocked",
            "detail": {
                "challenge_ids": [c["challenge_id"] for c in open_must_answer],
                "rules": sorted({c["rule"] for c in open_must_answer}),
            },
        }
    if status == "EXECUTING" and not (checkpoint and checkpoint.get("complete")):
        detail: dict = {"attempt": (checkpoint or {}).get("attempt")}
        if checkpoint and checkpoint.get("workstreams"):
            detail["completed_questions"] = sorted(
                qid
                for qid, e in checkpoint["workstreams"].items()
                if e.get("status") in ("completed", "empty")
            )
            detail["pending_questions"] = sorted(
                qid
                for qid, e in checkpoint["workstreams"].items()
                if e.get("status") not in ("completed", "empty")
            )
        return {"label": "interrupted", "detail": detail}
    failed = _failed_workstreams(store, case_id, report, checkpoint)
    if failed:
        return {"label": "workstream_failed", "detail": {"failed": failed}}
    standing_quarantine = [
        r for r in store.list_quarantine(case_id) if r["status"] == "QUARANTINED"
    ]
    if standing_quarantine and status != "FINAL":
        return {
            "label": "claims_quarantined",
            "detail": {
                "claims": [
                    {
                        "claim_id": r["claim_id"],
                        "quarantine_id": r["quarantine_id"],
                        "reason": r["reason"],
                    }
                    for r in standing_quarantine
                ]
            },
        }
    stale = case.get("stale_marks") or []
    if stale:
        return {
            "label": "stale_evidence",
            "detail": {
                "question_ids": sorted({m["question_id"] for m in stale}),
                "evidence_ids": sorted({m["evidence_id"] for m in stale}),
            },
        }
    if status == "PLAN_REJECTED":
        plan = store.get_plan(case_id) or {}
        return {
            "label": "plan_rejected",
            "detail": {
                "plan_version": plan.get("version"),
                "decided_by": plan.get("decided_by"),
                "note": plan.get("decision_note"),
            },
        }
    if status == "REPORT_REJECTED":
        return {
            "label": "report_rejected",
            "detail": {
                "decided_by": (report or {}).get("decided_by"),
                "note": (report or {}).get("decision_note"),
            },
        }
    if status == "FINAL":
        approved_coverage = (report or {}).get("approved_coverage")
        if approved_coverage:
            detail = {"decided_by": (report or {}).get("decided_by")}
            detail.update(_coverage_split(approved_coverage))
            return {"label": "completed_partial", "detail": detail}
        return {
            "label": "completed",
            "detail": {"decided_by": (report or {}).get("decided_by")},
        }
    detail = {"status": status}
    if status == "EXECUTING" and checkpoint:
        detail["checkpoint_outcome"] = checkpoint.get("outcome")
    return {"label": "in_progress", "detail": detail}


# ------------------------------------------------------------------- trace

# Audit event -> (act kind, act label). Every audit entry becomes
# exactly one act; an event missing from this table still appears,
# kind "event", with a humanised label — the trace drops nothing.
_ACT_KINDS: dict[str, tuple[str, str]] = {
    "case_created": ("intake", "Case created"),
    "evidence_ingested": ("intake", "Evidence ingested"),
    "evidence_added": ("intake", "Evidence added mid-case"),
    "budget_granted": ("run", "Run granted its budget"),
    "budget_consumed": ("run", "Run consumption recorded"),
    "budget_exceeded": ("budget", "Budget cap tripped"),
    "budget_amended": ("budget", "Budget amended"),
    "subagent_spawned": ("execution", "Sub-agent spawned"),
    "plan_generated": ("planning", "Plan generated"),
    "planning_budget_exceeded": ("planning", "Planning stopped on its budget"),
    "plan_approved": ("plan_gate", "Plan approved"),
    "plan_rejected": ("plan_gate", "Plan rejected"),
    "plan_revision_requested": ("planning", "Plan revision requested"),
    "execution_started": ("execution", "Execution started"),
    "execution_resumed": ("execution", "Execution resumed"),
    "execution_interruption_detected": ("execution", "Interruption detected"),
    "workstream_started": ("workstream", "Workstream started"),
    "workstream_handoff": ("workstream", "Findings handed off"),
    "workstream_finished": ("workstream", "Workstream finished"),
    "workstream_run_failed": ("workstream", "Workstream run failed"),
    "findings_reused": ("execution", "Findings carried over (reuse rule)"),
    "report_snapshotted": ("composition", "Report snapshot preserved"),
    "report_composed": ("composition", "Report composed"),
    "verification_completed": ("verification", "Verification completed"),
    "claims_quarantined": ("verification", "Claims quarantined"),
    "conflicts_detected": ("verification", "Conflicts detected"),
    "conflict_noted": ("decision", "Conflict noted"),
    "challenges_raised": ("challenge", "Challenges raised"),
    "challenge_rebutted": ("challenge", "Challenge rebutted"),
    "challenge_revised": ("challenge", "Challenge revised (claim sent back)"),
    "challenge_downgraded": ("challenge", "Challenge downgraded (claim quarantined)"),
    "claim_sent_back": ("decision", "Claim sent back for re-investigation"),
    "claim_dropped": ("decision", "Claim dropped"),
    "evidence_drift_detected": ("report_gate", "Evidence drift detected"),
    "report_claims_flagged_redaction": ("report_gate", "Claims flagged for redaction"),
    "redaction_claim_acknowledged": ("decision", "Redaction flag acknowledged"),
    "redaction_claim_sent_back": ("decision", "Redacted claim sent back"),
    "evidence_redacted": ("governance", "Evidence redacted"),
    "workstreams_marked_stale": ("governance", "Workstreams marked stale"),
    "staleness_proceeded": ("governance", "Staleness proceeded on (recorded)"),
    "report_approved": ("report_gate", "Report approved"),
    "report_rejected": ("report_gate", "Report rejected"),
    "case_note_added": ("note", "Analyst note added"),
    "report_section_note_added": ("note", "Report section note added"),
    "report_section_note_edited": ("note", "Report section note edited"),
    "case_assigned": ("triage", "Case assigned"),
    "case_priority_set": ("triage", "Priority set"),
    "case_assigned_to_engagement": ("triage", "Case assigned to an engagement"),
    "case_removed_from_engagement": ("triage", "Case removed from its engagement"),
    "legal_hold_set": ("governance", "Legal hold set"),
    "legal_hold_cleared": ("governance", "Legal hold cleared"),
    "source_registered": ("governance", "Source registered"),
    "case_deleted": ("governance", "Case deleted (tombstone written)"),
}


def _act_detail(payload: dict) -> dict:
    """An act's detail: the payload's own facts, kept readable —
    scalars ride as they are, dicts of scalars ride whole (caps,
    usage), lists of scalars ride when short, and anything larger
    is counted rather than dumped. Nothing is summarised into
    words the record does not contain."""
    detail: dict = {}
    for key, value in (payload or {}).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            detail[key] = value
        elif isinstance(value, dict) and all(
            isinstance(v, (str, int, float, bool)) or v is None
            for v in value.values()
        ):
            detail[key] = value
        elif isinstance(value, list) and all(
            isinstance(v, (str, int, float, bool)) for v in value
        ):
            detail[key] = value if len(value) <= 12 else {"count": len(value)}
        elif isinstance(value, list):
            detail[key] = {"count": len(value)}
        else:
            detail[key] = {"recorded": True}
    return detail


def _fold_runs(audit: list[dict]) -> list[dict]:
    """The phase runs, folded from the ledger events in order:
    each `budget_granted` opens a run, the guard's exhaustion and
    the run-end cost entry attach to it, and `budget_consumed`
    closes it with the run's usage. An execution run is stamped
    with its attempt — the count of `execution_started` events
    that precede it — so re-executions and resumes read as the
    separate attempts they were."""
    runs: list[dict] = []
    open_by_phase: dict[str, dict] = {}
    execution_attempts = 0
    for entry in audit:
        event = entry["event"]
        payload = entry.get("payload") or {}
        if event == "execution_started":
            execution_attempts += 1
        elif event == "budget_granted":
            phase = payload.get("phase")
            run = {
                "index": len(runs) + 1,
                "phase": phase,
                "attempt": execution_attempts if phase == "execution" else None,
                "granted_at": entry["ts"],
                "caps": payload.get("caps"),
                "case_caps": payload.get("case_caps"),
                "profile": payload.get("profile"),
                "consumed": None,
                "consumed_at": None,
                "exceeded": None,
                "cost": None,
            }
            runs.append(run)
            open_by_phase[phase] = run
        elif event == "budget_exceeded":
            run = open_by_phase.get(payload.get("phase"))
            if run is not None:
                run["exceeded"] = {
                    "kind": payload.get("kind"),
                    "limit": payload.get("limit"),
                    "observed": payload.get("observed"),
                }
        elif event == "cost_recorded":
            run = open_by_phase.get(payload.get("phase"))
            if run is not None:
                lines = payload.get("lines") or []
                total = sum(
                    Decimal(line["cost_usd"] or "0")
                    for line in lines
                    if line.get("priced")
                )
                run["cost"] = {
                    "total_usd": str(total),
                    "lines": lines,
                    "price_table": payload.get("price_table"),
                }
        elif event == "budget_consumed":
            run = open_by_phase.pop(payload.get("phase"), None)
            if run is not None:
                run["consumed"] = payload.get("usage")
                run["consumed_at"] = entry["ts"]
    return runs


def build_trace(store: Store, case_id: str) -> dict:
    """One case's run, reconstructed end to end from its stored
    records by this one function: the header (the case as it
    stands, its outcome under the taxonomy), the phase runs
    folded from the budget and cost ledgers, and the act sequence
    — every audit entry, in the trail's causal order, mapped to
    its act kind and labelled. The trace stores nothing and
    rewrites nothing; building it leaves the case's records
    exactly as they were."""
    case = store.get_case(case_id)
    if case is None:
        raise KeyError(case_id)
    audit = store.list_audit(case_id)
    report = store.get_report(case_id)
    verifier = store.get_verifier_report(case_id)
    findings = store.list_findings(case_id)
    acts = []
    for seq, entry in enumerate(audit):
        kind, label = _ACT_KINDS.get(
            entry["event"],
            ("event", entry["event"].replace("_", " ").capitalize()),
        )
        acts.append(
            {
                "seq": seq,
                "ts": entry["ts"],
                "kind": kind,
                "event": entry["event"],
                "label": label,
                "detail": _act_detail(entry.get("payload") or {}),
            }
        )
    findings_by_question: dict[str, int] = {}
    for finding in findings:
        qid = finding["question_id"]
        findings_by_question[qid] = findings_by_question.get(qid, 0) + 1
    challenges = store.list_challenges(case_id)
    cost = fold_cost_ledger(audit)
    return {
        "case_id": case_id,
        "subject": case["subject"],
        "status": case["status"],
        "created_at": case["created_at"],
        "updated_at": case["updated_at"],
        "profile": case.get("profile") or {},
        "outcome": classify_case_outcome(store, case_id),
        "counts": {
            "findings": len(findings),
            "findings_by_question": findings_by_question,
            "claims": len((report or {}).get("claims") or []),
            "quarantined_standing": sum(
                1
                for r in store.list_quarantine(case_id)
                if r["status"] == "QUARANTINED"
            ),
            "conflicts": len(store.list_conflicts(case_id)),
            "challenges_open": sum(
                1 for c in challenges if c["status"] == "OPEN"
            ),
            "challenges_total": len(challenges),
        },
        "verification": (
            {
                "run_id": verifier.get("run_id"),
                "checked": verifier.get("checked"),
                "supported": verifier.get("supported"),
                "quarantined": verifier.get("quarantined"),
                "by_reason": verifier.get("by_reason"),
                "verified_at": verifier.get("verified_at"),
            }
            if verifier
            else None
        ),
        "cost": {
            "total_usd": str(cost["totals"]["cost_usd"]),
            "runs_recorded": cost["runs_recorded"],
            "contains_estimates": cost["contains_estimates"],
        },
        "runs": _fold_runs(audit),
        "acts": acts,
        "timing": {
            "basis": (
                "audit-trail timestamps (UTC, second resolution); "
                "act order is the store's insertion order, which "
                "is the order the acts happened"
            ),
            "run_wall_seconds_recorded": True,
            "per_act_durations_recorded": False,
            "note": (
                "Each run's usage — steps, tokens, spawns, wall "
                "seconds — was recorded by the budget guard and "
                "rides the runs above. Per-act, per-workstream, "
                "and per-tool-call durations were never recorded, "
                "so the trace shows none rather than estimating "
                "them from the timestamps."
            ),
        },
    }


# ------------------------------------------------------------------ digest

_GATE_EVENTS = ("plan_approved", "plan_rejected", "report_approved", "report_rejected")
_CHALLENGE_ANSWER_EVENTS = (
    "challenge_rebutted",
    "challenge_revised",
    "challenge_downgraded",
)


def _day_of(ts: str) -> str:
    return (ts or "")[:10]


def fold_digest(store: Store, day: str, now: str | None = None) -> dict:
    """One day's operations fold, recomputed from the records at
    generation time — the digest the service stores is this fold
    plus its snapshot stamp, and nothing in it is a stored
    counter.

    Scope, stated plainly: a case belongs to the day it was
    OPENED (created) for the opened list, the status counts, and
    the taxonomy counts. Events belong to the day of their audit
    timestamp, whichever case they ride: gate decisions, the
    cost lines (each priced line rides its run's `cost_recorded`
    entry, so the day's cost is the day its runs ended), partial
    compositions, and challenge answers. A partial report is
    listed when it was composed that day and still stands — a
    case amended and re-run the same day no longer stands
    partial, and the digest says so by omission; its amendment
    is in the day's trail either way. Open must-answer
    challenges are desk-wide and current — the digest is taken
    now, so their ages run to now — because they are the desk's
    live exposure, not the day's history."""
    now_dt = datetime.fromisoformat(now or utcnow())
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    cases = store.list_cases()
    opened = [c for c in cases if _day_of(c["created_at"]) == day]
    gate_decisions: Counter = Counter()
    answered = 0
    closed_ids: list[str] = []
    cost_by_case: dict[str, Decimal] = {}
    cost_runs = 0
    cost_estimated = False
    partials: list[dict] = []
    for case in cases:
        case_id = case["case_id"]
        audit = store.list_audit(case_id)
        partial_composed_at: str | None = None
        for entry in audit:
            if _day_of(entry["ts"]) != day:
                continue
            event = entry["event"]
            payload = entry.get("payload") or {}
            if event in _GATE_EVENTS:
                gate_decisions[event] += 1
                if event == "report_approved" and case_id not in closed_ids:
                    closed_ids.append(case_id)
            elif event in _CHALLENGE_ANSWER_EVENTS:
                answered += 1
            elif event == "cost_recorded":
                cost_runs += 1
                total = Decimal(0)
                for line in payload.get("lines") or []:
                    if line.get("priced"):
                        total += Decimal(line["cost_usd"] or "0")
                    if line.get("usage_basis") != "measured":
                        cost_estimated = True
                if total:
                    cost_by_case[case_id] = (
                        cost_by_case.get(case_id, Decimal(0)) + total
                    )
            elif event == "report_composed" and payload.get("partial"):
                partial_composed_at = entry["ts"]
        if partial_composed_at is not None:
            report = store.get_report(case_id)
            if report and report.get("partial"):
                split = _coverage_split(report.get("coverage_detail"))
                partials.append(
                    {
                        "case_id": case_id,
                        "subject": case["subject"],
                        "composed_at": partial_composed_at,
                        "exceeded_kind": report.get("exceeded_kind"),
                        "exceeded_phase": report.get("exceeded_phase"),
                        "uncovered": split["uncovered"],
                        "partial": split["partial"],
                    }
                )
    taxonomy = Counter(
        classify_case_outcome(store, c["case_id"])["label"] for c in opened
    )
    raised = 0
    open_must_answer: list[dict] = []
    for case in cases:
        for challenge in store.list_challenges(case["case_id"]):
            if _day_of(challenge.get("created_at") or "") == day:
                raised += 1
            if (
                challenge["status"] == "OPEN"
                and challenge["severity"] == "must_answer"
            ):
                created = datetime.fromisoformat(challenge["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                open_must_answer.append(
                    {
                        "case_id": case["case_id"],
                        "challenge_id": challenge["challenge_id"],
                        "rule": challenge["rule"],
                        "claim_id": challenge.get("claim_id"),
                        "created_at": challenge["created_at"],
                        "age_seconds": int(
                            (now_dt - created).total_seconds()
                        ),
                    }
                )
    open_must_answer.sort(key=lambda c: c["created_at"])
    total_cost = sum(cost_by_case.values(), Decimal(0))
    return {
        "day": day,
        "cases": {
            "opened": [c["case_id"] for c in opened],
            "opened_count": len(opened),
            "closed_count": len(closed_ids),
            "closed": closed_ids,
            "by_status": dict(
                sorted(Counter(c["status"] for c in opened).items())
            ),
        },
        "gate_decisions": {
            event: gate_decisions.get(event, 0) for event in _GATE_EVENTS
        },
        "partial_reports": partials,
        "cost": {
            "total_usd": str(total_cost),
            "by_case": [
                {"case_id": case_id, "cost_usd": str(amount)}
                for case_id, amount in sorted(cost_by_case.items())
            ],
            "runs_recorded": cost_runs,
            "contains_estimates": cost_estimated,
        },
        "taxonomy": dict(sorted(taxonomy.items())),
        "challenges": {
            "raised": raised,
            "answered": answered,
            "open_must_answer": open_must_answer,
        },
    }
