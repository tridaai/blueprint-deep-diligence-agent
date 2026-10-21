"""The report composer — deterministic code, not a model.

It assembles the report draft from recorded findings, grouped by
question. A finding whose citations no longer resolve to case evidence
is refused here a second time (record_finding already enforces it at
write time) and named in the draft's method note. Every claim line
carries its evidence IDs; a question with no findings says so plainly.
A partial draft (budget exhausted) follows the partial-report
protocol: its banner names the cap and the phase that stopped, and
coverage is itemised per question — ANSWERED, PARTIAL (findings
exist but the run stopped mid-workstream), or NOT COVERED — derived
in code by `question_coverage` from the claims and the plan's
workstream mapping, and carried on the report record so the
analyst's approval records the coverage it approved. A completed
run whose workstreams did not all complete — one failed, one ran
empty — carries the same table outside the partial protocol, with
the reason named on each uncovered row: a failed branch is visible
in the report, never only in the audit trail.

Claims are composed in plan order (the brief's question order,
then fact ref), never in the order sub-agents happened to record
them, so the report is a function of the finding set: repeated
executions of the same case produce the same draft.

Composition is also when citations are *made*: each composed claim
gets a claim ID (CLM-0001, ...) and a citation-ledger entry recording
the hash of every cited evidence item's bytes at this moment (see
ledger.py). The report's References section is generated from that
ledger — the model never writes it.

After composition, the verifier (verifier.py) re-checks every claim
against its evidence; `apply_verification` then re-renders the draft
from the supported claims only — quarantined claims leave the body,
the ledger is rebuilt to match, and a Verification section records
the outcome in the draft itself.

Finally, `apply_conflicts` re-renders the draft with its Conflicts
section, generated from the conflict records (conflicts.py): where
supported claims assert different values for the same (entity,
metric, period), both sides are stated with their values verbatim
and their source labels. The section is generated from the records,
never hand-written, and a conflict the analyst has marked NOTED
renders with its note — it is never removed.

`apply_conflicts` is also the finalize step (report_model.py): it
builds the report's structured model — executive summary composed
from the supported claims under the summary-integrity check,
per-question support grades, risks and open items generated from
the case's records — and renders all three formats from that one
model: this markdown, a self-contained print-ready HTML page, and
the JSON export. Later re-renders (a conflict noted, staleness
proceeded on, a quarantined claim dropped, a redaction flag
acknowledged) pass through the same step, so the model and its
renderings always describe the records as they now stand.

The Method and limits section also carries the case's staleness
decisions (continuity.py): when evidence arrived after planning
and an analyst proceeded without re-planning, or re-planned around
it, the report says so — which evidence, which questions, whose
decision — because part of the draft then predates evidence the
reader should know about.
"""

from __future__ import annotations

from .ledger import build_ledger
from .objects import ObjectStore
from .report_model import (
    DRAFT_STATUS_LINE,
    build_report_model,
    render_html,
    skill_lines,
    staleness_lines,
)
from .store import Store, utcnow


def question_coverage(
    brief: dict,
    claims: list[dict],
    interrupted_questions: list[str] | tuple[str, ...] = (),
    workstream_status: list[dict] | None = None,
) -> list[dict]:
    """Coverage per brief question, derived in code from the
    report's claims and the plan's workstream mapping (the caller
    resolves which questions the stopped run was inside of, from
    the plan's workstream order and the spawns still open when the
    cap tripped):

    - ANSWERED — findings for the question are in the report and
      its workstream was not the one interrupted.
    - PARTIAL — findings exist, but the run stopped mid-workstream,
      so the question's investigation is incomplete.
    - NOT COVERED — no findings recorded for the question. When the
      execution record explains why — the workstream failed, ran
      empty, was interrupted, or never ran — the row carries that
      reason, so an uncovered question never reads as a mystery.
    """
    counts = _coverage(brief, claims)
    interrupted = set(interrupted_questions or ())
    statuses = {
        s["question_id"]: s for s in (workstream_status or [])
    }
    detail = []
    for question in brief["questions"]:
        found = counts.get(question["id"], 0)
        if found == 0:
            status = "NOT COVERED"
        elif question["id"] in interrupted:
            status = "PARTIAL"
        else:
            status = "ANSWERED"
        row = {"question_id": question["id"], "status": status, "findings": found}
        if status == "NOT COVERED":
            record = statuses.get(question["id"])
            if record is not None and record.get("reason"):
                row["reason"] = record["reason"]
        detail.append(row)
    return detail


def coverage_detail_for(
    brief: dict,
    claims: list[dict],
    *,
    partial: bool,
    interrupted_questions: list[str] | tuple[str, ...] = (),
    workstream_status: list[dict] | None = None,
) -> list[dict] | None:
    """The coverage table a report owes its analyst, or None.

    Partial reports always carry one (the R5 protocol). A completed
    run carries one when the execution record is not clean — any
    workstream failed, empty, interrupted, or never ran — so the
    failure is named in the report, not only in the audit trail. A
    clean run carries none, exactly as reports always have.
    """
    statuses = workstream_status or []
    if not partial and all(s["status"] == "completed" for s in statuses):
        return None
    return question_coverage(brief, claims, interrupted_questions, statuses)


def _summary_lines(summary: dict) -> list[str]:
    counts = summary["counts"]
    lines = [
        "## Executive summary",
        "",
        f"Supported claims: {counts['claims_supported']} · "
        f"Quarantined claims: {counts['claims_quarantined']} · "
        f"Conflicts open: {counts['conflicts_open']} · "
        f"Conflicts noted: {counts['conflicts_noted']} · "
        f"Questions not covered: {counts['questions_uncovered']} · "
        f"Questions partial: {counts['questions_partial']}",
        "",
    ]
    if summary["sentences"]:
        lines += [f"- {sentence}" for sentence in summary["sentences"]]
    elif counts["claims_supported"]:
        lines.append(
            "_Every summary sentence failed the integrity check "
            "against the supported claims and was dropped; the drops "
            "are recorded on the report record._"
        )
    else:
        lines.append(
            "_No findings survived verification; the summary has "
            "nothing to summarise._"
        )
    lines.append("")
    return lines


def _open_items_lines(items: dict) -> list[str]:
    if items["empty"]:
        return [
            "None recorded: no open or noted conflicts, no quarantined "
            "claims, no uncovered or partial questions, no staleness "
            "proceeded on, and no redaction flags.",
            "",
        ]
    lines: list[str] = []
    if items["conflicts"]:
        lines += ["### Conflicts", ""]
        for record in items["conflicts"]:
            period = record["period"] if record["period"] else "no period stated"
            line = (
                f"- {record['conflict_id']} — {record['metric']} · "
                f"{period}: {' vs '.join(record['values'])} "
                f"({record['status']}"
            )
            if record["status"] == "NOTED":
                line += f" by {record['noted_by']}: \"{record['note']}\""
            lines.append(line + ")")
        lines.append("")
    if items["quarantined"]:
        lines += ["### Quarantined claims", ""]
        for record in items["quarantined"]:
            line = (
                f"- {record['claim_id']} ({record['question_id']}) — "
                f"{record['reason']}; status {record['status']}"
            )
            if record["decided_by"]:
                line += f", decided by {record['decided_by']}"
            lines.append(line)
        lines.append("")
    if items["coverage_gaps"]:
        lines += ["### Coverage gaps", ""]
        for row in items["coverage_gaps"]:
            line = (
                f"- {row['question_id']} — {row['status']} "
                f"({row['findings']} finding(s) in this report)"
            )
            if row.get("reason"):
                line += f" — {row['reason']}"
            lines.append(line)
        lines.append("")
    if items["staleness_proceeds"]:
        lines += ["### Staleness proceeded on", ""]
        for decision in items["staleness_proceeds"]:
            line = (
                f"- Evidence {', '.join(decision['evidence_ids'])}; "
                f"questions {', '.join(decision['questions'])} — "
                f"proceeded by {decision['analyst']} on "
                f"{(decision['at'] or '')[:10]}"
            )
            if decision["note"]:
                line += f": \"{decision['note']}\""
            lines.append(line)
        lines.append("")
    if items["redaction_flags"]:
        lines += ["### Redaction flags", ""]
        for flag in items["redaction_flags"]:
            line = (
                f"- {flag['claim_id']} cites {flag['evidence_id']} — "
                f"{flag['status']}"
            )
            if flag["decided_by"]:
                line += f" by {flag['decided_by']}"
            lines.append(line)
        lines.append("")
    return lines


def _render_markdown(
    case: dict,
    claims: list[dict],
    evidence_count: int,
    entries: list[dict],
    *,
    partial: bool = False,
    exceeded_kind: str | None = None,
    exceeded_phase: str | None = None,
    coverage_detail: list[dict] | None = None,
    refused: list[str] | None = None,
    verification: dict | None = None,
    conflicts: list[dict] | None = None,
    staleness_decisions: list[dict] | None = None,
    skill_notes: list[str] | None = None,
    summary: dict | None = None,
    grades: dict | None = None,
    open_items: dict | None = None,
) -> str:
    brief = case["brief"]
    lines = [
        f"# Diligence report — {brief['subject']}",
        "",
        DRAFT_STATUS_LINE,
        f"Commissioned by: {brief['analyst']}. "
        f"Evidence in scope: {evidence_count} documents.",
        "",
    ]
    if summary is not None:
        lines += _summary_lines(summary)
    if partial:
        lines += [
            "> **Partial report.** The run stopped on its "
            f"{exceeded_kind or 'budget'} budget during the "
            f"{exceeded_phase or 'execution'} phase, before the plan "
            "was fully executed. Findings below are what was "
            "recorded before the stop; coverage is itemised per "
            "question — treat anything not ANSWERED as not fully "
            "investigated.",
            "",
        ]
    if coverage_detail:
        lines += ["Coverage by question:", ""]
        for row in coverage_detail:
            line = (
                f"- {row['question_id']} — {row['status']} "
                f"({row['findings']} finding(s) in this report)"
            )
            if row["status"] == "PARTIAL":
                line += " — the run stopped inside this workstream"
            elif row["status"] == "NOT COVERED":
                if row.get("reason"):
                    line += f" — {row['reason']}"
                else:
                    line += " — not investigated"
            lines.append(line)
        lines.append("")
    for question in brief["questions"]:
        header = f"## {question['id']} — {question['text']}"
        if grades and question["id"] in grades:
            header += f" · support: {grades[question['id']]}"
        lines += [header, ""]
        q_claims = [c for c in claims if c["question_id"] == question["id"]]
        if not q_claims:
            lines += ["_No findings recorded for this question._", ""]
        for claim in q_claims:
            citations = " ".join(f"[{eid}]" for eid in claim["evidence_ids"])
            lines.append(f"- {claim['claim']} {citations}")
        lines.append("")
    if conflicts:
        lines += [
            "## Conflicts",
            "",
            "The supported findings disagree on the items below. "
            "Both sides are reported, with the values as stated and "
            "the source each side rests on; the desk takes no "
            "position on which value is correct, and neither side "
            "has been suppressed or merged.",
            "",
        ]
        for record in conflicts:
            key = record["key"]
            period = key["period"] if key["period"] else "no period stated"
            lines.append(
                f"### {record['conflict_id']} — {key['metric']} · "
                f"{period} · {key['entity']}"
            )
            lines.append("")
            for side in record["sides"]:
                lines.append(f"- **{side['value_text']}**")
                for member in side["claims"]:
                    citations = " ".join(
                        f"[{eid}]" for eid in member["evidence_ids"]
                    )
                    lines.append(
                        f'  - {member["claim_id"]}: "{member["claim"]}" '
                        f"{citations}"
                    )
                    for source in member["sources"]:
                        context = (
                            f"Source: {source['source']}, dated "
                            f"{source['doc_date']} · "
                            f"{source['source_class']} source"
                        )
                        if member["more_recent"]:
                            context += " · more recent"
                        lines.append(f"    {context}")
            if record["status"] == "NOTED":
                lines.append(
                    f"  - Analyst note — marked NOTED by "
                    f"{record['noted_by']} on {record['noted_at']}: "
                    f'"{record["note"]}" The conflict remains part of '
                    "this report."
                )
            lines.append("")
    # References are generated from the citation ledger: the evidence
    # the report actually cites, each with the content hash recorded
    # at citation time, so a reader can check a source is unchanged.
    cited: dict[str, dict] = {}
    for entry in entries:
        for citation in entry["citations"]:
            slot = cited.setdefault(
                citation["evidence_id"], {**citation, "cited_by": []}
            )
            slot["cited_by"].append(entry["claim_id"])
    lines += ["## References", ""]
    if cited:
        for evidence_id in sorted(cited):
            citation = cited[evidence_id]
            lines.append(
                f"- [{evidence_id}] {citation['source']} — {citation['title']} "
                f"· sha256 {citation['sha256'][:12]}… "
                f"· cited by {', '.join(citation['cited_by'])}"
            )
    else:
        lines.append("_No cited evidence._")
    if verification is not None:
        lines += ["", "## Verification", ""]
        summary = (
            f"An independent verifier re-read every claim in this draft "
            f"against the evidence it cites — fetched from the evidence "
            f"store, not from the composer's working set. "
            f"{verification['checked']} claim(s) checked: "
            f"{verification['supported']} supported, "
            f"{verification['quarantined']} quarantined."
        )
        if verification["quarantined"]:
            reasons = ", ".join(
                f"{reason} ×{count}"
                for reason, count in sorted(verification["by_reason"].items())
            )
            summary += (
                f" Quarantined by reason: {reasons}. Quarantined claims "
                "are excluded from this report and held in the case's "
                "quarantine record; an analyst can send one back for "
                "re-investigation or drop it, but can never approve it "
                "into the report."
            )
        lines.append(summary)
    if open_items is not None:
        lines += ["", "## Risks and open items", ""]
        lines += _open_items_lines(open_items)
    lines += [
        "",
        "## Method and limits",
        "",
        "Assembled by the Deep Diligence Agent from the case evidence "
        "only. Each claim cites the evidence it rests on; the desk's "
        "decision is not part of this document and no recommendation "
        "is made.",
    ]
    if refused:
        lines.append(
            f"{len(refused)} recorded claim(s) were refused at "
            "composition because their evidence citations did not "
            "resolve: " + ", ".join(refused) + "."
        )
    lines += staleness_lines(staleness_decisions)
    lines += skill_notes or []
    return "\n".join(lines) + "\n"


def _coverage(brief: dict, claims: list[dict]) -> dict:
    coverage = {q["id"]: 0 for q in brief["questions"]}
    for claim in claims:
        coverage[claim["question_id"]] = coverage.get(claim["question_id"], 0) + 1
    return coverage


def _plan_order(brief: dict, findings: list[dict]) -> list[dict]:
    """Composition order: the brief's question order, then fact
    ref, then claim text — a pure function of the finding set.

    Findings are stored in the order sub-agents recorded them,
    which under parallel workstreams is completion order and can
    differ between two executions of the same case. Composing in
    plan order makes the report a function of WHAT was found, not
    of who finished first: repeated executions of the same case
    produce the same report.
    """
    question_index = {
        q["id"]: i for i, q in enumerate(brief["questions"])
    }
    return sorted(
        findings,
        key=lambda f: (
            question_index.get(f["question_id"], len(question_index)),
            f.get("fact_ref") or "",
            f["claim"],
            f["finding_id"],
        ),
    )


def compose_report(
    store: Store,
    objects: ObjectStore,
    case: dict,
    *,
    partial: bool = False,
    exceeded_kind: str | None = None,
    exceeded_phase: str | None = None,
    interrupted_questions: list[str] | tuple[str, ...] = (),
    workstream_status: list[dict] | None = None,
) -> dict:
    case_id = case["case_id"]
    brief = case["brief"]
    evidence = {ev["evidence_id"]: ev for ev in store.list_evidence(case_id)}
    claims: list[dict] = []
    refused: list[str] = []
    for finding in _plan_order(brief, store.list_findings(case_id)):
        valid = [e for e in finding["evidence_ids"] if e in evidence]
        if not valid:
            refused.append(finding["finding_id"])
            continue
        claims.append(
            {
                **finding,
                "claim_id": f"CLM-{len(claims) + 1:04d}",
                "evidence_ids": valid,
            }
        )
    entries = build_ledger(store, objects, case_id, claims)
    coverage_detail = coverage_detail_for(
        brief,
        claims,
        partial=partial,
        interrupted_questions=interrupted_questions,
        workstream_status=workstream_status,
    )
    report = {
        "case_id": case_id,
        "status": "DRAFT",
        "partial": partial,
        "exceeded_kind": exceeded_kind,
        "exceeded_phase": exceeded_phase,
        "interrupted_questions": list(interrupted_questions),
        "workstreams": list(workstream_status or []),
        "coverage_detail": coverage_detail,
        "approved_coverage": None,
        "markdown": _render_markdown(
            case,
            claims,
            len(evidence),
            entries,
            partial=partial,
            exceeded_kind=exceeded_kind,
            exceeded_phase=exceeded_phase,
            coverage_detail=coverage_detail,
            refused=refused,
            staleness_decisions=case.get("staleness_decisions"),
            skill_notes=skill_lines(store, case),
        ),
        "claims": claims,
        "refused_claims": refused,
        "coverage": _coverage(brief, claims),
        "ledger": {
            "entries": len(entries),
            "citations": sum(len(e["citations"]) for e in entries),
            "built_at": utcnow(),
        },
        "verification": None,
        "quarantined": [],
        "conflicts": [],
        "integrity": {"status": "OK", "checked_at": utcnow()},
        "composed_at": utcnow(),
        "decided_by": None,
        "decision_note": None,
    }
    return store.save_report(case_id, report)


def apply_verification(
    store: Store,
    objects: ObjectStore,
    case: dict,
    report: dict,
    verifier_report: dict,
    quarantine_records: list[dict],
) -> dict:
    """Re-render a composed draft from its supported claims only.

    Quarantined claims leave the report body; the citation ledger is
    rebuilt over the claims that remain, so the ledger and the
    References section describe the report the analyst actually sees.
    The verifier summary and the quarantined claims (with reasons)
    ride on the report record.
    """
    case_id = case["case_id"]
    brief = case["brief"]
    quarantined_ids = {r["claim_id"] for r in quarantine_records}
    claims = [c for c in report["claims"] if c["claim_id"] not in quarantined_ids]
    entries = build_ledger(store, objects, case_id, claims)
    summary = {
        "run_id": verifier_report["run_id"],
        "mode": verifier_report["mode"],
        "model": verifier_report["model"],
        "checked": verifier_report["checked"],
        "supported": verifier_report["supported"],
        "quarantined": verifier_report["quarantined"],
        "by_reason": verifier_report["by_reason"],
        "verified_at": verifier_report["verified_at"],
    }
    evidence_count = store.count_evidence(case_id)
    coverage_detail = coverage_detail_for(
        brief,
        claims,
        partial=report["partial"],
        interrupted_questions=report.get("interrupted_questions") or (),
        workstream_status=report.get("workstreams"),
    )
    report = {
        **report,
        "markdown": _render_markdown(
            case,
            claims,
            evidence_count,
            entries,
            partial=report["partial"],
            exceeded_kind=report["exceeded_kind"],
            exceeded_phase=report.get("exceeded_phase"),
            coverage_detail=coverage_detail,
            refused=report["refused_claims"],
            verification=summary,
            staleness_decisions=case.get("staleness_decisions"),
            skill_notes=skill_lines(store, case),
        ),
        "claims": claims,
        "coverage": _coverage(brief, claims),
        "coverage_detail": coverage_detail,
        "ledger": {
            "entries": len(entries),
            "citations": sum(len(e["citations"]) for e in entries),
            "built_at": utcnow(),
        },
        "verification": summary,
        "quarantined": [
            {
                "quarantine_id": r["quarantine_id"],
                "claim_id": r["claim_id"],
                "finding_id": r["finding_id"],
                "question_id": r["question_id"],
                "claim": r["claim"],
                "evidence_ids": r["evidence_ids"],
                "reason": r["reason"],
            }
            for r in quarantine_records
        ],
    }
    return store.save_report(case_id, report)


def apply_conflicts(
    store: Store,
    objects: ObjectStore,
    case: dict,
    report: dict,
    conflict_records: list[dict],
) -> dict:
    """Finalize a verified draft: its Conflicts section, its
    structured model, and every format rendered from that model.

    The Conflicts section is generated from the conflict records,
    so the report the analyst reads and the stored records cannot
    drift apart. Called once after conflict detection, and again
    whenever a later decision changes what the report must show —
    a conflict marked NOTED, staleness proceeded on, a quarantined
    claim dropped, a redaction flag acknowledged — so the note and
    the decision render into the draft and stay there. Claims,
    coverage, verification, and the citation ledger are unchanged
    by this step: it assembles the model (report_model.py) from
    the records as they stand and renders the markdown, the
    print-ready HTML, and the stored model the JSON export serves
    — one model, three formats, nothing assembled twice.
    """
    case_id = case["case_id"]
    claims = report["claims"]
    entries = build_ledger(store, objects, case_id, claims)
    evidence_count = store.count_evidence(case_id)
    model = build_report_model(store, case, report, entries, conflict_records)
    grades = {q["question_id"]: q["grade"] for q in model["questions"]}
    report = {
        **report,
        "markdown": _render_markdown(
            case,
            claims,
            evidence_count,
            entries,
            partial=report["partial"],
            exceeded_kind=report["exceeded_kind"],
            exceeded_phase=report.get("exceeded_phase"),
            coverage_detail=report.get("coverage_detail"),
            refused=report["refused_claims"],
            verification=report["verification"],
            conflicts=conflict_records,
            staleness_decisions=case.get("staleness_decisions"),
            skill_notes=skill_lines(store, case),
            summary=model["executive_summary"],
            grades=grades,
            open_items=model["risks_and_open_items"],
        ),
        "conflicts": [
            {
                "conflict_id": r["conflict_id"],
                "key": r["key"],
                "values": [side["value_text"] for side in r["sides"]],
                "status": r["status"],
                "noted_by": r["noted_by"],
                "note": r["note"],
            }
            for r in conflict_records
        ],
        "model": model,
        "html": render_html(model),
    }
    return store.save_report(case_id, report)
