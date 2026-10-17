"""The report composer — deterministic code, not a model.

It assembles the report draft from recorded findings, grouped by
question. A finding whose citations no longer resolve to case evidence
is refused here a second time (record_finding already enforces it at
write time) and named in the draft's method note. Every claim line
carries its evidence IDs; a question with no findings says so plainly;
a partial draft (budget exhausted) names the uncovered questions
instead of quietly omitting them.

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
"""

from __future__ import annotations

from .ledger import build_ledger
from .objects import ObjectStore
from .store import Store, utcnow


def _render_markdown(
    case: dict,
    claims: list[dict],
    evidence_count: int,
    entries: list[dict],
    *,
    partial: bool = False,
    exceeded_kind: str | None = None,
    refused: list[str] | None = None,
    verification: dict | None = None,
    conflicts: list[dict] | None = None,
) -> str:
    brief = case["brief"]
    coverage = {q["id"]: 0 for q in brief["questions"]}
    for claim in claims:
        coverage[claim["question_id"]] = coverage.get(claim["question_id"], 0) + 1
    lines = [
        f"# Diligence report — {brief['subject']}",
        "",
        "Status: DRAFT — not final until an analyst approves it.",
        f"Commissioned by: {brief['analyst']}. "
        f"Evidence in scope: {evidence_count} documents.",
        "",
    ]
    if partial:
        uncovered = [qid for qid, n in coverage.items() if n == 0]
        lines += [
            "> **Partial report.** The run stopped on its "
            f"{exceeded_kind or 'budget'} budget before the plan was fully "
            "executed. Findings below are what was recorded before the "
            "stop; treat uncovered questions as not investigated.",
            "",
        ]
        if uncovered:
            lines += [
                "Questions with no findings recorded: "
                + ", ".join(uncovered)
                + ".",
                "",
            ]
    for question in brief["questions"]:
        lines += [f"## {question['id']} — {question['text']}", ""]
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
    return "\n".join(lines) + "\n"


def _coverage(brief: dict, claims: list[dict]) -> dict:
    coverage = {q["id"]: 0 for q in brief["questions"]}
    for claim in claims:
        coverage[claim["question_id"]] = coverage.get(claim["question_id"], 0) + 1
    return coverage


def compose_report(
    store: Store,
    objects: ObjectStore,
    case: dict,
    *,
    partial: bool = False,
    exceeded_kind: str | None = None,
) -> dict:
    case_id = case["case_id"]
    brief = case["brief"]
    evidence = {ev["evidence_id"]: ev for ev in store.list_evidence(case_id)}
    claims: list[dict] = []
    refused: list[str] = []
    for finding in store.list_findings(case_id):
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
    report = {
        "case_id": case_id,
        "status": "DRAFT",
        "partial": partial,
        "exceeded_kind": exceeded_kind,
        "markdown": _render_markdown(
            case,
            claims,
            len(evidence),
            entries,
            partial=partial,
            exceeded_kind=exceeded_kind,
            refused=refused,
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
    report = {
        **report,
        "markdown": _render_markdown(
            case,
            claims,
            evidence_count,
            entries,
            partial=report["partial"],
            exceeded_kind=report["exceeded_kind"],
            refused=report["refused_claims"],
            verification=summary,
        ),
        "claims": claims,
        "coverage": _coverage(brief, claims),
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
    """Re-render a verified draft with its Conflicts section.

    The section is generated from the conflict records, so the
    report the analyst reads and the stored records cannot drift
    apart. Called once after conflict detection, and again whenever
    a conflict is marked NOTED, so the note renders into the draft
    and stays there. Claims, coverage, verification, and the
    citation ledger are unchanged — conflicts annotate the report,
    they do not alter its claims.
    """
    case_id = case["case_id"]
    claims = report["claims"]
    entries = build_ledger(store, objects, case_id, claims)
    evidence_count = store.count_evidence(case_id)
    report = {
        **report,
        "markdown": _render_markdown(
            case,
            claims,
            evidence_count,
            entries,
            partial=report["partial"],
            exceeded_kind=report["exceeded_kind"],
            refused=report["refused_claims"],
            verification=report["verification"],
            conflicts=conflict_records,
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
    }
    return store.save_report(case_id, report)
