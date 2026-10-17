"""The report composer — deterministic code, not a model.

It assembles the report draft from recorded findings, grouped by
question. A finding whose citations no longer resolve to case evidence
is refused here a second time (record_finding already enforces it at
write time) and named in the draft's method note. Every claim line
carries its evidence IDs; a question with no findings says so plainly;
a partial draft (budget exhausted) names the uncovered questions
instead of quietly omitting them.
"""

from __future__ import annotations

from .store import Store, utcnow


def compose_report(
    store: Store, case: dict, *, partial: bool = False, exceeded_kind: str | None = None
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
        claims.append({**finding, "evidence_ids": valid})
    coverage = {q["id"]: 0 for q in brief["questions"]}
    for claim in claims:
        coverage[claim["question_id"]] = coverage.get(claim["question_id"], 0) + 1

    lines = [
        f"# Diligence report — {brief['subject']}",
        "",
        "Status: DRAFT — not final until an analyst approves it.",
        f"Commissioned by: {brief['analyst']}. "
        f"Evidence in scope: {len(evidence)} documents.",
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
    lines += ["## Sources", ""]
    for evidence_id in sorted(evidence):
        ev = evidence[evidence_id]
        lines.append(
            f"- [{evidence_id}] {ev['title']} — {ev['source']}, {ev['doc_date']}"
        )
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
    report = {
        "case_id": case_id,
        "status": "DRAFT",
        "partial": partial,
        "exceeded_kind": exceeded_kind,
        "markdown": "\n".join(lines) + "\n",
        "claims": claims,
        "refused_claims": refused,
        "coverage": coverage,
        "composed_at": utcnow(),
        "decided_by": None,
        "decision_note": None,
    }
    return store.save_report(case_id, report)
