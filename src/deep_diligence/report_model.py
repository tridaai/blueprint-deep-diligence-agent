"""The report as a structured model — one model, three formats.

Until this round the composer produced markdown and only markdown:
the report record carried the claims and the rendered text, and any
other rendering would have meant a second assembly that could drift
from the first. The composer now builds a structured model of the
report first — executive summary, per-question sections with support
grades, conflicts, coverage, risks and open items, references,
verification — and the renderings are functions of that model: the
markdown (the draft the analyst reads at the gate), a self-contained
print-ready HTML page, and the JSON export. They cannot disagree,
because there is nothing to disagree with: one model, three
renderers. The model and the HTML ride on the report record; the
JSON export is the model plus the analyst's section annotations.

The executive summary is composed AFTER verification, from the
supported-claim set alone — the claims that survived the verifier.
Its sentences are headline claims, taken verbatim, one per question
in brief order; its counts are the report's own structural numbers
(claims supported, conflicts open, questions uncovered), computed in
code. The hard rule is enforced at composition time with the
verifier's own extraction machinery: a summary sentence may state
no figure, date, or named entity that does not appear in a
supported claim. A sentence that fails is DROPPED — recorded on the
report with the reasons, never silently rewritten and never
rendered. Headline sentences are claim texts, so a drop should
never happen on the standard path; the check stands so that any
summary source, present or future, passes through the same gate
before it can reach a reader.

Each question section carries a support grade, computed by the one
function `question_grade`: UNANSWERED (no supported claims), THIN
(a single claim, or a claim the verifier's model check marked
unsupported), STRONG (two or more claims, at least one resting on a
primary source), STANDARD (the rest — multiple claims on secondary
sources). Thinness outranks strength on purpose: one claim on an
excellent source is still one claim, and a claim the model check
flagged drags its question down however many others stand.

The Risks and open items section is generated entirely from
records — open and noted conflicts, quarantined claims with their
dispositions, questions the coverage table leaves PARTIAL or NOT
COVERED, staleness an analyst proceeded on, redaction flags — and
when every one of those sets is empty it says so plainly. Nothing
in it is free text; it is the report's honest edges, assembled from
the same records the rest of the pipeline wrote. One discipline
carries over from the verifier round: quarantined claims are
referenced by ID, reason, and disposition only, never quoted — a
quarantined claim's text, and the unsupported figure it may carry,
reaches no format of the report.
"""

from __future__ import annotations

import html

from .conflicts import source_class
from .store import Store
from .verifier import check_claim

# The named report sections an analyst may annotate (plus each
# brief question's own section, keyed by question id).
NAMED_SECTIONS: tuple[tuple[str, str], ...] = (
    ("executive_summary", "Executive summary"),
    ("conflicts", "Conflicts"),
    ("risks_and_open_items", "Risks and open items"),
    ("references", "References"),
    ("verification", "Verification"),
    ("method_and_limits", "Method and limits"),
)

GRADES = ("STRONG", "STANDARD", "THIN", "UNANSWERED")

# The status line every format of the report opens with. A draft
# carries DRAFT_STATUS_LINE; the report gate swaps it for the
# final_status_line when an analyst approves, so an approved
# report never reads as a draft (service.decide_report).
DRAFT_STATUS_LINE = "Status: DRAFT — not final until an analyst approves it."


def final_status_line(analyst: str, day: str) -> str:
    return f"Status: FINAL — approved by {analyst} on {day}."


def question_grade(
    claims: list[dict],
    evidence_by_id: dict[str, dict],
    weak_claim_ids: set[str] | frozenset[str] = frozenset(),
) -> str:
    """The one support-grade rule, applied to one question's
    supported claims:

    - UNANSWERED — no supported claims.
    - THIN — any claim the verifier's model check marked
      unsupported, or a single claim standing alone (however good
      its source: one claim is a thin basis for an answer).
    - STRONG — two or more claims, at least one resting on a
      primary source (per the source classes in conflicts.py).
    - STANDARD — two or more claims, all on secondary sources.
    """
    if not claims:
        return "UNANSWERED"
    if any(c["claim_id"] in weak_claim_ids for c in claims):
        return "THIN"
    if len(claims) == 1:
        return "THIN"
    for claim in claims:
        for evidence_id in claim["evidence_ids"]:
            ev = evidence_by_id.get(evidence_id)
            if ev is not None and source_class(ev["kind"])[0] == "primary":
                return "STRONG"
    return "STANDARD"


def check_summary_sentences(
    candidates: list[str], supported_texts: list[str]
) -> tuple[list[str], list[dict]]:
    """The summary-integrity check, run with the verifier's own
    extraction machinery: every figure, date, and named entity in a
    candidate sentence must appear in the supported claims. Returns
    (kept, dropped) — a dropped sentence is recorded with its
    failures and is never rendered; there is no rewrite path."""
    kept: list[str] = []
    dropped: list[dict] = []
    for sentence in candidates:
        failures = check_claim(sentence, supported_texts)
        if failures:
            dropped.append({"sentence": sentence, "failures": failures})
        else:
            kept.append(sentence)
    return kept, dropped


def build_summary(
    case: dict,
    claims: list[dict],
    *,
    open_conflicts: int,
    noted_conflicts: int,
    coverage_detail: list[dict] | None,
    quarantined_count: int,
) -> dict:
    """The executive summary, composed from the supported claims
    alone: one headline sentence per question in brief order (the
    question's first supported claim, verbatim), integrity-checked,
    plus the counts that matter — the report's own structural
    numbers, computed here, in code."""
    brief = case["brief"]
    supported_texts = [c["claim"] for c in claims]
    candidates: list[str] = []
    for question in brief["questions"]:
        first = next(
            (c for c in claims if c["question_id"] == question["id"]), None
        )
        if first is not None:
            candidates.append(first["claim"])
    kept, dropped = check_summary_sentences(candidates, supported_texts)
    if coverage_detail is not None:
        uncovered = sum(
            1 for row in coverage_detail if row["status"] == "NOT COVERED"
        )
        partial = sum(
            1 for row in coverage_detail if row["status"] == "PARTIAL"
        )
    else:
        answered = {c["question_id"] for c in claims}
        uncovered = sum(
            1 for q in brief["questions"] if q["id"] not in answered
        )
        partial = 0
    return {
        "sentences": kept,
        "dropped": dropped,
        "counts": {
            "claims_supported": len(claims),
            "claims_quarantined": quarantined_count,
            "conflicts_open": open_conflicts,
            "conflicts_noted": noted_conflicts,
            "questions_total": len(brief["questions"]),
            "questions_uncovered": uncovered,
            "questions_partial": partial,
        },
    }


def staleness_lines(decisions: list[dict] | None) -> list[str]:
    """The Method-and-limits lines for staleness decisions, shared
    by the markdown renderer and the report model so the two can
    never word the same decision differently."""
    lines: list[str] = []
    for decision in decisions or []:
        evidence_ids = ", ".join(decision["evidence_ids"])
        questions = ", ".join(decision["questions"])
        day = (decision.get("at") or "")[:10]
        if decision["action"] == "proceed":
            lines.append(
                f"New evidence arrived after planning ({evidence_ids}); "
                f"the desk marked {questions} stale, and "
                f"{decision['analyst']} recorded a decision to proceed "
                f"without re-planning on {day}"
                + (f': "{decision["note"]}"' if decision.get("note") else "")
                + ". Those workstreams were not re-run; their findings "
                "predate the new evidence."
            )
        else:
            lines.append(
                f"New evidence arrived after planning ({evidence_ids}); "
                f"the desk marked {questions} stale and re-planned on "
                f"{day} (analyst {decision['analyst']}). The marked "
                "workstreams were re-run against the full evidence; "
                "the others kept their findings, which rest on "
                "evidence the arrival did not touch."
            )
    return lines


def skill_lines(store: Store, case: dict) -> list[str]:
    """The Method-and-limits lines for the skills the case's
    workstreams ran under, folded from the findings' own
    provenance stamps — one line per question that ran under
    skills, naming the pinned versions the plan gate approved.
    Shared by the report model and the composer's markdown
    renderer, like staleness_lines, so no format can name a
    method another omits. A case that ran under no skills grows
    no line: the section names the methods that were actually in
    force, never the library's catalogue."""
    brief = case["brief"]
    skills_by_question: dict[str, list[dict]] = {}
    for finding in store.list_findings(case["case_id"]):
        for pin in finding.get("skills") or []:
            slot = skills_by_question.setdefault(finding["question_id"], [])
            if pin not in slot:
                slot.append(pin)
    lines: list[str] = []
    for question in brief["questions"]:
        pins = skills_by_question.get(question["id"])
        if pins:
            named = ", ".join(
                f"{pin['name']} v{pin['version']}" for pin in pins
            )
            lines.append(
                f"Workstream {question['id']} "
                f"({question['workstream']}) ran under governed "
                f"skill(s): {named}. The method text was loaded "
                "from the skills library at the pinned "
                "version(s), and the findings carry the pins as "
                "their provenance."
            )
    return lines


def build_open_items(
    store: Store,
    case: dict,
    report: dict,
    conflict_records: list[dict],
) -> dict:
    """Risks and open items, generated entirely from records: the
    conflict records (open and noted alike — a noted conflict
    remains part of the report), the quarantine records with the
    dispositions analysts have given them, the coverage rows that
    are not ANSWERED, the staleness decisions an analyst proceeded
    on, and the redaction flags (open ones block approval;
    acknowledged ones stay visible as part of the report's
    history)."""
    conflicts = [
        {
            "conflict_id": r["conflict_id"],
            "metric": r["key"]["metric"],
            "period": r["key"]["period"],
            "values": [side["value_text"] for side in r["sides"]],
            "status": r["status"],
            "noted_by": r["noted_by"],
            "note": r["note"],
        }
        for r in conflict_records
    ]
    # Quarantined claims are referenced by ID, reason, and
    # disposition only — never quoted. A quarantined claim's text
    # (and the unsupported figure it may carry) must not reach the
    # report in any format; that is the R3 zero-tolerance rule, and
    # it binds the open-items section and the export exactly as it
    # binds the body. The claim text lives in the quarantine
    # record, behind its own API.
    quarantined = [
        {
            "claim_id": q["claim_id"],
            "question_id": q["question_id"],
            "reason": q["reason"],
            "status": q["status"],
            "decided_by": q["decided_by"],
        }
        for q in store.list_quarantine(case["case_id"])
    ]
    coverage_gaps = [
        {
            "question_id": row["question_id"],
            "status": row["status"],
            "findings": row["findings"],
            "reason": row.get("reason"),
        }
        for row in report.get("coverage_detail") or []
        if row["status"] != "ANSWERED"
    ]
    proceeds = [
        {
            "evidence_ids": d["evidence_ids"],
            "questions": d["questions"],
            "analyst": d["analyst"],
            "at": d["at"],
            "note": d["note"],
        }
        for d in case.get("staleness_decisions") or []
        if d["action"] == "proceed"
    ]
    flags = [
        {
            "flag_id": f["flag_id"],
            "claim_id": f["claim_id"],
            "evidence_id": f["evidence_id"],
            "status": f["status"],
            "decided_by": f["decided_by"],
        }
        for f in store.list_redaction_flags(case["case_id"])
    ]
    return {
        "conflicts": conflicts,
        "quarantined": quarantined,
        "coverage_gaps": coverage_gaps,
        "staleness_proceeds": proceeds,
        "redaction_flags": flags,
        "empty": not (conflicts or quarantined or coverage_gaps or proceeds or flags),
    }


def model_sections(case: dict) -> list[dict]:
    """The report's sections, in order — the valid targets for an
    analyst's section note."""
    sections = [{"key": "executive_summary", "title": "Executive summary"}]
    for question in case["brief"]["questions"]:
        sections.append(
            {
                "key": question["id"],
                "title": f"{question['id']} — {question['text']}",
            }
        )
    sections += [
        {"key": key, "title": title} for key, title in NAMED_SECTIONS[1:]
    ]
    return sections


def build_report_model(
    store: Store,
    case: dict,
    report: dict,
    entries: list[dict],
    conflict_records: list[dict],
) -> dict:
    """Assemble the structured report model from the records: the
    claims the report carries, the citation-ledger entries built at
    composition, the conflict records, and the live governance
    records (quarantine dispositions, redaction flags) read from
    the store at render time — so a re-render after an analyst's
    decision shows the decision."""
    case_id = case["case_id"]
    brief = case["brief"]
    claims = report["claims"]
    evidence_by_id = {
        ev["evidence_id"]: ev for ev in store.list_evidence(case_id)
    }
    verifier_report = store.get_verifier_report(case_id) or {}
    weak_claim_ids = {
        check["claim_id"]
        for check in verifier_report.get("checks", [])
        if check.get("model_check") == "unsupported"
    }
    open_items = build_open_items(store, case, report, conflict_records)
    summary = build_summary(
        case,
        claims,
        open_conflicts=sum(
            1 for r in conflict_records if r["status"] == "OPEN"
        ),
        noted_conflicts=sum(
            1 for r in conflict_records if r["status"] == "NOTED"
        ),
        coverage_detail=report.get("coverage_detail"),
        quarantined_count=len(store.list_quarantine(case_id)),
    )
    questions = []
    for question in brief["questions"]:
        q_claims = [c for c in claims if c["question_id"] == question["id"]]
        questions.append(
            {
                "question_id": question["id"],
                "text": question["text"],
                "grade": question_grade(q_claims, evidence_by_id, weak_claim_ids),
                "claims": [
                    {
                        "claim_id": c["claim_id"],
                        "finding_id": c["finding_id"],
                        "claim": c["claim"],
                        "evidence_ids": list(c["evidence_ids"]),
                    }
                    for c in q_claims
                ],
            }
        )
    cited: dict[str, dict] = {}
    for entry in entries:
        for citation in entry["citations"]:
            slot = cited.setdefault(
                citation["evidence_id"], {**citation, "cited_by": []}
            )
            slot["cited_by"].append(entry["claim_id"])
    references = [
        {
            "evidence_id": evidence_id,
            "source": cited[evidence_id]["source"],
            "title": cited[evidence_id]["title"],
            "sha256": cited[evidence_id]["sha256"],
            "cited_by": cited[evidence_id]["cited_by"],
        }
        for evidence_id in sorted(cited)
    ]
    method_notes = [
        "Assembled by the Deep Diligence Agent from the case evidence "
        "only. Each claim cites the evidence it rests on; the desk's "
        "decision is not part of this document and no recommendation "
        "is made."
    ]
    if report["refused_claims"]:
        method_notes.append(
            f"{len(report['refused_claims'])} recorded claim(s) were "
            "refused at composition because their evidence citations "
            "did not resolve: " + ", ".join(report["refused_claims"]) + "."
        )
    method_notes += staleness_lines(case.get("staleness_decisions"))
    method_notes += skill_lines(store, case)
    return {
        "case_id": case_id,
        "subject": brief["subject"],
        "status": report["status"],
        "partial": report["partial"],
        "exceeded_kind": report["exceeded_kind"],
        "exceeded_phase": report["exceeded_phase"],
        "commissioned_by": brief["analyst"],
        "evidence_count": len(evidence_by_id),
        "executive_summary": summary,
        "questions": questions,
        "conflicts": [
            {
                "conflict_id": r["conflict_id"],
                "key": r["key"],
                "sides": [
                    {
                        "value_text": side["value_text"],
                        "claims": [
                            {
                                "claim_id": m["claim_id"],
                                "claim": m["claim"],
                                "evidence_ids": list(m["evidence_ids"]),
                                "sources": m["sources"],
                                "more_recent": m["more_recent"],
                            }
                            for m in side["claims"]
                        ],
                    }
                    for side in r["sides"]
                ],
                "status": r["status"],
                "noted_by": r["noted_by"],
                "note": r["note"],
                "noted_at": r["noted_at"],
            }
            for r in conflict_records
        ],
        "coverage": report.get("coverage_detail"),
        "risks_and_open_items": open_items,
        "references": references,
        "verification": report["verification"],
        "refused_claims": list(report["refused_claims"]),
        "method_notes": method_notes,
        "sections": model_sections(case),
    }


def export_report(store: Store, case: dict, report: dict) -> dict:
    """The JSON export: the report's structured model, the live
    report status, and the analyst's section annotations in their
    own field — annotations annotate the export; they are never
    part of the report body in any format. The caller passes a
    report that carries its model (the service re-renders a legacy
    record that predates models before exporting)."""
    model = report["model"]
    annotations: dict[str, list[dict]] = {}
    for note in store.list_section_notes(case["case_id"]):
        annotations.setdefault(note["section"], []).append(note)
    return {**model, "status": report["status"], "annotations": annotations}


# ---------------------------------------------------------------------------
# HTML rendering — self-contained, print-ready, house style.
# ---------------------------------------------------------------------------

_HTML_STYLE = """
body { background:#FFFFFF; color:#0A0A0A; font-family:-apple-system,
  "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin:40px auto;
  max-width:820px; padding:0 24px; line-height:1.55; }
h1 { font-size:26px; border-bottom:3px solid #4F46E5; padding-bottom:8px; }
h2 { font-size:19px; color:#4F46E5; margin-top:32px; }
h3 { font-size:15px; margin-top:20px; }
.muted { color:#555560; }
.banner { background:#FDF2F2; border:1px solid #F3C2C2; padding:10px 14px;
  border-radius:6px; }
.grade { display:inline-block; font-size:11px; font-weight:700;
  letter-spacing:.06em; padding:2px 8px; border-radius:999px; color:#fff;
  vertical-align:middle; }
.grade-STRONG { background:#4F46E5; }
.grade-STANDARD { background:#6B6B76; }
.grade-THIN { background:#B45309; }
.grade-UNANSWERED { background:#B42318; }
.cite { color:#4F46E5; font-size:12px; }
ul { padding-left:20px; }
li { margin:4px 0; }
.counts { color:#555560; font-size:14px; }
@media print { body { margin:0; max-width:none; } }
"""


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def render_html(model: dict) -> str:
    """Render the model as a self-contained HTML page: inline CSS,
    no external assets, every value escaped — a reader can save or
    print it and it will say exactly what the model says."""
    out: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>Diligence report — {_esc(model['subject'])}</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        f"<h1>Diligence report — {_esc(model['subject'])}</h1>",
        f"<p>{_esc(DRAFT_STATUS_LINE)}<br>",
        f"Commissioned by: {_esc(model['commissioned_by'])}. "
        f"Evidence in scope: {model['evidence_count']} documents.</p>",
    ]
    summary = model["executive_summary"]
    counts = summary["counts"]
    out.append("<h2>Executive summary</h2>")
    out.append(
        '<p class="counts">'
        f"Supported claims: {counts['claims_supported']} · "
        f"Quarantined claims: {counts['claims_quarantined']} · "
        f"Conflicts open: {counts['conflicts_open']} · "
        f"Conflicts noted: {counts['conflicts_noted']} · "
        f"Questions not covered: {counts['questions_uncovered']} · "
        f"Questions partial: {counts['questions_partial']}</p>"
    )
    if summary["sentences"]:
        out.append("<ul>")
        for sentence in summary["sentences"]:
            out.append(f"<li>{_esc(sentence)}</li>")
        out.append("</ul>")
    elif counts["claims_supported"]:
        out.append(
            "<p><em>Every summary sentence failed the integrity check "
            "against the supported claims and was dropped; the drops "
            "are recorded on the report record.</em></p>"
        )
    else:
        out.append(
            "<p><em>No findings survived verification; the summary "
            "has nothing to summarise.</em></p>"
        )
    if model["partial"]:
        out.append(
            '<p class="banner"><strong>Partial report.</strong> The '
            f"run stopped on its {_esc(model['exceeded_kind'] or 'budget')} "
            f"budget during the {_esc(model['exceeded_phase'] or 'execution')} "
            "phase, before the plan was fully executed. Coverage is "
            "itemised per question — treat anything not ANSWERED as "
            "not fully investigated.</p>"
        )
    if model["coverage"]:
        out.append("<p>Coverage by question:</p><ul>")
        for row in model["coverage"]:
            line = (
                f"{_esc(row['question_id'])} — {_esc(row['status'])} "
                f"({row['findings']} finding(s) in this report)"
            )
            if row["status"] == "PARTIAL":
                line += " — the run stopped inside this workstream"
            elif row["status"] == "NOT COVERED":
                line += f" — {_esc(row['reason'])}" if row.get("reason") else " — not investigated"
            out.append(f"<li>{line}</li>")
        out.append("</ul>")
    for question in model["questions"]:
        out.append(
            f"<h2>{_esc(question['question_id'])} — {_esc(question['text'])} "
            f'<span class="grade grade-{question["grade"]}">'
            f"{question['grade']}</span></h2>"
        )
        if not question["claims"]:
            out.append("<p><em>No findings recorded for this question.</em></p>")
        out.append("<ul>")
        for claim in question["claims"]:
            citations = " ".join(
                f'<span class="cite">[{_esc(eid)}]</span>'
                for eid in claim["evidence_ids"]
            )
            out.append(f"<li>{_esc(claim['claim'])} {citations}</li>")
        out.append("</ul>")
    if model["conflicts"]:
        out.append("<h2>Conflicts</h2>")
        out.append(
            "<p>The supported findings disagree on the items below. "
            "Both sides are reported, with the values as stated and "
            "the source each side rests on; the desk takes no "
            "position on which value is correct, and neither side "
            "has been suppressed or merged.</p>"
        )
        for record in model["conflicts"]:
            key = record["key"]
            period = key["period"] if key["period"] else "no period stated"
            out.append(
                f"<h3>{_esc(record['conflict_id'])} — {_esc(key['metric'])} · "
                f"{_esc(period)} · {_esc(key['entity'])}</h3><ul>"
            )
            for side in record["sides"]:
                out.append(f"<li><strong>{_esc(side['value_text'])}</strong><ul>")
                for member in side["claims"]:
                    citations = " ".join(
                        f'<span class="cite">[{_esc(eid)}]</span>'
                        for eid in member["evidence_ids"]
                    )
                    out.append(
                        f"<li>{_esc(member['claim_id'])}: "
                        f"&ldquo;{_esc(member['claim'])}&rdquo; {citations}"
                    )
                    for source in member["sources"]:
                        context = (
                            f"Source: {_esc(source['source'])}, dated "
                            f"{_esc(source['doc_date'])} · "
                            f"{_esc(source['source_class'])} source"
                        )
                        if member["more_recent"]:
                            context += " · more recent"
                        out.append(f"<br><span class=\"muted\">{context}</span>")
                    out.append("</li>")
                out.append("</ul></li>")
            if record["status"] == "NOTED":
                out.append(
                    f"<li>Analyst note — marked NOTED by "
                    f"{_esc(record['noted_by'])} on "
                    f"{_esc(record['noted_at'])}: "
                    f"&ldquo;{_esc(record['note'])}&rdquo; The conflict "
                    "remains part of this report.</li>"
                )
            out.append("</ul>")
    out.append("<h2>References</h2>")
    if model["references"]:
        out.append("<ul>")
        for ref in model["references"]:
            out.append(
                f"<li>[{_esc(ref['evidence_id'])}] {_esc(ref['source'])} — "
                f"{_esc(ref['title'])} · sha256 {_esc(ref['sha256'][:12])}… · "
                f"cited by {_esc(', '.join(ref['cited_by']))}</li>"
            )
        out.append("</ul>")
    else:
        out.append("<p><em>No cited evidence.</em></p>")
    if model["verification"] is not None:
        v = model["verification"]
        out.append("<h2>Verification</h2>")
        text = (
            f"An independent verifier re-read every claim in this draft "
            f"against the evidence it cites — fetched from the evidence "
            f"store, not from the composer's working set. "
            f"{v['checked']} claim(s) checked: {v['supported']} supported, "
            f"{v['quarantined']} quarantined."
        )
        if v["quarantined"]:
            reasons = ", ".join(
                f"{reason} ×{count}"
                for reason, count in sorted(v["by_reason"].items())
            )
            text += (
                f" Quarantined by reason: {reasons}. Quarantined claims "
                "are excluded from this report and held in the case's "
                "quarantine record; an analyst can send one back for "
                "re-investigation or drop it, but can never approve it "
                "into the report."
            )
        out.append(f"<p>{_esc(text)}</p>")
    out.append("<h2>Risks and open items</h2>")
    out += _open_items_html(model["risks_and_open_items"])
    out.append("<h2>Method and limits</h2>")
    for note in model["method_notes"]:
        out.append(f"<p>{_esc(note)}</p>")
    out.append("</body></html>")
    return "\n".join(out) + "\n"


def _open_items_html(items: dict) -> list[str]:
    if items["empty"]:
        return [
            "<p>None recorded: no open or noted conflicts, no "
            "quarantined claims, no uncovered or partial questions, "
            "no staleness proceeded on, and no redaction flags.</p>"
        ]
    out: list[str] = []
    if items["conflicts"]:
        out.append("<h3>Conflicts</h3><ul>")
        for record in items["conflicts"]:
            period = record["period"] if record["period"] else "no period stated"
            line = (
                f"{_esc(record['conflict_id'])} — {_esc(record['metric'])} · "
                f"{_esc(period)}: {_esc(' vs '.join(record['values']))} "
                f"({_esc(record['status'])}"
            )
            if record["status"] == "NOTED":
                line += (
                    f" by {_esc(record['noted_by'])}: "
                    f"&ldquo;{_esc(record['note'])}&rdquo;"
                )
            out.append(f"<li>{line})</li>")
        out.append("</ul>")
    if items["quarantined"]:
        out.append("<h3>Quarantined claims</h3><ul>")
        for record in items["quarantined"]:
            line = (
                f"{_esc(record['claim_id'])} "
                f"({_esc(record['question_id'])}) — "
                f"{_esc(record['reason'])}; status {_esc(record['status'])}"
            )
            if record["decided_by"]:
                line += f", decided by {_esc(record['decided_by'])}"
            out.append(f"<li>{line}</li>")
        out.append("</ul>")
    if items["coverage_gaps"]:
        out.append("<h3>Coverage gaps</h3><ul>")
        for row in items["coverage_gaps"]:
            line = (
                f"{_esc(row['question_id'])} — {_esc(row['status'])} "
                f"({row['findings']} finding(s) in this report)"
            )
            if row.get("reason"):
                line += f" — {_esc(row['reason'])}"
            out.append(f"<li>{line}</li>")
        out.append("</ul>")
    if items["staleness_proceeds"]:
        out.append("<h3>Staleness proceeded on</h3><ul>")
        for decision in items["staleness_proceeds"]:
            line = (
                f"Evidence {_esc(', '.join(decision['evidence_ids']))}; "
                f"questions {_esc(', '.join(decision['questions']))} — "
                f"proceeded by {_esc(decision['analyst'])} on "
                f"{_esc((decision['at'] or '')[:10])}"
            )
            if decision["note"]:
                line += f": &ldquo;{_esc(decision['note'])}&rdquo;"
            out.append(f"<li>{line}</li>")
        out.append("</ul>")
    if items["redaction_flags"]:
        out.append("<h3>Redaction flags</h3><ul>")
        for flag in items["redaction_flags"]:
            line = (
                f"{_esc(flag['claim_id'])} cites "
                f"{_esc(flag['evidence_id'])} — {_esc(flag['status'])}"
            )
            if flag["decided_by"]:
                line += f" by {_esc(flag['decided_by'])}"
            out.append(f"<li>{line}</li>")
        out.append("</ul>")
    return out
