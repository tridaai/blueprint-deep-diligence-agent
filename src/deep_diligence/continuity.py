"""Long-case continuity: which workstreams new evidence touches.

A diligence case runs for days. Documents arrive after the plan was
approved — sometimes after a draft exists — and the desk's rule is
that nothing re-runs silently and nothing pretends the new document
was always there. When evidence is added to a case that already has
a plan, the shell works out which workstreams the document speaks
to and marks those questions STALE on the case record; execution is
refused until an analyst decides — re-plan, or proceed with the
staleness on the record (see service.decide_staleness).

The keying is deterministic and uses the same governed keyword
stems the corpus routing uses (corpus.WORKSTREAM_KEYWORDS): a
workstream is touched when the document's text carries at least one
of its stems. The score that touched it rides on the mark, so the
analyst can see why a question was flagged rather than take the
shell's word for it. A document that speaks to no workstream marks
nothing — arrival alone is not staleness.
"""

from __future__ import annotations

from .corpus import WORKSTREAM_KEYWORDS

# A document speaks to a workstream when at least this many of the
# workstream's stems appear in it. One shared stem is cross-talk,
# not signal — the stem lists overlap by design ("adopt" sits in
# market-scan's list and inside every "adoption" a product document
# prints; "fund" sits inside "funded") — and marking a workstream
# stale is an interruption to the desk, so it takes real evidence
# of relevance, not a stray word.
MIN_STALE_SCORE = 2


def workstream_scores(text: str) -> dict[str, int]:
    """How strongly a document speaks to each workstream: the
    count of that workstream's keyword stems present in the text —
    the same measure the offline reader routes facts by."""
    lowered = text.lower()
    return {
        workstream: sum(1 for stem in stems if stem in lowered)
        for workstream, stems in WORKSTREAM_KEYWORDS.items()
    }


def question_workstreams(case: dict, plan: dict | None) -> dict[str, str]:
    """Question id -> workstream, from the current plan where one
    exists (the plan is what execution runs), else from the brief."""
    if plan and (plan.get("plan") or {}).get("workstreams"):
        return {
            ws["question_id"]: ws["workstream"]
            for ws in plan["plan"]["workstreams"]
        }
    return {q["id"]: q["workstream"] for q in case["brief"]["questions"]}


def stale_questions(case: dict, plan: dict | None, text: str) -> list[dict]:
    """The questions a newly arrived document makes stale: every
    question whose workstream the document speaks to (score at
    least MIN_STALE_SCORE), each with the score that touched it."""
    scores = workstream_scores(text)
    touched = {ws for ws, score in scores.items() if score >= MIN_STALE_SCORE}
    if not touched:
        return []
    return [
        {
            "question_id": question_id,
            "workstream": workstream,
            "score": scores[workstream],
        }
        for question_id, workstream in question_workstreams(case, plan).items()
        if workstream in touched
    ]
