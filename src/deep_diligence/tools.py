"""The workstream tools: corpus search, document read, finding record.

These are ordinary LangChain tools, closed over one case. The harness's
own filesystem/task/todo tools come from Deep Agents; these three are
ours, and `record_finding` is where the citation rule is enforced: a
finding cannot be recorded without at least one evidence ID that
exists in the case — evidence IDs resolve within the case only, so
another case's documents are unreachable from here — and a finding
that names a fact reference must match that fact as the cited
document's own Key facts ledger states it. A fact reference the
ledger does not carry, or a claim that rewrites the ledger's words,
is refused: evidence text is data a workstream reads, never a
channel through which a document can mint facts about itself.

When the case's approved plan pins skills for a question (skills.py),
`record_finding` also stamps the finding with those pins: the
provenance that says which governed methods the workstream ran
under when it produced the finding. The stamp comes from the plan,
never from anything the model says.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .evidence import evidence_text
from .objects import ObjectStore
from .store import Store
from .verifier import ledger_facts


def build_case_tools(
    store: Store,
    objects: ObjectStore,
    case: dict,
    skills_by_question: dict[str, list[dict]] | None = None,
) -> list:
    case_id = case["case_id"]
    question_ids = {q["id"] for q in case["brief"]["questions"]}
    # The skills each question's workstream runs under, pinned by
    # the plan gate (skills.py): a finding recorded for a question
    # carries the pins as its provenance — which governed methods
    # produced it. The mapping is read from the approved plan by
    # the caller; the tool only stamps what it is handed.
    skills_by_question = skills_by_question or {}

    @tool
    def search_corpus(query: str) -> str:
        """Search this case's evidence corpus for documents matching a query.

        Returns the in-scope evidence documents ranked by relevance,
        each with its evidence ID. Use read_document with an evidence ID
        to read a document in full.
        """
        tokens = [t for t in query.lower().split() if len(t) > 2]
        scored: list[tuple[int, dict]] = []
        for ev in store.list_evidence(case_id):
            text = evidence_text(objects, ev).lower()
            score = sum(text.count(tok) for tok in tokens)
            scored.append((score, ev))
        scored.sort(key=lambda item: (-item[0], item[1]["evidence_id"]))
        lines = [f"{len(scored)} evidence documents in scope for this case:"]
        for score, ev in scored[:10]:
            lines.append(
                f"- {ev['evidence_id']} · {ev['title']} "
                f"({ev['source']}, {ev['doc_date']}) · relevance {score}"
            )
        return "\n".join(lines)

    @tool
    def read_document(evidence_id: str) -> str:
        """Read one evidence document in full by its evidence ID (e.g. EV-0003)."""
        ev = store.get_evidence(case_id, evidence_id)
        if ev is None:
            return f"ERROR: {evidence_id} is not evidence in this case."
        return (
            f"EVIDENCE {evidence_id} — {ev['title']} "
            f"({ev['source']}, {ev['doc_date']})\n\n"
            + evidence_text(objects, ev)
        )

    @tool
    def record_finding(
        question_id: str, claim: str, evidence_ids: list[str], fact_ref: str = ""
    ) -> str:
        """Record a finding: one claim, the evidence IDs it rests on.

        A claim with no valid evidence ID is refused. fact_ref is the
        corpus fact identifier (e.g. F-HEL-101) when the claim comes
        from a document's Key facts ledger; leave it empty otherwise.
        """
        if question_id not in question_ids:
            return (
                f"ERROR: question {question_id} is not part of this brief. "
                f"Questions are: {sorted(question_ids)}"
            )
        if not evidence_ids:
            return "ERROR: a finding must cite at least one evidence ID."
        valid = []
        for evidence_id in evidence_ids:
            if store.get_evidence(case_id, evidence_id) is not None:
                valid.append(evidence_id)
        if not valid:
            return (
                "ERROR: none of the cited evidence IDs exist in this case: "
                f"{evidence_ids}"
            )
        if fact_ref:
            # A fact reference is a claim about the cited evidence:
            # that its Key facts ledger states this fact, in these
            # words. Check it against the ledger itself (verifier's
            # reader — the final ledger section, so a lookalike
            # section planted in a document's body mints nothing).
            ledger_texts = {}
            for evidence_id in valid:
                ev = store.get_evidence(case_id, evidence_id)
                if ev is not None:
                    ledger_texts.update(
                        ledger_facts(evidence_text(objects, ev))
                    )
            if fact_ref not in ledger_texts:
                return (
                    f"ERROR: fact {fact_ref} is not in the Key facts "
                    "ledger of any cited document; a fact reference "
                    "must name a fact the cited evidence states."
                )
            if claim.strip() != ledger_texts[fact_ref]:
                return (
                    f"ERROR: the claim does not match fact {fact_ref} "
                    "as the cited ledger states it; record the fact "
                    "in the ledger's words, or record the claim "
                    "without a fact reference."
                )
        for existing in store.list_findings(case_id):
            if (
                fact_ref
                and existing["question_id"] == question_id
                and existing.get("fact_ref") == fact_ref
            ):
                return f"Already recorded as {existing['finding_id']}."
        finding = store.add_finding_auto_id(
            case_id,
            {
                "question_id": question_id,
                "claim": claim.strip(),
                "evidence_ids": valid,
                "fact_ref": fact_ref or None,
                **(
                    {"skills": list(skills_by_question[question_id])}
                    if skills_by_question.get(question_id)
                    else {}
                ),
            },
        )
        return f"Recorded {finding['finding_id']} citing {', '.join(valid)}."

    return [search_corpus, read_document, record_finding]
