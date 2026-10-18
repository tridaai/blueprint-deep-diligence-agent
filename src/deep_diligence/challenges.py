"""The challenge pass: a structured second opinion on the draft.

The verifier (verifier.py) checks each claim against its own
evidence, and the conflict pass (conflicts.py) sets supported
claims side by side. Neither argues against the draft. A claim can
pass both and still be a weak thing to approve a report on: a
headline revenue figure resting on one news article, an answer
whose number comes from the oldest document in a live
disagreement, a STRONG grade that is one filing repeated four
times. This pass is the desk's second opinion — it examines the
verified draft for exactly those weaknesses and records each one
as a challenge the analyst must answer.

The pass is deterministic and independent in the verifier's
sense: it runs from the stored records — the report's supported
claims, the evidence metadata, the conflict records, the grades
the report model computed — never from composer internals, and
re-running it on an unchanged draft yields the identical
challenge set (the sync below is keyed by content fingerprint).
Each rule is named, and the record names the evidence the rule
read, so an analyst can check the objection itself.

Severity is the desk's doctrine about what else already has eyes
on a weakness. A must_answer challenge is a weakness nothing else
in the pipeline surfaces — today that is one shape: a headline
financial figure (revenue, margins, cash flow, debt, funding)
asserted by a claim that cites a single secondary source, with
no corroborating claim anywhere in the draft. Weaknesses adjacent
to something the desk already sees — a conflict side, a grade —
are advisory: recorded, counted at the gate, never blocking.

Challenges are internal review. They ride the case, never the
report record, so no delivered format — markdown, HTML, or the
JSON export — can carry one. When a re-composition moves the
draft on, the pass recomputes: an answered challenge whose target
has vanished (or whose substance changed) is marked SUPERSEDED,
never deleted — the history of what the desk objected to, and
what the analyst answered, stays on the case.
"""

from __future__ import annotations

import hashlib
import json

from .conflicts import extract_assertions, source_class
from .report_model import question_grade
from .store import Store, utcnow

# The headline financial figures of a diligence — the numbers a
# reader of the report will quote. A claim asserting one of these
# on a single secondary source, uncorroborated, is the one shape
# this pass insists on an answer for before approval.
HEADLINE_METRICS: frozenset[str] = frozenset(
    {
        "revenue",
        "revenue growth",
        "gross margin",
        "EBITDA margin",
        "operating cash flow",
        "net debt",
        "debt",
        "cash",
        "funding raised",
    }
)

RULE_SINGLE_SOURCE = "single_source_figure"
RULE_OLDEST_ON_CONTESTED = "oldest_source_on_contested_metric"
RULE_GRADE_ONE_DOCUMENT = "grade_rests_on_one_document"
RULE_ONE_DOCUMENT_CARRIES = "one_document_carries_questions"

MUST_ANSWER = "must_answer"
ADVISORY = "advisory"


def _fingerprint(body: dict) -> str:
    """The content identity of a computed challenge: the rule, its
    target, and the detail the rule read. Two passes over the same
    draft produce the same fingerprint; any change in the claim,
    its evidence, or the records the rule read produces a
    different one."""
    canonical = json.dumps(body, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _content(
    rule: str,
    *,
    key_part: str,
    claim_id: str | None,
    finding_id: str | None,
    question_id: str | None,
    claim: str | None,
    evidence_ids: list[str],
    severity: str,
    detail: dict,
) -> dict:
    body = {
        "rule": rule,
        "claim_id": claim_id,
        "question_id": question_id,
        "detail": detail,
    }
    return {
        "key": f"{rule}|{key_part}",
        "fingerprint": _fingerprint(body),
        "rule": rule,
        "severity": severity,
        "claim_id": claim_id,
        "finding_id": finding_id,
        "question_id": question_id,
        "claim": claim,
        "evidence_ids": evidence_ids,
        "detail": detail,
    }


def _evidence_view(ev: dict | None, evidence_id: str) -> dict:
    if ev is None:
        return {"evidence_id": evidence_id, "missing": True}
    cls, _rank = source_class(ev["kind"])
    return {
        "evidence_id": evidence_id,
        "source": ev["source"],
        "kind": ev["kind"],
        "doc_date": ev["doc_date"],
        "source_class": cls,
    }


def compute_challenges(store: Store, case: dict, report: dict) -> list[dict]:
    """Examine the draft's supported claims and return the
    challenge contents the records support — no IDs, no statuses;
    identity is the content fingerprint (see sync_challenges).

    Reads only stored records: the report's claims, the evidence
    list, the conflict records, the verifier report (for the
    model-check-weak set the grade rule consumes), and the brief.
    """
    case_id = case["case_id"]
    claims = report["claims"]
    evidence = {ev["evidence_id"]: ev for ev in store.list_evidence(case_id)}
    conflicts = store.list_conflicts(case_id)
    contents: list[dict] = []

    # Corroboration, keyed the way the conflict pass keys claims:
    # (metric, period, normalised value) -> the claims asserting it.
    # The same value from two claims is two voices; a value only
    # one claim asserts stands alone.
    assertions_by_claim: dict[str, list[dict]] = {}
    corroboration: dict[tuple, set[str]] = {}
    for claim in claims:
        assertions = extract_assertions(claim["claim"])
        assertions_by_claim[claim["claim_id"]] = assertions
        for assertion in assertions:
            key = (
                assertion["metric"],
                assertion["period"],
                tuple(assertion["value"]),
            )
            corroboration.setdefault(key, set()).add(claim["claim_id"])

    # Conflict membership: claim_id -> what the challenge needs
    # (the conflict, its own side's freshness, the conflict's
    # oldest document date and freshest side date).
    membership: dict[str, dict] = {}
    for conflict in conflicts:
        all_dates = [
            source["doc_date"]
            for side in conflict["sides"]
            for member in side["claims"]
            for source in member["sources"]
        ]
        side_latest = [
            max((m["latest_date"] for m in side["claims"]), default="")
            for side in conflict["sides"]
        ]
        freshest = max(side_latest, default="")
        oldest = min(all_dates, default="")
        for index, side in enumerate(conflict["sides"]):
            for member in side["claims"]:
                membership[member["claim_id"]] = {
                    "conflict": conflict,
                    "member": member,
                    "side_latest": side_latest[index],
                    "freshest": freshest,
                    "oldest": oldest,
                }

    # -- rule: single_source_figure --------------------------------------
    # A claim that asserts governed figures, cites exactly one
    # evidence item, and is corroborated by no other claim: the
    # figure stands on one document alone. Contested claims are
    # the next rule's ground — a conflict already puts both sides
    # in front of the analyst.
    for claim in claims:
        claim_id = claim["claim_id"]
        if claim_id in membership:
            continue
        assertions = assertions_by_claim[claim_id]
        if not assertions or len(claim["evidence_ids"]) != 1:
            continue
        uncorroborated = [
            a
            for a in assertions
            if corroboration[
                (a["metric"], a["period"], tuple(a["value"]))
            ]
            == {claim_id}
        ]
        if not uncorroborated:
            continue
        evidence_id = claim["evidence_ids"][0]
        ev = evidence.get(evidence_id)
        view = _evidence_view(ev, evidence_id)
        headline = any(
            a["metric"] in HEADLINE_METRICS for a in uncorroborated
        )
        severity = (
            MUST_ANSWER
            if headline and view.get("source_class") == "secondary"
            else ADVISORY
        )
        contents.append(
            _content(
                RULE_SINGLE_SOURCE,
                key_part=claim_id,
                claim_id=claim_id,
                finding_id=claim["finding_id"],
                question_id=claim["question_id"],
                claim=claim["claim"],
                evidence_ids=[evidence_id],
                severity=severity,
                detail={
                    "assertions": [
                        {
                            "metric": a["metric"],
                            "period": a["period"],
                            "value_text": a["value_text"],
                        }
                        for a in uncorroborated
                    ],
                    "evidence": view,
                    "headline_metric": headline,
                    "corroborated_by": [],
                },
            )
        )

    # -- rule: oldest_source_on_contested_metric --------------------------
    # A claim on a conflict side whose own evidence includes the
    # oldest document in the disagreement, while another side
    # rests on something fresher: the claim may be quoting a
    # number the record has already moved past. Advisory — the
    # conflict itself is in the report, permanently; the
    # challenge adds which side of it is stale.
    for claim in claims:
        claim_id = claim["claim_id"]
        info = membership.get(claim_id)
        if info is None:
            continue
        member = info["member"]
        own_dates = [s["doc_date"] for s in member["sources"]]
        if info["oldest"] not in own_dates:
            continue
        if not info["side_latest"] < info["freshest"]:
            continue
        conflict = info["conflict"]
        contents.append(
            _content(
                RULE_OLDEST_ON_CONTESTED,
                key_part=claim_id,
                claim_id=claim_id,
                finding_id=claim["finding_id"],
                question_id=claim["question_id"],
                claim=claim["claim"],
                evidence_ids=list(claim["evidence_ids"]),
                severity=ADVISORY,
                detail={
                    "conflict_id": conflict["conflict_id"],
                    "metric": conflict["key"]["metric"],
                    "period": conflict["key"]["period"],
                    "value_text": member["value_text"],
                    "oldest_doc_date": info["oldest"],
                    "freshest_side_date": info["freshest"],
                    "sources": [
                        {
                            "evidence_id": s["evidence_id"],
                            "kind": s["kind"],
                            "doc_date": s["doc_date"],
                            "source_class": s["source_class"],
                        }
                        for s in member["sources"]
                    ],
                },
            )
        )

    # -- question grades, by the one grade rule ----------------------------
    verifier_report = store.get_verifier_report(case_id) or {}
    weak_claim_ids = {
        check["claim_id"]
        for check in verifier_report.get("checks", [])
        if check.get("model_check") == "unsupported"
    }
    claims_by_question: dict[str, list[dict]] = {}
    for claim in claims:
        claims_by_question.setdefault(claim["question_id"], []).append(claim)

    # -- rule: grade_rests_on_one_document ---------------------------------
    # A STRONG or STANDARD grade whose claims, between them, cite
    # exactly one document: the grade counts claims, and four
    # claims repeating one filing are still one document's word.
    # (The grade rule itself needs two or more claims and, for
    # STRONG, a primary source — so a strong grade can never rest
    # on a single claim or a lone secondary source; repetition of
    # one document is the gap it cannot see.)
    sole_document_questions: dict[str, list[str]] = {}
    for question in case["brief"]["questions"]:
        question_id = question["id"]
        q_claims = claims_by_question.get(question_id, [])
        if not q_claims:
            continue
        grade = question_grade(q_claims, evidence, weak_claim_ids)
        union = sorted(
            {eid for c in q_claims for eid in c["evidence_ids"]}
        )
        if len(union) == 1:
            sole_document_questions.setdefault(union[0], []).append(
                question_id
            )
            if grade in ("STRONG", "STANDARD"):
                contents.append(
                    _content(
                        RULE_GRADE_ONE_DOCUMENT,
                        key_part=question_id,
                        claim_id=None,
                        finding_id=None,
                        question_id=question_id,
                        claim=None,
                        evidence_ids=union,
                        severity=ADVISORY,
                        detail={
                            "grade": grade,
                            "evidence": _evidence_view(
                                evidence.get(union[0]), union[0]
                            ),
                            "claim_ids": [c["claim_id"] for c in q_claims],
                        },
                    )
                )

    # -- rule: one_document_carries_questions ------------------------------
    # The concentration the per-question rules cannot see: one
    # document is the sole evidence behind two or more questions'
    # answers, so the draft as a whole leans on a single source.
    for evidence_id, question_ids in sorted(
        sole_document_questions.items()
    ):
        if len(question_ids) < 2:
            continue
        contents.append(
            _content(
                RULE_ONE_DOCUMENT_CARRIES,
                key_part=evidence_id,
                claim_id=None,
                finding_id=None,
                question_id=None,
                claim=None,
                evidence_ids=[evidence_id],
                severity=ADVISORY,
                detail={
                    "evidence": _evidence_view(
                        evidence.get(evidence_id), evidence_id
                    ),
                    "question_ids": question_ids,
                },
            )
        )

    return contents


def sync_challenges(store: Store, case: dict, report: dict) -> dict:
    """Reconcile the computed challenge set with the case's stored
    challenge history and persist the result.

    Identity is content: a computed challenge whose key and
    fingerprint match a live record leaves that record untouched
    (status, answers, and timestamps intact) — so re-running the
    pass on an unchanged draft is a no-op, byte for byte. A live
    record the new draft no longer computes — its target claim
    vanished, or the substance under its key changed — is marked
    SUPERSEDED, never deleted; a changed substance also mints the
    fresh challenge beside it. One exception to "answered stays
    answered": a challenge answered by sending its claim back
    whose claim returns byte-identical on re-investigation opens
    again — the send-back was supposed to change something, and
    the identical claim has not answered the objection.
    """
    case_id = case["case_id"]
    computed = compute_challenges(store, case, report)
    existing = store.list_challenges(case_id)
    history = [r for r in existing if r["status"] == "SUPERSEDED"]
    active = {r["key"]: r for r in existing if r["status"] != "SUPERSEDED"}
    next_number = (
        max(
            (int(r["challenge_id"].split("-")[1]) for r in existing),
            default=0,
        )
        + 1
    )
    kept: list[dict] = []
    minted: list[dict] = []
    superseded: list[dict] = []
    reopened: list[dict] = []
    seen_keys: set[str] = set()
    for content in computed:
        seen_keys.add(content["key"])
        old = active.get(content["key"])
        if old is not None and old["fingerprint"] == content["fingerprint"]:
            if (
                old["status"] == "ANSWERED"
                and old.get("disposition") == "revised"
            ):
                old = {**old, "status": "OPEN", "disposition": None}
                reopened.append(old)
            kept.append(old)
            continue
        if old is not None:
            old = {**old, "status": "SUPERSEDED", "superseded_at": utcnow()}
            superseded.append(old)
            kept.append(old)
        record = {
            **content,
            "challenge_id": f"CHL-{next_number:04d}",
            "case_id": case_id,
            "status": "OPEN",
            "created_at": utcnow(),
            "disposition": None,
            "answers": [],
            "superseded_at": None,
        }
        next_number += 1
        minted.append(record)
        kept.append(record)
    for key, old in active.items():
        if key not in seen_keys:
            old = {**old, "status": "SUPERSEDED", "superseded_at": utcnow()}
            superseded.append(old)
            kept.append(old)
    full = sorted(
        history + kept, key=lambda r: r["challenge_id"]
    )
    store.replace_challenges(case_id, full)
    return {
        "records": full,
        "minted": minted,
        "superseded": superseded,
        "reopened": reopened,
    }
