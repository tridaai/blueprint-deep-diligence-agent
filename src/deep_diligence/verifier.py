"""The verifier: an independent re-check of every claim in a report.

Composition checks that a citation *resolves* and, via the ledger,
that the cited bytes have not changed since. Neither checks that the
evidence actually *says* what the claim says. That is the verifier's
job, and it runs after composition, before the report reaches the
analyst gate.

Independence is structural, not a prompt instruction. The verifier
receives only the case store and the object store. It re-reads the
persisted report, and for every claim it fetches the cited evidence
bytes directly from the object store by ID and re-derives support
itself. It never reads the composer's in-memory structures or the
citation ledger — delete the ledger and the verifier's verdicts do
not change. Redacted evidence (governance.py) is treated as
missing: its bytes are a marker, and a marker supports no claim.

The gate checks are deterministic: every figure (money, percentages,
counts) and every date a claim states must appear in the evidence it
cites, and every named entity a claim mentions must be mentioned by
that evidence. A claim that fails is quarantined — removed from the
report body and recorded with its reason. When a real model is
configured, a model check runs alongside as an advisory signal: it
sees only the claim and its cited evidence (no findings, no other
claims), and its verdict is recorded on the check — but the
deterministic checks decide quarantine, so the gate holds offline.

Figures and dates are judged against the cited document's own fact
ledger — its "Key facts" section — when the document carries one.
A document's prose can say many things its ledger does not state:
a pro-forma aside, a quoted third party, a number its own results
table contradicts. The ledger is what the document asserts as fact,
so a figure that appears only in prose does not support a claim.
Documents with no ledger (an analyst's free-text upload) are judged
against their full text, as before. Named entities are still judged
against the full text: a name is context, not a figure.
"""

from __future__ import annotations

import re
import uuid

from .objects import ObjectStore
from .store import Store, utcnow

MONEY = re.compile(r"\$[\d,]+(?:\.\d+)?[MBK]?")
PERCENT = re.compile(r"\d+(?:\.\d+)?%")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONTH_YEAR = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b"
)
LEDGER_HEADING = "## Key facts"
LEDGER_LINE = re.compile(r"^- \[(F-[A-Z]+-\d+)\] (.*)$", re.MULTILINE)
CAP_TOKEN = re.compile(r"^[A-Z][A-Za-z0-9'’-]*$|^[A-Z]{2,}$")
CODE_TOKEN = re.compile(r"^[A-Z][A-Z0-9]*[-]?\d[\w-]*$|^\d{3,}$")
CONNECTORS = {"of", "the", "&", "and"}
REASON_ORDER = (
    "evidence_missing",
    "figure_not_in_evidence",
    "date_not_in_evidence",
    "entity_mismatch",
)


def ledger_facts(text: str) -> dict[str, str]:
    """The facts a document asserts, as {fact_id: fact_text}, read
    from its Key facts ledger. A document's ledger is the section
    the document format appends last, so when a body carries an
    earlier section with the same heading — a planted lookalike —
    the final one is the ledger and the lookalike is prose. A
    document with no ledger section asserts no facts and returns
    an empty dict."""
    index = text.rfind(f"\n{LEDGER_HEADING}\n")
    if index == -1:
        if not text.startswith(f"{LEDGER_HEADING}\n"):
            return {}
        section = text
    else:
        section = text[index + 1:]
    return {m.group(1): m.group(2).strip() for m in LEDGER_LINE.finditer(section)}


def ledger_text(text: str) -> str | None:
    """The document's ledger section as one text, or None when the
    document carries no ledger."""
    facts = ledger_facts(text)
    if not facts:
        return None
    return "\n".join(facts.values())


def _support_texts(texts: list[str]) -> list[str]:
    """What figures and dates are judged against: each cited
    document's fact ledger where it carries one, its full text
    where it does not."""
    return [ledger_text(text) or text for text in texts]


def _figures(text: str) -> list[str]:
    tokens: list[str] = []
    covered: list[tuple[int, int]] = []
    for pattern in (MONEY, PERCENT):
        for match in pattern.finditer(text):
            covered.append(match.span())
            if match.group() not in tokens:
                tokens.append(match.group())
    for match in NUMBER.finditer(text):
        # A bare number inside a money or percent token ($198M, 11%)
        # is the same figure, not a second one.
        if any(start <= match.start() < end for start, end in covered):
            continue
        if match.group() not in tokens:
            tokens.append(match.group())
    return tokens


def _dates(text: str) -> list[str]:
    tokens: list[str] = []
    for pattern in (ISO_DATE, MONTH_YEAR):
        for token in pattern.findall(text):
            if token not in tokens:
                tokens.append(token)
    return tokens


def _present(token: str, texts: list[str]) -> bool:
    """A token is present if it appears in the cited evidence, with a
    comma-insensitive fallback (4,800 in a claim, 4800 in a source)."""
    for text in texts:
        if token in text:
            return True
    bare = token.replace(",", "")
    if bare != token:
        return any(bare in text.replace(",", "") for text in texts)
    return False


def _entities(text: str) -> list[str]:
    """Named entities a claim mentions: multi-word proper-noun phrases
    (Helvion Grid Services), plus acronyms and product codes (FDA,
    AF-40, HGX-9). Single capitalised words are too noisy to check —
    they begin sentences — so they are not treated as entities."""
    words = [w.strip(" \t\"'()[]{}<>.,;:!?") for w in text.split()]
    entities: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 2:
            phrase = " ".join(run)
            if phrase not in entities:
                entities.append(phrase)
        elif len(run) == 1 and CODE_TOKEN.match(run[0]):
            if run[0] not in entities:
                entities.append(run[0])
        run.clear()

    for word in words:
        if not word:
            continue
        if CAP_TOKEN.match(word) or CODE_TOKEN.match(word):
            run.append(word)
        elif word.lower() in CONNECTORS and run:
            run.append(word)
        else:
            flush()
    flush()
    # Trailing connectors swept into a phrase are not part of the name.
    return [e for e in entities if e.split()[-1].lower() not in CONNECTORS]


def check_claim(claim: str, evidence_texts: list[str]) -> list[dict]:
    """Deterministic support check for one claim against the texts of
    the evidence it cites. Returns a list of failures (empty = the
    claim is supported by its citations)."""
    failures: list[dict] = []
    support = _support_texts(evidence_texts)
    for token in _figures(claim):
        if not _present(token, support):
            failures.append(
                {
                    "reason": "figure_not_in_evidence",
                    "detail": f"figure {token} does not appear in the cited evidence",
                }
            )
    for token in _dates(claim):
        if not _present(token, support):
            failures.append(
                {
                    "reason": "date_not_in_evidence",
                    "detail": f"date {token} does not appear in the cited evidence",
                }
            )
    joined = "\n".join(evidence_texts)
    for entity in _entities(claim):
        if entity in joined:
            continue
        tokens = [t for t in entity.split() if t.lower() not in CONNECTORS]
        if tokens and all(t in joined for t in tokens):
            continue
        failures.append(
            {
                "reason": "entity_mismatch",
                "detail": f"{entity} is not mentioned in the cited evidence",
            }
        )
    return failures


def _model_check(model, claim: str, evidence_texts: list[str]) -> str:
    """Advisory model verdict on one claim. The model sees ONLY the
    claim and the evidence it cites — no findings, no other claims,
    no verifier reasoning. Recorded on the check; never the gate."""
    from langchain_core.messages import HumanMessage

    prompt = (
        "You are verifying one claim from a diligence report against "
        "the evidence it cites. You are given ONLY the claim and the "
        "cited evidence.\n\n"
        f"CLAIM:\n{claim}\n\n"
        "CITED EVIDENCE:\n" + "\n\n".join(evidence_texts) + "\n\n"
        "Answer with exactly one word: SUPPORTED if every material "
        "fact in the claim is stated in the cited evidence, otherwise "
        "UNSUPPORTED."
    )
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content
        text = content if isinstance(content, str) else str(content)
        verdict = text.strip().upper()
        if verdict.startswith("UNSUPPORTED"):
            return "unsupported"
        if verdict.startswith("SUPPORTED"):
            return "supported"
    except Exception:
        pass
    return "unavailable"


def verify_case(
    store: Store,
    objects: ObjectStore,
    case_id: str,
    *,
    model=None,
    model_name: str | None = None,
) -> dict:
    """Re-check every claim in the case's persisted report against
    its cited evidence. Persists the verifier report and the
    quarantine set, and returns the verifier report.

    The report's claims come from the store (the persisted record),
    and every evidence text is fetched here, from the object store,
    by evidence ID — nothing is taken from the composer or the
    citation ledger.
    """
    report = store.get_report(case_id)
    if report is None:
        raise ValueError(f"case {case_id} has no report to verify")
    run_id = f"VR-{uuid.uuid4().hex[:8]}"
    checks: list[dict] = []
    quarantine: list[dict] = []
    by_reason: dict[str, int] = {}
    for claim in report["claims"]:
        failures: list[dict] = []
        texts: list[str] = []
        for evidence_id in claim["evidence_ids"]:
            ev = store.get_evidence(case_id, evidence_id)
            if ev is not None and ev.get("redacted"):
                # A redacted item's bytes are a governance marker,
                # not evidence: the claim is unverifiable against
                # it, exactly as if the item were missing — the
                # marker's own text must never "support" a figure.
                failures.append(
                    {
                        "reason": "evidence_missing",
                        "detail": f"cited evidence {evidence_id} has been redacted; a redaction marker cannot support a claim",
                    }
                )
                continue
            data: bytes | None = None
            if ev is not None:
                try:
                    data = objects.get(ev["object_key"])
                except Exception:
                    data = None
            if data is None:
                failures.append(
                    {
                        "reason": "evidence_missing",
                        "detail": f"cited evidence {evidence_id} could not be read from the evidence store",
                    }
                )
            else:
                texts.append(data.decode("utf-8", errors="replace"))
        if texts:
            failures.extend(check_claim(claim["claim"], texts))
        model_verdict = None
        if model is not None:
            model_verdict = (
                _model_check(model, claim["claim"], texts)
                if texts
                else "unavailable"
            )
        reasons = sorted(
            {f["reason"] for f in failures},
            key=REASON_ORDER.index,
        )
        verdict = "quarantined" if failures else "supported"
        checks.append(
            {
                "claim_id": claim["claim_id"],
                "finding_id": claim["finding_id"],
                "question_id": claim["question_id"],
                "verdict": verdict,
                "reasons": reasons,
                "failures": failures,
                "model_check": model_verdict,
            }
        )
        if failures:
            primary = reasons[0]
            by_reason[primary] = by_reason.get(primary, 0) + 1
            quarantine.append(
                {
                    "quarantine_id": f"Q-{len(quarantine) + 1:04d}",
                    "case_id": case_id,
                    "claim_id": claim["claim_id"],
                    "finding_id": claim["finding_id"],
                    "question_id": claim["question_id"],
                    "claim": claim["claim"],
                    "evidence_ids": claim["evidence_ids"],
                    "reason": primary,
                    "failures": failures,
                    "status": "QUARANTINED",
                    "verifier_run": run_id,
                    "created_at": utcnow(),
                    "decided_by": None,
                    "decision_note": None,
                    "decided_at": None,
                }
            )
    verifier_report = {
        "case_id": case_id,
        "run_id": run_id,
        "mode": "live-assisted" if model is not None else "offline",
        "model": model_name,
        "checked": len(checks),
        "supported": sum(1 for c in checks if c["verdict"] == "supported"),
        "quarantined": len(quarantine),
        "by_reason": by_reason,
        "checks": checks,
        "verified_at": utcnow(),
    }
    store.save_verifier_report(case_id, verifier_report)
    store.replace_quarantine(case_id, quarantine)
    return verifier_report
