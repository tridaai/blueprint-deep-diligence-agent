"""The citation ledger: a report's claims as a stored, checkable record.

Until this round, citations lived only inside findings, and the
composer checked that a cited evidence ID still resolved. The ledger
makes the claim→evidence mapping a first-class artifact of the case:
when a report is composed, every claim it carries gets a ledger entry
— claim ID, the finding behind it, and for each cited evidence item
the SHA-256 of its bytes *at citation time* (read live from the object
store while composing).

That recorded hash is what makes citation integrity checkable later.
`verify_ledger` re-reads the evidence bytes and compares them with
the hashes in the ledger (and with the hash recorded at ingestion).
Any mismatch means the evidence changed after the report cited it —
EVIDENCE_DRIFT — and the report gate refuses approval until a fresh
verification comes back clean. All of this is deterministic code;
no model is involved in checking citations.
"""

from __future__ import annotations

from .objects import ObjectStore, sha256_hex
from .store import Store, utcnow


def build_ledger(
    store: Store, objects: ObjectStore, case_id: str, claims: list[dict]
) -> list[dict]:
    """Persist one ledger entry per composed claim (replacing any prior
    ledger for the case, since a re-composition mints a new report)."""
    entries: list[dict] = []
    for claim in claims:
        citations: list[dict] = []
        for evidence_id in claim["evidence_ids"]:
            ev = store.get_evidence(case_id, evidence_id)
            citations.append(
                {
                    "evidence_id": evidence_id,
                    "doc_slug": ev["doc_slug"],
                    "source": ev["source"],
                    "title": ev["title"],
                    # The content hash at citation time: the live bytes,
                    # hashed now, not the metadata recorded at ingestion.
                    "sha256": sha256_hex(objects.get(ev["object_key"])),
                    "ingested_sha256": ev["sha256"],
                }
            )
        entries.append(
            {
                "claim_id": claim["claim_id"],
                "case_id": case_id,
                "finding_id": claim["finding_id"],
                "question_id": claim["question_id"],
                "claim": claim["claim"],
                "fact_ref": claim.get("fact_ref"),
                "citations": citations,
                "cited_at": utcnow(),
            }
        )
    return store.replace_ledger(case_id, entries)


def verify_ledger(store: Store, objects: ObjectStore, case_id: str) -> dict:
    """Re-verify every ledger citation against the evidence store.

    Deterministic: for each citation, the evidence must still exist
    and its current bytes must hash to the value recorded at citation
    time (which must also match the hash recorded at ingestion).
    """
    entries = store.list_ledger(case_id)
    drift: list[dict] = []
    checked = 0
    for entry in entries:
        for citation in entry["citations"]:
            checked += 1
            evidence_id = citation["evidence_id"]
            base = {
                "claim_id": entry["claim_id"],
                "finding_id": entry["finding_id"],
                "evidence_id": evidence_id,
                "cited_sha256": citation["sha256"],
            }
            ev = store.get_evidence(case_id, evidence_id)
            if ev is None:
                drift.append({**base, "current_sha256": None, "reason": "evidence_missing"})
                continue
            try:
                current = sha256_hex(objects.get(ev["object_key"]))
            except Exception:
                current = None
            if current != citation["sha256"]:
                drift.append(
                    {**base, "current_sha256": current, "reason": "content_changed"}
                )
    return {
        "status": "EVIDENCE_DRIFT" if drift else "OK",
        "entries": len(entries),
        "citations_checked": checked,
        "drift": drift,
        "verified_at": utcnow(),
    }
