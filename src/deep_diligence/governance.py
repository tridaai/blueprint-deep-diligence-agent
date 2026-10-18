"""Source governance: the registry, provenance, retention, redaction.

Until this round, an evidence item's `source` was a free-text label
that nothing checked — a document could name any source at all, a
brief could allow a source that did not exist, and there was no
answer to "where did this report's facts come from" better than
reading the citations one by one. Governance makes sources
first-class:

* **Registry.** Every source is a governed record — name, kind,
  owner, usage basis, notes — and evidence ingestion (the intake
  corpus and the analyst upload path alike) refuses a source the
  registry does not hold, naming it. Registering a source is a
  recorded act: the record carries who registered it and when.
* **Brief policy.** A brief's `allowed_sources` must resolve —
  each entry is a registered source name, a corpus document, or a
  document kind — or intake refuses the brief, naming the entry.
  Fail closed: a typo'd source scopes a case to nothing silently
  no longer; it stops the brief.
* **Provenance.** Per case, the audit view: every evidence item
  with its source's governance record, its content hash, how it
  arrived, and the findings and report claims that cite it.
* **Retention + legal hold.** A case carries a retention class.
  The sweep lists cases past their retention period for an
  analyst's decision; it deletes nothing itself. Deletion is an
  explicit recorded act (analyst + reason) that removes the case's
  data and objects and leaves only a tombstone. A case under
  legal hold refuses deletion in code.
* **Redaction.** An analyst may redact an evidence item for a
  recorded reason: the stored bytes are replaced by a marker, the
  redaction record keeps the original hash, and claims in the live
  report that cite the item are flagged — the report cannot be
  approved until each is acknowledged or sent back.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .corpus import CORPUS
from .store import Store, utcnow

SOURCE_KINDS = ("filing", "news", "internal dataset", "web")
USAGE_BASES = ("public", "licensed", "client-provided")
RETENTION_CLASSES = ("standard", "legal_hold")

# The bundled corpus's source labels, classified. The corpus is the
# desk's standing evidence universe, so its sources are registered
# at seed; everything beyond the corpus must be registered by an
# analyst before a document may cite it.
_CORPUS_SOURCE_SEED: dict[str, tuple[str, str, str, str]] = {
    # label: (kind, owner, usage basis, notes)
    "Company filing (annual report)": (
        "filing", "Subject company (issuer)", "public",
        "The subject's own audited annual filing.",
    ),
    "Company preliminary release (unaudited)": (
        "filing", "Subject company (issuer)", "public",
        "Unaudited preliminary figures from the subject; superseded "
        "by the annual filing where the two disagree.",
    ),
    "Company regulatory summary": (
        "filing", "Subject company (issuer)", "public",
        "The subject's summary of its regulatory position.",
    ),
    "Company operations brief": (
        "internal dataset", "Subject company (issuer)", "client-provided",
        "Operational detail the subject provides to the desk during "
        "an engagement.",
    ),
    "Investor update letter": (
        "filing", "Subject company (issuer)", "public",
        "The subject's letter to its investors.",
    ),
    "Company investor presentation": (
        "filing", "Subject company (issuer)", "public",
        "The subject's results presentation to investors; its "
        "tables restate the filing, its commentary is the "
        "company's own framing.",
    ),
    "Company investor relations release": (
        "news", "Subject company (issuer)", "public",
        "A release from the subject's investor-relations office.",
    ),
    "Vendor product documentation": (
        "web", "Subject company (vendor)", "public",
        "The vendor's published product documentation.",
    ),
    "Vendor pricing sheet": (
        "web", "Subject company (vendor)", "client-provided",
        "Commercial terms the vendor supplies for an engagement.",
    ),
    "Trade press article": (
        "news", "Trade press publisher", "public",
        "Coverage in the sector's trade press.",
    ),
    "Trade press profile": (
        "news", "Trade press publisher", "public",
        "A profile piece in the sector's trade press.",
    ),
    "Regional business journal": (
        "news", "Regional business journal publisher", "public",
        "Coverage in a regional business journal.",
    ),
    "Health-sector newsletter": (
        "news", "Sector newsletter publisher", "licensed",
        "A subscription sector newsletter, held under licence.",
    ),
    "Industry analyst briefing": (
        "news", "Industry analyst firm", "licensed",
        "A research briefing from an industry analyst firm, held "
        "under licence.",
    ),
    "Independent evaluation summary": (
        "internal dataset", "Independent evaluator", "licensed",
        "A third-party evaluation commissioned for diligence, held "
        "under licence.",
    ),
}


def seed_sources(store: Store) -> list[dict]:
    """Register the corpus's sources, once. Idempotent: a source
    the registry already holds (seeded, or registered by an analyst
    with richer detail) is left exactly as it stands."""
    seeded: list[dict] = []
    for label in sorted({doc.source for doc in CORPUS}):
        if store.get_source(label) is not None:
            continue
        kind, owner, basis, notes = _CORPUS_SOURCE_SEED.get(
            label, ("web", "Unknown publisher", "public", "")
        )
        record = {
            "name": label,
            "kind": kind,
            "owner": owner,
            "usage_basis": basis,
            "notes": notes,
            "registered_by": "system (corpus seed)",
            "registered_at": utcnow(),
            "history": [
                {
                    "action": "registered",
                    "analyst": "system (corpus seed)",
                    "at": utcnow(),
                }
            ],
        }
        seeded.append(store.add_source(record))
    return seeded


def require_registered_source(store: Store, source: str) -> dict:
    """The ingestion gate: a source the registry does not hold
    refuses the document, with the source named. Fail closed —
    there is no unregistered evidence."""
    record = store.get_source(source)
    if record is None:
        raise ValueError(
            f"source {source!r} is not in the source registry; "
            "register it before evidence may cite it"
        )
    return record


def brief_source_problems(store: Store, brief: dict) -> list[str]:
    """Resolve a brief's allowed_sources against the world: each
    entry must be a registered source name, a corpus document
    slug, or a corpus document kind (the three things the intake
    filter matches on). Anything else names nothing — a typo
    that would scope the case to a silent subset — and refuses
    the brief."""
    slugs = {doc.slug for doc in CORPUS}
    kinds = {doc.kind for doc in CORPUS}
    problems: list[str] = []
    for entry in brief.get("allowed_sources") or []:
        if (
            store.get_source(entry) is None
            and entry not in slugs
            and entry not in kinds
        ):
            problems.append(
                f"allowed source {entry!r} is not a registered "
                "source, a corpus document, or a document kind"
            )
    return problems


def redaction_marker(ev: dict, analyst: str, reason: str, at: str) -> str:
    """What replaces a redacted item's bytes in the object store.
    The marker is deliberately content-free: it names the act, not
    the document, so nothing downstream can mistake it for
    evidence — the verifier treats a redacted item as missing."""
    return (
        f"[REDACTED — evidence {ev['evidence_id']} ({ev['title']}) "
        f"was redacted by {analyst} at {at}. Reason: {reason}. "
        "The original content was removed under the desk's "
        "source-governance policy; the redaction record holds the "
        "original content hash. This marker is not evidence and "
        "supports no claim.]\n"
    )


def retention_due(case: dict, retention_days: int) -> datetime:
    created = datetime.fromisoformat(case["created_at"])
    return created + timedelta(days=retention_days)


def parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def case_provenance(store: Store, case: dict) -> dict:
    """The audit answer to "where did this report's facts come
    from": every evidence item in the case, its source's
    governance record, its hash, how it arrived (corpus intake or
    an analyst's upload), and the findings and report claims that
    cite it (claims read from the citation ledger)."""
    case_id = case["case_id"]
    findings = store.list_findings(case_id)
    ledger = store.list_ledger(case_id)
    cited_by_findings: dict[str, list[str]] = {}
    for finding in findings:
        for evidence_id in finding["evidence_ids"]:
            cited_by_findings.setdefault(evidence_id, []).append(
                finding["finding_id"]
            )
    cited_by_claims: dict[str, list[str]] = {}
    for entry in ledger:
        for citation in entry["citations"]:
            cited_by_claims.setdefault(citation["evidence_id"], []).append(
                entry["claim_id"]
            )
    items: list[dict] = []
    for ev in store.list_evidence(case_id):
        items.append(
            {
                "evidence_id": ev["evidence_id"],
                "title": ev["title"],
                "doc_slug": ev["doc_slug"],
                "kind": ev["kind"],
                "doc_date": ev["doc_date"],
                "source": ev["source"],
                "source_registry": store.get_source(ev["source"]),
                "sha256": ev["sha256"],
                "ingestion": ev.get("origin") or "corpus_intake",
                "retrieved_at": ev["retrieved_at"],
                "redacted": bool(ev.get("redacted")),
                "cited_by_findings": cited_by_findings.get(
                    ev["evidence_id"], []
                ),
                "cited_by_claims": cited_by_claims.get(
                    ev["evidence_id"], []
                ),
            }
        )
    return {
        "case_id": case_id,
        "subject": case["subject"],
        "generated_at": utcnow(),
        "items": items,
    }


def case_source_policy(store: Store, case: dict) -> dict:
    """The sources a case runs on, for the analyst at the plan
    gate: what the brief allowed (each entry resolved to the
    governance records it admits) and what the case's evidence
    actually came from."""
    brief = case["brief"]
    subject_docs = [d for d in CORPUS if d.company == case["subject_ref"]]
    allowed: list[dict] = []
    for entry in brief.get("allowed_sources") or []:
        record = store.get_source(entry)
        if record is not None:
            allowed.append(
                {"entry": entry, "resolves_to": "source", "sources": [record]}
            )
            continue
        matched = [
            d for d in subject_docs if d.slug == entry or d.kind == entry
        ]
        names = sorted({d.source for d in matched})
        allowed.append(
            {
                "entry": entry,
                "resolves_to": (
                    "document" if any(d.slug == entry for d in subject_docs)
                    else "kind"
                ),
                "sources": [
                    store.get_source(name) for name in names
                ],
            }
        )
    used_names = sorted(
        {ev["source"] for ev in store.list_evidence(case["case_id"])}
    )
    return {
        "case_id": case["case_id"],
        "allowed": allowed,
        "used": [store.get_source(name) for name in used_names],
    }
