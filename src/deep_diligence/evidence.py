"""Evidence ingestion: corpus documents become citable case evidence.

Every item gets an ID (EV-0001, ...), its source label, the retrieval
time, and a SHA-256 of its bytes. The bytes live in the object store;
the metadata lives in the case store. Findings may only cite evidence
IDs that exist in the case — that is checked where findings are
recorded, and again where the report is composed.

Source governance binds here, at the door: every item's source must
resolve in the source registry (governance.py), on the corpus path
and the analyst upload path alike. An unregistered source refuses
the document, with the source named — there is no unregistered
evidence.
"""

from __future__ import annotations

from .corpus import docs_for_subject
from .governance import require_registered_source
from .objects import ObjectStore, sha256_hex
from .store import Store, utcnow


def ingest_case_evidence(store: Store, objects: ObjectStore, case: dict) -> list[dict]:
    case_id = case["case_id"]
    allowed = set(case["brief"].get("allowed_sources") or [])
    ingested: list[dict] = []
    for doc in docs_for_subject(case["subject_ref"]):
        if (
            allowed
            and doc.slug not in allowed
            and doc.kind not in allowed
            and doc.source not in allowed
        ):
            continue
        require_registered_source(store, doc.source)
        data = doc.text.encode("utf-8")
        object_key = f"cases/{case_id}/evidence/{doc.slug}.md"
        objects.put(object_key, data)
        ev = {
            "evidence_id": f"EV-{store.count_evidence(case_id) + 1:04d}",
            "case_id": case_id,
            "doc_slug": doc.slug,
            "title": doc.title,
            "kind": doc.kind,
            "source": doc.source,
            "doc_date": doc.date,
            "sha256": sha256_hex(data),
            "object_key": object_key,
            "retrieved_at": utcnow(),
            "origin": "corpus_intake",
        }
        ingested.append(store.add_evidence(ev))
    return ingested


def evidence_text(objects: ObjectStore, ev: dict) -> str:
    return objects.get(ev["object_key"]).decode("utf-8")


def add_evidence_document(
    store: Store,
    objects: ObjectStore,
    case: dict,
    *,
    slug: str,
    title: str,
    kind: str,
    source: str,
    doc_date: str,
    text: str,
) -> dict:
    """An analyst-supplied document becomes case evidence, by the
    same rules as corpus ingestion: bytes into the object store,
    metadata with a SHA-256 into the case store, an ID from the
    case's sequence. The record names its origin — evidence that
    arrived mid-case is distinguishable from the intake corpus in
    the record, not just in the audit trail. The source gate is
    the same one the corpus passes: the registry must hold the
    source the upload names, or the document is refused."""
    require_registered_source(store, source)
    case_id = case["case_id"]
    data = text.encode("utf-8")
    object_key = f"cases/{case_id}/evidence/{slug}.md"
    objects.put(object_key, data)
    ev = {
        "evidence_id": f"EV-{store.count_evidence(case_id) + 1:04d}",
        "case_id": case_id,
        "doc_slug": slug,
        "title": title,
        "kind": kind,
        "source": source,
        "doc_date": doc_date,
        "sha256": sha256_hex(data),
        "object_key": object_key,
        "retrieved_at": utcnow(),
        "origin": "analyst_upload",
    }
    return store.add_evidence(ev)
