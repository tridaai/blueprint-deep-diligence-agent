-- Source governance records: the source registry, evidence
-- redactions and their claim flags, and case tombstones.
-- sources is the governed registry every evidence item's source
-- label must resolve against — one row per source name, payload
-- carrying kind, owner, usage basis, and the record of the acts
-- that registered or amended it. redactions records an evidence
-- item's redaction: the original content hash, the reason, and
-- the analyst — never the content itself. redaction_flags are the
-- claims in a live report that cite redacted evidence; a report
-- cannot be approved while one stands unacknowledged.
-- case_tombstones is all that survives a case deletion: the
-- minimal record that the case existed and who ended it, and why.
CREATE TABLE IF NOT EXISTS sources (
    name        TEXT PRIMARY KEY,
    payload     JSONB NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS redactions (
    redaction_id TEXT PRIMARY KEY,
    case_id      TEXT NOT NULL,
    evidence_id  TEXT NOT NULL,
    payload      JSONB NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS redaction_flags (
    flag_id     TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    claim_id    TEXT NOT NULL,
    status      TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_tombstones (
    case_id     TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    payload     JSONB NOT NULL,
    deleted_at  TEXT NOT NULL
);
