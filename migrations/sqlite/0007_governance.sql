-- Source governance records (SQLite local profile). Same tables
-- as the Postgres migration; payloads are TEXT carrying the same
-- JSON. sources is the governed registry every evidence item's
-- source label must resolve against. redactions records an
-- evidence redaction (original hash, reason, analyst — never the
-- content). redaction_flags are the live report claims citing
-- redacted evidence; approval is blocked while one is
-- unacknowledged. case_tombstones is all that survives a case
-- deletion: that the case existed, and who ended it, and why.
CREATE TABLE IF NOT EXISTS sources (
    name        TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS redactions (
    redaction_id TEXT PRIMARY KEY,
    case_id      TEXT NOT NULL,
    evidence_id  TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS redaction_flags (
    flag_id     TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    claim_id    TEXT NOT NULL,
    status      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_tombstones (
    case_id     TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    deleted_at  TEXT NOT NULL
);
