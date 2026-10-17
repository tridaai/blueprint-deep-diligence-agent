-- Core records for the Deep Diligence Agent governance shell (SQLite
-- local profile). Same tables as the Postgres migration; payloads are
-- TEXT carrying the same JSON.
CREATE TABLE IF NOT EXISTS cases (
    case_id    TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    subject    TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    case_id    TEXT NOT NULL,
    version    INTEGER NOT NULL,
    status     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (case_id, version)
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    doc_slug    TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (case_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id  TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    question_id TEXT NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (case_id, finding_id)
);

CREATE TABLE IF NOT EXISTS reports (
    case_id TEXT PRIMARY KEY,
    status  TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_files (
    case_id    TEXT NOT NULL,
    path       TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    size       INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (case_id, path)
);

CREATE TABLE IF NOT EXISTS audit (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    ts      TEXT NOT NULL,
    event   TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS audit_case_idx ON audit (case_id, id);
CREATE INDEX IF NOT EXISTS evidence_case_idx ON evidence (case_id);
CREATE INDEX IF NOT EXISTS findings_case_idx ON findings (case_id);
