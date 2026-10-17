-- Verifier records (SQLite local profile). Same tables as the
-- Postgres migration; payloads are TEXT carrying the same JSON.
CREATE TABLE IF NOT EXISTS verifier_reports (
    case_id    TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quarantine (
    quarantine_id TEXT NOT NULL,
    case_id       TEXT NOT NULL,
    claim_id      TEXT NOT NULL,
    status        TEXT NOT NULL,
    payload       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (case_id, quarantine_id)
);

CREATE INDEX IF NOT EXISTS quarantine_case_idx ON quarantine (case_id);
