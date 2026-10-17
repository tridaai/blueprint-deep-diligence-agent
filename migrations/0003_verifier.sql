-- Verifier records: the independent claim-check pass that runs
-- between composition and the report gate. verifier_reports holds
-- the latest verifier report per case (checked / supported /
-- quarantined, by reason). quarantine holds one row per claim the
-- verifier removed from the report body, with its reason; the
-- analyst may send a quarantined claim back for re-investigation or
-- drop it, but can never approve it into the report.
CREATE TABLE IF NOT EXISTS verifier_reports (
    case_id    TEXT PRIMARY KEY,
    payload    JSONB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quarantine (
    quarantine_id TEXT NOT NULL,
    case_id       TEXT NOT NULL,
    claim_id      TEXT NOT NULL,
    status        TEXT NOT NULL,
    payload       JSONB NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (case_id, quarantine_id)
);

CREATE INDEX IF NOT EXISTS quarantine_case_idx ON quarantine (case_id);
