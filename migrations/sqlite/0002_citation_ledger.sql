-- Citation ledger (SQLite local profile). Same table as the Postgres
-- migration; the payload is TEXT carrying the same JSON.
CREATE TABLE IF NOT EXISTS citation_ledger (
    case_id     TEXT NOT NULL,
    claim_id    TEXT NOT NULL,
    finding_id  TEXT NOT NULL,
    question_id TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (case_id, claim_id)
);

CREATE INDEX IF NOT EXISTS citation_ledger_case_idx ON citation_ledger (case_id);
