-- Citation ledger: one row per composed claim, mapping the claim to
-- the finding behind it and the evidence it cites, with each cited
-- item's content hash recorded at citation time (composition). The
-- report gate re-verifies the live evidence bytes against these
-- hashes; a mismatch flags the report EVIDENCE_DRIFT.
CREATE TABLE IF NOT EXISTS citation_ledger (
    case_id     TEXT NOT NULL,
    claim_id    TEXT NOT NULL,
    finding_id  TEXT NOT NULL,
    question_id TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (case_id, claim_id)
);

CREATE INDEX IF NOT EXISTS citation_ledger_case_idx ON citation_ledger (case_id);
