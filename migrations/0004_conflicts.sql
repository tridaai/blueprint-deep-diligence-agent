-- Conflict records: the deterministic post-verification pass that
-- keys supported findings by (entity, metric, period) and records
-- where two of them assert different values for the same key.
-- conflicts holds one row per disagreement, with both claims, both
-- values, their evidence, and each side's source class. The analyst
-- may mark a conflict NOTED; there is no delete or resolve path —
-- a conflict stays in the report permanently.
CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    status      TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (case_id, conflict_id)
);

CREATE INDEX IF NOT EXISTS conflicts_case_idx ON conflicts (case_id);
