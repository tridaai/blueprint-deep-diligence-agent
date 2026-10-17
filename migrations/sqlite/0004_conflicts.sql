-- Conflict records (SQLite local profile). Same tables as the
-- Postgres migration; payloads are TEXT carrying the same JSON.
-- A conflict is created when two supported findings assert
-- different values for the same (entity, metric, period) key. The
-- record holds both sides with their evidence and source class; an
-- analyst may mark it NOTED, and it is never deleted or resolved
-- away.
CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT NOT NULL,
    case_id     TEXT NOT NULL,
    status      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (case_id, conflict_id)
);

CREATE INDEX IF NOT EXISTS conflicts_case_idx ON conflicts (case_id);
