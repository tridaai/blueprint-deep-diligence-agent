-- Long-case continuity records: execution checkpoints, analyst
-- case notes, and report snapshots.
-- execution_checkpoints holds one row per case: the execution
-- record written as waves complete (per-workstream status, the
-- finding IDs recorded so far, wave position, and a budget
-- snapshot folded from the ledger), so an execution whose process
-- died can be detected and resumed from the record, not from any
-- harness memory. case_notes are the desk's memory of a case —
-- analyst-authored context, stored apart from evidence and never
-- citable. report_snapshots preserve a draft a plan revision
-- supersedes, so revising a plan mid-case never destroys the
-- report the desk already had.
CREATE TABLE IF NOT EXISTS execution_checkpoints (
    case_id     TEXT PRIMARY KEY,
    payload     JSONB NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_notes (
    note_id     TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    analyst     TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS report_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TEXT NOT NULL
);
