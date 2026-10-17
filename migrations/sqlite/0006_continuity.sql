-- Long-case continuity records (SQLite local profile). Same
-- tables as the Postgres migration; payloads are TEXT carrying the
-- same JSON. execution_checkpoints holds one row per case — the
-- execution record written as waves complete — so an interrupted
-- execution can be detected and resumed from the record.
-- case_notes are the desk's memory of a case: analyst context,
-- stored apart from evidence and never citable. report_snapshots
-- preserve a draft a plan revision supersedes.
CREATE TABLE IF NOT EXISTS execution_checkpoints (
    case_id     TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_notes (
    note_id     TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    analyst     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS report_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
