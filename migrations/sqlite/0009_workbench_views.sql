-- Analyst workbench views (SQLite local profile). Same table as
-- the Postgres migration; payloads are TEXT carrying the same
-- JSON. A view is a saved queue filter (act needed, assignee,
-- priority) under an analyst's name — working state for the
-- desk, never a case record and never a decision.
CREATE TABLE IF NOT EXISTS workbench_views (
    view_id     TEXT PRIMARY KEY,
    analyst     TEXT NOT NULL,
    name        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
