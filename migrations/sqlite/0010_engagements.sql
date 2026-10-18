-- Portfolio mode (SQLite local profile). Same table as the
-- Postgres migration; payloads are TEXT carrying the same JSON.
-- An engagement groups cases under one client mandate: the
-- record carries its name, client label, owner, and its act
-- history, while membership lives on each case's own record —
-- so the portfolio view is a fold over the cases, and the
-- engagement itself holds no case state and no decision.
CREATE TABLE IF NOT EXISTS engagements (
    engagement_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    payload       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
