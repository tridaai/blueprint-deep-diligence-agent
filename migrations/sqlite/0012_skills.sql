-- The skills library (SQLite local profile). Same table as the
-- Postgres migration; payloads are TEXT carrying the same JSON.
-- One row per (name, version), append-only: a published version
-- is what approved plans pin and what findings cite as their
-- provenance, so it is never edited — a change is a new version
-- beside the old. The shipped artifacts in skills/ are seeded
-- here at startup; later versions arrive by the publish act,
-- whose record (analyst, changelog, timestamp) is the row itself.
CREATE TABLE IF NOT EXISTS skills (
    name         TEXT NOT NULL,
    version      INTEGER NOT NULL,
    payload      TEXT NOT NULL,
    published_at TEXT NOT NULL,
    PRIMARY KEY (name, version)
);
