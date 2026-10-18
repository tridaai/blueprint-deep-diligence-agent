-- Operations digests (SQLite local profile). Same table as the
-- Postgres migration; payloads are TEXT carrying the same JSON.
-- One row per (day, version), append-only: regenerating a day's
-- digest mints the next version beside the stored ones, so an
-- earlier snapshot of the day is never rewritten by a later one.
CREATE TABLE IF NOT EXISTS operations_digests (
    day          TEXT NOT NULL,
    version      INTEGER NOT NULL,
    payload      TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    generated_by TEXT NOT NULL,
    PRIMARY KEY (day, version)
);
