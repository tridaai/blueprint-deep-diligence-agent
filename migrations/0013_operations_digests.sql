-- Operations digests: the desk's daily operational summary, held
-- as a stored, versioned record. One row per (day, version):
-- generating a day's digest appends the next version beside the
-- ones already stored — a digest is a snapshot the desk took at
-- a moment, and regenerating it later (after the day's cases
-- have moved on) must not rewrite what the earlier snapshot
-- said. The fold itself lives in observability.py and recomputes
-- everything from the case records at generation time; this
-- table is the only thing the round stores.
CREATE TABLE IF NOT EXISTS operations_digests (
    day          TEXT NOT NULL,
    version      INTEGER NOT NULL,
    payload      JSONB NOT NULL,
    generated_at TEXT NOT NULL,
    generated_by TEXT NOT NULL,
    PRIMARY KEY (day, version)
);
