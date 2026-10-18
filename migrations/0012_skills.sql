-- The skills library: governed, versioned method bundles.
-- One row per (name, version); rows are append-only — the store
-- exposes inserts and reads and no update or delete — because a
-- published version is what approved plans pin and what findings
-- cite as their provenance. Editing a version after the fact
-- would rewrite the method past cases already ran under; a change
-- is always a new version beside the old. The shipped artifacts
-- in skills/ are seeded here at startup; later versions arrive by
-- the publish act, whose record (analyst, changelog, timestamp)
-- is the row itself.
CREATE TABLE IF NOT EXISTS skills (
    name         TEXT NOT NULL,
    version      INTEGER NOT NULL,
    payload      JSONB NOT NULL,
    published_at TEXT NOT NULL,
    PRIMARY KEY (name, version)
);
