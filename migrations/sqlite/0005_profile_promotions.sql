-- Profile regression gate records (SQLite local profile). Same
-- tables as the Postgres migration; payloads are TEXT carrying the
-- same JSON. Both tables are append-only: profile_evaluations
-- holds one candidate's N gate runs plus the incumbent baseline
-- they were judged against, with the candidate file's SHA-256
-- snapshotted at evaluation time; profile_promotions holds the
-- promotion and rollback decisions. The active profile is whatever
-- the newest promotion record names; there is no other pointer.
CREATE TABLE IF NOT EXISTS profile_evaluations (
    seq            INTEGER NOT NULL,
    evaluation_id  TEXT PRIMARY KEY,
    candidate_id   TEXT NOT NULL,
    candidate_sha  TEXT NOT NULL,
    verdict        TEXT NOT NULL,
    payload        TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profile_promotions (
    seq            INTEGER NOT NULL,
    promotion_id   TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    candidate_id   TEXT NOT NULL,
    payload        TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
