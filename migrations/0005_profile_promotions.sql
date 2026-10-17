-- Profile regression gate records: evaluations and promotions.
-- Both tables are append-only — the store exposes inserts and reads
-- and no update or delete for either, because an evaluation or a
-- promotion that could be edited after the fact would prove
-- nothing. profile_evaluations holds one candidate's N gate runs
-- plus the incumbent baseline they were judged against, with the
-- candidate profile file's SHA-256 snapshotted at evaluation time.
-- profile_promotions holds the decisions: which evaluation a
-- promotion rests on, who decided, and (for a rollback) which
-- earlier record it reverses. The active profile is whatever the
-- newest promotion record names; there is no other pointer.
CREATE TABLE IF NOT EXISTS profile_evaluations (
    seq            INTEGER NOT NULL,
    evaluation_id  TEXT PRIMARY KEY,
    candidate_id   TEXT NOT NULL,
    candidate_sha  TEXT NOT NULL,
    verdict        TEXT NOT NULL,
    payload        JSONB NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profile_promotions (
    seq            INTEGER NOT NULL,
    promotion_id   TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    candidate_id   TEXT NOT NULL,
    payload        JSONB NOT NULL,
    created_at     TEXT NOT NULL
);
