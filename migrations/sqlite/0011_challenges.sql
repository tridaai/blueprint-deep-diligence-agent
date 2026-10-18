-- Challenge records (SQLite local profile). Same table as the
-- Postgres migration; payloads are TEXT carrying the same JSON.
-- A challenge is the challenge pass's structured objection to a
-- supported claim, a question's grade, or the draft's evidence
-- base, computed deterministically from the stored records. The
-- analyst answers each one — rebut, send back, or downgrade — and
-- an unanswered must_answer challenge blocks report approval.
-- Records are never deleted; a superseded challenge stays on the
-- case as the history of the desk's review.
CREATE TABLE IF NOT EXISTS challenges (
    challenge_id TEXT NOT NULL,
    case_id      TEXT NOT NULL,
    status       TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (case_id, challenge_id)
);

CREATE INDEX IF NOT EXISTS challenges_case_idx ON challenges (case_id);
