-- Challenge records: the deterministic second-opinion pass that
-- runs after verification and conflict detection, over the claims
-- that survived both. A challenge names a claim (or a question, or
-- the draft as a whole), the governed rule that fired, and the
-- evidence the rule read. Severity is advisory or must_answer; a
-- report with an unanswered must_answer challenge cannot be
-- approved. Challenges are the desk's internal review: they are
-- never part of the delivered report, and they are never deleted —
-- when a re-composition moves the draft on, an answered challenge
-- whose target is gone is marked SUPERSEDED and stays on the case.
CREATE TABLE IF NOT EXISTS challenges (
    challenge_id TEXT NOT NULL,
    case_id      TEXT NOT NULL,
    status       TEXT NOT NULL,
    payload      JSONB NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (case_id, challenge_id)
);

CREATE INDEX IF NOT EXISTS challenges_case_idx ON challenges (case_id);
