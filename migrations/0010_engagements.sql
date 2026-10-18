-- Portfolio mode: engagements, the grouping above a case.
-- An engagement is one client mandate under which several cases
-- run (a pipeline of vendor and acquisition diligences). The
-- record carries its name, client label, owner, and the history
-- of the acts that created it and moved cases in and out — the
-- membership itself lives on each case's own record (its
-- engagement_id), so the portfolio view folds from the case
-- records and can never drift from them. An engagement holds no
-- case state and no decision: gates, budgets, evidence, and
-- retention stay per case, exactly as before.
CREATE TABLE IF NOT EXISTS engagements (
    engagement_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    payload       JSONB NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
