-- Analyst workbench views: a saved queue filter, per analyst.
-- A view is the analyst's own working state — a name and the
-- filter set it applies to the workbench queue (act needed,
-- assignee, priority). It records no decision and gates nothing;
-- every gate decision stays one case, one analyst, made on the
-- case itself. Views live apart from the case tables because they
-- belong to the desk's workflow, not to any case's record.
CREATE TABLE IF NOT EXISTS workbench_views (
    view_id     TEXT PRIMARY KEY,
    analyst     TEXT NOT NULL,
    name        TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
