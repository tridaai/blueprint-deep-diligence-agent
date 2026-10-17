-- Report section notes (SQLite local profile). Same table as the
-- Postgres migration; payloads are TEXT carrying the same JSON.
-- A section note is an analyst's annotation on one named section
-- of a case's report. Notes live apart from the report record —
-- writing or editing one never touches the report — and surface
-- only in the JSON export's annotations field.
CREATE TABLE IF NOT EXISTS report_section_notes (
    note_id     TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    section     TEXT NOT NULL,
    analyst     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
