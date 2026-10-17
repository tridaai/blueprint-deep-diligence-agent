-- Report section notes: the analyst's annotations on a report.
-- A section note is margin commentary on one named section of a
-- case's report — the executive summary, a question's section, the
-- conflicts, the risks and open items. Notes live in their own
-- table, apart from the report record: writing or editing one
-- never touches the report, so a note can never change what the
-- report says (or its hash). They surface in exactly one place —
-- the JSON export's annotations field.
CREATE TABLE IF NOT EXISTS report_section_notes (
    note_id     TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    section     TEXT NOT NULL,
    analyst     TEXT NOT NULL,
    payload     JSONB NOT NULL,
    created_at  TEXT NOT NULL
);
