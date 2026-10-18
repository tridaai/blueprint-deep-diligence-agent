"""Case store: the governance shell's system of record.

One port, two adapters. SQLite is the local default (SQLITE_PATH);
PostgreSQL is the production profile (DATABASE_URL). Entity payloads are
JSON beside a few query columns, and numbered SQL migrations in
migrations/ (Postgres) and migrations/sqlite/ are applied by the runner
at startup. The SQL below is written with ? placeholders and translated
for psycopg.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = settings.store_backend
        self._lock = threading.RLock()
        if self.backend == "sqlite":
            path = Path(settings.sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        else:
            import psycopg
            from psycopg.rows import dict_row

            self._conn = psycopg.connect(settings.database_url, row_factory=dict_row)

    # -- plumbing ---------------------------------------------------------
    def _sql(self, sql: str) -> str:
        return sql if self.backend == "sqlite" else sql.replace("?", "%s")

    def _execute(self, sql: str, params: tuple = ()) -> Any:
        with self._lock:
            cur = self._conn.execute(self._sql(sql), params)
            self._conn.commit()
            return cur

    def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            cur = self._conn.execute(self._sql(sql), params)
            row = cur.fetchone()
            return dict(row) if row is not None else None

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(self._sql(sql), params)
            return [dict(r) for r in cur.fetchall()]

    def ping(self) -> bool:
        try:
            self._fetchone("SELECT 1 AS ok")
            return True
        except Exception:
            return False

    def _migrations_folder(self) -> Path:
        # MIGRATIONS_DIR wins; otherwise fall back to the migrations
        # directory beside the package (repo root / migrations), so the
        # app works from any working directory.
        bases = [
            Path(self.settings.migrations_dir),
            Path(__file__).resolve().parents[2] / "migrations",
        ]
        for base in bases:
            folder = base if self.backend == "postgres" else base / "sqlite"
            if folder.is_dir():
                return folder
        return bases[0] if self.backend == "postgres" else bases[0] / "sqlite"

    def migrate(self) -> None:
        folder = self._migrations_folder()
        self._execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        applied = {
            r["version"] for r in self._fetchall("SELECT version FROM schema_migrations")
        }
        for path in sorted(folder.glob("*.sql")):
            if path.name in applied:
                continue
            sql = path.read_text()
            with self._lock:
                if self.backend == "sqlite":
                    self._conn.executescript(sql)
                else:
                    self._conn.execute(sql)
                self._conn.execute(
                    self._sql(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)"
                    ),
                    (path.name, utcnow()),
                )
                self._conn.commit()

    # -- cases ------------------------------------------------------------
    def create_case(self, case: dict) -> dict:
        self._execute(
            "INSERT INTO cases (case_id, status, subject, payload, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                case["case_id"],
                case["status"],
                case["subject"],
                json.dumps(case),
                case["created_at"],
                case["updated_at"],
            ),
        )
        return case

    def get_case(self, case_id: str) -> dict | None:
        row = self._fetchone("SELECT payload FROM cases WHERE case_id = ?", (case_id,))
        return json.loads(row["payload"]) if row else None

    def list_cases(self) -> list[dict]:
        rows = self._fetchall("SELECT payload FROM cases ORDER BY created_at")
        return [json.loads(r["payload"]) for r in rows]

    def update_case(self, case_id: str, **fields: Any) -> dict:
        case = self.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        case.update(fields)
        case["updated_at"] = utcnow()
        self._execute(
            "UPDATE cases SET status = ?, payload = ?, updated_at = ? WHERE case_id = ?",
            (case["status"], json.dumps(case), case["updated_at"], case_id),
        )
        return case

    # -- plans --------------------------------------------------------------
    def save_plan(self, case_id: str, plan: dict) -> dict:
        row = self._fetchone(
            "SELECT COALESCE(MAX(version), 0) AS v FROM plans WHERE case_id = ?",
            (case_id,),
        )
        plan = {**plan, "version": int(row["v"]) + 1}
        self._execute(
            "INSERT INTO plans (case_id, version, status, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (case_id, plan["version"], plan["status"], json.dumps(plan), utcnow()),
        )
        return plan

    def get_plan(self, case_id: str, version: int | None = None) -> dict | None:
        if version is None:
            row = self._fetchone(
                "SELECT payload FROM plans WHERE case_id = ? ORDER BY version DESC LIMIT 1",
                (case_id,),
            )
        else:
            row = self._fetchone(
                "SELECT payload FROM plans WHERE case_id = ? AND version = ?",
                (case_id, version),
            )
        return json.loads(row["payload"]) if row else None

    def set_plan_status(self, case_id: str, version: int, status: str) -> None:
        self._execute(
            "UPDATE plans SET status = ? WHERE case_id = ? AND version = ?",
            (status, case_id, version),
        )

    def update_plan(self, case_id: str, plan: dict) -> dict:
        self._execute(
            "UPDATE plans SET status = ?, payload = ? WHERE case_id = ? AND version = ?",
            (plan["status"], json.dumps(plan), case_id, plan["version"]),
        )
        return plan

    # -- evidence -----------------------------------------------------------
    def add_evidence(self, ev: dict) -> dict:
        self._execute(
            "INSERT INTO evidence (evidence_id, case_id, doc_slug, sha256, payload)"
            " VALUES (?, ?, ?, ?, ?)",
            (ev["evidence_id"], ev["case_id"], ev["doc_slug"], ev["sha256"], json.dumps(ev)),
        )
        return ev

    def list_evidence(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM evidence WHERE case_id = ? ORDER BY evidence_id",
            (case_id,),
        )
        return [json.loads(r["payload"]) for r in rows]

    def get_evidence(self, case_id: str, evidence_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM evidence WHERE case_id = ? AND evidence_id = ?",
            (case_id, evidence_id),
        )
        return json.loads(row["payload"]) if row else None

    def count_evidence(self, case_id: str) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM evidence WHERE case_id = ?", (case_id,)
        )
        return int(row["n"])

    # -- findings -----------------------------------------------------------
    def add_finding(self, finding: dict) -> dict:
        self._execute(
            "INSERT INTO findings (finding_id, case_id, question_id, payload)"
            " VALUES (?, ?, ?, ?)",
            (
                finding["finding_id"],
                finding["case_id"],
                finding["question_id"],
                json.dumps(finding),
            ),
        )
        return finding

    def add_finding_auto_id(self, case_id: str, finding: dict) -> dict:
        # Sub-agents record findings in parallel; the count and the
        # insert must be one locked step or two workstreams mint the
        # same finding ID and collide on the primary key.
        with self._lock:
            row = self._fetchone(
                "SELECT COUNT(*) AS n FROM findings WHERE case_id = ?", (case_id,)
            )
            finding = {
                **finding,
                "finding_id": f"FND-{int(row['n']) + 1:04d}",
                "case_id": case_id,
            }
            cur = self._conn.execute(
                self._sql(
                    "INSERT INTO findings (finding_id, case_id, question_id, payload)"
                    " VALUES (?, ?, ?, ?)"
                ),
                (
                    finding["finding_id"],
                    case_id,
                    finding["question_id"],
                    json.dumps(finding),
                ),
            )
            self._conn.commit()
            return finding

    def list_findings(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM findings WHERE case_id = ? ORDER BY finding_id",
            (case_id,),
        )
        return [json.loads(r["payload"]) for r in rows]

    def count_findings(self, case_id: str) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM findings WHERE case_id = ?", (case_id,)
        )
        return int(row["n"])

    # -- reports --------------------------------------------------------------
    def save_report(self, case_id: str, report: dict) -> dict:
        existing = self.get_report(case_id)
        if existing is None:
            self._execute(
                "INSERT INTO reports (case_id, status, payload) VALUES (?, ?, ?)",
                (case_id, report["status"], json.dumps(report)),
            )
        else:
            self._execute(
                "UPDATE reports SET status = ?, payload = ? WHERE case_id = ?",
                (report["status"], json.dumps(report), case_id),
            )
        return report

    def get_report(self, case_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM reports WHERE case_id = ?", (case_id,)
        )
        return json.loads(row["payload"]) if row else None

    # -- citation ledger ------------------------------------------------------
    def replace_ledger(self, case_id: str, entries: list[dict]) -> list[dict]:
        # A re-composition mints a new report, so its ledger replaces
        # the old one wholesale; delete + inserts are one locked step
        # so a reader never sees a half-written ledger.
        with self._lock:
            self._conn.execute(
                self._sql("DELETE FROM citation_ledger WHERE case_id = ?"),
                (case_id,),
            )
            for entry in entries:
                self._conn.execute(
                    self._sql(
                        "INSERT INTO citation_ledger"
                        " (case_id, claim_id, finding_id, question_id, payload, created_at)"
                        " VALUES (?, ?, ?, ?, ?, ?)"
                    ),
                    (
                        case_id,
                        entry["claim_id"],
                        entry["finding_id"],
                        entry["question_id"],
                        json.dumps(entry),
                        entry["cited_at"],
                    ),
                )
            self._conn.commit()
        return entries

    def list_ledger(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM citation_ledger WHERE case_id = ? ORDER BY claim_id",
            (case_id,),
        )
        return [json.loads(r["payload"]) for r in rows]

    def delete_finding(self, case_id: str, finding_id: str) -> None:
        self._execute(
            "DELETE FROM findings WHERE case_id = ? AND finding_id = ?",
            (case_id, finding_id),
        )

    # -- verifier -------------------------------------------------------------
    def save_verifier_report(self, case_id: str, report: dict) -> dict:
        existing = self.get_verifier_report(case_id)
        if existing is None:
            self._execute(
                "INSERT INTO verifier_reports (case_id, payload, created_at)"
                " VALUES (?, ?, ?)",
                (case_id, json.dumps(report), report["verified_at"]),
            )
        else:
            self._execute(
                "UPDATE verifier_reports SET payload = ?, created_at = ?"
                " WHERE case_id = ?",
                (json.dumps(report), report["verified_at"], case_id),
            )
        return report

    def get_verifier_report(self, case_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM verifier_reports WHERE case_id = ?", (case_id,)
        )
        return json.loads(row["payload"]) if row else None

    def replace_quarantine(self, case_id: str, records: list[dict]) -> list[dict]:
        # A fresh verification supersedes the previous quarantine set
        # wholesale (analyst decisions on the old set live on in the
        # audit trail); delete + inserts are one locked step so a
        # reader never sees a half-written quarantine.
        with self._lock:
            self._conn.execute(
                self._sql("DELETE FROM quarantine WHERE case_id = ?"),
                (case_id,),
            )
            for record in records:
                self._conn.execute(
                    self._sql(
                        "INSERT INTO quarantine"
                        " (quarantine_id, case_id, claim_id, status, payload, created_at)"
                        " VALUES (?, ?, ?, ?, ?, ?)"
                    ),
                    (
                        record["quarantine_id"],
                        case_id,
                        record["claim_id"],
                        record["status"],
                        json.dumps(record),
                        record["created_at"],
                    ),
                )
            self._conn.commit()
        return records

    def list_quarantine(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM quarantine WHERE case_id = ?"
            " ORDER BY quarantine_id",
            (case_id,),
        )
        return [json.loads(r["payload"]) for r in rows]

    def get_quarantine(self, case_id: str, quarantine_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM quarantine"
            " WHERE case_id = ? AND quarantine_id = ?",
            (case_id, quarantine_id),
        )
        return json.loads(row["payload"]) if row else None

    def update_quarantine(self, case_id: str, record: dict) -> dict:
        self._execute(
            "UPDATE quarantine SET status = ?, payload = ?"
            " WHERE case_id = ? AND quarantine_id = ?",
            (
                record["status"],
                json.dumps(record),
                case_id,
                record["quarantine_id"],
            ),
        )
        return record

    # -- conflicts ------------------------------------------------------------
    def replace_conflicts(self, case_id: str, records: list[dict]) -> list[dict]:
        # A fresh detection pass supersedes the previous conflict set
        # wholesale (analyst NOTED decisions on the old set live on in
        # the audit trail); delete + inserts are one locked step so a
        # reader never sees a half-written set. Conflicts are never
        # deleted within a report's life by any analyst action — this
        # replacement happens only when a new draft is verified.
        with self._lock:
            self._conn.execute(
                self._sql("DELETE FROM conflicts WHERE case_id = ?"),
                (case_id,),
            )
            for record in records:
                self._conn.execute(
                    self._sql(
                        "INSERT INTO conflicts"
                        " (conflict_id, case_id, status, payload, created_at)"
                        " VALUES (?, ?, ?, ?, ?)"
                    ),
                    (
                        record["conflict_id"],
                        case_id,
                        record["status"],
                        json.dumps(record),
                        record["created_at"],
                    ),
                )
            self._conn.commit()
        return records

    def list_conflicts(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM conflicts WHERE case_id = ?"
            " ORDER BY conflict_id",
            (case_id,),
        )
        return [json.loads(r["payload"]) for r in rows]

    def get_conflict(self, case_id: str, conflict_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM conflicts"
            " WHERE case_id = ? AND conflict_id = ?",
            (case_id, conflict_id),
        )
        return json.loads(row["payload"]) if row else None

    def update_conflict(self, case_id: str, record: dict) -> dict:
        self._execute(
            "UPDATE conflicts SET status = ?, payload = ?"
            " WHERE case_id = ? AND conflict_id = ?",
            (
                record["status"],
                json.dumps(record),
                case_id,
                record["conflict_id"],
            ),
        )
        return record

    # -- virtual filesystem index ---------------------------------------------
    def put_file_meta(self, case_id: str, path: str, sha256: str, size: int) -> None:
        existing = self.get_file_meta(case_id, path)
        if existing is None:
            self._execute(
                "INSERT INTO case_files (case_id, path, sha256, size, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (case_id, path, sha256, size, utcnow()),
            )
        else:
            self._execute(
                "UPDATE case_files SET sha256 = ?, size = ?, updated_at = ?"
                " WHERE case_id = ? AND path = ?",
                (sha256, size, utcnow(), case_id, path),
            )

    def get_file_meta(self, case_id: str, path: str) -> dict | None:
        return self._fetchone(
            "SELECT * FROM case_files WHERE case_id = ? AND path = ?", (case_id, path)
        )

    def list_files(self, case_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM case_files WHERE case_id = ? ORDER BY path", (case_id,)
        )

    def delete_file_meta(self, case_id: str, path: str) -> None:
        self._execute(
            "DELETE FROM case_files WHERE case_id = ? AND path = ?", (case_id, path)
        )

    # -- profile regression gate ------------------------------------------------
    # Append-only by construction: these two tables have insert and
    # read methods and no update or delete anywhere in the store. An
    # evaluation or a promotion that could be rewritten after the
    # fact would prove nothing, so the port does not offer the verbs.
    def _next_seq(self, table: str) -> int:
        # Insertion order is the log's order: timestamps resolve to
        # the second, so each append takes the next sequence number
        # under the store lock instead of trusting the clock.
        row = self._fetchone(f"SELECT COALESCE(MAX(seq), 0) AS s FROM {table}")
        return int(row["s"]) + 1

    def add_evaluation(self, record: dict) -> dict:
        with self._lock:
            seq = self._next_seq("profile_evaluations")
            self._execute(
                "INSERT INTO profile_evaluations"
                " (seq, evaluation_id, candidate_id, candidate_sha, verdict, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    seq,
                    record["evaluation_id"],
                    record["candidate"]["id"],
                    record["candidate_sha256"],
                    record["verdict"],
                    json.dumps(record),
                    record["created_at"],
                ),
            )
        return record

    def get_evaluation(self, evaluation_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM profile_evaluations WHERE evaluation_id = ?",
            (evaluation_id,),
        )
        return json.loads(row["payload"]) if row else None

    def list_evaluations(self) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM profile_evaluations ORDER BY seq"
        )
        return [json.loads(r["payload"]) for r in rows]

    def add_promotion(self, record: dict) -> dict:
        with self._lock:
            seq = self._next_seq("profile_promotions")
            self._execute(
                "INSERT INTO profile_promotions"
                " (seq, promotion_id, kind, candidate_id, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    seq,
                    record["promotion_id"],
                    record["kind"],
                    record["candidate"]["id"],
                    json.dumps(record),
                    record["created_at"],
                ),
            )
        return record

    def list_promotions(self) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM profile_promotions ORDER BY seq"
        )
        return [json.loads(r["payload"]) for r in rows]

    def latest_promotion(self) -> dict | None:
        promotions = self.list_promotions()
        return promotions[-1] if promotions else None

    # -- execution checkpoints --------------------------------------------------
    def save_checkpoint(self, case_id: str, checkpoint: dict) -> dict:
        # One row per case, replaced as execution progresses: the
        # checkpoint is the execution's own record of where it got
        # to, written by the shell as waves complete — never derived
        # from harness state, so it survives the process.
        existing = self.get_checkpoint(case_id)
        if existing is None:
            self._execute(
                "INSERT INTO execution_checkpoints (case_id, payload, updated_at)"
                " VALUES (?, ?, ?)",
                (case_id, json.dumps(checkpoint), checkpoint["updated_at"]),
            )
        else:
            self._execute(
                "UPDATE execution_checkpoints SET payload = ?, updated_at = ?"
                " WHERE case_id = ?",
                (json.dumps(checkpoint), checkpoint["updated_at"], case_id),
            )
        return checkpoint

    def get_checkpoint(self, case_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM execution_checkpoints WHERE case_id = ?",
            (case_id,),
        )
        return json.loads(row["payload"]) if row else None

    # -- case notes -------------------------------------------------------------
    # The desk's memory of a case. Notes live in their own table —
    # not in the evidence table and not in the virtual filesystem —
    # so nothing in the citation machinery can mistake one for a
    # source: a note ID is never an evidence ID.
    def add_note(self, case_id: str, note: dict) -> dict:
        with self._lock:
            row = self._fetchone(
                "SELECT COUNT(*) AS n FROM case_notes WHERE case_id = ?",
                (case_id,),
            )
            note = {
                **note,
                "note_id": f"NOTE-{int(row['n']) + 1:04d}",
                "case_id": case_id,
            }
            self._conn.execute(
                self._sql(
                    "INSERT INTO case_notes"
                    " (note_id, case_id, analyst, payload, created_at)"
                    " VALUES (?, ?, ?, ?, ?)"
                ),
                (
                    note["note_id"],
                    case_id,
                    note["analyst"],
                    json.dumps(note),
                    note["created_at"],
                ),
            )
            self._conn.commit()
            return note

    def list_notes(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM case_notes WHERE case_id = ?"
            " ORDER BY note_id",
            (case_id,),
        )
        return [json.loads(r["payload"]) for r in rows]

    # -- report section notes -------------------------------------------------
    # An analyst's annotations on sections of a case's report.
    # Their own table, apart from the report record: adding or
    # editing a note never writes to the reports table, so a note
    # cannot change the report's body or its hash. The JSON export
    # is the only surface that carries them, in its annotations
    # field.
    def add_section_note(self, case_id: str, note: dict) -> dict:
        with self._lock:
            row = self._fetchone(
                "SELECT COUNT(*) AS n FROM report_section_notes"
                " WHERE case_id = ?",
                (case_id,),
            )
            note = {
                **note,
                "note_id": f"SNOTE-{int(row['n']) + 1:04d}",
                "case_id": case_id,
            }
            self._conn.execute(
                self._sql(
                    "INSERT INTO report_section_notes"
                    " (note_id, case_id, section, analyst, payload, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (
                    note["note_id"],
                    case_id,
                    note["section"],
                    note["analyst"],
                    json.dumps(note),
                    note["created_at"],
                ),
            )
            self._conn.commit()
            return note

    def list_section_notes(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM report_section_notes WHERE case_id = ?"
            " ORDER BY note_id",
            (case_id,),
        )
        return [json.loads(r["payload"]) for r in rows]

    def get_section_note(self, case_id: str, note_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM report_section_notes"
            " WHERE case_id = ? AND note_id = ?",
            (case_id, note_id),
        )
        return json.loads(row["payload"]) if row else None

    def update_section_note(self, case_id: str, note: dict) -> dict:
        self._execute(
            "UPDATE report_section_notes SET analyst = ?, payload = ?"
            " WHERE case_id = ? AND note_id = ?",
            (
                note["analyst"],
                json.dumps(note),
                case_id,
                note["note_id"],
            ),
        )
        return note

    # -- report snapshots ---------------------------------------------------------
    def add_report_snapshot(self, case_id: str, snapshot: dict) -> dict:
        with self._lock:
            row = self._fetchone(
                "SELECT COUNT(*) AS n FROM report_snapshots WHERE case_id = ?",
                (case_id,),
            )
            snapshot = {
                **snapshot,
                "snapshot_id": f"RSNAP-{int(row['n']) + 1:04d}",
                "case_id": case_id,
            }
            self._conn.execute(
                self._sql(
                    "INSERT INTO report_snapshots"
                    " (snapshot_id, case_id, payload, created_at)"
                    " VALUES (?, ?, ?, ?)"
                ),
                (
                    snapshot["snapshot_id"],
                    case_id,
                    json.dumps(snapshot),
                    snapshot["created_at"],
                ),
            )
            self._conn.commit()
            return snapshot

    def list_report_snapshots(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM report_snapshots WHERE case_id = ?"
            " ORDER BY snapshot_id",
            (case_id,),
        )
        return [json.loads(r["payload"]) for r in rows]

    def update_evidence(self, case_id: str, ev: dict) -> dict:
        self._execute(
            "UPDATE evidence SET sha256 = ?, payload = ?"
            " WHERE case_id = ? AND evidence_id = ?",
            (ev["sha256"], json.dumps(ev), case_id, ev["evidence_id"]),
        )
        return ev

    # -- source registry ----------------------------------------------------
    # The governed registry of sources (governance.py seeds it from
    # the corpus and owns its semantics). One row per source name;
    # the record carries its own act history, so registration is a
    # recorded act in the record itself, not a config edit.
    def add_source(self, record: dict) -> dict:
        self._execute(
            "INSERT INTO sources (name, payload, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (
                record["name"],
                json.dumps(record),
                record["registered_at"],
                record["registered_at"],
            ),
        )
        return record

    def get_source(self, name: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM sources WHERE name = ?", (name,)
        )
        return json.loads(row["payload"]) if row else None

    def list_sources(self) -> list[dict]:
        rows = self._fetchall("SELECT payload FROM sources ORDER BY name")
        return [json.loads(r["payload"]) for r in rows]

    def update_source(self, record: dict) -> dict:
        self._execute(
            "UPDATE sources SET payload = ?, updated_at = ? WHERE name = ?",
            (json.dumps(record), utcnow(), record["name"]),
        )
        return record

    # -- redactions -----------------------------------------------------------
    # A redaction record keeps the original content hash, the
    # reason, and the analyst — never the content. Flags are the
    # live report's claims that cite redacted evidence; they carry
    # the analyst's acknowledgement or send-back decision.
    def add_redaction(self, case_id: str, record: dict) -> dict:
        with self._lock:
            row = self._fetchone(
                "SELECT COUNT(*) AS n FROM redactions WHERE case_id = ?",
                (case_id,),
            )
            record = {
                **record,
                "redaction_id": f"RED-{int(row['n']) + 1:04d}",
                "case_id": case_id,
            }
            self._conn.execute(
                self._sql(
                    "INSERT INTO redactions"
                    " (redaction_id, case_id, evidence_id, payload, created_at)"
                    " VALUES (?, ?, ?, ?, ?)"
                ),
                (
                    record["redaction_id"],
                    case_id,
                    record["evidence_id"],
                    json.dumps(record),
                    record["redacted_at"],
                ),
            )
            self._conn.commit()
            return record

    def list_redactions(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM redactions WHERE case_id = ?"
            " ORDER BY redaction_id",
            (case_id,),
        )
        return [json.loads(r["payload"]) for r in rows]

    def add_redaction_flag(self, case_id: str, flag: dict) -> dict:
        with self._lock:
            row = self._fetchone(
                "SELECT COUNT(*) AS n FROM redaction_flags WHERE case_id = ?",
                (case_id,),
            )
            flag = {
                **flag,
                "flag_id": f"FLAG-{int(row['n']) + 1:04d}",
                "case_id": case_id,
            }
            self._conn.execute(
                self._sql(
                    "INSERT INTO redaction_flags"
                    " (flag_id, case_id, claim_id, status, payload, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (
                    flag["flag_id"],
                    case_id,
                    flag["claim_id"],
                    flag["status"],
                    json.dumps(flag),
                    flag["created_at"],
                ),
            )
            self._conn.commit()
            return flag

    def list_redaction_flags(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM redaction_flags WHERE case_id = ?"
            " ORDER BY flag_id",
            (case_id,),
        )
        return [json.loads(r["payload"]) for r in rows]

    def get_redaction_flag(self, case_id: str, flag_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM redaction_flags"
            " WHERE case_id = ? AND flag_id = ?",
            (case_id, flag_id),
        )
        return json.loads(row["payload"]) if row else None

    def update_redaction_flag(self, case_id: str, flag: dict) -> dict:
        self._execute(
            "UPDATE redaction_flags SET status = ?, payload = ?"
            " WHERE case_id = ? AND flag_id = ?",
            (flag["status"], json.dumps(flag), case_id, flag["flag_id"]),
        )
        return flag

    # -- case tombstones -------------------------------------------------------
    # All that survives a case deletion: the minimal record that
    # the case existed and was ended, by whom, why, and when, plus
    # the hash of the final report it carried, if it carried one.
    def add_tombstone(self, tombstone: dict) -> dict:
        self._execute(
            "INSERT INTO case_tombstones (case_id, subject, payload, deleted_at)"
            " VALUES (?, ?, ?, ?)",
            (
                tombstone["case_id"],
                tombstone["subject"],
                json.dumps(tombstone),
                tombstone["deleted_at"],
            ),
        )
        return tombstone

    def get_tombstone(self, case_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM case_tombstones WHERE case_id = ?",
            (case_id,),
        )
        return json.loads(row["payload"]) if row else None

    def list_tombstones(self) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM case_tombstones ORDER BY deleted_at"
        )
        return [json.loads(r["payload"]) for r in rows]

    def delete_case_data(self, case_id: str) -> None:
        # Every case-scoped table, in one locked step: a deletion
        # that fails halfway must not leave a half-case behind. The
        # tombstone is written by the caller, outside this sweep —
        # it is the one record that survives.
        tables = (
            "cases", "plans", "evidence", "findings", "reports",
            "citation_ledger", "verifier_reports", "quarantine",
            "conflicts", "case_files", "execution_checkpoints",
            "case_notes", "report_snapshots", "redactions",
            "redaction_flags", "audit",
        )
        with self._lock:
            for table in tables:
                self._conn.execute(
                    self._sql(f"DELETE FROM {table} WHERE case_id = ?"),
                    (case_id,),
                )
            self._conn.commit()

    # -- workbench views ----------------------------------------------------
    # An analyst's saved queue filters. Their own table because a
    # view belongs to the desk's workflow, not to a case: it is
    # keyed by analyst + name, carries no decision, and deleting
    # one removes nothing but the shortcut itself.
    def add_view(self, view: dict) -> dict:
        with self._lock:
            # Minted from the highest existing suffix, not a row
            # count: views can be deleted, and a reused ID would
            # collide on the primary key.
            rows = self._fetchall("SELECT view_id FROM workbench_views")
            highest = max(
                (int(r["view_id"].split("-")[1]) for r in rows),
                default=0,
            )
            view = {**view, "view_id": f"VIEW-{highest + 1:04d}"}
            self._conn.execute(
                self._sql(
                    "INSERT INTO workbench_views"
                    " (view_id, analyst, name, payload, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (
                    view["view_id"],
                    view["analyst"],
                    view["name"],
                    json.dumps(view),
                    view["created_at"],
                    view["updated_at"],
                ),
            )
            self._conn.commit()
            return view

    def list_views(self, analyst: str | None = None) -> list[dict]:
        if analyst is None:
            rows = self._fetchall(
                "SELECT payload FROM workbench_views ORDER BY view_id"
            )
        else:
            rows = self._fetchall(
                "SELECT payload FROM workbench_views"
                " WHERE analyst = ? ORDER BY view_id",
                (analyst,),
            )
        return [json.loads(r["payload"]) for r in rows]

    def get_view(self, view_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM workbench_views WHERE view_id = ?",
            (view_id,),
        )
        return json.loads(row["payload"]) if row else None

    def find_view(self, analyst: str, name: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM workbench_views"
            " WHERE analyst = ? AND name = ?",
            (analyst, name),
        )
        return json.loads(row["payload"]) if row else None

    def update_view(self, view: dict) -> dict:
        self._execute(
            "UPDATE workbench_views SET payload = ?, updated_at = ?"
            " WHERE view_id = ?",
            (json.dumps(view), view["updated_at"], view["view_id"]),
        )
        return view

    def delete_view(self, view_id: str) -> None:
        self._execute(
            "DELETE FROM workbench_views WHERE view_id = ?", (view_id,)
        )

    # -- engagements ----------------------------------------------------
    # Portfolio mode (R15): the grouping above a case. The record
    # carries the engagement's name, client label, owner, and the
    # history of the acts on it (creation, case assignments in and
    # out). Membership itself lives on the case records (their
    # engagement_id), so this table never holds a member list that
    # could drift from the cases it groups.
    def add_engagement(self, record: dict) -> dict:
        with self._lock:
            # Minted from the highest existing suffix, not a row
            # count, for the same reason as view IDs: a reused ID
            # would collide on the primary key.
            rows = self._fetchall("SELECT engagement_id FROM engagements")
            highest = max(
                (int(r["engagement_id"].split("-")[1]) for r in rows),
                default=0,
            )
            record = {**record, "engagement_id": f"ENG-{highest + 1:04d}"}
            self._conn.execute(
                self._sql(
                    "INSERT INTO engagements"
                    " (engagement_id, name, payload, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?)"
                ),
                (
                    record["engagement_id"],
                    record["name"],
                    json.dumps(record),
                    record["created_at"],
                    record["updated_at"],
                ),
            )
            self._conn.commit()
            return record

    def get_engagement(self, engagement_id: str) -> dict | None:
        row = self._fetchone(
            "SELECT payload FROM engagements WHERE engagement_id = ?",
            (engagement_id,),
        )
        return json.loads(row["payload"]) if row else None

    def list_engagements(self) -> list[dict]:
        rows = self._fetchall(
            "SELECT payload FROM engagements ORDER BY engagement_id"
        )
        return [json.loads(r["payload"]) for r in rows]

    def update_engagement(self, record: dict) -> dict:
        self._execute(
            "UPDATE engagements SET payload = ?, updated_at = ?"
            " WHERE engagement_id = ?",
            (
                json.dumps(record),
                record["updated_at"],
                record["engagement_id"],
            ),
        )
        return record

    # -- audit ------------------------------------------------------------------
    def add_audit(self, case_id: str, event: str, payload: dict | None = None) -> None:
        self._execute(
            "INSERT INTO audit (case_id, ts, event, payload) VALUES (?, ?, ?, ?)",
            (case_id, utcnow(), event, json.dumps(payload or {})),
        )

    def list_audit(self, case_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT ts, event, payload FROM audit WHERE case_id = ? ORDER BY id",
            (case_id,),
        )
        return [
            {"ts": r["ts"], "event": r["event"], "payload": json.loads(r["payload"])}
            for r in rows
        ]


def build_store(settings: Settings) -> Store:
    store = Store(settings)
    store.migrate()
    return store
