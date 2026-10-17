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
