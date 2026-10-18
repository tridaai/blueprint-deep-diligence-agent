"""PostgreSQL store conformance: the production store, exercised.

The unit suite builds every test its own SQLite store in a tmp
directory — per-test stores are its fixture contract, so it cannot
prove the Postgres adapter. This leg can. It is env-gated on
DD_TEST_DATABASE_URL naming a SCRATCH database: when the variable
is unset the leg skips, and when it is set the leg first wipes the
scratch database's tables (refusing, loudly, if the variable names
the same database as DATABASE_URL), then runs one gold case end to
end through the service on Postgres and reads the result back
through a second store connection — the bytes on disk, not one
connection's state.

It exists because of what it caught the first time it ran (R20):
every payload column is JSONB on Postgres and TEXT on SQLite, and
every store read assumed the SQLite wire shape — json.loads() on a
value psycopg had already parsed. The production store had never
actually run. The reads now go through store._loads, and this leg
is the regression test that keeps both wire shapes honest.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from deep_diligence.config import load_settings
from deep_diligence.corpus import GOLD_CASES
from deep_diligence.models import BriefIn
from deep_diligence.service import CaseService
from deep_diligence.store import Store

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_DSN = os.environ.get("DD_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not SCRATCH_DSN,
    reason="DD_TEST_DATABASE_URL not set — the Postgres conformance "
    "leg runs only against an explicitly named scratch database",
)


def _dbname(conninfo: dict) -> tuple:
    return (
        conninfo.get("host"),
        conninfo.get("port"),
        conninfo.get("dbname"),
    )


def _fresh_scratch_database(dsn: str) -> None:
    """Drop every table in the scratch database so the store's own
    migrations rebuild the schema on construction. Guarded: if
    DATABASE_URL names the same database, refuse instead of wiping
    what might be the deployment's real store."""
    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    target = _dbname(conninfo_to_dict(dsn))
    ambient = os.environ.get("DATABASE_URL")
    if ambient and _dbname(conninfo_to_dict(ambient)) == target:
        raise RuntimeError(
            "DD_TEST_DATABASE_URL names the same database as "
            "DATABASE_URL — refusing to wipe it; point the leg at a "
            "scratch database of its own"
        )
    with psycopg.connect(dsn, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        for (tablename,) in rows:
            conn.execute(f'DROP TABLE IF EXISTS public."{tablename}" CASCADE')


@pytest.fixture()
def pg_service(tmp_path, monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_PROFILE", raising=False)
    _fresh_scratch_database(SCRATCH_DSN)
    settings = load_settings(
        database_url=SCRATCH_DSN,
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
    )
    assert settings.store_backend == "postgres"
    return CaseService(settings)


def test_postgres_store_runs_a_gold_case_end_to_end(pg_service, tmp_path, monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_PROFILE", raising=False)
    service = pg_service
    assert service.store.backend == "postgres"

    gold = GOLD_CASES[0]
    case = service.create_case(BriefIn(**gold["brief"]))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Conformance", approve=True)
    service.run_execution(case_id)
    decided = service.decide_report(case_id, analyst="Conformance", approve=True)
    assert decided["status"] == "FINAL"

    # Read back through a second store on the same database: what
    # is proven is the persisted record, in Postgres's wire shape.
    settings = load_settings(
        database_url=SCRATCH_DSN,
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
    )
    store = Store(settings)
    stored_case = store.get_case(case_id)
    assert stored_case["status"] == "FINAL"
    assert stored_case["deployment"]["store"] == "postgres"

    report = store.get_report(case_id)
    assert report is not None
    assert report["claims"], "a final report carries its claims"

    # The read paths that carry JSONB payloads, exercised by name —
    # list_skills is where the wire-shape defect first surfaced.
    assert len(store.list_skills()) >= 1
    assert store.list_evidence(case_id), "evidence persisted"
    assert store.list_findings(case_id), "findings persisted"
    assert store.list_audit(case_id), "the audit trail persisted"
    verification = store.get_verifier_report(case_id)
    assert verification is not None
