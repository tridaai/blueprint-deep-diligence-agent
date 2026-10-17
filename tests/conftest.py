from __future__ import annotations

from pathlib import Path

import pytest

from deep_diligence.config import load_settings
from deep_diligence.corpus import GOLD_CASES
from deep_diligence.models import BriefIn
from deep_diligence.service import CaseService

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def settings(tmp_path):
    return load_settings(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
    )


@pytest.fixture()
def service(settings):
    return CaseService(settings)


@pytest.fixture()
def gold_brief():
    return GOLD_CASES[0]["brief"]


@pytest.fixture()
def case(service, gold_brief):
    return service.create_case(BriefIn(**gold_brief))


@pytest.fixture()
def approved_case(service, case):
    service.run_planning(case["case_id"])
    service.decide_plan(case["case_id"], analyst="Tester", approve=True)
    return service.store.get_case(case["case_id"])
