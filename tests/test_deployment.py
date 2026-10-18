"""Round 19: deployment profiles — named, validated runtime
compositions.

A deployment profile is chosen with DEPLOYMENT_PROFILE and
validated as a composition at settings load: local forbids live
models, a live default harness profile, and stray production
store settings; hosted requires DATABASE_URL and explicitly
chosen object storage; hosted-self requires its endpoint named
at OPENAI_BASE_URL, with no default. The one rule settings
cannot judge — live models only through a promoted harness
profile — is graded at service startup against the promotion
log. And the record is graded: a case and its runs carry the
deployment they ran under, and records from before this round
read as deployment-unknown, never backfilled.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.config import Settings, load_settings
from deep_diligence.corpus import GOLD_CASES
from deep_diligence.deployment import (
    ConfigError,
    deployment_view,
    unknown_deployment,
)
from deep_diligence.models import BriefIn, CaseOut
from deep_diligence.observability import _fold_runs
from deep_diligence.service import CaseService
from deep_diligence.store import utcnow

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_PROFILES = REPO_ROOT / "profiles"


def _paths(tmp_path) -> dict:
    return dict(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
    )


# -- the settings-load refusals -----------------------------------------------


def test_unknown_deployment_profile_refuses_listing_the_valid_set():
    with pytest.raises(ConfigError) as excinfo:
        load_settings(deployment_profile="bogus")
    message = str(excinfo.value)
    assert "bogus" in message
    for name in ("local", "hosted", "hosted-self"):
        assert name in message


def test_unnamed_settings_keep_their_historical_behaviour():
    """No profile named, no new refusals: the overrides the suite
    and the eval packs compose with — a live-target harness
    profile, a live model backend for a label — load exactly as
    they always have. The profiles bind deployments, and a
    deployment names its profile."""
    assert load_settings(profile_id="openai-gpt").profile_id == "openai-gpt"
    assert load_settings(llm_backend="openai").llm_backend == "openai"
    assert load_settings().deployment_profile == "local"


def test_local_refuses_a_live_model_backend():
    with pytest.raises(ConfigError) as excinfo:
        load_settings(deployment_profile="local", llm_backend="openai")
    message = str(excinfo.value)
    assert "local" in message and "forbidden" in message
    assert "openai" in message


def test_local_refuses_a_live_default_harness_profile():
    with pytest.raises(ConfigError) as excinfo:
        load_settings(deployment_profile="local", profile_id="openai-gpt")
    message = str(excinfo.value)
    assert "openai-gpt" in message and "offline" in message


def test_local_refuses_a_production_database_url():
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        load_settings(
            deployment_profile="local",
            database_url="postgresql://db/diligence",
        )


def test_local_refuses_an_s3_endpoint():
    with pytest.raises(ConfigError, match="S3_ENDPOINT_URL"):
        load_settings(
            deployment_profile="local",
            s3_endpoint_url="http://localhost:8333",
        )


def test_hosted_requires_database_url():
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        load_settings(deployment_profile="hosted")


def test_hosted_requires_explicit_object_storage(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        load_settings(
            deployment_profile="hosted",
            database_url="postgresql://db/diligence",
        )
    assert "OBJECT_DIR" in str(excinfo.value)


def test_hosted_settings_validate_when_the_composition_is_chosen(tmp_path):
    by_dir = load_settings(
        deployment_profile="hosted",
        database_url="postgresql://db/diligence",
        object_dir=str(tmp_path / "objects"),
    )
    assert by_dir.deployment_profile == "hosted"
    by_s3 = load_settings(
        deployment_profile="hosted",
        database_url="postgresql://db/diligence",
        s3_endpoint_url="http://localhost:8333",
    )
    assert by_s3.object_backend == "s3"


def test_hosted_self_requires_the_openai_compatible_boundary(tmp_path):
    with pytest.raises(ConfigError, match="LLM_BACKEND"):
        load_settings(
            deployment_profile="hosted-self",
            database_url="postgresql://db/diligence",
            object_dir=str(tmp_path / "objects"),
        )


def test_hosted_self_requires_a_base_url_with_no_default(tmp_path):
    with pytest.raises(ConfigError, match="OPENAI_BASE_URL"):
        load_settings(
            deployment_profile="hosted-self",
            database_url="postgresql://db/diligence",
            object_dir=str(tmp_path / "objects"),
            llm_backend="openai",
        )
    settings = load_settings(
        deployment_profile="hosted-self",
        database_url="postgresql://db/diligence",
        object_dir=str(tmp_path / "objects"),
        llm_backend="openai",
        openai_base_url="http://localhost:8000/v1",
    )
    assert settings.deployment_profile == "hosted-self"


# -- the record -----------------------------------------------------------------


def test_explicit_local_boots_stamps_and_reads_back(tmp_path):
    settings = load_settings(deployment_profile="local", **_paths(tmp_path))
    service = CaseService(settings)
    case = service.create_case(BriefIn(**GOLD_CASES[0]["brief"]))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)

    stamp = {"profile": "local", "harness_profile": "offline-stub",
             "store": "sqlite"}
    assert service.store.get_case(case_id)["deployment"] == stamp

    trace = service.case_trace(case_id)
    assert trace["deployment"] == stamp
    assert trace["runs"], "the gold run records phase runs"
    assert all(run["deployment"] == stamp for run in trace["runs"])

    client = TestClient(create_app(settings))
    body = client.get(f"/api/cases/{case_id}").json()
    assert body["deployment"] == stamp
    ready = client.get("/readiness").json()
    assert ready["deployment"] == "local"
    assert ready["harness_profile"] == "offline-stub"


def test_records_from_before_deployment_profiles_read_unknown(
    service, gold_brief
):
    # The view over a missing record is the unknown deployment —
    # never the current boot's composition.
    assert deployment_view(None) == unknown_deployment()
    assert deployment_view({}) == unknown_deployment()
    assert deployment_view({"profile": "local"})["profile"] == "local"

    # A run granted before the stamp existed folds as unknown.
    runs = _fold_runs(
        [
            {
                "event": "budget_granted",
                "ts": utcnow(),
                "payload": {"phase": "planning", "caps": {}},
            }
        ]
    )
    assert runs[0]["deployment"] == unknown_deployment()

    # A case whose record carries no stamp (written before R19):
    # the trace header and the API read both say unknown.
    case = service.create_case(BriefIn(**gold_brief))
    service.store.update_case(case["case_id"], deployment=None)
    trace = service.case_trace(case["case_id"])
    assert trace["deployment"] == unknown_deployment()
    record = service.store.get_case(case["case_id"])
    record.pop("deployment")
    assert CaseOut(**record).deployment == unknown_deployment()


# -- live permission at startup: the promotion log decides ---------------------


@pytest.fixture()
def shipped_profiles_dir(tmp_path):
    directory = tmp_path / "profiles"
    directory.mkdir()
    for name in ("offline-stub.json", "openai-gpt.json"):
        shutil.copy(SHIPPED_PROFILES / name, directory / name)
    return directory


def _hosted_settings(tmp_path, profiles_dir, **overrides) -> Settings:
    values = dict(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
        profile_dir=str(profiles_dir),
        profile_id="offline-stub",
        deployment_profile="hosted",
        llm_backend="openai",
    )
    values.update(overrides)
    return Settings(**values)


def test_hosted_live_model_is_refused_without_a_promotion(
    tmp_path, shipped_profiles_dir
):
    settings = _hosted_settings(tmp_path, shipped_profiles_dir)
    with pytest.raises(ConfigError) as excinfo:
        CaseService(settings)
    message = str(excinfo.value)
    assert "hosted" in message and "promoted" in message


def test_hosted_live_model_boots_behind_a_promotion(
    tmp_path, shipped_profiles_dir
):
    # Boot once on the stub and record the gate's act: openai-gpt
    # promoted. The record is written through the store the gate
    # itself writes — this test grades the permission check, not
    # the gate (round 7's suite grades the gate).
    stub_settings = _hosted_settings(
        tmp_path, shipped_profiles_dir, llm_backend="offline"
    )
    stub_service = CaseService(stub_settings)
    candidate = json.loads(
        (shipped_profiles_dir / "openai-gpt.json").read_text()
    )
    stub_service.store.add_promotion(
        {
            "promotion_id": "promo-test",
            "kind": "promotion",
            "candidate": {
                "id": candidate["id"],
                "version": candidate["version"],
                "target_model": candidate["target_model"],
            },
            "candidate_sha256": "test",
            "incumbent": {
                "id": "offline-stub",
                "version": 1,
                "target_model": "offline:stub",
            },
            "evaluation_id": "eval-test",
            "evidence_evaluation_id": "eval-test",
            "analyst": "Tester",
            "note": None,
            "created_at": utcnow(),
        }
    )

    live_settings = _hosted_settings(tmp_path, shipped_profiles_dir)
    service = CaseService(live_settings)
    assert service.profile.id == "openai-gpt"
    case = service.create_case(BriefIn(**GOLD_CASES[0]["brief"]))
    assert case["deployment"] == {
        "profile": "hosted",
        "harness_profile": "openai-gpt",
        "store": "sqlite",
    }
