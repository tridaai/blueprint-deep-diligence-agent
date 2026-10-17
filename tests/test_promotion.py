"""The profile regression gate: the split, the rule, the records,
and the controls.

The mechanics tests inject a fast suite runner — the gate's logic
does not care how a suite was produced. The two controls at the
bottom run the REAL suite: the shipped OpenAI profile must fail
the gate, and a copy of the incumbent must pass it and complete a
promotion + rollback cycle. They take minutes; that is the honest
cost of the evidence, and it is why they are two tests, not ten.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.config import load_settings
from deep_diligence.corpus import ALL_CASES, GOLD_CASES, PACK_SPLITS
from deep_diligence.models import BriefIn
from deep_diligence.promotion import (
    HELD_OUT_TOLERANCE,
    PROMOTION_RUNS,
    PromotionError,
    aggregate_run,
    judge_promotion,
    profile_file_sha256,
)
from deep_diligence.service import CaseService

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_PROFILES = REPO_ROOT / "profiles"


# -- fabricated suites --------------------------------------------------------


def _suite(profile_id, found=10, per_key=None, unsupported=0, citation=1.0,
           correctness=1.0):
    cases = []
    for key, split in PACK_SPLITS.items():
        f = (per_key or {}).get(key, found)
        cases.append(
            {
                "key": key,
                "split": split,
                "recall": f / 10,
                "found": f,
                "gold": 10,
                "citation": citation,
                "correctness": correctness,
                "unsupported": unsupported,
            }
        )
    return {"profile": {"id": profile_id}, "cases": cases}


def _runner(**profiles):
    def run(profile_id):
        return _suite(profile_id, **profiles[profile_id])

    return run


def _run(tuning_recall, held_out_recall, zero_tolerance_ok=True):
    subset = {
        "recall": 0.0,
        "unsupported": 0,
        "citation_failures": [],
        "correctness_failures": [],
        "zero_tolerance_ok": zero_tolerance_ok,
    }
    return {
        "cases": [],
        "subsets": {
            "tuning": {**subset, "recall": tuning_recall},
            "held_out": {**subset, "recall": held_out_recall},
        },
    }


@pytest.fixture()
def profiles_dir(tmp_path):
    directory = tmp_path / "profiles"
    directory.mkdir()
    base = json.loads((SHIPPED_PROFILES / "offline-stub.json").read_text())
    for pid in ("alpha", "beta"):
        doc = dict(base)
        doc["id"] = pid
        doc["version"] = 1
        (directory / f"{pid}.json").write_text(json.dumps(doc))
    return directory


@pytest.fixture()
def gate_settings(tmp_path, profiles_dir):
    return load_settings(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
        profile_id="alpha",
        profile_dir=str(profiles_dir),
    )


@pytest.fixture()
def gate_service(gate_settings):
    service = CaseService(gate_settings)
    service.gate.suite_runner = _runner(alpha={}, beta={})
    return service


# -- the split ------------------------------------------------------------------


def test_split_is_fixed_in_the_eval_data():
    assert set(PACK_SPLITS) == {c["key"] for c in ALL_CASES}
    assert {c["split"] for c in ALL_CASES} == {"tuning", "held_out"}
    for case in ALL_CASES:
        assert case["split"] == PACK_SPLITS[case["key"]]
    # The integrity pack — the stale-figure trap and the poisoned
    # findings — is held out: a promotion cannot be rehearsed on it.
    assert PACK_SPLITS["integrity-aster-citation"] == "held_out"
    assert PACK_SPLITS["gold-helvion-vendor"] == "tuning"


# -- the rule ---------------------------------------------------------------------


def test_judge_reads_the_worst_run_not_the_best():
    baseline = [_run(1.0, 1.0) for _ in range(PROMOTION_RUNS)]
    candidate = [_run(1.0, 1.0), _run(1.0, 1.0), _run(0.5, 1.0)]
    judged = judge_promotion(candidate, baseline)
    assert judged["verdict"] == "NOT_PROMOTABLE"
    assert judged["failures"][0]["run"] == 3
    assert judged["failures"][0]["rule"] == "tuning_recall"


def test_judge_held_out_tolerance_is_a_floor_not_a_vibe():
    baseline = [_run(1.0, 0.90) for _ in range(PROMOTION_RUNS)]
    bars_floor = 0.90 - HELD_OUT_TOLERANCE
    just_above = [_run(1.0, bars_floor + 0.005) for _ in range(PROMOTION_RUNS)]
    assert judge_promotion(just_above, baseline)["verdict"] == "PROMOTABLE"
    just_below = [_run(1.0, bars_floor - 0.005) for _ in range(PROMOTION_RUNS)]
    judged = judge_promotion(just_below, baseline)
    assert judged["verdict"] == "NOT_PROMOTABLE"
    assert {f["rule"] for f in judged["failures"]} == {"held_out_recall"}


def test_judge_tuning_allows_no_fall_at_all():
    baseline = [_run(0.95, 1.0) for _ in range(PROMOTION_RUNS)]
    candidate = [_run(0.949, 1.0) for _ in range(PROMOTION_RUNS)]
    judged = judge_promotion(candidate, baseline)
    assert judged["verdict"] == "NOT_PROMOTABLE"
    assert {f["rule"] for f in judged["failures"]} == {"tuning_recall"}


def test_judge_zero_tolerance_trumps_perfect_recall():
    baseline = [aggregate_run(_suite("alpha")) for _ in range(PROMOTION_RUNS)]
    poisoned = aggregate_run(_suite("beta", unsupported=0))
    poisoned["subsets"]["held_out"]["unsupported"] = 1
    poisoned["subsets"]["held_out"]["zero_tolerance_ok"] = False
    judged = judge_promotion([poisoned] * PROMOTION_RUNS, baseline)
    assert judged["verdict"] == "NOT_PROMOTABLE"
    assert {f["rule"] for f in judged["failures"]} == {"zero_tolerance"}


# -- the mechanics, on an injected runner -----------------------------------------


def test_evaluating_the_active_profile_is_refused(gate_service):
    with pytest.raises(PromotionError, match="already the active"):
        gate_service.evaluate_profile("alpha", "Tester")


def test_promotion_moves_the_pointer_for_new_cases_only(gate_service):
    before = gate_service.create_case(BriefIn(**GOLD_CASES[0]["brief"]))
    assert before["profile"]["id"] == "alpha"
    record = gate_service.evaluate_profile("beta", "Tester")
    assert record["verdict"] == "PROMOTABLE"
    assert record["runs_requested"] == PROMOTION_RUNS
    assert len(record["runs"]) == PROMOTION_RUNS
    gate_service.promote_profile(record["evaluation_id"], "Tester")
    assert gate_service.profile.id == "beta"
    after = gate_service.create_case(BriefIn(**GOLD_CASES[0]["brief"]))
    assert after["profile"]["id"] == "beta"
    # The case created before the promotion keeps its own stamp.
    assert gate_service.store.get_case(before["case_id"])["profile"]["id"] == "alpha"


def test_a_fresh_service_follows_the_log_not_the_env(gate_settings, gate_service):
    record = gate_service.evaluate_profile("beta", "Tester")
    gate_service.promote_profile(record["evaluation_id"], "Tester")
    # PROFILE_ID still says alpha; the promotion log governs.
    fresh = CaseService(gate_settings)
    assert fresh.profile.id == "beta"
    info = fresh.active_profile_info()
    assert info["source"].startswith("promotion ")


def test_promote_refuses_a_not_promotable_evaluation(gate_settings):
    service = CaseService(gate_settings)
    service.gate.suite_runner = _runner(alpha={}, beta={"found": 9})
    record = service.evaluate_profile("beta", "Tester")
    assert record["verdict"] == "NOT_PROMOTABLE"
    with pytest.raises(PromotionError, match="NOT_PROMOTABLE"):
        service.promote_profile(record["evaluation_id"], "Tester")
    assert service.profile.id == "alpha"


def test_promote_refuses_a_stale_evaluation(gate_service):
    first = gate_service.evaluate_profile("beta", "Tester")
    second = gate_service.evaluate_profile("beta", "Tester")
    gate_service.promote_profile(second["evaluation_id"], "Tester")
    # The first evaluation judged beta against alpha; alpha is no
    # longer the incumbent it would displace.
    with pytest.raises(PromotionError, match="stale"):
        gate_service.promote_profile(first["evaluation_id"], "Tester")


def test_promote_unknown_evaluation_raises_key_error(gate_service):
    with pytest.raises(KeyError):
        gate_service.promote_profile("eval-does-not-exist", "Tester")


def test_rollback_restores_the_previous_profile_as_its_own_record(gate_service):
    record = gate_service.evaluate_profile("beta", "Tester")
    promotion = gate_service.promote_profile(record["evaluation_id"], "Tester")
    rollback = gate_service.rollback_profile("Tester", note="beta misbehaves live")
    assert gate_service.profile.id == "alpha"
    assert rollback["kind"] == "rollback"
    assert rollback["reverses"] == promotion["promotion_id"]
    assert rollback["candidate"]["id"] == "alpha"
    assert rollback["incumbent"]["id"] == "beta"
    kinds = [p["kind"] for p in gate_service.store.list_promotions()]
    assert kinds == ["promotion", "rollback"]


def test_rollback_with_no_promotion_is_refused(gate_service):
    with pytest.raises(PromotionError, match="nothing to roll back"):
        gate_service.rollback_profile("Tester")


def test_the_gate_records_are_append_only(gate_service):
    store = gate_service.store
    for verb in (
        "update_evaluation",
        "delete_evaluation",
        "update_promotion",
        "delete_promotion",
        "set_active_profile",
    ):
        assert not hasattr(store, verb), verb
    assert not hasattr(gate_service, "set_active_profile")


# -- the tamper control -------------------------------------------------------------


def test_tamper_control_a_profile_edited_after_evaluation_cannot_promote(
    gate_service, profiles_dir
):
    record = gate_service.evaluate_profile("beta", "Tester")
    assert record["verdict"] == "PROMOTABLE"
    path = profiles_dir / "beta.json"
    doc = json.loads(path.read_text())
    doc["prompt_sections"]["planner-base"] += "\n(edited after evaluation)"
    path.write_text(json.dumps(doc))
    assert profile_file_sha256("beta", profiles_dir) != record["candidate_sha256"]
    with pytest.raises(PromotionError, match="changed after it was evaluated"):
        gate_service.promote_profile(record["evaluation_id"], "Tester")
    assert gate_service.profile.id == "alpha"


# -- the API surface ------------------------------------------------------------------


def test_gate_over_http(gate_settings, gate_service):
    client = TestClient(create_app(gate_settings, service=gate_service))
    info = client.get("/api/profiles").json()
    assert info["active"]["id"] == "alpha"
    assert info["source"].startswith("bootstrap")
    evaluated = client.post(
        "/api/profiles/evaluate",
        json={"candidate_id": "beta", "analyst": "HTTP Tester"},
    )
    assert evaluated.status_code == 200
    assert evaluated.json()["verdict"] == "PROMOTABLE"
    promoted = client.post(
        "/api/profiles/promote",
        json={
            "evaluation_id": evaluated.json()["evaluation_id"],
            "analyst": "HTTP Tester",
        },
    )
    assert promoted.status_code == 200
    assert promoted.json()["active"]["id"] == "beta"
    promotions = client.get("/api/profiles/promotions").json()
    assert [p["kind"] for p in promotions] == ["promotion"]
    missing = client.post(
        "/api/profiles/promote",
        json={"evaluation_id": "eval-nope", "analyst": "HTTP Tester"},
    )
    assert missing.status_code == 404
    rolled = client.post(
        "/api/profiles/rollback",
        json={"analyst": "HTTP Tester", "note": "http rollback"},
    )
    assert rolled.status_code == 200
    assert rolled.json()["active"]["id"] == "alpha"
    assert client.get("/api/profiles").json()["active"]["id"] == "alpha"


# -- the controls, on the real suite ---------------------------------------------------


def test_negative_control_the_openai_profile_fails_the_gate(tmp_path):
    settings = load_settings(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
        profile_id="offline-stub",
    )
    service = CaseService(settings)  # the real suite runner
    record = service.evaluate_profile("openai-gpt", "Control")
    assert record["verdict"] == "NOT_PROMOTABLE"
    rules = {f["rule"] for f in record["failures"]}
    # The narrowed market-scan loses gold facts on both subsets; the
    # held-out rule is among the failures, with numbers in the record.
    assert "held_out_recall" in rules
    assert "tuning_recall" in rules
    worst_held_out = min(
        r["subsets"]["held_out"]["recall"] for r in record["runs"]
    )
    bar = record["bars"]["held_out"] - HELD_OUT_TOLERANCE
    assert worst_held_out < bar
    # And it cannot be promoted past the verdict.
    with pytest.raises(PromotionError, match="NOT_PROMOTABLE"):
        service.promote_profile(record["evaluation_id"], "Control")


def test_positive_control_a_copy_of_the_incumbent_passes_and_cycles(tmp_path):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    for name in ("offline-stub.json", "openai-gpt.json"):
        (profiles_dir / name).write_bytes((SHIPPED_PROFILES / name).read_bytes())
    copy_doc = json.loads((SHIPPED_PROFILES / "offline-stub.json").read_text())
    # Identical in every behavioural field; only the identity fields
    # a distinct artifact must have (its id, its version) differ.
    copy_doc["id"] = "offline-stub-copy"
    copy_doc["version"] = 2
    (profiles_dir / "offline-stub-copy.json").write_text(json.dumps(copy_doc))
    settings = load_settings(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
        profile_id="offline-stub",
        profile_dir=str(profiles_dir),
    )
    service = CaseService(settings)  # the real suite runner
    record = service.evaluate_profile("offline-stub-copy", "Control")
    assert record["verdict"] == "PROMOTABLE", record["failures"]
    assert all(
        r["subsets"][split]["recall"] == 1.0
        for r in record["runs"]
        for split in ("tuning", "held_out")
    )
    service.promote_profile(record["evaluation_id"], "Control")
    assert service.profile.id == "offline-stub-copy"
    service.rollback_profile("Control", note="positive control rollback")
    assert service.profile.id == "offline-stub"
    kinds = [p["kind"] for p in service.store.list_promotions()]
    assert kinds == ["promotion", "rollback"]
