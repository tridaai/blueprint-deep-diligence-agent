#!/usr/bin/env python3
"""The profile regression gate's controls, run for real.

This is not the eval gate and it decides nothing about the default
profile. It exercises the promotion gate (promotion.py) end to end
with the real suite runner, on scratch stores, to prove the gate
discriminates:

  NEGATIVE control — the shipped openai-gpt profile as candidate
  against the offline-stub incumbent must come back
  NOT_PROMOTABLE. Its narrowed market-scan loses gold facts under
  the stub; the verdict must name the failing rules with numbers.

  POSITIVE control — a copy of the incumbent under a new id and
  version (identical in every behavioural field) must come back
  PROMOTABLE, and a promotion + rollback cycle must move the
  active profile and move it back, each as its own record.

Both controls run the gate's full discipline: PROMOTION_RUNS suite
passes per side, worst run decides. On the deterministic stub that
takes minutes, not seconds — that is the honest cost of the
evidence. Exit code 0 only when both controls behave as labelled.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_diligence.config import load_settings  # noqa: E402
from deep_diligence.promotion import PROMOTION_RUNS  # noqa: E402
from deep_diligence.service import CaseService  # noqa: E402

ANALYST = "Promotion Controls"


def _service(tmp: Path, profiles_dir: Path | None = None) -> CaseService:
    return CaseService(
        load_settings(
            sqlite_path=str(tmp / "diligence.db"),
            object_dir=str(tmp / "objects"),
            migrations_dir=str(REPO_ROOT / "migrations"),
            profile_id="offline-stub",
            profile_dir=str(profiles_dir) if profiles_dir else None,
        )
    )


def _worst(runs: list[dict], split: str) -> float:
    return min(r["subsets"][split]["recall"] for r in runs)


def negative_control() -> bool:
    print("=" * 72)
    print("NEGATIVE control: openai-gpt (candidate) vs offline-stub (incumbent)")
    service = _service(Path(tempfile.mkdtemp(prefix="dd-control-neg-")))
    record = service.evaluate_profile("openai-gpt", ANALYST)
    print(
        f"  verdict: {record['verdict']} after {record['runs_requested']} runs "
        f"(gate runs {PROMOTION_RUNS})"
    )
    print(
        f"  incumbent baseline bars: tuning {_worst(record['baseline']['runs'], 'tuning'):.3f}, "
        f"held-out {_worst(record['baseline']['runs'], 'held_out'):.3f}"
    )
    print(
        f"  candidate worst runs: tuning {_worst(record['runs'], 'tuning'):.3f}, "
        f"held-out {_worst(record['runs'], 'held_out'):.3f}"
    )
    for failure in record["failures"]:
        print(f"  failure: {failure['detail']}")
    held_out_failed = any(
        f["rule"] == "held_out_recall" for f in record["failures"]
    )
    ok = record["verdict"] == "NOT_PROMOTABLE" and held_out_failed
    print(f"  -> {'OK' if ok else 'CONTROL FAILED'}")
    return ok


def positive_control() -> bool:
    print("=" * 72)
    print("POSITIVE control: a copy of the incumbent under a new id/version")
    tmp = Path(tempfile.mkdtemp(prefix="dd-control-pos-"))
    profiles_dir = tmp / "profiles"
    profiles_dir.mkdir()
    shipped = REPO_ROOT / "profiles"
    for name in ("offline-stub.json", "openai-gpt.json"):
        (profiles_dir / name).write_bytes((shipped / name).read_bytes())
    copy_doc = json.loads((shipped / "offline-stub.json").read_text())
    copy_doc["id"] = "offline-stub-copy"
    copy_doc["version"] = 2
    (profiles_dir / "offline-stub-copy.json").write_text(
        json.dumps(copy_doc, indent=2)
    )
    print(
        "  the copy differs from the incumbent only in its id and "
        "version — the identity fields a distinct artifact must have"
    )
    service = _service(tmp / "store", profiles_dir)
    record = service.evaluate_profile("offline-stub-copy", ANALYST)
    print(
        f"  verdict: {record['verdict']} after {record['runs_requested']} runs"
    )
    print(
        f"  candidate worst runs: tuning {_worst(record['runs'], 'tuning'):.3f}, "
        f"held-out {_worst(record['runs'], 'held_out'):.3f}"
    )
    for failure in record["failures"]:
        print(f"  failure: {failure['detail']}")
    if record["verdict"] != "PROMOTABLE":
        print("  -> CONTROL FAILED")
        return False
    service.promote_profile(record["evaluation_id"], ANALYST)
    after_promote = service.profile.record()
    print(f"  after promotion, active: {after_promote['id']} v{after_promote['version']}")
    service.rollback_profile(ANALYST, note="Positive control rollback")
    after_rollback = service.profile.record()
    print(
        f"  after rollback, active: {after_rollback['id']} v{after_rollback['version']}"
    )
    promotions = service.store.list_promotions()
    kinds = [p["kind"] for p in promotions]
    ok = (
        after_promote["id"] == "offline-stub-copy"
        and after_rollback["id"] == "offline-stub"
        and kinds == ["promotion", "rollback"]
    )
    print(f"  promotion log kinds: {kinds}")
    print(f"  -> {'OK' if ok else 'CONTROL FAILED'}")
    return ok


def main() -> int:
    negative_ok = negative_control()
    positive_ok = positive_control()
    print("=" * 72)
    print(
        "CONTROLS:",
        "PASS" if (negative_ok and positive_ok) else "FAIL",
        "(negative:",
        "OK" if negative_ok else "FAILED",
        "· positive:",
        "OK" if positive_ok else "FAILED",
        ")",
    )
    return 0 if (negative_ok and positive_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
