"""The profile regression gate: a harness profile earns production
only by passing evals, and the decision is a record, not an edit.

The tuning loop this enforces (design brief, R7): propose a profile
change → run the eval packs → keep or roll back, with the evidence
kept. Concretely:

* SPLIT — the packs are partitioned in the eval data (corpus.py)
  into a TUNING subset and a HELD-OUT subset. A candidate is judged
  on both: it must match the incumbent on the cases profiles are
  tuned against, and must not regress on the cases it was not tuned
  against — the held-out cases are what stop a profile from being
  rehearsed into a pass.
* REPEATED — a promotion evaluation runs the candidate's full
  suite PROMOTION_RUNS times, because model behaviour is
  stochastic. Every run is kept, and the gate reads the WORST run:
  one lucky pass promotes nothing.
* RULE — on every run: tuning recall >= the incumbent's recorded
  baseline (the worst of the incumbent's own recorded runs);
  held-out recall >= that baseline minus HELD_OUT_TOLERANCE; and
  the zero-tolerance lines hold on both subsets (unsupported claims
  in final reports = 0, citation completeness = 1.000, citation
  correctness = 1.000). Any failure is NOT_PROMOTABLE, with the
  failing runs and metrics named in the record.
* INTEGRITY — every evaluation snapshots the candidate profile
  file's SHA-256. A file edited after evaluation no longer matches
  its evidence, and promotion is refused at decision time.
* DECISION — evaluations and promotions are append-only store
  records. The active profile pointer moves only through a
  recorded promotion by a named analyst; rollback is the same
  machinery promoting the previous profile back, judged on its
  recorded runs, with its own record. PROFILE_ID remains only the
  bootstrap default for a store with no promotions yet.

The gate grades with the same ruler as the eval runner:
`evaluation.evaluate_suite` runs the packs through the pipeline.
Tests inject a faster runner for the mechanics; the shipped
controls (tests/test_promotion.py, evals/promotion_controls.py)
run the real suite.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from .profiles import HarnessProfile, default_profiles_dir, load_profile
from .store import Store, utcnow

# A promotion evaluation runs the full suite this many times. Model
# behaviour is stochastic; one pass is an anecdote.
PROMOTION_RUNS = 3

# How far held-out recall may fall below the incumbent's recorded
# baseline before the candidate is a regression. The tuning subset
# allows no fall at all.
HELD_OUT_TOLERANCE = 0.02

SUBSETS = ("tuning", "held_out")


class PromotionError(RuntimeError):
    """A promotion or rollback the gate refuses. The message names why."""


def profile_file_sha256(profile_id: str, profiles_dir=None) -> str:
    directory = Path(profiles_dir) if profiles_dir else default_profiles_dir()
    path = directory / f"{profile_id}.json"
    if not path.is_file():
        raise PromotionError(f"profile {profile_id!r} not found: no file {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset_metrics(cases: list[dict]) -> dict:
    """Aggregate one suite run's per-case metrics by eval subset.

    Recall is pooled (gold facts found over gold facts in the
    subset) so a case counts in proportion to what it plants. The
    zero-tolerance lines are per-case absolutes: one case below
    1.000 on citation or correctness, or one unsupported claim,
    fails the subset's line and the case is named.
    """
    subsets: dict[str, dict] = {}
    for split in SUBSETS:
        members = [c for c in cases if c["split"] == split]
        found = sum(c["found"] for c in members)
        gold = sum(c["gold"] for c in members)
        subsets[split] = {
            "cases": [c["key"] for c in members],
            "recall": found / gold if gold else 1.0,
            "found": found,
            "gold": gold,
            "unsupported": sum(c["unsupported"] for c in members),
            "citation_failures": [c["key"] for c in members if c["citation"] != 1.0],
            "correctness_failures": [
                c["key"] for c in members if c["correctness"] != 1.0
            ],
        }
        subsets[split]["zero_tolerance_ok"] = (
            subsets[split]["unsupported"] == 0
            and not subsets[split]["citation_failures"]
            and not subsets[split]["correctness_failures"]
        )
    return subsets


def aggregate_run(suite: dict) -> dict:
    """One suite pass, reduced to what the gate reads."""
    return {"cases": suite["cases"], "subsets": subset_metrics(suite["cases"])}


def baseline_bars(baseline_runs: list[dict]) -> dict:
    """The bar the incumbent sets: the worst of its own recorded
    runs, per subset. A candidate is not compared against the
    incumbent's best day."""
    return {
        split: min(run["subsets"][split]["recall"] for run in baseline_runs)
        for split in SUBSETS
    }


def judge_promotion(candidate_runs: list[dict], baseline_runs: list[dict]) -> dict:
    """The promotion rule, applied to recorded runs. Pure: the same
    records always produce the same verdict, and every failure names
    the run, the subset, the rule, and the numbers."""
    bars = baseline_bars(baseline_runs)
    failures: list[dict] = []
    for index, run in enumerate(candidate_runs, start=1):
        tuning = run["subsets"]["tuning"]
        if tuning["recall"] < bars["tuning"]:
            failures.append(
                {
                    "run": index,
                    "subset": "tuning",
                    "rule": "tuning_recall",
                    "observed": tuning["recall"],
                    "required": bars["tuning"],
                    "detail": (
                        f"run {index} tuning recall {tuning['recall']:.3f} "
                        f"is below the incumbent baseline {bars['tuning']:.3f}"
                    ),
                }
            )
        held_out = run["subsets"]["held_out"]
        floor = bars["held_out"] - HELD_OUT_TOLERANCE
        if held_out["recall"] < floor:
            failures.append(
                {
                    "run": index,
                    "subset": "held_out",
                    "rule": "held_out_recall",
                    "observed": held_out["recall"],
                    "required": floor,
                    "detail": (
                        f"run {index} held-out recall {held_out['recall']:.3f} "
                        f"is below the incumbent baseline {bars['held_out']:.3f} "
                        f"minus tolerance {HELD_OUT_TOLERANCE}"
                    ),
                }
            )
        for split in SUBSETS:
            subset = run["subsets"][split]
            if not subset["zero_tolerance_ok"]:
                broken = []
                if subset["unsupported"]:
                    broken.append(
                        f"{subset['unsupported']} unsupported claim(s) in final reports"
                    )
                if subset["citation_failures"]:
                    broken.append(
                        "citation below 1.000 on "
                        + ", ".join(subset["citation_failures"])
                    )
                if subset["correctness_failures"]:
                    broken.append(
                        "correctness below 1.000 on "
                        + ", ".join(subset["correctness_failures"])
                    )
                failures.append(
                    {
                        "run": index,
                        "subset": split,
                        "rule": "zero_tolerance",
                        "observed": {
                            "unsupported": subset["unsupported"],
                            "citation_failures": subset["citation_failures"],
                            "correctness_failures": subset["correctness_failures"],
                        },
                        "required": "unsupported 0, citation 1.000, correctness 1.000",
                        "detail": f"run {index} {split}: " + "; ".join(broken),
                    }
                )
    return {
        "verdict": "PROMOTABLE" if not failures else "NOT_PROMOTABLE",
        "failures": failures,
        "bars": bars,
    }


def _default_suite_runner(profile_id: str, profiles_dir=None) -> dict:
    # Imported lazily: evaluation imports the service, and the
    # service imports this module.
    from .evaluation import evaluate_suite

    return evaluate_suite(profile_id, profiles_dir)


class ProfileGate:
    """Evaluations, promotions, and rollbacks against one store.

    The gate never decides which profile is active by itself — it
    writes records, and the service resolves the active profile
    from the newest promotion record. `suite_runner` is injectable
    so tests can exercise the mechanics without re-running the full
    packs; the shipped controls use the real runner.
    """

    def __init__(self, store: Store, profiles_dir=None, suite_runner=None):
        self.store = store
        self.profiles_dir = profiles_dir
        # The default runner grades in this gate's own profiles
        # directory: a candidate that only exists there must be
        # runnable there.
        self.suite_runner = suite_runner or (
            lambda profile_id: _default_suite_runner(profile_id, self.profiles_dir)
        )

    # -- evidence lookup -----------------------------------------------------
    def _recorded_runs_for(
        self, profile: HarnessProfile
    ) -> tuple[list[dict], str, str] | None:
        """Recorded gate runs for one profile, newest evidence
        first: its runs as a candidate, or as an evaluation's
        baseline, provided the profile file still hashes to what
        the record snapshotted. Returns (runs, evaluation_id, role)
        or None."""
        try:
            current_sha = profile_file_sha256(profile.id, self.profiles_dir)
        except PromotionError:
            return None
        target = profile.record()
        for record in reversed(self.store.list_evaluations()):
            if (
                record["candidate"] == target
                and record["candidate_sha256"] == current_sha
            ):
                return record["runs"], record["evaluation_id"], "candidate"
            baseline = record.get("baseline") or {}
            if (
                baseline.get("profile") == target
                and baseline.get("profile_sha256") == current_sha
            ):
                return baseline["runs"], record["evaluation_id"], "baseline"
        return None

    def _baseline_for(
        self, incumbent: HarnessProfile, runs: int
    ) -> dict:
        found = self._recorded_runs_for(incumbent)
        sha = profile_file_sha256(incumbent.id, self.profiles_dir)
        if found is not None:
            recorded_runs, source_id, _role = found
            return {
                "profile": incumbent.record(),
                "profile_sha256": sha,
                "source": f"recorded runs from evaluation {source_id}",
                "runs": recorded_runs,
            }
        fresh = [
            aggregate_run(self.suite_runner(incumbent.id)) for _ in range(runs)
        ]
        return {
            "profile": incumbent.record(),
            "profile_sha256": sha,
            "source": "fresh evaluation of the incumbent",
            "runs": fresh,
        }

    # -- evaluation ------------------------------------------------------------
    def evaluate(
        self,
        candidate_id: str,
        incumbent: HarnessProfile,
        evaluated_by: str,
        runs: int = PROMOTION_RUNS,
    ) -> dict:
        candidate = load_profile(candidate_id, self.profiles_dir)
        if candidate.record() == incumbent.record():
            raise PromotionError(
                f"{candidate_id} v{candidate.version} is already the active "
                "profile; there is nothing to promote"
            )
        candidate_sha = profile_file_sha256(candidate_id, self.profiles_dir)
        candidate_runs = [
            aggregate_run(self.suite_runner(candidate_id)) for _ in range(runs)
        ]
        baseline = self._baseline_for(incumbent, runs)
        judged = judge_promotion(candidate_runs, baseline["runs"])
        record = {
            "evaluation_id": f"eval-{uuid.uuid4().hex[:10]}",
            "candidate": candidate.record(),
            "candidate_sha256": candidate_sha,
            "incumbent": incumbent.record(),
            "runs_requested": runs,
            "runs": candidate_runs,
            "baseline": baseline,
            "bars": judged["bars"],
            "failures": judged["failures"],
            "verdict": judged["verdict"],
            "evaluated_by": evaluated_by,
            "created_at": utcnow(),
        }
        return self.store.add_evaluation(record)

    # -- decisions ---------------------------------------------------------------
    def promote(
        self, evaluation_id: str, analyst: str, active: HarnessProfile
    ) -> dict:
        record = self.store.get_evaluation(evaluation_id)
        if record is None:
            raise KeyError(evaluation_id)
        if record["verdict"] != "PROMOTABLE":
            named = "; ".join(
                f["detail"] for f in record["failures"][:3]
            ) or record["verdict"]
            raise PromotionError(
                f"evaluation {evaluation_id} is {record['verdict']}: {named}"
            )
        if record["incumbent"] != active.record():
            raise PromotionError(
                f"evaluation {evaluation_id} judged {record['candidate']['id']} "
                f"against {record['incumbent']['id']} v{record['incumbent']['version']}, "
                f"but the active profile is now {active.id} v{active.version}; "
                "the evidence is stale — re-evaluate against the current incumbent"
            )
        current_sha = profile_file_sha256(
            record["candidate"]["id"], self.profiles_dir
        )
        if current_sha != record["candidate_sha256"]:
            raise PromotionError(
                f"profile {record['candidate']['id']!r} changed after it was "
                "evaluated (file hash no longer matches the evaluation's "
                "snapshot); the evidence no longer describes the file — "
                "re-evaluate before promoting"
            )
        promotion = {
            "promotion_id": f"promo-{uuid.uuid4().hex[:10]}",
            "kind": "promotion",
            "candidate": record["candidate"],
            "candidate_sha256": record["candidate_sha256"],
            "incumbent": record["incumbent"],
            "evaluation_id": evaluation_id,
            "evidence_evaluation_id": evaluation_id,
            "analyst": analyst,
            "note": None,
            "created_at": utcnow(),
        }
        return self.store.add_promotion(promotion)

    def rollback(self, analyst: str, note: str | None, active: HarnessProfile) -> dict:
        latest = self.store.latest_promotion()
        if latest is None:
            raise PromotionError(
                "nothing to roll back: no promotion has been recorded, "
                "so the bootstrap profile is simply the configuration"
            )
        if latest["candidate"] != active.record():
            raise PromotionError(
                "the newest promotion record does not name the active "
                "profile; the promotion log and the service disagree, "
                "so the gate refuses to guess"
            )
        target = load_profile(latest["incumbent"]["id"], self.profiles_dir)
        if target.record() != latest["incumbent"]:
            raise PromotionError(
                f"the profile file for {target.id!r} is now v{target.version}, "
                f"but the promotion log recorded v{latest['incumbent']['version']}; "
                "rolling back to a different version than the one that ran "
                "is a new promotion, not a rollback — evaluate it instead"
            )
        evidence = self._recorded_runs_for(target)
        if evidence is None:
            raise PromotionError(
                f"no recorded gate runs for {target.id!r} match the file "
                "as it stands now; a rollback rests on recorded evidence, "
                "so evaluate the profile instead"
            )
        target_runs, evidence_id, role = evidence
        incumbent_evidence = self._recorded_runs_for(active)
        if incumbent_evidence is None:
            raise PromotionError(
                f"no recorded gate runs for the active profile {active.id!r} "
                "match its file; the rollback cannot be judged"
            )
        incumbent_runs, _eid, _role = incumbent_evidence
        judged = judge_promotion(target_runs, incumbent_runs)
        if judged["verdict"] != "PROMOTABLE":
            named = "; ".join(f["detail"] for f in judged["failures"][:3])
            raise PromotionError(
                f"the recorded runs for {target.id!r} do not pass the gate "
                f"against the active profile: {named}"
            )
        promotion = {
            "promotion_id": f"promo-{uuid.uuid4().hex[:10]}",
            "kind": "rollback",
            "candidate": target.record(),
            "candidate_sha256": profile_file_sha256(
                target.id, self.profiles_dir
            ),
            "incumbent": active.record(),
            "evaluation_id": None,
            "evidence_evaluation_id": evidence_id,
            "evidence_role": role,
            "reverses": latest["promotion_id"],
            "analyst": analyst,
            "note": note,
            "created_at": utcnow(),
        }
        return self.store.add_promotion(promotion)
