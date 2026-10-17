"""CaseService: the governance shell around the harness.

State machine (see models.STATUSES):
  INTAKE --plan--> PLAN_READY --approve--> APPROVED --execute-->
  DRAFT_READY --approve--> FINAL
Rejections return the case to PLAN_REJECTED / REPORT_REJECTED, from
which planning / execution may run again. A budget-exhausted execution
lands in BUDGET_EXHAUSTED with a partial draft, which the analyst may
still approve or reject — or amend the budget (recorded, never below
what the case already consumed) to return the case to APPROVED and
re-execute. Execution attempted from any other state is
a GateError — the plan gate is enforced here, in code, not by prompt.
"""

from __future__ import annotations

import json
import logging
import uuid

from langchain_core.messages import HumanMessage

from .backend import EvidenceBackend
from .budgets import (
    PHASES,
    BudgetExceededError,
    BudgetGuard,
    case_caps,
    fold_budget_ledger,
    resolve_phase_budget,
)
from .composer import apply_conflicts, apply_verification, compose_report
from .conflicts import detect_conflicts
from .config import Settings
from .corpus import companies
from .evidence import ingest_case_evidence
from .harness import build_agent, execute_message, plan_message
from .ledger import verify_ledger
from .llm import build_model
from .models import BudgetIn, BriefIn
from .objects import ObjectStore, build_object_store
from .profiles import ProfileError, load_profile
from .promotion import ProfileGate
from .schedule import build_handoff, plan_defects, waves, workstream_dependencies
from .store import Store, build_store, utcnow
from .tools import build_case_tools
from .verifier import verify_case

log = logging.getLogger("deep_diligence.service")


class GateError(RuntimeError):
    """A lifecycle transition the case state does not allow."""


def _aggregate_execution_usage(
    usages: list[dict], tripped: dict | None
) -> dict:
    """One execution's usage across its scheduled waves.

    Steps, tokens, spawns, and wall clock sum across waves; the
    budget ledger keeps each wave's own checkpoint, and this is the
    case-level picture the case record carries. A single-wave
    execution — every plan without dependencies — returns its one
    run's usage untouched, exactly as execution always reported.
    """
    if len(usages) == 1:
        return usages[0]
    total = {
        "phase": "execution",
        "steps": sum(u.get("steps", 0) for u in usages),
        "subagents": sum(u.get("subagents", 0) for u in usages),
        "tokens": sum(u.get("tokens", 0) for u in usages),
        "tool_calls": sum(u.get("tool_calls", 0) for u in usages),
        "wall_seconds": round(
            sum(float(u.get("wall_seconds", 0.0)) for u in usages), 2
        ),
        "exceeded": (tripped or {}).get("exceeded"),
        "interrupted_spawns": (tripped or {}).get("interrupted_spawns", []),
    }
    return total


class CaseService:
    def __init__(self, settings: Settings):
        self.settings = settings
        # The store is built first because the promotion log lives
        # in it. The active harness profile is then resolved before
        # any case work and validated at load: a profile that refuses
        # to load refuses the startup with it (fail-closed, no silent
        # fallback). The model is injected separately by the provider
        # boundary; the profile is configuration.
        self.store: Store = build_store(settings)
        self.profile = self._resolve_active_profile()
        self.gate = ProfileGate(self.store, settings.profile_dir)
        self.objects: ObjectStore = build_object_store(settings)
        log.info(
            "harness profile %s v%d (target %s)",
            self.profile.id,
            self.profile.version,
            self.profile.target_model,
        )

    def _resolve_active_profile(self):
        """PROFILE_ID is the bootstrap default only. Once a
        promotion is recorded, the newest promotion record names the
        active profile, and there is no other way to move the
        pointer: this method reads the log, nothing anywhere writes
        an active-profile field."""
        bootstrap = load_profile(
            self.settings.profile_id, self.settings.profile_dir
        )
        latest = self.store.latest_promotion()
        if latest is None:
            return bootstrap
        promoted = load_profile(
            latest["candidate"]["id"], self.settings.profile_dir
        )
        if promoted.record() != latest["candidate"]:
            raise ProfileError(
                f"promotion {latest['promotion_id']} made "
                f"{latest['candidate']['id']} v{latest['candidate']['version']} "
                f"active, but the profile file is now v{promoted.version}; "
                "the gate refuses to run a version the log did not promote"
            )
        return promoted

    # -- profile regression gate ----------------------------------------------
    def active_profile_info(self) -> dict:
        latest = self.store.latest_promotion()
        return {
            "active": self.profile.record(),
            "source": (
                f"promotion {latest['promotion_id']}"
                if latest is not None
                else "bootstrap (PROFILE_ID; no promotion recorded)"
            ),
            "bootstrap": {
                "id": self.settings.profile_id,
            },
            "latest_promotion": latest,
        }

    def evaluate_profile(self, candidate_id: str, analyst: str) -> dict:
        return self.gate.evaluate(candidate_id, self.profile, analyst)

    def promote_profile(self, evaluation_id: str, analyst: str) -> dict:
        record = self.gate.promote(evaluation_id, analyst, self.profile)
        # The pointer moves for new cases from this moment; cases
        # already created keep the profile stamp they were born with.
        self.profile = load_profile(
            record["candidate"]["id"], self.settings.profile_dir
        )
        return record

    def rollback_profile(self, analyst: str, note: str | None = None) -> dict:
        record = self.gate.rollback(analyst, note, self.profile)
        self.profile = load_profile(
            record["candidate"]["id"], self.settings.profile_dir
        )
        return record

    # -- helpers ------------------------------------------------------------
    def _require_case(self, case_id: str) -> dict:
        case = self.store.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        return case

    def _backend(self, case_id: str) -> EvidenceBackend:
        return EvidenceBackend(self.store, self.objects, case_id)

    def _flush_audit(self, case_id: str, sink: list) -> None:
        for event, payload in sink:
            self.store.add_audit(case_id, event, payload)

    def _run_agent(self, case: dict, message: str, phase: str):
        """One harness run, under the phase's resolved caps and the
        case's caps. Returns (usage, error).

        The run is a ledger sequence: a `budget_granted` entry names
        the caps it runs under, the guard's own events (spawns, the
        exhaustion if one trips) follow, and a `budget_consumed`
        checkpoint closes it. Prior case consumption — spawns and
        wall clock already spent by earlier phase runs — is folded
        from the ledger and seeds the guard, so the case-level caps
        bind across phases and across re-executions.
        """
        case_id = case["case_id"]
        backend = self._backend(case_id)
        tools = build_case_tools(self.store, self.objects, case)
        agent = build_agent(
            model=build_model(self.settings),
            backend=backend,
            case_tools=tools,
            profile=self.profile,
        )
        budget = case["brief"]["budget"]
        phase_caps = resolve_phase_budget(budget, phase)
        caps = case_caps(budget)
        consumed = fold_budget_ledger(self.store.list_audit(case_id))[
            "case_consumed"
        ]
        self.store.add_audit(
            case_id,
            "budget_granted",
            {
                "phase": phase,
                "caps": phase_caps,
                "case_caps": caps,
                # Every run records the profile it ran under, so the
                # audit trail attributes planning and execution to a
                # profile id + version, not just to a date.
                "profile": self.profile.record(),
            },
        )
        sink: list = []
        guard = BudgetGuard(
            {**phase_caps, **caps},
            audit_sink=sink,
            phase=phase,
            prior_subagents=consumed["subagents"],
            prior_wall_seconds=consumed["wall_seconds"],
        )
        config = {
            "callbacks": [guard],
            "recursion_limit": max(150, phase_caps["max_steps"] * 6),
        }
        error: BudgetExceededError | None = None
        failure: Exception | None = None
        try:
            agent.invoke({"messages": [HumanMessage(content=message)]}, config=config)
        except BudgetExceededError as exc:
            error = exc
        except Exception as exc:  # noqa: BLE001 — re-raised below
            # A crashed run still owes the ledger its events and its
            # consumption checkpoint; the caller decides what the
            # failure means for the workstreams it was running.
            failure = exc
        self._flush_audit(case_id, sink)
        usage = guard.usage()
        self.store.add_audit(
            case_id, "budget_consumed", {"phase": phase, "usage": usage}
        )
        if failure is not None:
            raise failure
        return usage, error

    # -- intake ---------------------------------------------------------------
    def create_case(self, brief: BriefIn) -> dict:
        known = {c["subject_ref"] for c in companies()}
        if brief.subject_ref not in known:
            raise ValueError(
                f"unknown subject_ref {brief.subject_ref!r}; "
                f"the offline corpus covers {sorted(known)}"
            )
        case_id = f"case-{uuid.uuid4().hex[:10]}"
        now = utcnow()
        case = {
            "case_id": case_id,
            "status": "INTAKE",
            "subject": brief.subject,
            "subject_ref": brief.subject_ref,
            "analyst": brief.analyst,
            "brief": brief.model_dump(),
            "usage": {},
            # The profile stamp: this case's planning and execution
            # run under this profile (id + version), and the case
            # view shows it.
            "profile": self.profile.record(),
            "created_at": now,
            "updated_at": now,
        }
        self.store.create_case(case)
        evidence = ingest_case_evidence(self.store, self.objects, case)
        self.store.add_audit(
            case_id,
            "case_created",
            {"subject": brief.subject, "profile": self.profile.record()},
        )
        self.store.add_audit(
            case_id, "evidence_ingested", {"count": len(evidence)}
        )
        log.info("case created %s subject=%s evidence=%d", case_id, brief.subject, len(evidence))
        return case

    # -- planning -------------------------------------------------------------
    def run_planning(self, case_id: str) -> dict:
        case = self._require_case(case_id)
        if case["status"] not in ("INTAKE", "PLAN_REJECTED"):
            raise GateError(
                f"cannot plan from status {case['status']}"
            )
        backend = self._backend(case_id)
        backend.delete_direct("/plan.json")
        backend.delete_direct("/plan.md")
        usage, error = self._run_agent(case, plan_message(case), "planning")
        case = self.store.update_case(
            case_id, usage={**case.get("usage", {}), "planning": usage}
        )
        if error is not None:
            self.store.add_audit(case_id, "planning_budget_exceeded", usage)
            raise GateError(f"planning stopped on the {error.kind} budget")
        raw = backend.read_text_direct("/plan.json")
        if raw is None:
            raise GateError("the planning run produced no /plan.json")
        try:
            plan_doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GateError(f"/plan.json is not valid JSON: {exc}") from exc
        markdown = backend.read_text_direct("/plan.md") or ""
        plan = self.store.save_plan(
            case_id,
            {
                "case_id": case_id,
                "status": "PENDING",
                "plan": plan_doc,
                "markdown": markdown,
                "decided_by": None,
                "decision_note": None,
            },
        )
        self.store.add_audit(case_id, "plan_generated", {"version": plan["version"]})
        self.store.update_case(case_id, status="PLAN_READY")
        log.info("case %s plan v%d ready", case_id, plan["version"])
        return plan

    def decide_plan(self, case_id: str, analyst: str, approve: bool, note: str = "") -> dict:
        case = self._require_case(case_id)
        if case["status"] != "PLAN_READY":
            raise GateError(f"no plan awaiting decision (status {case['status']})")
        plan = self.store.get_plan(case_id)
        if approve:
            # The plan gate checks the dependency graph in code
            # (schedule.py): an unknown dependency or a cycle means
            # the plan cannot be scheduled, so it cannot be approved
            # either. The defect is named, and the plan stays at the
            # gate for the analyst to send back.
            defects = plan_defects(
                (plan.get("plan") or {}).get("workstreams", [])
            )
            if defects:
                raise GateError(
                    "plan cannot be approved: " + "; ".join(defects)
                )
        plan["status"] = "APPROVED" if approve else "REJECTED"
        plan["decided_by"] = analyst
        plan["decision_note"] = note
        self.store.update_plan(case_id, plan)
        self.store.add_audit(
            case_id,
            "plan_approved" if approve else "plan_rejected",
            {"version": plan["version"], "analyst": analyst, "note": note},
        )
        return self.store.update_case(
            case_id, status="APPROVED" if approve else "PLAN_REJECTED"
        )

    # -- execution ------------------------------------------------------------
    def run_execution(self, case_id: str) -> dict:
        case = self._require_case(case_id)
        if case["status"] not in ("APPROVED", "REPORT_REJECTED"):
            raise GateError(
                f"execution requires an approved plan (status {case['status']})"
            )
        plan = self.store.get_plan(case_id)
        plan_doc = plan.get("plan") or {}
        workstreams = plan_doc.get("workstreams", [])
        # Defence in depth: the plan gate validated this graph at
        # approval, and an unschedulable graph must still never run.
        defects = plan_defects(workstreams)
        if defects:
            raise GateError(
                "plan cannot be executed: " + "; ".join(defects)
            )
        schedule = waves(workstreams)
        backend = self._backend(case_id)
        backend.delete_direct("/execution/notes.md")
        self.store.update_case(case_id, status="EXECUTING")
        self.store.add_audit(
            case_id,
            "execution_started",
            {
                "plan_version": plan["version"],
                "waves": [
                    [ws["question_id"] for ws in layer] for layer in schedule
                ],
            },
        )
        statuses: dict[str, dict] = {}
        usages: list[dict] = []
        error: BudgetExceededError | None = None
        interrupted: list[str] = []
        tripped_usage: dict | None = None
        for index, layer in enumerate(schedule, start=1):
            usage, error, layer_statuses, layer_interrupted = self._run_wave(
                case, plan_doc, layer, index
            )
            usages.append(usage)
            statuses.update(layer_statuses)
            if error is not None:
                interrupted = layer_interrupted
                tripped_usage = usage
                break
        if error is not None:
            # Waves the stop kept from ever starting: their
            # workstreams are part of the execution record too.
            for entry in workstreams:
                qid = entry["question_id"]
                if qid not in statuses:
                    statuses[qid] = {
                        "question_id": qid,
                        "workstream": entry["workstream"],
                        "status": "not_run",
                        "findings": 0,
                        "reason": (
                            "the run stopped on its budget before "
                            "this workstream started"
                        ),
                    }
        usage = _aggregate_execution_usage(usages, tripped_usage)
        status_list = [
            statuses[ws["question_id"]]
            for ws in workstreams
            if ws["question_id"] in statuses
        ]
        case = self.store.update_case(
            case_id, usage={**case.get("usage", {}), "execution": usage}
        )
        if error is not None:
            report = compose_report(
                self.store,
                self.objects,
                case,
                partial=True,
                exceeded_kind=error.kind,
                exceeded_phase="execution",
                interrupted_questions=interrupted,
                workstream_status=status_list,
            )
            self.store.add_audit(
                case_id, "report_composed", {"partial": True, "usage": usage}
            )
            report = self._verify_draft(case, report)
            self.store.update_case(case_id, status="BUDGET_EXHAUSTED")
            log.info("case %s budget exhausted (%s)", case_id, error.kind)
            return report
        report = compose_report(
            self.store, self.objects, case, workstream_status=status_list
        )
        self.store.add_audit(
            case_id, "report_composed", {"partial": False, "usage": usage}
        )
        report = self._verify_draft(case, report)
        self.store.update_case(case_id, status="DRAFT_READY")
        log.info("case %s draft ready", case_id)
        return report

    def _run_wave(
        self, case: dict, plan_doc: dict, layer: list[dict], wave_index: int
    ) -> tuple[dict, BudgetExceededError | None, dict, list[str]]:
        """One scheduled wave: a single harness run over one
        topological layer of the plan.

        The wave's workstreams are the run's whole world — the
        execute message carries only them, with each dependent's
        handoff (its prerequisites' recorded findings, verbatim)
        attached. Start and finish are audit events per workstream,
        so the schedule is provable after the fact: a dependent's
        `workstream_started` can only follow its prerequisites'
        `workstream_finished`, because the run that spawns it is
        not invoked until the wave that finished them has returned.

        Failure isolation lives here too: a wave whose run raises
        (anything but the budget stop, which _run_agent returns)
        marks its finding-less workstreams failed, with the error
        named, and execution continues with the next wave — one
        branch's failure never fails the whole case silently.
        Returns (usage, budget_error, statuses, interrupted).
        """
        case_id = case["case_id"]
        wave_entries: list[dict] = []
        findings_now = self.store.list_findings(case_id)
        for entry in layer:
            entry = dict(entry)
            deps = workstream_dependencies(entry)
            if deps:
                handoff = build_handoff(findings_now, entry)
                entry["prior_findings"] = handoff
                self.store.add_audit(
                    case_id,
                    "workstream_handoff",
                    {
                        "question_id": entry["question_id"],
                        "wave": wave_index,
                        "depends_on": deps,
                        "finding_ids": [h["finding_id"] for h in handoff],
                    },
                )
            wave_entries.append(entry)
            self.store.add_audit(
                case_id,
                "workstream_started",
                {
                    "question_id": entry["question_id"],
                    "workstream": entry["workstream"],
                    "wave": wave_index,
                },
            )
        wave_plan = {
            "plan": {
                "subject": plan_doc.get("subject"),
                "workstreams": wave_entries,
            }
        }
        usage: dict = {}
        error: BudgetExceededError | None = None
        failure: Exception | None = None
        try:
            usage, error = self._run_agent(
                case, execute_message(case, wave_plan), "execution"
            )
        except Exception as exc:  # noqa: BLE001 — isolation, see docstring
            failure = exc
            self.store.add_audit(
                case_id,
                "workstream_run_failed",
                {
                    "wave": wave_index,
                    "workstreams": [e["question_id"] for e in layer],
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            log.warning(
                "case %s wave %d run failed: %s", case_id, wave_index, exc
            )
        # The workstreams this run stopped inside of, resolved from
        # the wave's own order and the spawns still open when the
        # cap tripped. The composer turns these into the partial
        # report's PARTIAL rows.
        order = [entry["question_id"] for entry in layer]
        interrupted = [
            order[i - 1]
            for i in usage.get("interrupted_spawns", [])
            if 1 <= i <= len(order)
        ]
        interrupted_ids = set(interrupted)
        counts: dict[str, int] = {}
        for finding in self.store.list_findings(case_id):
            counts[finding["question_id"]] = (
                counts.get(finding["question_id"], 0) + 1
            )
        statuses: dict[str, dict] = {}
        for entry in layer:
            qid = entry["question_id"]
            found = counts.get(qid, 0)
            if found > 0:
                # Its output is in the case record, whatever else
                # happened in the wave around it.
                status = {
                    "question_id": qid,
                    "workstream": entry["workstream"],
                    "status": "completed",
                    "findings": found,
                    "reason": None,
                }
            elif failure is not None:
                status = {
                    "question_id": qid,
                    "workstream": entry["workstream"],
                    "status": "failed",
                    "findings": 0,
                    "reason": f"{type(failure).__name__}: {failure}",
                }
            elif error is not None and qid in interrupted_ids:
                status = {
                    "question_id": qid,
                    "workstream": entry["workstream"],
                    "status": "interrupted",
                    "findings": 0,
                    "reason": (
                        f"the run stopped on its {error.kind} budget "
                        "inside this workstream"
                    ),
                }
            elif error is not None:
                status = {
                    "question_id": qid,
                    "workstream": entry["workstream"],
                    "status": "not_run",
                    "findings": 0,
                    "reason": (
                        "the run stopped on its budget before this "
                        "workstream was spawned"
                    ),
                }
            else:
                status = {
                    "question_id": qid,
                    "workstream": entry["workstream"],
                    "status": "empty",
                    "findings": 0,
                    "reason": (
                        "the workstream ran and recorded no findings"
                    ),
                }
            statuses[qid] = status
            self.store.add_audit(
                case_id,
                "workstream_finished",
                {
                    "question_id": qid,
                    "workstream": entry["workstream"],
                    "wave": wave_index,
                    "status": status["status"],
                    "findings": found,
                    "reason": status["reason"],
                },
            )
        return usage, error, statuses, interrupted

    def workstream_statuses(self, case_id: str) -> list[dict]:
        """Per-workstream execution status for a case: the report's
        record once a report exists, else the audit fold (a
        workstream started but not yet finished reads as running)."""
        self._require_case(case_id)
        report = self.store.get_report(case_id)
        if report is not None and report.get("workstreams"):
            return report["workstreams"]
        statuses: dict[str, dict] = {}
        order: list[str] = []
        for entry in self.store.list_audit(case_id):
            payload = entry.get("payload") or {}
            qid = payload.get("question_id")
            if not qid:
                continue
            if entry["event"] == "workstream_started":
                if qid not in statuses:
                    order.append(qid)
                statuses[qid] = {
                    "question_id": qid,
                    "workstream": payload.get("workstream"),
                    "status": "running",
                    "findings": 0,
                    "reason": None,
                }
            elif entry["event"] == "workstream_finished":
                statuses[qid] = {
                    "question_id": qid,
                    "workstream": payload.get("workstream"),
                    "status": payload.get("status"),
                    "findings": payload.get("findings", 0),
                    "reason": payload.get("reason"),
                }
        return [statuses[qid] for qid in order]

    # -- budget amendment -----------------------------------------------------
    def amend_budget(
        self, case_id: str, analyst: str, reason: str, budget: BudgetIn
    ) -> dict:
        """Raise an exhausted case's budget and return it to
        APPROVED for re-execution.

        The amendment is a ledger event, not an edit: the original
        exhaustion stays in the audit trail, and the amendment —
        analyst, reason, the budget before and after — is recorded
        beside it permanently. One refusal is structural: no cap
        may be set below what the case has already consumed (per
        phase for steps and tokens, case-wide for spawns and wall
        clock, all folded from the ledger), because a budget that
        the recorded past already breaks would make the ledger a
        lie. Caps at or above consumption are the analyst's call.
        """
        case = self._require_case(case_id)
        if case["status"] != "BUDGET_EXHAUSTED":
            raise GateError(
                "budget amendment requires a budget-exhausted case "
                f"(status {case['status']})"
            )
        if not reason or not reason.strip():
            raise ValueError("a budget amendment requires a recorded reason")
        amended = budget.model_dump()
        fold = fold_budget_ledger(self.store.list_audit(case_id))
        problems: list[str] = []
        for phase in PHASES:
            caps = resolve_phase_budget(amended, phase)
            consumed = fold["phases"][phase]["consumed"]
            if caps["max_steps"] < consumed["steps"]:
                problems.append(
                    f"{phase} steps {caps['max_steps']} is below the "
                    f"{consumed['steps']} already consumed"
                )
            if caps["max_tokens"] < consumed["tokens"]:
                problems.append(
                    f"{phase} tokens {caps['max_tokens']} is below the "
                    f"{consumed['tokens']} already consumed"
                )
        caps = case_caps(amended)
        consumed = fold["case_consumed"]
        if caps["max_subagents"] < consumed["subagents"]:
            problems.append(
                f"sub-agents {caps['max_subagents']} is below the "
                f"{consumed['subagents']} already spawned case-wide"
            )
        if caps["max_wall_seconds"] < consumed["wall_seconds"]:
            problems.append(
                f"wall clock {caps['max_wall_seconds']}s is below the "
                f"{consumed['wall_seconds']}s already consumed case-wide"
            )
        if problems:
            raise ValueError(
                "an amendment may not set a cap below what the case "
                "has already consumed: " + "; ".join(problems)
            )
        previous = case["brief"]["budget"]
        brief = {**case["brief"], "budget": amended}
        self.store.add_audit(
            case_id,
            "budget_amended",
            {
                "analyst": analyst,
                "reason": reason,
                "previous": previous,
                "amended": amended,
            },
        )
        log.info(
            "case %s budget amended by %s: %s", case_id, analyst, reason
        )
        return self.store.update_case(case_id, brief=brief, status="APPROVED")

    # -- verification ---------------------------------------------------------
    def _verify_draft(self, case: dict, report: dict) -> dict:
        """The independent verifier pass, between composition and the
        analyst gate. Quarantined claims leave the report body here;
        what reaches the analyst is the verified draft, the verifier
        summary, and the quarantine record."""
        case_id = case["case_id"]
        model = None
        model_name = None
        if self.settings.llm_backend != "offline":
            model = build_model(self.settings)
            model_name = {
                "openai": self.settings.openai_model,
                "anthropic": self.settings.anthropic_model,
                "ollama": self.settings.ollama_model,
            }.get(self.settings.llm_backend)
        result = verify_case(
            self.store,
            self.objects,
            case_id,
            model=model,
            model_name=model_name,
        )
        self.store.add_audit(
            case_id,
            "verification_completed",
            {
                "run_id": result["run_id"],
                "checked": result["checked"],
                "supported": result["supported"],
                "quarantined": result["quarantined"],
                "by_reason": result["by_reason"],
            },
        )
        records = self.store.list_quarantine(case_id)
        if records:
            self.store.add_audit(
                case_id,
                "claims_quarantined",
                {
                    "claims": [
                        {"claim_id": r["claim_id"], "reason": r["reason"]}
                        for r in records
                    ]
                },
            )
        report = apply_verification(
            self.store, self.objects, case, report, result, records
        )
        log.info(
            "case %s verified: %d checked, %d supported, %d quarantined",
            case_id,
            result["checked"],
            result["supported"],
            result["quarantined"],
        )
        return self._detect_conflicts(case, report)

    def _detect_conflicts(self, case: dict, report: dict) -> dict:
        """The conflict pass, after verification: key the supported
        claims by (entity, metric, period) and record where they
        assert different values for the same key. Conflicts annotate
        the draft — no claim is removed, merged, or preferred — and
        the records ride into the report's Conflicts section."""
        case_id = case["case_id"]
        records = detect_conflicts(self.store, case, report["claims"])
        self.store.replace_conflicts(case_id, records)
        if records:
            self.store.add_audit(
                case_id,
                "conflicts_detected",
                {
                    "count": len(records),
                    "conflicts": [
                        {
                            "conflict_id": r["conflict_id"],
                            "metric": r["key"]["metric"],
                            "period": r["key"]["period"],
                            "values": [
                                side["value_text"] for side in r["sides"]
                            ],
                        }
                        for r in records
                    ],
                },
            )
            log.info(
                "case %s conflicts detected: %d", case_id, len(records)
            )
        return apply_conflicts(
            self.store, self.objects, case, report, records
        )

    def decide_conflict(
        self,
        case_id: str,
        conflict_id: str,
        analyst: str,
        note: str = "",
    ) -> dict:
        """The analyst's only move on a conflict: mark it NOTED,
        with a note.

        There is deliberately no resolve or delete move: a conflict
        is a fact about the evidence, not a task to clear. Marking
        NOTED records that the analyst has seen the disagreement and
        what they made of it; the record — and the report's
        Conflicts section, re-rendered with the note — stays
        permanently.
        """
        case = self._require_case(case_id)
        if case["status"] not in ("DRAFT_READY", "BUDGET_EXHAUSTED"):
            raise GateError(
                f"no report at the gate to note a conflict on "
                f"(status {case['status']})"
            )
        if not note.strip():
            raise ValueError("marking a conflict NOTED requires a note")
        record = self.store.get_conflict(case_id, conflict_id)
        if record is None:
            raise KeyError(conflict_id)
        if record["status"] != "OPEN":
            raise GateError(
                f"conflict {conflict_id} is already {record['status']}"
            )
        record["status"] = "NOTED"
        record["noted_by"] = analyst
        record["note"] = note
        record["noted_at"] = utcnow()
        self.store.update_conflict(case_id, record)
        self.store.add_audit(
            case_id,
            "conflict_noted",
            {
                "conflict_id": conflict_id,
                "metric": record["key"]["metric"],
                "period": record["key"]["period"],
                "analyst": analyst,
                "note": note,
            },
        )
        report = self.store.get_report(case_id)
        if report is not None:
            apply_conflicts(
                self.store,
                self.objects,
                case,
                report,
                self.store.list_conflicts(case_id),
            )
        return record

    def decide_quarantine(
        self,
        case_id: str,
        quarantine_id: str,
        analyst: str,
        action: str,
        note: str = "",
    ) -> dict:
        """An analyst's two moves on a quarantined claim — and only two.

        There is deliberately no "approve into the report" move: a
        claim the verifier quarantined can be sent back for
        re-investigation (the case returns to APPROVED, ready to
        execute again, with the analyst's note in the audit trail) or
        dropped. Either way the underlying finding leaves the case
        record, so the claim cannot quietly re-enter a later draft —
        a re-investigation has to re-derive it from the evidence.
        """
        case = self._require_case(case_id)
        if case["status"] not in ("DRAFT_READY", "BUDGET_EXHAUSTED"):
            raise GateError(
                f"no report at the gate to decide quarantine from "
                f"(status {case['status']})"
            )
        if action not in ("reinvestigate", "drop"):
            raise ValueError(
                f"unknown quarantine action {action!r}; "
                "expected 'reinvestigate' or 'drop'"
            )
        record = self.store.get_quarantine(case_id, quarantine_id)
        if record is None:
            raise KeyError(quarantine_id)
        if record["status"] != "QUARANTINED":
            raise GateError(
                f"quarantine record {quarantine_id} is already "
                f"{record['status']}"
            )
        record["status"] = "SENT_BACK" if action == "reinvestigate" else "DROPPED"
        record["decided_by"] = analyst
        record["decision_note"] = note
        record["decided_at"] = utcnow()
        self.store.update_quarantine(case_id, record)
        self.store.delete_finding(case_id, record["finding_id"])
        self.store.add_audit(
            case_id,
            "claim_sent_back" if action == "reinvestigate" else "claim_dropped",
            {
                "quarantine_id": quarantine_id,
                "claim_id": record["claim_id"],
                "finding_id": record["finding_id"],
                "reason": record["reason"],
                "analyst": analyst,
                "note": note,
            },
        )
        if action == "reinvestigate":
            self.store.update_case(case_id, status="APPROVED")
        return record

    def decide_report(self, case_id: str, analyst: str, approve: bool, note: str = "") -> dict:
        case = self._require_case(case_id)
        if case["status"] not in ("DRAFT_READY", "BUDGET_EXHAUSTED"):
            raise GateError(f"no report awaiting decision (status {case['status']})")
        report = self.store.get_report(case_id)
        if approve:
            # Citation integrity at the gate: re-verify the ledger
            # against the evidence store before this report may become
            # final. If any cited item's bytes no longer hash to the
            # value recorded at citation time, the evidence changed
            # after the fact — flag the report EVIDENCE_DRIFT and
            # refuse the approval until a re-check comes back clean.
            check = verify_ledger(self.store, self.objects, case_id)
            if check["status"] != "OK":
                report["integrity"] = {
                    "status": "EVIDENCE_DRIFT",
                    "drift": check["drift"],
                    "checked_at": check["verified_at"],
                }
                self.store.save_report(case_id, report)
                self.store.add_audit(
                    case_id, "evidence_drift_detected", {"drift": check["drift"]}
                )
                drifted = ", ".join(
                    sorted({d["evidence_id"] for d in check["drift"]})
                )
                raise GateError(
                    "report is flagged EVIDENCE_DRIFT: cited evidence "
                    f"changed after citation ({drifted}); it cannot be "
                    "approved until the ledger re-verifies clean"
                )
            report["integrity"] = {
                "status": "OK",
                "checked_at": check["verified_at"],
            }
        if approve and report.get("coverage_detail"):
            # Approving a partial report is the analyst's call — and
            # the approval record carries the coverage table it
            # approved, so a later reader can see exactly which
            # questions were ANSWERED, PARTIAL, or NOT COVERED at
            # the moment this draft became final.
            report["approved_coverage"] = report["coverage_detail"]
        report["status"] = "APPROVED" if approve else "REJECTED"
        report["decided_by"] = analyst
        report["decision_note"] = note
        self.store.save_report(case_id, report)
        audit_payload: dict = {"analyst": analyst, "note": note}
        if approve and report.get("coverage_detail"):
            audit_payload["partial"] = True
            audit_payload["coverage"] = report["coverage_detail"]
        self.store.add_audit(
            case_id,
            "report_approved" if approve else "report_rejected",
            audit_payload,
        )
        return self.store.update_case(
            case_id, status="FINAL" if approve else "REPORT_REJECTED"
        )
