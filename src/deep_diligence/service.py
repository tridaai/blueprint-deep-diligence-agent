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

import html
import json
import logging
import re
import uuid
from datetime import datetime
from decimal import Decimal

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
from .challenges import sync_challenges
from .composer import apply_conflicts, apply_verification, compose_report
from .conflicts import detect_conflicts
from .config import Settings
from .costs import cost_view, fold_cost_ledger, load_price_table, price_usage_records
from .continuity import stale_questions
from .corpus import companies
from .evidence import add_evidence_document, ingest_case_evidence
from .governance import (
    SOURCE_KINDS,
    USAGE_BASES,
    brief_source_problems,
    case_provenance,
    case_source_policy,
    parse_ts,
    redaction_marker,
    retention_due,
    seed_sources,
)
from .harness import build_agent, execute_message, plan_message
from .deployment import check_live_permission, deployment_record
from .ledger import verify_ledger
from .llm import build_model, model_label
from .models import PRIORITIES, BudgetIn, BriefIn
from .objects import ObjectStore, build_object_store, sha256_hex
from .observability import build_trace, classify_case_outcome, fold_digest
from .portfolio import build_portfolio, member_cases, summarize_portfolio
from .profiles import ProfileError, load_profile
from .promotion import ProfileGate
from .report_model import (
    DRAFT_STATUS_LINE,
    export_report,
    final_status_line,
    model_sections,
)
from .schedule import (
    build_handoff,
    plan_defects,
    plan_shape_defects,
    waves,
    workstream_dependencies,
)
from .skills import (
    SkillError,
    plan_skill_pins,
    resolve_plan_skills,
    seed_skills,
    skill_mount_files,
    validate_skill,
)
from .store import Store, build_store, utcnow
from .tools import build_case_tools
from .verifier import verify_case
from .workbench import build_queue, build_timeline, validate_filters
from .workbench import search_cases as _search_cases

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
        "spawned_workstreams": [
            qid for u in usages for qid in u.get("spawned_workstreams", [])
        ],
    }
    return total


def _validate_digest_day(day: str) -> str:
    """A digest is keyed by the day it summarises: an ISO
    calendar date, exactly YYYY-MM-DD, that actually exists. A
    digest keyed by a date nobody can read back — or a string
    that merely looks like one — is refused here, at generation
    and at read, in the service's own words."""
    try:
        parsed = datetime.strptime(day, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise ValueError(
            f"digest day {day!r} is not an ISO calendar date "
            "(YYYY-MM-DD); a digest is keyed by the day it "
            "summarises"
        ) from None
    if parsed.strftime("%Y-%m-%d") != day:
        raise ValueError(
            f"digest day {day!r} is not an ISO calendar date "
            "(YYYY-MM-DD); a digest is keyed by the day it "
            "summarises"
        )
    return day


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
        # The governed price table loads with the same fail-closed
        # discipline as the harness profile: a table that cannot
        # load refuses the startup, because a run priced against a
        # silent fallback would be a cost ledger nobody governed.
        self.price_table = load_price_table(settings.pricing_dir)
        # The source registry is seeded from the corpus before any
        # case work: evidence ingestion refuses unregistered
        # sources, and the corpus is the desk's standing universe,
        # so its sources are registered up front (idempotently —
        # analyst-registered records are never overwritten).
        seed_sources(self.store)
        self.profile = self._resolve_active_profile()
        # Deployment profile (R19, deployment.py): the composition
        # was validated at settings load; the one rule settings
        # cannot judge — under a hosted profile, live models are
        # permitted only through a promoted harness profile — is
        # checked here, against the promotion log the active
        # profile was just resolved from. It refuses the boot the
        # same way, before any case work.
        check_live_permission(
            settings,
            profile_promoted=self.store.latest_promotion() is not None,
        )
        # The skills library: the shipped artifacts are seeded
        # into the store — the runtime library of record — before
        # any case work. Seeding validates structurally and is
        # fail-closed (a malformed artifact refuses the startup);
        # the profile binding is checked where a skill is loaded
        # for a role: at publish and at the plan gate.
        self.skill_seeds = seed_skills(self.store, settings.skills_dir)
        self.gate = ProfileGate(self.store, settings.profile_dir)
        self.objects: ObjectStore = build_object_store(settings)
        log.info(
            "harness profile %s v%d (target %s); skills library holds "
            "%d version(s)",
            self.profile.id,
            self.profile.version,
            self.profile.target_model,
            len(self.store.list_skills()),
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

    def _deployment_stamp(self) -> dict:
        """The deployment record for work done now: the named
        composition this process booted under, with the harness
        profile as it stands at this moment (a promotion moves
        the pointer for later work, exactly as the profile stamp
        already behaves). Stamped on the case at intake and on
        every run's grant entry."""
        return deployment_record(self.settings, self.profile.id)

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

    # -- skills library -----------------------------------------------------
    def list_skills(self) -> list[dict]:
        return self.store.list_skills()

    def publish_skill(self, analyst: str, artifact: dict) -> dict:
        """Publishing a skill version is a recorded act — and the
        record is itself the gate. The artifact is validated
        fail-closed against the active profile BEFORE anything is
        stored, so a version that is malformed, or whose method
        names tools beyond a role's allowlist under this profile,
        can never enter the library. Versions extend a name's line
        by exactly one: re-publishing an existing version, or
        skipping ahead of the line, is refused — a published
        version is never edited and never forked."""
        skill = validate_skill(artifact, "publish request", self.profile)
        latest = self.store.latest_skill(skill.name)
        expected = latest["version"] + 1 if latest else 1
        if skill.version != expected:
            raise SkillError(
                f"skill {skill.name!r}: version {skill.version} cannot "
                f"be published — the library's line for this name is "
                f"at version {latest['version'] if latest else 0}, so "
                f"the next publishable version is {expected}"
            )
        record = {
            **skill.artifact(),
            "origin": "published",
            "published_by": analyst,
            "published_at": utcnow(),
        }
        return self.store.add_skill(record)

    # -- helpers ------------------------------------------------------------
    def _require_case(self, case_id: str) -> dict:
        case = self.store.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        return case

    def _backend(self, case_id: str) -> EvidenceBackend:
        # The case's filesystem mounts the skills its approved
        # plan pins, read-only at /skills/<name>.md (backend.py),
        # resolved from the stored plan on every build so a run
        # always sees exactly the pins in force for the case.
        return EvidenceBackend(
            self.store,
            self.objects,
            case_id,
            skill_files=skill_mount_files(self.store, case_id),
        )

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
        tools = build_case_tools(
            self.store,
            self.objects,
            case,
            skills_by_question=plan_skill_pins(self.store, case_id),
        )
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
                # And the deployment it ran under (R19): the named
                # composition — deployment profile, harness profile,
                # store backend — so the trace can say which
                # deployment produced this run.
                "deployment": self._deployment_stamp(),
            },
        )
        sink: list = []
        guard = BudgetGuard(
            {**phase_caps, **caps},
            audit_sink=sink,
            phase=phase,
            prior_subagents=consumed["subagents"],
            prior_wall_seconds=consumed["wall_seconds"],
            # Cost attribution: the model this run is priced
            # under, and — on the offline backend — the flag that
            # marks its usage figures as the stub's declared
            # per-call numbers (estimates), never as measured
            # provider usage.
            model=model_label(self.settings),
            usage_declared=self.settings.llm_backend == "offline",
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
        # The run's cost, priced now — at the end of the run, under
        # the price table as it stands today — and recorded as a
        # ledger entry beside the consumption checkpoint. Each
        # priced line carries the table version and rates it was
        # computed with, so the recorded ledger stands whatever
        # the table later becomes (costs.fold_cost_ledger reads
        # only these entries, never the current table).
        cost_lines = price_usage_records(
            usage.get("model_records") or [], self.price_table
        )
        if cost_lines:
            self.store.add_audit(
                case_id,
                "cost_recorded",
                {
                    "phase": phase,
                    "price_table": self.price_table.record(),
                    "lines": cost_lines,
                },
            )
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
        # Brief source policy, fail closed: every allowed_sources
        # entry must resolve — a registered source, a corpus
        # document, or a document kind. An entry that names
        # nothing would silently scope the case to a subset (or
        # to nothing); the brief is refused, with the entry named.
        problems = brief_source_problems(self.store, brief.model_dump())
        if problems:
            raise ValueError("; ".join(problems))
        case_id = f"case-{uuid.uuid4().hex[:10]}"
        now = utcnow()
        case = {
            "case_id": case_id,
            "status": "INTAKE",
            "subject": brief.subject,
            "subject_ref": brief.subject_ref,
            "analyst": brief.analyst,
            # Analyst workbench triage (R12): the desk's current
            # owner of the case, and how hard it should pull
            # attention. Both start unset/default; setting either
            # is a recorded act, and neither gates anything.
            "assignee": None,
            "priority": "normal",
            # Portfolio mode (R15): the engagement (client
            # mandate) this case runs under, or None. Starts
            # unset on every case; assignment is a recorded act,
            # and membership changes nothing inside the case.
            "engagement_id": None,
            "brief": brief.model_dump(),
            "usage": {},
            # Long-case continuity (R9): workstreams that evidence
            # arriving mid-case has marked stale and not yet had an
            # analyst decision, and the append-only record of those
            # decisions. Both start empty on every case.
            "stale_marks": [],
            "staleness_decisions": [],
            # Source governance (R10): the retention class and the
            # record of hold acts on the case. Every case starts on
            # standard retention; a legal hold is an analyst act,
            # recorded here and in the audit trail.
            "retention_class": "standard",
            "retention_history": [],
            # The profile stamp: this case's planning and execution
            # run under this profile (id + version), and the case
            # view shows it.
            "profile": self.profile.record(),
            # The deployment stamp (R19): the named composition the
            # case was born under — deployment profile, harness
            # profile, store backend. Cases created before
            # deployment profiles existed carry no stamp and read
            # as deployment-unknown, never backfilled.
            "deployment": self._deployment_stamp(),
            "created_at": now,
            "updated_at": now,
        }
        self.store.create_case(case)
        evidence = ingest_case_evidence(self.store, self.objects, case)
        self.store.add_audit(
            case_id,
            "case_created",
            {
                "subject": brief.subject,
                "profile": self.profile.record(),
                "deployment": self._deployment_stamp(),
            },
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
        # The desk's memory travels with the planning instruction:
        # on a re-plan the analyst notes are how what the desk
        # learned earlier reaches the new plan (bounded and
        # labelled in harness.notes_section; never evidence).
        notes = self.store.list_notes(case_id)
        usage, error = self._run_agent(
            case, plan_message(case, notes), "planning"
        )
        case = self.store.update_case(
            case_id, usage={**case.get("usage", {}), "planning": usage}
        )
        if error is not None:
            self.store.add_audit(case_id, "planning_budget_exceeded", usage)
            raise GateError(
                f"planning stopped on the {error.kind} budget; amend the "
                "budget to re-plan"
            )
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

    def _plan_gate_defects(self, case: dict, plan: dict) -> list[str]:
        """Every reason this plan cannot be scheduled, named: its
        shape against the brief and the active profile's roster
        first, then — on a well-formed plan — its dependency
        graph."""
        plan_doc = plan.get("plan")
        defects = plan_shape_defects(
            plan_doc,
            case["brief"]["questions"],
            [entry.name for entry in self.profile.subagents],
        )
        if defects:
            return defects
        return plan_defects(plan_doc["workstreams"])

    def decide_plan(self, case_id: str, analyst: str, approve: bool, note: str = "") -> dict:
        case = self._require_case(case_id)
        if case["status"] != "PLAN_READY":
            raise GateError(f"no plan awaiting decision (status {case['status']})")
        plan = self.store.get_plan(case_id)
        pins: dict[str, list[dict]] = {}
        if approve:
            # The plan gate checks the plan's shape and dependency
            # graph in code (schedule.py): a malformed plan, an
            # unknown dependency, or a cycle means the plan cannot
            # be scheduled, so it cannot be approved either. The
            # defect is named, and the plan stays at the gate for
            # the analyst to send back.
            defects = self._plan_gate_defects(case, plan)
            if defects:
                raise GateError(
                    "plan cannot be approved: " + "; ".join(defects)
                )
            workstreams = plan["plan"]["workstreams"]
            # The skill gate rides the same approval (skills.py):
            # every skill the plan names must be in the library,
            # must apply to the workstream's role, and must stay
            # inside the role's tool allowlist under the active
            # profile. Approval PINS the version each skill runs
            # under — the pin is stamped on the plan artifact here,
            # so execution, re-execution, and resume all run under
            # the version the analyst approved, even if a newer
            # version is published while the case is in flight.
            pins, problems = resolve_plan_skills(
                self.store, self.profile, workstreams
            )
            if problems:
                raise GateError(
                    "plan cannot be approved: " + "; ".join(problems)
                )
            for entry in workstreams:
                if entry["question_id"] in pins:
                    entry["skills"] = pins[entry["question_id"]]
        plan["status"] = "APPROVED" if approve else "REJECTED"
        plan["decided_by"] = analyst
        plan["decision_note"] = note
        self.store.update_plan(case_id, plan)
        audit_payload: dict = {
            "version": plan["version"],
            "analyst": analyst,
            "note": note,
        }
        if pins:
            audit_payload["skills"] = pins
        self.store.add_audit(
            case_id,
            "plan_approved" if approve else "plan_rejected",
            audit_payload,
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
        stale = case.get("stale_marks") or []
        if stale:
            questions = ", ".join(
                sorted({m["question_id"] for m in stale})
            )
            raise GateError(
                "new evidence arrived after planning and these "
                f"workstreams are marked stale: {questions}. An "
                "analyst must re-plan or record a decision to "
                "proceed before execution"
            )
        plan = self.store.get_plan(case_id)
        # Defence in depth: the plan gate validated this plan at
        # approval, and an unschedulable plan must still never run.
        defects = self._plan_gate_defects(case, plan)
        if defects:
            raise GateError(
                "plan cannot be executed: " + "; ".join(defects)
            )
        workstreams = plan["plan"]["workstreams"]
        skip = self._reuse_skip(case, workstreams)
        backend = self._backend(case_id)
        backend.delete_direct("/execution/notes.md")
        return self._execute(case, plan, skip=skip, resumed=False)

    def _execute(
        self,
        case: dict,
        plan: dict,
        *,
        skip: dict,
        resumed: bool,
        attempt: int | None = None,
    ) -> dict:
        """Run an approved plan's waves to a composed draft.

        `skip` holds workstreams that must not run — their recorded
        statuses are carried into the execution record instead.
        Two callers build it: a fresh execution after a
        staleness-driven re-plan (findings reuse, see _reuse_skip)
        and a resume (the checkpoint's completed workstreams, see
        resume_execution). An ordinary execution passes an empty
        skip and behaves exactly as execution always has.

        The execution writes its own checkpoint as it goes — one
        record of per-workstream status, finding IDs, wave
        position, and a budget snapshot folded from the ledger —
        so if the process dies mid-run, what remains is a case in
        EXECUTING with a checkpoint that says precisely where it
        stopped, and resume reads that record, never harness
        memory.
        """
        case_id = case["case_id"]
        plan_doc = plan.get("plan") or {}
        workstreams = plan_doc.get("workstreams", [])
        schedule = waves(workstreams)
        self.store.update_case(case_id, status="EXECUTING")
        started: dict = {
            "plan_version": plan["version"],
            "waves": [
                [ws["question_id"] for ws in layer] for layer in schedule
            ],
        }
        # The skills this execution runs under, by workstream —
        # the pins on the approved plan, recorded so the trail
        # attributes the run's findings to the method versions in
        # force, not just to a date.
        skill_pins = {
            entry["question_id"]: entry["skills"]
            for entry in workstreams
            if entry.get("skills")
        }
        if skill_pins:
            started["skills"] = skill_pins
        if resumed:
            started["resumed"] = True
        if skip:
            started["skipped"] = sorted(skip)
        self.store.add_audit(case_id, "execution_started", started)
        prior = self.store.get_checkpoint(case_id)
        if attempt is None:
            attempt = (prior or {}).get("attempt", 0) + 1
        checkpoint = self._open_checkpoint(
            case, plan, schedule, skip, attempt, prior
        )
        statuses: dict[str, dict] = dict(skip)
        usages: list[dict] = []
        error: BudgetExceededError | None = None
        interrupted: list[str] = []
        tripped_usage: dict | None = None
        for index, layer in enumerate(schedule, start=1):
            runnable = [
                ws for ws in layer if ws["question_id"] not in statuses
            ]
            if runnable:
                usage, error, layer_statuses, layer_interrupted = (
                    self._run_wave(case, plan_doc, runnable, index)
                )
                usages.append(usage)
                statuses.update(layer_statuses)
                if error is not None:
                    interrupted = layer_interrupted
                    tripped_usage = usage
            self._refresh_checkpoint(case_id, checkpoint, statuses)
            if error is not None:
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
            case_id,
            usage={**case.get("usage", {}), "execution": usage},
            # The reuse directive is one-shot: whichever execution
            # it governed has now run, and the next fresh execution
            # starts from the record as it stands then.
            reuse_findings=False,
            must_rerun=[],
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
            self._close_checkpoint(
                case_id, checkpoint, statuses, "budget_exhausted"
            )
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
        self._close_checkpoint(case_id, checkpoint, statuses, "draft_ready")
        log.info("case %s draft ready", case_id)
        return report

    # -- checkpoints ------------------------------------------------------------
    def _budget_snapshot(self, case_id: str) -> dict:
        """The case's budget position, folded from the ledger at
        this moment — the snapshot a checkpoint carries, so a
        resume can show exactly what the interrupted attempt had
        already spent."""
        fold = fold_budget_ledger(self.store.list_audit(case_id))
        return {
            "case_consumed": fold["case_consumed"],
            "phases": {
                phase: fold["phases"][phase]["consumed"] for phase in PHASES
            },
        }

    def _open_checkpoint(
        self,
        case: dict,
        plan: dict,
        schedule: list,
        skip: dict,
        attempt: int,
        prior: dict | None,
    ) -> dict:
        case_id = case["case_id"]
        recorded: dict[str, list[str]] = {}
        for finding in self.store.list_findings(case_id):
            recorded.setdefault(finding["question_id"], []).append(
                finding["finding_id"]
            )
        workstream_state: dict[str, dict] = {}
        for index, layer in enumerate(schedule, start=1):
            for entry in layer:
                qid = entry["question_id"]
                if qid in skip:
                    workstream_state[qid] = {
                        "status": skip[qid]["status"],
                        "findings": skip[qid]["findings"],
                        "finding_ids": recorded.get(qid, []),
                        "wave": index,
                        "reason": skip[qid].get("reason"),
                    }
                else:
                    workstream_state[qid] = {
                        "status": "pending",
                        "findings": len(recorded.get(qid, [])),
                        "finding_ids": recorded.get(qid, []),
                        "wave": index,
                        "reason": None,
                    }
        started_at = utcnow()
        if prior and prior.get("attempt") == attempt and prior.get("started_at"):
            # A resume continues the same attempt; its clock
            # started when the attempt did, not when it resumed.
            started_at = prior["started_at"]
        checkpoint = {
            "case_id": case_id,
            "plan_version": plan["version"],
            "attempt": attempt,
            "waves": [
                [ws["question_id"] for ws in layer] for layer in schedule
            ],
            "workstreams": workstream_state,
            "budget_snapshot": self._budget_snapshot(case_id),
            "complete": False,
            "outcome": None,
            "started_at": started_at,
            "updated_at": utcnow(),
        }
        return self.store.save_checkpoint(case_id, checkpoint)

    def _refresh_checkpoint(
        self, case_id: str, checkpoint: dict, statuses: dict
    ) -> dict:
        recorded: dict[str, list[str]] = {}
        for finding in self.store.list_findings(case_id):
            recorded.setdefault(finding["question_id"], []).append(
                finding["finding_id"]
            )
        for qid, status in statuses.items():
            entry = checkpoint["workstreams"].setdefault(qid, {"wave": None})
            entry.update(
                {
                    "status": status["status"],
                    "findings": status["findings"],
                    "finding_ids": recorded.get(qid, []),
                    "reason": status.get("reason"),
                }
            )
        checkpoint["budget_snapshot"] = self._budget_snapshot(case_id)
        checkpoint["updated_at"] = utcnow()
        return self.store.save_checkpoint(case_id, checkpoint)

    def _close_checkpoint(
        self, case_id: str, checkpoint: dict, statuses: dict, outcome: str
    ) -> dict:
        checkpoint = self._refresh_checkpoint(case_id, checkpoint, statuses)
        checkpoint["complete"] = True
        checkpoint["outcome"] = outcome
        checkpoint["updated_at"] = utcnow()
        return self.store.save_checkpoint(case_id, checkpoint)

    def _reuse_skip(self, case: dict, workstreams: list[dict]) -> dict:
        """The findings-reuse rule for a staleness-driven re-plan.

        The reuse directive is set only by replan_case, and only
        when staleness marks named the workstreams the new
        evidence touches. A workstream is skipped — its recorded
        findings carried into the new execution — when ALL of
        these hold: it is not one of the marked questions, it
        already holds findings, and every item of evidence those
        findings cite is still in the case with its bytes hashing
        to the value recorded when the evidence arrived (drifted
        evidence re-runs its workstream; nothing is reused on
        faith). A general plan revision sets no directive and
        re-executes in full, exactly as re-execution always has.
        """
        if not case.get("reuse_findings"):
            return {}
        case_id = case["case_id"]
        must_rerun = set(case.get("must_rerun") or [])
        by_question: dict[str, list[dict]] = {}
        for finding in self.store.list_findings(case_id):
            by_question.setdefault(finding["question_id"], []).append(finding)
        evidence = {
            ev["evidence_id"]: ev for ev in self.store.list_evidence(case_id)
        }
        skip: dict[str, dict] = {}
        for entry in workstreams:
            qid = entry["question_id"]
            found = by_question.get(qid) or []
            if qid in must_rerun or not found:
                continue
            unchanged = all(
                eid in evidence
                and sha256_hex(
                    self.objects.get(evidence[eid]["object_key"])
                )
                == evidence[eid]["sha256"]
                for finding in found
                for eid in finding["evidence_ids"]
            )
            if not unchanged:
                continue
            skip[qid] = {
                "question_id": qid,
                "workstream": entry["workstream"],
                "status": "completed",
                "findings": len(found),
                "reason": (
                    "findings reused from the previous plan — the "
                    "new evidence does not touch this workstream, "
                    "and the evidence its findings cite is unchanged"
                ),
                "reused": True,
            }
        if skip:
            self.store.add_audit(
                case_id,
                "findings_reused",
                {
                    "questions": sorted(skip),
                    "must_rerun": sorted(must_rerun),
                },
            )
        return skip

    # -- interrupted cases: detection + resume ------------------------------------
    def interrupted_cases(self) -> list[dict]:
        """Cases an interrupted execution left behind: status
        EXECUTING, checkpoint incomplete, no live run anywhere.

        Detection is recorded — the first scan that finds a case
        writes `execution_interruption_detected` to its audit
        trail, naming the workstreams that had completed and the
        ones still owed; later scans of the same interruption read
        the record back without writing it again.
        """
        found: list[dict] = []
        for case in self.store.list_cases():
            if case["status"] != "EXECUTING":
                continue
            case_id = case["case_id"]
            checkpoint = self.store.get_checkpoint(case_id)
            if checkpoint is not None and checkpoint.get("complete"):
                continue
            plan = self.store.get_plan(case_id)
            if plan and (plan.get("plan") or {}).get("workstreams"):
                all_questions = [
                    ws["question_id"]
                    for ws in plan["plan"]["workstreams"]
                ]
            else:
                all_questions = [
                    q["id"] for q in case["brief"]["questions"]
                ]
            state = (checkpoint or {}).get("workstreams") or {}
            completed = [
                qid
                for qid in all_questions
                if (state.get(qid) or {}).get("status")
                in ("completed", "empty")
            ]
            pending = [qid for qid in all_questions if qid not in completed]
            audit = self.store.list_audit(case_id)
            last_start = max(
                (
                    i
                    for i, entry in enumerate(audit)
                    if entry["event"] == "execution_started"
                ),
                default=-1,
            )
            already = any(
                entry["event"] == "execution_interruption_detected"
                for entry in audit[last_start + 1:]
            )
            if not already:
                self.store.add_audit(
                    case_id,
                    "execution_interruption_detected",
                    {
                        "plan_version": (plan or {}).get("version"),
                        "completed_questions": completed,
                        "pending_questions": pending,
                    },
                )
            found.append(
                {
                    "case": case,
                    "checkpoint": checkpoint,
                    "completed_questions": completed,
                    "pending_questions": pending,
                }
            )
        return found

    def resume_execution(self, case_id: str, analyst: str) -> dict:
        """Resume an interrupted case from its checkpoint.

        Workstreams the checkpoint records as finished are not
        re-run — their findings stand, under the IDs they were
        recorded with. Only the incomplete waves execute. Budget
        consumption carries over by construction: every run's
        guard seeds its case-level caps from the ledger, which
        already holds the interrupted attempt's checkpoints, so
        a resume cannot re-spend spawns the dead run used.
        """
        case = self._require_case(case_id)
        if case["status"] != "EXECUTING":
            raise GateError(
                "only an interrupted case can be resumed "
                f"(status {case['status']})"
            )
        plan = self.store.get_plan(case_id)
        if plan is None:
            raise GateError("the interrupted case has no plan to resume")
        defects = self._plan_gate_defects(case, plan)
        if defects:
            raise GateError(
                "plan cannot be executed: " + "; ".join(defects)
            )
        checkpoint = self.store.get_checkpoint(case_id)
        skip: dict[str, dict] = {}
        for entry in (plan.get("plan") or {}).get("workstreams", []):
            qid = entry["question_id"]
            saved = ((checkpoint or {}).get("workstreams") or {}).get(qid) or {}
            if saved.get("status") in ("completed", "empty"):
                skip[qid] = {
                    "question_id": qid,
                    "workstream": entry["workstream"],
                    "status": saved["status"],
                    "findings": saved.get("findings", 0),
                    "reason": saved.get("reason"),
                    "resumed": True,
                }
        self.store.add_audit(
            case_id,
            "execution_resumed",
            {
                "analyst": analyst,
                "skipped_questions": sorted(skip),
                "budget_carried": (checkpoint or {}).get("budget_snapshot"),
                "checkpoint_attempt": (checkpoint or {}).get("attempt"),
            },
        )
        log.info(
            "case %s execution resumed by %s; %d workstream(s) carried",
            case_id,
            analyst,
            len(skip),
        )
        return self._execute(
            case,
            plan,
            skip=skip,
            resumed=True,
            attempt=(checkpoint or {}).get("attempt") or 1,
        )

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
        # Which workstreams the run actually spawned, from the guard's
        # record. "Ran and found nothing" is only a claim the record
        # can make when every spawn is attributed to a workstream; a
        # spawn it cannot attribute leaves the label at empty, never
        # guessing which workstream went unspawned.
        spawned = usage.get("spawned_workstreams") or []
        spawned_ids = set(spawned)
        spawns_attributed = usage.get("subagents", 0) <= len(spawned)
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
            elif spawns_attributed and qid not in spawned_ids:
                status = {
                    "question_id": qid,
                    "workstream": entry["workstream"],
                    "status": "not_run",
                    "findings": 0,
                    "reason": (
                        "the run ended without spawning this "
                        "workstream's sub-agent"
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

    # -- evidence arrival + staleness ---------------------------------------------
    def add_case_evidence(
        self,
        case_id: str,
        analyst: str,
        *,
        title: str,
        kind: str,
        source: str,
        doc_date: str,
        text: str,
    ) -> dict:
        """A document arrives mid-case and becomes case evidence.

        Ingestion follows the same rules as the intake corpus —
        bytes to the object store, metadata and SHA-256 to the
        case store, an ID from the case's sequence. When the case
        already has a plan, the shell keys the document against
        the plan's workstreams (continuity.stale_questions) and
        marks the ones it speaks to STALE on the case record:
        nothing re-runs and nothing is silently treated as though
        it had always been there. Execution then refuses until an
        analyst decides — re-plan, or proceed with the staleness
        recorded. A FINAL case is closed: new evidence means a new
        case.
        """
        case = self._require_case(case_id)
        if case["status"] == "FINAL":
            raise GateError(
                "the case is FINAL; new evidence means a new case"
            )
        if not text or not text.strip():
            raise ValueError("an evidence document needs text")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
        slug = slug or "analyst-upload"
        taken = {ev["doc_slug"] for ev in self.store.list_evidence(case_id)}
        base, n = slug, 2
        while slug in taken:
            slug = f"{base}-{n}"
            n += 1
        ev = add_evidence_document(
            self.store,
            self.objects,
            case,
            slug=slug,
            title=title,
            kind=kind,
            source=source,
            doc_date=doc_date,
            text=text,
        )
        self.store.add_audit(
            case_id,
            "evidence_added",
            {
                "evidence_id": ev["evidence_id"],
                "slug": slug,
                "sha256": ev["sha256"],
                "analyst": analyst,
            },
        )
        stale: list[dict] = []
        if case["status"] != "INTAKE":
            plan = self.store.get_plan(case_id)
            stale = stale_questions(case, plan, text)
            if stale:
                marks = list(case.get("stale_marks") or [])
                known = {
                    (m["evidence_id"], m["question_id"]) for m in marks
                }
                for entry in stale:
                    key = (ev["evidence_id"], entry["question_id"])
                    if key in known:
                        continue
                    marks.append(
                        {
                            "evidence_id": ev["evidence_id"],
                            "question_id": entry["question_id"],
                            "workstream": entry["workstream"],
                            "score": entry["score"],
                            "marked_at": utcnow(),
                        }
                    )
                case = self.store.update_case(case_id, stale_marks=marks)
                self.store.add_audit(
                    case_id,
                    "workstreams_marked_stale",
                    {
                        "evidence_id": ev["evidence_id"],
                        "questions": [s["question_id"] for s in stale],
                    },
                )
        log.info(
            "case %s evidence %s added by %s; %d workstream(s) stale",
            case_id,
            ev["evidence_id"],
            analyst,
            len(stale),
        )
        return {"evidence": ev, "stale_questions": [s["question_id"] for s in stale], "case": case}

    def decide_staleness(
        self, case_id: str, analyst: str, action: str, note: str = ""
    ) -> dict:
        """The analyst's answer to marked staleness: RE-PLAN, or
        PROCEED with the staleness on the record.

        Either way the decision itself is recorded on the case —
        who, when, which evidence, which questions — and a proceed
        decision rides into the next report's limits (and re-renders
        into a draft that already exists), so a reader of the
        report can see that part of it predates evidence the desk
        chose not to re-run against.
        """
        case = self._require_case(case_id)
        if action not in ("proceed", "replan"):
            raise ValueError(
                f"unknown staleness decision {action!r}; "
                "expected 'proceed' or 'replan'"
            )
        marks = case.get("stale_marks") or []
        if not marks:
            raise GateError(
                "no unresolved staleness on this case to decide"
            )
        decision = {
            "action": action,
            "analyst": analyst,
            "note": note,
            "at": utcnow(),
            "evidence_ids": sorted({m["evidence_id"] for m in marks}),
            "questions": sorted({m["question_id"] for m in marks}),
        }
        if action == "replan":
            return self.replan_case(
                case_id, analyst, note=note, staleness_decision=decision
            )
        decisions = list(case.get("staleness_decisions") or [])
        decisions.append(decision)
        case = self.store.update_case(
            case_id, stale_marks=[], staleness_decisions=decisions
        )
        self.store.add_audit(case_id, "staleness_proceeded", decision)
        report = self.store.get_report(case_id)
        if report is not None:
            # The decision belongs in the draft the analyst is
            # looking at, not only in the next one: re-render it
            # (the conflicts pass re-renders the markdown from the
            # record, limits included).
            apply_conflicts(
                self.store,
                self.objects,
                case,
                report,
                self.store.list_conflicts(case_id),
            )
        return case

    def replan_case(
        self,
        case_id: str,
        analyst: str,
        note: str = "",
        staleness_decision: dict | None = None,
    ) -> dict:
        """Revise the plan mid-case: the re-gated path back to
        planning.

        The superseded plan stays in the versioned plan store; a
        draft the revision supersedes is snapshotted first, so
        revising a plan never destroys a report the desk already
        had. The case returns to PLAN_REJECTED, from which
        run_planning mints plan version n+1 — with the analyst
        notes in its context — and the new plan faces the plan
        gate again like any other.

        Findings reuse is set only when staleness marks named the
        workstreams the new evidence touches (the re-plan that
        answers an evidence arrival): the next execution re-runs
        exactly the marked workstreams and reuses the rest under
        _reuse_skip's evidence-unchanged rule. A general revision
        sets no directive and re-executes in full.
        """
        case = self._require_case(case_id)
        if case["status"] in ("INTAKE", "EXECUTING", "FINAL"):
            raise GateError(
                f"no plan to revise from status {case['status']}"
            )
        plan = self.store.get_plan(case_id)
        if plan is None:
            raise GateError("no plan to revise")
        marks = case.get("stale_marks") or []
        must_rerun = sorted({m["question_id"] for m in marks})
        snapshot_id = None
        report = self.store.get_report(case_id)
        if report is not None:
            snapshot = self.store.add_report_snapshot(
                case_id,
                {
                    "report": report,
                    "reason": "superseded by a plan revision",
                    "analyst": analyst,
                    "created_at": utcnow(),
                },
            )
            snapshot_id = snapshot["snapshot_id"]
            self.store.add_audit(
                case_id,
                "report_snapshotted",
                {
                    "snapshot_id": snapshot_id,
                    "plan_version": plan["version"],
                    "reason": "superseded by a plan revision",
                },
            )
        decisions = list(case.get("staleness_decisions") or [])
        if staleness_decision is not None:
            decisions.append(staleness_decision)
        from_status = case["status"]
        case = self.store.update_case(
            case_id,
            status="PLAN_REJECTED",
            stale_marks=[],
            staleness_decisions=decisions,
            reuse_findings=bool(marks),
            must_rerun=must_rerun,
        )
        self.store.add_audit(
            case_id,
            "plan_revision_requested",
            {
                "analyst": analyst,
                "note": note,
                "from_status": from_status,
                "superseded_plan_version": plan["version"],
                "must_rerun": must_rerun,
                "report_snapshot": snapshot_id,
            },
        )
        log.info(
            "case %s plan revision requested by %s (plan v%d superseded)",
            case_id,
            analyst,
            plan["version"],
        )
        return case

    # -- case notes -------------------------------------------------------------
    def add_case_note(self, case_id: str, analyst: str, text: str) -> dict:
        """Attach an analyst note to a case: the desk's memory.

        A note is context, never evidence. It lives in its own
        table, is never mounted in the case filesystem, and its ID
        is not an evidence ID — `record_finding` validates citations
        against the evidence table, so a note can never be cited.
        Notes reach the planner only as the bounded, labelled
        section in the planning instruction (harness.notes_section),
        and only when a plan is (re-)made.
        """
        self._require_case(case_id)
        if not text or not text.strip():
            raise ValueError("a case note needs text")
        note = self.store.add_note(
            case_id,
            {"analyst": analyst, "text": text.strip(), "created_at": utcnow()},
        )
        self.store.add_audit(
            case_id,
            "case_note_added",
            {"note_id": note["note_id"], "analyst": analyst},
        )
        return note

    # -- report formats + section notes ---------------------------------------
    def _report_or_gate(self, case_id: str) -> tuple[dict, dict]:
        case = self._require_case(case_id)
        report = self.store.get_report(case_id)
        if report is None:
            raise GateError(f"case {case_id} has no report yet")
        if report.get("model") is None:
            # A report record that predates the structured model:
            # re-render it through the finalize step so every
            # format the desk serves comes from the one model.
            report = apply_conflicts(
                self.store,
                self.objects,
                case,
                report,
                self.store.list_conflicts(case_id),
            )
        return case, report

    def report_html(self, case_id: str) -> str:
        """The print-ready HTML rendering of the case's report —
        the HTML stored on the report record at finalize time,
        rendered from the same model as the markdown."""
        _, report = self._report_or_gate(case_id)
        return report["html"]

    def report_export(self, case_id: str) -> dict:
        """The JSON export: the report's structured model plus the
        analyst's section annotations, in their own field."""
        case, report = self._report_or_gate(case_id)
        return export_report(self.store, case, report)

    def add_report_section_note(
        self, case_id: str, analyst: str, section: str, text: str
    ) -> dict:
        """Annotate one section of the case's report.

        A section note is margin commentary: it lives in its own
        table, never in the report body — the markdown and the HTML
        do not carry it, and writing or editing one never changes
        the report record (or its hash). Its only surface is the
        JSON export's annotations field. The section must be one of
        the report's sections: a question id, or a named section
        (executive_summary, conflicts, risks_and_open_items,
        references, verification, method_and_limits).
        """
        case = self._require_case(case_id)
        if self.store.get_report(case_id) is None:
            raise GateError(f"case {case_id} has no report to annotate")
        valid = {s["key"] for s in model_sections(case)}
        if section not in valid:
            raise ValueError(
                f"unknown report section {section!r}; "
                f"sections are: {', '.join(sorted(valid))}"
            )
        if not text or not text.strip():
            raise ValueError("a section note needs text")
        note = self.store.add_section_note(
            case_id,
            {
                "section": section,
                "analyst": analyst,
                "text": text.strip(),
                "created_at": utcnow(),
                "edited_at": None,
            },
        )
        self.store.add_audit(
            case_id,
            "report_section_note_added",
            {
                "note_id": note["note_id"],
                "section": section,
                "analyst": analyst,
            },
        )
        return note

    def edit_report_section_note(
        self, case_id: str, note_id: str, analyst: str, text: str
    ) -> dict:
        """Edit a section note's text. The edit is recorded (who,
        when) on the note and in the audit trail; the report record
        is not touched — an edit can no more change the report
        than the original note could."""
        self._require_case(case_id)
        note = self.store.get_section_note(case_id, note_id)
        if note is None:
            raise KeyError(note_id)
        if not text or not text.strip():
            raise ValueError("a section note needs text")
        note["text"] = text.strip()
        note["edited_at"] = utcnow()
        note["edited_by"] = analyst
        self.store.update_section_note(case_id, note)
        self.store.add_audit(
            case_id,
            "report_section_note_edited",
            {
                "note_id": note_id,
                "section": note["section"],
                "analyst": analyst,
            },
        )
        return note

    # -- source governance ------------------------------------------------------
    def register_source(
        self,
        analyst: str,
        *,
        name: str,
        kind: str,
        owner: str,
        usage_basis: str,
        notes: str = "",
    ) -> dict:
        """Register a source in the governance registry.

        Registration is a recorded act, not a config edit: the
        record carries the analyst, the moment, and its own act
        history. Until a source is registered, no evidence — from
        the corpus or from an analyst's upload — may cite it.
        """
        if not name or not name.strip():
            raise ValueError("a source needs a name")
        if kind not in SOURCE_KINDS:
            raise ValueError(
                f"unknown source kind {kind!r}; "
                f"expected one of {', '.join(SOURCE_KINDS)}"
            )
        if usage_basis not in USAGE_BASES:
            raise ValueError(
                f"unknown usage basis {usage_basis!r}; "
                f"expected one of {', '.join(USAGE_BASES)}"
            )
        if not owner or not owner.strip():
            raise ValueError("a source needs an owner")
        name = name.strip()
        if self.store.get_source(name) is not None:
            raise GateError(f"source {name!r} is already registered")
        record = {
            "name": name,
            "kind": kind,
            "owner": owner.strip(),
            "usage_basis": usage_basis,
            "notes": notes,
            "registered_by": analyst,
            "registered_at": utcnow(),
            "history": [
                {"action": "registered", "analyst": analyst, "at": utcnow()}
            ],
        }
        log.info("source %s registered by %s", name, analyst)
        return self.store.add_source(record)

    def case_sources(self, case_id: str) -> dict:
        """The governance records behind a case's sources: what
        the brief allowed, resolved entry by entry, and what the
        evidence actually came from (governance.case_source_policy)."""
        case = self._require_case(case_id)
        return case_source_policy(self.store, case)

    def case_provenance(self, case_id: str) -> dict:
        """The per-case provenance report (governance.case_provenance):
        every evidence item with its source's governance record,
        its hash, its ingestion path, and its citing findings and
        claims."""
        case = self._require_case(case_id)
        return case_provenance(self.store, case)

    def retention_sweep(self) -> list[dict]:
        """Cases past their retention period, listed for an
        analyst's decision. The sweep is a report, never an act:
        it deletes nothing, holds nothing, and changes no state —
        it names the standard-retention cases whose retention
        (policy: RETENTION_DAYS from creation) has run out.
        Cases under legal hold are never listed."""
        now = parse_ts(utcnow())
        due: list[dict] = []
        for case in self.store.list_cases():
            if (case.get("retention_class") or "standard") != "standard":
                continue
            due_at = retention_due(case, self.settings.retention_days)
            if now >= due_at:
                due.append(
                    {
                        "case_id": case["case_id"],
                        "subject": case["subject"],
                        "status": case["status"],
                        "created_at": case["created_at"],
                        "retention_class": "standard",
                        "retention_due_at": due_at.isoformat(
                            timespec="seconds"
                        ),
                    }
                )
        return due

    def set_legal_hold(
        self, case_id: str, analyst: str, hold: bool, reason: str = ""
    ) -> dict:
        """Set or clear a case's legal hold — a recorded analyst
        act either way (the act, the analyst, the reason, and the
        moment land on the case record and in the audit trail).
        While the hold stands, deletion refuses in code."""
        case = self._require_case(case_id)
        current = case.get("retention_class") or "standard"
        if hold and current == "legal_hold":
            raise GateError(f"case {case_id} is already under legal hold")
        if not hold and current != "legal_hold":
            raise GateError(f"case {case_id} is not under legal hold")
        entry = {
            "action": "set" if hold else "cleared",
            "analyst": analyst,
            "reason": reason,
            "at": utcnow(),
        }
        history = list(case.get("retention_history") or [])
        history.append(entry)
        case = self.store.update_case(
            case_id,
            retention_class="legal_hold" if hold else "standard",
            retention_history=history,
        )
        self.store.add_audit(
            case_id,
            "legal_hold_set" if hold else "legal_hold_cleared",
            entry,
        )
        log.info(
            "case %s legal hold %s by %s",
            case_id,
            "set" if hold else "cleared",
            analyst,
        )
        return case

    def delete_case(self, case_id: str, analyst: str, reason: str) -> dict:
        """End a case: the deletion act.

        Explicit and recorded: a named analyst, a stated reason.
        The case's rows and its objects (evidence originals and
        case files alike) are removed, and a tombstone is all
        that remains — case id, subject, who deleted it and why
        and when, how much evidence it held, and the hash of the
        final report it carried, if it carried one. A case under
        legal hold refuses here, in code, however it is asked.
        """
        if not reason or not reason.strip():
            raise ValueError("deleting a case requires a recorded reason")
        case = self._require_case(case_id)
        if (case.get("retention_class") or "standard") == "legal_hold":
            raise GateError(
                f"case {case_id} is under legal hold and cannot be "
                "deleted; clear the hold first, as its own recorded act"
            )
        report = self.store.get_report(case_id)
        final_hash = None
        if report is not None and report.get("markdown"):
            final_hash = sha256_hex(report["markdown"].encode("utf-8"))
        evidence_count = self.store.count_evidence(case_id)
        for key in self.objects.list_keys(f"cases/{case_id}/"):
            self.objects.delete(key)
        self.store.delete_case_data(case_id)
        tombstone = self.store.add_tombstone(
            {
                "case_id": case_id,
                "subject": case["subject"],
                "subject_ref": case["subject_ref"],
                "deleted_by": analyst,
                "reason": reason.strip(),
                "deleted_at": utcnow(),
                "final_report_sha256": final_hash,
                "evidence_count": evidence_count,
            }
        )
        log.info(
            "case %s deleted by %s: %s", case_id, analyst, reason.strip()
        )
        return tombstone

    def redact_evidence(
        self, case_id: str, evidence_id: str, analyst: str, reason: str
    ) -> dict:
        """Redact one evidence item, for a recorded reason.

        The stored bytes are replaced by a content-free marker;
        the redaction record keeps the item's original SHA-256,
        the reason, and the analyst — never the content. Claims in
        the case's live report that cite the item are flagged:
        until each is acknowledged or sent back, the report cannot
        be approved. The verifier treats a redacted item as
        missing from here on — a marker supports no claim.
        """
        if not reason or not reason.strip():
            raise ValueError("redacting evidence requires a recorded reason")
        self._require_case(case_id)
        ev = self.store.get_evidence(case_id, evidence_id)
        if ev is None:
            raise KeyError(evidence_id)
        if ev.get("redacted"):
            raise GateError(
                f"evidence {evidence_id} is already redacted"
            )
        at = utcnow()
        original_sha = ev["sha256"]
        redaction = self.store.add_redaction(
            case_id,
            {
                "evidence_id": evidence_id,
                "title": ev["title"],
                "original_sha256": original_sha,
                "analyst": analyst,
                "reason": reason.strip(),
                "redacted_at": at,
            },
        )
        self.objects.put(
            ev["object_key"],
            redaction_marker(ev, analyst, reason.strip(), at).encode("utf-8"),
        )
        ev["redacted"] = True
        ev["redaction"] = {
            "redaction_id": redaction["redaction_id"],
            "analyst": analyst,
            "reason": reason.strip(),
            "redacted_at": at,
            "original_sha256": original_sha,
        }
        self.store.update_evidence(case_id, ev)
        self.store.add_audit(
            case_id,
            "evidence_redacted",
            {
                "redaction_id": redaction["redaction_id"],
                "evidence_id": evidence_id,
                "original_sha256": original_sha,
                "analyst": analyst,
                "reason": reason.strip(),
            },
        )
        flags: list[dict] = []
        report = self.store.get_report(case_id)
        if report is not None:
            for claim in report.get("claims", []):
                if evidence_id in claim.get("evidence_ids", []):
                    flags.append(
                        self.store.add_redaction_flag(
                            case_id,
                            {
                                "claim_id": claim["claim_id"],
                                "finding_id": claim["finding_id"],
                                "evidence_id": evidence_id,
                                "redaction_id": redaction["redaction_id"],
                                "status": "OPEN",
                                "created_at": utcnow(),
                                "decided_by": None,
                                "decision_note": None,
                                "decided_at": None,
                            },
                        )
                    )
            if flags:
                self.store.add_audit(
                    case_id,
                    "report_claims_flagged_redaction",
                    {
                        "evidence_id": evidence_id,
                        "redaction_id": redaction["redaction_id"],
                        "claims": [f["claim_id"] for f in flags],
                    },
                )
        log.info(
            "case %s evidence %s redacted by %s; %d claim(s) flagged",
            case_id,
            evidence_id,
            analyst,
            len(flags),
        )
        return {"redaction": redaction, "flags": flags}

    def decide_redaction_flag(
        self,
        case_id: str,
        flag_id: str,
        analyst: str,
        action: str,
        note: str = "",
    ) -> dict:
        """The analyst's two moves on a redaction-flagged claim —
        the R3 quarantine pattern, applied to redaction.

        ACKNOWLEDGE: the analyst records that the claim stands
        with its evidence redacted (the acknowledgement, its
        author, and its note are on the flag forever) and the
        approval block lifts for that claim. SEND BACK: the
        underlying finding leaves the case record and the case
        returns to APPROVED for re-execution, exactly as a
        quarantined claim is sent back — a re-investigation has
        to re-derive the claim from evidence that still exists.
        There is no third move and no silent path.
        """
        case = self._require_case(case_id)
        if case["status"] not in ("DRAFT_READY", "BUDGET_EXHAUSTED"):
            raise GateError(
                f"no report at the gate to decide a redaction flag "
                f"from (status {case['status']})"
            )
        if action not in ("acknowledge", "send_back"):
            raise ValueError(
                f"unknown redaction flag action {action!r}; "
                "expected 'acknowledge' or 'send_back'"
            )
        flag = self.store.get_redaction_flag(case_id, flag_id)
        if flag is None:
            raise KeyError(flag_id)
        if flag["status"] != "OPEN":
            raise GateError(
                f"redaction flag {flag_id} is already {flag['status']}"
            )
        flag["status"] = (
            "ACKNOWLEDGED" if action == "acknowledge" else "SENT_BACK"
        )
        flag["decided_by"] = analyst
        flag["decision_note"] = note
        flag["decided_at"] = utcnow()
        self.store.update_redaction_flag(case_id, flag)
        if action == "send_back":
            self.store.delete_finding(case_id, flag["finding_id"])
        self.store.add_audit(
            case_id,
            (
                "redaction_claim_acknowledged"
                if action == "acknowledge"
                else "redaction_claim_sent_back"
            ),
            {
                "flag_id": flag_id,
                "claim_id": flag["claim_id"],
                "finding_id": flag["finding_id"],
                "evidence_id": flag["evidence_id"],
                "analyst": analyst,
                "note": note,
            },
        )
        if action == "send_back":
            self.store.update_case(case_id, status="APPROVED")
        if action == "acknowledge":
            # The acknowledged flag stands in the draft's Risks and
            # open items section with its disposition — re-render so
            # the report at the gate shows the decision.
            report = self.store.get_report(case_id)
            if report is not None:
                apply_conflicts(
                    self.store,
                    self.objects,
                    case,
                    report,
                    self.store.list_conflicts(case_id),
                )
        return flag

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
        if case["status"] != "BUDGET_EXHAUSTED" and not self._planning_stopped(
            case
        ):
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
        # An exhausted execution returns to APPROVED for re-execution;
        # a planning run that stopped on its budget leaves the case
        # where it was (INTAKE or PLAN_REJECTED), ready to re-plan.
        status = (
            "APPROVED" if case["status"] == "BUDGET_EXHAUSTED" else case["status"]
        )
        return self.store.update_case(case_id, brief=brief, status=status)

    def _planning_stopped(self, case: dict) -> bool:
        """True when the case's most recent planning outcome is a
        budget stop — no plan was minted and no amendment has been
        recorded since. Planning exhaustion leaves no draft, so the
        case never reaches BUDGET_EXHAUSTED; this is what lets the
        analyst amend it instead of re-planning into the same cap."""
        if case["status"] not in ("INTAKE", "PLAN_REJECTED"):
            return False
        for entry in reversed(self.store.list_audit(case["case_id"])):
            if entry["event"] == "planning_budget_exceeded":
                return True
            if entry["event"] in ("plan_generated", "budget_amended"):
                return False
        return False

    # -- cost accounting --------------------------------------------------------
    def cost_report(self, case_id: str) -> dict:
        """One case's budget-vs-actual report: the budget ledger's
        granted-vs-consumed picture joined with the cost ledger —
        what the case cost, by phase, by sub-agent role, and by
        model, folded from the priced lines recorded when each run
        ended (see costs.py). Reading a report prices nothing: the
        figures are the recorded ones."""
        case = self._require_case(case_id)
        return cost_view(
            case, self.store.list_audit(case_id), self.price_table
        )

    def cost_totals(self) -> dict:
        """Every case's recorded cost, summed — a totals view over
        the per-case ledgers, computed by folding each case's own
        ledger and adding the folds. Each case still owns its
        detail at its own cost report; this is the desk's total
        spend line and the per-case figures it is made of, nothing
        more. (The per-engagement roll-up is the portfolio fold's,
        in portfolio.py.)"""
        cases: list[dict] = []
        total_cost = Decimal(0)
        total_tokens = 0
        total_calls = 0
        contains_estimates = False
        for case in self.store.list_cases():
            fold = fold_cost_ledger(self.store.list_audit(case["case_id"]))
            totals = fold["totals"]
            total_cost += totals["cost_usd"]
            total_tokens += totals["total_tokens"]
            total_calls += totals["calls"]
            contains_estimates = (
                contains_estimates or fold["contains_estimates"]
            )
            cases.append(
                {
                    "case_id": case["case_id"],
                    "subject": case["subject"],
                    "status": case["status"],
                    "calls": totals["calls"],
                    "total_tokens": totals["total_tokens"],
                    "cost_usd": float(totals["cost_usd"]),
                    "contains_estimates": fold["contains_estimates"],
                    "has_unpriced": fold["has_unpriced"],
                }
            )
        return {
            "currency": self.price_table.currency,
            "cases": cases,
            "totals": {
                "cases": len(cases),
                "calls": total_calls,
                "total_tokens": total_tokens,
                "cost_usd": float(total_cost),
            },
            "contains_estimates": contains_estimates,
        }

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
        report = self._detect_conflicts(case, report)
        return self._challenge_draft(case, report)

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

    def _challenge_draft(self, case: dict, report: dict) -> dict:
        """The challenge pass, after verification and conflict
        detection: a deterministic second opinion over the draft's
        supported claims, computed from the stored records (never
        from composer internals) and stored as challenge records.
        The report itself is untouched — challenges are the desk's
        internal review, and no delivered format carries one."""
        case_id = case["case_id"]
        outcome = sync_challenges(self.store, case, report)
        records = outcome["records"]
        open_now = [r for r in records if r["status"] == "OPEN"]
        if outcome["minted"] or outcome["superseded"] or outcome["reopened"]:
            self.store.add_audit(
                case_id,
                "challenges_raised",
                {
                    "raised": [
                        {
                            "challenge_id": r["challenge_id"],
                            "rule": r["rule"],
                            "severity": r["severity"],
                            "claim_id": r["claim_id"],
                            "question_id": r["question_id"],
                        }
                        for r in outcome["minted"]
                    ],
                    "reopened": [
                        r["challenge_id"] for r in outcome["reopened"]
                    ],
                    "superseded": [
                        r["challenge_id"] for r in outcome["superseded"]
                    ],
                    "open_must_answer": sum(
                        1 for r in open_now if r["severity"] == "must_answer"
                    ),
                    "open_advisory": sum(
                        1 for r in open_now if r["severity"] == "advisory"
                    ),
                },
            )
        if open_now:
            log.info(
                "case %s challenges: %d open (%d must-answer)",
                case_id,
                len(open_now),
                sum(1 for r in open_now if r["severity"] == "must_answer"),
            )
        return report

    def run_challenge_pass(self, case_id: str) -> list[dict]:
        """Re-run the challenge pass over the case's current draft
        and return the case's full challenge history. The pass is
        idempotent by content — on an unchanged draft this changes
        nothing. It runs automatically on every fresh draft; this
        entry point exists so the desk (and the tests) can re-run
        it on demand."""
        case = self._require_case(case_id)
        report = self.store.get_report(case_id)
        if report is None:
            raise GateError(f"case {case_id} has no report to challenge")
        self._challenge_draft(case, report)
        return self.store.list_challenges(case_id)

    def decide_challenge(
        self,
        case_id: str,
        challenge_id: str,
        analyst: str,
        action: str,
        note: str = "",
        counter_evidence_ids: list[str] | tuple[str, ...] = (),
    ) -> dict:
        """The analyst's three answers to a challenge — and only three.

        rebut: the claim stands, and the analyst says why in
        writing, naming counter-evidence from the case. The
        rebuttal rides the case (this record and the audit trail),
        never the report body.

        revise: the objection lands — the claim is sent back for
        re-investigation, the reinvestigate path's effects applied
        to a supported claim: its finding leaves the case record,
        so a re-run has to re-derive it, and the case returns to
        APPROVED, ready to execute again.

        downgrade: the claim stops standing as support. It is
        quarantined through the existing shelf with reason
        challenge_downgraded and leaves the report body now, the
        draft re-rendered without it; the quarantine record can
        still be decided (sent back or dropped) like any other.

        revise and downgrade need a claim to act on, so they are
        refused for question- and case-level challenges — those
        are answered by rebuttal.
        """
        case = self._require_case(case_id)
        if case["status"] not in ("DRAFT_READY", "BUDGET_EXHAUSTED"):
            raise GateError(
                f"no report at the gate to answer challenges on "
                f"(status {case['status']})"
            )
        if action not in ("rebut", "revise", "downgrade"):
            raise ValueError(
                f"unknown challenge action {action!r}; expected "
                "'rebut', 'revise' or 'downgrade'"
            )
        record = self.store.get_challenge(case_id, challenge_id)
        if record is None:
            raise KeyError(challenge_id)
        if record["status"] != "OPEN":
            raise GateError(
                f"challenge {challenge_id} is already {record['status']}"
            )
        answer: dict = {"analyst": analyst, "note": note, "at": utcnow()}
        if action == "rebut":
            if not note.strip():
                raise ValueError("a rebuttal must say why the claim stands")
            named = [e for e in counter_evidence_ids if e]
            if not named:
                raise ValueError(
                    "a rebuttal must name counter-evidence from the case"
                )
            known = {
                ev["evidence_id"] for ev in self.store.list_evidence(case_id)
            }
            foreign = [e for e in named if e not in known]
            if foreign:
                raise ValueError(
                    "counter-evidence must come from this case; "
                    f"unknown evidence: {', '.join(sorted(foreign))}"
                )
            answer["disposition"] = "rebutted"
            answer["counter_evidence_ids"] = named
            record["disposition"] = "rebutted"
        elif action == "revise":
            if record["claim_id"] is None:
                raise GateError(
                    f"challenge {challenge_id} names no single claim "
                    "to send back; answer it with a rebuttal"
                )
            answer["disposition"] = "revised"
            record["disposition"] = "revised"
            self.store.delete_finding(case_id, record["finding_id"])
        else:  # downgrade
            if record["claim_id"] is None:
                raise GateError(
                    f"challenge {challenge_id} names no single claim "
                    "to downgrade; answer it with a rebuttal"
                )
            answer["disposition"] = "downgraded"
            record["disposition"] = "downgraded"
            answer["quarantine_id"] = self._quarantine_for_downgrade(
                case, record
            )
        record["status"] = "ANSWERED"
        record["answers"] = [*record.get("answers", []), answer]
        self.store.update_challenge(case_id, record)
        event = {
            "rebut": "challenge_rebutted",
            "revise": "challenge_revised",
            "downgrade": "challenge_downgraded",
        }[action]
        payload: dict = {
            "challenge_id": challenge_id,
            "rule": record["rule"],
            "severity": record["severity"],
            "claim_id": record["claim_id"],
            "question_id": record["question_id"],
            "analyst": analyst,
            "note": note,
        }
        if action == "rebut":
            payload["counter_evidence_ids"] = answer["counter_evidence_ids"]
        if action == "revise":
            payload["finding_id"] = record["finding_id"]
        if action == "downgrade":
            payload["quarantine_id"] = answer["quarantine_id"]
        self.store.add_audit(case_id, event, payload)
        log.info(
            "case %s challenge %s %s by %s",
            case_id,
            challenge_id,
            answer["disposition"],
            analyst,
        )
        if action == "revise":
            self.store.update_case(case_id, status="APPROVED")
        return record

    def _quarantine_for_downgrade(self, case: dict, challenge: dict) -> str:
        """Quarantine a challenged claim through the existing
        shelf, with reason challenge_downgraded, and re-render the
        draft without it — the finalize chain rebuilt over the
        claims that remain, so the ledger, the coverage table, and
        every grade describe the report the analyst now sees."""
        case_id = case["case_id"]
        existing = self.store.list_quarantine(case_id)
        number = (
            max(
                (int(r["quarantine_id"].split("-")[1]) for r in existing),
                default=0,
            )
            + 1
        )
        record = {
            "quarantine_id": f"Q-{number:04d}",
            "case_id": case_id,
            "claim_id": challenge["claim_id"],
            "finding_id": challenge["finding_id"],
            "question_id": challenge["question_id"],
            "claim": challenge["claim"],
            "evidence_ids": list(challenge["evidence_ids"]),
            "reason": "challenge_downgraded",
            "failures": [
                {
                    "reason": "challenge_downgraded",
                    "detail": (
                        f"downgraded at the report gate under challenge "
                        f"{challenge['challenge_id']} ({challenge['rule']}): "
                        "the claim no longer stands as support for its "
                        "question"
                    ),
                }
            ],
            "status": "QUARANTINED",
            "verifier_run": None,
            "origin": "challenge",
            "challenge_id": challenge["challenge_id"],
            "created_at": utcnow(),
            "decided_by": None,
            "decision_note": None,
            "decided_at": None,
        }
        self.store.replace_quarantine(case_id, [*existing, record])
        report = self.store.get_report(case_id)
        verifier_report = self.store.get_verifier_report(case_id)
        if report is not None and verifier_report is not None:
            report = apply_verification(
                self.store,
                self.objects,
                case,
                report,
                verifier_report,
                self.store.list_quarantine(case_id),
            )
            apply_conflicts(
                self.store,
                self.objects,
                case,
                report,
                self.store.list_conflicts(case_id),
            )
        return record["quarantine_id"]

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
        if action == "drop":
            # The disposition belongs in the draft the analyst is
            # looking at: the Risks and open items section is
            # generated from the quarantine records, so re-render.
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

    def decide_report(self, case_id: str, analyst: str, approve: bool, note: str = "") -> dict:
        case = self._require_case(case_id)
        if case["status"] not in ("DRAFT_READY", "BUDGET_EXHAUSTED"):
            raise GateError(f"no report awaiting decision (status {case['status']})")
        report = self.store.get_report(case_id)
        if approve and case.get("stale_marks"):
            # Staleness gate: evidence that arrived while the draft
            # was at the gate marked workstreams stale. Staleness is
            # decided, never silent — the analyst proceeds (recorded,
            # and rendered into the report's limits) or re-plans
            # before the report may become final.
            stale = ", ".join(
                sorted({m["question_id"] for m in case["stale_marks"]})
            )
            raise GateError(
                "new evidence arrived after planning and these "
                f"workstreams are marked stale: {stale}. An analyst "
                "must re-plan or record a decision to proceed before "
                "the report can be approved"
            )
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
            # Redaction gate: a claim in this report that cites
            # evidence redacted since composition stands flagged,
            # and the report cannot become final over an open flag.
            # The analyst's moves are on the flag — acknowledge it
            # (recorded) or send the claim back — not here.
            report_claim_ids = {
                c["claim_id"] for c in report.get("claims", [])
            }
            open_flags = [
                flag
                for flag in self.store.list_redaction_flags(case_id)
                if flag["status"] == "OPEN"
                and flag["claim_id"] in report_claim_ids
            ]
            if open_flags:
                flagged = ", ".join(
                    sorted({f["claim_id"] for f in open_flags})
                )
                raise GateError(
                    "report cites evidence that has been redacted "
                    f"({flagged}); each flagged claim must be "
                    "acknowledged or sent back before approval"
                )
            # Challenge gate: the challenge pass's must-answer
            # objections are the desk's second opinion in structured
            # form, and a report does not become final over one.
            # Each must be answered on its own record — rebutted
            # with counter-evidence, sent back for re-investigation,
            # or downgraded — never waved through here. Advisory
            # challenges never block; they are counted on the case
            # view for the analyst at this gate.
            open_must_answer = [
                c
                for c in self.store.list_challenges(case_id)
                if c["status"] == "OPEN" and c["severity"] == "must_answer"
            ]
            if open_must_answer:
                ids = ", ".join(
                    c["challenge_id"] for c in open_must_answer
                )
                raise GateError(
                    "report has unanswered must-answer challenges "
                    f"({ids}); each must be answered — rebutted with "
                    "counter-evidence, sent back for re-investigation, "
                    "or downgraded — before approval"
                )
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
        if approve:
            # The approved report must not read as a draft: swap the
            # status line in every stored format (markdown, HTML,
            # model). Nothing else in the report changes at the gate.
            final_line = final_status_line(analyst, utcnow()[:10])
            report["markdown"] = report["markdown"].replace(
                DRAFT_STATUS_LINE, final_line, 1
            )
            if report.get("html"):
                report["html"] = report["html"].replace(
                    html.escape(DRAFT_STATUS_LINE, quote=True),
                    html.escape(final_line, quote=True),
                    1,
                )
            if report.get("model"):
                report["model"] = {**report["model"], "status": "APPROVED"}
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

    # -- analyst workbench ----------------------------------------------------
    def assign_case(self, case_id: str, analyst: str, assignee: str) -> dict:
        """Record who at the desk currently owns this case.

        Assignment is triage, not authority: it orders and
        filters the workbench queue, and it confers no decision
        rights — every gate still takes any analyst's individual
        decision on the case itself. The act is recorded (who
        assigned, to whom, what it replaced); an empty assignee
        returns the case to the unassigned pool."""
        case = self._require_case(case_id)
        new = (assignee or "").strip() or None
        previous = case.get("assignee")
        case = self.store.update_case(case_id, assignee=new)
        self.store.add_audit(
            case_id,
            "case_assigned",
            {"analyst": analyst, "assignee": new, "previous": previous},
        )
        log.info("case %s assigned to %s by %s", case_id, new, analyst)
        return case

    def set_priority(self, case_id: str, analyst: str, priority: str) -> dict:
        """Set the case's triage priority (normal / high /
        urgent). A recorded act, like assignment — and, like
        assignment, it changes no gate, budget, or decision rule;
        it tells the desk what to pick up first."""
        case = self._require_case(case_id)
        if priority not in PRIORITIES:
            raise ValueError(
                f"unknown priority {priority!r}; "
                f"priorities are: {', '.join(PRIORITIES)}"
            )
        previous = case.get("priority") or "normal"
        case = self.store.update_case(case_id, priority=priority)
        self.store.add_audit(
            case_id,
            "case_priority_set",
            {"analyst": analyst, "priority": priority, "previous": previous},
        )
        log.info("case %s priority %s by %s", case_id, priority, analyst)
        return case

    def workbench_queue(self, filters: dict | None = None) -> dict:
        """The desk's queue: every case needing an analyst act,
        grouped by the act, derived from the records (see
        workbench.py). Building it runs the interrupted-case scan,
        because listing is that scan's detection point — a case a
        dead process stranded belongs in the queue the first time
        anyone looks."""
        return build_queue(
            self.store, self.interrupted_cases(), filters=filters
        )

    def case_timeline(self, case_id: str) -> list[dict]:
        """The case's story in one chronological view: the audit
        trail merged with the analyst's notes (enriched with
        their text), rendered from the records."""
        self._require_case(case_id)
        return build_timeline(self.store, case_id)

    def search_cases(
        self, query: str, engagement_id: str | None = None
    ) -> list[dict]:
        """Deterministic text search over subjects, brief
        questions, and finding claims (see workbench.py). With
        an engagement named, the search is scoped to that
        engagement's member cases; the results stay case-scoped
        and case-labelled either way — scoping narrows the desk,
        it never merges cases."""
        case_ids = None
        if engagement_id is not None:
            if self.store.get_engagement(engagement_id) is None:
                raise KeyError(engagement_id)
            case_ids = {
                case["case_id"]
                for case in member_cases(self.store, engagement_id)
            }
        return _search_cases(self.store, query, case_ids=case_ids)

    # -- portfolio mode -----------------------------------------------------
    def create_engagement(
        self, analyst: str, name: str, client: str, owner: str
    ) -> dict:
        """Open an engagement: a client mandate under which
        cases run. A governed record, created as a recorded act
        — the creator is on the record, and the record carries
        its own history from this first entry. An engagement
        groups cases; it holds no case state and decides
        nothing."""
        for field, value in (("name", name), ("client", client), ("owner", owner)):
            if not (value or "").strip():
                raise ValueError(f"an engagement needs a {field}")
        now = utcnow()
        record = self.store.add_engagement(
            {
                "name": name.strip(),
                "client": client.strip(),
                "owner": owner.strip(),
                "created_by": analyst,
                "created_at": now,
                "updated_at": now,
                "history": [
                    {"action": "created", "analyst": analyst, "at": now}
                ],
            }
        )
        log.info(
            "engagement %s created by %s (%s)",
            record["engagement_id"],
            analyst,
            record["name"],
        )
        return record

    def assign_case_engagement(
        self, case_id: str, engagement_id: str | None, analyst: str
    ) -> dict:
        """Put a case under an engagement — or take it out of
        one. A recorded act on both records: the case's audit
        trail names the engagement and what it replaced, and
        each engagement's history names the case that arrived or
        left, and from or to where. A case sits in at most one
        engagement, so a move writes both sides; nothing about
        the case itself — its gates, budget, evidence, or
        retention — is touched. Asking for the membership the
        case already has is not an act and is refused."""
        case = self._require_case(case_id)
        current_id = case.get("engagement_id") or None
        target = None
        if engagement_id is not None:
            target = self.store.get_engagement(engagement_id)
            if target is None:
                raise KeyError(engagement_id)
        if engagement_id == current_id:
            raise GateError(
                f"case {case_id} is already "
                + (
                    f"in engagement {engagement_id}"
                    if engagement_id
                    else "outside any engagement"
                )
                + "; a membership that does not change is not an act"
            )
        now = utcnow()
        case = self.store.update_case(case_id, engagement_id=engagement_id)
        previous = None
        if current_id is not None:
            previous = self.store.get_engagement(current_id)
            entry = {
                "action": "case_unassigned",
                "case_id": case_id,
                "analyst": analyst,
                "at": now,
                "to_engagement_id": engagement_id,
            }
            if previous is not None:
                previous["history"] = list(
                    previous.get("history") or []
                ) + [entry]
                previous["updated_at"] = now
                self.store.update_engagement(previous)
        if target is not None:
            entry = {
                "action": "case_assigned",
                "case_id": case_id,
                "analyst": analyst,
                "at": now,
                "from_engagement_id": current_id,
            }
            target["history"] = list(target.get("history") or []) + [entry]
            target["updated_at"] = now
            self.store.update_engagement(target)
        if engagement_id is not None:
            self.store.add_audit(
                case_id,
                "case_assigned_to_engagement",
                {
                    "analyst": analyst,
                    "engagement_id": engagement_id,
                    "engagement_name": target["name"],
                    "previous_engagement_id": current_id,
                },
            )
        else:
            previous_name = previous["name"] if previous else None
            self.store.add_audit(
                case_id,
                "case_removed_from_engagement",
                {
                    "analyst": analyst,
                    "engagement_id": current_id,
                    "engagement_name": previous_name,
                },
            )
        log.info(
            "case %s engagement %s -> %s by %s",
            case_id,
            current_id,
            engagement_id,
            analyst,
        )
        return case

    def portfolio_report(self, engagement_id: str) -> dict:
        """One engagement's portfolio report: the fold over its
        member cases (see portfolio.py) — counts by state, the
        gates awaiting decision across the portfolio (the
        workbench queue's own items, scoped to members), the
        open conflicts and quarantined claims by id, and the
        cost roll-up summed from the cases' own ledgers. The
        queue build runs the interrupted-case scan, as every
        queue read does."""
        engagement = self.store.get_engagement(engagement_id)
        if engagement is None:
            raise KeyError(engagement_id)
        return build_portfolio(
            self.store,
            engagement,
            self.workbench_queue(),
            self.settings.retention_days,
            currency=self.price_table.currency,
        )

    def list_portfolios(self) -> list[dict]:
        """Every engagement's summary, folded the same way as
        the full report (one queue build serves them all)."""
        queue = self.workbench_queue()
        return [
            summarize_portfolio(
                build_portfolio(
                    self.store,
                    engagement,
                    queue,
                    self.settings.retention_days,
                    currency=self.price_table.currency,
                )
            )
            for engagement in self.store.list_engagements()
        ]

    def save_view(self, analyst: str, name: str, filters: dict) -> dict:
        """Save a queue filter under the analyst's name.

        Saving under a name the analyst already holds replaces
        that view's filters — a view is a working shortcut, not a
        record of anything decided. The filter set is validated
        against the workbench's own keys, so a view can never name
        a filter the queue would not understand."""
        cleaned = validate_filters(filters)
        if not (name or "").strip():
            raise ValueError("a saved view needs a name")
        existing = self.store.find_view(analyst, name.strip())
        now = utcnow()
        if existing is not None:
            existing["filters"] = cleaned
            existing["updated_at"] = now
            return self.store.update_view(existing)
        return self.store.add_view(
            {
                "analyst": analyst,
                "name": name.strip(),
                "filters": cleaned,
                "created_at": now,
                "updated_at": now,
            }
        )

    def list_views(self, analyst: str | None = None) -> list[dict]:
        return self.store.list_views(analyst)

    def delete_view(self, analyst: str, view_id: str) -> dict:
        """Remove one of the analyst's own saved views. Only the
        view goes — no case, queue item, or decision is touched,
        because a view never held any of those."""
        view = self.store.get_view(view_id)
        if view is None or view["analyst"] != analyst:
            raise KeyError(view_id)
        self.store.delete_view(view_id)
        return view

    # -- observability ---------------------------------------------------------

    def case_trace(self, case_id: str) -> dict:
        """The case's run, reconstructed end to end from its
        stored records (observability.build_trace): the phase
        runs folded from the budget and cost ledgers, the act
        sequence in causal order, and the taxonomy's verdict on
        how the run ended. A pure read — building it writes
        nothing."""
        self._require_case(case_id)
        return build_trace(self.store, case_id)

    def case_outcome(self, case_id: str) -> dict:
        """The taxonomy's verdict on one case (observability.
        classify_case_outcome): exactly one label, with the
        records that decided it named in the detail."""
        self._require_case(case_id)
        return classify_case_outcome(self.store, case_id)

    def generate_digest(self, day: str, analyst: str) -> dict:
        """Generate a day's operations digest and store it as the
        day's next version. The fold (observability.fold_digest)
        recomputes everything from the case records as they stand
        now; the stored record is the snapshot — day, version,
        who generated it and when, and the fold itself. Earlier
        versions of the day are never rewritten."""
        day = _validate_digest_day(day)
        record = {
            "day": day,
            "generated_at": utcnow(),
            "generated_by": analyst,
            "digest": fold_digest(self.store, day),
        }
        return self.store.add_digest(record)

    def get_digest(self, day: str) -> dict | None:
        """The latest stored version of a day's digest, or None —
        an ungenerated day reads as absent."""
        return self.store.get_digest(_validate_digest_day(day))

    def list_digests(self) -> list[dict]:
        return self.store.list_digests()
