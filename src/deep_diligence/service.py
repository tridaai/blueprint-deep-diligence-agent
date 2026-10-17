"""CaseService: the governance shell around the harness.

State machine (see models.STATUSES):
  INTAKE --plan--> PLAN_READY --approve--> APPROVED --execute-->
  DRAFT_READY --approve--> FINAL
Rejections return the case to PLAN_REJECTED / REPORT_REJECTED, from
which planning / execution may run again. A budget-exhausted execution
lands in BUDGET_EXHAUSTED with a partial draft, which the analyst may
still approve or reject. Execution attempted from any other state is
a GateError — the plan gate is enforced here, in code, not by prompt.
"""

from __future__ import annotations

import json
import logging
import uuid

from langchain_core.messages import HumanMessage

from .backend import EvidenceBackend
from .budgets import BudgetExceededError, BudgetGuard
from .composer import compose_report
from .config import Settings
from .corpus import companies
from .evidence import ingest_case_evidence
from .harness import build_agent, execute_message, plan_message
from .llm import build_model
from .models import BriefIn
from .objects import ObjectStore, build_object_store
from .store import Store, build_store, utcnow
from .tools import build_case_tools

log = logging.getLogger("deep_diligence.service")


class GateError(RuntimeError):
    """A lifecycle transition the case state does not allow."""


class CaseService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store: Store = build_store(settings)
        self.objects: ObjectStore = build_object_store(settings)

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

    def _run_agent(self, case: dict, message: str):
        """One harness run under the case budget. Returns (usage, error)."""
        backend = self._backend(case["case_id"])
        tools = build_case_tools(self.store, self.objects, case)
        agent = build_agent(
            model=build_model(self.settings), backend=backend, case_tools=tools
        )
        sink: list = []
        guard = BudgetGuard(case["brief"]["budget"], audit_sink=sink)
        budget = case["brief"]["budget"]
        config = {
            "callbacks": [guard],
            "recursion_limit": max(150, int(budget["max_steps"]) * 6),
        }
        error: BudgetExceededError | None = None
        try:
            agent.invoke({"messages": [HumanMessage(content=message)]}, config=config)
        except BudgetExceededError as exc:
            error = exc
        self._flush_audit(case["case_id"], sink)
        return guard.usage(), error

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
            "created_at": now,
            "updated_at": now,
        }
        self.store.create_case(case)
        evidence = ingest_case_evidence(self.store, self.objects, case)
        self.store.add_audit(case_id, "case_created", {"subject": brief.subject})
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
        usage, error = self._run_agent(case, plan_message(case))
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
        backend = self._backend(case_id)
        backend.delete_direct("/execution/notes.md")
        self.store.update_case(case_id, status="EXECUTING")
        self.store.add_audit(case_id, "execution_started", {"plan_version": plan["version"]})
        usage, error = self._run_agent(case, execute_message(case, plan))
        case = self.store.update_case(
            case_id, usage={**case.get("usage", {}), "execution": usage}
        )
        if error is not None:
            report = compose_report(
                self.store, case, partial=True, exceeded_kind=error.kind
            )
            self.store.add_audit(
                case_id, "report_composed", {"partial": True, "usage": usage}
            )
            self.store.update_case(case_id, status="BUDGET_EXHAUSTED")
            log.info("case %s budget exhausted (%s)", case_id, error.kind)
            return report
        report = compose_report(self.store, case)
        self.store.add_audit(
            case_id, "report_composed", {"partial": False, "usage": usage}
        )
        self.store.update_case(case_id, status="DRAFT_READY")
        log.info("case %s draft ready", case_id)
        return report

    def decide_report(self, case_id: str, analyst: str, approve: bool, note: str = "") -> dict:
        case = self._require_case(case_id)
        if case["status"] not in ("DRAFT_READY", "BUDGET_EXHAUSTED"):
            raise GateError(f"no report awaiting decision (status {case['status']})")
        report = self.store.get_report(case_id)
        report["status"] = "APPROVED" if approve else "REJECTED"
        report["decided_by"] = analyst
        report["decision_note"] = note
        self.store.save_report(case_id, report)
        self.store.add_audit(
            case_id,
            "report_approved" if approve else "report_rejected",
            {"analyst": analyst, "note": note},
        )
        return self.store.update_case(
            case_id, status="FINAL" if approve else "REPORT_REJECTED"
        )
