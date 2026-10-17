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
from .composer import apply_conflicts, apply_verification, compose_report
from .conflicts import detect_conflicts
from .config import Settings
from .corpus import companies
from .evidence import ingest_case_evidence
from .harness import build_agent, execute_message, plan_message
from .ledger import verify_ledger
from .llm import build_model
from .models import BriefIn
from .objects import ObjectStore, build_object_store
from .store import Store, build_store, utcnow
from .tools import build_case_tools
from .verifier import verify_case

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
                self.store, self.objects, case, partial=True, exceeded_kind=error.kind
            )
            self.store.add_audit(
                case_id, "report_composed", {"partial": True, "usage": usage}
            )
            report = self._verify_draft(case, report)
            self.store.update_case(case_id, status="BUDGET_EXHAUSTED")
            log.info("case %s budget exhausted (%s)", case_id, error.kind)
            return report
        report = compose_report(self.store, self.objects, case)
        self.store.add_audit(
            case_id, "report_composed", {"partial": False, "usage": usage}
        )
        report = self._verify_draft(case, report)
        self.store.update_case(case_id, status="DRAFT_READY")
        log.info("case %s draft ready", case_id)
        return report

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
