"""FastAPI surface for the Deep Diligence Agent.

The console is served at /; the JSON API lives under /api. Lifecycle
transitions the case state forbids surface as 409 (the gates are the
product), unknown cases as 404, invalid briefs as 422.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .budgets import budget_view
from .config import Settings, load_settings
from .corpus import GOLD_CASES, companies
from .evidence import evidence_text
from .ledger import verify_ledger
from .logging_setup import configure_logging, new_request_id
from .models import (
    AssignIn,
    BriefIn,
    BudgetAmendIn,
    CaseOut,
    ChallengeDecisionIn,
    ConflictNoteIn,
    DecisionIn,
    DeleteCaseIn,
    EngagementAssignIn,
    EngagementIn,
    EvidenceIn,
    HoldIn,
    NoteIn,
    PriorityIn,
    ProfileEvaluateIn,
    ProfilePromoteIn,
    ProfileRollbackIn,
    QuarantineDecisionIn,
    RedactIn,
    RedactionDecisionIn,
    ReplanIn,
    ResumeIn,
    SectionNoteEditIn,
    SectionNoteIn,
    SourceIn,
    StalenessDecisionIn,
    WorkbenchViewIn,
)
from .profiles import ProfileError, list_profile_ids, load_profile
from .promotion import PromotionError
from .service import CaseService, GateError

log = logging.getLogger("deep_diligence.api")
STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


def create_app(
    settings: Settings | None = None, service: CaseService | None = None
) -> FastAPI:
    configure_logging()
    settings = settings or load_settings()
    service = service or CaseService(settings)
    app = FastAPI(title="Deep Diligence Agent — Trida AI Blueprint")
    app.state.service = service
    app.state.settings = settings

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = new_request_id()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="invalid API key")

    def get_case_or_404(case_id: str) -> dict:
        case = service.store.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")
        return case

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/readiness")
    def readiness() -> JSONResponse:
        store_ok = service.store.ping()
        objects_ok = service.objects.ping()
        ready = store_ok and objects_ok
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "ready": ready,
                "store": settings.store_backend if store_ok else "down",
                "objects": settings.object_backend if objects_ok else "down",
                "model": settings.llm_backend,
            },
        )

    @app.get("/")
    def console() -> FileResponse:
        return FileResponse(STATIC_INDEX)

    @app.get("/api/corpus")
    def corpus() -> dict:
        return {
            "companies": companies(),
            "sample_briefs": [
                {"key": gold["key"], "brief": gold["brief"]} for gold in GOLD_CASES
            ],
        }

    @app.post("/api/cases", status_code=201, dependencies=[Depends(require_api_key)])
    def create_case(brief: BriefIn) -> dict:
        try:
            case = service.create_case(brief)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return CaseOut(**case).model_dump()

    @app.get("/api/cases")
    def list_cases() -> list[dict]:
        return [CaseOut(**c).model_dump() for c in service.store.list_cases()]

    # Declared before /api/cases/{case_id} so the literal path is
    # not read as a case id.
    @app.get("/api/cases/interrupted")
    def list_interrupted() -> list[dict]:
        # Cases an interrupted execution left in EXECUTING: what
        # had completed, what is still owed, and the checkpoint the
        # resume will read. Listing is also the detection point —
        # the first scan records the interruption in the case's
        # audit trail.
        return [
            {
                "case_id": entry["case"]["case_id"],
                "subject": entry["case"]["subject"],
                "status": entry["case"]["status"],
                "checkpoint": entry["checkpoint"],
                "completed_questions": entry["completed_questions"],
                "pending_questions": entry["pending_questions"],
            }
            for entry in service.interrupted_cases()
        ]

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str) -> dict:
        return CaseOut(**get_case_or_404(case_id)).model_dump()

    @app.post("/api/cases/{case_id}/plan", dependencies=[Depends(require_api_key)])
    def run_planning(case_id: str) -> dict:
        get_case_or_404(case_id)
        try:
            return service.run_planning(case_id)
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/cases/{case_id}/plan")
    def get_plan(case_id: str) -> dict:
        get_case_or_404(case_id)
        plan = service.store.get_plan(case_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="no plan yet")
        return plan

    @app.post("/api/cases/{case_id}/plan/decision", dependencies=[Depends(require_api_key)])
    def decide_plan(case_id: str, decision: DecisionIn) -> dict:
        get_case_or_404(case_id)
        try:
            case = service.decide_plan(
                case_id, decision.analyst, decision.approve, decision.note
            )
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CaseOut(**case).model_dump()

    @app.post("/api/cases/{case_id}/execute", dependencies=[Depends(require_api_key)])
    def run_execution(case_id: str) -> dict:
        get_case_or_404(case_id)
        try:
            return service.run_execution(case_id)
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/cases/{case_id}/evidence")
    def list_evidence(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        return service.store.list_evidence(case_id)

    @app.get("/api/cases/{case_id}/evidence/{evidence_id}")
    def get_evidence(case_id: str, evidence_id: str) -> dict:
        get_case_or_404(case_id)
        ev = service.store.get_evidence(case_id, evidence_id)
        if ev is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        return {**ev, "text": evidence_text(service.objects, ev)}

    @app.get("/api/cases/{case_id}/findings")
    def list_findings(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        return service.store.list_findings(case_id)

    @app.get("/api/cases/{case_id}/ledger")
    def get_ledger(case_id: str) -> dict:
        get_case_or_404(case_id)
        return {
            "case_id": case_id,
            "entries": service.store.list_ledger(case_id),
            # Verification is computed live on read: the hashes in the
            # ledger against the evidence bytes as they are now.
            "verification": verify_ledger(
                service.store, service.objects, case_id
            ),
        }

    @app.get("/api/cases/{case_id}/report")
    def get_report(case_id: str) -> dict:
        get_case_or_404(case_id)
        report = service.store.get_report(case_id)
        if report is None:
            raise HTTPException(status_code=404, detail="no report yet")
        return report

    @app.get("/api/cases/{case_id}/report/html")
    def get_report_html(case_id: str) -> HTMLResponse:
        # The print-ready rendering: self-contained HTML (inline
        # CSS, no external assets), rendered at finalize time from
        # the same structured model as the markdown.
        get_case_or_404(case_id)
        try:
            return HTMLResponse(content=service.report_html(case_id))
        except GateError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/cases/{case_id}/report/export")
    def get_report_export(case_id: str) -> dict:
        # The JSON export: the report's structured model (questions,
        # claims with citations and support grades, conflicts,
        # coverage, risks and open items, references) plus the
        # analyst's section annotations in their own field.
        get_case_or_404(case_id)
        try:
            return service.report_export(case_id)
        except GateError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/cases/{case_id}/report/notes")
    def list_report_notes(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        return service.store.list_section_notes(case_id)

    @app.post(
        "/api/cases/{case_id}/report/notes",
        status_code=201,
        dependencies=[Depends(require_api_key)],
    )
    def add_report_note(case_id: str, note: SectionNoteIn) -> dict:
        get_case_or_404(case_id)
        try:
            return service.add_report_section_note(
                case_id, note.analyst, note.section, note.text
            )
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put(
        "/api/cases/{case_id}/report/notes/{note_id}",
        dependencies=[Depends(require_api_key)],
    )
    def edit_report_note(
        case_id: str, note_id: str, note: SectionNoteEditIn
    ) -> dict:
        get_case_or_404(case_id)
        try:
            return service.edit_report_section_note(
                case_id, note_id, note.analyst, note.text
            )
        except KeyError:
            raise HTTPException(
                status_code=404, detail="section note not found"
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/cases/{case_id}/verification")
    def get_verification(case_id: str) -> dict:
        get_case_or_404(case_id)
        result = service.store.get_verifier_report(case_id)
        if result is None:
            raise HTTPException(status_code=404, detail="no verification yet")
        return result

    @app.get("/api/cases/{case_id}/quarantine")
    def list_quarantine(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        return service.store.list_quarantine(case_id)

    @app.post(
        "/api/cases/{case_id}/quarantine/{quarantine_id}/decision",
        dependencies=[Depends(require_api_key)],
    )
    def decide_quarantine(
        case_id: str, quarantine_id: str, decision: QuarantineDecisionIn
    ) -> dict:
        get_case_or_404(case_id)
        try:
            record = service.decide_quarantine(
                case_id,
                quarantine_id,
                decision.analyst,
                decision.action,
                decision.note,
            )
        except KeyError:
            raise HTTPException(
                status_code=404, detail="quarantine record not found"
            ) from None
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        case = service.store.get_case(case_id)
        return {"quarantine": record, "case_status": case["status"]}

    @app.get("/api/cases/{case_id}/conflicts")
    def list_conflicts(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        return service.store.list_conflicts(case_id)

    @app.post(
        "/api/cases/{case_id}/conflicts/{conflict_id}/decision",
        dependencies=[Depends(require_api_key)],
    )
    def decide_conflict(
        case_id: str, conflict_id: str, decision: ConflictNoteIn
    ) -> dict:
        get_case_or_404(case_id)
        try:
            record = service.decide_conflict(
                case_id,
                conflict_id,
                decision.analyst,
                decision.note,
            )
        except KeyError:
            raise HTTPException(
                status_code=404, detail="conflict record not found"
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        case = service.store.get_case(case_id)
        return {"conflict": record, "case_status": case["status"]}

    @app.get("/api/cases/{case_id}/challenges")
    def list_challenges(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        return service.store.list_challenges(case_id)

    @app.post(
        "/api/cases/{case_id}/challenges/{challenge_id}/decision",
        dependencies=[Depends(require_api_key)],
    )
    def decide_challenge(
        case_id: str, challenge_id: str, decision: ChallengeDecisionIn
    ) -> dict:
        get_case_or_404(case_id)
        try:
            record = service.decide_challenge(
                case_id,
                challenge_id,
                decision.analyst,
                decision.action,
                decision.note,
                decision.counter_evidence_ids,
            )
        except KeyError:
            raise HTTPException(
                status_code=404, detail="challenge record not found"
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        case = service.store.get_case(case_id)
        return {"challenge": record, "case_status": case["status"]}

    @app.post("/api/cases/{case_id}/report/decision", dependencies=[Depends(require_api_key)])
    def decide_report(case_id: str, decision: DecisionIn) -> dict:
        get_case_or_404(case_id)
        try:
            case = service.decide_report(
                case_id, decision.analyst, decision.approve, decision.note
            )
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CaseOut(**case).model_dump()

    @app.get("/api/cases/{case_id}/audit")
    def get_audit(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        return service.store.list_audit(case_id)

    @app.get("/api/cases/{case_id}/timeline")
    def get_timeline(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        # The case's story in one view: the audit trail merged
        # with the analyst's notes, gate decisions and state
        # changes in chronological order — rendered from the
        # records, nothing re-recorded to feed it.
        return service.case_timeline(case_id)

    @app.post(
        "/api/cases/{case_id}/assignment",
        dependencies=[Depends(require_api_key)],
    )
    def assign_case(case_id: str, request: AssignIn) -> dict:
        get_case_or_404(case_id)
        case = service.assign_case(case_id, request.analyst, request.assignee)
        return CaseOut(**case).model_dump()

    @app.post(
        "/api/cases/{case_id}/priority",
        dependencies=[Depends(require_api_key)],
    )
    def set_priority(case_id: str, request: PriorityIn) -> dict:
        get_case_or_404(case_id)
        try:
            case = service.set_priority(
                case_id, request.analyst, request.priority
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return CaseOut(**case).model_dump()

    @app.post(
        "/api/cases/{case_id}/engagement",
        dependencies=[Depends(require_api_key)],
    )
    def set_case_engagement(
        case_id: str, request: EngagementAssignIn
    ) -> dict:
        # Membership is a recorded act on both records (the
        # case's audit trail, the engagement's history). It is
        # triage, not a decision: it moves no gate, budget,
        # evidence, or retention state.
        get_case_or_404(case_id)
        try:
            case = service.assign_case_engagement(
                case_id, request.engagement_id, request.analyst
            )
        except KeyError:
            raise HTTPException(
                status_code=404, detail="engagement not found"
            ) from None
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CaseOut(**case).model_dump()

    @app.get("/api/cases/{case_id}/workstreams")
    def get_workstreams(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        # Per-workstream execution status (completed / empty /
        # failed / interrupted / not_run, with the reason where one
        # applies): the report's record once composed, else folded
        # from the audit trail's start/finish events.
        return service.workstream_statuses(case_id)

    @app.get("/api/cases/{case_id}/budget")
    def get_budget(case_id: str) -> dict:
        case = get_case_or_404(case_id)
        # Granted comes from the case's current budget; consumed is
        # folded from the budget ledger in the audit trail.
        return budget_view(case, service.store.list_audit(case_id))

    @app.get("/api/cases/{case_id}/cost")
    def get_cost(case_id: str) -> dict:
        get_case_or_404(case_id)
        # Budget-vs-actual joined with the cost ledger: granted vs
        # consumed from the budget ledger, and the priced cost —
        # totals and breakdowns by phase, sub-agent role, and
        # model — folded from the cost lines recorded when each
        # run ended, each pinned to its price-table version.
        return service.cost_report(case_id)

    @app.get("/api/costs")
    def get_costs() -> dict:
        # Every case's recorded cost and the desk's total: a sum
        # of the per-case ledgers, each of which keeps its own
        # detail at /api/cases/{id}/cost.
        return service.cost_totals()

    @app.post(
        "/api/cases/{case_id}/budget/amendment",
        dependencies=[Depends(require_api_key)],
    )
    def amend_budget(case_id: str, amendment: BudgetAmendIn) -> dict:
        get_case_or_404(case_id)
        try:
            case = service.amend_budget(
                case_id,
                amendment.analyst,
                amendment.reason,
                amendment.budget,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CaseOut(**case).model_dump()

    # -- long-case continuity ------------------------------------------------
    @app.post(
        "/api/cases/{case_id}/resume", dependencies=[Depends(require_api_key)]
    )
    def resume_execution(case_id: str, request: ResumeIn) -> dict:
        get_case_or_404(case_id)
        try:
            return service.resume_execution(case_id, request.analyst)
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/cases/{case_id}/checkpoint")
    def get_checkpoint(case_id: str) -> dict:
        get_case_or_404(case_id)
        checkpoint = service.store.get_checkpoint(case_id)
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="no checkpoint yet")
        return checkpoint

    @app.post(
        "/api/cases/{case_id}/evidence",
        status_code=201,
        dependencies=[Depends(require_api_key)],
    )
    def add_evidence(case_id: str, document: EvidenceIn) -> dict:
        get_case_or_404(case_id)
        try:
            result = service.add_case_evidence(
                case_id,
                document.analyst,
                title=document.title,
                kind=document.kind,
                source=document.source,
                doc_date=document.doc_date,
                text=document.text,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "evidence": result["evidence"],
            "stale_questions": result["stale_questions"],
        }

    @app.get("/api/cases/{case_id}/notes")
    def list_notes(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        return service.store.list_notes(case_id)

    @app.post(
        "/api/cases/{case_id}/notes",
        status_code=201,
        dependencies=[Depends(require_api_key)],
    )
    def add_note(case_id: str, note: NoteIn) -> dict:
        get_case_or_404(case_id)
        try:
            return service.add_case_note(case_id, note.analyst, note.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/api/cases/{case_id}/staleness/decision",
        dependencies=[Depends(require_api_key)],
    )
    def decide_staleness(case_id: str, decision: StalenessDecisionIn) -> dict:
        get_case_or_404(case_id)
        try:
            case = service.decide_staleness(
                case_id, decision.analyst, decision.action, decision.note
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CaseOut(**case).model_dump()

    @app.post(
        "/api/cases/{case_id}/replan", dependencies=[Depends(require_api_key)]
    )
    def replan_case(case_id: str, request: ReplanIn) -> dict:
        get_case_or_404(case_id)
        try:
            case = service.replan_case(case_id, request.analyst, request.note)
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CaseOut(**case).model_dump()

    @app.get("/api/cases/{case_id}/report/versions")
    def list_report_versions(case_id: str) -> list[dict]:
        get_case_or_404(case_id)
        # Drafts a plan revision superseded, preserved when the
        # revision was requested — the current report lives at
        # /report; these are its predecessors.
        return [
            {
                "snapshot_id": s["snapshot_id"],
                "reason": s["reason"],
                "analyst": s["analyst"],
                "created_at": s["created_at"],
                "report_status": s["report"]["status"],
                "partial": s["report"]["partial"],
                "composed_at": s["report"]["composed_at"],
                "markdown": s["report"]["markdown"],
            }
            for s in service.store.list_report_snapshots(case_id)
        ]

    # -- source governance ---------------------------------------------------
    @app.get("/api/sources")
    def list_sources() -> list[dict]:
        # The governed registry: every source evidence may cite,
        # with its classification and its registration record.
        return service.store.list_sources()

    @app.post(
        "/api/sources", status_code=201, dependencies=[Depends(require_api_key)]
    )
    def register_source(source: SourceIn) -> dict:
        try:
            return service.register_source(
                source.analyst,
                name=source.name,
                kind=source.kind,
                owner=source.owner,
                usage_basis=source.usage_basis,
                notes=source.notes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/retention/sweep")
    def retention_sweep() -> list[dict]:
        # Cases past their retention, listed for an analyst's
        # decision. The sweep is a report: it deletes nothing.
        return service.retention_sweep()

    @app.get("/api/retention/tombstones")
    def list_tombstones() -> list[dict]:
        # All that survives deleted cases: that they existed, and
        # who ended them, and why.
        return service.store.list_tombstones()

    @app.get("/api/cases/{case_id}/sources")
    def get_case_sources(case_id: str) -> dict:
        get_case_or_404(case_id)
        # The governance records behind the case's sources: the
        # brief's allowed entries resolved, and the sources the
        # evidence actually came from.
        return service.case_sources(case_id)

    @app.get("/api/cases/{case_id}/provenance")
    def get_provenance(case_id: str) -> dict:
        get_case_or_404(case_id)
        # Every evidence item with its source's governance record,
        # its hash, its ingestion path, and the findings and report
        # claims that cite it.
        return service.case_provenance(case_id)

    @app.post(
        "/api/cases/{case_id}/evidence/{evidence_id}/redact",
        dependencies=[Depends(require_api_key)],
    )
    def redact_evidence(
        case_id: str, evidence_id: str, request: RedactIn
    ) -> dict:
        get_case_or_404(case_id)
        try:
            return service.redact_evidence(
                case_id, evidence_id, request.analyst, request.reason
            )
        except KeyError:
            raise HTTPException(
                status_code=404, detail="evidence not found"
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/cases/{case_id}/redactions")
    def list_redactions(case_id: str) -> dict:
        get_case_or_404(case_id)
        return {
            "redactions": service.store.list_redactions(case_id),
            "flags": service.store.list_redaction_flags(case_id),
        }

    @app.post(
        "/api/cases/{case_id}/redaction-flags/{flag_id}/decision",
        dependencies=[Depends(require_api_key)],
    )
    def decide_redaction_flag(
        case_id: str, flag_id: str, decision: RedactionDecisionIn
    ) -> dict:
        get_case_or_404(case_id)
        try:
            flag = service.decide_redaction_flag(
                case_id,
                flag_id,
                decision.analyst,
                decision.action,
                decision.note,
            )
        except KeyError:
            raise HTTPException(
                status_code=404, detail="redaction flag not found"
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        case = service.store.get_case(case_id)
        return {"flag": flag, "case_status": case["status"]}

    @app.post(
        "/api/cases/{case_id}/retention/hold",
        dependencies=[Depends(require_api_key)],
    )
    def set_legal_hold(case_id: str, request: HoldIn) -> dict:
        get_case_or_404(case_id)
        try:
            case = service.set_legal_hold(
                case_id,
                request.analyst,
                request.action == "set",
                request.reason,
            )
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CaseOut(**case).model_dump()

    @app.post(
        "/api/cases/{case_id}/delete", dependencies=[Depends(require_api_key)]
    )
    def delete_case(case_id: str, request: DeleteCaseIn) -> dict:
        get_case_or_404(case_id)
        try:
            tombstone = service.delete_case(
                case_id, request.analyst, request.reason
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"tombstone": tombstone}

    # -- analyst workbench -------------------------------------------------
    # The desk's working view, derived from the case records on
    # every read. Note what is NOT here: no bulk decision route.
    # Assignment and priority are triage; every gate decision
    # stays one case, one analyst, on the case's own endpoints.
    @app.get("/api/workbench/queue")
    def workbench_queue(
        act: str | None = None,
        assignee: str | None = None,
        priority: str | None = None,
    ) -> dict:
        filters = {"act": act, "assignee": assignee, "priority": priority}
        try:
            return service.workbench_queue(filters)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/workbench/search")
    def workbench_search(
        q: str = "", engagement_id: str | None = None
    ) -> list[dict]:
        # With an engagement named, the search is scoped to its
        # member cases; hits stay case-scoped and case-labelled
        # either way.
        try:
            return service.search_cases(q, engagement_id=engagement_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail="engagement not found"
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/workbench/views")
    def list_views(analyst: str | None = None) -> list[dict]:
        return service.list_views(analyst)

    @app.post(
        "/api/workbench/views",
        status_code=201,
        dependencies=[Depends(require_api_key)],
    )
    def save_view(view: WorkbenchViewIn) -> dict:
        try:
            return service.save_view(view.analyst, view.name, view.filters)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete(
        "/api/workbench/views/{view_id}",
        dependencies=[Depends(require_api_key)],
    )
    def delete_view(view_id: str, analyst: str) -> dict:
        try:
            view = service.delete_view(analyst, view_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail="saved view not found"
            ) from None
        return {"deleted": view["view_id"], "name": view["name"]}

    # -- portfolio mode ------------------------------------------------------
    # Engagements group cases under a client mandate. These
    # routes create the grouping and read the fold over it —
    # note what is NOT here, by design: no portfolio-level
    # decision, approval, or deletion route. Every gate stays
    # one case, one analyst, on the case's own endpoints.
    @app.post(
        "/api/engagements",
        status_code=201,
        dependencies=[Depends(require_api_key)],
    )
    def create_engagement(request: EngagementIn) -> dict:
        try:
            return service.create_engagement(
                request.analyst, request.name, request.client, request.owner
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/engagements")
    def list_engagements() -> list[dict]:
        return service.store.list_engagements()

    @app.get("/api/engagements/{engagement_id}")
    def get_engagement(engagement_id: str) -> dict:
        record = service.store.get_engagement(engagement_id)
        if record is None:
            raise HTTPException(status_code=404, detail="engagement not found")
        return record

    @app.get("/api/portfolio")
    def list_portfolios() -> list[dict]:
        # Every engagement's summary: counts by state, open
        # gates with the oldest wait, the cost total, and the
        # held / past-retention member counts.
        return service.list_portfolios()

    @app.get("/api/portfolio/{engagement_id}")
    def get_portfolio(engagement_id: str) -> dict:
        # The full fold: member cases with grades and retention
        # standing, the portfolio's gates, open conflicts and
        # quarantined claims by id, and the cost roll-up.
        try:
            return service.portfolio_report(engagement_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail="engagement not found"
            ) from None

    # -- profile regression gate -------------------------------------------
    # The active profile is read from the promotion log; the only
    # writes are an evaluation (evidence), a promotion (decision),
    # and a rollback (decision). There is deliberately no endpoint
    # that sets the active profile directly.
    @app.get("/api/profiles")
    def get_profiles() -> dict:
        profiles = []
        for profile_id in list_profile_ids(settings.profile_dir):
            try:
                profiles.append(load_profile(profile_id, settings.profile_dir).record())
            except ProfileError as exc:
                profiles.append({"id": profile_id, "error": str(exc)})
        return {**service.active_profile_info(), "profiles": profiles}

    @app.get("/api/profiles/evaluations")
    def list_evaluations() -> list[dict]:
        return service.store.list_evaluations()

    @app.get("/api/profiles/promotions")
    def list_promotions() -> list[dict]:
        return service.store.list_promotions()

    @app.post(
        "/api/profiles/evaluate", dependencies=[Depends(require_api_key)]
    )
    def evaluate_profile(request: ProfileEvaluateIn) -> dict:
        try:
            return service.evaluate_profile(request.candidate_id, request.analyst)
        except ProfileError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except PromotionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/profiles/promote", dependencies=[Depends(require_api_key)])
    def promote_profile(request: ProfilePromoteIn) -> dict:
        try:
            record = service.promote_profile(request.evaluation_id, request.analyst)
        except KeyError:
            raise HTTPException(
                status_code=404, detail="evaluation not found"
            ) from None
        except PromotionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"promotion": record, "active": service.profile.record()}

    @app.post("/api/profiles/rollback", dependencies=[Depends(require_api_key)])
    def rollback_profile(request: ProfileRollbackIn) -> dict:
        try:
            record = service.rollback_profile(
                request.analyst, request.note or None
            )
        except PromotionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"promotion": record, "active": service.profile.record()}

    return app


app = create_app()
