"""FastAPI surface for the Deep Diligence Agent.

The console is served at /; the JSON API lives under /api. Lifecycle
transitions the case state forbids surface as 409 (the gates are the
product), unknown cases as 404, invalid briefs as 422.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .budgets import budget_view
from .config import Settings, load_settings
from .corpus import GOLD_CASES, companies
from .evidence import evidence_text
from .ledger import verify_ledger
from .logging_setup import configure_logging, new_request_id
from .models import (
    BriefIn,
    BudgetAmendIn,
    CaseOut,
    ConflictNoteIn,
    DecisionIn,
    ProfileEvaluateIn,
    ProfilePromoteIn,
    ProfileRollbackIn,
    QuarantineDecisionIn,
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

    @app.get("/api/cases/{case_id}/budget")
    def get_budget(case_id: str) -> dict:
        case = get_case_or_404(case_id)
        # Granted comes from the case's current budget; consumed is
        # folded from the budget ledger in the audit trail.
        return budget_view(case, service.store.list_audit(case_id))

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
