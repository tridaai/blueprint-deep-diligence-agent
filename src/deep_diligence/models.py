"""Pydantic schemas for the API boundary and the case record."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

WORKSTREAMS = ("financial-scan", "product-scan", "market-scan")

# Case lifecycle. Execution is impossible before PLAN_APPROVED; a report
# is a draft until REPORT decision approves it.
STATUSES = (
    "INTAKE",            # created, no plan yet
    "PLAN_READY",        # plan artifact stored, awaiting the analyst
    "PLAN_REJECTED",     # analyst sent the plan back; may re-plan
    "APPROVED",          # plan approved; execution may start
    "EXECUTING",         # run in flight (transient inside one request)
    "DRAFT_READY",       # report draft composed, awaiting the analyst
    "BUDGET_EXHAUSTED",  # run stopped on a cap; partial draft composed
    "FINAL",             # report approved
    "REPORT_REJECTED",   # analyst sent the draft back; may re-execute
)


class PhaseBudgetIn(BaseModel):
    """Per-phase overrides for the two caps a single run consumes on
    its own. A field left unset derives from the brief's flat value,
    so a brief written before phases existed behaves exactly as it
    did. Sub-agent spawns and wall clock are not phase caps: they
    are case-level, consumed across both phases (see budgets.py)."""

    max_steps: int | None = Field(default=None, ge=1, le=2000)
    max_tokens: int | None = Field(default=None, ge=1, le=10_000_000)


class BudgetIn(BaseModel):
    max_steps: int = Field(default=80, ge=1, le=2000)
    max_subagents: int = Field(default=8, ge=0, le=64)
    max_tokens: int = Field(default=250_000, ge=1, le=10_000_000)
    max_wall_seconds: int = Field(default=900, ge=5, le=86_400)
    planning: PhaseBudgetIn = Field(default_factory=PhaseBudgetIn)
    execution: PhaseBudgetIn = Field(default_factory=PhaseBudgetIn)


class QuestionIn(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    text: str = Field(min_length=8, max_length=600)
    workstream: Literal["financial-scan", "product-scan", "market-scan"]


class BriefIn(BaseModel):
    subject: str = Field(min_length=2, max_length=200)
    subject_ref: str = Field(min_length=2, max_length=80)
    analyst: str = Field(min_length=1, max_length=120)
    questions: list[QuestionIn] = Field(min_length=1, max_length=6)
    allowed_sources: list[str] = Field(default_factory=list, max_length=40)
    budget: BudgetIn = Field(default_factory=BudgetIn)

    @field_validator("questions")
    @classmethod
    def _unique_question_ids(cls, value: list[QuestionIn]) -> list[QuestionIn]:
        ids = [q.id for q in value]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return value


class DecisionIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    approve: bool
    note: str = Field(default="", max_length=2000)


class BudgetAmendIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    # An amendment without a recorded reason is not an amendment;
    # the reason is part of the budget ledger, permanently.
    reason: str = Field(min_length=1, max_length=2000)
    budget: BudgetIn


class ConflictNoteIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    # Marking a conflict NOTED is the analyst's only move on it —
    # there is no resolve or delete — and it requires the note that
    # says what the analyst made of the disagreement.
    note: str = Field(min_length=1, max_length=2000)


class QuarantineDecisionIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    # A quarantined claim has exactly two analyst moves; there is no
    # "approve into the report" action, by design.
    action: Literal["reinvestigate", "drop"]
    note: str = Field(default="", max_length=2000)


class ProfileEvaluateIn(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=120)
    analyst: str = Field(min_length=1, max_length=120)


class ProfilePromoteIn(BaseModel):
    evaluation_id: str = Field(min_length=1, max_length=120)
    analyst: str = Field(min_length=1, max_length=120)


class ProfileRollbackIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)


class CaseOut(BaseModel):
    case_id: str
    status: str
    subject: str
    subject_ref: str
    analyst: str
    brief: dict[str, Any]
    usage: dict[str, Any] = Field(default_factory=dict)
    # The harness profile (id, version, target model) stamped on the
    # case at intake — the profile that ran its planning and
    # execution. Empty for cases created before profiles existed.
    profile: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
