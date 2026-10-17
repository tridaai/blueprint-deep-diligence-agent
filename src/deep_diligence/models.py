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


class BudgetIn(BaseModel):
    max_steps: int = Field(default=80, ge=1, le=2000)
    max_subagents: int = Field(default=8, ge=0, le=64)
    max_tokens: int = Field(default=250_000, ge=1, le=10_000_000)
    max_wall_seconds: int = Field(default=900, ge=5, le=86_400)


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


class CaseOut(BaseModel):
    case_id: str
    status: str
    subject: str
    subject_ref: str
    analyst: str
    brief: dict[str, Any]
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
