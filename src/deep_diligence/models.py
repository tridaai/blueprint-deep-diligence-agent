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
    # Question ids within this brief whose workstream must finish
    # before this question's workstream starts (schedule.py). The
    # planner carries the declaration onto the plan, and the plan
    # gate validates the resulting graph before approval.
    depends_on: list[str] = Field(default_factory=list, max_length=6)


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


class EvidenceIn(BaseModel):
    """A document an analyst adds to a case mid-flight. It becomes
    case evidence under the same rules as the intake corpus; the
    free-text body is the document, stored byte for byte."""
    analyst: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=2, max_length=200)
    kind: str = Field(min_length=2, max_length=40)
    source: str = Field(min_length=2, max_length=200)
    doc_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    text: str = Field(min_length=1, max_length=200_000)


class NoteIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=4000)


class SectionNoteIn(BaseModel):
    """An analyst's annotation on one section of a case's report.
    Margin commentary only: it is stored apart from the report and
    surfaces solely in the JSON export's annotations field."""
    analyst: str = Field(min_length=1, max_length=120)
    section: str = Field(min_length=1, max_length=60)
    text: str = Field(min_length=1, max_length=4000)


class SectionNoteEditIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=4000)


class ResumeIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)


class StalenessDecisionIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    # The only two answers to marked staleness: re-plan around the
    # new evidence, or proceed with the staleness on the record.
    action: Literal["proceed", "replan"]
    note: str = Field(default="", max_length=2000)


class ReplanIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)


class SourceIn(BaseModel):
    """Registering a source is a recorded act: who registered it
    is part of the record, alongside the classification."""
    analyst: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=2, max_length=200)
    kind: Literal["filing", "news", "internal dataset", "web"]
    owner: str = Field(min_length=1, max_length=200)
    usage_basis: Literal["public", "licensed", "client-provided"]
    notes: str = Field(default="", max_length=2000)


class RedactIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    # A redaction without a recorded reason is not a redaction;
    # the reason is part of the redaction record, permanently.
    reason: str = Field(min_length=1, max_length=2000)


class RedactionDecisionIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    # A redaction-flagged claim has exactly two analyst moves —
    # acknowledge it, or send it back for re-investigation. There
    # is no "approve over it" action, by design.
    action: Literal["acknowledge", "send_back"]
    note: str = Field(default="", max_length=2000)


class HoldIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    action: Literal["set", "clear"]
    reason: str = Field(default="", max_length=2000)


class DeleteCaseIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=120)
    # Deletion is a recorded act; the reason lands on the
    # tombstone, which is all of the case that survives.
    reason: str = Field(min_length=1, max_length=2000)


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
    # Long-case continuity: workstreams marked stale by evidence
    # that arrived mid-case and awaits an analyst decision, and the
    # decisions recorded on them. Empty on a case nothing arrived
    # late for.
    stale_marks: list[dict[str, Any]] = Field(default_factory=list)
    staleness_decisions: list[dict[str, Any]] = Field(default_factory=list)
    # Source governance: the case's retention class and the record
    # of hold acts on it. A case under legal_hold refuses deletion.
    retention_class: str = "standard"
    retention_history: list[dict[str, Any]] = Field(default_factory=list)
    # The harness profile (id, version, target model) stamped on the
    # case at intake — the profile that ran its planning and
    # execution. Empty for cases created before profiles existed.
    profile: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
