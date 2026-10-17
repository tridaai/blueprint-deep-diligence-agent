"""Budget enforcement — in our code, around the harness.

A brief's budget has a phase structure. Steps and tokens are
per-phase caps: the planning run and the execution run each run
under their own resolved caps (a phase override in the brief wins;
an unset field derives from the brief's flat value, so older briefs
behave exactly as they did). Sub-agent spawns and wall clock are
case-level caps, consumed across both phases — the guard is seeded
with what the case has already spent, which it reads from the
budget ledger in the audit trail, so a second execution attempt
cannot quietly re-spend the case's spawns.

A BudgetGuard rides on the agent run as a LangChain callback
handler. It counts model steps, tokens (from response usage
metadata, with a character-based estimate when a provider omits
it), sub-agent spawns (the harness's `task` tool), and wall-clock
time. The moment a cap is crossed it raises BudgetExceededError,
which aborts the run; the findings recorded so far stay in the case
store and the composer produces a partial report that itemises
coverage per question. Caps are never enforced by prompt.

Every budget-relevant event is a ledger entry in the case's audit
trail: `budget_granted` when a phase run starts (the resolved caps
it runs under), `budget_consumed` when a run ends (what it spent),
`budget_exceeded` when a cap trips (which cap, which phase, usage
at the stop), and `budget_amended` when an analyst raises the
budget on an exhausted case. `fold_budget_ledger` folds those
entries into the case's consumption; `budget_view` renders the
per-phase granted-vs-consumed picture from the ledger, not from
mutable counters.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

PHASES = ("planning", "execution")

BUDGET_EVENTS = (
    "budget_granted",
    "budget_consumed",
    "budget_exceeded",
    "budget_amended",
)


def resolve_phase_budget(budget: dict, phase: str) -> dict:
    """The step/token caps one phase run runs under: the phase's
    override when the brief sets one, else the flat value."""
    override = budget.get(phase) or {}
    steps = override.get("max_steps")
    tokens = override.get("max_tokens")
    return {
        "max_steps": int(steps if steps is not None else budget["max_steps"]),
        "max_tokens": int(
            tokens if tokens is not None else budget["max_tokens"]
        ),
    }


def case_caps(budget: dict) -> dict:
    """The case-level caps: total sub-agent spawns and total
    wall-clock seconds across every phase run of the case."""
    return {
        "max_subagents": int(budget["max_subagents"]),
        "max_wall_seconds": int(budget["max_wall_seconds"]),
    }


class BudgetExceededError(RuntimeError):
    def __init__(self, kind: str, limit: int, observed: int, phase: str | None = None):
        super().__init__(
            f"budget exceeded: {kind} (limit {limit}, observed {observed})"
        )
        self.kind = kind
        self.limit = limit
        self.observed = observed
        self.phase = phase


class BudgetGuard(BaseCallbackHandler):
    # Raise callback errors into the run instead of swallowing them.
    raise_error = True

    def __init__(
        self,
        budget: dict,
        audit_sink: list | None = None,
        *,
        phase: str = "execution",
        prior_subagents: int = 0,
        prior_wall_seconds: float = 0.0,
    ):
        # `budget` carries both cap families, already resolved by
        # the caller: max_steps / max_tokens are this phase's caps;
        # max_subagents / max_wall_seconds are the case's caps,
        # checked against prior consumption plus this run's.
        self.phase = phase
        self.max_steps = int(budget["max_steps"])
        self.max_subagents = int(budget["max_subagents"])
        self.max_tokens = int(budget["max_tokens"])
        self.max_wall_seconds = int(budget["max_wall_seconds"])
        self.prior_subagents = int(prior_subagents)
        self.prior_wall_seconds = float(prior_wall_seconds)
        self.audit_sink = audit_sink if audit_sink is not None else []
        self.started = time.monotonic()
        self.steps = 0
        self.subagents = 0
        self.tokens = 0
        self.tool_calls = 0
        self.exceeded: str | None = None
        self._seen_model_runs: set[str] = set()
        # `task` calls started but not yet ended, run_id -> spawn
        # index. When a cap trips, the spawns still in this map are
        # the workstreams the run stopped inside.
        self._open_tasks: dict[str, int] = {}
        self.interrupted_spawns: list[int] = []

    # -- helpers ------------------------------------------------------------
    def _wall_elapsed(self) -> float:
        return self.prior_wall_seconds + (time.monotonic() - self.started)

    def _snapshot(self) -> dict:
        return {
            "steps": self.steps,
            "subagents": self.subagents,
            "tokens": self.tokens,
            "tool_calls": self.tool_calls,
            "wall_seconds": round(time.monotonic() - self.started, 2),
            "case_subagents": self.prior_subagents + self.subagents,
            "case_wall_seconds": round(self._wall_elapsed(), 2),
        }

    def _fail(self, kind: str, limit: int, observed: int) -> None:
        if self.exceeded is None:
            self.exceeded = kind
            self.interrupted_spawns = sorted(self._open_tasks.values())
        self.audit_sink.append(
            (
                "budget_exceeded",
                {
                    "phase": self.phase,
                    "kind": kind,
                    "limit": limit,
                    "observed": observed,
                    "usage": self._snapshot(),
                },
            )
        )
        raise BudgetExceededError(kind, limit, observed, phase=self.phase)

    def _check_wall(self) -> None:
        elapsed = self._wall_elapsed()
        if elapsed > self.max_wall_seconds:
            self._fail("wall_seconds", self.max_wall_seconds, int(elapsed))

    # -- LangChain callback surface -------------------------------------------
    def _count_model_start(self, kwargs: dict) -> None:
        # Chat models fire on_chat_model_start; plain LLMs fire
        # on_llm_start; some stacks fire both for one call. Deduplicate
        # on run_id so a model call is counted exactly once.
        run_id = str(kwargs.get("run_id") or "")
        if run_id:
            if run_id in self._seen_model_runs:
                return
            self._seen_model_runs.add(run_id)
        self.steps += 1
        if self.steps > self.max_steps:
            self._fail("steps", self.max_steps, self.steps)
        self._check_wall()

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs: Any) -> None:
        self._count_model_start(kwargs)

    def on_chat_model_start(
        self, serialized: dict, messages: list, **kwargs: Any
    ) -> None:
        self._count_model_start(kwargs)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        total = 0
        for generations in getattr(response, "generations", []) or []:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage and usage.get("total_tokens"):
                    total += int(usage["total_tokens"])
                elif message is not None:
                    content = getattr(message, "content", "") or ""
                    total += max(1, len(str(content)) // 4)
        self.tokens += total
        if self.tokens > self.max_tokens:
            self._fail("tokens", self.max_tokens, self.tokens)

    def on_tool_start(
        self, serialized: dict, input_str: str, **kwargs: Any
    ) -> None:
        self.tool_calls += 1
        name = (serialized or {}).get("name", "")
        if name == "task":
            self.subagents += 1
            run_id = str(kwargs.get("run_id") or "")
            if run_id:
                self._open_tasks[run_id] = self.subagents
            self.audit_sink.append(
                (
                    "subagent_spawned",
                    {
                        "phase": self.phase,
                        "spawn_index": self.subagents,
                        "case_spawn_index": self.prior_subagents + self.subagents,
                    },
                )
            )
            case_total = self.prior_subagents + self.subagents
            if case_total > self.max_subagents:
                self._fail("subagents", self.max_subagents, case_total)
        self._check_wall()

    def _close_task(self, kwargs: dict) -> None:
        run_id = str(kwargs.get("run_id") or "")
        self._open_tasks.pop(run_id, None)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._close_task(kwargs)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self._close_task(kwargs)

    def usage(self) -> dict:
        return {
            "phase": self.phase,
            "steps": self.steps,
            "subagents": self.subagents,
            "tokens": self.tokens,
            "tool_calls": self.tool_calls,
            "wall_seconds": round(time.monotonic() - self.started, 2),
            "exceeded": self.exceeded,
            "interrupted_spawns": list(self.interrupted_spawns),
        }


# -- the budget ledger --------------------------------------------------------


def _empty_consumed() -> dict:
    return {"steps": 0, "tokens": 0, "subagents": 0, "wall_seconds": 0.0}


def fold_budget_ledger(audit: list[dict]) -> dict:
    """Fold a case's audit trail into its budget consumption.

    Consumption comes only from `budget_consumed` checkpoints —
    one per phase run, written when the run ends — so the numbers
    are a fold of the ledger, never a mutable counter that could
    drift from it. Steps and tokens accumulate per phase; spawns
    and wall clock accumulate per phase and case-wide.
    """
    phases = {
        phase: {"consumed": _empty_consumed(), "runs": 0} for phase in PHASES
    }
    case_consumed = {"subagents": 0, "wall_seconds": 0.0}
    exhaustions: list[dict] = []
    amendments: list[dict] = []
    events: list[dict] = []
    for entry in audit:
        event = entry["event"]
        if event not in BUDGET_EVENTS:
            continue
        payload = entry.get("payload") or {}
        events.append({"ts": entry["ts"], "event": event, "payload": payload})
        if event == "budget_consumed":
            phase = payload.get("phase")
            usage = payload.get("usage") or {}
            if phase in phases:
                consumed = phases[phase]["consumed"]
                consumed["steps"] += int(usage.get("steps", 0))
                consumed["tokens"] += int(usage.get("tokens", 0))
                consumed["subagents"] += int(usage.get("subagents", 0))
                consumed["wall_seconds"] = round(
                    consumed["wall_seconds"]
                    + float(usage.get("wall_seconds", 0.0)),
                    2,
                )
                phases[phase]["runs"] += 1
            case_consumed["subagents"] += int(usage.get("subagents", 0))
            case_consumed["wall_seconds"] = round(
                case_consumed["wall_seconds"]
                + float(usage.get("wall_seconds", 0.0)),
                2,
            )
        elif event == "budget_exceeded":
            exhaustions.append({"ts": entry["ts"], **payload})
        elif event == "budget_amended":
            amendments.append({"ts": entry["ts"], **payload})
    return {
        "phases": phases,
        "case_consumed": case_consumed,
        "exhaustions": exhaustions,
        "amendments": amendments,
        "events": events,
    }


def budget_view(case: dict, audit: list[dict]) -> dict:
    """The case-level budget picture: per-phase granted (resolved
    from the case's current budget) against consumed (folded from
    the ledger), the case-level caps the same way, and the
    exhaustion and amendment history."""
    budget = case["brief"]["budget"]
    fold = fold_budget_ledger(audit)
    return {
        "budget": budget,
        "usage": case.get("usage", {}),
        "view": {
            "phases": {
                phase: {
                    "granted": resolve_phase_budget(budget, phase),
                    "consumed": fold["phases"][phase]["consumed"],
                    "runs": fold["phases"][phase]["runs"],
                }
                for phase in PHASES
            },
            "case": {
                "granted": case_caps(budget),
                "consumed": fold["case_consumed"],
            },
            "exhaustions": fold["exhaustions"],
            "amendments": fold["amendments"],
            "ledger": fold["events"],
        },
    }
