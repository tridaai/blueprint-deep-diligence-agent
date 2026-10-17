"""Budget enforcement — in our code, around the harness.

A BudgetGuard rides on the agent run as a LangChain callback handler.
It counts model steps, tokens (from response usage metadata, with a
character-based estimate when a provider omits it), sub-agent spawns
(the harness's `task` tool), and wall-clock time. The moment a cap is
crossed it raises BudgetExceededError, which aborts the run; the
findings recorded so far stay in the case store and the composer
produces a partial report that names what was not covered. Caps are
never enforced by prompt.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class BudgetExceededError(RuntimeError):
    def __init__(self, kind: str, limit: int, observed: int):
        super().__init__(f"budget exceeded: {kind} (limit {limit}, observed {observed})")
        self.kind = kind
        self.limit = limit
        self.observed = observed


class BudgetGuard(BaseCallbackHandler):
    # Raise callback errors into the run instead of swallowing them.
    raise_error = True

    def __init__(self, budget: dict, audit_sink: list | None = None):
        self.max_steps = int(budget["max_steps"])
        self.max_subagents = int(budget["max_subagents"])
        self.max_tokens = int(budget["max_tokens"])
        self.max_wall_seconds = int(budget["max_wall_seconds"])
        self.audit_sink = audit_sink if audit_sink is not None else []
        self.started = time.monotonic()
        self.steps = 0
        self.subagents = 0
        self.tokens = 0
        self.tool_calls = 0
        self.exceeded: str | None = None
        self._seen_model_runs: set[str] = set()

    # -- helpers ------------------------------------------------------------
    def _fail(self, kind: str, limit: int, observed: int) -> None:
        self.exceeded = kind
        self.audit_sink.append(
            ("budget_exceeded", {"kind": kind, "limit": limit, "observed": observed})
        )
        raise BudgetExceededError(kind, limit, observed)

    def _check_wall(self) -> None:
        elapsed = int(time.monotonic() - self.started)
        if elapsed > self.max_wall_seconds:
            self._fail("wall_seconds", self.max_wall_seconds, elapsed)

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
            self.audit_sink.append(
                ("subagent_spawned", {"spawn_index": self.subagents})
            )
            if self.subagents > self.max_subagents:
                self._fail("subagents", self.max_subagents, self.subagents)
        self._check_wall()

    def usage(self) -> dict:
        return {
            "steps": self.steps,
            "subagents": self.subagents,
            "tokens": self.tokens,
            "tool_calls": self.tool_calls,
            "wall_seconds": round(time.monotonic() - self.started, 2),
            "exceeded": self.exceeded,
        }
