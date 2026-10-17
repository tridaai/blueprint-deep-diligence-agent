"""Harness profiles: per-model configuration artifacts for the harness.

A profile is a versioned JSON file in ``profiles/`` that shapes how
`create_deep_agent` is built for one target model: which prompt
sections are assembled (in order) for the planner and for each
sub-agent, which of the case tools each sub-agent role may use, which
sub-agents exist, and the middleware switches (context compression,
large-result offloading). A profile changes harness configuration
only — never model weights, never governance: budgets, gates, and
citation rules live outside profiles and hold under every profile.

The active profile is configuration, not model identity: the service
loads the profile named by ``PROFILE_ID`` (default ``offline-stub``)
at startup and stamps its id + version on every case, while the model
itself is injected separately by the provider boundary (`llm.py`). A
profile can therefore be exercised with the offline stub model
standing in for its target, which is how the real-provider profile is
tested and evaluated before any promotion decision.

Validation is fail-closed: a profile file with an unknown key, a
missing section, a tool name that does not exist, a middleware switch
left unset, or a roster entry pointing at a prompt section that is
not defined refuses to load with a named `ProfileError`, and the
service refuses to start on it. Nothing falls back silently.

Relationship to the harness package's own profiles: `deepagents`
resolves its internal `HarnessProfile` from the model's identity
through a process-global registry. That cannot express "run this
build under profile X with the stub standing in", so this blueprint
keeps its own artifacts and applies them through public seams — the
prompts, tool lists, and roster it passes to `create_deep_agent`, and
caller middleware, which the harness merges into its stack by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

# The case-tool registry: the tools this blueprint owns (tools.py).
# Harness built-ins (filesystem, task, todos) come from Deep Agents
# and are not a profile's to grant or withhold.
KNOWN_TOOLS = ("search_corpus", "read_document", "record_finding")

_TOP_KEYS = {
    "id",
    "version",
    "target_model",
    "description",
    "prompt_sections",
    "planner",
    "middleware",
    "subagents",
}
_PLANNER_KEYS = {"sections"}
_MIDDLEWARE_KEYS = {
    "context_compression",
    "large_result_offloading",
    "offload_threshold_chars",
}
_ROSTER_KEYS = {"name", "description", "prompt_sections", "tools"}


class ProfileError(RuntimeError):
    """A profile that refuses to load. The message names the problem."""


@dataclass(frozen=True)
class SubagentProfile:
    name: str
    description: str
    prompt_sections: tuple[str, ...]
    tools: tuple[str, ...]


@dataclass(frozen=True)
class HarnessProfile:
    id: str
    version: int
    target_model: str
    description: str
    prompt_sections: dict[str, str]
    planner_sections: tuple[str, ...]
    middleware: dict
    subagents: tuple[SubagentProfile, ...]

    def record(self) -> dict:
        """The stamp a case and its audit trail carry for this profile."""
        return {
            "id": self.id,
            "version": self.version,
            "target_model": self.target_model,
        }

    def _assemble(self, names: tuple[str, ...], workstream: str | None) -> str:
        parts = []
        for name in names:
            text = self.prompt_sections[name]
            if workstream is not None:
                text = text.replace("{workstream}", workstream)
            parts.append(text)
        return "\n\n".join(parts)

    def planner_prompt(self) -> str:
        return self._assemble(self.planner_sections, None)

    def subagent(self, name: str) -> SubagentProfile:
        for entry in self.subagents:
            if entry.name == name:
                return entry
        raise ProfileError(f"profile {self.id!r} has no sub-agent {name!r}")

    def subagent_prompt(self, name: str) -> str:
        entry = self.subagent(name)
        return self._assemble(entry.prompt_sections, name)


def default_profiles_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "profiles"


def _fail(source: str, problem: str) -> ProfileError:
    return ProfileError(f"profile {source}: {problem}")


def _require_keys(
    obj: dict, allowed: set[str], required: set[str], source: str, where: str
) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise _fail(source, f"{where} has unknown keys: {', '.join(unknown)}")
    missing = sorted(required - set(obj))
    if missing:
        raise _fail(source, f"{where} is missing required keys: {', '.join(missing)}")


def _section_refs(value, source: str, where: str, sections: dict) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise _fail(source, f"{where} must name at least one prompt section")
    refs = []
    for name in value:
        if not isinstance(name, str) or name not in sections:
            raise _fail(
                source,
                f"{where} references prompt section {name!r}, "
                "which is not defined in prompt_sections",
            )
        refs.append(name)
    return tuple(refs)


def _validate(raw, source: str) -> HarnessProfile:
    if not isinstance(raw, dict):
        raise _fail(source, "the profile file must contain a JSON object")
    _require_keys(raw, _TOP_KEYS, _TOP_KEYS, source, "profile")
    profile_id = raw["id"]
    if not isinstance(profile_id, str) or not profile_id:
        raise _fail(source, "'id' must be a non-empty string")
    stem = Path(source).stem
    if stem != profile_id:
        raise _fail(
            source,
            f"profile id {profile_id!r} does not match its file name {stem!r}",
        )
    version = raw["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise _fail(source, "'version' must be an integer >= 1")
    target = raw["target_model"]
    if not isinstance(target, str) or not target:
        raise _fail(source, "'target_model' must be a non-empty string")
    if not isinstance(raw["description"], str):
        raise _fail(source, "'description' must be a string")
    sections = raw["prompt_sections"]
    if not isinstance(sections, dict) or not sections:
        raise _fail(source, "'prompt_sections' must be a non-empty object")
    for name, text in sections.items():
        if not isinstance(text, str) or not text.strip():
            raise _fail(source, f"prompt section {name!r} must be non-empty text")
    planner = raw["planner"]
    if not isinstance(planner, dict):
        raise _fail(source, "'planner' must be an object")
    _require_keys(planner, _PLANNER_KEYS, _PLANNER_KEYS, source, "planner")
    planner_sections = _section_refs(
        planner["sections"], source, "planner", sections
    )
    middleware = raw["middleware"]
    if not isinstance(middleware, dict):
        raise _fail(source, "'middleware' must be an object")
    _require_keys(middleware, _MIDDLEWARE_KEYS, _MIDDLEWARE_KEYS, source, "middleware")
    for switch in ("context_compression", "large_result_offloading"):
        if not isinstance(middleware[switch], bool):
            raise _fail(source, f"middleware switch {switch!r} must be true or false")
    threshold = middleware["offload_threshold_chars"]
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
        or threshold < 500
    ):
        raise _fail(source, "'offload_threshold_chars' must be an integer >= 500")
    roster = raw["subagents"]
    if not isinstance(roster, list) or not roster:
        raise _fail(source, "'subagents' must be a non-empty list")
    subagents: list[SubagentProfile] = []
    seen: set[str] = set()
    for entry in roster:
        if not isinstance(entry, dict):
            raise _fail(source, "each subagents entry must be an object")
        name = entry.get("name")
        where = f"sub-agent {name!r}" if isinstance(name, str) else "sub-agent entry"
        _require_keys(entry, _ROSTER_KEYS, _ROSTER_KEYS, source, where)
        if not isinstance(name, str) or not name:
            raise _fail(source, "each sub-agent needs a non-empty 'name'")
        if name in seen:
            raise _fail(source, f"duplicate sub-agent name {name!r}")
        seen.add(name)
        if not isinstance(entry["description"], str) or not entry["description"]:
            raise _fail(source, f"sub-agent {name!r} needs a non-empty 'description'")
        refs = _section_refs(
            entry["prompt_sections"], source, f"sub-agent {name!r}", sections
        )
        tools = entry["tools"]
        if not isinstance(tools, list) or not tools:
            raise _fail(source, f"sub-agent {name!r} must allow at least one tool")
        for tool_name in tools:
            if tool_name not in KNOWN_TOOLS:
                raise _fail(
                    source,
                    f"sub-agent {name!r} allows tool {tool_name!r}, "
                    f"which does not exist; known tools: {', '.join(KNOWN_TOOLS)}",
                )
        if len(set(tools)) != len(tools):
            raise _fail(source, f"sub-agent {name!r} lists a tool twice")
        subagents.append(
            SubagentProfile(
                name=name,
                description=entry["description"],
                prompt_sections=refs,
                tools=tuple(tools),
            )
        )
    return HarnessProfile(
        id=profile_id,
        version=version,
        target_model=target,
        description=raw["description"],
        prompt_sections=dict(sections),
        planner_sections=planner_sections,
        middleware=dict(middleware),
        subagents=tuple(subagents),
    )


def load_profile(profile_id: str, profiles_dir=None) -> HarnessProfile:
    """Load and validate one profile artifact, or refuse with a
    named ProfileError. There is no fallback profile: a service that
    cannot load its configured profile does not start."""
    directory = Path(profiles_dir) if profiles_dir else default_profiles_dir()
    path = directory / f"{profile_id}.json"
    if not path.is_file():
        raise ProfileError(
            f"profile {profile_id!r} not found: no file {path} "
            f"(profiles live in {directory}; PROFILE_ID selects one)"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"profile {path}: not valid JSON: {exc}") from exc
    return _validate(raw, str(path))


def list_profile_ids(profiles_dir=None) -> list[str]:
    directory = Path(profiles_dir) if profiles_dir else default_profiles_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


# -- the middleware the profile switches control ---------------------------


class DisabledSummarizationMiddleware(AgentMiddleware):
    """A pass-through that takes the built-in summarization
    middleware's place in the stack.

    `create_deep_agent` merges caller middleware into its base stack
    by name, replacing a same-named entry in place, and the built-in
    context-compression middleware ships under the public name
    "SummarizationMiddleware". A profile with context compression
    off passes this middleware, so the assembled stack carries no
    compression at all — long contexts are the model's problem under
    that profile, which is exactly the trade the profile records.
    """

    @property
    def name(self) -> str:
        return "SummarizationMiddleware"

    def wrap_model_call(self, request, handler):
        return handler(request)


class LargeResultOffloadMiddleware(AgentMiddleware):
    """Large-result offloading, at this blueprint's layer.

    A tool result longer than the profile's threshold is written in
    full to the case filesystem under /offloaded/ and the model sees
    a short pointer plus a preview, so one oversized document cannot
    flood the context of every later step. The full text stays one
    read_file away, in the case's own files.
    """

    def __init__(self, backend, threshold_chars: int):
        super().__init__()
        self._backend = backend
        self._threshold = threshold_chars
        self._counter = 0

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        if not isinstance(result, ToolMessage):
            return result
        content = result.content
        if not isinstance(content, str) or len(content) <= self._threshold:
            return result
        self._counter += 1
        tool_name = (request.tool_call or {}).get("name", "tool")
        path = f"/offloaded/{tool_name}-{self._counter:03d}.md"
        self._backend.write_text_direct(path, content)
        pointer = (
            f"[large tool result offloaded by the harness profile: "
            f"{len(content)} characters from {tool_name} written to {path} "
            f"— read it back with read_file if you need the full text]\n\n"
            + content[:600]
        )
        return ToolMessage(
            content=pointer,
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
        )


def build_middleware(profile: HarnessProfile, backend) -> list:
    """The caller middleware a profile contributes to the harness
    stack. Under the offline profile's switches (compression on,
    offloading off) this is empty — the build passes exactly what it
    passed before profiles existed."""
    stack: list = []
    if not profile.middleware["context_compression"]:
        stack.append(DisabledSummarizationMiddleware())
    if profile.middleware["large_result_offloading"]:
        stack.append(
            LargeResultOffloadMiddleware(
                backend, profile.middleware["offload_threshold_chars"]
            )
        )
    return stack
