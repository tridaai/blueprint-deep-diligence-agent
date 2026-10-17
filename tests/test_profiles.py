"""Harness profiles: the artifacts, the fail-closed loader, the
build that consumes them, the middleware switches, and the record a
case carries.

The pinned invariant of this round: under the shipped offline-stub
profile the harness build is byte-for-byte the build this blueprint
had before profiles existed — same planner prompt, same workstream
prompts, same tools on every role, no caller middleware.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage

from deep_diligence import harness as harness_module
from deep_diligence.api import create_app
from deep_diligence.backend import EvidenceBackend
from deep_diligence.config import load_settings
from deep_diligence.corpus import GOLD_CASES
from deep_diligence.harness import (
    PLANNER_SYSTEM,
    WORKSTREAM_DESCRIPTIONS,
    build_agent,
    workstream_system,
)
from deep_diligence.models import WORKSTREAMS, BriefIn
from deep_diligence.profiles import (
    KNOWN_TOOLS,
    DisabledSummarizationMiddleware,
    LargeResultOffloadMiddleware,
    ProfileError,
    build_middleware,
    list_profile_ids,
    load_profile,
)
from deep_diligence.service import CaseService
from deep_diligence.stub_model import ScriptedDiligenceModel
from deep_diligence.tools import build_case_tools

REPO_ROOT = Path(__file__).resolve().parents[1]


# -- the shipped artifacts ---------------------------------------------------


def test_two_profiles_ship_and_load():
    assert list_profile_ids() == ["offline-stub", "openai-gpt"]
    stub = load_profile("offline-stub")
    assert stub.record() == {
        "id": "offline-stub",
        "version": 1,
        "target_model": "offline:stub",
    }
    real = load_profile("openai-gpt")
    assert real.record() == {
        "id": "openai-gpt",
        "version": 1,
        "target_model": "openai:gpt-5.1",
    }


def test_offline_profile_reproduces_the_pre_profile_build():
    profile = load_profile("offline-stub")
    assert profile.planner_prompt() == PLANNER_SYSTEM
    assert [entry.name for entry in profile.subagents] == list(WORKSTREAMS)
    for entry in profile.subagents:
        assert entry.description == WORKSTREAM_DESCRIPTIONS[entry.name]
        assert profile.subagent_prompt(entry.name) == workstream_system(entry.name)
        assert entry.tools == KNOWN_TOOLS
    assert profile.middleware == {
        "context_compression": True,
        "large_result_offloading": False,
        "offload_threshold_chars": 12000,
    }
    # The switches at their defaults contribute no caller middleware:
    # the built stack is exactly the harness's own.
    assert build_middleware(profile, backend=None) == []


def test_openai_profile_differs_materially():
    profile = load_profile("openai-gpt")
    offline = load_profile("offline-stub")
    # Different prompt suffix sections, planner and workstreams.
    assert profile.planner_prompt() != offline.planner_prompt()
    assert profile.planner_prompt().startswith(
        offline.prompt_sections["planner-base"]
    )
    assert profile.planner_prompt().endswith(
        profile.prompt_sections["planner-suffix-gpt"]
    )
    financial = profile.subagent("financial-scan")
    assert profile.subagent_prompt("financial-scan").endswith(
        profile.prompt_sections["workstream-suffix-gpt"]
    )
    assert financial.tools == KNOWN_TOOLS
    # Middleware switches flipped against the offline profile.
    assert profile.middleware["context_compression"] is False
    assert profile.middleware["large_result_offloading"] is True
    # One sub-agent's tool set narrowed: market-scan cannot read
    # documents, it works from search results only.
    market = profile.subagent("market-scan")
    assert market.tools == ("search_corpus", "record_finding")
    assert profile.subagent_prompt("market-scan").endswith(
        profile.prompt_sections["workstream-suffix-gpt-market"]
    )
    stack = build_middleware(profile, backend=None)
    assert [type(m).__name__ for m in stack] == [
        "DisabledSummarizationMiddleware",
        "LargeResultOffloadMiddleware",
    ]
    assert stack[0].name == "SummarizationMiddleware"
    assert stack[1]._threshold == 20000


# -- the build consumes the profile ------------------------------------------


def _capture_build(monkeypatch, service, case, profile):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "AGENT"

    monkeypatch.setattr(
        harness_module, "create_deep_agent", fake_create_deep_agent
    )
    tools = build_case_tools(service.store, service.objects, case)
    result = build_agent(
        model=ScriptedDiligenceModel(),
        backend=EvidenceBackend(service.store, service.objects, case["case_id"]),
        case_tools=tools,
        profile=profile,
    )
    assert result == "AGENT"
    return captured


def test_build_agent_consumes_the_offline_profile(monkeypatch, service, case):
    captured = _capture_build(
        monkeypatch, service, case, load_profile("offline-stub")
    )
    assert captured["system_prompt"] == PLANNER_SYSTEM
    assert captured["middleware"] == []
    specs = {spec["name"]: spec for spec in captured["subagents"]}
    assert list(specs) == list(WORKSTREAMS)
    for name, spec in specs.items():
        assert spec["system_prompt"] == workstream_system(name)
        assert spec["description"] == WORKSTREAM_DESCRIPTIONS[name]
        assert [t.name for t in spec["tools"]] == list(KNOWN_TOOLS)


def test_build_agent_consumes_the_openai_profile(monkeypatch, service, case):
    profile = load_profile("openai-gpt")
    captured = _capture_build(monkeypatch, service, case, profile)
    assert captured["system_prompt"] == profile.planner_prompt()
    assert captured["system_prompt"] != PLANNER_SYSTEM
    specs = {spec["name"]: spec for spec in captured["subagents"]}
    assert [t.name for t in specs["market-scan"]["tools"]] == [
        "search_corpus",
        "record_finding",
    ]
    assert [t.name for t in specs["financial-scan"]["tools"]] == list(KNOWN_TOOLS)
    assert [m.name for m in captured["middleware"]] == [
        "SummarizationMiddleware",
        "LargeResultOffloadMiddleware",
    ]


# -- fail-closed validation ----------------------------------------------------


def _write_profile(tmp_path, profile_id="testprof", mutate=None, raw=None):
    if raw is None:
        raw = json.loads(
            (REPO_ROOT / "profiles" / "offline-stub.json").read_text()
        )
        raw["id"] = profile_id
    if mutate is not None:
        mutate(raw)
    path = tmp_path / f"{profile_id}.json"
    path.write_text(json.dumps(raw))
    return path


def _expect_profile_error(tmp_path, fragment, **kwargs):
    _write_profile(tmp_path, **kwargs)
    with pytest.raises(ProfileError) as excinfo:
        load_profile("testprof", tmp_path)
    assert fragment in str(excinfo.value)


def test_validation_unknown_top_level_key(tmp_path):
    def mutate(raw):
        raw["surprise"] = True

    _expect_profile_error(tmp_path, "unknown keys: surprise", mutate=mutate)


def test_validation_missing_required_key(tmp_path):
    def mutate(raw):
        del raw["middleware"]

    _expect_profile_error(tmp_path, "missing required keys: middleware", mutate=mutate)


def test_validation_planner_section_not_defined(tmp_path):
    def mutate(raw):
        raw["planner"]["sections"] = ["planner-base", "planner-ghost"]

    _expect_profile_error(tmp_path, "'planner-ghost'", mutate=mutate)


def test_validation_roster_entry_without_prompt_section(tmp_path):
    def mutate(raw):
        raw["subagents"][0]["prompt_sections"] = ["workstream-ghost"]

    _expect_profile_error(
        tmp_path, "sub-agent 'financial-scan'", mutate=mutate
    )
    _expect_profile_error(tmp_path, "'workstream-ghost'", mutate=mutate)


def test_validation_roster_entry_with_no_sections_at_all(tmp_path):
    def mutate(raw):
        raw["subagents"][1]["prompt_sections"] = []

    _expect_profile_error(
        tmp_path, "must name at least one prompt section", mutate=mutate
    )


def test_validation_unknown_tool_name(tmp_path):
    def mutate(raw):
        raw["subagents"][2]["tools"] = ["search_corpus", "teleport"]

    _expect_profile_error(tmp_path, "'teleport'", mutate=mutate)
    _expect_profile_error(tmp_path, "does not exist", mutate=mutate)


def test_validation_unknown_middleware_key(tmp_path):
    def mutate(raw):
        raw["middleware"]["turbo_mode"] = True

    _expect_profile_error(tmp_path, "unknown keys: turbo_mode", mutate=mutate)


def test_validation_middleware_switch_left_unset(tmp_path):
    def mutate(raw):
        del raw["middleware"]["context_compression"]

    _expect_profile_error(
        tmp_path, "missing required keys: context_compression", mutate=mutate
    )


def test_validation_duplicate_roster_name(tmp_path):
    def mutate(raw):
        raw["subagents"].append(dict(raw["subagents"][0]))

    _expect_profile_error(tmp_path, "duplicate sub-agent name", mutate=mutate)


def test_validation_id_must_match_file_name(tmp_path):
    def mutate(raw):
        raw["id"] = "someone-else"

    _write_profile(tmp_path, mutate=mutate)
    with pytest.raises(ProfileError) as excinfo:
        load_profile("testprof", tmp_path)
    assert "does not match" in str(excinfo.value)


def test_validation_malformed_json(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    with pytest.raises(ProfileError) as excinfo:
        load_profile("broken", tmp_path)
    assert "not valid JSON" in str(excinfo.value)


def test_missing_profile_file_is_a_named_refusal(tmp_path):
    with pytest.raises(ProfileError) as excinfo:
        load_profile("ghost", tmp_path)
    assert "not found" in str(excinfo.value)


# -- the middleware the switches control ---------------------------------------


class _FakeBackend:
    def __init__(self):
        self.writes: dict[str, str] = {}

    def write_text_direct(self, path, content):
        self.writes[path] = content


def test_disabled_summarization_takes_the_builtin_name_and_passes_through():
    middleware = DisabledSummarizationMiddleware()
    # The name is the mechanism: caller middleware merges into the
    # harness stack by name, replacing the same-named built-in.
    assert middleware.name == "SummarizationMiddleware"
    sentinel = object()
    assert middleware.wrap_model_call(None, lambda request: sentinel) is sentinel


def test_offload_middleware_writes_large_results_to_the_case_files():
    backend = _FakeBackend()
    middleware = LargeResultOffloadMiddleware(backend, threshold_chars=100)
    big = "FINDING BODY " * 400
    message = ToolMessage(
        content=big, tool_call_id="call_9", name="read_document"
    )
    request = SimpleNamespace(
        tool_call={"name": "read_document", "args": {}, "id": "call_9"}
    )
    result = middleware.wrap_tool_call(request, lambda req: message)
    assert backend.writes == {"/offloaded/read_document-001.md": big}
    assert result.tool_call_id == "call_9"
    assert result.content.startswith("[large tool result offloaded")
    assert "/offloaded/read_document-001.md" in result.content
    assert big[:600] in result.content
    assert len(result.content) < len(big)


def test_offload_middleware_leaves_small_and_non_text_results_alone():
    backend = _FakeBackend()
    middleware = LargeResultOffloadMiddleware(backend, threshold_chars=100)
    small = ToolMessage(content="short", tool_call_id="c1", name="search_corpus")
    request = SimpleNamespace(tool_call={"name": "search_corpus"})
    assert middleware.wrap_tool_call(request, lambda req: small) is small
    not_a_message = object()
    assert middleware.wrap_tool_call(request, lambda req: not_a_message) is (
        not_a_message
    )
    assert backend.writes == {}


# -- selection + recording -----------------------------------------------------


def test_case_records_the_profile_and_the_audit_trail_shows_it(service, case):
    assert case["profile"] == {
        "id": "offline-stub",
        "version": 1,
        "target_model": "offline:stub",
    }
    audit = service.store.list_audit(case["case_id"])
    created = next(a for a in audit if a["event"] == "case_created")
    assert created["payload"]["profile"]["id"] == "offline-stub"
    service.run_planning(case["case_id"])
    audit = service.store.list_audit(case["case_id"])
    grants = [a for a in audit if a["event"] == "budget_granted"]
    assert grants and all(
        g["payload"]["profile"] == case["profile"] for g in grants
    )


def test_case_view_exposes_the_profile(settings, case):
    client = TestClient(create_app(settings))
    body = client.get(f"/api/cases/{case['case_id']}").json()
    assert body["profile"]["id"] == "offline-stub"
    assert body["profile"]["version"] == 1


def test_service_refuses_to_start_on_an_unknown_profile(tmp_path):
    settings = load_settings(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
        profile_id="no-such-profile",
    )
    with pytest.raises(ProfileError, match="not found"):
        CaseService(settings)


def test_service_refuses_to_start_on_an_invalid_profile(tmp_path):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    raw = json.loads((REPO_ROOT / "profiles" / "offline-stub.json").read_text())
    raw["id"] = "broken"
    raw["subagents"][0]["tools"] = ["search_corpus", "teleport"]
    (profiles_dir / "broken.json").write_text(json.dumps(raw))
    settings = load_settings(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
        profile_id="broken",
        profile_dir=str(profiles_dir),
    )
    with pytest.raises(ProfileError, match="teleport"):
        CaseService(settings)


def test_openai_profile_runs_a_case_with_the_stub_standing_in(tmp_path):
    """The profile is configuration, the model is injected: the
    offline stub drives the openai-gpt profile end to end. The run
    completes and the profile is recorded; the narrowed market-scan
    records nothing (it has no document reader under this profile),
    which is the cost the per-profile eval run reports."""
    settings = load_settings(
        sqlite_path=str(tmp_path / "diligence.db"),
        object_dir=str(tmp_path / "objects"),
        migrations_dir=str(REPO_ROOT / "migrations"),
        profile_id="openai-gpt",
    )
    service = CaseService(settings)
    gold = GOLD_CASES[0]  # Helvion
    case = service.create_case(BriefIn(**gold["brief"]))
    case_id = case["case_id"]
    assert case["profile"]["id"] == "openai-gpt"
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    case = service.store.get_case(case_id)
    assert case["status"] == "DRAFT_READY"
    market_q = next(
        q["id"] for q in gold["brief"]["questions"]
        if q["workstream"] == "market-scan"
    )
    other_qs = {
        q["id"] for q in gold["brief"]["questions"]
        if q["workstream"] != "market-scan"
    }
    findings = service.store.list_findings(case_id)
    assert [f for f in findings if f["question_id"] == market_q] == []
    assert {f["question_id"] for f in findings} == other_qs
    grants = [
        a
        for a in service.store.list_audit(case_id)
        if a["event"] == "budget_granted"
    ]
    assert {g["payload"]["profile"]["id"] for g in grants} == {"openai-gpt"}


# -- per-profile eval runs -----------------------------------------------------


def _load_eval_runner():
    spec = importlib.util.spec_from_file_location(
        "run_evals", REPO_ROOT / "evals" / "run_evals.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_runner_attributes_results_to_the_profile():
    runner = _load_eval_runner()
    gold = GOLD_CASES[1]  # Copperline
    default = runner.run_case(gold)
    assert default["profile"]["id"] == "offline-stub"
    assert default["recall"] == 1.0
    under_openai = runner.run_case(gold, profile_id="openai-gpt")
    assert under_openai["profile"] == {
        "id": "openai-gpt",
        "version": 1,
        "target_model": "openai:gpt-5.1",
    }
    # The narrowed market-scan cannot read its documents from the
    # stub, so the profile's cost shows up in its own column.
    assert under_openai["recall"] < 1.0
