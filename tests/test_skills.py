"""Round 17: the skills library.

Skills are governed, versioned method bundles. These tests pin:
the seeded library loads under the strict loader and lands in the
store exactly once; the loader refuses malformed artifacts with
named errors; a skill naming tools beyond a role's allowlist
under a profile is refused wherever it is validated against that
profile (publish, plan gate); brief-declared skills ride the plan
artifact, approval pins the version in force, and a role mismatch
or an unknown name refuses at the gate with the plan left
standing; execution mounts the pinned method text read-only at
/skills/<name>.md, findings carry the pins as provenance, and the
report's Method section folds them; a version published mid-case
does not move a case whose plan already pinned the older one; and
a case that names no skills produces exactly the records it always
did — no skills key on its plan, its findings, or its report.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.config import load_settings
from deep_diligence.corpus import GOLD_CASES
from deep_diligence.models import BriefIn
from deep_diligence.profiles import load_profile
from deep_diligence.service import CaseService, GateError
from deep_diligence.skills import (
    SkillError,
    load_skill_file,
    seed_skills,
    validate_skill,
)

GOLD = GOLD_CASES[0]


def _artifact(**overrides) -> dict:
    artifact = {
        "name": "desk-method",
        "version": 1,
        "applies_to": ["financial-scan"],
        "tools": ["search_corpus", "read_document", "record_finding"],
        "description": "A test method.",
        "method": "METHOD: do the work carefully.",
        "changelog": "Initial version.",
    }
    artifact.update(overrides)
    return artifact


def _skills_brief(**question_skills) -> dict:
    brief = {**GOLD["brief"], "questions": [dict(q) for q in GOLD["brief"]["questions"]]}
    for question in brief["questions"]:
        if question["id"] in question_skills:
            question["skills"] = question_skills[question["id"]]
    return brief


# -- the loader ---------------------------------------------------------------

def test_seed_library_loads_and_seeds_exactly_three_skills(service):
    library = service.list_skills()
    assert [(s["name"], s["version"]) for s in library] == [
        ("contradiction-hunting", 1),
        ("revenue-bridge", 1),
        ("source-triangulation", 1),
    ]
    assert all(s["origin"] == "seed" for s in library)
    revenue = next(s for s in library if s["name"] == "revenue-bridge")
    assert revenue["applies_to"] == ["financial-scan"]
    assert "REVENUE BRIDGE" in revenue["method"]
    triangulation = next(s for s in library if s["name"] == "source-triangulation")
    assert triangulation["applies_to"] == ["market-scan"]


def test_seeding_is_idempotent(settings):
    service = CaseService(settings)
    assert len(service.list_skills()) == 3
    again = CaseService(settings)
    assert len(again.list_skills()) == 3
    assert again.skill_seeds == []


def test_loader_refuses_a_file_whose_name_does_not_match_its_contents(tmp_path):
    path = tmp_path / "renamed.v9.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    with pytest.raises(SkillError) as excinfo:
        load_skill_file(path)
    assert "does not match" in str(excinfo.value)


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"surprise": True}, "unknown keys"),
        ({"tools": ["search_corpus", "delete_case"]}, "not a case tool"),
        ({"applies_to": ["legal-scan"]}, "not a workstream"),
        ({"applies_to": []}, "at least one workstream"),
        ({"method": "  "}, "'method' must be non-empty"),
        ({"tools": ["search_corpus", "search_corpus"]}, "lists a tool twice"),
        ({"version": 0}, "'version' must be an integer >= 1"),
        ({"name": "Not A Slug"}, "'name' must be a slug"),
    ],
)
def test_loader_refuses_malformed_artifacts(overrides, fragment):
    with pytest.raises(SkillError) as excinfo:
        validate_skill(_artifact(**overrides), "test")
    assert fragment in str(excinfo.value)


def test_loader_refuses_a_missing_key():
    artifact = _artifact()
    del artifact["changelog"]
    with pytest.raises(SkillError) as excinfo:
        validate_skill(artifact, "test")
    assert "missing required keys: changelog" in str(excinfo.value)


def test_tool_widening_is_judged_against_the_loading_profile():
    # contradiction-hunting names read_document for every role.
    # Under the offline profile every role allows it; under
    # openai-gpt the market-scan role is narrowed to search +
    # record, so the same artifact refuses there — named.
    artifact = _artifact(
        name="contradiction-hunting",
        applies_to=["financial-scan", "product-scan", "market-scan"],
    )
    validate_skill(artifact, "test", load_profile("offline-stub"))
    with pytest.raises(SkillError) as excinfo:
        validate_skill(artifact, "test", load_profile("openai-gpt"))
    message = str(excinfo.value)
    assert "read_document" in message and "market-scan" in message
    assert "never widen its tools" in message


# -- publishing ---------------------------------------------------------------

def test_publish_extends_the_line_and_records_the_act(service):
    record = service.publish_skill(
        "Tester",
        _artifact(
            name="revenue-bridge",
            version=2,
            method="METHOD: REVENUE BRIDGE v2 — the revised method.",
            changelog="Tightened the bridge walk after the spring review.",
        ),
    )
    assert record["origin"] == "published"
    assert record["published_by"] == "Tester"
    assert service.store.latest_skill("revenue-bridge")["version"] == 2
    # The pinned past does not move: v1 is still in the library,
    # byte for byte the seeded artifact.
    assert "spring review" not in service.store.get_skill("revenue-bridge", 1)["changelog"]


def test_publish_refuses_versions_that_skip_or_repeat(service):
    with pytest.raises(SkillError) as excinfo:
        service.publish_skill("Tester", _artifact(name="revenue-bridge", version=3))
    assert "next publishable version is 2" in str(excinfo.value)
    with pytest.raises(SkillError):
        service.publish_skill("Tester", _artifact(name="revenue-bridge", version=1))
    with pytest.raises(SkillError):
        service.publish_skill("Tester", _artifact(name="brand-new", version=2))
    assert service.store.latest_skill("brand-new") is None


def test_publish_refuses_a_tool_widening_skill(service):
    with pytest.raises(SkillError) as excinfo:
        service.publish_skill(
            "Tester",
            _artifact(name="desk-method", tools=["search_corpus", "write_file"]),
        )
    assert "write_file" in str(excinfo.value)
    assert service.store.get_skill("desk-method", 1) is None


def test_publish_refuses_what_the_active_profile_narrows_out(settings):
    narrowed = CaseService(
        load_settings(
            sqlite_path=settings.sqlite_path,
            object_dir=settings.object_dir,
            migrations_dir=settings.migrations_dir,
            profile_id="openai-gpt",
        )
    )
    with pytest.raises(SkillError) as excinfo:
        narrowed.publish_skill(
            "Tester",
            _artifact(
                name="market-method",
                applies_to=["market-scan"],
                tools=["search_corpus", "read_document"],
            ),
        )
    assert "market-scan" in str(excinfo.value)


# -- the plan gate --------------------------------------------------------------

def test_approval_pins_the_version_and_the_audit_carries_it(service):
    case = service.create_case(BriefIn(**_skills_brief(q1=["revenue-bridge"])))
    case_id = case["case_id"]
    plan = service.run_planning(case_id)
    entry = plan["plan"]["workstreams"][0]
    # The planner carried the name; the readable plan shows it too.
    assert entry["skills"] == ["revenue-bridge"]
    assert "skills: revenue-bridge" in plan["markdown"]
    service.decide_plan(case_id, analyst="Tester", approve=True)
    plan = service.store.get_plan(case_id)
    assert plan["plan"]["workstreams"][0]["skills"] == [
        {"name": "revenue-bridge", "version": 1}
    ]
    approved = next(
        a for a in service.store.list_audit(case_id) if a["event"] == "plan_approved"
    )
    assert approved["payload"]["skills"] == {
        "q1": [{"name": "revenue-bridge", "version": 1}]
    }


def test_role_mismatch_refuses_at_the_gate_and_the_plan_stands(service):
    # revenue-bridge applies to financial-scan; q3 is market-scan.
    case = service.create_case(BriefIn(**_skills_brief(q3=["revenue-bridge"])))
    case_id = case["case_id"]
    service.run_planning(case_id)
    with pytest.raises(GateError) as excinfo:
        service.decide_plan(case_id, analyst="Tester", approve=True)
    message = str(excinfo.value)
    assert "revenue-bridge" in message and "market-scan" in message
    # The refusal changes nothing: the plan still awaits a decision.
    assert service.store.get_case(case_id)["status"] == "PLAN_READY"
    assert service.store.get_plan(case_id)["status"] == "PENDING"
    decided = service.decide_plan(case_id, analyst="Tester", approve=False)
    assert decided["status"] == "PLAN_REJECTED"


def test_unknown_skill_refuses_at_the_gate(service):
    case = service.create_case(BriefIn(**_skills_brief(q1=["no-such-skill"])))
    case_id = case["case_id"]
    service.run_planning(case_id)
    with pytest.raises(GateError) as excinfo:
        service.decide_plan(case_id, analyst="Tester", approve=True)
    assert "no-such-skill" in str(excinfo.value)
    assert "not in the skills library" in str(excinfo.value)


def test_gate_refuses_a_skill_the_active_profile_narrows_out(settings):
    narrowed = CaseService(
        load_settings(
            sqlite_path=settings.sqlite_path,
            object_dir=settings.object_dir,
            migrations_dir=settings.migrations_dir,
            profile_id="openai-gpt",
        )
    )
    # contradiction-hunting applies to market-scan but names
    # read_document, which openai-gpt's market-scan may not use.
    case = narrowed.create_case(
        BriefIn(**_skills_brief(q3=["contradiction-hunting"]))
    )
    case_id = case["case_id"]
    narrowed.run_planning(case_id)
    with pytest.raises(GateError) as excinfo:
        narrowed.decide_plan(case_id, analyst="Tester", approve=True)
    message = str(excinfo.value)
    assert "contradiction-hunting" in message and "read_document" in message


# -- execution: mount, provenance, pinning --------------------------------------

def _run_skills_case(service) -> str:
    case = service.create_case(
        BriefIn(
            **_skills_brief(
                q1=["revenue-bridge"], q2=["contradiction-hunting"]
            )
        )
    )
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    return case_id


def test_findings_carry_the_pins_and_recall_holds(service):
    case_id = _run_skills_case(service)
    findings = service.store.list_findings(case_id)
    by_question: dict[str, set] = {}
    for finding in findings:
        pins = {
            (p["name"], p["version"]) for p in finding.get("skills", [])
        }
        by_question.setdefault(finding["question_id"], set()).add(
            pins or None
        )
    assert by_question["q1"] == {(("revenue-bridge", 1),)}
    assert by_question["q2"] == {(("contradiction-hunting", 1),)}
    # A workstream that ran under no skill stamps nothing.
    assert by_question["q3"] == {None}
    assert all("skills" not in f for f in findings if f["question_id"] == "q3")
    found = {f["fact_ref"] for f in findings if f.get("fact_ref")}
    assert found == set(GOLD["gold_facts"])


def test_the_mounted_skill_file_is_the_pinned_text_and_read_only(service):
    case_id = _run_skills_case(service)
    backend = service._backend(case_id)
    seeded = service.store.get_skill("revenue-bridge", 1)
    result = backend.read("/skills/revenue-bridge.md")
    assert result.error is None
    assert result.file_data["content"] == seeded["method"]
    listing = backend.ls("/skills/")
    assert {e["path"] for e in listing.entries} == {
        "/skills/revenue-bridge.md",
        "/skills/contradiction-hunting.md",
    }
    assert backend.write("/skills/revenue-bridge.md", "tampered").error == (
        "Skill files are read-only"
    )
    assert backend.edit(
        "/skills/revenue-bridge.md", "METHOD", "HACKED"
    ).error == "Skill files are read-only"
    assert backend.delete("/skills/revenue-bridge.md").error == (
        "Skill files are read-only"
    )
    upload = backend.upload_files([("/skills/revenue-bridge.md", b"x")])
    assert upload[0].error == "permission_denied"
    # And the library record itself is untouched by all of it.
    assert service.store.get_skill("revenue-bridge", 1) == seeded


def test_a_case_runs_under_its_pin_after_a_newer_version_publishes(service):
    case_id = _run_skills_case(service)
    service.publish_skill(
        "Tester",
        _artifact(
            name="revenue-bridge",
            version=2,
            method="METHOD: REVENUE BRIDGE v2 — changed method text.",
            changelog="Revised after the case pinned v1.",
        ),
    )
    # The mounted text is still v1's: the plan pinned it.
    backend = service._backend(case_id)
    result = backend.read("/skills/revenue-bridge.md")
    assert "changed method text" not in result.file_data["content"]
    # A case approved after the publish pins v2 instead.
    later = service.create_case(BriefIn(**_skills_brief(q1=["revenue-bridge"])))
    service.run_planning(later["case_id"])
    service.decide_plan(later["case_id"], analyst="Tester", approve=True)
    plan = service.store.get_plan(later["case_id"])
    assert plan["plan"]["workstreams"][0]["skills"] == [
        {"name": "revenue-bridge", "version": 2}
    ]


def test_the_report_method_section_names_the_skills_that_ran(service):
    case_id = _run_skills_case(service)
    report = service.store.get_report(case_id)
    notes = report["model"]["method_notes"]
    skill_notes = [n for n in notes if "governed skill" in n]
    assert len(skill_notes) == 2
    assert any("revenue-bridge v1" in n and "q1" in n for n in skill_notes)
    assert any("contradiction-hunting v1" in n and "q2" in n for n in skill_notes)
    assert "revenue-bridge v1" in report["markdown"]


def test_a_skill_free_case_records_nothing_new(service):
    case = service.create_case(BriefIn(**GOLD["brief"]))
    case_id = case["case_id"]
    plan = service.run_planning(case_id)
    assert all("skills" not in w for w in plan["plan"]["workstreams"])
    service.decide_plan(case_id, analyst="Tester", approve=True)
    approved = next(
        a for a in service.store.list_audit(case_id) if a["event"] == "plan_approved"
    )
    assert "skills" not in approved["payload"]
    service.run_execution(case_id)
    findings = service.store.list_findings(case_id)
    assert findings and all("skills" not in f for f in findings)
    report = service.store.get_report(case_id)
    assert all(
        "governed skill" not in n for n in report["model"]["method_notes"]
    )


# -- API -------------------------------------------------------------------------

@pytest.fixture()
def client(settings):
    return TestClient(create_app(settings))


def test_api_lists_reads_and_publishes_skills(client):
    listing = client.get("/api/skills")
    assert listing.status_code == 200
    names = {(s["name"], s["version"]) for s in listing.json()["skills"]}
    assert ("revenue-bridge", 1) in names
    one = client.get("/api/skills/revenue-bridge")
    assert one.status_code == 200
    assert one.json()["versions"][0]["method"].startswith("METHOD: REVENUE BRIDGE")
    version = client.get("/api/skills/revenue-bridge/versions/1")
    assert version.status_code == 200
    assert client.get("/api/skills/nope").status_code == 404
    assert client.get("/api/skills/revenue-bridge/versions/9").status_code == 404
    published = client.post(
        "/api/skills",
        json={
            "analyst": "Console Tester",
            "name": "revenue-bridge",
            "version": 2,
            "applies_to": ["financial-scan"],
            "tools": ["search_corpus", "read_document", "record_finding"],
            "description": "The revised revenue method.",
            "method": "METHOD: REVENUE BRIDGE v2.",
            "changelog": "Second version.",
        },
    )
    assert published.status_code == 201
    assert published.json()["published_by"] == "Console Tester"
    widening = client.post(
        "/api/skills",
        json={
            "analyst": "Console Tester",
            "name": "revenue-bridge",
            "version": 3,
            "applies_to": ["financial-scan"],
            "tools": ["search_corpus", "execute_plan"],
            "description": "A widening method.",
            "method": "METHOD: reach for tools the role does not hold.",
            "changelog": "Must never land.",
        },
    )
    assert widening.status_code == 422
    assert "execute_plan" in widening.json()["detail"]
    # The refused version is not in the library.
    assert client.get("/api/skills/revenue-bridge/versions/3").status_code == 404


def test_no_route_edits_or_deletes_a_skill(settings):
    """A published version never changes: asserted against the
    route table itself — the only skills write is the publish."""
    app = create_app(settings)
    writes = []
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) or set()
        if not path.startswith("/api/skills"):
            continue
        assert "DELETE" not in methods and "PUT" not in methods, (
            f"{methods} on {path}"
        )
        if "POST" in methods:
            writes.append(path)
    assert writes == ["/api/skills"]


def test_seed_skills_reports_what_it_inserted(settings, tmp_path):
    # Seeding against an empty directory is a no-op, not an error:
    # a deployment may run with the library held only in the store.
    store_service = CaseService(settings)
    assert seed_skills(store_service.store, tmp_path) == []
