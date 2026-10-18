"""Investigation skills: governed, versioned method bundles.

A skill is a reusable piece of investigative METHOD — how a
workstream should go about its question — written once by the desk,
versioned, and loaded into a sub-agent's context when a case's plan
names it. It is not case memory (analyst notes are that), not
evidence (a skill can never be cited by a finding), and not a
profile (a profile shapes the harness build for a model; a skill is
method for whoever runs under it).

Artifacts live as JSON files in ``skills/``, one file per version
(``<name>.v<version>.json``), validated by a fail-closed strict
loader in the profiles discipline: unknown or missing keys, a name
that does not match its file, an unknown workstream, a tool that is
not a case tool — each refuses with a named `SkillError`, and
nothing falls back silently. At startup the service seeds the
shipped artifacts into the store's skills table (the runtime
library of record), idempotently, the way the source registry is
seeded from the corpus. Publishing a new version after that is a
recorded analyst act against the store — the record carries the
analyst, the changelog, and the timestamp, in the append-only
pattern of the profile promotion log. A published version is never
edited; a change is a new version.

The tool rule is the governance heart of the feature: a skill names
the case tools its method instructs the agent to use, and a skill
may narrow a role's method but NEVER widen its tools. Wherever a
skill is validated against a harness profile — at publish time, and
again at the plan gate — every tool it names must already sit in
that role's allowlist under that profile, or the validation refuses
with the excess named. Under a profile that narrows a role, a skill
the desk published happily under a wider profile stops being
approvable for that role until the profiles or the skill change:
fail-closed, at the gate, in the open.

Cases pin versions: a brief question may declare skill names, the
planner carries the names onto the plan artifact, and the plan gate
resolves each name to the library's latest version at the moment of
approval and stamps the pin on the plan. Execution runs under the
pins — a version published mid-case changes nothing for a case
whose plan is already approved; it is picked up by the next plan
the desk approves.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .models import WORKSTREAMS
from .profiles import KNOWN_TOOLS
from .store import Store, utcnow

_TOP_KEYS = {
    "name",
    "version",
    "applies_to",
    "tools",
    "description",
    "method",
    "changelog",
}
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")

SKILLS_PREFIX = "/skills/"


class SkillError(RuntimeError):
    """A skill artifact or act that refuses. The message names it."""


@dataclass(frozen=True)
class Skill:
    name: str
    version: int
    applies_to: tuple[str, ...]
    tools: tuple[str, ...]
    description: str
    method: str
    changelog: str

    def record(self) -> dict:
        """The pin a plan, a finding, and an audit entry carry."""
        return {"name": self.name, "version": self.version}

    def mount_path(self) -> str:
        return f"{SKILLS_PREFIX}{self.name}.md"

    def artifact(self) -> dict:
        """The governed artifact itself, as published and seeded."""
        return {
            "name": self.name,
            "version": self.version,
            "applies_to": list(self.applies_to),
            "tools": list(self.tools),
            "description": self.description,
            "method": self.method,
            "changelog": self.changelog,
        }


def default_skills_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "skills"


def _fail(source: str, problem: str) -> SkillError:
    return SkillError(f"skill {source}: {problem}")


def validate_skill(raw, source: str, profile=None) -> Skill:
    """Validate one skill artifact, fail-closed.

    Structural rules always bind: exact keys, a slug name, an
    integer version, named workstreams, named case tools, real
    text. When a harness profile is supplied, the profile rules
    bind too: every role the skill applies to must exist in the
    profile's roster, and every tool the skill names must already
    be in that role's allowlist under the profile — a skill may
    narrow a role's method, never widen its tools.
    """
    if not isinstance(raw, dict):
        raise _fail(source, "the skill artifact must be a JSON object")
    unknown = sorted(set(raw) - _TOP_KEYS)
    if unknown:
        raise _fail(source, f"unknown keys: {', '.join(unknown)}")
    missing = sorted(_TOP_KEYS - set(raw))
    if missing:
        raise _fail(source, f"missing required keys: {', '.join(missing)}")
    name = raw["name"]
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise _fail(
            source,
            "'name' must be a slug (lowercase letters, digits, hyphens)",
        )
    version = raw["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise _fail(source, "'version' must be an integer >= 1")
    applies_to = raw["applies_to"]
    if not isinstance(applies_to, list) or not applies_to:
        raise _fail(source, "'applies_to' must name at least one workstream")
    for role in applies_to:
        if role not in WORKSTREAMS:
            raise _fail(
                source,
                f"'applies_to' names {role!r}, which is not a workstream; "
                f"workstreams are: {', '.join(WORKSTREAMS)}",
            )
    if len(set(applies_to)) != len(applies_to):
        raise _fail(source, "'applies_to' lists a workstream twice")
    tools = raw["tools"]
    if not isinstance(tools, list) or not tools:
        raise _fail(source, "'tools' must name at least one case tool")
    for tool_name in tools:
        if tool_name not in KNOWN_TOOLS:
            raise _fail(
                source,
                f"names tool {tool_name!r}, which is not a case tool; "
                f"a skill's method may only use the case tools: "
                f"{', '.join(KNOWN_TOOLS)}",
            )
    if len(set(tools)) != len(tools):
        raise _fail(source, "'tools' lists a tool twice")
    for field in ("description", "method", "changelog"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise _fail(source, f"'{field}' must be non-empty text")
    if profile is not None:
        for role in applies_to:
            try:
                entry = profile.subagent(role)
            except Exception as exc:  # ProfileError: role not in roster
                raise _fail(
                    source,
                    f"applies to {role!r}, which profile {profile.id!r} "
                    "has no sub-agent for",
                ) from exc
            excess = [t for t in tools if t not in entry.tools]
            if excess:
                raise _fail(
                    source,
                    f"names tool(s) {', '.join(excess)} for {role}, whose "
                    f"allowlist under profile {profile.id} v{profile.version} "
                    f"is {', '.join(entry.tools)} — a skill may narrow a "
                    "role's method, never widen its tools",
                )
    return Skill(
        name=name,
        version=version,
        applies_to=tuple(applies_to),
        tools=tuple(tools),
        description=raw["description"],
        method=raw["method"],
        changelog=raw["changelog"],
    )


def load_skill_file(path, profile=None) -> Skill:
    """Load and validate one versioned artifact file. The file
    name is part of the artifact's identity: ``<name>.v<version>``
    must match the name and version inside, so a renamed or
    hand-edited file refuses rather than drifting."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SkillError(f"skill {path}: not valid JSON: {exc}") from exc
    except OSError as exc:
        raise SkillError(f"skill {path}: cannot be read: {exc}") from exc
    skill = validate_skill(raw, str(path))
    expected_stem = f"{skill.name}.v{skill.version}"
    if path.stem != expected_stem:
        raise _fail(
            str(path),
            f"file name {path.stem!r} does not match the artifact "
            f"it carries ({expected_stem!r})",
        )
    if profile is not None:
        validate_skill(raw, str(path), profile)
    return skill


def list_skill_files(skills_dir=None) -> list[Path]:
    directory = Path(skills_dir) if skills_dir else default_skills_dir()
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def seed_skills(store: Store, skills_dir=None) -> list[dict]:
    """Seed the runtime library from the shipped artifacts.

    Runs at service startup, idempotently (the source-registry
    pattern): a (name, version) already in the store is the desk's
    record and is never overwritten by the file that shipped it.
    Validation here is structural — the profile binding is checked
    where a skill is loaded for a role: at publish and at the plan
    gate. A malformed seed artifact refuses the startup with it:
    the desk does not boot on a library it cannot read.
    """
    inserted: list[dict] = []
    for path in list_skill_files(skills_dir):
        skill = load_skill_file(path)
        if store.get_skill(skill.name, skill.version) is not None:
            continue
        record = {
            **skill.artifact(),
            "origin": "seed",
            "published_by": "skills library seed",
            "published_at": utcnow(),
        }
        store.add_skill(record)
        inserted.append(record)
    return inserted


def declared_skill_names(entry: dict) -> list[str]:
    """The skill names one plan workstream entry declares. Entries
    written before approval carry names (strings, as the planner
    wrote them); an entry being re-read after approval carries
    pins ({"name", "version"} records). Both read as names here.
    Anything else in the list is malformed and reported as its own
    repr, so the gate's refusal can name it."""
    names: list[str] = []
    for item in entry.get("skills") or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
        else:
            names.append(repr(item))
    return names


def resolve_plan_skills(store: Store, profile, workstreams: list[dict]):
    """The plan gate's skill check: resolve every skill a plan
    names to the library version it will run under.

    Returns (pins, problems): pins maps question_id -> the list of
    {"name", "version"} records to stamp on that workstream's plan
    entry; problems names every defect that refuses approval — a
    skill not in the library, a skill that does not apply to the
    workstream's role, a skill whose tools exceed the role's
    allowlist under the active profile. Every named skill is
    checked, so the analyst sees the whole list at once, in the
    schedule gate's manner.
    """
    pins: dict[str, list[dict]] = {}
    problems: list[str] = []
    for entry in workstreams:
        qid = entry.get("question_id")
        role = entry.get("workstream")
        resolved: list[dict] = []
        for name in declared_skill_names(entry):
            record = store.latest_skill(name)
            if record is None:
                problems.append(
                    f"workstream {qid} names skill {name!r}, "
                    "which is not in the skills library"
                )
                continue
            try:
                artifact = {key: record[key] for key in _TOP_KEYS}
                skill = validate_skill(
                    artifact, f"{name} v{record['version']}", profile
                )
            except SkillError as exc:
                problems.append(f"workstream {qid}: {exc}")
                continue
            if role not in skill.applies_to:
                problems.append(
                    f"workstream {qid} names skill {name!r}, which "
                    f"applies to {', '.join(skill.applies_to)} — "
                    f"not to {role}"
                )
                continue
            resolved.append(skill.record())
        if resolved:
            pins[qid] = resolved
    return pins, problems


def plan_skill_pins(store: Store, case_id: str) -> dict[str, list[dict]]:
    """The skills a case's current plan pins, by question: read
    from the stored plan artifact, never re-resolved — execution
    runs under the pins the analyst approved, even if a newer
    version has been published since. Empty for a case whose plan
    names no skills (or has no plan yet)."""
    plan = store.get_plan(case_id)
    if not plan:
        return {}
    pins: dict[str, list[dict]] = {}
    for entry in (plan.get("plan") or {}).get("workstreams", []):
        resolved = [
            {"name": item["name"], "version": item["version"]}
            for item in entry.get("skills") or []
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("version"), int)
        ]
        if resolved:
            pins[entry["question_id"]] = resolved
    return pins


def skill_mount_files(store: Store, case_id: str) -> dict[str, str]:
    """The skill files a case's virtual filesystem mounts:
    /skills/<name>.md -> the pinned version's method text,
    verbatim. A pin whose record has vanished from the library
    mounts nothing (the pins were validated at the gate; a missing
    record is a store integrity problem the read will surface as
    an absent file, not a silent substitution of another version).
    """
    files: dict[str, str] = {}
    for pin_list in plan_skill_pins(store, case_id).values():
        for pin in pin_list:
            record = store.get_skill(pin["name"], pin["version"])
            if record is not None:
                files[f"{SKILLS_PREFIX}{pin['name']}.md"] = record["method"]
    return files
