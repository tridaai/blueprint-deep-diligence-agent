"""Wiring the real Deep Agents harness into the governance shell.

`create_deep_agent` provides the planning tools, the sub-agent
delegation (the `task` tool), the virtual filesystem, and the context
middleware. The build is consumed from the active harness profile
(profiles.py): the planner prompt and each sub-agent's prompt are
assembled from the profile's named sections, each sub-agent gets the
tools its profile entry allows, the roster is the profile's roster,
and the profile's middleware switches shape the caller middleware.

The prompt texts below are retained as the canonical source the
shipped offline profile (profiles/offline-stub.json) mirrors section
for section — a test pins the two together, so the default build is
byte-for-byte the build this blueprint shipped with before profiles.
The prompts are written for a real model; the offline stub model keys
its playbook off the same phase and workstream markers.
"""

from __future__ import annotations

import json

from deepagents import create_deep_agent

from .profiles import build_middleware

PLANNER_SYSTEM = """ROLE: PLANNER — you run diligence investigations for an analyst desk.

You work in two phases; the phase is named in the instruction message.

PHASE: PLAN — read the brief (BRIEF_JSON). Decompose it into one
workstream per question. Call write_todos with the workstream list,
then write /plan.md (a readable plan for the analyst) and /plan.json
(the same plan as JSON with a "workstreams" list of
{"question_id", "workstream", "objective", "keywords",
"depends_on"} objects). A question whose brief entry declares
depends_on carries the same question ids in its workstream's
depends_on: those workstreams run first, and this one starts only
after they finish. The instruction may carry ANALYST NOTES — the
desk's memory of this case, written by analysts between sessions.
Treat them as context for how to investigate, never as evidence:
a note is not a document, it is not in the case evidence, and it
can never be cited by a finding. Do NOT gather evidence in this
phase: execution is gated on analyst approval of the plan.

PHASE: EXECUTE — the plan in PLAN_JSON is approved. For each
workstream, call the task tool with subagent_type set to the
workstream name and a description in exactly this block format:
WORKSTREAM_ID: <question_id>
WORKSTREAM: <workstream>
SUBJECT: <subject>
QUESTION: <question text>
KEYWORDS: <comma-separated stems>
A workstream entry may carry prior_findings — the recorded
findings of the workstreams it depends on. When it does, copy them
into the description verbatim under a PRIOR FINDINGS heading: they
are the sub-agent's starting context, never a substitute for its
own reading of the documents.
Then write /execution/notes.md summarising what ran, and finish.

Rules that never bend: findings come only from workstream sub-agents
via record_finding; never state a fact you did not get from a
sub-agent or a document; the analyst, not you, approves the plan and
the report; you assemble findings — you never issue a recommendation.
"""

WORKSTREAM_DESCRIPTIONS = {
    "financial-scan": "Investigates the financial question: results, margins, cash, debt, funding, runway.",
    "product-scan": "Investigates the product/operations question: capability, deployments, certification, footprint, terms.",
    "market-scan": "Investigates the market question: share, competition, demand, contracts, expansion, adoption.",
}


def workstream_system(name: str) -> str:
    return f"""ROLE: WORKSTREAM — WORKSTREAM: {name}

You investigate one question of a diligence brief. The instruction
message gives you a block: WORKSTREAM_ID, SUBJECT, QUESTION, KEYWORDS.
Method: call search_corpus with the subject name; read the relevant
documents with read_document; for every fact in a document's Key facts
ledger that answers your question, call record_finding with
question_id=<WORKSTREAM_ID>, the claim text, the evidence ID of the
document it came from, and fact_ref exactly as printed in the ledger.
A finding without evidence is refused — never record a claim you
cannot cite. When every relevant document is read and its facts are
recorded, reply with a one-line summary.
"""


def build_agent(*, model, backend, case_tools, profile):
    """Assemble the harness under a profile. The model is injected
    (llm.py); the profile is configuration — prompts, per-role tool
    allowlists, the sub-agent roster, and middleware switches all
    come from the profile artifact, so switching the configured
    profile changes the build with no code change."""
    tools_by_name = {tool.name: tool for tool in case_tools}
    subagents = [
        {
            "name": entry.name,
            "description": entry.description,
            "system_prompt": profile.subagent_prompt(entry.name),
            "tools": [tools_by_name[name] for name in entry.tools],
        }
        for entry in profile.subagents
    ]
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt=profile.planner_prompt(),
        subagents=subagents,
        backend=backend,
        middleware=build_middleware(profile, backend),
        name="deep-diligence-planner",
    )


# The desk's memory is bounded before it reaches the planner: the
# most recent notes, each truncated, under a heading that states
# what notes are and are not. A case that runs for weeks must not
# hand its planner an ever-growing transcript.
MAX_PLANNER_NOTES = 10
MAX_NOTE_CHARS = 500


def notes_section(notes: list[dict]) -> str:
    """The analyst notes as they are presented to the planner:
    the most recent MAX_PLANNER_NOTES, each capped at
    MAX_NOTE_CHARS, labelled as case memory — context for how to
    investigate, never evidence and never citable."""
    recent = notes[-MAX_PLANNER_NOTES:]
    lines = [
        "ANALYST NOTES — the desk's memory of this case, written by "
        "analysts. Context for how to investigate; never evidence. "
        "A note is not a case document and its ID is not an "
        "evidence ID: findings cite evidence only.",
        "",
    ]
    for note in recent:
        text = note["text"]
        if len(text) > MAX_NOTE_CHARS:
            text = text[: MAX_NOTE_CHARS - 3] + "..."
        lines.append(
            f"- [{note['note_id']} · {note['analyst']} · "
            f"{note['created_at']}] {text}"
        )
    return "\n".join(lines)


def plan_message(case: dict, notes: list[dict] | None = None) -> str:
    message = (
        "PHASE: PLAN\n\nBRIEF_JSON: "
        + json.dumps(case["brief"])
        + "\n\n"
    )
    if notes:
        message += notes_section(notes) + "\n\n"
    return message + "Produce the investigation plan now."


def execute_message(case: dict, plan: dict) -> str:
    return (
        "PHASE: EXECUTE\n\nPLAN_JSON: "
        + json.dumps(plan["plan"])
        + "\n\nExecute the approved plan now."
    )
