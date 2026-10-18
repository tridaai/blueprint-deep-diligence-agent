"""The deterministic offline model.

A scripted stand-in for a real chat model, used by tests, evals, and
the no-key demo. It drives the REAL Deep Agents harness — planning
todos, sub-agent delegation, filesystem writes, corpus tools — through
a fixed playbook keyed on the phase and workstream markers in the
prompts, and it reads facts from the corpus documents' Key facts
ledgers, judging relevance by the workstream keyword stems in
corpus.py. It is a labelled offline double: with a provider configured
(see llm.py) a real model follows the same prompts and tools.

Skills reach the stub through the same seams they reach a real
model — the plan artifact names them, the planner's task block
carries a SKILLS line, the method files are mounted on the case
filesystem — and the stub workstream reads each mounted file with
read_file before it searches. What the stub cannot do is follow
prose: its fact selection is the keyword playbook, unchanged by
whatever the method text says. The skills machinery therefore
grades loading, gating, and provenance offline, never method
compliance; a live model is where method text starts to steer.
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .corpus import WORKSTREAM_KEYWORDS

_call_ids = itertools.count(1)
FACT_LINE = re.compile(r"^- \[(F-[A-Z]+-\d+)\] (.*)$", re.MULTILINE)
EVIDENCE_ID = re.compile(r"\b(EV-\d{4})\b")


def _system_text(messages: list[BaseMessage]) -> str:
    for message in messages:
        if isinstance(message, SystemMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def _human_texts(messages: list[BaseMessage]) -> list[str]:
    return [
        m.content if isinstance(m.content, str) else str(m.content)
        for m in messages
        if isinstance(m, HumanMessage)
    ]


def _tool_calls(messages: list[BaseMessage], name: str | None = None) -> list[dict]:
    calls: list[dict] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if name is None or call["name"] == name:
                    calls.append(call)
    return calls


def _results_by_call_id(messages: list[BaseMessage]) -> dict[str, str]:
    results: dict[str, str] = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            content = message.content
            results[message.tool_call_id] = (
                content if isinstance(content, str) else str(content)
            )
    return results


def _results_for(messages: list[BaseMessage], name: str) -> list[str]:
    by_id = _results_by_call_id(messages)
    return [
        by_id[call["id"]] for call in _tool_calls(messages, name) if call["id"] in by_id
    ]


def _json_block(text: str, marker: str) -> dict:
    index = text.index(marker) + len(marker)
    return json.JSONDecoder().raw_decode(text[index:].lstrip())[0]


def _kv_block(text: str) -> dict[str, str]:
    block: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            block[key.strip()] = value.strip()
    return block


def _score(text: str, workstream: str) -> int:
    lowered = text.lower()
    return sum(1 for stem in WORKSTREAM_KEYWORDS[workstream] if stem in lowered)


def _ai_tool_calls(calls: list[tuple[str, dict]]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"call_{next(_call_ids)}",
                "type": "tool_call",
            }
            for name, args in calls
        ],
        usage_metadata={"input_tokens": 150, "output_tokens": 40, "total_tokens": 190},
    )


def _ai_text(text: str) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 150, "output_tokens": 60, "total_tokens": 210},
    )


class ScriptedDiligenceModel(BaseChatModel):
    """Deterministic playbook model for the offline profile."""

    @property
    def _llm_type(self) -> str:
        return "scripted-diligence-model"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        # langchain-core's default bind_tools raises NotImplementedError;
        # a scripted model only needs the tool list passed through.
        return self.bind(tools=list(tools))

    # -- planner ---------------------------------------------------------
    def _planner(self, messages: list[BaseMessage]) -> AIMessage:
        human = _human_texts(messages)[-1] if _human_texts(messages) else ""
        if "PHASE: PLAN" in human:
            brief = _json_block(human, "BRIEF_JSON:")
            questions = brief["questions"]
            if not _tool_calls(messages, "write_todos"):
                todos = [
                    {"content": "Draft the investigation plan", "status": "completed"}
                ] + [
                    {
                        "content": f"Workstream {q['workstream']} for {q['id']}: {q['text']}",
                        "status": "pending",
                    }
                    for q in questions
                ] + [{"content": "Analyst plan gate", "status": "pending"}]
                return _ai_tool_calls([("write_todos", {"todos": todos})])
            writes = {
                call["args"].get("file_path")
                for call in _tool_calls(messages, "write_file")
            }
            plan_entries = []
            for q in questions:
                entry = {
                    "question_id": q["id"],
                    "workstream": q["workstream"],
                    "objective": q["text"],
                    "keywords": WORKSTREAM_KEYWORDS[q["workstream"]],
                    "depends_on": list(q.get("depends_on") or []),
                }
                if q.get("skills"):
                    # The brief's declaration rides the plan
                    # verbatim: the names are what the approver
                    # sees at the gate; approval pins the versions.
                    entry["skills"] = list(q["skills"])
                plan_entries.append(entry)
            plan = {
                "subject": brief["subject"],
                "workstreams": plan_entries,
            }
            if "/plan.md" not in writes:
                lines = [
                    f"# Investigation plan — {brief['subject']}",
                    "",
                    f"Commissioned by {brief['analyst']}. One workstream per "
                    "question; each works only from the case evidence.",
                    "",
                ] + [
                    f"- **{q['id']}** ({q['workstream']}): {q['text']}"
                    + (
                        f" — skills: {', '.join(q['skills'])}"
                        if q.get("skills")
                        else ""
                    )
                    for q in questions
                ]
                return _ai_tool_calls(
                    [("write_file", {"file_path": "/plan.md", "content": "\n".join(lines)})]
                )
            if "/plan.json" not in writes:
                return _ai_tool_calls(
                    [
                        (
                            "write_file",
                            {
                                "file_path": "/plan.json",
                                "content": json.dumps(plan, indent=2),
                            },
                        )
                    ]
                )
            return _ai_text(
                "PLAN_READY: the plan is written to /plan.md and /plan.json "
                "and awaits analyst approval at the plan gate."
            )
        if "PHASE: EXECUTE" in human:
            plan = _json_block(human, "PLAN_JSON:")
            workstreams = plan["workstreams"]
            if not _tool_calls(messages, "write_todos"):
                todos = [
                    {
                        "content": f"Run workstream {ws['workstream']} for {ws['question_id']}",
                        "status": "pending",
                    }
                    for ws in workstreams
                ] + [{"content": "Compose the report draft", "status": "pending"}]
                return _ai_tool_calls([("write_todos", {"todos": todos})])
            tasked = set()
            for call in _tool_calls(messages, "task"):
                block = _kv_block(call["args"].get("description", ""))
                if "WORKSTREAM_ID" in block:
                    tasked.add(block["WORKSTREAM_ID"])
            remaining = [
                ws for ws in workstreams if ws["question_id"] not in tasked
            ]
            if remaining:
                calls = []
                for ws in remaining:
                    lines = [
                        f"WORKSTREAM_ID: {ws['question_id']}",
                        f"WORKSTREAM: {ws['workstream']}",
                        f"SUBJECT: {plan['subject']}",
                        f"QUESTION: {ws['objective']}",
                        f"KEYWORDS: {', '.join(ws['keywords'])}",
                    ]
                    prior = ws.get("prior_findings") or []
                    if prior:
                        lines.append(
                            "PRIOR FINDINGS — recorded by the workstream(s) "
                            "this one depends on; starting context only, "
                            "your own findings still come from the documents:"
                        )
                        for finding in prior:
                            lines.append(
                                f"- [{finding['fact_ref'] or 'no ref'}] "
                                f"{finding['claim']} "
                                f"(evidence: {', '.join(finding['evidence_ids'])})"
                            )
                    skill_names = [
                        s["name"] if isinstance(s, dict) else s
                        for s in ws.get("skills") or []
                    ]
                    if skill_names:
                        lines.append(f"SKILLS: {', '.join(skill_names)}")
                        lines.append(
                            "Read each skill's method file at "
                            "/skills/<name>.md before you start; a "
                            "skill is method, never evidence."
                        )
                    lines.append(
                        "Search the corpus, read the relevant documents, "
                        "and record a finding for every Key facts ledger "
                        "entry that answers the question."
                    )
                    description = "\n".join(lines)
                    calls.append(
                        (
                            "task",
                            {
                                "description": description,
                                "subagent_type": ws["workstream"],
                            },
                        )
                    )
                return _ai_tool_calls(calls)
            writes = {
                call["args"].get("file_path")
                for call in _tool_calls(messages, "write_file")
            }
            if "/execution/notes.md" not in writes:
                return _ai_tool_calls(
                    [
                        (
                            "write_file",
                            {
                                "file_path": "/execution/notes.md",
                                "content": (
                                    "Execution complete. Workstreams run: "
                                    + ", ".join(
                                        ws["question_id"] for ws in workstreams
                                    )
                                    + ". Findings are in the case record via "
                                    "record_finding."
                                ),
                            },
                        )
                    ]
                )
            return _ai_text(
                "EXECUTION_COMPLETE: all approved workstreams ran; findings "
                "are recorded and the report draft can be composed."
            )
        return _ai_text("Offline stub model: unrecognised instruction phase.")

    # -- workstream sub-agent ---------------------------------------------
    def _workstream(self, messages: list[BaseMessage], system: str) -> AIMessage:
        first_line = system.splitlines()[0] if system else ""
        workstream = first_line.rsplit("WORKSTREAM:", 1)[1].strip()
        humans = _human_texts(messages)
        block = _kv_block(humans[0]) if humans else {}
        question_id = block.get("WORKSTREAM_ID", "")
        subject = block.get("SUBJECT", "")
        # Governed method first: when the block names skills, the
        # workstream reads each mounted method file before it
        # searches. The scripted model cannot follow prose method —
        # it reads the files because the plan says the methods are
        # loaded, and the loading is what the record proves: the
        # mount, the read on the transcript, and the provenance
        # stamp record_finding puts on this workstream's findings.
        skill_names = [
            name.strip()
            for name in block.get("SKILLS", "").split(",")
            if name.strip()
        ]
        if skill_names:
            read_paths = {
                call["args"].get("file_path")
                for call in _tool_calls(messages, "read_file")
            }
            unread = [
                name
                for name in skill_names
                if f"/skills/{name}.md" not in read_paths
            ]
            if unread:
                return _ai_tool_calls(
                    [
                        ("read_file", {"file_path": f"/skills/{name}.md"})
                        for name in unread
                    ]
                )
        if not _tool_calls(messages, "search_corpus"):
            return _ai_tool_calls([("search_corpus", {"query": subject})])
        search_results = _results_for(messages, "search_corpus")
        evidence_ids: list[str] = []
        if search_results:
            for match in EVIDENCE_ID.findall(search_results[-1]):
                if match not in evidence_ids:
                    evidence_ids.append(match)
        read_calls = _tool_calls(messages, "read_document")
        read_ids = [call["args"].get("evidence_id") for call in read_calls]
        unread = [eid for eid in evidence_ids if eid not in read_ids]
        if unread:
            return _ai_tool_calls(
                [("read_document", {"evidence_id": unread[0]})]
            )
        by_id = _results_by_call_id(messages)
        read_texts = {
            call["args"].get("evidence_id"): by_id[call["id"]]
            for call in read_calls
            if call["id"] in by_id
        }
        recorded_refs = {
            call["args"].get("fact_ref")
            for call in _tool_calls(messages, "record_finding")
        }
        pending: list[tuple[str, dict]] = []
        for evidence_id, text in read_texts.items():
            for fact_id, fact_text in FACT_LINE.findall(text):
                if fact_id in recorded_refs:
                    continue
                scores = {
                    ws: _score(fact_text, ws) for ws in WORKSTREAM_KEYWORDS
                }
                own = scores.get(workstream, 0)
                if own >= 1 and own == max(scores.values()):
                    pending.append(
                        (
                            "record_finding",
                            {
                                "question_id": question_id,
                                "claim": fact_text,
                                "evidence_ids": [evidence_id],
                                "fact_ref": fact_id,
                            },
                        )
                    )
        if pending:
            return _ai_tool_calls(pending)
        return _ai_text(
            f"Workstream {question_id} complete: read {len(read_texts)} "
            f"documents, recorded {len(recorded_refs)} findings."
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = _system_text(messages)
        # The planner's own prompt quotes the WORKSTREAM: block format,
        # so route on the role line each prompt starts with, never on a
        # bare substring.
        if system.startswith("ROLE: WORKSTREAM"):
            message = self._workstream(messages, system)
        else:
            message = self._planner(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])
