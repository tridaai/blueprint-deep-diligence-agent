"""The workstream schedule: dependencies, waves, and handoffs.

A plan's workstreams may declare dependencies on other workstreams
in the same plan (`depends_on`, by question id): a market read that
needs the product facts first, say. The declaration lives on the
plan artifact, and the rules around it live here, in code — not in
the planner's prompt:

- VALIDATION (the plan gate): a malformed plan (no workstreams
  list, an entry without a question id, a workstream outside the
  profile's roster, a brief question left unplanned), an unknown
  dependency, a workstream depending on itself, or a dependency
  cycle refuses plan approval, with the defect named. A plan with no dependencies validates
  trivially and executes exactly as plans always have.
- SCHEDULING: execution runs in topological waves computed here.
  Wave 1 is every workstream with no unmet dependencies; a
  workstream enters a later wave only when all its prerequisites
  sit in earlier waves. The service invokes one harness run per
  wave, so a dependent workstream's sub-agent cannot be spawned
  before its prerequisites have finished — the schedule is enforced
  by when runs start, not by asking the model to wait.
- HANDOFF: when a wave is assembled, each dependent workstream
  receives its prerequisites' recorded findings — claim and
  evidence IDs, verbatim — as starting context, and the handoff
  (which finding IDs went to which workstream) is written to the
  audit trail. Nothing else crosses: no transcripts, no other
  workstream's scratch. Context isolation is the default; the
  handoff is the one declared exception.
"""

from __future__ import annotations


def workstream_dependencies(entry: dict) -> list[str]:
    """The question ids one plan workstream declares it depends on."""
    deps = entry.get("depends_on") or []
    return [str(d) for d in deps]


def plan_shape_defects(
    plan_doc, brief_questions: list[dict], roster: list[str]
) -> list[str]:
    """Every defect in the plan artifact's shape, named — checked
    before the dependency graph, because the graph checks assume a
    well-formed plan.

    The planner is a model, and a model can write a plan that is
    valid JSON but not a plan: the workstreams under another key,
    an entry without a question id, a workstream the profile's
    roster cannot spawn, or a brief question no workstream answers.
    Each of those must stop at the gate, worded for the analyst who
    sends the plan back — never approve into an execution that
    silently answers nothing, and never crash the gate.
    """
    if not isinstance(plan_doc, dict):
        return ["the plan is not a JSON object"]
    workstreams = plan_doc.get("workstreams")
    if not isinstance(workstreams, list):
        return ["the plan has no 'workstreams' list"]
    defects: list[str] = []
    for index, entry in enumerate(workstreams, start=1):
        if not isinstance(entry, dict):
            defects.append(f"workstream entry {index} is not an object")
            continue
        qid = entry.get("question_id")
        if not isinstance(qid, str) or not qid:
            defects.append(f"workstream entry {index} has no question_id")
            continue
        name = entry.get("workstream")
        if name not in roster:
            defects.append(
                f"workstream {qid} names {name!r}, which is not a "
                f"sub-agent in the active profile ({', '.join(roster)})"
            )
        for field in ("depends_on", "keywords"):
            value = entry.get(field)
            if value is not None and not (
                isinstance(value, list)
                and all(isinstance(item, str) for item in value)
            ):
                defects.append(
                    f"workstream {qid} has a {field} that is not a "
                    "list of strings"
                )
        if not isinstance(entry.get("skills", []), list):
            defects.append(f"workstream {qid} has a skills that is not a list")
    if defects:
        return defects
    brief_ids = [q["id"] for q in brief_questions]
    planned = {entry["question_id"] for entry in workstreams}
    for qid in sorted(planned - set(brief_ids)):
        defects.append(
            f"workstream {qid} is not a question in the brief"
        )
    for qid in brief_ids:
        if qid not in planned:
            defects.append(
                f"the plan has no workstream for brief question {qid}"
            )
    return defects


def plan_defects(workstreams: list[dict]) -> list[str]:
    """Every structural defect in a plan's dependency graph, named.

    Empty list = the graph is schedulable. The defects are worded
    for the analyst who has to send the plan back: each names the
    workstream and the problem.
    """
    defects: list[str] = []
    ids = [ws["question_id"] for ws in workstreams]
    known = set(ids)
    if len(known) != len(ids):
        defects.append("the plan lists the same question twice")
    graph: dict[str, list[str]] = {}
    for entry in workstreams:
        qid = entry["question_id"]
        deps = workstream_dependencies(entry)
        graph[qid] = deps
        for dep in deps:
            if dep == qid:
                defects.append(f"workstream {qid} depends on itself")
            elif dep not in known:
                defects.append(
                    f"workstream {qid} depends on {dep}, "
                    "which is not a workstream in this plan"
                )
    # Cycle detection: walk each node's dependency chain; a node
    # that reaches itself again closes a cycle, and the cycle path
    # is named so the analyst can see the loop, not just its fact.
    for start in ids:
        path: list[str] = []
        on_path: set[str] = set()

        def visit(node: str) -> bool:
            if node in on_path:
                cycle = path[path.index(node):] + [node]
                defects.append(
                    "dependency cycle: " + " -> ".join(cycle)
                )
                return True
            if node not in graph:
                return False
            on_path.add(node)
            path.append(node)
            found = any(visit(dep) for dep in graph[node])
            path.pop()
            on_path.discard(node)
            return found

        if visit(start):
            break
    return defects


def waves(workstreams: list[dict]) -> list[list[dict]]:
    """Topological layers of a plan, in plan order within a layer.

    Assumes a validated graph (see plan_defects): layer 1 is every
    workstream with no dependencies; each later layer is the
    workstreams whose prerequisites all sit in earlier layers. A
    plan with no dependencies is a single wave — the pre-schedule
    behaviour, one fan-out.
    """
    placed: set[str] = set()
    remaining = list(workstreams)
    layers: list[list[dict]] = []
    while remaining:
        layer = [
            ws
            for ws in remaining
            if all(dep in placed for dep in workstream_dependencies(ws))
        ]
        if not layer:  # unreachable on a validated graph; fail loudly
            raise ValueError(
                "workstream dependencies cannot be scheduled: "
                + "; ".join(plan_defects(workstreams))
            )
        layers.append(layer)
        placed.update(ws["question_id"] for ws in layer)
        remaining = [ws for ws in remaining if ws not in layer]
    return layers


def build_handoff(findings: list[dict], entry: dict) -> list[dict]:
    """The findings a dependent workstream starts from: exactly the
    recorded findings of its declared prerequisites, verbatim —
    claim, evidence IDs, and fact ref, tagged with the workstream
    they came from. Findings recorded for any other question are
    not the dependent's business and never cross."""
    deps = set(workstream_dependencies(entry))
    if not deps:
        return []
    return [
        {
            "from_question_id": f["question_id"],
            "finding_id": f["finding_id"],
            "claim": f["claim"],
            "evidence_ids": list(f["evidence_ids"]),
            "fact_ref": f.get("fact_ref"),
        }
        for f in findings
        if f["question_id"] in deps
    ]
