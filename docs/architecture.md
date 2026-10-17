# Architecture — Deep Diligence Agent

## The split that defines the design

Two layers, with a hard boundary between them:

- **The harness** (LangChain `deepagents`, pinned in pyproject) does
  the deep work: it plans with `write_todos`, decomposes the brief,
  delegates to sub-agents with the `task` tool, works in a virtual
  filesystem, and manages context with its middleware stack.
- **The governance shell** (this repo's own code) owns everything the
  desk is accountable for: case scope, the plan gate, budgets,
  evidence, citations, the report gate, and the audit trail.

The harness never decides what a case may spend, which sources are in
scope, or whether a report is final. Those are enforced in code around
it, so they hold for any model the harness runs.

## Components

| Piece | File | Responsibility |
|---|---|---|
| API | `api.py` | FastAPI surface + console; gates surface as 409 |
| Service | `service.py` | Case lifecycle state machine; the plan/report gates |
| Harness wiring | `harness.py` | `create_deep_agent` with planner + 3 workstream sub-agents |
| Prompts | `harness.py` | Planner and workstream system prompts (phase/workstream markers) |
| Budgets | `budgets.py` | Callback guard: steps, sub-agents, tokens, wall clock |
| Tools | `tools.py` | `search_corpus`, `read_document`, `record_finding` (citation rule) |
| Backend | `backend.py` | Deep Agents filesystem protocol over the evidence store |
| Evidence | `evidence.py` | Corpus ingestion: IDs, source, hash, retrieval time |
| Composer | `composer.py` | Deterministic report assembly from findings |
| Ledger | `ledger.py` | Citation ledger: claim→evidence entries with citation-time hashes; re-verification |
| Store | `store.py` | Case store port: SQLite (local) / PostgreSQL (production) |
| Objects | `objects.py` | Object-store port: local dir / S3-compatible |
| Corpus | `corpus.py` | Generated evidence corpus + gold case packs |
| Models | `llm.py`, `stub_model.py` | Provider boundary; the deterministic offline stub |

## Lifecycle

```
INTAKE → (planning run) → PLAN_READY → approve → APPROVED
       → (execution run) → DRAFT_READY → approve → FINAL
```

Rejection at either gate parks the case in `PLAN_REJECTED` /
`REPORT_REJECTED`, from which the phase may run again (a re-plan mints
plan version n+1). A budget-exhausted execution composes a **partial**
draft and lands in `BUDGET_EXHAUSTED`; the report gate still applies.

## A run, step by step

1. **Intake.** The brief (subject, questions with workstreams, allowed
   sources, budget) is validated; the scoped corpus documents become
   case evidence — bytes in the object store, metadata (ID, source,
   hash) in the case store.
2. **Planning.** The planner agent writes `/plan.md` and `/plan.json`
   into the case filesystem (one workstream per question). The plan is
   stored as a versioned artifact. Nothing else can happen until an
   analyst approves it — the state machine refuses execution from any
   other state.
3. **Execution.** The planner delegates each workstream via the `task`
   tool to a specialist sub-agent (financial-scan, product-scan,
   market-scan) with an isolated context. Sub-agents search the case
   evidence, read documents, and record findings through
   `record_finding`, which refuses a claim without a valid evidence
   ID. Findings land directly in the case store, so a stopped run
   keeps what it found.
4. **Composition.** Deterministic code groups findings by question,
   re-verifies every citation against the case evidence (a claim that
   fails is refused and named), and renders the draft. Each composed
   claim gets a claim ID and a **citation-ledger** entry: the finding
   behind it, and for each cited evidence item the SHA-256 of its
   bytes at citation time, read live from the object store. The
   report's References section is generated from the ledger. The
   composer — never the model — writes the report.
5. **Report gate.** An analyst approves or rejects. Approval first
   re-verifies the ledger: every cited item's current bytes must
   hash to the value recorded at citation time. A mismatch means the
   evidence changed after the fact — the report is flagged
   **EVIDENCE_DRIFT**, the drift is named in the audit trail, and the
   approval is refused until a re-verification comes back clean.
   Approval is what makes a report final; the agent itself never
   issues a recommendation.

## Budgets

`BudgetGuard` is a LangChain callback handler with `raise_error` set:
crossing a cap raises `BudgetExceededError` inside the run, aborting
it. Counters: model steps (deduplicated on run id, because chat
models fire `on_chat_model_start` and some stacks also fire
`on_llm_start`), tokens (from `usage_metadata`, with a character
estimate when a provider omits it), `task` spawns, and wall clock.
Caps apply **per run** — planning and execution are each guarded by
the case budget. Exhaustion is an audit event, and the partial-report
protocol names the uncovered questions rather than truncating
silently.

## The virtual filesystem

`EvidenceBackend` implements the Deep Agents `BackendProtocol` over
the object store plus a `case_files` index in the case store. Case
files (`/plan.json`, `/execution/notes.md`, …) are writable; evidence
is mounted read-only at `/evidence/<EV-ID>.md`, so the agent reads
sources through the same `read_file` tool it uses for its own notes,
and cannot alter them — a write to `/evidence/` returns an error.
File metadata (path, SHA-256, size) is indexed in the store; bytes
live in the object store.

## Data model

Payload-JSON tables, Postgres and SQLite variants of the same schema
(numbered migrations in `migrations/` and `migrations/sqlite/`):
`cases`, `plans` (versioned), `evidence`, `findings`, `reports`,
`citation_ledger`, `case_files`, `audit`. The audit trail records case creation,
evidence ingestion, plan generation and decisions, sub-agent spawns,
budget events, composition, evidence-drift detections, and report decisions.

## Models

One boundary (`llm.py`): `offline` returns the scripted stub; the
real-provider profiles build `ChatOpenAI` (any OpenAI-compatible
endpoint, including NVIDIA NIM and Ollama's compatible surface) or
`ChatAnthropic` from the `llm` extra. Prompts are written for real
models; the stub keys its deterministic playbook off the same phase
and workstream markers, and reads facts from the corpus documents'
Key facts ledgers. LangSmith is not required anywhere — tracing is
the audit trail plus structured JSON logs with request IDs.

## Deliberate limits of this round

- Contradictions are planted in the corpus and both sides surface as
  findings; explicit conflict records in the report are a later round.
- Web fetch is not wired; the evidence universe is the case corpus.
- The verifier pass (independent claim re-check by a second agent)
  is a later round; the citation defence today is record-time
  enforcement, composer re-verification, and the ledger's
  deterministic hash re-verification at the report gate — none of
  which re-reads a claim against its evidence for *meaning*, only
  for presence, resolution, and byte identity.
