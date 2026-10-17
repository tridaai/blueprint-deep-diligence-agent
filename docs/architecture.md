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
| Budgets | `budgets.py` | Callback guard + budget ledger: per-phase step/token caps, case-level spawn/wall caps, grant/consume/exhaust/amend entries folded from the audit trail |
| Tools | `tools.py` | `search_corpus`, `read_document`, `record_finding` (citation rule) |
| Backend | `backend.py` | Deep Agents filesystem protocol over the evidence store |
| Evidence | `evidence.py` | Corpus ingestion: IDs, source, hash, retrieval time |
| Composer | `composer.py` | Deterministic report assembly from findings; re-renders the draft from supported claims after verification |
| Ledger | `ledger.py` | Citation ledger: claim→evidence entries with citation-time hashes; re-verification |
| Verifier | `verifier.py` | Independent claim re-check: reads evidence bytes from the object store and re-derives support; quarantines unsupported claims |
| Conflicts | `conflicts.py` | Deterministic keying of supported claims by (entity, metric, period); conflict records with both sides, source classes, and recency labels |
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
draft and lands in `BUDGET_EXHAUSTED`; the report gate still applies,
and the analyst has one further move from that state — **amend the
budget** (a recorded raise, never below what the case has already
consumed), which returns the case to `APPROVED` for re-execution.

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
5. **Verification.** Before the draft reaches the analyst, the
   verifier re-checks every claim against the evidence it cites. It
   is independent by construction: it re-reads the persisted report
   and fetches each cited item's bytes directly from the object
   store, never touching the composer's structures or the citation
   ledger. The gate checks are deterministic — every figure and date
   a claim states must appear in its cited evidence, and every named
   entity it mentions must be mentioned there; when a real model is
   configured, an advisory model check (claim + evidence only, no
   other context) is recorded alongside, but never decides. A claim
   that fails is **quarantined**: removed from the report body (the
   composer re-renders from the supported claims and rebuilds the
   ledger), recorded with its reason, and summarised in the draft's
   Verification section. The analyst's only moves on a quarantined
   claim are *send back for re-investigation* — the case returns to
   APPROVED with the note in the audit trail — or *drop it*. There
   is no path that approves a quarantined claim into the report.
6. **Conflict detection.** Verification judges claims one at a time;
   disagreement is a relation between claims, so it is detected by a
   separate deterministic pass over the supported claims. Each claim
   is reduced to assertions — (metric, period, value) — extracted in
   code: a governed metric lexicon finds the metric mentions, figures
   are typed (money / percent / count) and assigned to the nearest
   mentioned metric of their type, and each figure takes the period
   stated nearest it, normalised to the year (FY2025 and 2025 are
   the same period; FY2024 is a different one — period-distinct
   facts are different keys and never conflict). Where two supported
   claims assert different values for the same (entity, metric,
   period), a **conflict record** is stored holding both sides: the
   claims, their evidence, both values verbatim, each side's source
   class applied in code from the evidence kind (primary — filings,
   regulatory notes, official reports — ranks above secondary —
   news-style coverage, vendor product and pricing material), and a
   more-recent marker on the side resting on the later-dated
   document. The composer renders a Conflicts section from the
   records. The labels are context, not resolution: no side is
   suppressed, merged, or preferred, and the same value asserted by
   two sources is corroboration — its claims sit together on one
   side. The analyst's only move is to mark the conflict **NOTED**
   with a note, which re-renders into the report; there is no
   delete or resolve path.
7. **Report gate.** An analyst approves or rejects. Approval first
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

The brief's budget has a **phase structure**. Steps and tokens are
per-phase caps: `resolve_phase_budget` resolves the caps for the
planning run and the execution run independently — a phase override
in the brief wins, an unset field derives from the brief's flat
value, so a brief written before phases existed resolves exactly as
it used to. Spawns and wall clock are **case-level** caps, consumed
across phases: the guard is seeded with the case's prior consumption
(folded from the ledger) and checks `task` spawns and elapsed wall
time against prior-plus-current, so a re-execution cannot re-spend
spawns an earlier attempt already used.

Every budget-relevant event is a **ledger entry** in the case's
audit trail, in order: `budget_granted` (phase, its resolved caps,
the case caps) when a phase run starts; the guard's own
`subagent_spawned` and `budget_exceeded` events (the exceeded entry
names the cap, the phase, the limit, the observed count, and a
usage snapshot); `budget_consumed` (the run's final usage) when the
run ends; and `budget_amended` when an analyst amends.
`fold_budget_ledger` folds those entries into per-phase and
case-level consumption, and `GET /api/cases/{id}/budget` renders
granted-vs-consumed from that fold — there is no separate mutable
counter that could drift from the ledger.

**Partial-report protocol.** When execution stops on a cap, the
composer itemises coverage per brief question, derived in code from
the report's claims and the plan's workstream mapping: ANSWERED
(findings in the report, workstream not interrupted), PARTIAL
(findings exist, but the run stopped mid-workstream — the guard
tracks which `task` calls were still open when the cap tripped,
and the service maps those spawns back to questions through the
plan's workstream order), or NOT COVERED (no findings recorded).
The banner names the exhausted cap and phase; the table rides on
the report record as `coverage_detail`. Approving a partial report
stamps that table onto the approval record.

**Amendment.** From `BUDGET_EXHAUSTED` only, an analyst may amend
the budget with a recorded reason. The service refuses any cap set
below what the ledger says the case already consumed — per phase
for steps and tokens, case-wide for spawns and wall clock — then
records the amendment (analyst, reason, previous and amended
budgets) as a ledger entry and returns the case to `APPROVED`. The
original exhaustion is never removed; grant, exhaustion,
amendment, and the completing run's consumption read in order.

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
`citation_ledger`, `verifier_reports`, `quarantine`, `conflicts`,
`case_files`, `audit`. The audit trail records case creation,
evidence ingestion, plan generation and decisions, sub-agent spawns,
budget ledger entries (grants, consumption checkpoints,
exhaustions, amendments), composition, verification and quarantine
events, conflict detections and NOTED decisions, evidence-drift
detections, and report decisions.

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

- Conflict detection is only as wide as the governed metric
  lexicon: a disagreement stated without a recognised metric,
  figure, or comparable value sits in the report as two findings and
  is not flagged as a conflict.
- Web fetch is not wired; the evidence universe is the case corpus.
- The verifier checks what the cited evidence *says* only in the
  deterministic sense: figures, dates, and named entities present in
  the text. It does not judge narrative plausibility or reconcile
  conflicting sources — reconciliation is the conflict pass's job
  only in the sense of surfacing the disagreement as a record; the
  system never decides which side is true. A claim whose figure
  matches a *stale* cited document passes the verifier while
  failing the eval pack's citation-correctness grading. The
  optional model check is advisory and recorded; it is never the
  gate.
