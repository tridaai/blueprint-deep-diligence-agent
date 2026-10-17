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
| Harness wiring | `harness.py` | `create_deep_agent` built from the active profile: assembled prompts, per-role tools, the profile's roster |
| Prompts | `harness.py`, `profiles/` | Canonical prompt texts in `harness.py`; the shipped offline profile mirrors them section for section (a test pins the two together) |
| Profiles | `profiles.py`, `profiles/*.json` | Harness profile artifacts: strict fail-closed loader, prompt assembly from named sections, middleware the switches control |
| Evaluation | `evaluation.py` | The case-pack grading (recall, citation, correctness, quarantine, conflicts), shared by the eval runner and the promotion gate |
| Promotion gate | `promotion.py` | Split evaluation, worst-of-three verdicts against the incumbent's recorded baseline, file-hash evidence, append-only promotion/rollback records |
| Budgets | `budgets.py` | Callback guard + budget ledger: per-phase step/token caps, case-level spawn/wall caps, grant/consume/exhaust/amend entries folded from the audit trail |
| Schedule | `schedule.py` | Workstream dependencies: plan-gate graph validation, topological waves, and the findings handoff a dependent starts from |
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
Two continuity moves sit alongside the lifecycle (detailed under
"Long-case continuity" below): a case stranded in `EXECUTING` by a
dead process is **resumed** from its checkpoint rather than
restarted, and an analyst may request a **plan revision** from any
planned state, which returns the case to `PLAN_REJECTED` — the
re-gated path to plan version n+1 — with the superseded draft
preserved.

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
3. **Execution.** The service computes the approved plan's schedule
   (see "Parallel workstreams" below) and runs one harness run per
   wave; inside a run, the planner delegates the wave's workstreams
   via the `task` tool to specialist sub-agents (financial-scan,
   product-scan, market-scan) with isolated contexts. Sub-agents
   search the case evidence, read documents, and record findings
   through `record_finding`, which refuses a claim without a valid
   evidence ID. Findings land directly in the case store, so a
   stopped run keeps what it found.
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
tripped wave's workstream order), or NOT COVERED (no findings
recorded). The banner names the exhausted cap and phase; the table
rides on the report record as `coverage_detail`. Approving a
partial report stamps that table onto the approval record. The
same table — outside the partial protocol, with the reason named
on each uncovered row — appears on a completed run whose
workstreams did not all complete (see below).

**Amendment.** From `BUDGET_EXHAUSTED` only, an analyst may amend
the budget with a recorded reason. The service refuses any cap set
below what the ledger says the case already consumed — per phase
for steps and tokens, case-wide for spawns and wall clock — then
records the amendment (analyst, reason, previous and amended
budgets) as a ledger entry and returns the case to `APPROVED`. The
original exhaustion is never removed; grant, exhaustion,
amendment, and the completing run's consumption read in order.

## Parallel workstreams

Workstreams fan out in parallel, but the fan-out is scheduled, not
improvised (`schedule.py`). A plan workstream may declare
`depends_on` — the question ids whose workstreams must finish
before it starts — carried from the brief onto the plan artifact by
the planner.

**The gate validates the graph.** `decide_plan` runs the plan's
dependency graph through `plan_defects` before approval: an
unknown dependency, a self-dependency, or a cycle (named as a path,
`q1 -> q2 -> q1`) refuses approval with the defect in the message,
and the plan stays at the gate. Execution re-checks the graph
before scheduling it, so an unschedulable plan can never run.

**Waves.** Execution computes topological layers of the graph:
wave 1 is every workstream with no unmet dependencies, and a
workstream enters a later wave only when all its prerequisites sit
in earlier waves. The service invokes one harness run per wave,
with that wave's workstreams as the run's whole plan. A dependent
workstream therefore cannot be spawned early — the run that would
spawn it has not been invoked — and inside a wave the workstreams
run as the harness's parallel fan-out with isolated contexts, as
before. A plan with no dependencies is a single wave, which is the
execution this blueprint always had. Start and finish are audit
events per workstream (`workstream_started` /
`workstream_finished`), and `execution_started` records the wave
layout, so the schedule is provable from the trail after the fact.

**Handoff.** When a wave is assembled, each dependent workstream
receives exactly its prerequisites' recorded findings — claim,
evidence IDs, fact ref, verbatim — attached to its plan entry as
`prior_findings` and passed by the planner into the sub-agent's
task description under a PRIOR FINDINGS heading. The audit records
the handoff (`workstream_handoff`: dependent, prerequisites, the
finding ids carried). Nothing else crosses between workstreams: no
transcripts, no scratch. Isolation is the default; the handoff is
the one declared exception.

**Statuses and failure isolation.** Every workstream ends an
execution with a status on the report record (and at
`GET /api/cases/{id}/workstreams`): `completed` (findings in the
record), `empty` (ran, recorded none), `failed` (its wave's run
raised and it recorded none; the error is the reason),
`interrupted` (a budget stop inside it), or `not_run` (a budget
stop before it). If a wave's run raises, the wave's finding-less
workstreams are marked failed with the error named, and later
waves still run — a dependent of a failed prerequisite runs with
whatever handoff exists, possibly none, recorded as such. The
case still composes and drafts; the failed or empty branch's
question appears in the report's coverage table as NOT COVERED
with the reason, so one branch's failure is a reported fact, never
a silent gap or a failed case.

**Determinism.** Findings are stored in recording order, which
under parallel workstreams is completion order. The composer does
not compose in that order: it sorts the finding set by the brief's
question order, then fact ref, then claim text, and assigns claim
ids in that order. The report is therefore a function of what was
found, not of who finished first — repeated executions of the same
case compose byte-identical drafts (a test runs the dependency
case twice in fresh stores and diffs them, and another records
findings in scrambled order and checks the composed order).

## Long-case continuity

Everything in this section is record-based: no continuity behaviour
depends on a process staying alive, because the cases that need it
are exactly the ones where a process does not.

**Checkpoints.** `_execute` — the single wave loop behind fresh
executions, resumes, and re-executions — opens a checkpoint when
execution starts and rewrites it after every wave:
`execution_checkpoints` holds one row per case with the plan
version, the attempt number, the wave layout, per-workstream
entries (status, finding count, finding IDs, wave, reason), a
budget snapshot folded from the ledger at write time, and
`complete`/`outcome` once the execution ends by design
(`draft_ready` / `budget_exhausted`). A process that dies
mid-execution closes nothing — which is precisely the signal the
rest of the machinery reads.

**Interrupted cases and resume.** A case in EXECUTING whose
checkpoint is missing or incomplete is interrupted.
`interrupted_cases()` scans for them and reports completed vs
pending questions from the checkpoint; the first scan after an
interruption writes `execution_interruption_detected` to the audit
trail, and later scans of the same interruption (recognised as
those after the latest `execution_started`) read it back without
writing it again. `resume_execution` is an analyst action and a
gate — anything but EXECUTING is refused. Workstreams the
checkpoint records as `completed` or `empty` are carried into the
execution record without running: their findings stand, under
their original IDs (finding IDs are minted under the store lock
from the case's own sequence, and `record_finding` dedups by
(question, fact_ref), so even a wave that is re-run cannot
duplicate a fact's finding). Only incomplete waves execute.
Budget carry-over needs no dedicated machinery: every run's guard
already seeds its case-level caps from the ledger fold, and the
interrupted attempt's `budget_consumed` checkpoints are in that
ledger — a resume cannot re-spend spawns the dead run used. The
honest edge: a wave that died mid-flight consumed steps after its
last completed wave, and those never reached the ledger; the
checkpoint's snapshot is the position at the last completed wave,
and `execution_resumed` carries that snapshot in its payload so
the record shows exactly what the resume inherited.

**Evidence arrival and staleness.** `add_case_evidence` ingests an
analyst-supplied document under intake's rules (object store,
SHA-256, the case's next evidence ID) with `origin:
analyst_upload` on the record. On a case past INTAKE,
`continuity.stale_questions` scores the document against each
workstream with the governed keyword stems from `corpus.py` — the
same lists the offline reader routes facts by — and requires a
score of at least 2: the stem lists overlap by design ("adopt"
sits in market-scan's list and inside every product document's
"adoption"; "fund" sits inside "funded"), so a single shared stem
is cross-talk, not signal, and marking a workstream stale
interrupts the desk. Marks land on the case record as
`stale_marks` (evidence ID, question, workstream, score, time);
`run_execution` refuses while any are open. `decide_staleness`
records the answer on the case either way: **proceed** clears the
marks, appends the decision to `staleness_decisions`, and
re-renders the current draft so the decision appears in its
Method and limits; **re-plan** routes into `replan_case` with the
decision attached, and the next draft's limits carry it the same
way.

**Re-plan and the reuse rule.** `replan_case` snapshots the
current draft into `report_snapshots` before anything changes
(plans are already versioned; reports were not), clears open
staleness into the decision record, and returns the case to
PLAN_REJECTED, from which `run_planning` mints plan version n+1 —
with the analyst notes in its context — and the revision faces
the plan gate like any plan. Reuse is set only when the revision
answers staleness marks: the case carries `reuse_findings` plus
the marked questions as `must_rerun`, and at the next execution
`_reuse_skip` carries a workstream's recorded findings into the
new execution only when all three hold — the question is
unmarked, it holds findings, and every evidence item those
findings cite still hashes to the SHA-256 recorded when the
evidence arrived. Anything else re-runs. The directive is
one-shot, cleared by whichever execution it governed. A general
revision sets no directive and re-executes in full: reuse is a
staleness answer, not a default, because a scope change the desk
chose deliberately should not silently inherit findings recorded
under a different plan.

**Analyst notes.** Notes are the one channel by which what the
desk learned outside the documents reaches the next plan, and
they are engineered to be unable to become evidence: their own
table (`case_notes`, IDs `NOTE-####`), no presence in the case
filesystem, and citation validation that resolves IDs against the
evidence table only. They enter the planner's context solely as
the bounded section `harness.notes_section` builds — most recent
10, each truncated at 500 characters, under a heading that states
they are context and never evidence — and only when a plan is
(re-)made, never during execution.

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
`case_files`, `audit`, `execution_checkpoints` (one row per case),
`case_notes`, `report_snapshots`. Case records carry the active harness
profile's stamp (id, version, target model) from intake, and their
continuity state: open `stale_marks` and the append-only
`staleness_decisions`. The audit
trail records case creation, evidence ingestion, plan generation
and decisions, sub-agent spawns, budget ledger entries (grants —
each naming the profile the phase ran under — consumption
checkpoints, exhaustions, amendments), composition, verification
and quarantine events, conflict detections and NOTED decisions,
evidence-drift detections, and report decisions — plus the
continuity events: mid-case evidence arrival and staleness
marking, staleness decisions, plan revisions and report snapshots,
findings reuse, interruption detections, resumes, and case notes.

## Models

One boundary (`llm.py`): `offline` returns the scripted stub; the
real-provider backends build `ChatOpenAI` (any OpenAI-compatible
endpoint, including NVIDIA NIM and Ollama's compatible surface) or
`ChatAnthropic` from the `llm` extra. Prompts are written for real
models; the stub keys its deterministic playbook off the same phase
and workstream markers, and reads facts from the corpus documents'
Key facts ledgers. LangSmith is not required anywhere — tracing is
the audit trail plus structured JSON logs with request IDs.

## Harness profiles

A harness profile is a versioned JSON artifact in `profiles/`
holding, for one target model: the prompt sections and the order
they assemble in (planner sections, and per sub-agent a section
list whose text may carry the `{workstream}` placeholder), a
roster of sub-agents (name, description, prompt sections, and a
tool allowlist over the blueprint's case-tool registry), and the
middleware switches (`context_compression`,
`large_result_offloading`, `offload_threshold_chars`).

**Selection and recording.** `PROFILE_ID` (default `offline-stub`)
names the bootstrap artifact; once a promotion is recorded, the
newest promotion record names the active one instead (see the
regression gate, below). The service resolves the active profile
at startup and stamps its id + version + target onto every case record, onto
the `case_created` audit entry, and onto each phase's
`budget_granted` entry — planning and execution are both attributed
to the profile that ran them, in the case view, the console, and
the audit trail. Switching profiles is a configuration change; the
model is injected separately by `llm.py`, so any profile can be
exercised with the stub model standing in for its target.

**Fail-closed validation.** The loader rejects, with a named
`ProfileError` per problem: unknown keys at any level, missing
required keys, a planner or roster entry referencing a prompt
section that is not defined, a roster entry naming no section at
all, a tool name outside the case-tool registry, a middleware
switch left unset, a duplicate roster name, an id that does not
match its file name, and a file that is not valid JSON. The service
constructs nothing until its profile has loaded — a bad profile is
a refused startup, never a silent fallback to defaults.

**How the switches act.** Prompts, tool lists, and the roster are
passed straight to `create_deep_agent`, so their effect is the
build itself. The middleware switches work through the caller
middleware the harness merges into its stack by name. Context
compression off passes a pass-through middleware named
`SummarizationMiddleware`, which takes the built-in compression
middleware's slot in place — the assembled stack carries no
compression. Large-result offloading on adds this blueprint's own
middleware: a tool result over the profile's threshold is written
in full to `/offloaded/` in the case filesystem and the model sees
a pointer plus a preview. Under the offline profile's switches
(compression on, offloading off) the contributed middleware list is
empty — the default build is unchanged.

**Two profiles ship.** `offline-stub` v1 reproduces the
pre-profile build byte-for-byte (its sections are the canonical
prompts in `harness.py`; a test pins them equal). `openai-gpt` v1
targets `openai:gpt-5.1`: model-tuned suffix sections on the
planner and workstreams, context compression off with offloading
on (threshold 20,000 chars), and a market-scan whose allowlist
drops `read_document` — it works from search results only. Run
under the eval packs with the stub standing in, its gold-pack
recall falls to 0.700–0.769 while citation discipline holds: the
tuning's cost, attributed to the profile that caused it. Whether
such a profile may replace the default is decided by the profile
regression gate, next.

**Why not the package's own profiles.** `deepagents` resolves its
internal `HarnessProfile` from the model's identity through a
process-global, additive registry. That cannot express this
blueprint's selection rule — the profile named in configuration
runs the build, whatever model is injected — and a global registry
would let one service's tuning leak into another's. The blueprint
therefore keeps its own artifacts and applies them through the
public seams above.

## Profile regression gate

The gate (`promotion.py`) is how a profile stops being an artifact
someone configured and becomes the profile production runs. It is
a workflow over append-only records, not a setting.

**The split.** The eval packs are partitioned in the eval data
(`corpus.py`, `PACK_SPLITS`) — Helvion and Copperline are the
TUNING subset, Selene and the citation-integrity pack are
HELD-OUT. The split is read off the case records; no run chooses
it. The integrity pack is held out on purpose: its stale-figure
trap and poisoned findings are the behaviour a promotion must not
be able to rehearse against.

**The evaluation.** Judging a candidate takes `PROMOTION_RUNS`
(3) full suite passes, because model behaviour is stochastic;
every run's per-subset metrics are recorded, and the verdict reads
the worst run. The bar is the incumbent's own recorded baseline —
the worst of the incumbent's recorded runs, per subset. Recorded
baseline runs are reused while the incumbent's file hash still
matches them; a first evaluation on a fresh store pays for the
baseline runs itself. The grading is `evaluation.evaluate_suite`,
the same code the eval runner gates on.

**The rule.** On every candidate run: tuning recall ≥ the
baseline bar (no tolerance — the tuning cases are the ones the
author iterated against); held-out recall ≥ the bar minus
`HELD_OUT_TOLERANCE` (0.02); and the zero-tolerance lines hold on
both subsets — unsupported claims in final reports = 0, citation
completeness = 1.000, citation correctness = 1.000. Any failure
makes the evaluation NOT_PROMOTABLE, with each failure naming the
run, subset, rule, and the observed vs required numbers.

**The evidence.** An evaluation record carries the candidate's
id + version, the incumbent it was judged against, all runs, the
baseline and its source, the verdict — and the candidate profile
file's SHA-256 at evaluation time. At promotion, the file is
hashed again: a mismatch refuses the promotion, because the
evidence no longer describes the file. An evaluation judged
against a profile that is no longer the incumbent is likewise
stale and refused.

**The decision.** Promotions live in `profile_promotions`
(migration 0005), append-only like the evaluations; the store
offers no update or delete for either table. The active profile
is resolved at startup from the newest promotion record, and a
live promotion moves the pointer for new cases only — existing
cases keep their stamp. `PROFILE_ID` is the bootstrap default for
a store with no promotions. There is no code path that sets the
active profile directly: the service exposes evaluate, promote,
and rollback, and the API and console surface exactly those.

**Rollback.** Rolling back promotes the previous profile again,
as its own record naming the record it reverses. It is judged, not
rubber-stamped: the restored profile's recorded runs (found in
the evaluation history, as that evaluation's candidate or its
baseline, hash-checked against the file as it stands) must pass
the same rule against the current incumbent's recorded runs. A
profile whose file changed since its runs were recorded cannot
be rolled back to — it must be evaluated as a candidate instead.

**The controls.** Two controls run against the real suite, in
`tests/test_promotion.py` and standalone as
`evals/promotion_controls.py`. NEGATIVE: `openai-gpt` against
`offline-stub` returns NOT_PROMOTABLE on all three runs — tuning
recall 0.750 and held-out 0.812 against a 1.000 baseline, failing
both recall rules, its zero-tolerance lines intact. POSITIVE: a
copy of the incumbent under a new id and version (identical in
every behavioural field) returns PROMOTABLE and completes a
promotion + rollback cycle. A third control tampers with a
candidate file after evaluation and shows promotion refused on
the hash check.

## Deliberate limits of this round

- Harness profiles shape this blueprint's own seams — prompts,
  case-tool allowlists, the roster, caller middleware. The
  harness's built-in filesystem tools are required scaffolding in
  `deepagents` and are not a profile's to grant or withhold;
  likewise the middleware switches act on the assembled stacks
  through the caller-middleware merge, not on the package's
  internal per-model profile registry, which this blueprint
  deliberately does not use.
- The regression gate grades on the deterministic stub over the
  offline corpus, like every gate in this blueprint: it proves the
  promotion discipline and measures profile behaviour under the
  stub standing in. A promotion decision for a live deployment
  should rest on a live-mode evaluation of the same packs (the
  runner's per-profile mode is the same ruler pointed at a real
  model), and the gate's records say which model produced the
  evidence they hold.
- A rollback rests on recorded runs only. If the previous
  profile's file was edited after its runs were recorded, the hash
  check finds no evidence for the file as it stands and the gate
  refuses the rollback — the profile re-enters as a candidate
  through a fresh evaluation instead.
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
