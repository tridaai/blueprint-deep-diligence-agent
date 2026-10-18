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
| Budgets | `budgets.py` | Callback guard + budget ledger: per-phase step/token caps, case-level spawn/wall caps, grant/consume/exhaust/amend entries folded from the audit trail; the guard also records each model call (phase, role, model, token split, usage basis) for cost accounting |
| Costs | `costs.py`, `pricing/price-table.json` | Cost accounting: the governed price table (per-model rates as versioned data, fail-closed loader), run-end pricing of the guard's records into `cost_recorded` ledger entries pinned to table version + rates, the cost fold by phase/role/model, and the budget-vs-actual view |
| Schedule | `schedule.py` | Workstream dependencies: plan-gate graph validation, topological waves, and the findings handoff a dependent starts from |
| Tools | `tools.py` | `search_corpus`, `read_document`, `record_finding` (citation rule) |
| Backend | `backend.py` | Deep Agents filesystem protocol over the evidence store |
| Evidence | `evidence.py` | Corpus ingestion: IDs, source, hash, retrieval time |
| Composer | `composer.py` | Deterministic report assembly from findings; re-renders the draft from supported claims after verification; its finalize step builds the report model and renders every format from it |
| Report model | `report_model.py` | The report as a structured model: executive summary from supported claims under the integrity check, per-question support grades, risks and open items from records, the print-ready HTML renderer, and the JSON export (model + section annotations) |
| Ledger | `ledger.py` | Citation ledger: claim→evidence entries with citation-time hashes; re-verification |
| Verifier | `verifier.py` | Independent claim re-check: reads evidence bytes from the object store and re-derives support; quarantines unsupported claims |
| Conflicts | `conflicts.py` | Deterministic keying of supported claims by (entity, metric, period); conflict records with both sides, source classes, and recency labels |
| Workbench | `workbench.py` | The analyst queue derived from case records (grouped by act needed, oldest first), the merged case timeline, and deterministic case search; saved views in the store |
| Observability | `observability.py` | The run trace folded from the stored records (runs from the budget/cost ledgers, every audit entry as one act), the failure taxonomy (one function, total by construction), and the operations digest fold — stored versioned by the service, never a counter |
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
   evidence ID — IDs resolve within the case only, so another
   case's documents are unreachable — and refuses a claim whose
   fact reference the cited documents' Key facts ledgers do not
   state, or whose words rewrite the ledger's. Findings land
   directly in the case store, so a stopped run keeps what it
   found.
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
   a claim states must appear in its cited evidence, judged against
   each document's own Key facts ledger where it carries one (the
   ledger is what the document asserts as fact; prose alone —
   a pro-forma aside, a number the document's own table
   contradicts — supports nothing), and every named entity it
   mentions must be mentioned in the full text; when a real model is
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
7. **Challenge pass.** A deterministic second opinion over the
   verified draft, computed from the stored records — the claims,
   their evidence metadata, the conflict records, the grades — and
   stored as **challenge records**: each names the claim, question,
   or draft it targets, the governed rule that fired, and the
   evidence the rule read. The rules look for what verification
   and conflict detection do not test: a figure claim resting on a
   single source with no corroboration, a claim leaning on the
   oldest document in a live conflict, a grade carried by one
   document repeated, one document behind whole questions. A
   headline financial figure on a single secondary source is
   **must-answer**; the rest are **advisory**. The full design is
   under "The challenge pass" below.
8. **Report gate.** An analyst approves or rejects. Approval first
   re-verifies the ledger: every cited item's current bytes must
   hash to the value recorded at citation time. A mismatch means the
   evidence changed after the fact — the report is flagged
   **EVIDENCE_DRIFT**, the drift is named in the audit trail, and the
   approval is refused until a re-verification comes back clean.
   Approval is also refused while a redaction flag stands open on a
   claim in the report, and while any must-answer challenge is
   unanswered — the refusal names the open challenges, and each is
   answered on its own record (rebutted with counter-evidence,
   sent back for re-investigation, or downgraded), never waved
   through at the gate. Approval is what makes a report final; the
   agent itself never issues a recommendation.

## The report model

Composition ends in a finalize step (`composer.apply_conflicts`,
backed by `report_model.py`): from the supported claims, the
citation-ledger entries, the conflict records, and the case's
governance records, it builds a **structured report model** and
renders every format from that one model — the markdown draft,
a self-contained print-ready HTML page (inline CSS, house style,
no external assets), and the JSON export. The model and the HTML
ride on the report record; the export is the model plus the
analyst's section annotations. Every later re-render (a conflict
noted, staleness proceeded on, a quarantined claim dropped, a
redaction flag acknowledged) passes through the same step, so the
model and its renderings always describe the records as they now
stand.

The model's **executive summary** is composed after verification,
from the supported-claim set alone: one headline claim per
question in brief order, plus structural counts computed in code
(claims supported and quarantined, conflicts open and noted,
questions uncovered and partial). The hard rule is enforced at
composition time with the verifier's own extraction machinery
(`verifier.check_claim`): a summary sentence may state no figure,
date, or named entity that does not appear in a supported claim.
A failing sentence is dropped and the drop recorded on the report
record (`executive_summary.dropped`); there is no rewrite path.
The eval runner re-checks the invariant independently on every
pack — summary figures, dates, and entities ⊆ the supported
claims', zero violations tolerated — and grades cross-format
identity: every claim in the model must appear in both the
markdown and the HTML.

Each question carries a **support grade**, computed by the single
function `report_model.question_grade` from the question's
supported claims, their evidence's source classes, and the
verifier report's model-check marks: UNANSWERED (no supported
claims), THIN (a claim the model check marked unsupported, or a
single claim standing alone), STRONG (two or more claims with at
least one on a primary source), STANDARD (the rest). The grade
renders in the section header in every format.

The **Risks and open items** section is generated entirely from
records: conflict records (open and noted), quarantine records
with their dispositions, coverage rows that are not ANSWERED,
staleness-proceed decisions, and redaction flags. Quarantined
claims are referenced by ID, reason, and disposition only — never
quoted: a quarantined claim's text reaches no format, the same
zero-tolerance rule the body lives under. When every set is
empty, the section states that plainly.

**Section notes** are an analyst's annotations on a report
section (a question id, or a named section: executive summary,
conflicts, risks and open items, references, verification,
method and limits). They are stored in their own table
(`report_section_notes`, migration 0008), apart from the report
record: adding or editing one writes no report field, so a note
can never change the report body or its hash. Their only surface
is the JSON export's `annotations` field.

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

## Cost accounting

The budget ledger says what a case consumed; the cost ledger
(`costs.py`) prices it, in the same recorded style — no new store
table, everything folded from the audit trail.

**Recording.** The BudgetGuard is the seam: at `on_llm_end` it
already totalled a call's tokens for the budget, and it now also
keeps a per-call consumption record — phase, role, model, and the
input/output split. The role comes from the harness's own naming
(the callback metadata's agent name: the planner's agent reads as
the main agent, a sub-agent's roster name is its role; the
assembled system prompt is the fallback), and the model from the
provider boundary's label (`llm.model_label`, the same
provider:model shape the price table keys on). A provider total
that carries tokens its split does not itemise keeps the
remainder as `other_tokens`: counted in the budget total, never
priced as input or output. Each record carries the basis of its
figures — `measured` when a real provider returned usage
metadata, `estimated` for the offline stub's declared per-call
figures (150 input tokens a call; 40 output for a tool call, 60
for text — the stub's own scripted numbers, labelled estimated
however they arrive) and for the character estimate used when a
provider returns no usage at all. A record's token total is
exactly the number the guard counted against the budget, so the
cost ledger's tokens reconcile with the budget ledger's by
construction, phase for phase.

**Pricing.** Rates live in a governed price table,
`pricing/price-table.json`: per-model input/output rates in USD
per million tokens, held as exact decimal strings under a version
stamp, loaded fail-closed at startup with the profile loader's
discipline (unknown keys, duplicate models, float rates, an id
that does not match the file all refuse). The shipped v1 prices
the OpenAI and Anthropic defaults at their providers' published
list rates (checked 2026-10-17), the self-hosted Ollama model at
zero (no per-token charge exists; the table says why), and the
offline stub at a shadow rate — the table's gpt-5.1 list rate —
so offline runs can be compared with real ones, always labelled
estimated. At the end of every run the service prices that run's
records with Decimal arithmetic and writes a `cost_recorded`
audit entry beside the `budget_consumed` checkpoint: the phase,
the table's id + version, and the priced lines aggregated by
(phase, role, model), each carrying its rates, its input/output
costs, its usage basis, and — for a model the table does not
list — `priced: false` with the reason, its tokens counted and
its cost absent rather than invented.

**The fold and the report.** `fold_cost_ledger` reads only
`cost_recorded` entries and aggregates their lines into totals
and breakdowns by phase, by role, and by model. It never consults
the current price table: an edited or re-versioned table prices
later runs only, and the versions a case was priced under are
named on its report. There is no re-pricing view; the recorded
ledger stands. `cost_view` joins that fold with the budget
ledger's granted-vs-consumed (steps and tokens per phase, spawns
and wall clock case-wide, exhaustions, amendments) into the
per-case **budget-vs-actual** report — the API's
`GET /api/cases/{id}/cost` and the console's cost block under its
existing budget bars — with the estimated/measured label the
case's figures carry. `GET /api/costs` sums every case's fold
into an all-cases totals line; portfolio comparison across cases
is round 15's work, not this round's.

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

## Source governance

Everything a report asserts traces to evidence, and evidence
traces to a source. Round 10 makes the source itself a governed
record and gives the case a retention and redaction lifecycle,
in `governance.py` with the service methods around it.

**The registry.** A `sources` table holds one record per source:
name, kind (`filing`, `news`, `internal dataset`, `web`), owner,
usage basis (`public`, `licensed`, `client-provided`), notes, and
an append-only history of registration acts. At service startup
`seed_sources` registers every corpus source with its corpus
classification — idempotently, never overwriting a record the
desk has since amended. Registration afterwards is a service/API
act (`POST /api/sources`), recorded with its analyst, never a
config edit or a code change.

**Ingestion validates against it.** Both intake paths — corpus
ingestion (`evidence.ingest_case_evidence`) and the mid-case
analyst upload (`add_evidence_document`) — call
`require_registered_source` and refuse a document whose source
has no registry record, with the source named in the error. A
brief's `allowed_sources` are checked at case creation
(`brief_source_problems`): an entry matches if it names a
registered source, a corpus document slug, or a corpus document
kind (the three scoping styles briefs already used); anything
else refuses the brief, fail closed. `case_source_policy` then
resolves each allowed entry to the governance records it admits,
so the analyst sees — in the console and at `GET
/api/cases/{id}/sources` — on what basis the case's sources are
being used, not just their labels.

**Provenance.** `case_provenance` joins, per evidence item, the
item's own record (source, SHA-256, ingestion path — corpus
intake or analyst upload, from the `origin` stamp ingestion now
writes) with its source's registry record, the findings that
cite it, and the report claims the citation ledger maps to it.
It reads only persisted records and the ledger; it is the audit
answer to "where did this report's facts come from".

**Retention and legal hold.** New cases carry `retention_class`
(`standard`) and an append-only `retention_history`.
`retention_sweep` lists standard cases whose creation date plus
the configured retention period (`RETENTION_DAYS`, default 365)
has passed — a listing only, by design: nothing in the sweep
path can delete. Deletion is `delete_case`: an analyst act with
a required reason that removes the case's evidence and file
objects, deletes its rows across the case-scoped tables, and
writes a tombstone — case id, subject, actor, reason, date, the
final report's SHA-256 if one exists, and the evidence count —
so the register can always answer that the case existed and who
closed it. `set_legal_hold` flips the class as a recorded act;
a case under hold is excluded from the sweep and `delete_case`
refuses it outright (GateError, HTTP 409). Holds are set and
cleared, never edited away: both acts stay in
`retention_history` and the audit trail.

**Redaction.** `redact_evidence` replaces an item's stored
bytes with a content-free marker (redaction id, analyst, reason,
date) and writes a `redactions` record holding the item's
ORIGINAL SHA-256, the reason, and the actor — the original
content survives nowhere. Three consumers react. The verifier
treats a redacted cited item as `evidence_missing` ("a redaction
marker cannot support a claim") and never reads marker bytes as
evidence text. The citation ledger's gate check reports redacted
citations separately from drift: redaction is an authorised act,
not tampering, so it neither passes as verified nor blocks as
drift. And every claim in the live draft citing the item gets a
`redaction_flags` record: `decide_report` refuses approval while
any OPEN flag's claim is in the current report, and the analyst
either acknowledges the flag (recorded, claim stays) or sends
it back — the round-3 pattern verbatim: the claim's finding is
deleted and the case returns to APPROVED for re-execution. Flags
are recomputed from the live report at each gate, so a stale
flag on a superseded report never blocks a clean one.

## Analyst workbench

The workbench (`workbench.py`, reached through the service and
the API) composes the surfaces above into the desk's working
view. It stores exactly one new thing — saved views — and
derives everything else from records that already exist.

**The queue.** Each case is read for the acts it owes, in seven
groups: plan awaiting decision (status PLAN_READY; waiting since
the plan was generated), draft awaiting decision (DRAFT_READY;
since the draft was composed), quarantined claims awaiting
disposition (records still QUARANTINED), conflicts not yet
noted (records still OPEN), redaction flags open (flags still
OPEN whose claim is in the live report — the same set that
blocks report approval), interrupted cases awaiting resume (the
interrupted scan's result; building the queue runs the scan,
because listing is its detection point), and budget-exhausted
cases awaiting amendment (BUDGET_EXHAUSTED; since the cap
tripped — that group also represents the partial draft's
decision, so the case is not listed twice). The record-level
groups count only while the case sits at a gate where the
decision can actually be taken (DRAFT_READY or
BUDGET_EXHAUSTED), mirroring the decision endpoints' own gates.
Items sort oldest-waiting first; each group reports its count
and its oldest wait; ages are computed against the store clock
(`utcnow`) at read time. Filters — act, assignee, priority —
narrow every group the same way. Nothing about the queue is
persisted: it is what the records say, recomputed per read, so
it cannot drift from the cases it describes.

**The timeline.** A case's audit trail already carries every
gate decision and state change; the timeline renders it as one
chronological view, folding the analyst's notes in at their own
events and enriching those entries with the note text (case
notes and report section notes alike, looked up by the note id
the audit payload carries), with a human-readable label and the
acting analyst per entry. It records nothing of its own.

**Triage.** A case record carries an `assignee` (or none) and a
`priority` (`normal`, `high`, `urgent`). Setting either is a
recorded act — the audit event names the actor, the new value,
and what it replaced — and either can be read off the case, the
queue items, and the search results. Triage confers no decision
rights and changes no gate: it exists to order the desk's work.
There is deliberately no bulk decision in the workbench or its
API — a gate decision is one case, one analyst, made on the
case's own endpoints — and a test asserts that against the
route table itself.

**Saved views.** `workbench_views` (migration 0009) holds one
analyst's named queue filters: the filter keys are the queue's
own (act, assignee, priority) and are validated on save, so a
view can never name a filter the queue does not implement.
Saving under a name the analyst already holds replaces that
view's filters; a view can be deleted by its owner. A view holds
no case state and no decision — it is a shortcut, and deleting
one touches nothing else.

**Search.** Deterministic, case-insensitive substring search
over case subjects, brief question texts, and finding claims.
Each hit names the field it came from and the question or
finding id, so the analyst lands on the matched text, not just
the case. No index is maintained: the desk's case volume makes
a scan the honest implementation, and the result order follows
the case list.

## Portfolio mode

The workbench sees the whole desk or one case; a desk also
thinks in mandates. An **engagement** is the grouping above a
case — one client's pipeline of vendor and acquisition
diligences. It is a governed record in its own table
(`engagements`): name, client label, owner, created by/at, and
an append-only history of the acts on it. Membership lives on
the case records: a case carries at most one `engagement_id`,
so the engagement holds no member list that could drift from
the cases it groups.

Assignment is a recorded act on both records, in the triage
pattern. The case's audit trail gains
`case_assigned_to_engagement` (naming the engagement and the
one it replaced) or `case_removed_from_engagement`; the
engagement's history gains `case_assigned` / `case_unassigned`
entries naming the case and the other side of the move. A
no-op assignment is refused — a membership that does not
change is not an act.

The portfolio view (`portfolio.py`, `GET /api/portfolio` and
`/api/portfolio/{engagement_id}`) is a derivation, like the
queue: folded from the member cases' records on every read.
Counts by case state come off the case records. The gate list
is the workbench queue's own items filtered to members — same
groups, same waits, so the portfolio can never show a gate the
desk queue does not, and the oldest-waiting item is simply the
earliest of them. The cost roll-up sums `fold_cost_ledger` over
each member's audit trail with Decimal arithmetic and converts
once at the boundary; budgets are never merged. Per-question
grades are recomputed by the one documented rule
(`report_model.question_grade`) over each case's own report, so
a portfolio row cannot disagree with the report it summarises.
Open conflicts and quarantined claims are listed by id,
labelled with their case. Each member's retention standing is
computed with the governance helpers and surfaced as flags —
`legal_hold`, `past_retention` — read-only: surfacing them is
what keeps the grouping from ever reading as a bulk handle.

Isolation is the invariant. Membership writes exactly one
field on the case record and opens no path between cases:
evidence, findings, citations, budgets, gates, and retention
all stay case-scoped, resolved through the same per-case store
reads as before (the R14 defences are untouched — an evidence
ID still names a document in its own case only, even when a
sibling case in the same engagement holds a lookalike ID).
Search gains a scope, not a merge: `search_cases` takes an
optional case set, and the engagement-scoped search reads only
member cases, returning the same case-labelled hits. There is
no portfolio-level decision or deletion route — the route
table is tested for their absence, as the workbench's was.

## The challenge pass

Verification asks of each claim: does its evidence support it?
Conflict detection asks of the claim set: do any two disagree?
Neither argues against the draft — and a draft can pass both with
claims that are weak in ways a diligence partner would catch on a
read: a headline revenue figure resting on a single news article,
an answer whose number comes from the oldest document in a live
disagreement, a STRONG grade that is one filing repeated four
times. The challenge pass (`challenges.py`) is that read, made
structural: a governed rule set in code, run after verification
and conflict detection, recording each objection as a **challenge
record** the analyst must answer before the report gate.

**Independence and determinism.** Like the verifier, the pass
runs from the stored records only — the report's supported
claims, the evidence list (kinds, dates, source classes from
`conflicts.py`), the conflict records, and the grades by the one
grade rule in `report_model.py`. It never reads composer
internals. Identity is content: each computed challenge carries a
fingerprint (SHA-256 over the rule, its target, and the detail the
rule read), and the sync that persists the pass is keyed by it —
re-running the pass on an unchanged draft yields the identical
challenge set, byte for byte (proven by test). When a
re-composition moves the draft on, the sync recomputes: a live
record whose target vanished, or whose substance changed under its
key, is marked **SUPERSEDED**, never deleted, and the fresh
challenge is minted beside it. One refinement: a challenge answered
by sending its claim back whose claim returns byte-identical on
re-investigation **reopens** — the send-back was supposed to change
something, and the identical claim has not answered the objection.

**The rules.** Four, each named on its record with the evidence it
read:

- `single_source_figure` — a claim asserting governed-metric
  figures (the conflict pass's lexicon), citing exactly one
  evidence item, corroborated by no other claim. Contested claims
  are excluded; they are the next rule's ground.
- `oldest_source_on_contested_metric` — a claim on a conflict side
  whose own evidence includes the oldest document in the
  disagreement, while another side rests on fresher evidence.
- `grade_rests_on_one_document` — a question graded STRONG or
  STANDARD whose claims, between them, cite exactly one document.
  The grade rule counts claims and requires a primary source for
  STRONG, so a strong grade can never rest on a single claim or a
  lone secondary source; repetition of one document is the gap it
  cannot see, and it is the honest form of "a grade that outruns
  its evidence" in this design.
- `one_document_carries_questions` — one document is the sole
  evidence behind two or more questions' answers: the draft as a
  whole leans on a single source. (Per-question workstream
  concentration cannot occur here — a question has exactly one
  workstream by design — so the pass reads concentration where it
  can actually happen: the evidence.)

**Severity is a doctrine about what else has eyes on the
weakness.** A **must-answer** challenge is one nothing else in the
pipeline surfaces: today, exactly the `single_source_figure` shape
where the figure is a headline financial metric (revenue, revenue
growth, gross margin, EBITDA margin, operating cash flow, net
debt, debt, cash, funding raised) and the single source is
secondary class. Everything else is **advisory** — recorded,
counted on the case view at the gate, never blocking — because
each sits beside something the desk already sees: the conflict is
in the report permanently, the grade is printed on the question.

**The gate and the answers.** `decide_report` refuses approval
while any must-answer challenge is OPEN, naming the open
challenges (rejection is never blocked). The analyst answers each
challenge on its own record, with exactly three dispositions:

- **rebut** — the claim stands. Requires a written rebuttal and
  counter-evidence named from the case (IDs the service validates
  against the case's evidence). The rebuttal rides the case — the
  challenge record and the audit trail — never the report body.
- **revise** — the claim is sent back for re-investigation: the
  reinvestigate path's effects applied to a supported claim. Its
  finding leaves the case record and the case returns to APPROVED.
  (No quarantine record is created: quarantine is the verifier's
  shelf, and this claim was never quarantined.)
- **downgrade** — the claim stops standing as support. It is
  quarantined through the existing shelf with reason
  `challenge_downgraded` and leaves the report body immediately,
  the draft re-rendered through the finalize chain so coverage,
  grades, and the ledger describe what remains. The shelf record
  can then be decided like any other (sent back or dropped).

Question- and case-level challenges name no single claim, so
revise and downgrade are refused for them; they are answered by
rebuttal.

**Internal review, never delivered.** Challenges ride the case —
the `challenges` table (migration 0011), `GET
/api/cases/{id}/challenges`, the console's Challenge pass panel —
and never the report record, so no delivered format (markdown,
HTML, JSON export) can carry one; the eval pack asserts their
absence from all three. The report's readers see the desk's
findings; the desk sees its own second opinion.

**Deployment variant.** The graded implementation is the
deterministic rule set above, per the house rule that judgement
lives in code and the gate must hold offline. A deployment that
wants a model-driven challenger can put one behind the same
record shape: a challenger step proposing challenge contents
(rule name, target, evidence read, severity) that are validated
into challenge records — named rules only, evidence IDs checked
against the case, severity capped by the same doctrine — after
which the gate, the dispositions, and the supersede semantics
apply unchanged. The records are the contract; who or what
proposes them is a deployment choice.

## The skills library

A skill is governed, versioned **method**: how a workstream should
run its investigation, written once by the desk and loaded into a
sub-agent's context when a case's plan names it. It is distinct
from the neighbouring concepts by what it is not: not case memory
(analyst notes are one case's facts and context), not evidence (a
skill can never be cited by a finding), and not a harness profile
(a profile shapes the build for a model — prompts it assembles,
tools it allows; a skill is method that runs under whichever
profile is active).

**Artifacts and the loader.** A skill is a JSON artifact — name,
version, the workstreams it applies to, the case tools its method
uses, a description, the method text, a changelog — one file per
version under `skills/` (`<name>.v<version>.json`), validated by a
fail-closed strict loader (`skills.py`) in the profiles
discipline: exact keys, the file name part of the artifact's
identity, every refusal a named `SkillError`. The shipped library
is three skills: `revenue-bridge` (financial-scan: reconcile a
headline figure across documents, naming every bridge item),
`contradiction-hunting` (all roles: hunt the second source on
every primary figure), `source-triangulation` (market-scan: no
market claim from a single secondary source). At startup the
artifacts seed into the store's `skills` table (migration 0012) —
the runtime library of record — idempotently, the source
registry's pattern: a version already in the store is never
overwritten by the file that shipped it.

**The tool rule.** A skill may narrow a role's method but never
widen its tools. Wherever a skill is validated against a harness
profile — at publish, and again at the plan gate — every tool it
names must already sit in that role's allowlist under that
profile, or validation refuses with the excess named. The rule is
profile-relative on purpose: under `openai-gpt`, whose market-scan
has no document reader, a market skill naming `read_document`
refuses, while the same artifact is unremarkable under the wider
offline profile. A profile promotion that narrows a role can
therefore retire a skill from approvability — at the gate, in the
open, never silently at runtime.

**Selection and the pin.** Brief questions declare skills by name
(`QuestionIn.skills`), the planner carries the names onto the
plan artifact and its readable form, and the **plan gate** is
where the library binds to the case: `decide_plan`, on approval,
resolves every named skill (`skills.resolve_plan_skills`) —
unknown names, role mismatches, and tools beyond the role's
allowlist each refuse approval with the defect named and the plan
left standing — and stamps the resolved **pins** ({name, version}
of the library's latest) onto the plan entries, recording them on
the `plan_approved` audit event. Execution, re-execution, and
resume all run under the pins: a version published mid-case is
picked up by the next plan the desk approves, never by a case
already approved. This is the plan-version discipline applied to
method.

**Loading and provenance.** At execution the pinned method text
mounts on the case's virtual filesystem at `/skills/<name>.md`,
read-only under the evidence mount's discipline (below); the
planner's task block names the workstream's skills, and the
sub-agent reads the files before it searches. `record_finding`
stamps the question's pins onto every finding the workstream
records — the stamp comes from the approved plan, never from
anything the model says — and the report model folds the stamps
into the Method-and-limits section, one line per workstream that
ran under skills, naming the pinned versions. The execution
record names the pins too (`execution_started`).

**Publishing.** `POST /api/skills` is the library's only write:
the analyst, the artifact, and its changelog are the record (the
publish pattern of the promotion log — append-only rows, no
update or delete in the store, no edit or delete route).
Validation against the active profile runs **before** anything
is stored, so a malformed or tool-widening version can never
enter the library, and versions extend a name's line by exactly
one — re-publishing an existing version, or skipping ahead, is
refused. Publishing writes to the store, not the `skills/`
directory: the files are the seed set, a deployment's library
lives with its data, and a store reset returns to the seeds.

**What the offline gate can and cannot grade.** Skills reach the
scripted stub through the same seams they reach a real model —
plan artifact, task block, filesystem mount — and the stub
workstream reads each mounted file with `read_file` as a playbook
step before it searches. The stub cannot follow prose: its fact
selection is the keyword playbook, unchanged by the method text.
The skills eval pack therefore grades loading, gating, pinning,
and provenance exactly, and claims no outcome difference for the
method text itself; method compliance is a live-model property.

## Observability

Everything above writes records; round 18 builds the reader
side (`observability.py`). Three reads, all folds over the
stored records — none of them writes to a case, keeps a
counter, or asks the harness anything — plus the one stored
snapshot the round adds.

**The run trace.** `build_trace` reconstructs one case's run
end to end. The **runs** fold from the budget ledger in audit
order: each `budget_granted` opens a run (phase, the caps it
was granted, the profile it ran under), the guard's
`budget_exceeded` and the run's `cost_recorded` entry attach
to it, and `budget_consumed` closes it with the run's usage;
an execution run is stamped with its attempt, the count of
`execution_started` events before it, so re-executions and
resumes read as the separate attempts they were. The **acts**
are the audit trail itself, mapped entry for entry: intake,
planning, the plan gate, execution with its waves and
workstream finishes, composition, verification (quarantine,
conflicts), the challenge pass and its dispositions, the
report gate. Every entry becomes exactly one act in the
trail's insertion order — causal order — and an event the act
table does not name still appears, under a humanised label;
the trace drops nothing. Beside the acts ride the counts
(findings by question, claims, standing quarantine, conflicts,
challenges), the verifier report's summary, and the case's
priced cost total from the cost ledger's own fold. Timing is
stated, not decorated: the trail's stamps are second-resolution
and each run's usage (wall seconds included) was recorded, but
no per-act, per-workstream, or per-tool-call durations exist
anywhere in the records, so the trace shows none.

**The failure taxonomy.** `classify_case_outcome` is one
function with a documented rule order, first match wins:
`evidence_drift` (the report record's flag) →
`budget_exhausted` (cap, phase, and uncovered questions from
the coverage table) → `challenge_blocked` (the open
must-answer IDs) → `interrupted` (EXECUTING with an unclosed
checkpoint; the attempt and its completed/pending split) →
`workstream_failed` (the latest execution's failed branches,
with their recorded errors) → `claims_quarantined` (the
standing quarantine, with the verifier's reasons) →
`stale_evidence` (the open marks) → `plan_rejected` /
`report_rejected` (the gates' recorded refusals) →
`completed_partial` (FINAL over the approval record's
coverage table) → `completed` → `in_progress`, the fallback
that names the standing status and makes the classification
**total**: every case maps to exactly one label, so an
unclassifiable outcome is impossible by construction, and the
eval pack's sweep asserts the property over every case the
other packs produce — a documented label for each, and no
terminal case in the fallback. Where a fact competes (a draft
can hold quarantined claims and open challenges at once), the
rule order is the precedence, and the losing facts remain
visible in the trace and the case view.

**The operations digest.** `fold_digest` summarises one day
across the desk, and the service stores the fold as a
**versioned record** (`operations_digests`, migration 0013):
generating is a recorded act (analyst, timestamp), appends the
day's next version, and never rewrites an earlier snapshot;
an ungenerated day reads as absent (`GET /api/digests/{day}`
answers 404). Scope is stated on the fold: a case belongs to
the day it was **opened** for the opened list, status counts,
and taxonomy counts; events belong to the day of their audit
timestamp — gate decisions, challenge answers, and cost (each
priced line rides its run's `cost_recorded` entry, so a day's
cost is the day its runs ended), summed in Decimal from the
recorded lines by case and in total, never from a stored
counter. A partial report is listed when it was composed that
day and still stands, with the coverage table's uncovered and
partial questions; a case amended and re-run the same day no
longer stands partial, and the digest says so by omission.
The challenge block counts the records minted that day and the
dispositions taken that day, and lists the must-answer
challenges still open **desk-wide**, each with its age at
generation — the desk's live exposure, not the day's history.
The console renders the trace on the case (outcome, runs, act
table) and the digest in its own panel, with the stored
versions listed.

## The virtual filesystem

`EvidenceBackend` implements the Deep Agents `BackendProtocol` over
the object store plus a `case_files` index in the case store. Case
files (`/plan.json`, `/execution/notes.md`, …) are writable; evidence
is mounted read-only at `/evidence/<EV-ID>.md`, so the agent reads
sources through the same `read_file` tool it uses for its own notes,
and cannot alter them — a write to `/evidence/` returns an error.
The skills the case's approved plan pins mount the same way at
`/skills/<name>.md` — the library's method text verbatim, resolved
from the plan's pins when the backend is built, read-only under the
same refusal. File metadata (path, SHA-256, size) is indexed in the store; bytes
live in the object store.

## Data model

Payload-JSON tables, Postgres and SQLite variants of the same schema
(numbered migrations in `migrations/` and `migrations/sqlite/`):
`cases`, `plans` (versioned), `evidence`, `findings`, `reports`,
`citation_ledger`, `verifier_reports`, `quarantine`, `conflicts`,
`case_files`, `audit`, `execution_checkpoints` (one row per case),
`case_notes`, `report_snapshots`, `report_section_notes` (the
analyst's report annotations, apart from the report record),
`sources` (the registry, keyed
by name), `redactions`, `redaction_flags`, `case_tombstones`,
`workbench_views` (an analyst's saved queue filters, keyed by
analyst and name — the one workbench table, holding no case
state), `engagements` (the grouping above a case: name, client
label, owner, and the record's own act history — membership
itself lives on the case records, never here), `challenges`
(the challenge pass's records: rule, target, the evidence read,
severity, status — OPEN, ANSWERED with its dispositions, or
SUPERSEDED — and the content fingerprint the sync keys on;
migration 0011), `skills` (the library: one append-only row per
(name, version) — the artifact, its origin (seed or published),
the publishing analyst, and the timestamp; migration 0012 — a
global table, not case-scoped, so case deletion never touches
it, and findings carry skill pins as plain payload fields).
Case records carry the active harness
profile's stamp (id, version, target model) from intake, and their
continuity state: open `stale_marks` and the append-only
`staleness_decisions`, plus their `retention_class` and the
append-only `retention_history` of hold acts, plus their
workbench triage state: an `assignee` and a `priority`, each set
as a recorded act, plus their portfolio membership: an
`engagement_id`, set by the recorded assignment acts. The audit
trail records case creation, evidence ingestion, plan generation
and decisions, sub-agent spawns, budget ledger entries (grants —
each naming the profile the phase ran under — consumption
checkpoints, exhaustions, amendments), cost ledger entries (one
`cost_recorded` per run: its priced lines, each pinned to the
price-table version and rates that produced it), composition, verification
and quarantine events, conflict detections and NOTED decisions,
evidence-drift detections, and report decisions — plus the
continuity events: mid-case evidence arrival and staleness
marking, staleness decisions, plan revisions and report snapshots,
findings reuse, interruption detections, resumes, and case notes —
plus the governance events: source registrations, legal-hold acts,
evidence redactions with their claim flags and flag decisions, and
case deletions (the last event a case ever records; the tombstone
outlives it) — plus report section notes added and edited — plus
the workbench triage acts: case assignments and priority changes,
each naming the actor and the value it replaced — plus the
portfolio acts: cases assigned to and removed from engagements,
each naming the engagement and the membership it replaced —
plus the challenge events: passes that raised, reopened, or
superseded challenges, with the open counts by severity, and the
three dispositions (rebutted with its counter-evidence named,
revised, downgraded with its quarantine record named).

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

## Red-team pack

Round 14 turns the shell's defences into graded targets. The pack
(`RED_TEAM_CASES`, outside the promotion split like the
dependency packs) attacks with two purpose-built subjects and
three brief/case-level scenarios:

- **Poisoned evidence** — a presentation whose prose touts a
  figure its own results table contradicts; a claim built on the
  prose figure must quarantine. This attack succeeded against the
  whole-text figure rule and is why the verifier now judges
  figures against the cited document's ledger (see step 5 above).
- **Citation bait** — a document titled for a topic its body never
  covers; claims citing it for figures or relationships it does
  not state must fail the figure/entity checks, while its real
  facts are still recorded.
- **Injection in evidence** — a document carrying a fake system
  notice, a fake prior finding, and a fake Key facts section in
  its body. Evidence is data only: the recording check (step 3)
  refuses fact references the real ledger — the document's final
  Key facts section — does not state, in words it does not use,
  and planning never reads evidence, so document text cannot
  move a plan.
- **Brief attacks** — an unregistered `allowed_sources` entry
  (refused at intake, R10), a question whose only sources are out
  of scope (NOT COVERED, never answered from evidence the case
  does not hold), and an instruction smuggled into a question
  (the plan gate is a state machine; it refuses regardless).
- **Cross-case citation** — evidence IDs resolve within the case
  only: a foreign-only citation is refused, and a mixed citation
  records citing only the case's own documents.

Grading is per attack and zero tolerance
(`evaluate_red_team_case`): injected figures in final reports,
cross-case citations accepted, out-of-scope answers, and
poisoned claims in reports must all be zero. The committed
report — attacks, expected behaviour, and the observed result on
this build, including the two attacks that succeeded before
their defences were built — is `docs/red-team.md`.

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
- Cost accounting prices harness runs only. The verifier's
  optional advisory model check is invoked outside a run (and
  outside the budget guard), so it leaves no consumption record
  to price; cases that ran before the cost ledger existed carry
  no cost lines and are never reconstructed. The price table's
  provider rates are published list rates, with caching and
  batch discounts unmodelled — the ledger is a costing record
  at governed rates, not a provider invoice.
- Conflict detection is only as wide as the governed metric
  lexicon: a disagreement stated without a recognised metric,
  figure, or comparable value sits in the report as two findings and
  is not flagged as a conflict.
- Web fetch is not wired; the evidence universe is the case corpus.
- Source governance validates the source as a governed record, not
  the document's truth: a registered source can still publish a
  wrong figure — that is the verifier's and the conflict pass's
  territory, not the registry's. The retention clock runs from
  case creation, not from the case's last activity; a long-running
  case's sweep listing is a prompt for an analyst's judgement, not
  an expiry. Redaction is irreversible by design: the original
  bytes are destroyed, and the redaction record's hash is the only
  proof of what was there.
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
- Skills govern loading, not compliance. The library, the gate,
  and the mount guarantee which method text a workstream runs
  under and prove it afterwards from the record; whether a model
  follows the method is observable only in live mode, and the
  offline stub — which reads the files and then runs its fixed
  playbook — is the standing demonstration of the difference.
  Selection is per brief question, declared in the brief and
  approved at the plan gate; there is no per-case library
  override and no way for a plan to name a version directly —
  the pin is always the latest at approval, so a desk that needs
  an older version approves a plan that names it only by holding
  the newer publish back, a deliberate consequence of keeping
  one pin rule.
- Observability reads what the records hold, and no more. The
  trace cannot show per-act or per-step timing because no run
  records it — the trail's stamps are second-resolution and the
  guard's per-run usage is the finest clock the system keeps.
  The taxonomy classifies a case where it stands when asked:
  its counts in a digest are the day's cases as they stand at
  generation, so a regenerated digest may legitimately disagree
  with its earlier version about the same day — which is why
  both are kept. Gate refusals that leave no record (a defective
  plan refused at approval, a skill the gate cannot pin) are
  not outcomes: nothing terminal happened, and the case reads
  in_progress at its gate, with the refusal living in the API's
  409 and nowhere else.
