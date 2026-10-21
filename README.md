<p align="center">
  <img src="https://trida.ai/logos-rounded/trida-rounded.png" alt="Trida AI" width="96">
</p>

# Blueprint — Deep Diligence Agent

A diligence pack looks simple from the outside: a partner asks what a
company earns, what it sells, and where it stands, and someone has to
come back with answers that hold up. The work underneath is not
simple. The facts are scattered across filings, product documents,
pricing sheets, and press coverage; some of them contradict each
other; and every claim in the final pack has to trace to a source a
reviewer can open. Junior analysts burn days on the gathering. The
judgment was never the slow part — the assembly was.

This blueprint puts a deep agent on the assembly. It is built on
LangChain's **Deep Agents** harness — planning tools, sub-agents with
isolated contexts, a virtual filesystem, context management — and
wrapped in the governance a diligence desk actually needs. An analyst
hands the agent a brief: the subject, the questions the pack must
answer, the sources it may use, and a budget. The agent plans the
investigation and stops. An analyst approves the plan before anything
runs. Specialist sub-agents then work the questions in parallel,
gathering evidence into the case filesystem, and every finding they
record must cite evidence that exists in the case — a claim without a
citation is refused in code, not discouraged in a prompt. A
deterministic composer assembles the report draft from the findings,
and an analyst approves the draft before it is final. The agent never
issues a recommendation. It assembles cited findings; the decision
stays with people.

The setting is a diligence desk at a professional-services firm.
Every company, person, and document in the bundled corpus is
fictional.

> Reference prototype. Not connected to any real firm, data provider,
> or client system.

> **Work in progress** — this blueprint is built and passing its automated test and eval gates, but it has not yet finished hands-on local testing, and its companion blog post is not yet published. Expect changes. This notice comes off when both are done.

## Run it with a real model

The planner and the workstream sub-agents are built for a real LLM.
Set one provider and they use it; the governance around them — gates,
budgets, citation enforcement, composition — does not change.

```bash
cp .env.example .env   # then set ONE provider
```

- **OpenAI-compatible** — `OPENAI_API_KEY`, plus `OPENAI_BASE_URL` for
  any compatible endpoint: NVIDIA NIM hosted
  (`https://integrate.api.nvidia.com/v1`) or self-hosted
  (`http://localhost:8000/v1`), vLLM, and so on. `OPENAI_MODEL` picks
  the model.
- **Anthropic** — `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`).
- **Ollama** — `OLLAMA_BASE_URL` for a local model.
- **Offline** — `LLM_BACKEND=offline` (the default with no key): a
  labelled deterministic stub model drives the same harness over the
  bundled corpus. That is what the tests, the evals, and the demo run.

Real-provider packages are an extra: `uv sync --extra llm`. Pair the
provider with the harness profile tuned for it — `PROFILE_ID` (see
"Harness profiles" below); the default `offline-stub` profile is the
build the tests and evals pin.

## Quickstart (offline, no keys)

```bash
make install     # uv — or: make install-pip
make demo        # one gold case end to end: intake → plan gate → execution → report gate
make test        # the pytest suite
make evals       # the gold case-pack gate
make serve       # API + console on http://localhost:8000
```

## How it works

**The harness does the deep work; the shell owns the accountability.**
LangChain Deep Agents (`create_deep_agent`, pinned in
`pyproject.toml`) provides the planning tools, sub-agent delegation,
the virtual filesystem, and context middleware. Everything the desk
is accountable for lives in our code around it:

- **Case intake.** A brief becomes a scoped case. The corpus documents
  in scope are ingested as evidence — each gets an ID, a source
  label, a retrieval time, and a SHA-256 — with bytes in the object
  store and metadata in the case store.
- **The plan gate.** The planner decomposes the brief into one
  workstream per question and writes the plan (`/plan.md`,
  `/plan.json`) as a stored, versioned artifact. Execution is
  impossible until an analyst approves it; the lifecycle is a state
  machine in `service.py`, and the API answers out-of-order calls
  with 409.
- **Sub-agent workstreams, run to a schedule.** On approval, the
  planner delegates each workstream with the harness's `task` tool
  to a specialist (financial-scan, product-scan, market-scan)
  running in an isolated context. Sub-agents search the case
  evidence and read documents through the virtual filesystem, where
  evidence is mounted read-only at `/evidence/<EV-ID>.md` — the
  agent cannot alter its sources. A workstream may declare
  **dependencies** on other workstreams in the plan (the market
  read needs the product facts first): the plan gate refuses an
  unknown dependency or a cycle, with the defect named, and
  execution runs in topological **waves** computed in code — one
  harness run per wave, so a dependent workstream cannot be spawned
  before its prerequisites finish. When a wave assembles, each
  dependent receives its prerequisites' recorded findings —
  verbatim, nothing else crosses — as starting context, and the
  handoff is written to the audit trail. Every workstream ends with
  a recorded status (completed, empty, failed, interrupted, not
  run); a wave that crashes fails only its own branch, the other
  waves still run, and the case composes with the failure named in
  its coverage table. The composer orders claims by plan, not by
  completion order, so repeated executions of a case produce the
  same draft.
- **Continuity: checkpoints, interruption, resume.** Cases run for
  days, so execution keeps its own record. As waves complete, the
  shell writes a **checkpoint**: per-workstream status, the finding
  IDs recorded so far, wave position, and a budget snapshot folded
  from the ledger. A case left in EXECUTING with an incomplete
  checkpoint — the process died, a deploy restarted it — is an
  **interrupted case**: `interrupted_cases()` finds them, the
  first scan records `execution_interruption_detected` in the
  case's audit trail (later scans do not repeat it), and the desk
  lists them at `GET /api/cases/interrupted` and in the console.
  **Resume** is an explicit analyst action, never automatic:
  workstreams the checkpoint records as finished are not re-run —
  their findings stand under their original IDs — only the
  incomplete waves execute, and budget consumption carries over
  because every run's guard seeds the case-level caps from the
  ledger, which already holds the interrupted attempt's spend.
- **Evidence arrives mid-case; staleness is decided, never
  silent.** An analyst can add a document to a live case (console
  form or `POST /api/cases/{id}/evidence`): it is ingested under
  intake's rules — object store, SHA-256, the case's next evidence
  ID, `origin: analyst_upload` on the record. When the case
  already has a plan, the shell keys the document against the
  plan's workstreams in code (the governed keyword stems, with a
  two-stem minimum, because a single shared stem is cross-talk)
  and marks the questions it speaks to **stale** on the case
  record. Execution then refuses until an analyst decides:
  **proceed** — the decision is recorded on the case and lands in
  the report's Method and limits — or **re-plan**. A re-plan
  snapshots the current draft into the report-version store (the
  plan store already versions plans), returns the case to the
  plan gate, and the new plan is version n+1. When the re-plan
  answers staleness, the next execution re-runs exactly the marked
  workstreams and **reuses** the rest under a stated rule: a
  workstream's findings carry over only while every item of
  evidence they cite is still in the case with its recorded hash
  intact — drifted evidence re-runs its workstream. A general
  plan revision, with no staleness involved, sets no reuse
  directive and re-executes in full.
- **Analyst notes: the desk's memory, never evidence.** Analysts
  attach timestamped notes to a case (`NOTE-####`, console + API).
  On any (re-)plan, the notes reach the planner's instruction in
  a bounded, labelled section — the most recent ten, each
  truncated at 500 characters — as context for how to investigate.
  Notes live in their own table: never in the evidence store,
  never mounted in the case filesystem, and citation validation
  resolves IDs against the evidence table, so a note ID is refused
  exactly like any other non-evidence ID.
- **Citations, enforced at every step.** `record_finding` refuses a
  claim with no valid evidence ID. The report composer re-verifies
  every citation while assembling the draft — a claim whose evidence
  no longer resolves is refused again and named in the method note —
  and composition writes the **citation ledger**: a stored record
  mapping every claim in the report (CLM-0001, ...) to the finding
  behind it and the SHA-256 of each cited item's bytes at citation
  time. The report's References section is generated from the ledger,
  and the ledger is queryable in the console and at
  `GET /api/cases/{id}/ledger`.
- **Budgets in code, per phase, on a ledger.** Each run is wrapped
  by a callback guard that counts model steps, sub-agent spawns,
  tokens, and wall-clock time. Steps and tokens are per-phase caps —
  the brief can give the planning run and the execution run
  different limits, and an unset phase field derives from the
  brief's flat value — while spawns and wall clock are case-level
  caps consumed across both phases, seeded from what the case has
  already spent. Crossing a cap aborts the run; the findings
  recorded so far survive, and the composer produces a partial
  report whose banner names the cap and the phase that stopped,
  with coverage itemised per question: ANSWERED, PARTIAL (findings
  exist but the run stopped mid-workstream), or NOT COVERED. Every
  budget event — the grant a phase runs under, its consumption
  checkpoint, an exhaustion, an amendment — is an entry in the
  case's audit-trail budget ledger, and the console's budget view
  is folded from that ledger, not from counters. An analyst may
  approve a partial report (the approval record carries the
  coverage table it approved) or **amend the budget**: a recorded
  raise, never below what the case has already consumed, that
  returns the case to APPROVED for re-execution while the original
  exhaustion stays in the ledger. A planning run that stops on
  its cap writes no plan, so its amendment leaves the case where it
  was, ready to re-plan.
- **Cost accounting: what a case cost, priced in code.** The same
  guard records every model call as it ends — phase, role (the
  main agent, or financial-scan / product-scan / market-scan),
  model, and the token split — with the basis of the figures: a
  real provider's returned usage is **measured**; the offline
  stub's declared per-call figures are **estimated**, and are
  labelled that way everywhere they surface. When a run ends, the
  service prices its records against a governed price table
  (`pricing/price-table.json` — per-model input/output rates held
  as data, versioned like the harness profiles, loaded
  fail-closed) and records the priced lines in the audit trail.
  Each line carries the table version and the rates it was
  computed with, so editing the table afterwards re-prices
  nothing: the fold reads recorded lines only. The per-case
  **budget-vs-actual** report (`GET /api/cases/{id}/cost`, and
  the console's budget panel) joins the budget ledger's granted
  vs consumed with the cost totals and their breakdowns by phase,
  by sub-agent role, and by model — priced in Decimal, so the
  parts sum to the total exactly, and the cost ledger's token
  totals are the budget ledger's own tokens, not a second count.
  A model the table does not list is recorded **unpriced** — its
  tokens count, its cost is not invented — and `GET /api/costs`
  sums every case's recorded ledger into the desk's total.
- **Independent verification, before the gate.** A citation can
  resolve and still not support the claim — the figure can be
  overstated, the entity absent. So after composition, a verifier
  re-checks every claim against the evidence it cites. It is
  independent by construction: it re-reads the persisted report and
  fetches each cited item's bytes directly from the object store,
  never the composer's working set or the citation ledger (delete
  the ledger and its verdicts do not change). The checks are
  deterministic — every figure and date in a claim must appear in
  its cited evidence, and every named entity must be mentioned
  there. With a real model configured, an advisory model check
  (claim and evidence only) is recorded beside them, but the
  deterministic checks decide. A claim that fails is **quarantined**:
  removed from the report body, recorded with its reason
  (`figure_not_in_evidence`, `entity_mismatch`,
  `evidence_missing`, …), and summarised in the draft's Verification
  section, the console's quarantine panel, and
  `GET /api/cases/{id}/verification`. The analyst can send a
  quarantined claim back for re-investigation or drop it — there is
  no action that approves one into the report.
- **Conflicts, surfaced and never resolved away.** The verifier
  judges each claim against its own evidence; it cannot see two
  claims side by side. So after verification, a deterministic pass
  keys the supported claims by (entity, metric, period) — the key
  extracted in code from each claim, with periods normalised to the
  year — and where two claims assert different values for the same
  key, it creates a stored **conflict record**: both claims, both
  values verbatim, their evidence, and each side's source class
  (primary sources — filings and official reports — rank above
  secondary ones such as press coverage and vendor marketing; for
  same-period facts the later-dated source is marked more recent).
  The draft gains a Conflicts section generated from the records,
  and the records are queryable at `GET /api/cases/{id}/conflicts`.
  Nothing in the pipeline suppresses, merges, or silently prefers
  either side, and the desk takes no position on which value is
  true. The analyst's only move is to mark a conflict **NOTED** with
  a note; the record and its section stay in the report permanently.
  Facts from different periods are different keys and never
  conflict; the same value from two sources is corroboration, and
  its claims sit together on one side of the record.
- **A second opinion the analyst must answer.** Verification and
  conflict detection between them still leave a draft nobody has
  argued against, so a third deterministic pass — the **challenge
  pass** — examines the verified draft for the weaknesses those
  passes do not test, and records each one as a **challenge**: the
  claim (or question, or draft) it targets, the governed rule that
  fired, and the evidence the rule read. The rules: a figure claim
  resting on a single source with no corroboration; a claim whose
  evidence is the oldest document on a contested metric; a support
  grade resting on one document repeated; one document carrying
  whole questions. A headline financial figure (revenue, margins,
  cash flow, debt, funding) standing on a single secondary source
  is **must-answer** — `decide_report` refuses approval while one
  stands open, naming it — and the rest are **advisory**: recorded,
  counted on the case view, never blocking. The analyst answers
  each challenge on its record, three ways only: **rebut** it in
  writing, naming counter-evidence from the case; **revise** —
  send the claim back for re-investigation, its finding deleted
  and the case returned to APPROVED; or **downgrade** — the claim
  is quarantined with reason `challenge_downgraded` and leaves the
  report body, the draft re-rendered without it. The pass runs
  from the stored records, never composer internals, and is
  idempotent by content: re-running it on an unchanged draft
  yields the identical set, and when a re-composition moves the
  draft on, answered challenges whose target vanished are marked
  SUPERSEDED, never deleted. Challenges are the desk's internal
  review — they ride the case (`GET /api/cases/{id}/challenges`,
  the console's Challenge pass panel) and never the report record,
  so no delivered format carries one.
- **Governed skills: method as a versioned library.** The desk's
  investigative method lives in a **skills library** — reusable
  instruction bundles (`revenue-bridge` for the financial scan,
  `contradiction-hunting` for every role, `source-triangulation`
  for the market scan) held as versioned artifacts under `skills/`
  behind a fail-closed strict loader, seeded into the store at
  startup and extended only by **publishing**: a recorded act whose
  artifact is validated against the active profile before anything
  is stored. The loader's tool rule is the governance heart: a
  skill names the case tools its method uses, and a name outside
  the loading role's profile allowlist refuses — a skill may
  narrow a role's method, never widen its tools. A brief question
  may declare skills by name; the planner carries the names onto
  the plan artifact, and **the plan gate pins the version**:
  approval resolves each name to the library version in force,
  stamps the pin on the plan, and refuses — named — a skill that
  is unknown, that does not apply to the workstream's role, or
  that the profile has narrowed out. At execution the pinned
  method text mounts read-only at `/skills/<name>.md` on the
  case filesystem (the evidence mount's discipline), the
  workstream reads it before it searches, and **every finding
  records the pins** — provenance that folds into the report's
  Method section. A version published mid-case moves nothing for
  a case whose plan already pinned the older one; there is no
  edit or delete route for a published version, because the pins
  and the findings that cite them must keep meaning what they
  meant. One honesty note: the offline stub reads the mounted
  files through the same playbook step a real model would, but
  it cannot follow prose — the offline gates grade loading,
  gating, and provenance, not method compliance.
- **One report model, three formats.** The composed report is a
  structured model first; the markdown draft, a self-contained
  print-ready HTML page (inline CSS, no external assets), and a
  JSON export (`GET /api/cases/{id}/report/html`,
  `GET /api/cases/{id}/report/export`) all render that one model,
  so the formats cannot disagree. The model opens with an
  **executive summary composed after verification, from the
  supported claims alone**: one headline claim per question in
  brief order, plus the counts that matter (claims supported,
  conflicts open, questions uncovered). A hard rule is enforced in
  code at composition time, using the verifier's own extraction:
  the summary may state no figure, date, or named entity that does
  not appear in a supported claim — a sentence that fails is
  dropped and the drop recorded, never silently rewritten. Each
  question section carries a **support grade** computed by one
  documented rule — STRONG (multiple claims, at least one on a
  primary source), STANDARD (multiple claims on secondary
  sources), THIN (a single claim, or one the verifier's model
  check marked weak), UNANSWERED (no supported claims) — rendered
  in the section header in every format. A **Risks and open
  items** section is generated entirely from records: open and
  noted conflicts, quarantined claims with their dispositions
  (referenced by ID and reason, never quoted — a quarantined
  claim's text reaches no format), questions the coverage table
  leaves PARTIAL or NOT COVERED, staleness an analyst proceeded
  on, and redaction flags; when every set is empty it says so
  plainly. Analysts can attach **section notes** to any report
  section (`POST /api/cases/{id}/report/notes`, console
  affordance included): margin commentary stored apart from the
  report, carried only in the JSON export's `annotations` field —
  writing or editing one never changes the report or its hash.
- **The report gate.** The draft is final only when an analyst
  approves it — and approval first re-verifies the citation ledger
  against the evidence store. If a cited item's bytes no longer hash
  to the value recorded at citation time, the evidence changed after
  the fact: the report is flagged **EVIDENCE_DRIFT** and cannot be
  approved until the ledger re-verifies clean. Rejection at either
  gate is a normal state, with the analyst's note kept on the record.
- **Harness profiles.** How the harness is built for a model —
  which prompt sections are assembled for the planner and each
  sub-agent, which case tools each role may use, which sub-agents
  exist, and the middleware switches (context compression,
  large-result offloading) — is a versioned configuration artifact
  in `profiles/`, not code. `PROFILE_ID` selects the active profile
  (default `offline-stub`); the service loads and strictly
  validates it at startup, and an invalid profile refuses startup
  with a named error. Switching profiles needs no code change.
  Every case is stamped with the profile id + version that ran its
  planning and execution, in the case record, the console, and the
  audit trail. Two profiles ship: `offline-stub` (the behaviour the
  gates pin) and `openai-gpt`, a candidate tuning for the
  OpenAI-compatible frontier profile with model-tuned prompt
  suffixes, context compression off and large-result offloading on,
  and a market-scan narrowed to search results only. A profile
  changes harness configuration only — never model weights, and
  never the governance: gates, budgets, and citation rules hold
  under every profile. The model itself is injected separately, so
  a profile can be run and evaluated with the offline stub standing
  in for its target.
- **Profile regression gate.** A profile replaces the active one
  only by passing evals, as a recorded decision. The packs are
  split in the eval data into a **tuning** subset and a **held-out**
  subset. A promotion evaluation runs the candidate's full suite
  three times and reads the worst run: tuning recall must match
  the incumbent's recorded baseline, held-out recall must stay
  within 0.02 of it, and the zero-tolerance lines (unsupported
  claims 0, citation and correctness 1.000) must hold on both
  subsets, every run. The evaluation snapshots the candidate
  file's SHA-256 — edit the file afterwards and the evidence no
  longer matches, so promotion is refused. Promotions and
  rollbacks are append-only records naming the analyst; the
  active profile is whatever the newest record says, and
  `PROFILE_ID` is only the bootstrap default. The shipped
  controls prove the gate discriminates: `openai-gpt` comes back
  **NOT_PROMOTABLE** (tuning recall 0.750 and held-out 0.812
  against a 1.000 baseline, on every run), while a copy of the
  incumbent under a new id passes and completes a promotion +
  rollback cycle (`python evals/promotion_controls.py` runs both,
  for real).
- **Audit trail.** Case creation, evidence ingestion, plan versions
  and decisions, every sub-agent spawn, budget events, composition,
  and report decisions are recorded per case and visible in the
  console.
- **Observability: the trace, the taxonomy, the digest.** The
  records above are the system of record, and three reads fold
  them for the desk. The **run trace** (`GET
  /api/cases/{id}/trace`, and a console panel) reconstructs one
  case's run end to end: the phase runs folded from the budget
  ledger — the caps each run was granted, what it consumed, what
  it cost, the cap it tripped if it tripped one — and every
  recorded act in causal order, from intake through the plan
  gate, the waves and their outcomes, composition, verification,
  the challenge pass, and the report gate. It stores nothing,
  and it shows no per-act timings, because none were ever
  recorded. The **failure taxonomy** (`GET
  /api/cases/{id}/outcome`) is one function that names how a
  case's run ended or what stands in its way — evidence drift,
  budget exhaustion with its uncovered questions, a
  challenge-blocked gate, an interrupted execution, a failed
  workstream, standing quarantine, stale evidence, a gate
  rejection, a partial or full completion — with the deciding
  records named, under a documented rule order whose fallback
  makes the classification total: every case gets exactly one
  label. The **operations digest** (`POST /api/digests/{day}`)
  folds a day across the desk — cases opened and closed, gate
  decisions, partial reports still standing with their uncovered
  questions, the day's priced cost recomputed from the cost
  ledgers, taxonomy counts, and the must-answer challenges
  still open desk-wide with their ages — and stores it as a
  **versioned record**: regenerating a day mints the next
  version beside the old ones, and an ungenerated day reads as
  absent.
- **Source governance.** Every source is a governed record — name,
  kind (filing, news, internal dataset, web), owner, usage basis
  (public, licensed, client-provided) — in a registry seeded from
  the corpus. Evidence ingestion refuses an unregistered source on
  both paths (corpus intake and analyst uploads), naming it, and a
  brief whose `allowed_sources` name nothing real is refused at
  intake, fail closed. Each case carries a provenance report —
  every evidence item with its source's governance record, its
  SHA-256, how it arrived, and the findings and claims that cite
  it — and a retention class: a sweep lists cases past their
  retention (`RETENTION_DAYS`) for an analyst's decision and
  deletes nothing itself; deletion is a recorded act that leaves
  only a tombstone, and a case under legal hold refuses it in
  code. An analyst may redact an evidence item for a recorded
  reason: its bytes are replaced by a marker, the redaction record
  keeps the original hash (never the content), the verifier treats
  the item as missing, and every live-report claim citing it is
  flagged — the report cannot be approved until each flag is
  acknowledged or its claim sent back.
- **The analyst workbench.** Everything a case owes a person, in
  one derived view (`GET /api/workbench/queue`, and a console
  panel): plans awaiting decision, drafts awaiting decision,
  quarantined claims awaiting disposition, conflicts not yet
  noted, open redaction flags, interrupted cases awaiting
  resume, and budget-exhausted cases awaiting amendment —
  grouped by the act needed, oldest waiting first, each item
  naming how long it has waited. The queue is computed from the
  case records on every read and stores nothing, so it cannot
  drift from the cases it describes. A per-case **timeline**
  merges the audit trail with the analyst's notes into one
  chronological view, and **search** runs deterministically over
  subjects, brief questions, and finding claims, naming the field
  each hit came from. Cases carry **triage** — an assignee and a
  priority (normal / high / urgent), each set as a recorded act —
  and analysts can save queue filters as named **views**. Triage
  orders the desk's work and gates nothing: there is no bulk
  decision anywhere, and every gate decision remains one case,
  one analyst, made on the case itself.
- **Portfolio mode.** A desk runs a client's pipeline, not one
  case at a time — so cases group under an **engagement**: a
  governed record (name, client label, owner, created by/at)
  with its own act history. Membership lives on the case
  records, and assigning, moving, or removing a case is a
  recorded act on both records — a move names where the case
  came from and where it went, and nothing moves silently.
  The **portfolio view** (`GET /api/portfolio/{engagement_id}`,
  and a console panel) folds from the member cases' records on
  every read: counts by case state, the gates awaiting decision
  across the portfolio (the workbench queue's own items, scoped
  to the engagement, with the oldest wait), each case's
  per-question support grades, open conflicts and quarantined
  claims by id and case, and the cost roll-up summed from the
  cases' own ledgers. Nothing is stored that could drift, and
  membership changes nothing inside a case: gates, budgets,
  evidence, and retention stay per case — search scoped to an
  engagement returns case-scoped hits labelled by case, a
  finding's evidence still resolves only inside its own case,
  and held or past-retention members are surfaced distinctly
  precisely because the portfolio is never a bulk handle on
  them. There is no portfolio-level decision or deletion route.

`docs/architecture.md` has the component map and the details.

## Evals

The gate is the case packs in the corpus: three gold packs (a vendor
diligence, an acquisition screen, and a partnership diligence), each
with planted facts distributed across documents, planted
contradictions, and known-correct outcomes — plus a
citation-integrity pack whose corpus holds a near-duplicate document
carrying a stale figure, where the current figure's finding must cite
the current document. `make evals` runs every case end to end on the
offline model and grades:

- **Finding recall** against the planted gold facts — gate ≥ 0.90
- **Citation completeness** on composed claims — gate 100%
- **Citation correctness** — every figure a claim states must appear
  in the evidence it cites — gate 100%
- **Budget compliance** — every run finishes inside its caps
- **Verifier quarantine** — the integrity pack also injects two
  poisoned findings (a revenue figure overstated against its cited
  filing, and a claim naming an entity its cited document never
  mentions); both must be quarantined, with the expected reasons,
  and **unsupported claims in final reports must be 0** across
  every pack
- **Conflict detection** — every pack plants a contradiction
  between documents; each must yield exactly one conflict record
  carrying both gold values, and no pack may yield any other
  conflict (period-distinct facts are graded as non-conflicts)
- **Summary integrity** — every figure, date, and named entity in
  each pack's executive summary must appear in a supported claim
  (gate: zero violations), and every claim in the report model
  must appear in both the markdown and the HTML rendering
- **Budget hardening** — a fifth case runs a gold brief with an
  execution step cap sized to exhaust mid-run: the case must land
  in BUDGET_EXHAUSTED with a partial report whose coverage table
  names at least one NOT COVERED question, no cap may be violated,
  an amendment below consumption must be refused, and a recorded
  amendment plus re-run must reach DRAFT_READY with full recall
- **Scheduled workstreams** — two dependency packs grade the
  schedule itself: a brief whose market workstream depends on its
  product workstream (the financial workstream is an independent
  branch) must show, in the audit trail, the dependent starting
  only after its prerequisite finished, with a handoff carrying
  exactly the prerequisite's findings; the failure-injection
  variant scopes the independent branch's only document out of the
  case, and that branch must end empty — its question NOT COVERED
  with the reason named — while the case still drafts
- **Red-team pack** — six adversarial scenarios grade the defences
  themselves, per attack and at zero tolerance: poisoned evidence
  whose prose contradicts its own table, a citation-bait document
  titled for claims it never makes, instruction injection inside
  a document (a fake system notice, a fake prior finding, a fake
  fact ledger planted in the body), brief-level attacks (an
  unregistered source, a question whose only sources are out of
  scope, an instruction smuggled into a question), and cross-case
  citation. Every attack must be held, and injected figures in
  final reports, cross-case citations accepted, out-of-scope
  answers, and poisoned claims in reports must all be 0. Two of
  the attacks succeeded against the defences as they stood and
  drove fixes (the verifier now judges figures against the cited
  document's own fact ledger; `record_finding` checks a fact
  reference against that ledger) — the full account, before and
  after, is `docs/red-team.md`
- **Portfolio pack** — one engagement with three cases at three
  states (one FINAL, one at the report gate, one
  BUDGET_EXHAUSTED) grades the portfolio fold exactly: the
  counts by state, the gate list against the workbench queue
  recomputed independently, the cost total recomputed from the
  cases' own ledgers, the per-question grades, and a held
  member surfaced as held. A second scenario puts the red-team
  pack's two cross-case subjects inside one engagement and
  replays the citation attacks there — foreign and unknown
  evidence IDs refused, the foreign ID dropped from a mixed
  citation, the sibling case's fact refused against this case's
  own evidence — with cross-case citations accepted at 0, and
  engagement-scoped search answering member hits only

Current results (offline stub): recall **1.000** on all four cases,
citation completeness and correctness **1.000**, budgets compliant;
the verifier checks 43 claims across the packs, quarantines exactly
the two poisoned findings, and final reports carry **0** unsupported
claims. Each pack's planted contradiction yields exactly its one
conflict record — $412M vs $421M (Helvion revenue), 1,150 vs 1,300
(Copperline headcount), 25 vs 41 minutes (Selene result time),
$186M vs $171M (Aster revenue, audited filing against its stale
preliminary release) — and no other conflicts. The budget case
exhausts on its step cap during execution, refuses a
below-consumption amendment, and after amendment re-runs to
DRAFT_READY with recall **1.000**. The dependency packs both pass:
the chain's schedule and handoff grade clean at recall **1.000**,
and the failure variant's missing branch ends empty with its
question NOT COVERED and the reason named, the case at
DRAFT_READY. The red-team pack holds all 21
attacks with its four counters at zero. The portfolio pack
grades its three-state fold exactly — counts, gates, and the
cost total recomputed from the cases' ledgers all match — and
its in-engagement citation attacks are all refused, with
cross-case citations accepted at zero. The challenge pack
grades the second opinion on two crafted subjects: Kestrel
Avionics records exactly its three challenges (two must-answer
single-source headline figures, one advisory stale side of a
conflict), approval refuses while a must-answer stands, and the
rebut, revise, and downgrade paths each answer one; Alder &
Voss, whose figures are each stated by two documents that
agree, records no challenges and approves unimpeded. The skills
pack grades the library on a brief that names two skills:
the names ride the plan, approval pins version 1 of each, the
pinned text mounts read-only, every finding of the named
workstreams carries its pins with recall at the full gold set,
and the report's Method section names what ran — while a
role-mismatched name and an unknown name refuse at the gate,
and a tool-widening version and a line-skipping version refuse
at publish, the mid-case publish leaving the pinned case on
version 1 and the next approved plan pinning version 2. The
observability pack grades the reader side: a gold case's trace
covers every audit entry in causal order and folds its runs
and priced cost from the budget and cost ledgers, the
budget-tight case traces to its partial-report outcome with
its class and uncovered questions named, the challenge pack's
crafted case classifies challenge-blocked with its open
must-answer pair, and the day's digest folds to the totals
recomputed from the cases' own ledgers and keeps its figures
when a second generation adds a case — while a sweep
classifies every case the other packs produce (gold,
integrity, both dependency packs, the clean challenge case,
the red-team case, a rejected plan, a fresh intake) and
asserts the taxonomy is total, with no terminal case in the
in-progress fallback. The deployment pack grades the boot seam:
compositions are selected through the environment, the way a
deployment selects them — local boots and runs the gold case
end to end on the stub with the deployment recorded on the
case, its runs, and its trace, while hosted without a database
URL, local with a live default, hosted-self without its base
URL, and an unknown profile name each refuse by name. The pytest suite
(319 tests — 318 run everywhere, plus a Postgres conformance leg
that skips unless `DD_TEST_DATABASE_URL` names a scratch database)
covers the gates, the budget caps, the evidence backend,
the citation rules, the ledger's drift gate, the verifier and
quarantine, conflict detection and handling, per-phase budgets,
the budget ledger, the partial-report protocol, amendment, the
harness profiles, the profile regression gate — including its
two controls run against the real suite (below) — the
workstream schedule: dependency validation at the plan gate,
waves, handoffs, failure isolation, and deterministic
composition, long-case continuity: checkpoints,
interrupted-case detection and resume, evidence arrival with
staleness marking, the proceed and re-plan decisions, findings
reuse under its evidence-unchanged rule, and analyst notes, and
source governance: the seeded registry and recorded registration,
ingestion refusal on both paths, the brief source policy,
provenance, the retention sweep, legal hold, tombstoned
deletion, and redaction with its approval-blocking claim flags,
and report depth: the executive summary and its integrity check,
the support-grade rule, risks and open items from records,
cross-format identity of claims and figures, and section notes
that annotate the export without touching the report, and the
analyst workbench: the queue derived per group from planted
states with exact waits, recorded triage, the merged timeline,
saved views, field-named search, and the route table's proof
that no bulk decision exists, and cost accounting: the price
table's fail-closed loader and grounded rates, the guard's
per-call consumption records with their measured/estimated
bases, the cost ledger folding exactly by phase, role, and model
against the budget ledger's own token counts, the price-version
pinning (an edited table moves nothing recorded), the unpriced
model, and the budget-vs-actual report on the budget-tight case
through amendment and re-run, and the red-team defences: the
verifier's ledger rule, record_finding's ledger and case-scope
checks, the intake refusal, and the pack itself held as
regression cases, and portfolio mode: engagement records and
their act history, assignment and moves recorded on both
records, the fold graded against hand-built member records
(counts, gates, costs, grades, open items), held and
past-retention members surfaced, engagement-scoped search,
cross-case citation inside one engagement refused, and the
route table's proof that no portfolio-level decision or
deletion route exists, and the challenge pass: the crafted
set recorded exactly, the approval gate over open must-answer
challenges, the three dispositions and their validations, the
advisory tier never blocking, idempotency by content,
supersede-never-delete on a moved draft, and no challenge in
any delivered format, and the skills library: the strict
loader's refusals, the publish discipline, the plan-gate pins
and refusals, the profile-relative tool rule at publish and at
the gate, the read-only mount, finding provenance, and the
byte-stability of skill-free cases, and observability: the
trace's act-for-act reconstruction and its ledger-folded runs,
the taxonomy's twelve labels each met on its own scenario and
its totality over the states a desk produces, and the digest
fold — hand-computed totals, stacked versions, snapshots that
do not move, and a generation that writes nothing to any case,
and deployment profiles: the settings-load refusals by name,
the unnamed-settings seam standing, the deployment record on
the case, its runs, and its trace, records from before the
round reading unknown, and live permission at startup decided
by the promotion log.

The eval runner also runs the packs under any other profile —
`python evals/run_evals.py --profile openai-gpt` — with the report
header naming the profile and version, so results are attributable
per profile. The default profile is the gate; another profile's run
is reported, not gated. Under `openai-gpt` with the stub standing
in, the gold packs score recall **0.700–0.769**: its narrowed
market-scan has no document reader, so the market questions come
back empty — the cost of that tuning choice, in its own column,
while citation completeness and the zero-unsupported-claims line
hold. (The citation-integrity pack, whose questions are financial
and product, still scores 1.000.) The runner also reports the two
eval subsets separately — tuning (Helvion + Copperline, 24 gold
facts) and held-out (Selene + the integrity pack, 16 gold facts) —
because the profile regression gate judges candidates on them:
run through the gate, `openai-gpt` is **NOT_PROMOTABLE** with
worst-run recalls of 0.750 (tuning) and 0.812 (held-out) against
the incumbent's 1.000 baseline, while a behavioural copy of the
incumbent is PROMOTABLE and cycles through promotion and
rollback. `python evals/promotion_controls.py` reproduces both
controls end to end.

## Production profile

`docker compose up --build` runs the production topology: the
stateless API, PostgreSQL (pgvector image; the store port is the same
one SQLite serves locally), and SeaweedFS's S3 gateway (service name
`minio`, bucket `diligence-evidence`) for evidence and file bytes.
Migrations in `migrations/` are numbered and applied at startup.
`/readiness` reports the store, object, and model profiles, and —
since the deployment-profiles round — the deployment profile and
the active harness profile. The
development profile needs none of it: SQLite at `./data/diligence.db`
(env `SQLITE_PATH`) and a local object directory.

Verified live on 2026-10-18 against PostgreSQL 16.15 + pgvector
0.8.7 — the release-hardening round ran the production store for
the first time, and it had a real defect: every payload read
assumed SQLite's wire shape (Postgres's JSONB columns arrive from
psycopg already parsed), so the hosted profile could not boot at
all. Fixed at the store layer, with the fix's regression test kept
in the suite as the env-gated conformance leg described in the
Evals section. After the fix, the hosted profile booted against
Postgres and ran a gold case end to end over HTTP — plan,
approval, execution, report approval — to FINAL, with the result
read back from the database itself; a fresh-database prod smoke
passed the same way. The unit suite keeps per-test SQLite stores
(that is its fixture contract); the Postgres adapter is proven by
the conformance leg and the hosted boot, not by re-running the
unit suite against a shared database. The compose topology itself
was not run — there is no Docker in the verification environment —
so the Docker build remains the one unverified piece of the
production story.

## Deployment profiles

A deployment is a composition — store, object storage, harness
default, live-model permission — and `DEPLOYMENT_PROFILE` names
it as one of three, validated as a whole at settings load. An
invalid combination refuses loudly by name (`ConfigError`)
instead of half-working:

- **`local`** — SQLite, the local object directory, the offline
  stub harness as the default profile, live models forbidden.
  A `DATABASE_URL` or S3 endpoint in the environment contradicts
  the composition and refuses, rather than silently re-pointing
  a local boot at production stores.
- **`hosted`** — PostgreSQL via `DATABASE_URL` (required; there
  is no default connection string), object storage chosen
  explicitly (`OBJECT_DIR` or `S3_ENDPOINT_URL`), and the harness
  default is whatever the profile regression gate has promoted.
  Live models are permitted only through a promoted harness
  profile: a hosted boot that pairs a live model with no
  promotion on record refuses at startup. The compose topology
  above boots as `hosted` — so a provider key alone no longer
  takes a fresh deployment live; the profile must first earn
  promotion through the eval gate, or the deployment runs the
  stub until one does.
- **`hosted-self`** — hosted, for a self-hosted model endpoint
  (NVIDIA NIM and its kin): the model is reached through the
  OpenAI-compatible boundary, and `OPENAI_BASE_URL` is required
  with no default on purpose — a silent default would send
  diligence traffic to a public API the deployment never chose.

Every case, and every run within it, records the deployment it
ran under — profile name, harness profile id, store backend —
on the case record, in the trace, and on the case read; work
recorded before deployment profiles existed reads as
deployment-unknown, never backfilled with the current boot's
values. Naming a profile is the act that binds its rules:
settings built without naming one (the test suite's and eval
packs' compositions) keep the behaviour they always had.

## Honest limits

- The offline stub reads facts from the corpus documents' structured
  Key facts ledgers. A real model reads the prose; the offline gate
  measures the harness and the governance, not a real model's
  extraction quality. A live-mode eval is the way to measure that.
- The evidence universe is the case corpus. Web fetch is deliberately
  not wired in this round; sources enter a case only through
  ingestion, with provenance attached — and since the governance
  round, only against the source registry: a document whose source
  nobody has registered is refused at the door, corpus or upload
  alike.
- Contradictions are planted in the corpus and surface as conflict
  records, but detection is only as wide as the governed metric
  lexicon in `conflicts.py`: a disagreement stated without a
  recognised metric, figure, or comparable value will sit in the
  report as two findings and not be flagged. Harness profiles are
  live as configuration, and one replaces the active profile only
  through the regression gate's recorded promotion — nothing here
  auto-promotes, and the gate's baseline is the incumbent's own
  recorded runs, so a store whose profiles were never evaluated
  pays for a fresh baseline (three more suite runs) on its first
  promotion evaluation. A profile's tool allowlist governs this
  blueprint's case tools; the
  harness's built-in filesystem tools are its own scaffolding and
  are not a profile's to grant or withhold. The verifier checks
  figures, dates, and named entities against cited text — it does
  not judge narrative plausibility, and its optional model check is
  advisory only.
- Skills steer by prose, and prose can be governed only so far.
  The library controls which method text a workstream loads, at
  which pinned version, under which profile's tool allowlist —
  it cannot make a model follow the method, and the offline stub
  demonstrably does not: its fact selection is a fixed keyword
  playbook whatever the method text says, which is why the
  offline gates grade the loading, the gating, and the
  provenance, and method compliance is a live-model property
  that only a live-mode run can measure. Publishing writes to
  the store, never to the repo's `skills/` directory, so a
  deployment's published versions live with its data and the
  shipped files remain the seed set; a store reset returns the
  library to the seeds.
- Observability reads what the records hold, and no more. The
  trace shows no per-act or per-step timing because no run
  records it — the audit trail's second-resolution stamps and
  the guard's per-run usage are the finest clocks the system
  keeps. A digest classifies its cases as they stand when it
  is generated, so a regenerated digest may disagree with its
  own earlier version about the same day; both are kept,
  because each was the truth when it was taken.
- Budgets cap each run (planning and execution are guarded
  separately) rather than the case as a whole.
- Costs are priced from the governed table's list rates, not from
  a provider invoice: prompt-caching and batch discounts are not
  modelled, so a discounted run's ledger figure reads high against
  its bill, never low — and what a provider actually charged is a
  finance record, not this ledger. The offline stub's dollars are
  a shadow price for comparing runs, never a spend. The verifier's
  optional advisory model check runs outside the harness runs and
  the budget guard, so it has no consumption record and is not
  costed; in the offline default it does not run at all. Cases run
  before cost accounting existed carry no cost lines, and their
  ledgers read empty rather than reconstructed.
- A checkpoint's budget snapshot is the position at the last
  completed wave. A wave that dies mid-flight consumed steps the
  ledger never saw, and those are not reconstructed: resume
  carries the recorded position forward, and the resume's audit
  event carries the snapshot it resumed from.
- LangSmith is not required anywhere. Tracing is the audit trail plus
  structured JSON logs with request IDs.

## Repository layout

```
src/deep_diligence/
  api.py          FastAPI surface + console routes
  service.py      case lifecycle, plan/report gates
  harness.py      create_deep_agent wiring, built from the profile
  profiles.py     harness profiles: the strict loader, prompt
                  assembly, and the middleware the switches control
  evaluation.py   the case-pack grading, shared by the eval runner
                  and the promotion gate (one ruler for both)
  promotion.py    the profile regression gate: split evaluation,
                  worst-of-three verdicts, the promotion and
                  rollback records that move the active profile
  stub_model.py   the deterministic offline model
  llm.py          provider boundary (OpenAI-compatible, Anthropic, Ollama)
  schedule.py     workstream dependencies: gate validation, waves,
                  and the findings handoff
  budgets.py      budget guard enforced in code around the harness;
                  also records each model call for cost accounting
  costs.py        cost accounting: the governed price table's
                  strict loader, run-end pricing of the guard's
                  per-call records, the cost ledger folded from
                  the recorded lines, and the budget-vs-actual view
  tools.py        search_corpus / read_document / record_finding
  backend.py      virtual filesystem over the evidence store
  evidence.py     corpus ingestion with IDs, sources, hashes;
                  mid-case evidence arrival under the same rules
  ledger.py       the citation ledger: claim→evidence record with
                  citation-time hashes, re-verified at the gate
  verifier.py     the independent verifier: re-checks every claim
                  against its cited evidence bytes; quarantines
                  unsupported claims before the report gate
  conflicts.py    conflict detection: keys supported claims by
                  (entity, metric, period) and records where they
                  assert different values, with source classes
  challenges.py   the challenge pass: the deterministic second
                  opinion over the verified draft, its governed
                  rules, content-keyed sync (supersede, never
                  delete), and the analyst's three dispositions
  skills.py       the skills library: versioned method artifacts,
                  the fail-closed loader with its tool rule, the
                  startup seed, and the plan gate's pin resolution
  observability.py the run trace folded from the stored records,
                  the failure taxonomy (one function, total by
                  construction), and the operations digest fold
  continuity.py   staleness keying: which workstreams a document
                  that arrived mid-case speaks to, scored in code
  governance.py   source governance: the registry and its corpus
                  seed, the brief source policy, provenance,
                  retention and legal hold, redaction
  workbench.py    the analyst workbench: the queue derived from
                  case records, the merged case timeline, saved
                  views, and deterministic case search
  portfolio.py    portfolio mode: engagements and the portfolio
                  fold — counts, gates, grades, open items, and
                  cost summed from the member cases' records
  composer.py     deterministic report assembly from findings
  report_model.py the report as a structured model: executive
                  summary under its integrity check, support
                  grades, risks and open items from records, and
                  the HTML renderer + JSON export of the one model
  corpus.py       the bundled evidence corpus (generated for this
                  blueprint), the gold case packs, and the
                  red-team pack
  store.py        case store port (SQLite local, PostgreSQL production)
  objects.py      object-store port (local directory, S3-compatible)
  static/         the console (one static HTML file)
profiles/         the harness profile artifacts (offline-stub,
                  openai-gpt), versioned JSON, strictly validated
skills/           the shipped skill artifacts (revenue-bridge,
                  contradiction-hunting, source-triangulation),
                  one versioned JSON file per version, strictly
                  validated and seeded into the store at startup
pricing/          the governed price table (price-table.json):
                  per-model input/output rates, versioned JSON,
                  strictly validated like the profiles
migrations/       numbered SQL migrations (+ sqlite/ variants)
evals/            the gold case-pack gate (--profile for per-profile
                  runs) and promotion_controls.py, the gate's two
                  controls run for real
tests/            the pytest suite
docs/             architecture notes
```

## License

Apache 2.0. This blueprint is an independent Trida AI reference
implementation built on the open-source LangChain Deep Agents
harness; it is not affiliated with or endorsed by LangChain.
