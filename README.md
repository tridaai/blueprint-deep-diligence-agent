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
- **Sub-agent workstreams.** On approval, the planner delegates each
  workstream with the harness's `task` tool to a specialist
  (financial-scan, product-scan, market-scan) running in an isolated
  context. Sub-agents search the case evidence and read documents
  through the virtual filesystem, where evidence is mounted read-only
  at `/evidence/<EV-ID>.md` — the agent cannot alter its sources.
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
  exhaustion stays in the ledger.
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
- **Budget hardening** — a fifth case runs a gold brief with an
  execution step cap sized to exhaust mid-run: the case must land
  in BUDGET_EXHAUSTED with a partial report whose coverage table
  names at least one NOT COVERED question, no cap may be violated,
  an amendment below consumption must be refused, and a recorded
  amendment plus re-run must reach DRAFT_READY with full recall

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
DRAFT_READY with recall **1.000**. The pytest suite
(111 tests) covers the gates, the budget caps, the evidence backend,
the citation rules, the ledger's drift gate, the verifier and
quarantine, conflict detection and handling, per-phase budgets,
the budget ledger, the partial-report protocol, amendment, the
harness profiles, and the profile regression gate — including its
two controls run against the real suite (below).

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
`/readiness` reports the store, object, and model profiles. The
development profile needs none of it: SQLite at `./data/diligence.db`
(env `SQLITE_PATH`) and a local object directory.

## Honest limits

- The offline stub reads facts from the corpus documents' structured
  Key facts ledgers. A real model reads the prose; the offline gate
  measures the harness and the governance, not a real model's
  extraction quality. A live-mode eval is the way to measure that.
- The evidence universe is the case corpus. Web fetch is deliberately
  not wired in this round; sources enter a case only through
  ingestion, with provenance attached.
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
- Budgets cap each run (planning and execution are guarded
  separately) rather than the case as a whole.
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
  budgets.py      budget guard enforced in code around the harness
  tools.py        search_corpus / read_document / record_finding
  backend.py      virtual filesystem over the evidence store
  evidence.py     corpus ingestion with IDs, sources, hashes
  ledger.py       the citation ledger: claim→evidence record with
                  citation-time hashes, re-verified at the gate
  verifier.py     the independent verifier: re-checks every claim
                  against its cited evidence bytes; quarantines
                  unsupported claims before the report gate
  conflicts.py    conflict detection: keys supported claims by
                  (entity, metric, period) and records where they
                  assert different values, with source classes
  composer.py     deterministic report assembly from findings
  corpus.py       the bundled evidence corpus (generated for this
                  blueprint) and the gold case packs
  store.py        case store port (SQLite local, PostgreSQL production)
  objects.py      object-store port (local directory, S3-compatible)
  static/         the console (one static HTML file)
profiles/         the harness profile artifacts (offline-stub,
                  openai-gpt), versioned JSON, strictly validated
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
