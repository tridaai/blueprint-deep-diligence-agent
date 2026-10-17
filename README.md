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

Real-provider packages are an extra: `uv sync --extra llm`.

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
- **Budgets in code.** Each run is wrapped by a callback guard that
  counts model steps, sub-agent spawns, tokens, and wall-clock time
  against the brief's budget. Crossing a cap aborts the run; the
  findings recorded so far survive, and the composer produces a
  partial report that names the questions it did not cover instead
  of truncating silently.
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
- **The report gate.** The draft is final only when an analyst
  approves it — and approval first re-verifies the citation ledger
  against the evidence store. If a cited item's bytes no longer hash
  to the value recorded at citation time, the evidence changed after
  the fact: the report is flagged **EVIDENCE_DRIFT** and cannot be
  approved until the ledger re-verifies clean. Rejection at either
  gate is a normal state, with the analyst's note kept on the record.
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

Current results (offline stub): recall **1.000** on all four cases,
citation completeness and correctness **1.000**, budgets compliant;
the verifier checks 43 claims across the packs, quarantines exactly
the two poisoned findings, and final reports carry **0** unsupported
claims. The pytest suite (39 tests) covers the gates, the budget
caps, the evidence backend, the citation rules, the ledger's drift
gate, and the verifier and quarantine separately.

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
- Contradictions are planted in the corpus and both sides surface as
  findings; explicit conflict records in the report are a later
  round, as is harness-profile promotion. The verifier checks
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
  harness.py      create_deep_agent wiring, prompts, sub-agent roster
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
  composer.py     deterministic report assembly from findings
  corpus.py       the bundled evidence corpus (generated for this
                  blueprint) and the gold case packs
  store.py        case store port (SQLite local, PostgreSQL production)
  objects.py      object-store port (local directory, S3-compatible)
  static/         the console (one static HTML file)
migrations/       numbered SQL migrations (+ sqlite/ variants)
evals/            the gold case-pack gate
tests/            the pytest suite
docs/             architecture notes
```

## License

Apache 2.0. This blueprint is an independent Trida AI reference
implementation built on the open-source LangChain Deep Agents
harness; it is not affiliated with or endorsed by LangChain.
