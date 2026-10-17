# Contributing

This is a Trida AI reference blueprint. The bar is production-shaped,
original work: no copied code or prose from other projects, and every
behaviour the README claims is covered by a test or an eval.

## Workflow

```bash
make install     # uv sync --extra dev
make test        # pytest — must be green before anything else
make evals       # gold case-pack gate — must print PASS
make demo        # one case end to end, offline
```

Primary workflow is uv. Without uv: `make install-pip` installs the
frozen constraint set first and the package second (the two steps are
deliberate — pip refuses editable installs under a hashed constraints
file).

## Rules of the house

- **Gates are code.** Lifecycle rules (plan gate, report gate) live in
  `service.py`; budget caps live in `budgets.py`; the citation rule
  lives in `tools.record_finding` and again in `composer.py`. Do not
  move a rule into a prompt.
- **The harness is real.** The agent core is LangChain's `deepagents`
  package. Do not re-implement planning, sub-agent delegation, or the
  filesystem — configure the harness, and put governance around it.
- **Offline first.** Tests and evals run keyless on the stub model and
  the bundled corpus. A change that needs a live key to verify does
  not merge.
- **Budgets per run.** Each harness run (planning, execution) is
  guarded separately by the case budget. Changing that semantic is a
  design change — write it down in docs/architecture.md first.
- **Commits** are small, human-voiced, and never squashed.
