"""Cost accounting: what a case cost to run, priced in code.

Round 5 gave the desk a budget ledger — steps, spawns, tokens, and
wall clock, granted against consumed, folded from the audit trail.
This module prices that consumption. The chain of custody is the
same as for budgets, one link longer:

  * the BudgetGuard (budgets.py) records every model call as it
    ends — phase, role (the main agent, or the sub-agent role the
    harness named the call for), model, and the token split the
    call reported, with the basis of those figures: measured
    (a real provider returned usage metadata), or estimated (the
    offline stub's declared per-call figures, or a character
    estimate when a provider returns no usage at all);
  * when a run ends, the service prices those records against the
    governed price table (``pricing/price-table.json`` — per-model
    input/output rates held as data, versioned like the harness
    profiles) and records the priced lines as a `cost_recorded`
    entry in the case's audit trail. Each line carries the price
    table's id + version and the rates it was priced with;
  * `fold_cost_ledger` folds those recorded lines — never the
    current table — into the per-case ledger: totals and the
    breakdowns by phase, by sub-agent role, and by model.

Because the fold reads only recorded lines, editing the price
table afterwards changes nothing historical: a later version
prices later runs, and the versions a case was priced under are
named on its ledger. There is deliberately no re-pricing view —
the recorded ledger stands.

Honesty rules, enforced by the shape of the data:

  * An offline run's figures are labelled `estimated` everywhere
    they surface. The stub's token counts are its declared
    per-call figures, and its dollars are a shadow price at the
    table's reference rate — useful for comparing runs, never a
    provider figure, and the ledger says so.
  * A model with no entry in the price table is recorded
    `unpriced` with the reason named. Its tokens still count; its
    cost is absent, never invented, and totals say how many
    tokens stand unpriced beside the priced total.
  * Costs are computed with Decimal and recorded as exact strings
    on the audit lines; the views convert to JSON numbers at the
    boundary, rounded to 8 decimal places — lossless at the
    table's rates, so a breakdown's parts sum to its total.

The verifier's optional advisory model check is not costed: it
runs outside the harness runs (and outside the budget guard), so
it has no consumption record to price. In the offline default it
does not run at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from .budgets import budget_view

COST_EVENT = "cost_recorded"
ROLES = ("main", "financial-scan", "product-scan", "market-scan")
_USAGE_BASES = ("measured", "estimated")
_RATE_BASES = ("provider_rate", "estimated", "local_zero")
_MILLION = Decimal(1_000_000)
_EIGHT_PLACES = Decimal("0.00000001")

_TOP_KEYS = {"id", "version", "currency", "note", "models"}
_MODEL_KEYS = {"model", "input_usd_per_million", "output_usd_per_million", "basis", "note"}


class PriceError(RuntimeError):
    """A price table that refuses to load. The message names the problem."""


@dataclass(frozen=True)
class ModelRate:
    model: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    basis: str
    note: str


@dataclass(frozen=True)
class PriceTable:
    id: str
    version: int
    currency: str
    note: str
    models: tuple[ModelRate, ...]

    def record(self) -> dict:
        """The stamp a recorded cost line carries for this table."""
        return {"id": self.id, "version": self.version, "currency": self.currency}

    def rate_for(self, model: str) -> ModelRate | None:
        for entry in self.models:
            if entry.model == model:
                return entry
        return None


def default_pricing_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "pricing"


def _fail(source: str, problem: str) -> PriceError:
    return PriceError(f"price table {source}: {problem}")


def _rate(value, source: str, model: str, field: str) -> Decimal:
    # Rates are stored as strings in the table so the Decimal the
    # ledger is priced with is exactly the decimal that was
    # written, never a binary float's approximation of it.
    if not isinstance(value, str):
        raise _fail(
            source,
            f"model {model!r} rate {field!r} must be a string "
            "(rates are exact decimals, not floats)",
        )
    try:
        rate = Decimal(value)
    except InvalidOperation:
        raise _fail(
            source, f"model {model!r} rate {field!r} is not a decimal: {value!r}"
        ) from None
    if not rate.is_finite() or rate < 0:
        raise _fail(source, f"model {model!r} rate {field!r} must be >= 0")
    return rate


def _validate(raw, source: str) -> PriceTable:
    if not isinstance(raw, dict):
        raise _fail(source, "the price table file must contain a JSON object")
    unknown = sorted(set(raw) - _TOP_KEYS)
    if unknown:
        raise _fail(source, f"unknown keys: {', '.join(unknown)}")
    missing = sorted(_TOP_KEYS - set(raw))
    if missing:
        raise _fail(source, f"missing required keys: {', '.join(missing)}")
    table_id = raw["id"]
    if not isinstance(table_id, str) or not table_id:
        raise _fail(source, "'id' must be a non-empty string")
    stem = Path(source).stem
    if stem != table_id:
        raise _fail(
            source,
            f"table id {table_id!r} does not match its file name {stem!r}",
        )
    version = raw["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise _fail(source, "'version' must be an integer >= 1")
    if not isinstance(raw["currency"], str) or not raw["currency"]:
        raise _fail(source, "'currency' must be a non-empty string")
    if not isinstance(raw["note"], str):
        raise _fail(source, "'note' must be a string")
    raw_models = raw["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise _fail(source, "'models' must be a non-empty list")
    models: list[ModelRate] = []
    seen: set[str] = set()
    for entry in raw_models:
        if not isinstance(entry, dict):
            raise _fail(source, "each models entry must be an object")
        model = entry.get("model")
        where = f"model {model!r}" if isinstance(model, str) else "models entry"
        unknown = sorted(set(entry) - _MODEL_KEYS)
        if unknown:
            raise _fail(source, f"{where} has unknown keys: {', '.join(unknown)}")
        missing = sorted(_MODEL_KEYS - set(entry))
        if missing:
            raise _fail(source, f"{where} is missing required keys: {', '.join(missing)}")
        if not isinstance(model, str) or not model:
            raise _fail(source, "each models entry needs a non-empty 'model'")
        if model in seen:
            raise _fail(source, f"duplicate model entry {model!r}")
        seen.add(model)
        if entry["basis"] not in _RATE_BASES:
            raise _fail(
                source,
                f"model {model!r} basis {entry['basis']!r} is not one of: "
                + ", ".join(_RATE_BASES),
            )
        if not isinstance(entry["note"], str):
            raise _fail(source, f"model {model!r} 'note' must be a string")
        models.append(
            ModelRate(
                model=model,
                input_usd_per_million=_rate(
                    entry["input_usd_per_million"], source, model,
                    "input_usd_per_million",
                ),
                output_usd_per_million=_rate(
                    entry["output_usd_per_million"], source, model,
                    "output_usd_per_million",
                ),
                basis=entry["basis"],
                note=entry["note"],
            )
        )
    return PriceTable(
        id=table_id,
        version=version,
        currency=raw["currency"],
        note=raw["note"],
        models=tuple(models),
    )


def load_price_table(pricing_dir=None, table_id: str = "price-table") -> PriceTable:
    """Load and validate the governed price table, or refuse with a
    named PriceError. Like a harness profile, a table that cannot
    load refuses the startup — costs are never priced against a
    silent fallback."""
    directory = Path(pricing_dir) if pricing_dir else default_pricing_dir()
    path = directory / f"{table_id}.json"
    if not path.is_file():
        raise PriceError(
            f"price table {table_id!r} not found: no file {path} "
            f"(price tables live in {directory}; PRICING_DIR selects the directory)"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PriceError(f"price table {path}: not valid JSON: {exc}") from exc
    return _validate(raw, str(path))


# -- pricing one run's consumption -------------------------------------------


def _money(value: Decimal) -> Decimal:
    return value.quantize(_EIGHT_PLACES, rounding=ROUND_HALF_UP)


def _combined_basis(bases) -> str:
    distinct = set(bases)
    if len(distinct) == 1:
        return distinct.pop()
    return "mixed"


def price_usage_records(records: list[dict], table: PriceTable) -> list[dict]:
    """Price one run's per-call consumption records (from the
    BudgetGuard) into ledger lines aggregated by phase, role, and
    model. Money is Decimal, recorded on each line as an exact
    string alongside the table version and the rates used — the
    line is the historical record; nothing later re-prices it."""
    grouped: dict[tuple, dict] = {}
    for record in records:
        key = (
            record.get("phase") or "execution",
            record.get("role") or "main",
            record.get("model") or "unknown",
        )
        line = grouped.setdefault(
            key,
            {
                "phase": key[0],
                "role": key[1],
                "model": key[2],
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "other_tokens": 0,
                "total_tokens": 0,
                "_bases": [],
            },
        )
        line["calls"] += int(record.get("calls", 1))
        line["input_tokens"] += int(record.get("input_tokens", 0))
        line["output_tokens"] += int(record.get("output_tokens", 0))
        line["other_tokens"] += int(record.get("other_tokens", 0))
        line["total_tokens"] += int(record.get("total_tokens", 0))
        line["_bases"].append(record.get("usage_basis") or "estimated")
    lines: list[dict] = []
    phase_order = {"planning": 0, "execution": 1}
    role_order = {role: i for i, role in enumerate(ROLES)}
    for line in sorted(
        grouped.values(),
        key=lambda item: (
            phase_order.get(item["phase"], 9),
            role_order.get(item["role"], 9),
            item["model"],
        ),
    ):
        bases = line.pop("_bases")
        rate = table.rate_for(line["model"])
        priced = rate is not None
        if priced:
            input_cost = _money(
                Decimal(line["input_tokens"]) * rate.input_usd_per_million / _MILLION
            )
            output_cost = _money(
                Decimal(line["output_tokens"]) * rate.output_usd_per_million / _MILLION
            )
            line.update(
                {
                    "priced": True,
                    "rate_basis": rate.basis,
                    "rates": {
                        "input_usd_per_million": str(rate.input_usd_per_million),
                        "output_usd_per_million": str(rate.output_usd_per_million),
                    },
                    "input_cost_usd": str(input_cost),
                    "output_cost_usd": str(output_cost),
                    "cost_usd": str(input_cost + output_cost),
                    "unpriced_reason": None,
                }
            )
        else:
            line.update(
                {
                    "priced": False,
                    "rate_basis": None,
                    "rates": None,
                    "input_cost_usd": None,
                    "output_cost_usd": None,
                    "cost_usd": None,
                    "unpriced_reason": (
                        f"no price-table entry for model {line['model']!r} "
                        f"in {table.id} v{table.version}; its tokens are "
                        "counted and its cost is not invented"
                    ),
                }
            )
        line["usage_basis"] = _combined_basis(bases)
        line["usage_counts"] = {
            basis: bases.count(basis) for basis in sorted(set(bases))
        }
        line["price_table"] = {"id": table.id, "version": table.version}
        line["currency"] = table.currency
        lines.append(line)
    return lines


# -- the cost ledger ------------------------------------------------------------


def _empty_bucket() -> dict:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "other_tokens": 0,
        "total_tokens": 0,
        "cost": Decimal(0),
        "priced_calls": 0,
        "unpriced_calls": 0,
        "unpriced_tokens": 0,
        "_bases": set(),
    }


def _absorb(bucket: dict, line: dict) -> None:
    bucket["calls"] += int(line.get("calls", 0))
    bucket["input_tokens"] += int(line.get("input_tokens", 0))
    bucket["output_tokens"] += int(line.get("output_tokens", 0))
    bucket["other_tokens"] += int(line.get("other_tokens", 0))
    bucket["total_tokens"] += int(line.get("total_tokens", 0))
    bucket["_bases"].add(line.get("usage_basis") or "estimated")
    if line.get("priced"):
        bucket["priced_calls"] += int(line.get("calls", 0))
        bucket["cost"] += Decimal(line.get("cost_usd") or "0")
    else:
        bucket["unpriced_calls"] += int(line.get("calls", 0))
        bucket["unpriced_tokens"] += int(line.get("total_tokens", 0))


def _finish_bucket(bucket: dict) -> dict:
    bases = bucket.pop("_bases")
    cost = bucket.pop("cost")
    return {
        **bucket,
        "cost_usd": _money(cost),
        "usage_basis": _combined_basis(bases) if bases else None,
    }


def fold_cost_ledger(audit: list[dict]) -> dict:
    """Fold a case's audit trail into its cost ledger.

    Reads only `cost_recorded` entries — the priced lines a run
    recorded when it ended, each pinned to the price-table version
    and rates it was computed with. The current price table is
    never consulted here, which is what makes the fold historical:
    edit the table and this fold does not move. Costs are Decimal
    in the fold; the views convert at the JSON boundary.
    """
    total = _empty_bucket()
    by_phase: dict[str, dict] = {}
    by_role: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    lines: list[dict] = []
    price_tables: list[dict] = []
    runs = 0
    for entry in audit:
        if entry["event"] != COST_EVENT:
            continue
        runs += 1
        payload = entry.get("payload") or {}
        stamp = payload.get("price_table") or {}
        if stamp and stamp not in price_tables:
            price_tables.append(stamp)
        for line in payload.get("lines") or []:
            lines.append(line)
            _absorb(total, line)
            _absorb(by_phase.setdefault(line["phase"], _empty_bucket()), line)
            _absorb(by_role.setdefault(line["role"], _empty_bucket()), line)
            _absorb(by_model.setdefault(line["model"], _empty_bucket()), line)
    total = _finish_bucket(total)
    return {
        "runs_recorded": runs,
        "lines": lines,
        "totals": total,
        "by_phase": {
            key: _finish_bucket(by_phase[key])
            for key in sorted(by_phase, key=lambda p: {"planning": 0, "execution": 1}.get(p, 9))
        },
        "by_role": {
            key: _finish_bucket(by_role[key])
            for key in sorted(by_role, key=lambda r: {role: i for i, role in enumerate(ROLES)}.get(r, 9))
        },
        "by_model": {key: _finish_bucket(by_model[key]) for key in sorted(by_model)},
        "price_tables": price_tables,
        "contains_estimates": any(
            line.get("usage_basis") != "measured" for line in lines
        ),
        "has_unpriced": any(not line.get("priced") for line in lines),
    }


# -- views ---------------------------------------------------------------------


def _jsonable(value):
    if isinstance(value, Decimal):
        return float(_money(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def cost_view(case: dict, audit: list[dict], current_table: PriceTable | None = None) -> dict:
    """The per-case budget-vs-actual report: the R5 budget ledger's
    granted-vs-consumed picture (steps, tokens, spawns, wall clock)
    joined with the priced cost ledger (totals and breakdowns by
    phase, by sub-agent role, and by model), folded from the same
    audit trail. The cost figures are the recorded ones; the
    current price table is reported for reference only, beside the
    versions the case was actually priced under."""
    fold = fold_cost_ledger(audit)
    budget = budget_view(case, audit)["view"]
    contains_estimates = fold["contains_estimates"]
    return _jsonable(
        {
            "case_id": case["case_id"],
            "currency": (
                current_table.currency if current_table else "USD"
            ),
            "budget_vs_actual": {
                "phases": budget["phases"],
                "case": budget["case"],
                "exhaustions": budget["exhaustions"],
                "amendments": budget["amendments"],
            },
            "cost": {
                "totals": fold["totals"],
                "by_phase": fold["by_phase"],
                "by_role": fold["by_role"],
                "by_model": fold["by_model"],
                "lines": fold["lines"],
                "runs_recorded": fold["runs_recorded"],
            },
            "price_tables_recorded": fold["price_tables"],
            "price_table_current": (
                current_table.record() if current_table else None
            ),
            "contains_estimates": contains_estimates,
            "has_unpriced": fold["has_unpriced"],
            "usage_label": (
                "estimated — token figures are the offline stub's "
                "declared per-call figures (or character estimates "
                "where a provider returned no usage), and costs are "
                "shadow-priced at the price table's recorded rates; "
                "not a provider figure"
                if contains_estimates
                else "measured — token figures are the usage the "
                "provider returned, priced at the price table's "
                "recorded rates"
            ),
        }
    )
