"""Round 13 — cost accounting.

The ledger folds exactly: on a case run end to end, the priced
lines recorded at each run's end sum — by phase, by sub-agent
role, and by model — to the case total, and their token totals
are the budget ledger's own consumption, not a second count.
The price version is pinned: a line carries the table version
and rates it was priced with, and editing the table afterwards
moves nothing recorded. The offline stub's figures are labelled
estimated wherever they surface; a model the table does not know
is recorded unpriced, never priced at an invented rate.
"""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.budgets import BudgetGuard, budget_view
from deep_diligence.config import load_settings
from deep_diligence.costs import (
    PriceError,
    fold_cost_ledger,
    load_price_table,
    price_usage_records,
)
from deep_diligence.llm import model_label
from deep_diligence.models import BriefIn, BudgetIn
from deep_diligence.service import CaseService

REPO_ROOT = __file__.rsplit("/tests/", 1)[0]


@pytest.fixture()
def client(settings):
    return TestClient(create_app(settings))


def _run_full_case(service, gold_brief):
    case = service.create_case(BriefIn(**gold_brief))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    return case_id


def _fold(service, case_id):
    return fold_cost_ledger(service.store.list_audit(case_id))


def _dec(value) -> Decimal:
    return Decimal(str(value))


# -- the price table -----------------------------------------------------------


def test_shipped_price_table_loads_with_grounded_rates():
    table = load_price_table()
    assert table.record() == {"id": "price-table", "version": 1, "currency": "USD"}
    stub = table.rate_for("offline:stub")
    assert stub.basis == "estimated"
    gpt = table.rate_for("openai:gpt-5.1")
    assert gpt.input_usd_per_million == Decimal("1.25")
    assert gpt.output_usd_per_million == Decimal("10.00")
    assert gpt.basis == "provider_rate"
    claude = table.rate_for("anthropic:claude-sonnet-4-5-20250929")
    assert claude.input_usd_per_million == Decimal("3.00")
    assert claude.output_usd_per_million == Decimal("15.00")
    ollama = table.rate_for("ollama:llama3.1")
    assert ollama.input_usd_per_million == Decimal("0")
    assert ollama.basis == "local_zero"
    assert table.rate_for("openai:no-such-model") is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update(surprise=True),
        lambda raw: raw["models"].append(dict(raw["models"][0])),
        lambda raw: raw["models"][0].update(input_usd_per_million=1.25),
        lambda raw: raw["models"][0].update(input_usd_per_million="-1"),
        lambda raw: raw.update(id="other-table"),
        lambda raw: raw["models"][0].update(basis="guesswork"),
        lambda raw: raw.update(version=0),
    ],
)
def test_price_table_loader_refuses_bad_tables(tmp_path, mutate):
    raw = json.loads(
        open(REPO_ROOT + "/pricing/price-table.json", encoding="utf-8").read()
    )
    mutate(raw)
    (tmp_path / "price-table.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PriceError):
        load_price_table(tmp_path)


def test_missing_price_table_refuses_to_load(tmp_path):
    with pytest.raises(PriceError, match="not found"):
        load_price_table(tmp_path)


def test_model_labels_match_the_price_table_keys(settings):
    assert model_label(settings) == "offline:stub"
    assert model_label(load_settings(llm_backend="openai")) == "openai:gpt-5.1"
    assert (
        model_label(load_settings(llm_backend="anthropic"))
        == "anthropic:claude-sonnet-4-5-20250929"
    )
    assert model_label(load_settings(llm_backend="ollama")) == "ollama:llama3.1"
    table = load_price_table()
    for backend in ("offline", "openai", "anthropic", "ollama"):
        label = model_label(load_settings(llm_backend=backend))
        assert table.rate_for(label) is not None


# -- the guard's consumption records --------------------------------------------


def _response(usage=None, content=""):
    message = SimpleNamespace(content=content, usage_metadata=usage)
    generation = SimpleNamespace(message=message)
    return SimpleNamespace(generations=[[generation]])


def _start_and_end(guard, *, agent_name, usage=None, content=""):
    guard.on_chat_model_start(
        {},
        [[SimpleNamespace(content="system", type="system")]],
        run_id=f"run-{len(guard.model_records) + guard.steps}",
        metadata={"lc_agent_name": agent_name},
    )
    guard.on_llm_end(
        _response(usage, content),
        run_id=f"run-{len(guard.model_records) + guard.steps - 1}",
    )


def _guard(**kwargs):
    return BudgetGuard(
        {"max_steps": 100, "max_subagents": 10, "max_tokens": 10_000_000,
         "max_wall_seconds": 3600},
        **kwargs,
    )


def test_guard_records_measured_usage_with_role_and_split():
    guard = _guard(model="openai:gpt-5.1", usage_declared=False)
    _start_and_end(
        guard,
        agent_name="financial-scan",
        usage={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
    )
    (record,) = guard.model_records
    assert record["role"] == "financial-scan"
    assert record["model"] == "openai:gpt-5.1"
    assert record["usage_basis"] == "measured"
    assert record["input_tokens"] == 1000
    assert record["output_tokens"] == 200
    assert record["other_tokens"] == 0
    assert record["total_tokens"] == 1200 == guard.tokens


def test_guard_records_unitemised_total_as_other_never_priced_as_input():
    guard = _guard(model="openai:gpt-5.1", usage_declared=False)
    _start_and_end(
        guard,
        agent_name="deep-diligence-planner",
        usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 200},
    )
    (record,) = guard.model_records
    assert record["role"] == "main"
    assert record["other_tokens"] == 50
    assert record["total_tokens"] == 200


def test_guard_labels_stub_declared_usage_as_estimated():
    guard = _guard(model="offline:stub", usage_declared=True)
    _start_and_end(
        guard,
        agent_name="market-scan",
        usage={"input_tokens": 150, "output_tokens": 40, "total_tokens": 190},
    )
    (record,) = guard.model_records
    assert record["usage_basis"] == "estimated"
    assert record["total_tokens"] == 190


def test_guard_labels_character_fallback_as_estimated():
    guard = _guard(model="openai:gpt-5.1", usage_declared=False)
    _start_and_end(guard, agent_name="product-scan", usage=None, content="x" * 400)
    (record,) = guard.model_records
    assert record["usage_basis"] == "estimated"
    assert record["output_tokens"] == 100
    assert record["total_tokens"] == guard.tokens == 100


# -- pricing records --------------------------------------------------------------


def test_price_records_exact_decimals_and_unpriced_model():
    table = load_price_table()
    records = [
        {"phase": "execution", "role": "financial-scan", "model": "openai:gpt-5.1",
         "calls": 1, "input_tokens": 1_000_000, "output_tokens": 500_000,
         "other_tokens": 0, "total_tokens": 1_500_000, "usage_basis": "measured"},
        {"phase": "execution", "role": "main", "model": "acme:unlisted-1",
         "calls": 1, "input_tokens": 10, "output_tokens": 5,
         "other_tokens": 0, "total_tokens": 15, "usage_basis": "measured"},
    ]
    lines = price_usage_records(records, table)
    priced = next(line for line in lines if line["priced"])
    # 1M input at $1.25 + 0.5M output at $10.00 = $6.25, exactly.
    assert Decimal(priced["cost_usd"]) == Decimal("6.25")
    assert priced["price_table"] == {"id": "price-table", "version": 1}
    assert priced["rates"] == {
        "input_usd_per_million": "1.25",
        "output_usd_per_million": "10.00",
    }
    unpriced = next(line for line in lines if not line["priced"])
    assert unpriced["cost_usd"] is None
    assert "no price-table entry" in unpriced["unpriced_reason"]


# -- the ledger on a real case -----------------------------------------------------


def test_cost_ledger_folds_exactly_on_a_full_case(service, gold_brief):
    case_id = _run_full_case(service, gold_brief)
    fold = _fold(service, case_id)
    totals = fold["totals"]
    # The stub's declared figures make the case's cost a known
    # quantity: 32 calls, 6,180 tokens, $0.0198 at the shadow rate.
    assert totals["calls"] == 32
    assert totals["total_tokens"] == 6180
    assert totals["cost_usd"] == Decimal("0.0198")
    # Every breakdown sums to the total — to the cent and beyond,
    # because the money is Decimal end to end.
    for breakdown in (fold["by_phase"], fold["by_role"], fold["by_model"]):
        assert sum(b["cost_usd"] for b in breakdown.values()) == totals["cost_usd"]
        assert sum(b["total_tokens"] for b in breakdown.values()) == 6180
        assert sum(b["calls"] for b in breakdown.values()) == 32
    assert sum(
        Decimal(line["cost_usd"]) for line in fold["lines"]
    ) == totals["cost_usd"]
    # By phase, against the known stub run.
    assert fold["by_phase"]["planning"]["cost_usd"] == Decimal("0.00255")
    assert fold["by_phase"]["execution"]["cost_usd"] == Decimal("0.01725")
    # By role: the main agent plans and orchestrates; each scan
    # runs its own workstream.
    assert set(fold["by_role"]) == {
        "main", "financial-scan", "product-scan", "market-scan",
    }
    assert fold["by_role"]["main"]["cost_usd"] == Decimal("0.0051")
    for role in ("financial-scan", "product-scan", "market-scan"):
        assert fold["by_role"][role]["cost_usd"] == Decimal("0.0049")
    assert set(fold["by_model"]) == {"offline:stub"}
    # Every line pins its price-table version and rates.
    for line in fold["lines"]:
        assert line["price_table"] == {"id": "price-table", "version": 1}
        assert line["rates"] == {
            "input_usd_per_million": "1.25",
            "output_usd_per_million": "10.00",
        }
        assert line["usage_basis"] == "estimated"
    assert fold["price_tables"] == [
        {"id": "price-table", "version": 1, "currency": "USD"}
    ]
    assert fold["contains_estimates"] is True
    assert fold["has_unpriced"] is False


def test_cost_tokens_are_the_budget_ledgers_own_tokens(service, gold_brief):
    case_id = _run_full_case(service, gold_brief)
    case = service.store.get_case(case_id)
    audit = service.store.list_audit(case_id)
    view = budget_view(case, audit)["view"]
    fold = fold_cost_ledger(audit)
    for phase in ("planning", "execution"):
        assert (
            fold["by_phase"][phase]["total_tokens"]
            == view["phases"][phase]["consumed"]["tokens"]
        )
        assert (
            fold["by_phase"][phase]["calls"]
            == view["phases"][phase]["consumed"]["steps"]
        )
    assert fold["totals"]["total_tokens"] == sum(
        view["phases"][p]["consumed"]["tokens"] for p in ("planning", "execution")
    )


def test_budget_vs_actual_report_joins_both_ledgers(service, gold_brief):
    case_id = _run_full_case(service, gold_brief)
    report = service.cost_report(case_id)
    actual = report["budget_vs_actual"]
    assert actual["phases"]["planning"]["granted"] == {
        "max_steps": 120, "max_tokens": 400000,
    }
    assert actual["phases"]["execution"]["consumed"]["steps"] == 28
    assert actual["phases"]["execution"]["consumed"]["tokens"] == 5400
    assert actual["case"]["granted"] == {"max_subagents": 6, "max_wall_seconds": 600}
    assert actual["case"]["consumed"]["subagents"] == 3
    assert _dec(report["cost"]["totals"]["cost_usd"]) == Decimal("0.0198")
    # The estimated label rides on the report itself.
    assert report["contains_estimates"] is True
    assert report["usage_label"].startswith("estimated")
    assert report["price_tables_recorded"] == [
        {"id": "price-table", "version": 1, "currency": "USD"}
    ]
    assert report["price_table_current"]["version"] == 1


def test_price_table_edit_never_rewrites_a_recorded_ledger(
    service, gold_brief, tmp_path
):
    case_id = _run_full_case(service, gold_brief)
    before = _fold(service, case_id)
    # A new version of the table, with the stub's shadow rate
    # moved two orders of magnitude.
    raw = json.loads(
        open(REPO_ROOT + "/pricing/price-table.json", encoding="utf-8").read()
    )
    raw["version"] = 2
    for entry in raw["models"]:
        if entry["model"] == "offline:stub":
            entry["input_usd_per_million"] = "125.00"
            entry["output_usd_per_million"] = "1000.00"
    (tmp_path / "price-table.json").write_text(json.dumps(raw), encoding="utf-8")
    edited = load_price_table(tmp_path)
    assert edited.version == 2
    # The recorded ledger is a fold of recorded lines: it does not
    # consult the table, so it cannot move.
    after = _fold(service, case_id)
    assert after["totals"] == before["totals"]
    assert after["lines"] == before["lines"]
    assert after["price_tables"] == [
        {"id": "price-table", "version": 1, "currency": "USD"}
    ]
    # Pricing the same consumption under v2 is a different, larger
    # number — which is exactly why the fold never does it silently.
    repriced = price_usage_records(
        [
            {"phase": "planning", "role": "main", "model": "offline:stub",
             "calls": 4, "input_tokens": 600, "output_tokens": 180,
             "other_tokens": 0, "total_tokens": 780, "usage_basis": "estimated"}
        ],
        edited,
    )
    assert Decimal(repriced[0]["cost_usd"]) > Decimal("0.00255")
    assert repriced[0]["price_table"]["version"] == 2


def test_cost_on_the_budget_tight_case_matches_the_r5_ledger(
    service, gold_brief
):
    brief = BriefIn(
        **{
            **gold_brief,
            "budget": {**gold_brief["budget"], "execution": {"max_steps": 6}},
        }
    )
    case = service.create_case(brief)
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Tester", approve=True)
    service.run_execution(case_id)
    assert service.store.get_case(case_id)["status"] == "BUDGET_EXHAUSTED"

    audit = service.store.list_audit(case_id)
    view = budget_view(service.store.get_case(case_id), audit)["view"]
    fold = fold_cost_ledger(audit)
    # The stopped run still recorded its cost, and its tokens are
    # the budget ledger's consumed tokens, phase for phase.
    for phase in ("planning", "execution"):
        assert (
            fold["by_phase"][phase]["total_tokens"]
            == view["phases"][phase]["consumed"]["tokens"]
        )
    report = service.cost_report(case_id)
    assert report["budget_vs_actual"]["exhaustions"]
    assert report["budget_vs_actual"]["exhaustions"][0]["kind"] == "steps"
    first_total = fold["totals"]["cost_usd"]
    assert first_total > 0

    # Amend and re-run: both attempts stand in the ledger, and the
    # totals accumulate — the first run's cost is not overwritten.
    service.amend_budget(
        case_id,
        analyst="Tester",
        reason="The step cap under-scoped a three-workstream brief",
        budget=BudgetIn(
            **{**gold_brief["budget"], "execution": {"max_steps": 120}}
        ),
    )
    service.run_execution(case_id)
    fold_after = _fold(service, case_id)
    assert fold_after["totals"]["cost_usd"] > first_total
    assert fold_after["runs_recorded"] >= 3  # planning + two executions
    events = [a["event"] for a in service.store.list_audit(case_id)]
    assert events.count("cost_recorded") == fold_after["runs_recorded"]


def test_case_with_no_runs_has_an_empty_honest_ledger(service, gold_brief):
    case = service.create_case(BriefIn(**gold_brief))
    report = service.cost_report(case["case_id"])
    assert report["cost"]["totals"]["calls"] == 0
    assert _dec(report["cost"]["totals"]["cost_usd"]) == Decimal("0")
    assert report["cost"]["lines"] == []
    assert report["contains_estimates"] is False


def test_all_cases_totals_sum_the_case_ledgers(service, gold_brief):
    first = _run_full_case(service, gold_brief)
    totals = service.cost_totals()
    assert totals["totals"]["cases"] == 1
    assert _dec(totals["totals"]["cost_usd"]) == Decimal("0.0198")
    assert totals["totals"]["total_tokens"] == 6180
    (entry,) = totals["cases"]
    assert entry["case_id"] == first
    assert entry["contains_estimates"] is True


# -- HTTP surface -----------------------------------------------------------------


def test_cost_endpoints_over_http(client, service, gold_brief):
    case_id = _run_full_case(service, gold_brief)
    body = client.get(f"/api/cases/{case_id}/cost")
    assert body.status_code == 200
    body = body.json()
    assert _dec(body["cost"]["totals"]["cost_usd"]) == Decimal("0.0198")
    assert body["usage_label"].startswith("estimated")
    assert set(body["cost"]["by_role"]) == {
        "main", "financial-scan", "product-scan", "market-scan",
    }
    totals = client.get("/api/costs").json()
    assert _dec(totals["totals"]["cost_usd"]) == Decimal("0.0198")
    assert client.get("/api/cases/case-nope/cost").status_code == 404


def test_console_shows_the_cost_ledger(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Cost — what this case cost to run" in page.text
    assert "/cost`" in page.text
