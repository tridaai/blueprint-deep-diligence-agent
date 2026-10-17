"""One gold case graded the way the eval runner grades all three."""

from deep_diligence.corpus import GOLD_CASES
from deep_diligence.models import BriefIn


def test_gold_case_recall_and_citations(service):
    gold = GOLD_CASES[1]  # Copperline
    case = service.create_case(BriefIn(**gold["brief"]))
    case_id = case["case_id"]
    service.run_planning(case_id)
    service.decide_plan(case_id, analyst="Eval", approve=True)
    report = service.run_execution(case_id)
    found = {
        f["fact_ref"] for f in service.store.list_findings(case_id) if f.get("fact_ref")
    }
    gold_facts = set(gold["gold_facts"])
    assert found == gold_facts, f"missing {gold_facts - found}, extra {found - gold_facts}"
    assert report["refused_claims"] == []
    assert all(c["evidence_ids"] for c in report["claims"])
