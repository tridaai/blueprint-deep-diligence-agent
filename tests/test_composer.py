from deep_diligence.composer import compose_report


def test_claims_render_with_citations(service, approved_case):
    case_id = approved_case["case_id"]
    service.run_execution(case_id)
    report = service.store.get_report(case_id)
    assert report["refused_claims"] == []
    assert report["coverage"] == {"q1": 5, "q2": 5, "q3": 3}
    for claim in report["claims"]:
        assert claim["evidence_ids"], "every composed claim must cite evidence"
        for eid in claim["evidence_ids"]:
            assert f"[{eid}]" in report["markdown"]


def test_composer_refuses_uncited_or_bogus_claims(service, case):
    case_id = case["case_id"]
    service.store.add_finding(
        {
            "finding_id": "FND-9001",
            "case_id": case_id,
            "question_id": "q1",
            "claim": "A claim resting on evidence that does not exist.",
            "evidence_ids": ["EV-9999"],
            "fact_ref": None,
        }
    )
    service.store.add_finding(
        {
            "finding_id": "FND-9002",
            "case_id": case_id,
            "question_id": "q1",
            "claim": "A claim with no citation at all.",
            "evidence_ids": [],
            "fact_ref": None,
        }
    )
    report = compose_report(service.store, service.store.get_case(case_id))
    assert report["refused_claims"] == ["FND-9001", "FND-9002"]
    assert "A claim resting on evidence" not in report["markdown"]
    assert report["claims"] == []


def test_record_finding_tool_enforces_citations(service, case):
    from deep_diligence.tools import build_case_tools

    tools = {t.name: t for t in build_case_tools(service.store, service.objects, case)}
    refused = tools["record_finding"].invoke(
        {"question_id": "q1", "claim": "No evidence.", "evidence_ids": []}
    )
    assert "ERROR" in refused
    refused = tools["record_finding"].invoke(
        {"question_id": "q1", "claim": "Bogus evidence.", "evidence_ids": ["EV-9999"]}
    )
    assert "ERROR" in refused
    ok = tools["record_finding"].invoke(
        {
            "question_id": "q1",
            "claim": "Revenue grew 18% year over year to $412M in FY2025.",
            "evidence_ids": ["EV-0001"],
            "fact_ref": "F-HEL-101",
        }
    )
    assert ok.startswith("Recorded FND-")
    again = tools["record_finding"].invoke(
        {
            "question_id": "q1",
            "claim": "Revenue grew 18% year over year to $412M in FY2025.",
            "evidence_ids": ["EV-0001"],
            "fact_ref": "F-HEL-101",
        }
    )
    assert again.startswith("Already recorded")
    assert service.store.count_findings(case["case_id"]) == 1
