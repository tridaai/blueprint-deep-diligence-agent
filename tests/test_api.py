import pytest
from fastapi.testclient import TestClient

from deep_diligence.api import create_app
from deep_diligence.corpus import GOLD_CASES


@pytest.fixture()
def client(settings):
    return TestClient(create_app(settings))


def test_health_and_readiness(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/readiness")
    assert ready.status_code == 200
    body = ready.json()
    assert body["ready"] is True
    assert body["store"] == "sqlite"
    assert body["model"] == "offline"


def test_console_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Deep Diligence Agent" in page.text
    assert "View all blueprints" in page.text


def test_corpus_endpoint(client):
    body = client.get("/api/corpus").json()
    refs = {c["subject_ref"] for c in body["companies"]}
    assert {"helvion-grid", "copperline-provisions", "selene-diagnostics"} <= refs
    assert len(body["sample_briefs"]) == 3


def test_full_flow_over_http(client):
    brief = GOLD_CASES[2]["brief"]  # Selene: the smallest case
    created = client.post("/api/cases", json=brief)
    assert created.status_code == 201
    case_id = created.json()["case_id"]
    # The plan gate holds over HTTP too.
    assert client.post(f"/api/cases/{case_id}/execute").status_code == 409
    plan = client.post(f"/api/cases/{case_id}/plan")
    assert plan.status_code == 200
    assert plan.json()["status"] == "PENDING"
    decided = client.post(
        f"/api/cases/{case_id}/plan/decision",
        json={"analyst": "API Tester", "approve": True, "note": ""},
    )
    assert decided.json()["status"] == "APPROVED"
    report = client.post(f"/api/cases/{case_id}/execute")
    assert report.status_code == 200
    assert report.json()["coverage"] == {"q1": 4, "q2": 3, "q3": 3}
    evidence = client.get(f"/api/cases/{case_id}/evidence").json()
    assert len(evidence) == 6
    detail = client.get(f"/api/cases/{case_id}/evidence/EV-0001").json()
    assert "Key facts" in detail["text"]
    budget = client.get(f"/api/cases/{case_id}/budget").json()
    assert budget["usage"]["execution"]["subagents"] == 3
    final = client.post(
        f"/api/cases/{case_id}/report/decision",
        json={"analyst": "API Tester", "approve": True, "note": ""},
    )
    assert final.json()["status"] == "FINAL"


def test_unknown_case_404(client):
    assert client.get("/api/cases/case-nope").status_code == 404
    assert client.post("/api/cases/case-nope/plan").status_code == 404


def test_invalid_brief_422(client):
    assert client.post("/api/cases", json={"subject": "x"}).status_code == 422


def test_budget_view_and_amendment_over_http(client):
    brief = {
        **GOLD_CASES[0]["brief"],
        "budget": {
            **GOLD_CASES[0]["brief"]["budget"],
            "execution": {"max_steps": 6},
        },
    }
    created = client.post("/api/cases", json=brief)
    assert created.status_code == 201
    case_id = created.json()["case_id"]
    client.post(f"/api/cases/{case_id}/plan")
    client.post(
        f"/api/cases/{case_id}/plan/decision",
        json={"analyst": "API Tester", "approve": True, "note": ""},
    )
    report = client.post(f"/api/cases/{case_id}/execute")
    assert report.status_code == 200
    assert report.json()["partial"] is True
    assert report.json()["exceeded_phase"] == "execution"
    assert report.json()["coverage_detail"]

    # Amendment from any other state is a 409; here the case is
    # exhausted, so the state is right but a below-consumption cap
    # is a 422.
    body = client.get(f"/api/cases/{case_id}/budget").json()
    view = body["view"]
    assert view["phases"]["execution"]["granted"] == {
        "max_steps": 6,
        "max_tokens": 400000,
    }
    assert view["exhaustions"][0]["kind"] == "steps"
    consumed_steps = view["phases"]["execution"]["consumed"]["steps"]
    refused = client.post(
        f"/api/cases/{case_id}/budget/amendment",
        json={
            "analyst": "API Tester",
            "reason": "Too low, on purpose",
            "budget": {
                **GOLD_CASES[0]["brief"]["budget"],
                "execution": {"max_steps": consumed_steps - 1},
            },
        },
    )
    assert refused.status_code == 422

    amended = client.post(
        f"/api/cases/{case_id}/budget/amendment",
        json={
            "analyst": "API Tester",
            "reason": "The step cap under-scoped the brief",
            "budget": {
                **GOLD_CASES[0]["brief"]["budget"],
                "execution": {"max_steps": 120},
            },
        },
    )
    assert amended.status_code == 200
    assert amended.json()["status"] == "APPROVED"

    # The case is no longer exhausted: a second amendment is a 409.
    again = client.post(
        f"/api/cases/{case_id}/budget/amendment",
        json={
            "analyst": "API Tester",
            "reason": "Once more",
            "budget": GOLD_CASES[0]["brief"]["budget"],
        },
    )
    assert again.status_code == 409

    report = client.post(f"/api/cases/{case_id}/execute")
    assert report.status_code == 200
    assert report.json()["partial"] is False
