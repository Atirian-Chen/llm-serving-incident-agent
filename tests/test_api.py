import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from incident_agent.api import app
from incident_agent.rag.errors import RAGModelError


def test_health_endpoint_without_llm_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CONFIG_FILE", str(tmp_path / "missing.toml"))
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "degraded"
        assert payload["llm_provider"] == "deepseek"
        assert "LLM unavailable" in payload["llm_error"]


def test_diagnose_reports_llm_unavailable_without_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CONFIG_FILE", str(tmp_path / "missing.toml"))
    with TestClient(app) as client:
        response = client.post(
            "/diagnose",
            json={"service_id": "demo-oom", "symptom": "GPU memory is exhausted"},
        )
        assert response.status_code == 503
        assert "LLM unavailable" in response.json()["detail"]


def test_health_exposes_strict_rag_failure(monkeypatch):
    import incident_agent.api as api_module

    def fail_engine():
        raise RAGModelError("RAG unavailable: CUDA is required")

    monkeypatch.setattr(api_module, "create_engine", fail_engine)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "degraded"
        assert "RAG unavailable" in payload["llm_error"]
