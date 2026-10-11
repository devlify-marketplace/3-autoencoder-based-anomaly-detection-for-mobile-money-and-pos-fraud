from fastapi.testclient import TestClient

from fraud_autoencoder.api import app

client = TestClient(app)


def test_health_reports_proxy_scope_and_model_readiness():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "European credit-card" in payload["benchmark"]
    assert "Not a Nigerian" in payload["scope_warning"]
    assert isinstance(payload["model_ready"], bool)


def test_missing_artifacts_do_not_return_fake_metrics():
    response = client.get("/api/metrics")
    assert response.status_code in (200, 404)
    if response.status_code == 404:
        assert "Train the model" in response.json()["detail"]
