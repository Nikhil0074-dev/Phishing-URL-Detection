"""Tests for the Flask application and its API blueprints.

These tests use the trained model artifacts under models/. Run
'python -m app.ml.train' first if models/trained is empty.
"""

import pytest

from app.application import create_app
from config.config import SCALER_PATH, TRAINED_MODEL_DIR

MODELS_TRAINED = SCALER_PATH.exists() and any(TRAINED_MODEL_DIR.glob("*.pkl"))


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


# ------------------------------------------------------------------ pages
def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"URL" in response.data


def test_dashboard_page_loads(client):
    assert client.get("/dashboard").status_code == 200


def test_models_page_loads(client):
    assert client.get("/models").status_code == 200


def test_analysis_page_loads(client):
    assert client.get("/analysis").status_code == 200


def test_predictions_page_loads(client):
    assert client.get("/predictions").status_code == 200


def test_reports_page_loads(client):
    assert client.get("/reports").status_code == 200


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_unknown_page_returns_404(client):
    response = client.get("/this-page-does-not-exist")
    assert response.status_code == 404


# --------------------------------------------------------------- validate
def test_validate_accepts_good_url(client):
    response = client.post("/api/validate", json={"url": "https://example.com"})
    assert response.status_code == 200
    assert response.get_json()["valid"] is True


def test_validate_rejects_empty_url(client):
    response = client.post("/api/validate", json={"url": ""})
    assert response.status_code == 400
    assert response.get_json()["valid"] is False


def test_validate_rejects_malformed_url(client):
    response = client.post("/api/validate", json={"url": "not a url"})
    assert response.status_code == 400


# --------------------------------------------------------------- features
def test_features_endpoint_returns_feature_vector(client):
    response = client.post("/api/features", json={"url": "https://example.com/login"})
    assert response.status_code == 200
    payload = response.get_json()
    assert "features" in payload
    assert "feature_groups" in payload
    assert payload["feature_count"] == len(payload["features"])


def test_features_endpoint_rejects_invalid_url(client):
    response = client.post("/api/features", json={"url": "not a url"})
    assert response.status_code == 400


def test_features_batch_endpoint(client):
    response = client.post(
        "/api/features/batch", json={"urls": ["https://a.com", "https://b.com"]}
    )
    assert response.status_code == 200
    assert response.get_json()["count"] == 2


def test_features_batch_rejects_empty_list(client):
    response = client.post("/api/features/batch", json={"urls": []})
    assert response.status_code == 400


def test_features_batch_rejects_too_many_urls(client):
    urls = [f"https://example{i}.com" for i in range(501)]
    response = client.post("/api/features/batch", json={"urls": urls})
    assert response.status_code == 400


# ------------------------------------------------------------------ model
def test_models_list_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.get_json()["models"]
    assert len(models) == 7


def test_models_status_endpoint(client):
    response = client.get("/api/models/status")
    assert response.status_code == 200
    assert "ready" in response.get_json()


@pytest.mark.skipif(not MODELS_TRAINED, reason="models not trained")
def test_models_comparison_endpoint_when_trained(client):
    response = client.get("/api/models/comparison")
    assert response.status_code == 200
    assert "comparison" in response.get_json()


# ------------------------------------------------------------- prediction
@pytest.mark.skipif(not MODELS_TRAINED, reason="models not trained")
def test_predict_endpoint_returns_verdict(client):
    response = client.post(
        "/api/predict", json={"url": "https://example.com", "store": False}
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] in {"phishing", "legitimate"}
    assert "contributing_factors" in payload
    assert "indicators" in payload


@pytest.mark.skipif(not MODELS_TRAINED, reason="models not trained")
def test_predict_endpoint_with_explicit_model(client):
    response = client.post(
        "/api/predict",
        json={"url": "https://example.com", "model": "random_forest", "store": False},
    )
    assert response.status_code == 200
    assert response.get_json()["model"] == "random_forest"


def test_predict_endpoint_rejects_bad_url(client):
    response = client.post("/api/predict", json={"url": "not a url"})
    assert response.status_code == 400


def test_predict_endpoint_rejects_unknown_model(client):
    response = client.post(
        "/api/predict", json={"url": "https://example.com", "model": "not_a_model"}
    )
    assert response.status_code == 400


@pytest.mark.skipif(not MODELS_TRAINED, reason="models not trained")
def test_predict_compare_endpoint(client):
    response = client.post(
        "/api/predict/compare", json={"url": "https://example.com", "store": False}
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "consensus" in payload
    assert len(payload["model_results"]) >= 1


def test_predictions_history_endpoint(client):
    response = client.get("/api/predictions?limit=5")
    assert response.status_code == 200
    payload = response.get_json()
    assert "results" in payload
    assert "summary" in payload


# ----------------------------------------------------------------- report
@pytest.mark.skipif(not MODELS_TRAINED, reason="models not trained")
def test_model_comparison_report_endpoint(client):
    response = client.get("/api/reports/model-comparison")
    assert response.status_code == 200
    assert response.get_json()["available"] is True
