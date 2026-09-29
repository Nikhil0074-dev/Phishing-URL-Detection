"""Tests for app.ml.predict.

These tests use the trained model artifacts under models/. Run
'python -m app.ml.train' first if models/trained is empty.
"""

import pytest

from app.ml.predict import ModelNotTrainedError, PredictionEngine
from app.ml.registry import MODEL_NAMES
from config.config import SCALER_PATH, TRAINED_MODEL_DIR

pytestmark = pytest.mark.skipif(
    not SCALER_PATH.exists() or not any(TRAINED_MODEL_DIR.glob("*.pkl")),
    reason="Models are not trained yet. Run 'python -m app.ml.train' first.",
)


@pytest.fixture(scope="module")
def engine():
    return PredictionEngine()


def test_available_models_returns_known_names(engine):
    available = engine.available_models()
    assert available
    assert set(available).issubset(set(MODEL_NAMES))


def test_is_ready_true_when_models_exist(engine):
    assert engine.is_ready() is True


def test_predict_url_returns_expected_keys(engine):
    result = engine.predict_url("https://example.com", "random_forest")
    for key in ("url", "model", "prediction", "label", "risk_score", "features"):
        assert key in result
    assert result["prediction"] in {"phishing", "legitimate"}
    assert 0.0 <= result["risk_score"] <= 1.0


def test_predict_url_raises_for_unknown_model(engine):
    with pytest.raises(KeyError):
        engine.predict_url("https://example.com", "not_a_model")


def test_predict_url_uses_default_model_when_none_given(engine):
    result = engine.predict_url("https://example.com")
    assert result["model"] in engine.available_models()


def test_predict_all_models_returns_consensus(engine):
    result = engine.predict_all_models("http://secure-login.example-verify.xyz/account")
    assert "consensus" in result
    consensus = result["consensus"]
    assert consensus["prediction"] in {"phishing", "legitimate"}
    assert consensus["total_models"] == len(result["model_results"])
    assert 0 <= consensus["phishing_votes"] <= consensus["total_models"]


def test_prediction_engine_raises_before_training():
    engine = PredictionEngine(model_dir="/tmp/does-not-exist", scaler_path="/tmp/no-scaler.pkl")
    with pytest.raises(ModelNotTrainedError):
        engine.predict_url("https://example.com", "random_forest")


def test_clear_cache_forces_reload(engine):
    engine.get_model("random_forest")
    assert "random_forest" in engine._models
    engine.clear_cache()
    assert engine._models == {}
