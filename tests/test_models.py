"""Tests for app.ml.registry and the individual model builders."""

import numpy as np
import pytest

from app.ml.registry import MODEL_NAMES, MODEL_REGISTRY, build_model, get_model_spec
from app.ml.train import predict_with_probability


def _toy_data():
    rng = np.random.default_rng(0)
    x_legit = rng.normal(loc=0.0, scale=1.0, size=(60, 4))
    x_phish = rng.normal(loc=3.0, scale=1.0, size=(60, 4))
    features = np.vstack([x_legit, x_phish])
    labels = np.array([0] * 60 + [1] * 60)
    return features, labels


def test_registry_has_seven_models():
    assert len(MODEL_NAMES) == 7


def test_registry_contains_expected_models():
    expected = {
        "logistic_regression",
        "naive_bayes",
        "decision_tree",
        "random_forest",
        "svm",
        "xgboost",
        "ann",
    }
    assert set(MODEL_NAMES) == expected


def test_get_model_spec_is_case_and_separator_insensitive():
    assert get_model_spec("Random Forest").name == "random_forest"
    assert get_model_spec("random-forest").name == "random_forest"
    assert get_model_spec("RANDOM_FOREST").name == "random_forest"


def test_get_model_spec_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_model_spec("not_a_real_model")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_builds_and_predicts(name):
    features, labels = _toy_data()
    model = build_model(name)
    model.fit(features, labels)
    predictions, probabilities = predict_with_probability(model, features)

    assert predictions.shape == (120,)
    assert set(np.unique(predictions)).issubset({0, 1})
    if probabilities is not None:
        assert probabilities.shape == (120,)
        assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_needs_scaling_flags_are_consistent_with_registry():
    scale_sensitive = {"logistic_regression", "naive_bayes", "svm", "ann"}
    for name, spec in MODEL_REGISTRY.items():
        assert spec.needs_scaling == (name in scale_sensitive)
