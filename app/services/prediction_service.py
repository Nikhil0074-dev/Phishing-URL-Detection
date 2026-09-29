"""Prediction service: classify a URL, explain it and record the result."""

from __future__ import annotations

from typing import Dict, Optional

from app.database import db
from app.explainability.feature_importance import (
    explain_prediction,
    model_feature_importance,
    warning_indicators,
)
from app.feature_engineering.extractor import FEATURE_NAMES
from app.ml.predict import ModelNotTrainedError, get_engine

RISK_BANDS = (
    (0.80, "high"),
    (0.60, "elevated"),
    (0.40, "uncertain"),
    (0.20, "low"),
)


def risk_band(score: float) -> str:
    """Turn a probability into a coarse band used by the interface."""
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "minimal"


def _importances_for(model_name: str) -> Dict[str, float]:
    try:
        model = get_engine().get_model(model_name)
    except (ModelNotTrainedError, KeyError):
        return {}
    return model_feature_importance(model, FEATURE_NAMES)


def analyse_url(
    url: str,
    model_name: Optional[str] = None,
    store: bool = True,
    include_features: bool = False,
) -> Dict[str, object]:
    """Classify one URL and build the explained response."""
    result = get_engine().predict_url(url, model_name)
    features = result.pop("features")

    importances = _importances_for(result["model"])
    result["risk_band"] = risk_band(result["risk_score"])
    result["risk_percent"] = round(result["risk_score"] * 100, 2)
    result["contributing_factors"] = explain_prediction(features, importances)
    result["indicators"] = warning_indicators(features)
    result["explanation_note"] = (
        "The factors listed above contributed to the model's output. "
        "They describe the model's reasoning and do not, on their own, "
        "establish that the URL is malicious."
    )
    if include_features:
        result["features"] = features

    if store:
        try:
            result["prediction_id"] = db.save_prediction(
                url=result["url"],
                prediction=result["prediction"],
                risk_score=result["risk_score"],
                model_name=result["model"],
                features=features,
            )
        except Exception:  # pragma: no cover - persistence must never break inference
            result["prediction_id"] = None
    return result


def compare_models_on_url(url: str, store: bool = False) -> Dict[str, object]:
    """Run every trained model on the same URL and report the vote."""
    result = get_engine().predict_all_models(url)
    features = result.pop("features")
    consensus = result["consensus"]

    importances = _importances_for(result["model_results"][0]["model"])
    result["contributing_factors"] = explain_prediction(features, importances)
    result["indicators"] = warning_indicators(features)
    consensus["risk_band"] = risk_band(consensus["mean_risk_score"])

    for item in result["model_results"]:
        item["risk_percent"] = round(item["risk_score"] * 100, 2)

    if store:
        try:
            db.save_prediction(
                url=result["url"],
                prediction=consensus["prediction"],
                risk_score=consensus["mean_risk_score"],
                model_name="consensus",
                features=features,
            )
        except Exception:  # pragma: no cover
            pass
    return result


def history(limit: int = 50, offset: int = 0):
    return db.get_predictions(limit=limit, offset=offset)


def summary():
    return db.prediction_summary()
