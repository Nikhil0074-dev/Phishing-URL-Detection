"""Inference: load the trained models and classify a single URL."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from app.feature_engineering.extractor import (
    FEATURE_NAMES,
    extract_features_from_parsed,
    features_to_vector,
)
from app.feature_engineering.url_parser import parse_url
from app.ml.registry import MODEL_NAMES, get_model_spec
from config.config import DEFAULT_MODEL, SCALER_PATH, TRAINED_MODEL_DIR
from config.logging_config import get_logger

LOGGER = get_logger(__name__)


class ModelNotTrainedError(RuntimeError):
    """Raised when a prediction is requested before training has been run."""


class PredictionEngine:
    """Lazy loading cache around the trained models and the scaler."""

    def __init__(
        self,
        model_dir: Path | str = TRAINED_MODEL_DIR,
        scaler_path: Path | str = SCALER_PATH,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.scaler_path = Path(scaler_path)
        self._models: Dict[str, object] = {}
        self._scaler = None

    # ------------------------------------------------------------- loading
    def available_models(self) -> List[str]:
        if not self.model_dir.exists():
            return []
        return [
            name
            for name in MODEL_NAMES
            if (self.model_dir / f"{name}.pkl").exists()
        ]

    def is_ready(self) -> bool:
        return bool(self.available_models()) and self.scaler_path.exists()

    def get_scaler(self):
        if self._scaler is None:
            if not self.scaler_path.exists():
                raise ModelNotTrainedError(
                    f"Scaler not found at {self.scaler_path}. "
                    "Run 'python -m app.ml.train' first."
                )
            self._scaler = joblib.load(self.scaler_path)
        return self._scaler

    def get_model(self, name: str):
        spec = get_model_spec(name)
        if spec.name not in self._models:
            path = self.model_dir / f"{spec.name}.pkl"
            if not path.exists():
                raise ModelNotTrainedError(
                    f"Model file not found: {path}. Run 'python -m app.ml.train' first."
                )
            self._models[spec.name] = joblib.load(path)
        return self._models[spec.name]

    def clear_cache(self) -> None:
        self._models.clear()
        self._scaler = None

    # ---------------------------------------------------------- prediction
    def _prepare_vector(self, vector: np.ndarray, needs_scaling: bool) -> np.ndarray:
        return self.get_scaler().transform(vector) if needs_scaling else vector

    def predict_vector(self, vector: np.ndarray, model_name: str) -> Dict:
        spec = get_model_spec(model_name)
        model = self.get_model(spec.name)
        prepared = self._prepare_vector(vector, spec.needs_scaling)

        start = time.perf_counter()
        label = int(np.asarray(model.predict(prepared)).ravel()[0])
        if hasattr(model, "predict_proba"):
            probability = float(np.asarray(model.predict_proba(prepared))[0, 1])
        elif hasattr(model, "decision_function"):
            score = float(np.asarray(model.decision_function(prepared)).ravel()[0])
            probability = float(1.0 / (1.0 + np.exp(-score)))
        else:  # pragma: no cover - every registry model supports one of the above
            probability = float(label)
        elapsed = time.perf_counter() - start

        return {
            "model": spec.name,
            "model_display_name": spec.display_name,
            "prediction": "phishing" if label == 1 else "legitimate",
            "label": label,
            "risk_score": round(probability, 4),
            "prediction_time_ms": round(elapsed * 1000, 3),
        }

    def predict_url(self, url: str, model_name: Optional[str] = None) -> Dict:
        """Classify one URL with a single model."""
        parsed = parse_url(url)
        features = extract_features_from_parsed(parsed)
        vector = features_to_vector(features)
        name = model_name or self._default_model()
        result = self.predict_vector(vector, name)
        result["url"] = parsed.url
        result["features"] = features
        return result

    def predict_all_models(self, url: str) -> Dict:
        """Classify one URL with every trained model and take a majority vote."""
        parsed = parse_url(url)
        features = extract_features_from_parsed(parsed)
        vector = features_to_vector(features)

        results = []
        for name in self.available_models():
            try:
                results.append(self.predict_vector(vector, name))
            except Exception as error:  # pragma: no cover - defensive
                LOGGER.warning("Model %s failed to predict: %s", name, error)

        if not results:
            raise ModelNotTrainedError(
                "No trained models found. Run 'python -m app.ml.train' first."
            )

        phishing_votes = sum(1 for item in results if item["label"] == 1)
        mean_risk = float(np.mean([item["risk_score"] for item in results]))
        majority = 1 if phishing_votes * 2 > len(results) else 0

        return {
            "url": parsed.url,
            "features": features,
            "model_results": results,
            "consensus": {
                "prediction": "phishing" if majority == 1 else "legitimate",
                "label": majority,
                "phishing_votes": phishing_votes,
                "total_models": len(results),
                "mean_risk_score": round(mean_risk, 4),
            },
        }

    def _default_model(self) -> str:
        available = self.available_models()
        if not available:
            raise ModelNotTrainedError(
                "No trained models found. Run 'python -m app.ml.train' first."
            )
        return DEFAULT_MODEL if DEFAULT_MODEL in available else available[0]


_ENGINE: PredictionEngine | None = None


def get_engine() -> PredictionEngine:
    """Return the process wide prediction engine."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PredictionEngine()
    return _ENGINE


def predict(url: str, model_name: Optional[str] = None) -> Dict:
    return get_engine().predict_url(url, model_name)


__all__ = [
    "FEATURE_NAMES",
    "ModelNotTrainedError",
    "PredictionEngine",
    "get_engine",
    "predict",
]
