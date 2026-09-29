"""Access to trained model metadata and comparison metrics."""

from __future__ import annotations

from typing import Dict, List

from app.ml.evaluate import load_metrics, results_to_frame
from app.ml.predict import get_engine
from app.ml.registry import MODEL_REGISTRY


def list_models() -> List[Dict[str, object]]:
    """Every registered model with its training status."""
    engine = get_engine()
    trained = set(engine.available_models())
    metrics = load_metrics()
    models = []
    for name, spec in MODEL_REGISTRY.items():
        payload = metrics.get(name, {})
        models.append(
            {
                "name": name,
                "display_name": spec.display_name,
                "needs_scaling": spec.needs_scaling,
                "trained": name in trained,
                "accuracy": payload.get("accuracy"),
                "f1_score": payload.get("f1_score"),
                "roc_auc": payload.get("roc_auc"),
                "training_time": payload.get("training_time"),
            }
        )
    return models


def comparison_table() -> List[Dict[str, object]]:
    """The model comparison table, best F1 first."""
    metrics = load_metrics()
    if not metrics:
        return []
    frame = results_to_frame(metrics)
    return frame.to_dict(orient="records")


def best_model() -> Dict[str, object] | None:
    table = comparison_table()
    return table[0] if table else None


def is_ready() -> bool:
    return get_engine().is_ready()
