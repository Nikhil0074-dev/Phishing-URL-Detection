"""Report generation: comparison tables, feature importance and experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.explainability.feature_importance import model_feature_importance
from app.feature_engineering.extractor import FEATURE_NAMES
from app.ml.evaluate import load_metrics, results_to_frame
from app.ml.predict import ModelNotTrainedError, get_engine
from app.services.model_service import best_model, comparison_table
from config.config import EXPERIMENTS_PATH, GENERATED_REPORT_DIR


def model_comparison_report() -> Dict[str, object]:
    """Structured report of the latest training run."""
    metrics = load_metrics()
    table = comparison_table()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models_evaluated": len(table),
        "best_model": best_model(),
        "comparison": table,
        "available": bool(metrics),
    }


def feature_importance_report(model_name: str | None = None) -> Dict[str, object]:
    """Feature importance for one model, best model by default."""
    engine = get_engine()
    if model_name is None:
        best = best_model()
        model_name = best["model"] if best else None
    if model_name is None:
        return {"available": False, "reason": "No trained models found."}

    try:
        model = engine.get_model(model_name)
    except (ModelNotTrainedError, KeyError) as error:
        return {"available": False, "reason": str(error)}

    importances = model_feature_importance(model, FEATURE_NAMES)
    if not importances:
        return {
            "available": False,
            "model": model_name,
            "reason": "This model does not expose feature importance.",
        }
    return {
        "available": True,
        "model": model_name,
        "feature_importance": importances,
        "top_features": list(importances)[:10],
    }


def experiment_report() -> Dict[str, object]:
    """Load the results written by scripts/run_experiments.py."""
    path = Path(EXPERIMENTS_PATH)
    if not path.exists():
        return {
            "available": False,
            "reason": "Run 'python scripts/run_experiments.py' to generate this report.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["available"] = True
    return payload


def export_comparison_csv(
    path: Path | str = GENERATED_REPORT_DIR / "model_comparison.csv",
) -> Path | None:
    metrics = load_metrics()
    if not metrics:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_to_frame(metrics).to_csv(path, index=False)
    return path


def dashboard_payload() -> Dict[str, object]:
    """Everything the dashboard template needs in one call."""
    table = comparison_table()
    importance = feature_importance_report()
    frame = pd.DataFrame(table) if table else pd.DataFrame()
    top_features: List[Dict[str, float]] = []
    if importance.get("available"):
        top_features = [
            {"feature": name, "importance": value}
            for name, value in list(importance["feature_importance"].items())[:10]
        ]
    return {
        "comparison": table,
        "best_model": best_model(),
        "models_tested": len(table),
        "feature_count": len(FEATURE_NAMES),
        "mean_accuracy": float(frame["accuracy"].mean()) if not frame.empty else None,
        "top_features": top_features,
    }
