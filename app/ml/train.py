"""Train and compare every model on an identical protocol.

Run it with::

    python -m app.ml.train
    python -m app.ml.train --models random_forest xgboost --rebuild
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Sequence

import joblib
import numpy as np
import pandas as pd

from app.explainability.feature_importance import model_feature_importance
from app.feature_engineering.extractor import FEATURE_NAMES
from app.ml import evaluate as evaluation
from app.ml.preprocessing import (
    class_distribution,
    get_processed_dataset,
    prepare_training_data,
    save_feature_order,
)
from app.ml.registry import MODEL_NAMES, get_model_spec
from config.config import (
    FIGURE_DIR,
    METRICS_PATH,
    PROCESSED_DATASET,
    RAW_DATASET,
    TRAINED_MODEL_DIR,
    ensure_directories,
)
from config.logging_config import get_logger

LOGGER = get_logger(__name__)


def predict_with_probability(model, features: np.ndarray):
    """Return ``(labels, probabilities)`` for any estimator in the registry."""
    predictions = model.predict(features)
    predictions = np.asarray(predictions).astype(int).ravel()
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))[:, 1]
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
    return predictions, probabilities


def train_single_model(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    save_dir: Path = TRAINED_MODEL_DIR,
) -> Dict:
    """Train one model, evaluate it and persist it to disk."""
    spec = get_model_spec(name)
    model = spec.build()

    train_features = scaler.transform(x_train) if spec.needs_scaling else x_train
    test_features = scaler.transform(x_test) if spec.needs_scaling else x_test

    LOGGER.info("Training %s", spec.display_name)
    start = time.perf_counter()
    model.fit(train_features, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions, probabilities = predict_with_probability(model, test_features)
    prediction_time = time.perf_counter() - start

    metrics = evaluation.compute_metrics(y_test, predictions, probabilities)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{spec.name}.pkl"
    joblib.dump(model, model_path)

    LOGGER.info(
        "%s -> accuracy %.4f | f1 %.4f | fpr %.4f | %.2fs",
        spec.display_name,
        metrics["accuracy"],
        metrics["f1_score"],
        metrics["false_positive_rate"],
        training_time,
    )

    return {
        "model": model,
        "payload": {
            "name": spec.name,
            "display_name": spec.display_name,
            "needs_scaling": spec.needs_scaling,
            "training_time": float(training_time),
            "prediction_time": float(prediction_time),
            "model_path": str(model_path),
            "metrics": metrics,
            **metrics,
        },
        "predictions": predictions,
        "probabilities": probabilities,
    }


def train_all(
    model_names: Sequence[str] | None = None,
    raw_path: Path | str = RAW_DATASET,
    processed_path: Path | str = PROCESSED_DATASET,
    rebuild: bool = False,
    make_figures: bool = True,
) -> Dict[str, dict]:
    """Train every requested model and write metrics plus figures."""
    ensure_directories()
    model_names = list(model_names) if model_names else list(MODEL_NAMES)

    x_train, x_test, y_train, y_test, scaler = prepare_training_data(
        raw_path, processed_path, FEATURE_NAMES, rebuild=rebuild
    )
    save_feature_order(FEATURE_NAMES)
    LOGGER.info(
        "Training on %d samples, testing on %d samples, %d features",
        len(x_train),
        len(x_test),
        x_train.shape[1],
    )

    results: Dict[str, dict] = {}
    curves: Dict[str, tuple] = {}
    trained_models: Dict[str, object] = {}

    for name in model_names:
        outcome = train_single_model(
            name, x_train, y_train, x_test, y_test, scaler
        )
        results[name] = outcome["payload"]
        trained_models[name] = outcome["model"]
        if outcome["probabilities"] is not None:
            curves[outcome["payload"]["display_name"]] = (
                y_test,
                outcome["probabilities"],
            )

    comparison = evaluation.results_to_frame(results)
    evaluation.save_metrics(results, METRICS_PATH)
    comparison.to_csv(METRICS_PATH.parent / "model_comparison.csv", index=False)
    print("\nModel comparison\n")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))

    if make_figures:
        _write_figures(
            results, comparison, curves, trained_models, scaler,
            x_test, y_test, processed_path,
        )
    return results


def _write_figures(
    results, comparison, curves, trained_models, scaler, x_test, y_test, processed_path
) -> None:
    frame = get_processed_dataset(processed_path=processed_path)
    evaluation.plot_class_distribution(frame["label"].tolist())
    evaluation.plot_correlation_matrix(frame, FEATURE_NAMES)
    evaluation.plot_model_comparison(comparison)
    if curves:
        evaluation.plot_roc_curves(curves)
        evaluation.plot_precision_recall_curves(curves)

    best = comparison.iloc[0]["model"]
    spec = get_model_spec(best)
    best_features = scaler.transform(x_test) if spec.needs_scaling else x_test
    predictions, _ = predict_with_probability(trained_models[best], best_features)
    evaluation.plot_confusion_matrix(
        y_test,
        predictions,
        spec.display_name,
        FIGURE_DIR / "confusion_matrix.png",
    )
    print(f"\nClassification report - {spec.display_name}\n")
    print(evaluation.classification_text_report(y_test, predictions))

    importances = model_feature_importance(trained_models[best], FEATURE_NAMES)
    if importances:
        evaluation.plot_feature_importance(
            importances, f"Feature importance - {spec.display_name}"
        )
        pd.DataFrame(
            {"feature": list(importances), "importance": list(importances.values())}
        ).to_csv(METRICS_PATH.parent / "feature_importance.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phishing URL detection models.")
    parser.add_argument("--models", nargs="*", default=None, help="subset of models")
    parser.add_argument("--dataset", default=str(RAW_DATASET), help="raw CSV path")
    parser.add_argument(
        "--rebuild", action="store_true", help="re-extract features from the raw CSV"
    )
    parser.add_argument("--no-figures", action="store_true", help="skip figures")
    arguments = parser.parse_args()

    frame = get_processed_dataset(
        raw_path=arguments.dataset, rebuild=arguments.rebuild
    )
    print("Class distribution:", class_distribution(frame))

    train_all(
        model_names=arguments.models,
        raw_path=arguments.dataset,
        rebuild=arguments.rebuild,
        make_figures=not arguments.no_figures,
    )


if __name__ == "__main__":
    main()
