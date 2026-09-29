"""Run the five research experiments described in the project proposal.

Experiment 1 - Algorithm comparison: same dataset, same features, all models.
Experiment 2 - Feature comparison: lexical only vs structural only vs both.
Experiment 3 - Feature selection: all features vs a selected subset.
Experiment 4 - Generalisation: train on the main set, evaluate on a
               held-out set generated with a different seed and class mix.
Experiment 5 - Error analysis: inspect false positives and false negatives.

Also performs the statistical test for the hypotheses in section 19 of the
proposal (H0: no significant difference between algorithms; H1: there is).

Usage::

    python scripts/run_experiments.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.feature_engineering.extractor import FEATURE_NAMES, FEATURE_SUBSETS  # noqa: E402
from app.feature_engineering.feature_selector import (  # noqa: E402
    correlation_with_target,
    random_forest_importance,
    select_k_best,
)
from app.ml.cross_validation import cross_validate_all, friedman_test  # noqa: E402
from app.ml.preprocessing import (  # noqa: E402
    build_feature_frame,
    get_processed_dataset,
    load_raw_dataset,
    split_features_labels,
    train_test_split_data,
)
from app.ml.registry import MODEL_NAMES, get_model_spec  # noqa: E402
from app.ml.train import predict_with_probability  # noqa: E402
from app.ml import evaluate as evaluation  # noqa: E402
from config.config import (  # noqa: E402
    EXPERIMENTS_PATH,
    EXTERNAL_DATASET,
    PROCESSED_DATASET,
    PROCESSED_HOLDOUT,
    RANDOM_STATE,
    RAW_DATASET,
    ensure_directories,
)
from config.logging_config import get_logger  # noqa: E402

LOGGER = get_logger(__name__)


def _fit_and_score(name: str, x_train, y_train, x_test, y_test) -> Dict[str, float]:
    """Fit one model (with its own scaling) and return its test metrics."""
    from sklearn.preprocessing import StandardScaler

    spec = get_model_spec(name)
    model = spec.build()
    if spec.needs_scaling:
        scaler = StandardScaler().fit(x_train)
        model.fit(scaler.transform(x_train), y_train)
        predictions, probabilities = predict_with_probability(
            model, scaler.transform(x_test)
        )
    else:
        model.fit(x_train, y_train)
        predictions, probabilities = predict_with_probability(model, x_test)
    return evaluation.compute_metrics(y_test, predictions, probabilities), model, predictions


# ------------------------------------------------------------ Experiment 1
def experiment_algorithm_comparison(frame: pd.DataFrame) -> Dict:
    """Same dataset, same features, every algorithm - the headline table."""
    features, labels = split_features_labels(frame, FEATURE_NAMES)
    x_train, x_test, y_train, y_test = train_test_split_data(features, labels)

    cross_val = cross_validate_all(features, labels, MODEL_NAMES, folds=5, scoring="f1")
    hypothesis_test = friedman_test(cross_val)

    rows = []
    for name in MODEL_NAMES:
        metrics, _, _ = _fit_and_score(name, x_train, y_train, x_test, y_test)
        spec = get_model_spec(name)
        rows.append(
            {
                "model": name,
                "display_name": spec.display_name,
                "test_f1": metrics["f1_score"],
                "cv_f1_mean": cross_val[name]["mean"],
                "cv_f1_std": cross_val[name]["std"],
            }
        )
    rows.sort(key=lambda row: row["test_f1"], reverse=True)

    return {
        "name": "Experiment 1 - Algorithm comparison",
        "summary": (
            f"{rows[0]['display_name']} scored highest on the held-out test set "
            f"(F1 = {rows[0]['test_f1']:.4f}); "
            f"{rows[-1]['display_name']} scored lowest (F1 = {rows[-1]['test_f1']:.4f})."
        ),
        "results": rows,
        "cross_validation": cross_val,
        "hypothesis_test": hypothesis_test,
    }


# ------------------------------------------------------------ Experiment 2
def experiment_feature_comparison(frame: pd.DataFrame) -> Dict:
    """Lexical only vs structural only vs lexical+structural vs all features."""
    labels = frame["label"].astype(int).to_numpy()
    results = []
    for subset_name, columns in FEATURE_SUBSETS.items():
        features = frame[columns].astype(float).to_numpy()
        x_train, x_test, y_train, y_test = train_test_split_data(features, labels)
        metrics, _, _ = _fit_and_score(
            "random_forest", x_train, y_train, x_test, y_test
        )
        results.append(
            {
                "feature_subset": subset_name,
                "num_features": len(columns),
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    results.sort(key=lambda row: row["f1_score"], reverse=True)
    best = results[0]
    return {
        "name": "Experiment 2 - Feature group comparison",
        "summary": (
            f"'{best['feature_subset']}' ({best['num_features']} features) gave the "
            f"best Random Forest F1 ({best['f1_score']:.4f}) among the feature subsets tested."
        ),
        "results": results,
    }


# ------------------------------------------------------------ Experiment 3
def experiment_feature_selection(frame: pd.DataFrame, k: int = 25) -> Dict:
    """All features vs a mutual-information selected subset of size k."""
    features, labels = split_features_labels(frame, FEATURE_NAMES)
    x_train, x_test, y_train, y_test = train_test_split_data(features, labels)

    full_metrics, _, _ = _fit_and_score(
        "random_forest", x_train, y_train, x_test, y_test
    )

    selected_names = select_k_best(x_train, y_train, FEATURE_NAMES, k=k)
    selected_index = [FEATURE_NAMES.index(name) for name in selected_names]
    selected_metrics, _, _ = _fit_and_score(
        "random_forest",
        x_train[:, selected_index],
        y_train,
        x_test[:, selected_index],
        y_test,
    )

    correlation = correlation_with_target(frame, labels, FEATURE_NAMES)
    rf_importance = random_forest_importance(x_train, y_train, FEATURE_NAMES)

    return {
        "name": "Experiment 3 - Feature selection",
        "summary": (
            f"All {len(FEATURE_NAMES)} features scored F1 = {full_metrics['f1_score']:.4f}; "
            f"the top {k} selected features scored F1 = {selected_metrics['f1_score']:.4f} "
            f"({'no loss' if selected_metrics['f1_score'] >= full_metrics['f1_score'] - 0.01 else 'a small loss'})."
        ),
        "all_features": {"count": len(FEATURE_NAMES), **full_metrics},
        "selected_features": {
            "count": k,
            "features": selected_names,
            **selected_metrics,
        },
        "top_correlated_features": dict(list(correlation.items())[:15]),
        "top_random_forest_features": dict(list(rf_importance.items())[:15]),
    }


# ------------------------------------------------------------ Experiment 4
def experiment_generalisation(
    train_frame: pd.DataFrame, holdout_path: Path = EXTERNAL_DATASET
) -> Dict:
    """Train on the main set, evaluate on an independently generated holdout."""
    if not Path(holdout_path).exists():
        return {
            "name": "Experiment 4 - Generalisation",
            "summary": "Holdout dataset not found; run scripts/build_dataset.py first.",
            "results": [],
        }

    holdout_raw = load_raw_dataset(holdout_path)
    holdout_frame = build_feature_frame(holdout_raw, save_path=PROCESSED_HOLDOUT)

    x_train, y_train = split_features_labels(train_frame, FEATURE_NAMES)
    x_holdout, y_holdout = split_features_labels(holdout_frame, FEATURE_NAMES)

    rows = []
    for name in MODEL_NAMES:
        from sklearn.preprocessing import StandardScaler

        spec = get_model_spec(name)
        model = spec.build()
        if spec.needs_scaling:
            scaler = StandardScaler().fit(x_train)
            model.fit(scaler.transform(x_train), y_train)
            predictions, probabilities = predict_with_probability(
                model, scaler.transform(x_holdout)
            )
        else:
            model.fit(x_train, y_train)
            predictions, probabilities = predict_with_probability(model, x_holdout)
        metrics = evaluation.compute_metrics(y_holdout, predictions, probabilities)
        rows.append({"model": name, "display_name": spec.display_name, **metrics})

    rows.sort(key=lambda row: row["f1_score"], reverse=True)
    best = rows[0]
    return {
        "name": "Experiment 4 - Generalisation to unseen data",
        "summary": (
            f"On {len(holdout_frame)} independently generated holdout URLs, "
            f"{best['display_name']} generalised best (F1 = {best['f1_score']:.4f})."
        ),
        "holdout_size": len(holdout_frame),
        "results": rows,
    }


# ------------------------------------------------------------ Experiment 5
def experiment_error_analysis(frame: pd.DataFrame, model_name: str = "random_forest") -> Dict:
    """Inspect false positives and false negatives for the leading model."""
    features, labels = split_features_labels(frame, FEATURE_NAMES)
    x_train, x_test, y_train, y_test = train_test_split_data(features, labels)
    urls_train, urls_test = train_test_split_data(
        frame["url"].to_numpy().reshape(-1, 1), labels
    )[0:2]

    metrics, model, predictions = _fit_and_score(
        model_name, x_train, y_train, x_test, y_test
    )

    false_positive_mask = (y_test == 0) & (predictions == 1)
    false_negative_mask = (y_test == 1) & (predictions == 0)

    false_positive_urls = urls_test[false_positive_mask].ravel().tolist()[:15]
    false_negative_urls = urls_test[false_negative_mask].ravel().tolist()[:15]

    return {
        "name": "Experiment 5 - Error analysis",
        "summary": (
            f"{model_name} produced {int(false_positive_mask.sum())} false positives and "
            f"{int(false_negative_mask.sum())} false negatives out of {len(y_test)} test URLs "
            f"(FPR = {metrics['false_positive_rate']:.4f}, FNR = {metrics['false_negative_rate']:.4f})."
        ),
        "model": model_name,
        "false_positive_count": int(false_positive_mask.sum()),
        "false_negative_count": int(false_negative_mask.sum()),
        "false_positive_rate": metrics["false_positive_rate"],
        "false_negative_rate": metrics["false_negative_rate"],
        "false_positive_examples": false_positive_urls,
        "false_negative_examples": false_negative_urls,
    }


def main() -> None:
    ensure_directories()
    LOGGER.info("Loading processed dataset")
    frame = get_processed_dataset(RAW_DATASET, PROCESSED_DATASET)

    start = time.perf_counter()
    experiments: List[Dict] = []

    LOGGER.info("Running Experiment 1 - algorithm comparison")
    exp1 = experiment_algorithm_comparison(frame)
    experiments.append(exp1)

    LOGGER.info("Running Experiment 2 - feature comparison")
    experiments.append(experiment_feature_comparison(frame))

    LOGGER.info("Running Experiment 3 - feature selection")
    experiments.append(experiment_feature_selection(frame))

    LOGGER.info("Running Experiment 4 - generalisation")
    experiments.append(experiment_generalisation(frame))

    LOGGER.info("Running Experiment 5 - error analysis")
    experiments.append(experiment_error_analysis(frame))

    payload = {
        "generated_in_seconds": round(time.perf_counter() - start, 2),
        "dataset_size": len(frame),
        "feature_count": len(FEATURE_NAMES),
        "experiments": experiments,
        "hypothesis_test": exp1["hypothesis_test"],
    }

    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_PATH.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    LOGGER.info("Experiments written to %s", EXPERIMENTS_PATH)

    for experiment in experiments:
        print(f"\n{experiment['name']}\n{'-' * len(experiment['name'])}")
        print(experiment["summary"])
    print(f"\nHypothesis test: {exp1['hypothesis_test']['interpretation']}")


if __name__ == "__main__":
    main()
