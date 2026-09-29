"""Evaluation metrics, confusion matrices, curves and comparison figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.config import FIGURE_DIR, METRICS_PATH  # noqa: E402
from config.logging_config import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

METRIC_COLUMNS = [
    "model",
    "display_name",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "roc_auc",
    "pr_auc",
    "training_time",
    "prediction_time",
]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """Return the full metric set for one model."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(true_negative / max(true_negative + false_positive, 1)),
        "false_positive_rate": float(
            false_positive / max(false_positive + true_negative, 1)
        ),
        "false_negative_rate": float(
            false_negative / max(false_negative + true_positive, 1)
        ),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def classification_text_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(
        y_true, y_pred, target_names=["Legitimate", "Phishing"], zero_division=0
    )


def results_to_frame(results: Dict[str, dict]) -> pd.DataFrame:
    """Convert the raw results dictionary into a tidy comparison table."""
    rows = []
    for name, payload in results.items():
        metrics = payload.get("metrics", {})
        row = {"model": name, "display_name": payload.get("display_name", name)}
        for column in METRIC_COLUMNS[2:]:
            row[column] = payload.get(column, metrics.get(column, float("nan")))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    return frame.sort_values("f1_score", ascending=False).reset_index(drop=True)


def save_metrics(results: Dict[str, dict], path: Path | str = METRICS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    LOGGER.info("Metrics written to %s", path)


def load_metrics(path: Path | str = METRICS_PATH) -> Dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ plots
def plot_class_distribution(
    labels: Sequence[int], path: Path | str = FIGURE_DIR / "class_distribution.png"
) -> Path:
    series = pd.Series(list(labels)).map({0: "Legitimate", 1: "Phishing"})
    counts = series.value_counts()
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(counts.index, counts.to_numpy(), color=["#2d7dd2", "#d7263d"])
    axis.set_title("Class distribution")
    axis.set_ylabel("Number of URLs")
    for index, value in enumerate(counts.to_numpy()):
        axis.text(index, value, str(int(value)), ha="center", va="bottom")
    return _save(figure, path)


def plot_correlation_matrix(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    path: Path | str = FIGURE_DIR / "correlation_matrix.png",
) -> Path:
    subset = frame[list(feature_names)].astype(float)
    matrix = subset.corr().fillna(0.0)
    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    axis.set_yticks(range(len(matrix.columns)))
    axis.set_yticklabels(matrix.columns, fontsize=6)
    axis.set_title("Feature correlation matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save(figure, path)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path | str
) -> Path:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4.5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Legitimate", "Phishing"])
    axis.set_yticks([0, 1], ["Legitimate", "Phishing"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"Confusion matrix - {title}")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )
    return _save(figure, path)


def plot_roc_curves(
    curves: Dict[str, tuple], path: Path | str = FIGURE_DIR / "roc_curves.png"
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 6))
    for label, (y_true, y_proba) in curves.items():
        if y_proba is None:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_proba)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{label} (AUC = {auc(false_positive_rate, true_positive_rate):.3f})",
        )
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curves")
    axis.legend(fontsize=8, loc="lower right")
    return _save(figure, path)


def plot_precision_recall_curves(
    curves: Dict[str, tuple],
    path: Path | str = FIGURE_DIR / "precision_recall_curves.png",
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 6))
    for label, (y_true, y_proba) in curves.items():
        if y_proba is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        axis.plot(recall, precision, label=label)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curves")
    axis.legend(fontsize=8, loc="lower left")
    return _save(figure, path)


def plot_model_comparison(
    frame: pd.DataFrame, path: Path | str = FIGURE_DIR / "model_comparison.png"
) -> Path:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = frame["display_name"].tolist()
    positions = np.arange(len(labels))
    width = 0.2

    figure, axis = plt.subplots(figsize=(11, 6))
    for index, metric in enumerate(metrics):
        axis.bar(
            positions + index * width,
            frame[metric].to_numpy(),
            width,
            label=metric.replace("_", " "),
        )
    axis.set_xticks(positions + width * 1.5, labels, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Model comparison")
    axis.legend(fontsize=9)
    return _save(figure, path)


def plot_feature_importance(
    importances: Dict[str, float],
    title: str = "Feature importance",
    top_n: int = 20,
    path: Path | str = FIGURE_DIR / "feature_importance.png",
) -> Path:
    items = list(importances.items())[:top_n][::-1]
    names = [name for name, _ in items]
    values = [value for _, value in items]
    figure, axis = plt.subplots(figsize=(9, max(4, 0.35 * len(names))))
    axis.barh(names, values, color="#2d7dd2")
    axis.set_title(title)
    axis.set_xlabel("Importance")
    axis.tick_params(axis="y", labelsize=8)
    return _save(figure, path)


def _save(figure, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    LOGGER.info("Figure saved to %s", path)
    return path
