"""SHAP based explanations.

SHAP is an optional dependency. When it is unavailable, or when a model is
not supported by the tree explainer, the functions fall back to the
model's own feature importance so the rest of the system keeps working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from app.explainability.feature_importance import model_feature_importance
from app.feature_engineering.extractor import FEATURE_NAMES
from config.config import FIGURE_DIR
from config.logging_config import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover - depends on the environment
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False


def _positive_class_values(values) -> np.ndarray:
    """Normalise the many shapes SHAP can return into a 2-D array."""
    array = np.asarray(values)
    if array.ndim == 3:
        return array[:, :, -1] if array.shape[-1] > 1 else array[:, :, 0]
    return array


def shap_values_for(model, features: np.ndarray, background: np.ndarray | None = None):
    """Return SHAP values for the positive class, or ``None`` on failure."""
    if not SHAP_AVAILABLE:
        return None
    try:
        explainer = shap.Explainer(model, background if background is not None else features)
        values = explainer(features, check_additivity=False)
        return _positive_class_values(values.values)
    except Exception as error:  # pragma: no cover - explainer support varies
        LOGGER.warning("SHAP explanation unavailable: %s", error)
        return None


def global_feature_importance(
    model,
    features: np.ndarray,
    feature_names: Sequence[str] = FEATURE_NAMES,
    sample_size: int = 300,
) -> Dict[str, float]:
    """Mean absolute SHAP value per feature, with an importance fallback."""
    sample = features[:sample_size]
    values = shap_values_for(model, sample)
    if values is None:
        return model_feature_importance(model, feature_names)

    magnitudes = np.abs(values).mean(axis=0)
    total = magnitudes.sum()
    if total > 0:
        magnitudes = magnitudes / total
    ranked = dict(zip(feature_names, (float(value) for value in magnitudes)))
    return dict(sorted(ranked.items(), key=lambda item: item[1], reverse=True))


def local_explanation(
    model,
    vector: np.ndarray,
    background: np.ndarray,
    feature_names: Sequence[str] = FEATURE_NAMES,
    top_n: int = 8,
) -> Optional[Dict[str, float]]:
    """Signed SHAP contributions for a single prediction."""
    values = shap_values_for(model, vector, background)
    if values is None:
        return None
    contributions = dict(zip(feature_names, (float(value) for value in values[0])))
    ordered = sorted(
        contributions.items(), key=lambda item: abs(item[1]), reverse=True
    )
    return dict(ordered[:top_n])


def save_summary_plot(
    model,
    features: np.ndarray,
    feature_names: Sequence[str] = FEATURE_NAMES,
    path: Path | str = FIGURE_DIR / "shap_summary.png",
    sample_size: int = 300,
) -> Optional[Path]:
    """Write a SHAP summary plot; returns ``None`` when SHAP is unavailable."""
    if not SHAP_AVAILABLE:
        LOGGER.info("SHAP is not installed, skipping the summary plot.")
        return None

    sample = features[:sample_size]
    values = shap_values_for(model, sample)
    if values is None:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        values, sample, feature_names=list(feature_names), show=False, max_display=20
    )
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    LOGGER.info("SHAP summary plot saved to %s", path)
    return path
