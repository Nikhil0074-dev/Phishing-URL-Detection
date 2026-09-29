"""Stratified k-fold cross validation and the hypothesis test for H0/H1."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.registry import MODEL_NAMES, get_model_spec
from config.config import CV_FOLDS, RANDOM_STATE
from config.logging_config import get_logger

LOGGER = get_logger(__name__)


def build_pipeline(name: str) -> Pipeline:
    """Wrap a model in a pipeline so scaling is fitted inside each fold."""
    spec = get_model_spec(name)
    steps = []
    if spec.needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", spec.build()))
    return Pipeline(steps)


def cross_validate_model(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    folds: int = CV_FOLDS,
    scoring: str = "f1",
) -> Dict[str, object]:
    """Return per fold scores for one model."""
    from sklearn.model_selection import cross_val_score

    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_STATE
    )
    scores = cross_val_score(
        build_pipeline(name), features, labels, cv=splitter, scoring=scoring, n_jobs=1
    )
    spec = get_model_spec(name)
    LOGGER.info(
        "%s cross validation %s: %.4f (+/- %.4f)",
        spec.display_name,
        scoring,
        scores.mean(),
        scores.std(),
    )
    return {
        "model": spec.name,
        "display_name": spec.display_name,
        "scoring": scoring,
        "fold_scores": [float(score) for score in scores],
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def cross_validate_all(
    features: np.ndarray,
    labels: np.ndarray,
    model_names: Sequence[str] | None = None,
    folds: int = CV_FOLDS,
    scoring: str = "f1",
) -> Dict[str, dict]:
    names = list(model_names) if model_names else list(MODEL_NAMES)
    return {
        name: cross_validate_model(name, features, labels, folds, scoring)
        for name in names
    }


def friedman_test(results: Dict[str, dict]) -> Dict[str, object]:
    """Friedman test over the per fold scores of every model.

    The test decides whether H0 (no significant difference between the
    algorithms) can be rejected at the 0.05 level.
    """
    usable = {
        name: payload["fold_scores"]
        for name, payload in results.items()
        if len(payload.get("fold_scores", [])) >= 3
    }
    if len(usable) < 3:
        return {
            "test": "friedman",
            "performed": False,
            "reason": "At least three models with three folds each are required.",
        }

    statistic, p_value = stats.friedmanchisquare(*usable.values())
    return {
        "test": "friedman",
        "performed": True,
        "models": list(usable),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": 0.05,
        "reject_null_hypothesis": bool(p_value < 0.05),
        "interpretation": (
            "H0 is rejected: the algorithms differ significantly."
            if p_value < 0.05
            else "H0 is not rejected: no significant difference was detected."
        ),
    }


def paired_t_test(
    first: Sequence[float], second: Sequence[float]
) -> Dict[str, object]:
    """Paired t-test between the fold scores of two models."""
    statistic, p_value = stats.ttest_rel(list(first), list(second))
    return {
        "test": "paired_t_test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
