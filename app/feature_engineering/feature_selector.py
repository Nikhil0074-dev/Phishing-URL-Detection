"""Feature selection utilities used by Module 4 and Experiment 3."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from config.config import RANDOM_STATE


def correlation_with_target(
    frame: pd.DataFrame, target: Sequence[int], feature_names: Sequence[str]
) -> Dict[str, float]:
    """Absolute Pearson correlation between each feature and the label."""
    target_series = pd.Series(list(target), index=frame.index, dtype=float)
    scores: Dict[str, float] = {}
    for name in feature_names:
        column = frame[name].astype(float)
        if column.std() == 0:
            scores[name] = 0.0
            continue
        value = column.corr(target_series)
        scores[name] = 0.0 if pd.isna(value) else abs(float(value))
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def highly_correlated_pairs(
    frame: pd.DataFrame, feature_names: Sequence[str], threshold: float = 0.95
) -> List[tuple]:
    """Return feature pairs whose absolute correlation exceeds ``threshold``."""
    matrix = frame[list(feature_names)].astype(float).corr().abs()
    pairs: List[tuple] = []
    names = list(feature_names)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            value = matrix.loc[first, second]
            if pd.notna(value) and value >= threshold:
                pairs.append((first, second, float(value)))
    return sorted(pairs, key=lambda item: item[2], reverse=True)


def mutual_information_scores(
    features: np.ndarray, target: np.ndarray, feature_names: Sequence[str]
) -> Dict[str, float]:
    """Mutual information between each feature and the label."""
    scores = mutual_info_classif(features, target, random_state=RANDOM_STATE)
    ranked = dict(zip(feature_names, (float(score) for score in scores)))
    return dict(sorted(ranked.items(), key=lambda item: item[1], reverse=True))


def random_forest_importance(
    features: np.ndarray, target: np.ndarray, feature_names: Sequence[str]
) -> Dict[str, float]:
    """Impurity based importance from a small random forest."""
    forest = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    forest.fit(features, target)
    ranked = dict(
        zip(feature_names, (float(value) for value in forest.feature_importances_))
    )
    return dict(sorted(ranked.items(), key=lambda item: item[1], reverse=True))


def select_k_best(
    features: np.ndarray, target: np.ndarray, feature_names: Sequence[str], k: int = 20
) -> List[str]:
    """Return the ``k`` best features according to mutual information."""
    k = min(k, len(feature_names))
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE),
        k=k,
    )
    selector.fit(features, target)
    mask = selector.get_support()
    return [name for name, keep in zip(feature_names, mask) if keep]
