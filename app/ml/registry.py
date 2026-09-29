"""Registry of every comparable model.

A single dictionary keeps the model key, its display name, its factory and
whether it requires standardised inputs, so the training script, the
evaluation script and the web application all agree on the model list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from app.ml.models import (
    ann,
    decision_tree,
    logistic_regression,
    naive_bayes,
    random_forest,
    svm,
    xgboost_model,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    build: Callable
    needs_scaling: bool
    param_grid: Dict


_MODULES = [
    logistic_regression,
    naive_bayes,
    decision_tree,
    random_forest,
    svm,
    xgboost_model,
    ann,
]

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    module.NAME: ModelSpec(
        name=module.NAME,
        display_name=module.DISPLAY_NAME,
        build=module.build_model,
        needs_scaling=module.NEEDS_SCALING,
        param_grid=module.PARAM_GRID,
    )
    for module in _MODULES
}

MODEL_NAMES: List[str] = list(MODEL_REGISTRY)


def get_model_spec(name: str) -> ModelSpec:
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available models: {', '.join(MODEL_NAMES)}"
        )
    return MODEL_REGISTRY[key]


def build_model(name: str, **overrides):
    return get_model_spec(name).build(**overrides)


def display_name(name: str) -> str:
    try:
        return get_model_spec(name).display_name
    except KeyError:
        return str(name)
