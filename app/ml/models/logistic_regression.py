"""Logistic Regression baseline."""

from sklearn.linear_model import LogisticRegression

from config.config import RANDOM_STATE

NAME = "logistic_regression"
DISPLAY_NAME = "Logistic Regression"
NEEDS_SCALING = True


def build_model(**overrides):
    params = {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return LogisticRegression(**params)


PARAM_GRID = {"C": [0.1, 1.0, 10.0]}
