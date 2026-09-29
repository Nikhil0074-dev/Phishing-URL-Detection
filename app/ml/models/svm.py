"""Support Vector Machine classifier."""

from sklearn.svm import SVC

from config.config import RANDOM_STATE

NAME = "svm"
DISPLAY_NAME = "SVM"
NEEDS_SCALING = True


def build_model(**overrides):
    params = {
        "C": 2.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
        "class_weight": "balanced",
        "cache_size": 500,
        "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return SVC(**params)


PARAM_GRID = {"C": [1.0, 2.0, 10.0], "gamma": ["scale", 0.1]}
