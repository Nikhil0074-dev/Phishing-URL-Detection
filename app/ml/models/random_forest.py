"""Random Forest ensemble."""

from sklearn.ensemble import RandomForestClassifier

from config.config import RANDOM_STATE

NAME = "random_forest"
DISPLAY_NAME = "Random Forest"
NEEDS_SCALING = False


def build_model(**overrides):
    params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return RandomForestClassifier(**params)


PARAM_GRID = {"n_estimators": [200, 300, 500], "max_depth": [16, 24, None]}
