"""Decision Tree classifier."""

from sklearn.tree import DecisionTreeClassifier

from config.config import RANDOM_STATE

NAME = "decision_tree"
DISPLAY_NAME = "Decision Tree"
NEEDS_SCALING = False


def build_model(**overrides):
    params = {
        "criterion": "gini",
        "max_depth": 14,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return DecisionTreeClassifier(**params)


PARAM_GRID = {"max_depth": [8, 14, 20, None], "min_samples_leaf": [1, 4, 8]}
