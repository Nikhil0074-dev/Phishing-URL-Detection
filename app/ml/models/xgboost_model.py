"""XGBoost gradient boosting classifier.

XGBoost is an optional dependency. When it is not installed the registry
falls back to scikit-learn's HistGradientBoostingClassifier so that the
whole pipeline still runs end to end.
"""

from config.config import RANDOM_STATE

NAME = "xgboost"
DISPLAY_NAME = "XGBoost"
NEEDS_SCALING = False

try:  # pragma: no cover - depends on the environment
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier


def build_model(**overrides):
    if XGBOOST_AVAILABLE:
        params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
        params.update(overrides)
        return XGBClassifier(**params)

    params = {
        "max_iter": 400,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return HistGradientBoostingClassifier(**params)


PARAM_GRID = {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1, 0.2]}
