"""Artificial Neural Network (multi layer perceptron)."""

from sklearn.neural_network import MLPClassifier

from config.config import RANDOM_STATE

NAME = "ann"
DISPLAY_NAME = "ANN"
NEEDS_SCALING = True


def build_model(**overrides):
    params = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 128,
        "learning_rate_init": 1e-3,
        "max_iter": 300,
        "early_stopping": True,
        "n_iter_no_change": 12,
        "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return MLPClassifier(**params)


PARAM_GRID = {"hidden_layer_sizes": [(32,), (64, 32), (128, 64)], "alpha": [1e-4, 1e-3]}
