"""Gaussian Naive Bayes baseline."""

from sklearn.naive_bayes import GaussianNB

NAME = "naive_bayes"
DISPLAY_NAME = "Naive Bayes"
NEEDS_SCALING = True


def build_model(**overrides):
    params = {"var_smoothing": 1e-9}
    params.update(overrides)
    return GaussianNB(**params)


PARAM_GRID = {"var_smoothing": [1e-9, 1e-8, 1e-7]}
