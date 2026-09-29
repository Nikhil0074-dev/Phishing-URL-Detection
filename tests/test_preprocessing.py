"""Tests for app.ml.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from app.ml.preprocessing import (
    DatasetError,
    class_distribution,
    clean_dataset,
    encode_labels,
    split_features_labels,
    train_test_split_data,
    validate_dataset,
)


def test_validate_dataset_accepts_case_insensitive_columns():
    frame = pd.DataFrame({"URL": ["https://a.com"], "Label": ["legitimate"]})
    validated = validate_dataset(frame)
    assert list(validated.columns) == ["url", "label"]


def test_validate_dataset_missing_url_column_raises():
    with pytest.raises(DatasetError):
        validate_dataset(pd.DataFrame({"link": ["https://a.com"], "label": ["legitimate"]}))


def test_validate_dataset_missing_label_column_raises():
    with pytest.raises(DatasetError):
        validate_dataset(pd.DataFrame({"url": ["https://a.com"]}))


def test_encode_labels_accepts_known_synonyms():
    encoded = encode_labels(["legitimate", "phishing", "PHISH", "Benign", "1", "0"])
    assert encoded.tolist() == [0, 1, 1, 0, 1, 0]


def test_encode_labels_rejects_unknown_values():
    with pytest.raises(DatasetError):
        encode_labels(["legitimate", "unknown_label"])


def test_clean_dataset_drops_missing_and_duplicates():
    frame = pd.DataFrame(
        {
            "url": ["https://a.com", "https://a.com", None, "https://b.com"],
            "label": ["legitimate", "legitimate", "phishing", "phishing"],
        }
    )
    cleaned = clean_dataset(frame)
    assert len(cleaned) == 2
    assert set(cleaned["url"]) == {"https://a.com", "https://b.com"}


def test_clean_dataset_strips_whitespace():
    frame = pd.DataFrame({"url": ["  https://a.com  "], "label": ["legitimate"]})
    cleaned = clean_dataset(frame)
    assert cleaned.iloc[0]["url"] == "https://a.com"


def test_class_distribution_counts_correctly():
    frame = pd.DataFrame({"url": ["a", "b", "c"], "label": [0, 0, 1]})
    distribution = class_distribution(frame)
    assert distribution == {"legitimate": 2, "phishing": 1, "total": 3}


def test_split_features_labels_shapes():
    frame = pd.DataFrame(
        {"feature_a": [1.0, 2.0, 3.0], "feature_b": [4.0, 5.0, 6.0], "label": [0, 1, 0]}
    )
    features, labels = split_features_labels(frame, ["feature_a", "feature_b"])
    assert features.shape == (3, 2)
    assert labels.tolist() == [0, 1, 0]


def test_split_features_labels_missing_column_raises():
    frame = pd.DataFrame({"feature_a": [1.0], "label": [0]})
    with pytest.raises(DatasetError):
        split_features_labels(frame, ["feature_a", "feature_missing"])


def test_train_test_split_data_is_stratified():
    features = np.arange(200).reshape(100, 2).astype(float)
    labels = np.array([0] * 70 + [1] * 30)
    x_train, x_test, y_train, y_test = train_test_split_data(
        features, labels, test_size=0.2, random_state=0
    )
    assert len(x_train) + len(x_test) == 100
    assert abs(y_test.mean() - labels.mean()) < 0.1
