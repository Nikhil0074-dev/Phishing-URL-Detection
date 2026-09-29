"""Dataset loading, cleaning, feature building, splitting and scaling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.feature_engineering.extractor import (
    FEATURE_NAMES,
    extract_features_dataframe,
)
from config.config import (
    FEATURE_ORDER_PATH,
    PROCESSED_DATASET,
    RANDOM_STATE,
    RAW_DATASET,
    SCALER_PATH,
    TEST_SIZE,
    ensure_directories,
)
from config.logging_config import get_logger

LOGGER = get_logger(__name__)

LABEL_MAP = {
    "legitimate": 0,
    "legit": 0,
    "benign": 0,
    "good": 0,
    "0": 0,
    "phishing": 1,
    "phish": 1,
    "malicious": 1,
    "bad": 1,
    "1": 1,
}


class DatasetError(ValueError):
    """Raised when a dataset cannot be used for training."""


# ------------------------------------------------------------------ loading
def load_raw_dataset(path: Path | str = RAW_DATASET) -> pd.DataFrame:
    """Load a CSV with ``url`` and ``label`` columns."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at {path}. Run 'python scripts/build_dataset.py' first."
        )
    frame = pd.read_csv(path)
    return validate_dataset(frame)


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the required columns exist and normalise their names."""
    columns = {str(column).strip().lower(): column for column in frame.columns}
    if "url" not in columns:
        raise DatasetError("Dataset must contain a 'url' column.")
    if "label" not in columns:
        raise DatasetError("Dataset must contain a 'label' column.")
    frame = frame.rename(columns={columns["url"]: "url", columns["label"]: "label"})
    return frame[["url", "label"]]


def encode_labels(values: Iterable) -> pd.Series:
    """Map textual or numeric labels onto 0 (legitimate) and 1 (phishing)."""
    series = pd.Series(list(values), dtype="object")
    encoded = series.map(
        lambda value: LABEL_MAP.get(str(value).strip().lower(), np.nan)
    )
    if encoded.isna().any():
        unknown = sorted(
            {
                str(value)
                for value, code in zip(series, encoded)
                if pd.isna(code)
            }
        )[:5]
        raise DatasetError(f"Unrecognised label values: {unknown}")
    return encoded.astype(int)


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop missing values and duplicates, then encode the labels."""
    cleaned = frame.copy()
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["url", "label"])
    cleaned["url"] = cleaned["url"].astype(str).str.strip()
    cleaned = cleaned[cleaned["url"] != ""]
    cleaned = cleaned.drop_duplicates(subset=["url"], keep="first")
    cleaned["label"] = encode_labels(cleaned["label"])
    cleaned = cleaned.reset_index(drop=True)
    LOGGER.info("Cleaned dataset: %d rows kept out of %d", len(cleaned), before)
    return cleaned


def class_distribution(frame: pd.DataFrame) -> Dict[str, int]:
    counts = frame["label"].value_counts().to_dict()
    return {
        "legitimate": int(counts.get(0, 0)),
        "phishing": int(counts.get(1, 0)),
        "total": int(len(frame)),
    }


# ------------------------------------------------------- feature building
def build_feature_frame(
    frame: pd.DataFrame, save_path: Path | str | None = PROCESSED_DATASET
) -> pd.DataFrame:
    """Extract features for every URL and attach the label column."""
    cleaned = clean_dataset(frame)
    features = extract_features_dataframe(cleaned["url"].tolist())
    labels = cleaned.set_index("url")["label"]
    features["label"] = features["url"].map(labels).astype(int)
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(save_path, index=False)
        LOGGER.info("Processed dataset written to %s", save_path)
    return features


def load_processed_dataset(path: Path | str = PROCESSED_DATASET) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Processed dataset not found at {path}.")
    return pd.read_csv(path)


def get_processed_dataset(
    raw_path: Path | str = RAW_DATASET,
    processed_path: Path | str = PROCESSED_DATASET,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Return the processed dataset, building it from the raw CSV if needed."""
    processed_path = Path(processed_path)
    if processed_path.exists() and not rebuild:
        frame = pd.read_csv(processed_path)
        if set(FEATURE_NAMES).issubset(frame.columns):
            return frame
        LOGGER.info("Processed dataset is stale, rebuilding it.")
    return build_feature_frame(load_raw_dataset(raw_path), processed_path)


# --------------------------------------------------------------- splitting
def split_features_labels(
    frame: pd.DataFrame, feature_names: Sequence[str] = FEATURE_NAMES
) -> Tuple[np.ndarray, np.ndarray]:
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise DatasetError(f"Missing feature columns: {missing[:5]}")
    features = frame[list(feature_names)].astype(float).to_numpy()
    labels = frame["label"].astype(int).to_numpy()
    return features, labels


def train_test_split_data(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


# ----------------------------------------------------------------- scaling
def fit_scaler(features: np.ndarray, save_path: Path | str | None = SCALER_PATH):
    scaler = StandardScaler()
    scaler.fit(features)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        LOGGER.info("Scaler saved to %s", save_path)
    return scaler


def load_scaler(path: Path | str = SCALER_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found at {path}. Train the models first.")
    return joblib.load(path)


# ---------------------------------------------------------- feature order
def save_feature_order(
    feature_names: Sequence[str] = FEATURE_NAMES,
    path: Path | str = FEATURE_ORDER_PATH,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(feature_names), indent=2), encoding="utf-8")


def load_feature_order(path: Path | str = FEATURE_ORDER_PATH) -> List[str]:
    path = Path(path)
    if not path.exists():
        return list(FEATURE_NAMES)
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_training_data(
    raw_path: Path | str = RAW_DATASET,
    processed_path: Path | str = PROCESSED_DATASET,
    feature_names: Sequence[str] = FEATURE_NAMES,
    rebuild: bool = False,
):
    """Return ``(X_train, X_test, y_train, y_test, scaler)`` ready for training."""
    ensure_directories()
    frame = get_processed_dataset(raw_path, processed_path, rebuild=rebuild)
    features, labels = split_features_labels(frame, feature_names)
    x_train, x_test, y_train, y_test = train_test_split_data(features, labels)
    scaler = fit_scaler(x_train)
    save_feature_order(feature_names)
    return x_train, x_test, y_train, y_test, scaler
