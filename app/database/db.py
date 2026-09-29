"""SQLite persistence for predictions, extracted features and model results."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from config.config import DATABASE_PATH, SCHEMA_PATH
from config.logging_config import get_logger

LOGGER = get_logger(__name__)

# Mapping from the canonical feature names to the url_features columns.
FEATURE_COLUMN_MAP = {
    "url_length": "url_length",
    "domain_length": "domain_length",
    "path_length": "path_length",
    "num_dots": "dot_count",
    "num_hyphens": "hyphen_count",
    "num_digits": "digit_count",
    "num_special_characters": "special_character_count",
    "num_subdomains": "subdomain_count",
    "has_https": "https_flag",
    "url_entropy": "entropy",
}


@contextmanager
def get_connection(path: Path | str = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row access by column name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database(
    path: Path | str = DATABASE_PATH, schema_path: Path | str = SCHEMA_PATH
) -> None:
    """Create the tables if they do not exist yet."""
    schema = Path(schema_path).read_text(encoding="utf-8")
    with get_connection(path) as connection:
        connection.executescript(schema)
    LOGGER.info("Database initialised at %s", path)


def save_prediction(
    url: str,
    prediction: str,
    risk_score: float,
    model_name: str,
    features: Optional[Dict[str, float]] = None,
    user_id: Optional[int] = None,
    path: Path | str = DATABASE_PATH,
) -> int:
    """Persist one prediction and its features; returns the prediction id."""
    with get_connection(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO url_predictions (url, prediction, risk_score, model_name, user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (url, prediction, float(risk_score), model_name, user_id),
        )
        prediction_id = int(cursor.lastrowid)

        if features:
            columns = ["prediction_id"]
            values = [prediction_id]
            for feature_name, column in FEATURE_COLUMN_MAP.items():
                if feature_name in features:
                    columns.append(column)
                    values.append(float(features[feature_name]))
            placeholders = ", ".join("?" for _ in values)
            connection.execute(
                f"INSERT INTO url_features ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
    return prediction_id


def get_predictions(
    limit: int = 50, offset: int = 0, path: Path | str = DATABASE_PATH
) -> List[Dict]:
    """Return the most recent predictions."""
    with get_connection(path) as connection:
        rows = connection.execute(
            """
            SELECT prediction_id, url, prediction, risk_score, model_name, prediction_time
            FROM url_predictions
            ORDER BY prediction_id DESC
            LIMIT ? OFFSET ?
            """,
            (int(limit), int(offset)),
        ).fetchall()
    return [dict(row) for row in rows]


def get_prediction_features(
    prediction_id: int, path: Path | str = DATABASE_PATH
) -> Optional[Dict]:
    with get_connection(path) as connection:
        row = connection.execute(
            "SELECT * FROM url_features WHERE prediction_id = ?", (int(prediction_id),)
        ).fetchone()
    return dict(row) if row else None


def prediction_summary(path: Path | str = DATABASE_PATH) -> Dict[str, int]:
    """Counts used by the dashboard."""
    with get_connection(path) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN prediction = 'phishing' THEN 1 ELSE 0 END) AS phishing,
                SUM(CASE WHEN prediction = 'legitimate' THEN 1 ELSE 0 END) AS legitimate
            FROM url_predictions
            """
        ).fetchone()
    return {
        "total": int(row["total"] or 0),
        "phishing": int(row["phishing"] or 0),
        "legitimate": int(row["legitimate"] or 0),
    }


def save_model_results(
    results: Dict[str, dict], path: Path | str = DATABASE_PATH
) -> int:
    """Store a training run's metrics; returns the number of rows written."""
    rows = []
    for name, payload in results.items():
        metrics = payload.get("metrics", payload)
        rows.append(
            (
                payload.get("display_name", name),
                float(metrics.get("accuracy", 0.0)),
                float(metrics.get("precision", 0.0)),
                float(metrics.get("recall", 0.0)),
                float(metrics.get("f1_score", 0.0)),
                float(metrics.get("roc_auc", 0.0) or 0.0),
                float(payload.get("training_time", 0.0)),
            )
        )
    with get_connection(path) as connection:
        connection.executemany(
            """
            INSERT INTO model_results
                (model_name, accuracy, precision_score, recall, f1_score, roc_auc, training_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def get_model_results(path: Path | str = DATABASE_PATH) -> List[Dict]:
    with get_connection(path) as connection:
        rows = connection.execute(
            "SELECT * FROM model_results ORDER BY result_id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def clear_predictions(path: Path | str = DATABASE_PATH) -> None:
    with get_connection(path) as connection:
        connection.execute("DELETE FROM url_features")
        connection.execute("DELETE FROM url_predictions")
