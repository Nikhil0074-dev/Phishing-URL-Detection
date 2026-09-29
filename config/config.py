"""Central configuration for the Phishing URL Detection project."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODEL_DIR = BASE_DIR / "models"
TRAINED_MODEL_DIR = MODEL_DIR / "trained"
SCALER_DIR = MODEL_DIR / "scalers"

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
GENERATED_REPORT_DIR = REPORT_DIR / "generated"

RAW_DATASET = RAW_DATA_DIR / "urls.csv"
EXTERNAL_DATASET = EXTERNAL_DATA_DIR / "urls_holdout.csv"
PROCESSED_DATASET = PROCESSED_DATA_DIR / "processed_urls.csv"
PROCESSED_HOLDOUT = PROCESSED_DATA_DIR / "processed_urls_holdout.csv"

SCALER_PATH = SCALER_DIR / "scaler.pkl"
FEATURE_ORDER_PATH = MODEL_DIR / "feature_order.json"
METRICS_PATH = GENERATED_REPORT_DIR / "model_results.json"
EXPERIMENTS_PATH = GENERATED_REPORT_DIR / "experiments.json"

DATABASE_PATH = BASE_DIR / "data" / "phishing.db"
SCHEMA_PATH = BASE_DIR / "app" / "database" / "schema.sql"

# ---------------------------------------------------------------- experiment
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Models that need standardised inputs.
SCALE_SENSITIVE_MODELS = {
    "logistic_regression",
    "svm",
    "naive_bayes",
    "ann",
}

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "random_forest")

# ---------------------------------------------------------------- flask
class FlaskConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key")
    JSON_SORT_KEYS = False
    HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
    PORT = int(os.environ.get("FLASK_PORT", "5000"))
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"


def ensure_directories() -> None:
    """Create every directory the pipeline writes to."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        TRAINED_MODEL_DIR,
        SCALER_DIR,
        FIGURE_DIR,
        GENERATED_REPORT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
