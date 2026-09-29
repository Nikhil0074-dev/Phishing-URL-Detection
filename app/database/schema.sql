-- Schema for the Phishing URL Detection application (SQLite dialect).

CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'user',
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS url_predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT    NOT NULL,
    prediction      TEXT    NOT NULL CHECK (prediction IN ('phishing', 'legitimate')),
    risk_score      REAL    NOT NULL,
    model_name      TEXT    NOT NULL,
    user_id         INTEGER,
    prediction_time TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_time ON url_predictions (prediction_time);
CREATE INDEX IF NOT EXISTS idx_predictions_url  ON url_predictions (url);

CREATE TABLE IF NOT EXISTS url_features (
    feature_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id           INTEGER NOT NULL,
    url_length              REAL,
    domain_length           REAL,
    path_length             REAL,
    dot_count               REAL,
    hyphen_count            REAL,
    digit_count             REAL,
    special_character_count REAL,
    subdomain_count         REAL,
    https_flag              REAL,
    entropy                 REAL,
    FOREIGN KEY (prediction_id) REFERENCES url_predictions (prediction_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_features_prediction ON url_features (prediction_id);

CREATE TABLE IF NOT EXISTS model_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT    NOT NULL,
    accuracy        REAL,
    precision_score REAL,
    recall          REAL,
    f1_score        REAL,
    roc_auc         REAL,
    training_time   REAL,
    evaluation_date TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_model_results_name ON model_results (model_name);
