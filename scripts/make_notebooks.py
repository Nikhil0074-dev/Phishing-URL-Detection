"""Generate the seven project notebooks under notebooks/.

Each notebook is a thin, runnable wrapper around the pipeline modules in
app/. Run once from the project root:

    python scripts/make_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


SETUP = code(
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
    "sys.path.insert(0, str(PROJECT_ROOT))\n"
    "\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "pd.set_option('display.max_columns', 50)"
)


def build_01():
    cells = [
        md("# 01 - Dataset analysis\n\n"
           "Load the raw URL dataset, clean it, and inspect the class distribution."),
        SETUP,
        code(
            "from config.config import RAW_DATASET, EXTERNAL_DATASET\n"
            "from app.ml.preprocessing import load_raw_dataset, clean_dataset, class_distribution\n"
            "\n"
            "raw = load_raw_dataset(RAW_DATASET)\n"
            "print(f'{len(raw)} rows loaded from {RAW_DATASET.name}')\n"
            "raw.head()"
        ),
        code(
            "cleaned = clean_dataset(raw)\n"
            "print(f'{len(cleaned)} rows after cleaning ({len(raw) - len(cleaned)} dropped)')\n"
            "class_distribution(cleaned)"
        ),
        md("## Class balance"),
        code(
            "counts = cleaned['label'].map({0: 'Legitimate', 1: 'Phishing'}).value_counts()\n"
            "counts.plot(kind='bar', color=['#2d7dd2', '#d7263d'])\n"
            "plt.title('Class distribution')\n"
            "plt.ylabel('Number of URLs')\n"
            "plt.show()"
        ),
        md("## URL length by class (a first look at separability)"),
        code(
            "cleaned['length'] = cleaned['url'].str.len()\n"
            "cleaned.groupby('label')['length'].describe()"
        ),
        md("The two classes overlap on simple statistics such as raw length; "
           "see notebook 02 for the full engineered feature set."),
    ]
    return notebook(cells)


def build_02():
    cells = [
        md("# 02 - URL feature extraction\n\n"
           "Run the feature extractor on a handful of URLs and on the full dataset."),
        SETUP,
        code(
            "from app.feature_engineering.extractor import extract_features, FEATURE_NAMES, FEATURE_GROUPS\n"
            "\n"
            "print(f'{len(FEATURE_NAMES)} features across {len(FEATURE_GROUPS)} groups')\n"
            "{group: len(names) for group, names in FEATURE_GROUPS.items()}"
        ),
        code(
            "samples = [\n"
            "    'https://github.com/explore',\n"
            "    'http://secure-login.paypal-verify9.xyz/account/confirm.php',\n"
            "    'http://192.168.1.20:8080/signin/update',\n"
            "]\n"
            "for url in samples:\n"
            "    features = extract_features(url)\n"
            "    print(url)\n"
            "    print('  length=%d entropy=%.2f subdomains=%d suspicious_kw=%d https=%d'\n"
            "          % (features['url_length'], features['url_entropy'],\n"
            "             features['num_subdomains'], features['suspicious_keyword_count'],\n"
            "             features['has_https']))"
        ),
        md("## Extract features for the whole dataset"),
        code(
            "from app.ml.preprocessing import get_processed_dataset\n"
            "\n"
            "frame = get_processed_dataset()\n"
            "print(frame.shape)\n"
            "frame.head()"
        ),
    ]
    return notebook(cells)


def build_03():
    cells = [
        md("# 03 - Feature analysis\n\n"
           "Correlation, mutual information and Random Forest importance over the "
           "extracted feature set."),
        SETUP,
        code(
            "from app.ml.preprocessing import get_processed_dataset, split_features_labels\n"
            "from app.feature_engineering.extractor import FEATURE_NAMES\n"
            "from app.feature_engineering.feature_selector import (\n"
            "    correlation_with_target, mutual_information_scores,\n"
            "    random_forest_importance, highly_correlated_pairs,\n"
            ")\n"
            "\n"
            "frame = get_processed_dataset()\n"
            "features, labels = split_features_labels(frame, FEATURE_NAMES)\n"
            "features.shape, labels.shape"
        ),
        md("## Correlation with the label"),
        code(
            "correlation = correlation_with_target(frame, labels, FEATURE_NAMES)\n"
            "pd.Series(correlation).head(15)"
        ),
        md("## Mutual information"),
        code(
            "mutual_info = mutual_information_scores(features, labels, FEATURE_NAMES)\n"
            "pd.Series(mutual_info).head(15)"
        ),
        md("## Random Forest importance"),
        code(
            "rf_importance = random_forest_importance(features, labels, FEATURE_NAMES)\n"
            "pd.Series(rf_importance).head(15).plot(kind='barh')\n"
            "plt.gca().invert_yaxis()\n"
            "plt.title('Random Forest feature importance (top 15)')\n"
            "plt.show()"
        ),
        md("## Redundant feature pairs (|r| >= 0.95)"),
        code(
            "pairs = highly_correlated_pairs(frame, FEATURE_NAMES, threshold=0.95)\n"
            "pairs[:10]"
        ),
    ]
    return notebook(cells)


def build_04():
    cells = [
        md("# 04 - Model training\n\n"
           "Train all seven models with the shared preprocessing pipeline."),
        SETUP,
        code(
            "from app.ml.train import train_all\n"
            "\n"
            "results = train_all(make_figures=True)\n"
            "list(results)"
        ),
        md("## Comparison table"),
        code(
            "from app.ml.evaluate import results_to_frame\n"
            "\n"
            "results_to_frame(results)"
        ),
    ]
    return notebook(cells)


def build_05():
    cells = [
        md("# 05 - Model comparison\n\n"
           "Load the saved metrics and compare the seven models on every metric, "
           "not just accuracy."),
        SETUP,
        code(
            "from app.ml.evaluate import load_metrics, results_to_frame\n"
            "\n"
            "metrics = load_metrics()\n"
            "if not metrics:\n"
            "    raise SystemExit(\"No metrics found - run notebook 04 or 'python -m app.ml.train' first.\")\n"
            "comparison = results_to_frame(metrics)\n"
            "comparison"
        ),
        md("## Why not accuracy alone"),
        code(
            "ax = comparison.set_index('display_name')[['accuracy', 'f1_score', 'false_positive_rate']].plot(\n"
            "    kind='bar', figsize=(9, 5))\n"
            "ax.set_ylabel('Score')\n"
            "ax.set_title('Accuracy vs F1 vs false positive rate')\n"
            "plt.xticks(rotation=30, ha='right')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("Naive Bayes typically shows the gap clearly: high precision and a very "
           "low false positive rate, but much lower recall, so its accuracy and F1 "
           "disagree noticeably - the reason the project does not rank models on "
           "accuracy alone."),
        md("## Cross validation and the H0 / H1 hypothesis test"),
        code(
            "from app.ml.cross_validation import cross_validate_all, friedman_test\n"
            "from app.ml.preprocessing import get_processed_dataset, split_features_labels\n"
            "from app.feature_engineering.extractor import FEATURE_NAMES\n"
            "from app.ml.registry import MODEL_NAMES\n"
            "\n"
            "frame = get_processed_dataset()\n"
            "features, labels = split_features_labels(frame, FEATURE_NAMES)\n"
            "cv_results = cross_validate_all(features, labels, MODEL_NAMES, folds=5, scoring='f1')\n"
            "friedman_test(cv_results)"
        ),
    ]
    return notebook(cells)


def build_06():
    cells = [
        md("# 06 - Error analysis\n\n"
           "Inspect false positives and false negatives for the leading model."),
        SETUP,
        code(
            "from app.ml.preprocessing import get_processed_dataset, split_features_labels, train_test_split_data\n"
            "from app.feature_engineering.extractor import FEATURE_NAMES\n"
            "from app.ml.train import train_single_model, predict_with_probability\n"
            "from app.ml.preprocessing import fit_scaler\n"
            "\n"
            "frame = get_processed_dataset()\n"
            "features, labels = split_features_labels(frame, FEATURE_NAMES)\n"
            "x_train, x_test, y_train, y_test = train_test_split_data(features, labels)\n"
            "scaler = fit_scaler(x_train, save_path=None)\n"
            "outcome = train_single_model('random_forest', x_train, y_train, x_test, y_test, scaler)\n"
            "outcome['payload']['metrics']"
        ),
        md("## Locate the errors"),
        code(
            "urls_train, urls_test, _, _ = train_test_split_data(\n"
            "    frame['url'].to_numpy().reshape(-1, 1), labels)\n"
            "predictions = outcome['predictions']\n"
            "\n"
            "false_positive = (y_test == 0) & (predictions == 1)\n"
            "false_negative = (y_test == 1) & (predictions == 0)\n"
            "print(f'False positives: {false_positive.sum()}  False negatives: {false_negative.sum()}')"
        ),
        code(
            "print('Sample false positives (legitimate flagged as phishing):')\n"
            "for url in urls_test[false_positive].ravel()[:10]:\n"
            "    print(' ', url)"
        ),
        code(
            "print('Sample false negatives (phishing flagged as legitimate):')\n"
            "for url in urls_test[false_negative].ravel()[:10]:\n"
            "    print(' ', url)"
        ),
    ]
    return notebook(cells)


def build_07():
    cells = [
        md("# 07 - Explainable AI\n\n"
           "Feature importance and per-prediction explanations."),
        SETUP,
        code(
            "from app.ml.predict import get_engine\n"
            "from app.explainability.feature_importance import (\n"
            "    model_feature_importance, explain_prediction, warning_indicators,\n"
            ")\n"
            "from app.feature_engineering.extractor import FEATURE_NAMES, extract_features\n"
            "\n"
            "engine = get_engine()\n"
            "if not engine.is_ready():\n"
            "    raise SystemExit(\"Train the models first: python -m app.ml.train\")\n"
            "model = engine.get_model('random_forest')\n"
            "importances = model_feature_importance(model, FEATURE_NAMES)\n"
            "pd.Series(importances).head(15)"
        ),
        md("## Explain one prediction"),
        code(
            "url = 'http://secure-login.paypal-verify9.xyz/account/confirm.php'\n"
            "result = engine.predict_url(url, 'random_forest')\n"
            "print(f\"{url}\\nPrediction: {result['prediction'].upper()}  Risk: {result['risk_score']:.2%}\")"
        ),
        code(
            "features = result['features']\n"
            "for factor in explain_prediction(features, importances):\n"
            "    print(f\"  {factor['label']:<26} {factor['level']:<8} weight={factor['importance']:.3f}\")"
        ),
        code(
            "print('Warning indicators:')\n"
            "for note in warning_indicators(features):\n"
            "    print('  -', note)"
        ),
        md("## SHAP summary (optional)"),
        code(
            "from app.explainability.shap_analysis import SHAP_AVAILABLE, global_feature_importance\n"
            "from app.ml.preprocessing import get_processed_dataset, split_features_labels\n"
            "\n"
            "if SHAP_AVAILABLE:\n"
            "    frame = get_processed_dataset()\n"
            "    features_matrix, _ = split_features_labels(frame, FEATURE_NAMES)\n"
            "    shap_importance = global_feature_importance(model, features_matrix, FEATURE_NAMES)\n"
            "    print(pd.Series(shap_importance).head(15))\n"
            "else:\n"
            "    print('SHAP is not installed; falling back to the importance shown above.')"
        ),
    ]
    return notebook(cells)


BUILDERS = {
    "01_dataset_analysis.ipynb": build_01,
    "02_url_feature_extraction.ipynb": build_02,
    "03_feature_analysis.ipynb": build_03,
    "04_model_training.ipynb": build_04,
    "05_model_comparison.ipynb": build_05,
    "06_error_analysis.ipynb": build_06,
    "07_explainable_ai.ipynb": build_07,
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in BUILDERS.items():
        path = NOTEBOOK_DIR / filename
        nbf.write(builder(), path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
