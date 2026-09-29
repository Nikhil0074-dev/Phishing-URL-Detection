# Methodology

## Pipeline

```
Dataset -> Cleaning -> Feature extraction -> Feature engineering
        -> Feature selection -> Train/test split -> Model training
        -> Model evaluation -> Comparative analysis
```

## Data cleaning

`app/ml/preprocessing.clean_dataset()`: drops missing `url`/`label` values,
strips whitespace, removes duplicate URLs, and encodes labels to
`{0: legitimate, 1: phishing}` via a synonym map (`LABEL_MAP`).

## Feature extraction (62 features, four groups)

| Group | Count | Examples |
|---|---|---|
| Lexical | 26 | length, token counts, character ratios |
| Structural | 16 | HTTPS, IP host, port, subdomain count, path depth |
| Domain / keyword | 14 | TLD category, suspicious keywords, brand tokens |
| Entropy | 6 | Shannon entropy of the URL, host, path and domain label |

Every feature is derived from the URL string alone - no network lookups, no
WHOIS, no page content - so extraction is fast and works offline.

## Feature engineering

Ratios and derived counts (`special_character_ratio`, `digit_letter_ratio`,
`subdomain_ratio`, `vowel_consonant_ratio`, ...) are computed directly inside
each feature module rather than as a separate pass.

## Feature selection

`app/feature_engineering/feature_selector.py` implements four techniques:
Pearson correlation with the label, mutual information, Random Forest
impurity importance, and `SelectKBest`. Experiment 3
(`scripts/run_experiments.py`) compares the full feature set against the
best `k` under mutual information.

## Train/test protocol

An 80/20 stratified split (`RANDOM_STATE = 42`, `TEST_SIZE = 0.2`) is used
for every model, so the comparison in `reports/generated/model_results.json`
isolates the effect of the algorithm rather than the split. 5-fold
stratified cross validation (`app/ml/cross_validation.py`) gives a second,
split-independent estimate and feeds the Friedman test.

## Models

Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM,
XGBoost (falls back to `HistGradientBoostingClassifier` if the `xgboost`
package is unavailable) and an ANN (`MLPClassifier`). Models needing
standardised inputs (Logistic Regression, Naive Bayes, SVM, ANN) are trained
on `StandardScaler`-transformed features; tree/boosting models are trained
on the raw feature matrix.

## Evaluation metrics

Accuracy, precision, recall, F1, specificity, false positive rate, false
negative rate, ROC-AUC, PR-AUC, training time and prediction time
(`app/ml/evaluate.compute_metrics`). Accuracy alone is never used to declare
a winner, in line with the project's stated principle for imbalanced data.

## Statistical significance

A Friedman test over the 5-fold cross-validated F1 scores of all seven
models tests:

- **H0**: no significant difference in detection performance among the
  algorithms.
- **H1**: there is a significant difference.

`app/ml/cross_validation.friedman_test()` rejects H0 at alpha = 0.05 when
`p_value < 0.05`. The result is written to
`reports/generated/experiments.json` by `scripts/run_experiments.py`.

## Explainability

Global importance comes from each model's own `feature_importances_` or
`coef_` (`app/explainability/feature_importance.py`), with SHAP
(`app/explainability/shap_analysis.py`) used when installed for a more
faithful per-prediction attribution. Per-prediction explanations report a
feature's raw value, a LOW/MEDIUM/HIGH level and its global importance
weight - framed explicitly as *factors that contributed to the model's
output*, not as independent proof of malicious intent.
