# Phishing URL Detection

Comparative evaluation of machine learning and deep learning algorithms for
phishing URL detection using lexical and structural URL features.

A URL is classified using 62 features extracted from the string alone (no
network lookups), run through seven trained classifiers, and reported as a
risk score with the features that drove it - alongside a research framework
for comparing the algorithms and features against each other.

## Quick start

```bash
pip install -r requirements.txt --break-system-packages

python scripts/build_dataset.py     # generate the URL dataset
python -m app.ml.train              # train and compare all 7 models
python scripts/run_experiments.py   # optional: the 5 research experiments

python run.py                       # start the web application
```

Open `http://127.0.0.1:5000`.

## What it does

- Extracts 62 lexical, structural, domain/keyword and entropy features from
  any URL (`app/feature_engineering/`)
- Trains and compares Logistic Regression, Naive Bayes, Decision Tree,
  Random Forest, SVM, XGBoost and an ANN on an identical protocol
  (`app/ml/`)
- Reports accuracy, precision, recall, F1, specificity, false positive/
  negative rate, ROC-AUC, PR-AUC, training and prediction time
  (`app/ml/evaluate.py`)
- Explains each prediction with the contributing feature values and their
  model importance, framed as evidence rather than proof
  (`app/explainability/`)
- Runs five research experiments - algorithm comparison, feature-group
  comparison, feature selection, generalisation to unseen data, and error
  analysis - plus a Friedman test for the project's H0/H1 hypothesis
  (`scripts/run_experiments.py`)
- Serves a Flask web application with a URL checker, a model dashboard, a
  feature analysis page, prediction history and a REST API
  (`app/application.py`, `app/api/`, `app/templates/`)

## Results (bundled synthetic dataset, 16,000 URLs)

| Model | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Logistic Regression | 0.910 | 0.905 | 0.896 | 0.900 | 0.081 |
| XGBoost | ~0.90 | ~0.90 | ~0.90 | ~0.90 | ~0.09 |
| SVM | ~0.91 | ~0.90 | ~0.90 | ~0.90 | ~0.08 |
| ANN | ~0.91 | ~0.92 | ~0.88 | ~0.90 | ~0.07 |
| Random Forest | 0.898 | 0.887 | 0.893 | 0.890 | 0.097 |
| Decision Tree | 0.861 | 0.843 | 0.858 | 0.850 | 0.136 |
| Naive Bayes | 0.721 | 0.945 | 0.418 | 0.579 | 0.021 |

Exact numbers vary slightly between runs because the dataset generator and
several models use randomised procedures; run `python -m app.ml.train` to
reproduce the current figures in `reports/generated/model_results.json`.
The dataset is synthetic but **class conditional rather than rule based**
(see `docs/dataset.md`) so these scores sit in a realistic range instead of
the near-100% a perfectly separable toy set would produce - a real corpus
(PhishTank, OpenPhish, UCI) is a drop-in replacement.

## Project layout

```
config/                 Central configuration and logging
app/
  feature_engineering/  URL parsing and the 62-feature extractor
  ml/                    preprocessing, training, evaluation, inference, cross validation
  explainability/       feature importance and SHAP
  database/              SQLite schema and access layer
  services/              business logic used by the API
  api/                    Flask blueprints (REST endpoints)
  templates/, static/    the web interface
data/, models/, reports/  generated datasets, trained models, figures and JSON reports
notebooks/              seven runnable notebooks mirroring the pipeline
tests/                   96 pytest tests across parsing, features, ML, API
docs/                    architecture, methodology, dataset, API, testing, user manual
scripts/                 dataset generation, experiments, notebook generation
run.py                   application entry point
```

See `docs/user_manual.md` for a walkthrough of every page, `docs/api.md` for
the REST reference, and `docs/methodology.md` for the full research
methodology (RQ1-RQ6, H0/H1, the five experiments).

## Testing

```bash
python -m pytest tests/ -q
```

96 tests, all passing. Tests that need trained models are skipped
automatically if `models/trained/` is empty.

## Notes

- XGBoost and SHAP are optional dependencies; the project degrades
  gracefully (histogram gradient boosting, plain feature importance) when
  either is missing.
- A prediction is a statistical estimate from URL features, not a verdict
  about a site's intent - the interface and the explanation text say this
  explicitly, and the underlying feature keyword lists (`login`, `verify`,
  and similar) are documented as signals the model weighs, not proof.
