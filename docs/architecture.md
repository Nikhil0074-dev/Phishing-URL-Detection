# Architecture

## Two subsystems

**Research / ML pipeline** — `scripts/build_dataset.py`, `app/ml/train.py`,
`scripts/run_experiments.py`. Produces trained models, metrics, figures and
the experiment report under `models/` and `reports/`.

**Web application** — `run.py` -> `app/application.py` (Flask factory) ->
`app/api/*` (blueprints) -> `app/services/*` (business logic) ->
`app/ml/predict.py` (inference) and `app/database/db.py` (SQLite).

## Request flow for a prediction

```
Browser
  -> POST /api/predict {url, model?}
  -> app/api/prediction_routes.py
  -> app/services/prediction_service.analyse_url()
       -> app/ml/predict.PredictionEngine.predict_url()
            -> app/feature_engineering/extractor.extract_features()
            -> loads models/trained/<model>.pkl (+ scaler if needed)
       -> app/explainability/feature_importance (contributing factors)
       -> app/database/db.save_prediction() (history)
  <- JSON: prediction, risk_score, contributing_factors, indicators
```

## Layers

| Layer | Responsibility | Location |
|---|---|---|
| Feature engineering | URL -> 62 numeric features | `app/feature_engineering/` |
| ML | training, evaluation, cross validation, inference | `app/ml/` |
| Explainability | feature importance, SHAP, human readable factors | `app/explainability/` |
| Services | glue between ML/DB and the API | `app/services/` |
| API | REST endpoints | `app/api/` |
| Web | Flask app, HTML templates, static assets | `app/application.py`, `app/templates/`, `app/static/` |
| Persistence | prediction history | `app/database/` |

## Design decisions

- **Feature order is canonical.** `FEATURE_NAMES` in `extractor.py` is the
  single source of truth; training, the scaler and inference all read
  columns in that order, so a mismatch cannot silently corrupt a prediction.
- **Scaling is model specific.** Tree and boosting models train on raw
  features; linear/margin/probabilistic/neural models train on the scaler's
  output. `ModelSpec.needs_scaling` encodes this per model.
- **XGBoost and SHAP are optional.** Both have a fallback (histogram
  gradient boosting; the model's own feature importance) so the project runs
  even in a minimal environment.
- **Synthetic, class-overlapping dataset.** URLs are generated from shared
  vocabulary pools with different *probabilities* per class rather than
  disjoint rule sets, so accuracy stays in a realistic band (~85-91%) instead
  of the near-100% a perfectly separable toy set would produce.
