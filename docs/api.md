# API reference

All endpoints return JSON. Errors return `{"error": "..."}` with a 4xx/5xx
status.

## URL

### `POST /api/validate`
Body: `{"url": "..."}`. Returns `{"valid": bool, "url": str, "error": str|null}`.

### `POST /api/features`
Body: `{"url": "..."}`. Returns the 62 extracted features grouped by family,
plus the parsed URL components.

### `POST /api/features/batch`
Body: `{"urls": ["...", "..."]}` (max 500). Returns one feature record per
valid URL.

## Prediction

### `POST /api/predict`
Body: `{"url": "...", "model": "random_forest"?, "store": true?, "include_features": false?}`

`model` defaults to `DEFAULT_MODEL` (see `.env.example`) when omitted.
Returns:

```json
{
  "url": "https://example.com/login",
  "model": "random_forest",
  "model_display_name": "Random Forest",
  "prediction": "legitimate",
  "risk_score": 0.08,
  "risk_percent": 8.0,
  "risk_band": "minimal",
  "contributing_factors": [ { "feature": "...", "label": "...", "value": 0, "level": "LOW", "importance": 0.12 } ],
  "indicators": ["..."],
  "explanation_note": "...",
  "prediction_id": 1
}
```

### `POST /api/predict/compare`
Body: `{"url": "...", "store": false?}`. Runs every trained model and
returns each model's result plus a majority-vote `consensus`.

### `GET /api/predictions?limit=50&offset=0`
Recent prediction history plus a phishing/legitimate summary count.

## Models

### `GET /api/models`
Every registered model with its trained status and, once trained, its
headline metrics.

### `GET /api/models/comparison`
The full comparison table, best F1 first, and the current best model.
503 if no metrics exist yet.

### `GET /api/models/status`
`{"ready": bool, "models": [...]}` - whether at least one trained model and
the scaler are present.

## Reports

### `GET /api/reports/model-comparison`
Structured comparison report (also used by the "Reports" page).

### `GET /api/reports/feature-importance?model=random_forest`
Feature importance for one model (best model if `model` is omitted).

### `GET /api/reports/experiments`
The five research experiments and the hypothesis test, once
`scripts/run_experiments.py` has been run.

## Pages

`/`, `/dashboard`, `/models`, `/analysis`, `/predictions`, `/reports`,
`/health` (JSON status) and `/report-figures/<filename>` (serves PNGs from
`reports/figures/`).
