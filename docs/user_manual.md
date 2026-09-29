# User manual

## 1. Install

```
pip install -r requirements.txt --break-system-packages
```

## 2. Build the dataset and train

```
python scripts/build_dataset.py
python -m app.ml.train
python scripts/run_experiments.py   # optional: research experiments
```

Training writes `models/trained/*.pkl`, `models/scalers/scaler.pkl`,
`reports/generated/model_results.json` and six figures under
`reports/figures/`.

## 3. Run the application

```
python run.py
```

Open `http://127.0.0.1:5000`.

## 4. Pages

- **Check a URL** (`/`) - paste a URL, pick a model or leave it on "best
  available", and read the risk score, contributing factors and warning
  indicators. "Run all models" shows every model's vote and the consensus.
- **Dashboard** (`/dashboard`) - headline metrics, the full comparison
  table, top features for the best model, and the training figures.
- **Models** (`/models`) - the model registry, trained/not trained status,
  and a short description of what each algorithm does.
- **Features** (`/analysis`) - the four feature groups, top features, and
  (once generated) the research experiment summaries and hypothesis test.
- **History** (`/predictions`) - every URL checked through the interface.
- **Reports** (`/reports`) - the comparison table and experiment summaries
  in report form.

## 5. API

See `docs/api.md` for the full endpoint reference. Quick example:

```
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/login"}'
```

## 6. Configuration

Copy `.env.example` to `.env` and adjust `SECRET_KEY`, `FLASK_PORT`, and
`DEFAULT_MODEL` as needed. `config/config.py` reads these via
`os.environ`.
