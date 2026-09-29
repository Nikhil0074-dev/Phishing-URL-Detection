# Testing

```
pip install -r requirements.txt --break-system-packages
python -m pytest tests/ -q
```

## Layout

| File | Covers |
|---|---|
| `test_url_parser.py` | URL normalisation, parsing, subdomain/suffix logic |
| `test_features.py` | All four feature groups and the combined extractor |
| `test_preprocessing.py` | Dataset validation, cleaning, label encoding, splitting |
| `test_models.py` | Every model builds and fits on synthetic data |
| `test_prediction.py` | The inference engine, using the trained artifacts |
| `test_api.py` | Every Flask page and API endpoint |

Tests in `test_prediction.py` and parts of `test_api.py` are skipped
automatically when `models/trained/` is empty - run
`python -m app.ml.train` first to exercise them.

## Notes

- `test_models.py` fits each model on a small synthetic two-cluster dataset
  built in the test itself, so it does not depend on `data/raw/urls.csv`.
- The Flask tests use `app.test_client()`; no server needs to be running.
