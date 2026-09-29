# Dataset

## Format

Two columns: `url` (string) and `label` (`legitimate`/`phishing`, or
`0`/`1`, or several other accepted synonyms - see `LABEL_MAP` in
`app/ml/preprocessing.py`).

## Generating the bundled dataset

```
python scripts/build_dataset.py --size 16000 --holdout-size 4000
```

Writes `data/raw/urls.csv` (training/evaluation) and
`data/external/urls_holdout.csv` (used only by Experiment 4, generalisation).

The generator is **class conditional**: legitimate and phishing URLs are
built from the same pools of domains, path words, action words and TLDs, but
with different probabilities per class (see `LEGITIMATE_PROFILE` and
`PHISHING_PROFILE` in `scripts/build_dataset.py`). This intentionally
produces overlap between the classes - a share of legitimate URLs contain
words like "login" or "verify", and a share of phishing URLs are short,
HTTPS and keyword free - which keeps the reported metrics realistic.

## Using a real dataset

Replace `data/raw/urls.csv` with any corpus that has `url` and `label`
columns - PhishTank, OpenPhish, the UCI phishing datasets, or a Kaggle
malicious-URL collection all work directly. Then:

```
python -m app.ml.train --rebuild
```

`--rebuild` forces feature re-extraction instead of reusing the cached
`data/processed/processed_urls.csv`.

## Processed dataset

`data/processed/processed_urls.csv` holds the 62 extracted features plus the
`url` and `label` columns. It is what `app/ml/train.py` actually trains on;
delete it (or pass `--rebuild`) after changing `data/raw/urls.csv`.
