"""Build the URL dataset used by the experiments.

The project ships a reproducible, synthetically generated dataset so the
whole pipeline runs out of the box without downloading anything.

The generator is class conditional rather than rule based: both classes
draw from the *same* pools of hostnames, paths, schemes and tokens, and
only the probabilities differ. Legitimate URLs therefore sometimes carry
words such as "login", long session identifiers or an uncommon top level
domain, while a share of phishing URLs look completely ordinary. The
classes overlap on purpose, so the measured scores stay in a realistic
range instead of being an artefact of a perfectly separable toy set.

Replace ``data/raw/urls.csv`` with a real corpus (PhishTank, OpenPhish,
the UCI phishing dataset, Kaggle malicious URL collections) whenever one
is available. The rest of the pipeline only expects two columns,
``url`` and ``label``.

Usage::

    python scripts/build_dataset.py
    python scripts/build_dataset.py --size 20000 --holdout-size 4000
"""

from __future__ import annotations

import argparse
import random
import string
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (  # noqa: E402
    EXTERNAL_DATASET,
    RANDOM_STATE,
    RAW_DATASET,
    ensure_directories,
)

# ---------------------------------------------------------------- vocabulary
POPULAR_DOMAINS = [
    "google.com", "youtube.com", "wikipedia.org", "amazon.com", "github.com",
    "stackoverflow.com", "linkedin.com", "reddit.com", "microsoft.com",
    "apple.com", "netflix.com", "nytimes.com", "bbc.co.uk", "theguardian.com",
    "cloudflare.com", "mozilla.org", "python.org", "docker.com", "gitlab.com",
    "atlassian.com", "slack.com", "zoom.us", "dropbox.com", "adobe.com",
    "oracle.com", "ibm.com", "intel.com", "nvidia.com", "samsung.com",
    "spotify.com", "twitch.tv", "medium.com", "quora.com", "coursera.org",
    "edx.org", "udemy.com", "mit.edu", "stanford.edu", "ox.ac.uk",
    "iitb.ac.in", "nic.in", "irctc.co.in", "flipkart.com", "zomato.com",
    "paytm.com", "hdfcbank.com", "icicibank.com", "sbi.co.in", "who.int",
    "nasa.gov", "europa.eu", "gov.uk", "springer.com", "ieee.org", "arxiv.org",
    "nature.com", "kaggle.com", "huggingface.co", "salesforce.com",
    "shopify.com", "booking.com", "airbnb.com", "tripadvisor.com", "uber.com",
    "makemytrip.com", "espn.com", "cricbuzz.com", "cnn.com", "reuters.com",
    "bloomberg.com", "ndtv.com", "thehindu.com", "indianexpress.com",
]

# Ordinary English words used to build small-business style domains. Both
# classes draw from this pool, which is where most of the overlap comes from.
DOMAIN_WORDS = [
    "green", "valley", "north", "bridge", "silver", "stone", "bright", "wave",
    "urban", "meadow", "pine", "harbor", "summit", "clear", "swift", "prime",
    "aster", "maple", "cedar", "orbit", "delta", "vertex", "nimbus", "copper",
    "crest", "lumen", "quartz", "ridge", "atlas", "fable", "coral", "onyx",
    "media", "works", "group", "labs", "store", "shop", "cloud", "systems",
    "digital", "solutions", "studio", "market", "tech", "care", "health",
]

PATH_WORDS = [
    "about", "products", "services", "blog", "news", "docs", "documentation",
    "support", "help", "contact", "pricing", "features", "careers", "team",
    "search", "articles", "category", "tags", "archive", "download",
    "community", "forum", "events", "courses", "tutorials", "guide",
    "reference", "api", "developers", "status", "legal", "privacy", "terms",
    "index", "home", "portal", "app", "media", "images", "posts", "page",
]

# Words tied to a sensitive action. Legitimate sites use them too, which is
# exactly why the model must weigh them instead of treating them as proof.
ACTION_WORDS = [
    "login", "signin", "verify", "verification", "secure", "security",
    "account", "update", "confirm", "recovery", "unlock", "suspended",
    "billing", "payment", "invoice", "alert", "support", "auth", "password",
]

BRANDS = [
    "paypal", "apple", "google", "microsoft", "amazon", "netflix", "facebook",
    "instagram", "whatsapp", "linkedin", "dropbox", "outlook", "office365",
    "icloud", "chase", "hsbc", "barclays", "wellsfargo", "citibank", "sbi",
    "icici", "hdfc", "axisbank", "paytm", "binance", "coinbase", "steam",
    "dhl", "fedex", "usps", "irs", "spotify", "adobe",
]

SUB_LABELS = [
    "www", "secure", "login", "account", "my", "id", "auth", "cdn", "docs",
    "blog", "support", "api", "shop", "mail", "web", "portal",
]

UNCOMMON_TLDS = [
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "click", "link",
    "work", "loan", "men", "date", "icu", "cfd", "rest",
]

COMMON_TLDS = ["com", "net", "org", "info", "site", "online", "co", "io", "in"]

FILE_EXTENSIONS = [".php", ".html", ".htm", ".asp", ".aspx", ".jsp", ""]

# ----------------------------------------------------------------- profiles
# Every probability below is deliberately non-zero for both classes, except
# the two patterns (raw IP host, credentials before the host) that genuine
# public web sites effectively never use.
LEGITIMATE_PROFILE: Dict[str, float] = {
    "popular_domain": 0.45,
    "word_domain": 0.44,
    "random_domain": 0.09,
    "ip_host": 0.00,
    "brand_lookalike": 0.02,
    "https": 0.90,
    "uncommon_tld": 0.06,
    "subdomain_extra": 0.22,
    "action_word_host": 0.10,
    "action_word_path": 0.22,
    "random_path_token": 0.20,
    "long_query": 0.20,
    "hyphen_domain": 0.18,
    "digits_in_domain": 0.07,
    "file_extension": 0.12,
    "port": 0.01,
    "at_symbol": 0.00,
    "deep_path": 0.20,
}

PHISHING_PROFILE: Dict[str, float] = {
    "popular_domain": 0.02,
    "word_domain": 0.34,
    "random_domain": 0.34,
    "ip_host": 0.09,
    "brand_lookalike": 0.21,
    "https": 0.42,
    "uncommon_tld": 0.40,
    "subdomain_extra": 0.46,
    "action_word_host": 0.48,
    "action_word_path": 0.62,
    "random_path_token": 0.55,
    "long_query": 0.45,
    "hyphen_domain": 0.46,
    "digits_in_domain": 0.30,
    "file_extension": 0.40,
    "port": 0.05,
    "at_symbol": 0.03,
    "deep_path": 0.45,
}


def _token(rng: random.Random, low: int = 6, high: int = 14) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(low, high)))


def _tld(rng: random.Random, profile: Dict[str, float]) -> str:
    if rng.random() < profile["uncommon_tld"]:
        return rng.choice(UNCOMMON_TLDS)
    return rng.choice(COMMON_TLDS)


def _word_domain(rng: random.Random, profile: Dict[str, float]) -> str:
    parts = [rng.choice(DOMAIN_WORDS) for _ in range(rng.randint(1, 3))]
    separator = "-" if rng.random() < profile["hyphen_domain"] else ""
    label = separator.join(parts)
    if rng.random() < profile["digits_in_domain"]:
        label += str(rng.randint(1, 999))
    return f"{label}.{_tld(rng, profile)}"


def _brand_host(rng: random.Random, profile: Dict[str, float]) -> str:
    brand = rng.choice(BRANDS)
    tld = _tld(rng, profile)
    style = rng.random()
    if style < 0.35:
        return f"{brand}-{rng.choice(ACTION_WORDS)}.{tld}"
    if style < 0.60:
        return f"{brand}{rng.randint(1, 9999)}.{tld}"
    if style < 0.80:
        return f"{brand}.{rng.choice(DOMAIN_WORDS)}.{tld}"
    return f"{brand}{rng.choice(['s', 'hq', 'web', 'inc', 'care'])}.{tld}"


def _host(rng: random.Random, profile: Dict[str, float]) -> str:
    draw = rng.random()
    cumulative = profile["popular_domain"]
    if draw < cumulative:
        host = rng.choice(POPULAR_DOMAINS)
    elif draw < (cumulative := cumulative + profile["brand_lookalike"]):
        host = _brand_host(rng, profile)
    elif draw < cumulative + profile["ip_host"]:
        host = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        if rng.random() < 0.4:
            host += f":{rng.choice([8080, 8000, 8888, 9090])}"
        return host
    elif draw < cumulative + profile["ip_host"] + profile["random_domain"]:
        host = f"{_token(rng, 7, 16)}.{_tld(rng, profile)}"
    else:
        host = _word_domain(rng, profile)

    labels = []
    if rng.random() < profile["action_word_host"]:
        labels.append(rng.choice(ACTION_WORDS))
    if rng.random() < profile["subdomain_extra"]:
        labels.append(rng.choice(SUB_LABELS))
    if rng.random() < profile["subdomain_extra"] * 0.4:
        labels.append(rng.choice(SUB_LABELS + ACTION_WORDS))
    if labels:
        host = ".".join(labels) + "." + host
    if rng.random() < profile["port"] and ":" not in host:
        host += f":{rng.choice([8080, 8000, 8443, 3000])}"
    return host


def _path(rng: random.Random, profile: Dict[str, float]) -> str:
    deep = rng.random() < profile["deep_path"]
    depth = rng.randint(2, 5) if deep else rng.randint(0, 2)
    segments = []
    for _ in range(depth):
        if rng.random() < profile["action_word_path"]:
            segments.append(rng.choice(ACTION_WORDS))
        elif rng.random() < profile["random_path_token"]:
            segments.append(_token(rng, 6, 18))
        else:
            segments.append(rng.choice(PATH_WORDS))
    if not segments:
        return ""
    path = "/" + "/".join(segments)
    if rng.random() < profile["file_extension"]:
        path += rng.choice(FILE_EXTENSIONS)
    return path


def _query(rng: random.Random, profile: Dict[str, float]) -> str:
    if rng.random() >= profile["long_query"]:
        return f"?q={_token(rng, 3, 8)}" if rng.random() < 0.15 else ""
    pairs = [
        f"session={_token(rng, 10, 26)}",
        f"id={rng.randint(1000, 9999999)}",
        f"ref={_token(rng, 5, 12)}",
        f"utm_source={_token(rng, 4, 9)}",
        f"email={_token(rng, 5, 10)}%40mail.com",
        f"redirect={rng.choice(['home', 'index', 'account'])}",
    ]
    rng.shuffle(pairs)
    return "?" + "&".join(pairs[: rng.randint(2, 4)])


def generate_url(rng: random.Random, profile: Dict[str, float]) -> str:
    """Sample one URL from a class conditional profile."""
    scheme = "https" if rng.random() < profile["https"] else "http"
    host = _host(rng, profile)
    url = f"{scheme}://{host}{_path(rng, profile)}{_query(rng, profile)}"
    if rng.random() < profile["at_symbol"]:
        url = f"{scheme}://{rng.choice(BRANDS)}.com@{url.split('://', 1)[1]}"
    return url


def generate_dataset(
    size: int, seed: int, phishing_ratio: float = 0.45
) -> pd.DataFrame:
    """Generate a labelled URL corpus."""
    rng = random.Random(seed)
    phishing_target = int(size * phishing_ratio)

    rows = []
    for _ in range(size - phishing_target):
        rows.append(
            {"url": generate_url(rng, LEGITIMATE_PROFILE), "label": "legitimate"}
        )
    for _ in range(phishing_target):
        rows.append({"url": generate_url(rng, PHISHING_PROFILE), "label": "phishing"})

    frame = pd.DataFrame(rows).drop_duplicates(subset=["url"])
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the URL dataset.")
    parser.add_argument("--size", type=int, default=16000, help="training corpus size")
    parser.add_argument(
        "--holdout-size", type=int, default=4000, help="external holdout size"
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    arguments = parser.parse_args()

    ensure_directories()

    main_set = generate_dataset(arguments.size, arguments.seed)
    main_set.to_csv(RAW_DATASET, index=False)
    print(f"Wrote {len(main_set)} rows to {RAW_DATASET}")

    # The holdout uses a different seed and a different class balance, so
    # Experiment 4 measures generalisation to data never used for fitting.
    holdout = generate_dataset(
        arguments.holdout_size, arguments.seed + 977, phishing_ratio=0.38
    )
    holdout = holdout[~holdout["url"].isin(set(main_set["url"]))]
    holdout.to_csv(EXTERNAL_DATASET, index=False)
    print(f"Wrote {len(holdout)} rows to {EXTERNAL_DATASET}")
    print(main_set["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
