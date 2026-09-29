"""Model feature importance and human readable prediction explanations."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from app.feature_engineering.extractor import FEATURE_NAMES

# Thresholds used to turn a raw feature value into LOW / MEDIUM / HIGH.
# They are reporting aids only; the model prediction does not depend on them.
LEVEL_THRESHOLDS = {
    "url_length": (45, 80),
    "num_subdomains": (1, 2),
    "suspicious_keyword_count": (1, 2),
    "num_special_characters": (4, 9),
    "url_entropy": (3.9, 4.4),
    "hostname_entropy": (3.3, 3.9),
    "num_digits": (2, 6),
    "path_depth": (2, 4),
    "domain_length": (15, 25),
    "num_hyphens": (1, 3),
}

# Features reported to the user, with the wording used in the interface.
REPORTED_FEATURES = [
    ("url_length", "URL length"),
    ("suspicious_keyword_count", "Suspicious keywords"),
    ("has_https", "HTTPS"),
    ("num_subdomains", "Subdomain count"),
    ("num_special_characters", "Special characters"),
    ("url_entropy", "Character entropy"),
    ("has_ip_host", "IP address host"),
    ("is_suspicious_tld", "Top level domain"),
    ("brand_outside_domain", "Brand name outside domain"),
    ("num_digits", "Digit count"),
]

BINARY_LABELS = {
    "has_https": ("ABSENT", "PRESENT"),
    "has_ip_host": ("NO", "YES"),
    "is_suspicious_tld": ("COMMON", "UNCOMMON"),
    "brand_outside_domain": ("NO", "YES"),
    "has_at_before_host": ("NO", "YES"),
    "is_shortened_url": ("NO", "YES"),
}


def model_feature_importance(
    model, feature_names: Sequence[str] = FEATURE_NAMES
) -> Dict[str, float]:
    """Return normalised importance per feature, sorted high to low.

    Tree models expose ``feature_importances_``; linear models expose
    ``coef_``. Models with neither (for example Naive Bayes) return an
    empty dictionary.
    """
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()

    if values is None or values.size != len(feature_names):
        return {}

    total = values.sum()
    if total > 0:
        values = values / total
    ranked = dict(zip(feature_names, (float(value) for value in values)))
    return dict(sorted(ranked.items(), key=lambda item: item[1], reverse=True))


def value_level(name: str, value: float) -> str:
    """Describe a feature value as LOW, MEDIUM or HIGH (or a binary label)."""
    if name in BINARY_LABELS:
        negative, positive = BINARY_LABELS[name]
        return positive if float(value) >= 0.5 else negative
    low, high = LEVEL_THRESHOLDS.get(name, (1, 3))
    if value >= high:
        return "HIGH"
    if value >= low:
        return "MEDIUM"
    return "LOW"


def explain_prediction(
    features: Dict[str, float],
    importances: Dict[str, float] | None = None,
    top_n: int = 6,
) -> List[Dict[str, object]]:
    """Build the contributing factor list shown next to a prediction.

    Each entry reports the raw feature value, a readable level and the
    model importance of that feature. A factor contributes to the model's
    output; on its own it does not establish that a URL is malicious.
    """
    importances = importances or {}
    factors = []
    for name, label in REPORTED_FEATURES:
        if name not in features:
            continue
        value = float(features[name])
        factors.append(
            {
                "feature": name,
                "label": label,
                "value": round(value, 4),
                "level": value_level(name, value),
                "importance": round(float(importances.get(name, 0.0)), 5),
            }
        )

    factors.sort(
        key=lambda item: (item["importance"], item["level"] in {"HIGH", "YES"}),
        reverse=True,
    )
    return factors[:top_n]


def warning_indicators(features: Dict[str, float]) -> List[str]:
    """Plain language notes about patterns present in the URL."""
    notes: List[str] = []
    if features.get("url_length", 0) >= 80:
        notes.append("Unusually long URL")
    if features.get("num_subdomains", 0) >= 2:
        notes.append("Multiple subdomains")
    if features.get("suspicious_keyword_count", 0) >= 1:
        notes.append("Contains sensitive-action keywords")
    if features.get("has_https", 1) < 0.5:
        notes.append("No HTTPS")
    if features.get("has_ip_host", 0) >= 0.5:
        notes.append("Host is a raw IP address")
    if features.get("url_entropy", 0) >= 4.4:
        notes.append("High character entropy")
    if features.get("brand_outside_domain", 0) >= 0.5:
        notes.append("Brand name appears outside the registered domain")
    if features.get("is_suspicious_tld", 0) >= 0.5:
        notes.append("Uncommon top level domain")
    if features.get("has_at_before_host", 0) >= 0.5:
        notes.append("'@' symbol before the host")
    if features.get("num_special_characters", 0) >= 9:
        notes.append("Unusual special-character pattern")
    return notes
