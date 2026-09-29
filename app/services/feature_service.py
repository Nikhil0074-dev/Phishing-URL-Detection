"""Feature extraction service."""

from __future__ import annotations

from typing import Dict, Iterable, List

from app.feature_engineering.extractor import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    extract_features,
    extract_features_dataframe,
)
from app.feature_engineering.url_parser import parse_url


def features_for_url(url: str) -> Dict[str, object]:
    """Return the full feature vector for one URL, grouped for display."""
    parsed = parse_url(url)
    features = extract_features(parsed.url)
    grouped = {
        group: {name: features[name] for name in names}
        for group, names in FEATURE_GROUPS.items()
    }
    return {
        "url": parsed.url,
        "components": parsed.to_dict(),
        "feature_count": len(FEATURE_NAMES),
        "features": features,
        "feature_groups": grouped,
    }


def features_for_many(urls: Iterable[str]) -> List[Dict[str, float]]:
    frame = extract_features_dataframe(urls)
    return frame.to_dict(orient="records")
