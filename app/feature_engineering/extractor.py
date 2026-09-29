"""Single entry point for URL feature extraction.

``FEATURE_NAMES`` defines the canonical column order. Training, evaluation
and inference all read it from here, so a model can never be fed a feature
vector whose columns are in a different order from the one it learned.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from app.feature_engineering.domain_features import (
    DOMAIN_FEATURE_NAMES,
    extract_domain_features,
)
from app.feature_engineering.entropy_features import (
    ENTROPY_FEATURE_NAMES,
    extract_entropy_features,
)
from app.feature_engineering.lexical_features import (
    LEXICAL_FEATURE_NAMES,
    extract_lexical_features,
)
from app.feature_engineering.structural_features import (
    STRUCTURAL_FEATURE_NAMES,
    extract_structural_features,
)
from app.feature_engineering.url_parser import InvalidURLError, ParsedURL, parse_url

FEATURE_GROUPS: Dict[str, List[str]] = {
    "lexical": LEXICAL_FEATURE_NAMES,
    "structural": STRUCTURAL_FEATURE_NAMES,
    "domain": DOMAIN_FEATURE_NAMES,
    "entropy": ENTROPY_FEATURE_NAMES,
}

FEATURE_NAMES: List[str] = (
    LEXICAL_FEATURE_NAMES
    + STRUCTURAL_FEATURE_NAMES
    + DOMAIN_FEATURE_NAMES
    + ENTROPY_FEATURE_NAMES
)

# Feature subsets used by the feature-ablation experiment.
FEATURE_SUBSETS: Dict[str, List[str]] = {
    "lexical_only": LEXICAL_FEATURE_NAMES,
    "structural_only": STRUCTURAL_FEATURE_NAMES + DOMAIN_FEATURE_NAMES,
    "entropy_only": ENTROPY_FEATURE_NAMES,
    "lexical_structural": LEXICAL_FEATURE_NAMES
    + STRUCTURAL_FEATURE_NAMES
    + DOMAIN_FEATURE_NAMES,
    "all_features": FEATURE_NAMES,
}


def extract_features_from_parsed(parsed: ParsedURL) -> Dict[str, float]:
    """Extract every feature group from an already parsed URL."""
    features: Dict[str, float] = {}
    features.update(extract_lexical_features(parsed))
    features.update(extract_structural_features(parsed))
    features.update(extract_domain_features(parsed))
    features.update(extract_entropy_features(parsed))
    return {name: float(features[name]) for name in FEATURE_NAMES}


def extract_features(url: str) -> Dict[str, float]:
    """Extract every feature from a raw URL string.

    Raises :class:`InvalidURLError` when the URL cannot be parsed.
    """
    return extract_features_from_parsed(parse_url(url))


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert a feature dictionary to a 1 x n array in canonical order."""
    return np.array(
        [[float(features.get(name, 0.0)) for name in FEATURE_NAMES]], dtype=float
    )


def extract_features_dataframe(
    urls: Iterable[str], skip_invalid: bool = True
) -> pd.DataFrame:
    """Extract features for many URLs and return a DataFrame.

    The returned frame carries a ``url`` column plus one column per feature.
    Invalid URLs are dropped when ``skip_invalid`` is true.
    """
    rows: List[Dict[str, float]] = []
    kept: List[str] = []
    for url in urls:
        try:
            features = extract_features(url)
        except InvalidURLError:
            if skip_invalid:
                continue
            raise
        rows.append(features)
        kept.append(url)

    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    frame.insert(0, "url", kept)
    return frame
