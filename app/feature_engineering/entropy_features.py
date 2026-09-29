"""Entropy and randomness features.

Algorithmically generated hostnames and random path tokens tend to have a
higher Shannon entropy and longer consonant runs than hand written ones.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict

from app.feature_engineering.url_parser import ParsedURL

CONSONANT_RUN_PATTERN = re.compile(r"[bcdfghjklmnpqrstvwxyz]{2,}")

ENTROPY_FEATURE_NAMES = [
    "url_entropy",
    "hostname_entropy",
    "path_entropy",
    "domain_entropy",
    "longest_consonant_run",
    "unique_character_ratio",
]


def shannon_entropy(text: str) -> float:
    """Return the Shannon entropy, in bits per character, of ``text``."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )


def extract_entropy_features(parsed: ParsedURL) -> Dict[str, float]:
    """Return the entropy feature dictionary for a parsed URL."""
    url = parsed.url
    hostname = parsed.hostname
    domain_label = parsed.registered_domain.split(".")[0]

    runs = CONSONANT_RUN_PATTERN.findall(hostname.lower())
    longest_run = max((len(run) for run in runs), default=0)

    return {
        "url_entropy": shannon_entropy(url),
        "hostname_entropy": shannon_entropy(hostname),
        "path_entropy": shannon_entropy(parsed.path),
        "domain_entropy": shannon_entropy(domain_label),
        "longest_consonant_run": float(longest_run),
        "unique_character_ratio": len(set(url)) / max(len(url), 1),
    }
