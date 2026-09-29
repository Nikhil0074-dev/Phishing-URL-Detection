"""Lexical (character level) URL features.

These features describe the URL purely as a string: how long it is, how many
digits or special characters it contains, and how its tokens are shaped.
None of them look the domain up on the network, so extraction is fast and
completely offline.
"""

from __future__ import annotations

import re
from typing import Dict

from app.feature_engineering.url_parser import ParsedURL

SPECIAL_CHARACTERS = set("-_@?=&%#~+,;:!$*()[]{}|\\^'\"<>")
TOKEN_SPLIT_PATTERN = re.compile(r"[/\-._?=&%#~+,;:@]+")

LEXICAL_FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "fragment_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_letters",
    "num_special_characters",
    "num_at_symbols",
    "num_question_marks",
    "num_equals",
    "num_ampersands",
    "num_percent",
    "num_query_parameters",
    "num_tokens",
    "avg_token_length",
    "longest_token_length",
    "digit_ratio",
    "letter_ratio",
    "special_character_ratio",
    "digit_letter_ratio",
    "vowel_consonant_ratio",
]


def _tokens(parsed: ParsedURL) -> list:
    raw = TOKEN_SPLIT_PATTERN.split(parsed.without_scheme)
    return [token for token in raw if token]


def extract_lexical_features(parsed: ParsedURL) -> Dict[str, float]:
    """Return the lexical feature dictionary for a parsed URL."""
    url = parsed.url
    length = max(len(url), 1)

    digits = sum(character.isdigit() for character in url)
    letters = sum(character.isalpha() for character in url)
    specials = sum(character in SPECIAL_CHARACTERS for character in url)

    lowered = url.lower()
    vowels = sum(lowered.count(vowel) for vowel in "aeiou")
    consonants = max(letters - vowels, 0)

    tokens = _tokens(parsed)
    token_lengths = [len(token) for token in tokens] or [0]

    return {
        "url_length": float(len(url)),
        "hostname_length": float(len(parsed.hostname)),
        "path_length": float(len(parsed.path)),
        "query_length": float(len(parsed.query)),
        "fragment_length": float(len(parsed.fragment)),
        "num_dots": float(url.count(".")),
        "num_hyphens": float(url.count("-")),
        "num_underscores": float(url.count("_")),
        "num_slashes": float(parsed.without_scheme.count("/")),
        "num_digits": float(digits),
        "num_letters": float(letters),
        "num_special_characters": float(specials),
        "num_at_symbols": float(url.count("@")),
        "num_question_marks": float(url.count("?")),
        "num_equals": float(url.count("=")),
        "num_ampersands": float(url.count("&")),
        "num_percent": float(url.count("%")),
        "num_query_parameters": float(len(parsed.query_parameters)),
        "num_tokens": float(len(tokens)),
        "avg_token_length": float(sum(token_lengths) / len(token_lengths)),
        "longest_token_length": float(max(token_lengths)),
        "digit_ratio": digits / length,
        "letter_ratio": letters / length,
        "special_character_ratio": specials / length,
        "digit_letter_ratio": digits / max(letters, 1),
        "vowel_consonant_ratio": vowels / max(consonants, 1),
    }
