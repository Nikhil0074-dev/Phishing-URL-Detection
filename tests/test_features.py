"""Tests for app.feature_engineering.extractor and its sub-modules."""

import math

import pytest

from app.feature_engineering.domain_features import extract_domain_features
from app.feature_engineering.entropy_features import (
    extract_entropy_features,
    shannon_entropy,
)
from app.feature_engineering.extractor import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_SUBSETS,
    extract_features,
    extract_features_dataframe,
    features_to_vector,
)
from app.feature_engineering.lexical_features import extract_lexical_features
from app.feature_engineering.structural_features import extract_structural_features
from app.feature_engineering.url_parser import InvalidURLError, parse_url


def test_feature_names_are_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_feature_groups_cover_all_feature_names():
    grouped = sum(FEATURE_GROUPS.values(), [])
    assert sorted(grouped) == sorted(FEATURE_NAMES)


def test_extract_features_returns_every_feature():
    features = extract_features("https://example.com/login")
    assert set(features) == set(FEATURE_NAMES)
    assert all(isinstance(value, float) for value in features.values())


def test_extract_features_invalid_url_raises():
    with pytest.raises(InvalidURLError):
        extract_features("not a url")


def test_features_to_vector_matches_feature_order():
    features = extract_features("https://example.com")
    vector = features_to_vector(features)
    assert vector.shape == (1, len(FEATURE_NAMES))
    assert vector[0, FEATURE_NAMES.index("has_https")] == 1.0


def test_lexical_features_url_length():
    url = "http://example.com/login"
    parsed = parse_url(url)
    features = extract_lexical_features(parsed)
    assert features["url_length"] == float(len(url))


def test_lexical_features_digit_and_letter_ratio():
    parsed = parse_url("http://a1b2c3.com")
    features = extract_lexical_features(parsed)
    assert 0.0 <= features["digit_ratio"] <= 1.0
    assert 0.0 <= features["letter_ratio"] <= 1.0


def test_structural_features_https_flag():
    assert extract_structural_features(parse_url("https://example.com"))["has_https"] == 1.0
    assert extract_structural_features(parse_url("http://example.com"))["has_https"] == 0.0


def test_structural_features_detect_ip_host():
    features = extract_structural_features(parse_url("http://192.168.1.5/login"))
    assert features["has_ip_host"] == 1.0


def test_structural_features_detect_at_symbol():
    features = extract_structural_features(parse_url("http://paypal.com@evil.com/login"))
    assert features["has_at_before_host"] == 1.0


def test_structural_features_shortener_flag():
    features = extract_structural_features(parse_url("http://bit.ly/abc123"))
    assert features["is_shortened_url"] == 1.0


def test_domain_features_keyword_detection():
    features = extract_domain_features(parse_url("http://example.com/account/login"))
    assert features["has_suspicious_keyword"] == 1.0
    assert features["suspicious_keyword_count"] >= 1.0


def test_domain_features_no_keywords():
    features = extract_domain_features(parse_url("http://example.com/about"))
    assert features["has_suspicious_keyword"] == 0.0


def test_domain_features_suspicious_tld():
    features = extract_domain_features(parse_url("http://example.xyz"))
    assert features["is_suspicious_tld"] == 1.0


def test_domain_features_brand_outside_domain():
    features = extract_domain_features(parse_url("http://paypal.login-security.info"))
    assert features["brand_outside_domain"] == 1.0


def test_shannon_entropy_of_empty_string_is_zero():
    assert shannon_entropy("") == 0.0


def test_shannon_entropy_of_repeated_character_is_zero():
    assert shannon_entropy("aaaaaa") == 0.0


def test_shannon_entropy_increases_with_variety():
    assert shannon_entropy("abcdabcd") > shannon_entropy("aabbaabb")


def test_entropy_features_are_non_negative():
    features = extract_entropy_features(parse_url("http://x8j2k9f1.tk/abc"))
    assert features["url_entropy"] >= 0.0
    assert not math.isnan(features["url_entropy"])


def test_feature_subsets_are_non_empty_and_valid():
    for name, columns in FEATURE_SUBSETS.items():
        assert columns, f"subset {name} is empty"
        assert set(columns).issubset(set(FEATURE_NAMES))


def test_extract_features_dataframe_skips_invalid_by_default():
    frame = extract_features_dataframe(["https://example.com", "not a url", "bit.ly/x"])
    assert len(frame) == 2
    assert list(frame.columns[1:]) == FEATURE_NAMES


def test_extract_features_dataframe_raises_when_not_skipping():
    with pytest.raises(InvalidURLError):
        extract_features_dataframe(["not a url"], skip_invalid=False)
