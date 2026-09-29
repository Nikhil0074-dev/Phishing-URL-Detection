"""Tests for app.feature_engineering.url_parser."""

import pytest

from app.feature_engineering.url_parser import (
    InvalidURLError,
    is_valid_url,
    normalise_url,
    parse_url,
)


def test_normalise_adds_scheme():
    assert normalise_url("example.com") == "http://example.com"


def test_normalise_keeps_existing_scheme():
    assert normalise_url("https://example.com") == "https://example.com"


def test_normalise_strips_whitespace_and_quotes():
    assert normalise_url('  "example.com" \n') == "http://example.com"


def test_normalise_rejects_empty_string():
    with pytest.raises(InvalidURLError):
        normalise_url("   ")


def test_normalise_rejects_none():
    with pytest.raises(InvalidURLError):
        normalise_url(None)


def test_parse_basic_url():
    parsed = parse_url("https://www.example.com/path?x=1#frag")
    assert parsed.scheme == "https"
    assert parsed.hostname == "www.example.com"
    assert parsed.path == "/path"
    assert parsed.query == "x=1"
    assert parsed.fragment == "frag"
    assert parsed.query_parameters == [("x", "1")]


def test_parse_rejects_url_without_host():
    with pytest.raises(InvalidURLError):
        parse_url("not a url")


def test_parse_rejects_whitespace_in_host():
    with pytest.raises(InvalidURLError):
        parse_url("http://exa mple.com")


def test_is_valid_url_true_and_false():
    assert is_valid_url("https://google.com") is True
    assert is_valid_url("definitely not a url") is False


def test_is_ip_host_true_for_ipv4():
    parsed = parse_url("http://192.168.1.1/login")
    assert parsed.is_ip_host is True


def test_is_ip_host_false_for_domain():
    parsed = parse_url("http://192-168-1-1.com/login")
    assert parsed.is_ip_host is False


def test_registered_domain_simple():
    parsed = parse_url("https://mail.google.com")
    assert parsed.registered_domain == "google.com"


def test_registered_domain_compound_suffix():
    parsed = parse_url("https://shop.example.co.uk")
    assert parsed.registered_domain == "example.co.uk"


def test_subdomain_labels_excludes_www():
    parsed = parse_url("https://www.example.com")
    assert parsed.subdomain_labels == []


def test_subdomain_labels_detects_multiple():
    parsed = parse_url("https://login.secure.example.com")
    assert parsed.subdomain_labels == ["login", "secure"]


def test_port_is_parsed():
    parsed = parse_url("http://example.com:8080/path")
    assert parsed.port == 8080


def test_url_is_truncated_when_extremely_long():
    long_url = "http://example.com/" + "a" * 3000
    parsed = parse_url(long_url)
    assert len(parsed.url) <= 2048
