"""Structural URL features.

These describe how the URL is assembled: scheme, host composition, port,
path depth and the presence of patterns that legitimate sites rarely use
(an embedded IP address, an "@" before the host, a double slash inside the
path). A structural feature is evidence for the model to weigh, never proof
of malicious intent on its own.
"""

from __future__ import annotations

from typing import Dict

from app.feature_engineering.url_parser import ParsedURL

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "rebrand.ly", "bit.do", "t.ly",
    "tiny.cc", "adf.ly", "s.id", "shorte.st",
}

STRUCTURAL_FEATURE_NAMES = [
    "has_https",
    "has_ip_host",
    "has_port",
    "is_non_standard_port",
    "num_subdomains",
    "path_depth",
    "has_at_before_host",
    "has_double_slash_in_path",
    "has_hyphen_in_domain",
    "has_digit_in_domain",
    "is_shortened_url",
    "has_punycode",
    "has_file_extension",
    "has_query",
    "has_fragment",
    "subdomain_ratio",
]

EXECUTABLE_EXTENSIONS = {
    ".exe", ".zip", ".rar", ".scr", ".apk", ".js", ".jar", ".msi", ".bat",
    ".php", ".html", ".htm", ".asp", ".aspx", ".jsp",
}


def extract_structural_features(parsed: ParsedURL) -> Dict[str, float]:
    """Return the structural feature dictionary for a parsed URL."""
    host_labels = parsed.host_labels
    subdomains = parsed.subdomain_labels
    path = parsed.path
    body = parsed.without_scheme
    host_part = body.split("/", 1)[0]

    path_segments = [segment for segment in path.split("/") if segment]
    last_segment = path_segments[-1].lower() if path_segments else ""
    extension = ""
    if "." in last_segment:
        extension = "." + last_segment.rsplit(".", 1)[-1]

    port = parsed.port
    default_ports = {"http": 80, "https": 443}

    return {
        "has_https": 1.0 if parsed.scheme == "https" else 0.0,
        "has_ip_host": 1.0 if parsed.is_ip_host else 0.0,
        "has_port": 1.0 if port is not None else 0.0,
        "is_non_standard_port": 1.0
        if port is not None and port != default_ports.get(parsed.scheme)
        else 0.0,
        "num_subdomains": float(len(subdomains)),
        "path_depth": float(len(path_segments)),
        "has_at_before_host": 1.0 if "@" in host_part else 0.0,
        "has_double_slash_in_path": 1.0 if "//" in path else 0.0,
        "has_hyphen_in_domain": 1.0 if "-" in parsed.registered_domain else 0.0,
        "has_digit_in_domain": 1.0
        if any(character.isdigit() for character in parsed.registered_domain)
        else 0.0,
        "is_shortened_url": 1.0
        if parsed.registered_domain in SHORTENER_DOMAINS
        else 0.0,
        "has_punycode": 1.0 if "xn--" in parsed.hostname else 0.0,
        "has_file_extension": 1.0 if extension in EXECUTABLE_EXTENSIONS else 0.0,
        "has_query": 1.0 if parsed.query else 0.0,
        "has_fragment": 1.0 if parsed.fragment else 0.0,
        "subdomain_ratio": len(subdomains) / max(len(host_labels), 1),
    }
