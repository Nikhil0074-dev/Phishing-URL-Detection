"""URL normalisation and parsing helpers.

Every feature module works on a :class:`ParsedURL`, so a URL is parsed once
per prediction instead of once per feature group.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Dict, List
from urllib.parse import parse_qsl, urlsplit

SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
IPV4_PATTERN = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

# Multi part suffixes that must not be mistaken for a subdomain boundary.
COMPOUND_SUFFIXES = {
    "co.uk", "ac.uk", "gov.uk", "org.uk", "co.in", "net.in", "org.in",
    "ac.in", "gov.in", "co.jp", "ne.jp", "or.jp", "com.au", "net.au",
    "org.au", "com.br", "com.cn", "com.mx", "com.sg", "co.za", "co.kr",
    "com.tr", "com.ar", "co.nz", "com.my", "com.ph", "com.hk", "com.tw",
}


class InvalidURLError(ValueError):
    """Raised when a string cannot be treated as a URL."""


@dataclass
class ParsedURL:
    """A normalised URL together with its structural components."""

    raw: str
    url: str
    scheme: str
    hostname: str
    port: int | None
    path: str
    query: str
    fragment: str
    query_parameters: List[tuple] = field(default_factory=list)

    @property
    def without_scheme(self) -> str:
        return self.url.split("://", 1)[-1]

    @property
    def host_labels(self) -> List[str]:
        return [label for label in self.hostname.split(".") if label]

    @property
    def is_ip_host(self) -> bool:
        if IPV4_PATTERN.match(self.hostname):
            try:
                ipaddress.ip_address(self.hostname)
                return True
            except ValueError:
                return False
        return False

    @property
    def suffix(self) -> str:
        """Return the public suffix using a small built in list."""
        labels = self.host_labels
        if self.is_ip_host or len(labels) < 2:
            return ""
        last_two = ".".join(labels[-2:])
        if last_two in COMPOUND_SUFFIXES and len(labels) >= 3:
            return last_two
        return labels[-1]

    @property
    def registered_domain(self) -> str:
        labels = self.host_labels
        if self.is_ip_host or not labels:
            return self.hostname
        suffix_parts = len(self.suffix.split(".")) if self.suffix else 0
        if len(labels) <= suffix_parts:
            return self.hostname
        return ".".join(labels[-(suffix_parts + 1):])

    @property
    def subdomain_labels(self) -> List[str]:
        labels = self.host_labels
        if self.is_ip_host:
            return []
        registered = self.registered_domain.split(".")
        if len(labels) <= len(registered):
            return []
        subdomains = labels[: len(labels) - len(registered)]
        return [label for label in subdomains if label != "www"]

    def to_dict(self) -> Dict[str, object]:
        return {
            "url": self.url,
            "scheme": self.scheme,
            "hostname": self.hostname,
            "port": self.port,
            "path": self.path,
            "query": self.query,
            "fragment": self.fragment,
            "registered_domain": self.registered_domain,
            "subdomains": self.subdomain_labels,
        }


def normalise_url(url: str) -> str:
    """Strip whitespace and add a scheme when the user omitted one."""
    if url is None:
        raise InvalidURLError("URL is None")
    candidate = str(url).strip().strip('"').strip("'")
    if not candidate:
        raise InvalidURLError("URL is empty")
    if len(candidate) > 2048:
        candidate = candidate[:2048]
    if not SCHEME_PATTERN.match(candidate):
        candidate = "http://" + candidate
    return candidate


def parse_url(url: str) -> ParsedURL:
    """Parse a URL string into a :class:`ParsedURL`.

    Raises :class:`InvalidURLError` when the string has no usable host.
    """
    normalised = normalise_url(url)
    try:
        split = urlsplit(normalised)
    except ValueError as exc:  # pragma: no cover - urlsplit rarely raises
        raise InvalidURLError(str(exc)) from exc

    hostname = (split.hostname or "").lower()
    if not hostname or "." not in hostname and not IPV4_PATTERN.match(hostname):
        raise InvalidURLError(f"URL has no valid host: {url!r}")
    if any(character in hostname for character in " \t\n"):
        raise InvalidURLError(f"Host contains whitespace: {url!r}")

    try:
        port = split.port
    except ValueError:
        port = None

    return ParsedURL(
        raw=str(url),
        url=normalised,
        scheme=split.scheme.lower(),
        hostname=hostname,
        port=port,
        path=split.path,
        query=split.query,
        fragment=split.fragment,
        query_parameters=parse_qsl(split.query, keep_blank_values=True),
    )


def is_valid_url(url: str) -> bool:
    try:
        parse_url(url)
        return True
    except InvalidURLError:
        return False
