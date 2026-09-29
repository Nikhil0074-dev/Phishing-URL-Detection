"""URL validation and inspection helpers used by the API layer."""

from __future__ import annotations

from typing import Dict

from app.feature_engineering.url_parser import InvalidURLError, parse_url

MAX_URL_LENGTH = 2048


def validate_url(url: str) -> Dict[str, object]:
    """Validate a user supplied URL.

    Returns ``{"valid": bool, "url": str, "error": str | None}`` instead of
    raising, because the API turns this straight into a JSON response.
    """
    if url is None or not str(url).strip():
        return {"valid": False, "url": "", "error": "URL must not be empty."}
    if len(str(url)) > MAX_URL_LENGTH:
        return {
            "valid": False,
            "url": str(url)[:80],
            "error": f"URL exceeds {MAX_URL_LENGTH} characters.",
        }
    try:
        parsed = parse_url(url)
    except InvalidURLError as error:
        return {"valid": False, "url": str(url), "error": str(error)}
    return {"valid": True, "url": parsed.url, "error": None}


def describe_url(url: str) -> Dict[str, object]:
    """Return the parsed components of a URL."""
    return parse_url(url).to_dict()
