"""Domain, TLD and keyword based URL features.

The keyword lists below are deliberately treated as *features*. A URL that
contains the word "login" is not phishing because of that word; the model
decides how much weight the signal carries once it is combined with the
rest of the feature vector.
"""

from __future__ import annotations

from typing import Dict

from app.feature_engineering.url_parser import ParsedURL

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "secure", "security",
    "account", "update", "confirm", "password", "credential", "bank",
    "billing", "invoice", "payment", "wallet", "recover", "unlock",
    "suspend", "alert", "limited", "support", "service", "webscr",
]

BRAND_TOKENS = [
    "paypal", "apple", "google", "microsoft", "amazon", "netflix", "facebook",
    "instagram", "whatsapp", "linkedin", "dropbox", "outlook", "office365",
    "icloud", "chase", "hsbc", "barclays", "wellsfargo", "citibank", "sbi",
    "icici", "hdfc", "axisbank", "paytm", "binance", "coinbase", "steam",
]

# Low cost TLDs that appear frequently in abuse reports. Presence is a
# feature, not a verdict.
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "click", "link",
    "work", "country", "kim", "loan", "men", "date", "download", "review",
    "zip", "mov", "rest", "cam", "sbs", "cfd", "icu",
}

COMMON_TLDS = {"com", "org", "net", "edu", "gov", "int", "mil"}

DOMAIN_FEATURE_NAMES = [
    "domain_length",
    "tld_length",
    "is_suspicious_tld",
    "is_common_tld",
    "suspicious_keyword_count",
    "has_suspicious_keyword",
    "keyword_in_hostname",
    "keyword_in_path",
    "brand_token_count",
    "brand_outside_domain",
    "num_host_labels",
    "longest_host_label_length",
    "domain_digit_count",
    "domain_hyphen_count",
]


def extract_domain_features(parsed: ParsedURL) -> Dict[str, float]:
    """Return the domain and keyword feature dictionary for a parsed URL."""
    lowered = parsed.url.lower()
    hostname = parsed.hostname
    path_and_query = (parsed.path + "?" + parsed.query).lower()
    registered = parsed.registered_domain
    domain_label = registered.split(".")[0] if registered else ""
    suffix = parsed.suffix
    host_labels = parsed.host_labels

    keyword_hits = [word for word in SUSPICIOUS_KEYWORDS if word in lowered]
    brand_hits = [brand for brand in BRAND_TOKENS if brand in lowered]

    # A brand name that appears anywhere except the registered domain is a
    # classic impersonation pattern (for example paypal.secure-login.xyz).
    brand_outside_domain = any(
        brand in lowered and brand not in registered for brand in BRAND_TOKENS
    )

    return {
        "domain_length": float(len(registered)),
        "tld_length": float(len(suffix)),
        "is_suspicious_tld": 1.0 if suffix.split(".")[-1] in SUSPICIOUS_TLDS else 0.0,
        "is_common_tld": 1.0 if suffix.split(".")[-1] in COMMON_TLDS else 0.0,
        "suspicious_keyword_count": float(len(keyword_hits)),
        "has_suspicious_keyword": 1.0 if keyword_hits else 0.0,
        "keyword_in_hostname": 1.0
        if any(word in hostname for word in SUSPICIOUS_KEYWORDS)
        else 0.0,
        "keyword_in_path": 1.0
        if any(word in path_and_query for word in SUSPICIOUS_KEYWORDS)
        else 0.0,
        "brand_token_count": float(len(brand_hits)),
        "brand_outside_domain": 1.0 if brand_outside_domain else 0.0,
        "num_host_labels": float(len(host_labels)),
        "longest_host_label_length": float(
            max((len(label) for label in host_labels), default=0)
        ),
        "domain_digit_count": float(
            sum(character.isdigit() for character in domain_label)
        ),
        "domain_hyphen_count": float(domain_label.count("-")),
    }
