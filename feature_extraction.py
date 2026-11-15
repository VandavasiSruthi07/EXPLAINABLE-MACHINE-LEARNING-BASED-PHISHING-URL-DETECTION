"""Lexical URL feature extraction. Does not fetch or open the submitted URL."""
import ipaddress
import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = (
    "login", "verify", "secure", "update", "account", "bank",
    "password", "signin", "confirm", "wallet", "bonus", "free"
)

def extract_features(url: str) -> dict:
    raw = (url or "").strip()
    candidate = raw if "://" in raw else "http://" + raw
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower()
    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0

    labels = [part for part in host.split(".") if part]
    return {
        "url_length": len(raw),
        "hostname_length": len(host),
        "dot_count": raw.count("."),
        "hyphen_count": raw.count("-"),
        "at_count": raw.count("@"),
        "question_count": raw.count("?"),
        "ampersand_count": raw.count("&"),
        "equals_count": raw.count("="),
        "slash_count": raw.count("/"),
        "https": int(parsed.scheme.lower() == "https"),
        "has_ip": has_ip,
        "subdomain_count": max(0, len(labels) - 2),
        "suspicious_word_count": sum(
            len(re.findall(rf"\b{re.escape(word)}\b", raw.lower()))
            for word in SUSPICIOUS_WORDS
        ),
        "digit_count": sum(ch.isdigit() for ch in raw),
    }

FEATURE_NAMES = list(extract_features("https://example.com").keys())

def feature_vector(url: str):
    values = extract_features(url)
    return [values[name] for name in FEATURE_NAMES]
