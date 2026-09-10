"""Access-key authentication helpers for the Streamlit application."""
from __future__ import annotations

import hmac
import os

ACCESS_KEY_ENV = "SALESFORECAST_ACCESS_KEY"
MIN_ACCESS_KEY_LENGTH = 16


def configured_access_key(secret_value: str = "") -> str:
    """Return the configured access key, preferring the environment variable."""
    return (os.getenv(ACCESS_KEY_ENV) or secret_value or "").strip()


def is_authorized(provided_key: str, configured_key: str) -> bool:
    """Check a supplied key without accepting weak or missing configuration."""
    expected = (configured_key or "").strip()
    provided = (provided_key or "").strip()
    return len(expected) >= MIN_ACCESS_KEY_LENGTH and hmac.compare_digest(provided, expected)
