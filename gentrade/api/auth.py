"""API-key authentication.

Read once at process start from ``GENTRADE_API_KEY`` (or override per-test
with ``configure_api_key``). All non-public endpoints depend on
``require_api_key``, which compares the ``X-API-Key`` header against the
configured key in constant time.

`/healthz` is intentionally unauthenticated — load balancers and uptime
monitors need it to work without secrets.
"""
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status

_API_KEY: str | None = None


def configure_api_key(key: str | None) -> None:
    """Override the API key (test hook). Pass ``None`` to clear."""
    global _API_KEY
    _API_KEY = key


def _current_key() -> str | None:
    if _API_KEY is not None:
        return _API_KEY
    return os.getenv("GENTRADE_API_KEY")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: 401 unless the request carries the right key.

    If no key is configured anywhere, every request is rejected — failing
    closed avoids accidentally exposing the API on the public internet
    without auth.
    """
    expected = _current_key()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key not configured on the server",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )
