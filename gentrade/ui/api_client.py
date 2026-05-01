"""Thin HTTP client for the gentrade API.

Lives outside the Streamlit pages so we can unit-test it without spinning
up a Streamlit runtime. Reads the API base URL and key from
``GENTRADE_API_URL`` / ``GENTRADE_API_KEY`` by default; tests can pass
``base_url`` and ``api_key`` directly.

The client raises ``ApiError`` on non-2xx responses with the server's
``detail`` string surfaced — Streamlit pages catch and ``st.error()`` it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


class ApiError(RuntimeError):
    """Surface-level wrapper for non-2xx responses from the API."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str


def config_from_env() -> Config:
    return Config(
        base_url=os.environ.get("GENTRADE_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        api_key=os.environ.get("GENTRADE_API_KEY", ""),
    )


class ApiClient:
    """Synchronous httpx wrapper. Intentionally tiny — Streamlit pages
    are top-down scripts; an async client would just complicate things."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if base_url is None or api_key is None:
            cfg = config_from_env()
            base_url = base_url or cfg.base_url
            api_key = api_key or cfg.api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        # Allow injection so tests can plug in `httpx.Client(transport=ASGITransport(app=...))`.
        self._client = client or httpx.Client(timeout=10.0)

    # ----- HTTP plumbing -----

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        url = f"{self.base_url}{path}"
        r = self._client.request(method, url, headers=self.headers, **kw)
        if not r.is_success:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise ApiError(r.status_code, str(detail))
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    # ----- endpoints -----

    def healthz(self) -> dict[str, str]:
        return self._request("GET", "/healthz")

    def list_assets(self) -> list[dict]:
        return self._request("GET", "/assets")

    def list_runs(self) -> list[dict]:
        return self._request("GET", "/runs")

    def get_run(self, run_id: str) -> dict:
        return self._request("GET", f"/runs/{run_id}")

    def get_generation(self, run_id: str, n: int) -> dict:
        return self._request("GET", f"/runs/{run_id}/generations/{n}")

    def get_strategy(self, run_id: str, strategy_id: str) -> dict:
        return self._request("GET", f"/runs/{run_id}/strategies/{strategy_id}")

    def create_run(self, body: dict) -> dict:
        return self._request("POST", "/runs", json=body)

    def post_backtest(self, run_id: str, strategy_id: str) -> dict:
        return self._request(
            "POST", "/backtests", json={"run_id": run_id, "strategy_id": strategy_id}
        )

    def post_cross_asset(
        self, run_id: str, strategy_id: str, assets: list[str]
    ) -> dict:
        return self._request(
            "POST",
            "/backtests/cross_asset",
            json={"run_id": run_id, "strategy_id": strategy_id, "assets": assets},
        )
