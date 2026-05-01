"""Tests for the Streamlit UI's HTTP client.

We mount the FastAPI test app via Starlette's TestClient (a sync
httpx.Client subclass) so the client makes real HTTP calls in-process,
without binding a port.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gentrade.api import assets as asset_registry
from gentrade.api.app import create_app
from gentrade.api.assets import AssetEntry
from gentrade.api.auth import configure_api_key
from gentrade.persistence import init_db
from gentrade.ui.api_client import ApiClient, ApiError, config_from_env

API_KEY = "ui-test-key"


def _bars_csv(path) -> str:
    n = 60
    base = pd.Timestamp("2022-01-01", tz="UTC")
    closes = np.array([100.0 + (i % 10) * 0.2 for i in range(n)])
    opens = np.concatenate([[closes[0]], closes[:-1]])
    df = pd.DataFrame(
        {
            "open_ts": [(base + pd.Timedelta(minutes=15 * i)).isoformat() for i in range(n)],
            "open": opens,
            "high": np.maximum(opens, closes) + 0.01,
            "low": np.minimum(opens, closes) - 0.01,
            "close": closes,
            "volume": np.full(n, 10.0),
        }
    )
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    bars = _bars_csv(tmp_path / "bars.csv")
    asset_registry.configure_registry(
        [AssetEntry(asset="TEST-15m", exchange="test", interval="15m", path=bars)]
    )
    configure_api_key(API_KEY)

    # Patch _generate_strategies so we don't depend on the full signal catalogue.
    from gentrade.api import jobs

    def _fake(population_size: int, seed: int) -> list[dict]:
        return [
            {
                "id": f"s{i}",
                "indicators": [
                    {"absolute": True, "indicator": "close", "op": ">=", "abs_value": 100.0 + i * 0.5}
                ],
                "conjunctions": [],
            }
            for i in range(population_size)
        ]

    monkeypatch.setattr(jobs, "_generate_strategies", _fake)

    engine = init_db(f"sqlite:///{tmp_path}/g.db")
    app = create_app(engine=engine, log_dir=tmp_path / "runs")

    test_client = TestClient(app)
    yield ApiClient(base_url="", api_key=API_KEY, client=test_client)

    asset_registry.configure_registry(None)
    configure_api_key(None)


def test_config_from_env_defaults(monkeypatch):
    monkeypatch.delenv("GENTRADE_API_URL", raising=False)
    monkeypatch.delenv("GENTRADE_API_KEY", raising=False)
    cfg = config_from_env()
    assert cfg.base_url == "http://127.0.0.1:8000"
    assert cfg.api_key == ""


def test_config_from_env_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("GENTRADE_API_URL", "http://example/")
    monkeypatch.setenv("GENTRADE_API_KEY", "abc")
    cfg = config_from_env()
    assert cfg.base_url == "http://example"
    assert cfg.api_key == "abc"


def test_healthz(api_client):
    assert api_client.healthz() == {"status": "ok"}


def test_list_assets(api_client):
    assets = api_client.list_assets()
    assert len(assets) == 1
    assert assets[0]["asset"] == "TEST-15m"


def test_list_runs_initially_empty(api_client):
    assert api_client.list_runs() == []


def test_create_run_then_list(api_client):
    body = api_client.create_run(
        {"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42}
    )
    assert "run_id" in body
    runs = api_client.list_runs()
    assert any(r["id"] == body["run_id"] for r in runs)


def test_unknown_run_raises_api_error(api_client):
    with pytest.raises(ApiError) as exc:
        api_client.get_run("not-a-real-id")
    assert exc.value.status == 404
    assert "not found" in exc.value.detail


def test_bad_api_key_raises_401(tmp_path):
    """If the configured key on the client doesn't match server's, 401 is surfaced."""
    bars = _bars_csv(tmp_path / "bars.csv")
    asset_registry.configure_registry(
        [AssetEntry(asset="TEST-15m", exchange="test", interval="15m", path=bars)]
    )
    configure_api_key("server-key")
    try:
        engine = init_db(f"sqlite:///{tmp_path}/g.db")
        app = create_app(engine=engine, log_dir=tmp_path / "runs")
        client = ApiClient(base_url="", api_key="wrong-key", client=TestClient(app))
        with pytest.raises(ApiError) as exc:
            client.list_runs()
        assert exc.value.status == 401
    finally:
        asset_registry.configure_registry(None)
        configure_api_key(None)
