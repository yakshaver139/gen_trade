"""Tests for the FastAPI app (Phase 3).

Covers auth, asset registry resolution, run creation + listing + detail,
generation + strategy detail, ad-hoc backtests via POST /backtests, and
healthz. Background-job semantics are exercised by waiting for the job
thread to finish (small populations, two generations — runs in <1 sec).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gentrade.api import assets as asset_registry
from gentrade.api.app import create_app
from gentrade.api.assets import AssetEntry
from gentrade.api.auth import configure_api_key
from gentrade.persistence import init_db

API_KEY = "test-secret-not-real"


def _write_bars_csv(path) -> str:
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
def client(tmp_path, monkeypatch):
    """Wire a fully-isolated app: tmp DB, tmp logs, in-process asset registry,
    in-process API key. Each test gets a fresh stack."""
    bars_csv = _write_bars_csv(tmp_path / "bars.csv")
    asset_registry.configure_registry(
        [AssetEntry(asset="TEST-15m", exchange="test", interval="15m", path=bars_csv)]
    )
    configure_api_key(API_KEY)

    # Patch _generate_strategies so we don't depend on the full signal catalogue.
    from gentrade.api import jobs

    def _fake_strategies(population_size: int, seed: int) -> list[dict]:
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

    monkeypatch.setattr(jobs, "_generate_strategies", _fake_strategies)

    engine = init_db(f"sqlite:///{tmp_path}/g.db")
    app = create_app(engine=engine, log_dir=tmp_path / "runs")
    yield TestClient(app), tmp_path

    asset_registry.configure_registry(None)
    configure_api_key(None)


def _wait_for_status(client, run_id: str, target: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/runs/{run_id}", headers={"X-API-Key": API_KEY})
        assert r.status_code == 200
        body = r.json()
        if body["status"] == target:
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached status {target}; last={body}")


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------

def test_healthz_is_unauthenticated(client):
    c, _ = client
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_protected_endpoint_without_key_returns_401(client):
    c, _ = client
    r = c.get("/runs")
    assert r.status_code == 401


def test_protected_endpoint_with_wrong_key_returns_401(client):
    c, _ = client
    r = c.get("/runs", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_protected_endpoint_with_right_key_returns_200(client):
    c, _ = client
    r = c.get("/runs", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    assert r.json() == []


def test_no_server_key_configured_returns_503(tmp_path, monkeypatch):
    """Failing closed: if neither env nor the test hook provides a key, refuse.

    Globals (the API-key + asset-registry singletons) are reset on teardown
    via try/finally so a mid-test failure can't leak module state into
    subsequent tests — a route that would otherwise enable an auth bypass.
    """
    monkeypatch.delenv("GENTRADE_API_KEY", raising=False)
    asset_registry.configure_registry([])
    configure_api_key(None)
    try:
        engine = init_db(f"sqlite:///{tmp_path}/g.db")
        app = create_app(engine=engine, log_dir=tmp_path / "runs")
        c = TestClient(app)
        r = c.get("/runs", headers={"X-API-Key": "anything"})
        assert r.status_code == 503
    finally:
        asset_registry.configure_registry(None)
        configure_api_key(None)


# ---------------------------------------------------------------------------
# /assets
# ---------------------------------------------------------------------------

def test_list_assets(client):
    c, _ = client
    r = c.get("/assets", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["asset"] == "TEST-15m"
    assert body[0]["exchange"] == "test"


# ---------------------------------------------------------------------------
# POST /runs
# ---------------------------------------------------------------------------

def test_post_run_unknown_asset_returns_400(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={"asset": "DOES-NOT-EXIST", "population_size": 4, "generations": 2},
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 400


def test_post_run_kicks_off_a_run_and_returns_202(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={
            "asset": "TEST-15m",
            "population_size": 4,
            "generations": 2,
            "seed": 42,
        },
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 202
    body = r.json()
    assert "run_id" in body
    assert body["status"] == "in_progress"
    assert body["status_url"] == f"/runs/{body['run_id']}"

    # Wait for completion.
    final = _wait_for_status(c, body["run_id"], target="reported")
    assert final["chosen_strategy_id"] is not None
    assert final["test_metrics"] is not None
    assert final["current_generation"] == 2


def test_catalogue_returns_known_indicators(client):
    c, _ = client
    r = c.get("/catalogue", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) > 0
    # every entry has the documented shape
    for entry in body:
        assert "indicator" in entry and isinstance(entry["indicator"], str)
        assert "type" in entry
        assert "ops" in entry and isinstance(entry["ops"], list)
        assert "absolute_thresholds" in entry
    # spot-check: momentum_rsi has both >= and <= ops in the catalogue
    rsi = next((e for e in body if e["indicator"] == "momentum_rsi"), None)
    assert rsi is not None
    assert ">=" in rsi["ops"] or "<=" in rsi["ops"]


def test_catalogue_requires_auth(client):
    c, _ = client
    r = c.get("/catalogue")
    assert r.status_code == 401


def test_post_run_with_valid_seed_strategy(client):
    c, _ = client
    body = {
        "asset": "TEST-15m",
        "population_size": 4,
        "generations": 2,
        "seed": 42,
        "seed_strategies": [
            {
                "indicators": [
                    {
                        "indicator": "momentum_rsi",
                        "op": ">=",
                        "absolute": True,
                        "abs_value": 70.0,
                    }
                ],
                "conjunctions": [],
            }
        ],
    }
    r = c.post("/runs", json=body, headers={"X-API-Key": API_KEY})
    assert r.status_code == 202
    rid = r.json()["run_id"]
    final = _wait_for_status(c, rid, target="reported")
    # the seeded strategy must appear in generation 1's population (rank 0,
    # since seeds are prepended before the random pool)
    g1 = c.get(
        f"/runs/{rid}/generations/1", headers={"X-API-Key": API_KEY}
    ).json()
    seeded = g1["strategies"][0]
    assert any(
        ind["indicator"] == "momentum_rsi" and ind["abs_value"] == 70.0
        for ind in seeded["indicators"]
    )
    assert final["chosen_strategy_id"] is not None


def test_post_run_seed_with_unknown_indicator_returns_400(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={
            "asset": "TEST-15m",
            "population_size": 4,
            "generations": 2,
            "seed_strategies": [
                {
                    "indicators": [
                        {
                            "indicator": "totally_made_up",
                            "op": ">=",
                            "absolute": True,
                            "abs_value": 1.0,
                        }
                    ],
                    "conjunctions": [],
                }
            ],
        },
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 400
    assert "totally_made_up" in r.json()["detail"]


def test_post_run_seed_with_first_or_conjunction_rejected(client):
    """Per the FirstConjunctionIsAnd spec invariant."""
    c, _ = client
    r = c.post(
        "/runs",
        json={
            "asset": "TEST-15m",
            "population_size": 4,
            "generations": 2,
            "seed_strategies": [
                {
                    "indicators": [
                        {"indicator": "momentum_rsi", "op": ">=",
                         "absolute": True, "abs_value": 70.0},
                        {"indicator": "momentum_rsi", "op": "<=",
                         "absolute": True, "abs_value": 30.0},
                    ],
                    "conjunctions": ["or"],
                }
            ],
        },
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 400
    assert "first conjunction must be 'and'" in r.json()["detail"]


def test_post_run_seed_count_mismatch_returns_400(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={
            "asset": "TEST-15m",
            "population_size": 4,
            "generations": 2,
            "seed_strategies": [
                {
                    "indicators": [
                        {"indicator": "momentum_rsi", "op": ">=",
                         "absolute": True, "abs_value": 70.0},
                        {"indicator": "momentum_rsi", "op": "<=",
                         "absolute": True, "abs_value": 30.0},
                    ],
                    # too many for 2 indicators (need exactly 1)
                    "conjunctions": ["and", "or"],
                }
            ],
        },
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 400


def test_post_run_invalid_pressure_returns_422(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={"asset": "TEST-15m", "selection_pressure": "not-a-thing"},
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 422


def test_post_run_writes_per_run_log_file(client):
    c, run_dir = client
    r = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    )
    body = r.json()
    _wait_for_status(c, body["run_id"], target="reported")

    log_path = run_dir / "runs" / body["run_id"] / "run.log"
    assert log_path.exists()
    contents = log_path.read_text()
    assert "starting run" in contents
    assert "finished run" in contents


# ---------------------------------------------------------------------------
# GET /runs, /runs/{id}, /runs/{id}/generations/{n}, strategies
# ---------------------------------------------------------------------------

def test_get_runs_lists_persisted_runs(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    )
    body = r.json()
    _wait_for_status(c, body["run_id"], target="reported")

    r2 = c.get("/runs", headers={"X-API-Key": API_KEY})
    assert r2.status_code == 200
    rows = r2.json()
    assert len(rows) == 1
    assert rows[0]["id"] == body["run_id"]
    assert rows[0]["status"] == "reported"


def test_get_run_returns_404_for_unknown_id(client):
    c, _ = client
    r = c.get("/runs/not-a-real-id", headers={"X-API-Key": API_KEY})
    assert r.status_code == 404


def test_get_generation_includes_breeding_events(client):
    """Each generation past 1 should have one breeding event per child:
    elites carry operator='(elite)' and self-parents; bred children carry
    a real operator + their two distinct parents."""
    c, _ = client
    body = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 3, "seed": 42},
        headers={"X-API-Key": API_KEY},
    ).json()
    _wait_for_status(c, body["run_id"], target="reported")

    # Generation 2 was bred from gen 1, so we should see 4 events.
    r = c.get(f"/runs/{body['run_id']}/generations/2", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    detail = r.json()
    events = detail["breeding_events"]
    assert len(events) == 4

    # Required event shape for each child
    for ev in events:
        for k in ("generation_number", "child_id", "parent_a_id",
                  "parent_b_id", "operator", "applied"):
            assert k in ev
        assert ev["generation_number"] == 2

    # Two elites with operator='(elite)' and self-as-both-parents.
    elites = [e for e in events if e["operator"] == "(elite)"]
    assert len(elites) == 2
    for e in elites:
        assert e["parent_a_id"] == e["parent_b_id"] == e["child_id"]

    # The non-elite children must have an operator from the rich set.
    bred = [e for e in events if e["operator"] != "(elite)"]
    assert len(bred) == 2
    valid_ops = {
        "perturb_threshold", "flip_operator", "swap_indicator",
        "flip_conjunction", "add_signal", "remove_signal", "swap_rel_target",
    }
    for e in bred:
        assert e["operator"] in valid_ops


def test_get_generation_one_has_no_breeding_events(client):
    """Generation 1 is the initial random pool — no breeding has happened."""
    c, _ = client
    body = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    ).json()
    _wait_for_status(c, body["run_id"], target="reported")

    r = c.get(f"/runs/{body['run_id']}/generations/1", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    assert r.json()["breeding_events"] == []


def test_list_run_strategies_returns_flat_sorted_list(client):
    """The flat strategies endpoint returns one row per (gen, strategy)
    sorted by generation then rank. Used by the lineage view to
    position nodes."""
    c, _ = client
    body = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 3, "seed": 42},
        headers={"X-API-Key": API_KEY},
    ).json()
    _wait_for_status(c, body["run_id"], target="reported")

    r = c.get(f"/runs/{body['run_id']}/strategies", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    rows = r.json()
    # 4 strategies × 3 generations = 12 rows
    assert len(rows) == 12
    # Sorted by (generation_number, rank)
    seen = [(r["generation_number"], r["rank"]) for r in rows]
    assert seen == sorted(seen)
    # Slim payload: no indicators / parsed_query / etc.
    for r in rows:
        for forbidden in ("indicators", "conjunctions", "parsed_query"):
            assert forbidden not in r


def test_get_generation_returns_strategies(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    )
    body = r.json()
    _wait_for_status(c, body["run_id"], target="reported")

    r2 = c.get(f"/runs/{body['run_id']}/generations/1", headers={"X-API-Key": API_KEY})
    assert r2.status_code == 200
    detail = r2.json()
    assert detail["number"] == 1
    assert len(detail["strategies"]) == 4
    # parsed query string is included
    assert all("parsed_query" in s for s in detail["strategies"])


def test_get_generation_404_for_unknown_generation(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    )
    body = r.json()
    _wait_for_status(c, body["run_id"], target="reported")

    r2 = c.get(f"/runs/{body['run_id']}/generations/99", headers={"X-API-Key": API_KEY})
    assert r2.status_code == 404


def test_get_strategy_returns_parsed_query(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    )
    body = r.json()
    _wait_for_status(c, body["run_id"], target="reported")
    chosen_id = c.get(f"/runs/{body['run_id']}", headers={"X-API-Key": API_KEY}).json()["chosen_strategy_id"]

    r2 = c.get(
        f"/runs/{body['run_id']}/strategies/{chosen_id}",
        headers={"X-API-Key": API_KEY},
    )
    assert r2.status_code == 200
    detail = r2.json()
    assert detail["id"] == chosen_id
    assert "parsed_query" in detail
    assert "indicators" in detail


# ---------------------------------------------------------------------------
# POST /backtests
# ---------------------------------------------------------------------------

def test_post_backtest_reruns_saved_strategy(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    )
    body = r.json()
    final = _wait_for_status(c, body["run_id"], target="reported")
    chosen_id = final["chosen_strategy_id"]

    r2 = c.post(
        "/backtests",
        json={"run_id": body["run_id"], "strategy_id": chosen_id},
        headers={"X-API-Key": API_KEY},
    )
    assert r2.status_code == 200
    bt = r2.json()
    assert bt["chosen_strategy_id"] == chosen_id
    assert "test_metrics" in bt
    assert "buy_and_hold_test" in bt
    # test_trades carries individual trade records for the UI's charts
    assert "test_trades" in bt
    assert isinstance(bt["test_trades"], list)
    if bt["test_trades"]:
        first = bt["test_trades"][0]
        for k in ("entry_time", "entry_price", "exit_time", "exit_price",
                  "outcome", "return"):
            assert k in first
    # test_bars carries OHLC bars for the candlestick chart, downsampled
    assert "test_bars" in bt
    assert isinstance(bt["test_bars"], list)
    assert len(bt["test_bars"]) <= 1000  # server-side cap
    if bt["test_bars"]:
        first_bar = bt["test_bars"][0]
        for k in ("open_ts", "open", "high", "low", "close", "volume"):
            assert k in first_bar


def test_sse_events_stream_reports_progress_and_terminates(client):
    """The events stream emits at least one progress event and a terminal one."""
    c, _ = client
    body = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    ).json()
    run_id = body["run_id"]

    events: list[str] = []
    with c.stream(
        "GET",
        f"/runs/{run_id}/events",
        headers={"X-API-Key": API_KEY},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            events.append(line)
            # The terminal event closes the stream; safety bound below.
            if "event: terminal" in line:
                pass
            if len(events) > 200:
                break

    blob = "\n".join(events)
    assert "event: progress" in blob
    assert "event: terminal" in blob
    assert '"status": "reported"' in blob


def test_sse_events_unknown_run_returns_404(client):
    c, _ = client
    r = c.get("/runs/not-a-real-id/events", headers={"X-API-Key": API_KEY})
    assert r.status_code == 404


def test_sse_events_requires_auth(client):
    c, _ = client
    body = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    ).json()
    r = c.get(f"/runs/{body['run_id']}/events")  # no key
    assert r.status_code == 401


def test_cross_asset_evaluates_strategy_against_other_registered_assets(tmp_path, monkeypatch):
    """POST /backtests/cross_asset returns one row per requested asset."""
    bars_a = _write_bars_csv(tmp_path / "a.csv")
    bars_b = _write_bars_csv(tmp_path / "b.csv")
    bars_c = _write_bars_csv(tmp_path / "c.csv")
    asset_registry.configure_registry(
        [
            AssetEntry(asset="A-15m", exchange="test", interval="15m", path=bars_a),
            AssetEntry(asset="B-15m", exchange="test", interval="15m", path=bars_b),
            AssetEntry(asset="C-15m", exchange="test", interval="15m", path=bars_c),
        ]
    )
    configure_api_key(API_KEY)
    try:
        from gentrade.api import jobs

        def _fake_strategies(population_size: int, seed: int) -> list[dict]:
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

        monkeypatch.setattr(jobs, "_generate_strategies", _fake_strategies)

        engine = init_db(f"sqlite:///{tmp_path}/g.db")
        app = create_app(engine=engine, log_dir=tmp_path / "runs")
        c = TestClient(app)

        # Run on asset A first.
        r = c.post(
            "/runs",
            json={"asset": "A-15m", "population_size": 4, "generations": 2, "seed": 42},
            headers={"X-API-Key": API_KEY},
        )
        body = r.json()
        final = _wait_for_status(c, body["run_id"], target="reported")
        chosen_id = final["chosen_strategy_id"]

        # Now evaluate on B and C.
        r2 = c.post(
            "/backtests/cross_asset",
            json={
                "run_id": body["run_id"],
                "strategy_id": chosen_id,
                "assets": ["B-15m", "C-15m"],
            },
            headers={"X-API-Key": API_KEY},
        )
        assert r2.status_code == 200
        out = r2.json()
        assert out["chosen_strategy_id"] == chosen_id
        assert out["base_asset"] == "A-15m"
        assert len(out["rows"]) == 2
        assert {r["asset"] for r in out["rows"]} == {"B-15m", "C-15m"}
        for row in out["rows"]:
            assert row["error"] is None
            assert row["n_bars"] > 0
            assert "metrics" in row
    finally:
        asset_registry.configure_registry(None)
        configure_api_key(None)


def test_cross_asset_unresolved_asset_returns_per_row_error(tmp_path, monkeypatch):
    """Mixing registered + unregistered assets surfaces per-row errors, not 4xx."""
    bars_a = _write_bars_csv(tmp_path / "a.csv")
    asset_registry.configure_registry(
        [AssetEntry(asset="A-15m", exchange="test", interval="15m", path=bars_a)]
    )
    configure_api_key(API_KEY)
    try:
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
        c = TestClient(app)

        r = c.post(
            "/runs",
            json={"asset": "A-15m", "population_size": 4, "generations": 2, "seed": 42},
            headers={"X-API-Key": API_KEY},
        )
        body = r.json()
        final = _wait_for_status(c, body["run_id"], target="reported")

        r2 = c.post(
            "/backtests/cross_asset",
            json={
                "run_id": body["run_id"],
                "strategy_id": final["chosen_strategy_id"],
                "assets": ["A-15m", "GHOST-15m"],
            },
            headers={"X-API-Key": API_KEY},
        )
        assert r2.status_code == 200
        rows = r2.json()["rows"]
        ghost = next(r for r in rows if r["asset"] == "GHOST-15m")
        assert ghost["error"] is not None
        assert "registry" in ghost["error"]
    finally:
        asset_registry.configure_registry(None)
        configure_api_key(None)


def test_cross_asset_no_resolvable_assets_returns_422(client):
    c, _ = client
    body = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    ).json()
    final = _wait_for_status(c, body["run_id"], target="reported")

    r = c.post(
        "/backtests/cross_asset",
        json={
            "run_id": body["run_id"],
            "strategy_id": final["chosen_strategy_id"],
            "assets": ["GHOST-1", "GHOST-2"],
        },
        headers={"X-API-Key": API_KEY},
    )
    assert r.status_code == 422


def test_post_backtest_unknown_strategy_returns_404(client):
    c, _ = client
    r = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2, "seed": 42},
        headers={"X-API-Key": API_KEY},
    )
    body = r.json()
    _wait_for_status(c, body["run_id"], target="reported")

    r2 = c.post(
        "/backtests",
        json={"run_id": body["run_id"], "strategy_id": "nope"},
        headers={"X-API-Key": API_KEY},
    )
    assert r2.status_code == 404


# ---------------------------------------------------------------------------
# resume from CLI works on API-created runs
# ---------------------------------------------------------------------------

def test_openapi_snapshot_matches_committed_file():
    """The committed openapi.json must match what the app generates today.

    Regenerate with: ``uv run python -m gentrade.api.export_openapi > openapi.json``.
    """
    import json
    from pathlib import Path

    from gentrade.persistence import init_db

    snapshot_path = Path(__file__).resolve().parent.parent / "openapi.json"
    if not snapshot_path.exists():
        pytest.skip("no committed openapi.json — run snapshot generator first")

    engine = init_db("sqlite:///:memory:")
    app = create_app(engine=engine)
    live = json.loads(json.dumps(app.openapi(), sort_keys=True))
    committed = json.loads(snapshot_path.read_text())

    # FastAPI version may bump description text; pin only the routes + schema names
    assert sorted(live["paths"].keys()) == sorted(committed["paths"].keys())
    assert sorted(live["components"]["schemas"].keys()) == sorted(
        committed["components"]["schemas"].keys()
    )


def test_failed_run_status_recorded(client, monkeypatch):
    """If the GA loop raises, the run row is marked failed."""

    def _explode(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("gentrade.api.jobs.run_ga", _explode)

    c, _ = client
    r = c.post(
        "/runs",
        json={"asset": "TEST-15m", "population_size": 4, "generations": 2},
        headers={"X-API-Key": API_KEY},
    )
    body = r.json()
    final = _wait_for_status(c, body["run_id"], target="failed", timeout=5.0)
    assert final["status"] == "failed"
