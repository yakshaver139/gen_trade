"""Tests for the gentrade CLI (Phase 2).

Tests call `cli.main(argv)` directly rather than going through subprocess
— that's both faster and lets us inspect the in-memory database engine
without serialising state to disk for assertions.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

from gentrade.cli import main
from gentrade.persistence import init_db, list_runs


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


def _write_strategies_json(path) -> str:
    strats = [
        {
            "id": f"s{i}",
            "indicators": [
                {"absolute": True, "indicator": "close", "op": ">=", "abs_value": 100.0 + i * 0.5}
            ],
            "conjunctions": [],
        }
        for i in range(4)
    ]
    with open(path, "w") as f:
        json.dump(strats, f)
    return str(path)


# ---------------------------------------------------------------------------
# list (empty)
# ---------------------------------------------------------------------------

def test_list_on_empty_db_prints_no_runs(tmp_path):
    db = tmp_path / "g.db"
    init_db(f"sqlite:///{db}")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["list", "--db-url", f"sqlite:///{db}"])
    assert rc == 0
    assert "no runs" in buf.getvalue().lower()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_creates_persisted_run(tmp_path):
    bars_csv = _write_bars_csv(tmp_path / "bars.csv")
    strats_json = _write_strategies_json(tmp_path / "strats.json")
    db_url = f"sqlite:///{tmp_path}/g.db"
    init_db(db_url)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "run",
                "--data", bars_csv,
                "--strategies", strats_json,
                "--population-size", "4",
                "--generations", "2",
                "--seed", "42",
                "--db-url", db_url,
                "--allow-dirty",
            ]
        )
    assert rc == 0
    out = buf.getvalue()
    assert "run_id" in out

    engine = init_db(db_url)
    rows = list_runs(engine)
    assert len(rows) == 1
    assert rows[0]["status"] == "reported"


def test_list_shows_persisted_runs(tmp_path):
    bars_csv = _write_bars_csv(tmp_path / "bars.csv")
    strats_json = _write_strategies_json(tmp_path / "strats.json")
    db_url = f"sqlite:///{tmp_path}/g.db"
    init_db(db_url)
    main([
        "run", "--data", bars_csv, "--strategies", strats_json,
        "--population-size", "4", "--generations", "2", "--seed", "42",
        "--db-url", db_url, "--allow-dirty",
    ])

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["list", "--db-url", db_url])
    assert rc == 0
    out = buf.getvalue()
    assert "reported" in out


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

def test_resume_completes_an_interrupted_run(tmp_path):
    bars_csv = _write_bars_csv(tmp_path / "bars.csv")
    strats_json = _write_strategies_json(tmp_path / "strats.json")
    db_url = f"sqlite:///{tmp_path}/g.db"
    init_db(db_url)

    # Run, but stop after gen 1 (simulates a kill).
    main([
        "run", "--data", bars_csv, "--strategies", strats_json,
        "--population-size", "4", "--generations", "3", "--seed", "42",
        "--db-url", db_url, "--allow-dirty",
        "--stop-after-generation", "1",
    ])

    engine = init_db(db_url)
    rows = list_runs(engine)
    assert len(rows) == 1
    assert rows[0]["status"] == "in_progress"
    run_id = rows[0]["id"]

    # Resume.
    rc = main([
        "resume", run_id,
        "--data", bars_csv,
        "--db-url", db_url,
    ])
    assert rc == 0

    rows_after = list_runs(engine)
    assert rows_after[0]["status"] == "reported"
    assert rows_after[0]["current_generation"] == 3


# ---------------------------------------------------------------------------
# argparse / help
# ---------------------------------------------------------------------------

def test_unknown_command_prints_help_and_returns_nonzero(tmp_path):
    with pytest.raises(SystemExit):
        main(["wat"])


def test_ingest_downloads_and_writes_parquet(tmp_path, monkeypatch):
    """`gentrade ingest` writes a Parquet file with OHLCV + indicator columns."""

    from gentrade.ingest import TIMEFRAMES

    bar_ms = TIMEFRAMES["15m"]
    fake_bars = [
        [1_640_995_200_000 + i * bar_ms, 100 + i * 0.1, 101 + i * 0.1,
         99 + i * 0.1, 100.5 + i * 0.1, 5.0]
        for i in range(200)
    ]

    class FakeExchange:
        rateLimit = 0  # noqa: N815 — ccxt's name

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
            return list(fake_bars)

    # Patch the default exchange factory so the CLI doesn't try to hit Binance.
    from gentrade import ingest

    monkeypatch.setattr(
        ingest, "_default_exchange_factory", lambda exchange_id: lambda: FakeExchange()
    )

    out_path = tmp_path / "BTCUSDT-15m.parquet"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "ingest",
            "--exchange", "binance",
            "--asset", "BTC/USDT",
            "--interval", "15m",
            "--since", "2022-01-01",
            "--out", str(out_path),
        ])
    assert rc == 0
    assert out_path.exists()
    out_text = buf.getvalue()
    # The CLI prints a registry snippet ready to paste.
    assert "BTCUSDT-15m" in out_text
    assert "binance" in out_text

    # Round-trip check: file is loadable via the standard loader.
    from gentrade.ingest import load_bars

    df = load_bars(out_path)
    assert "open" in df.columns
    assert any(c.startswith("momentum_") for c in df.columns)


def test_ingest_no_indicators_writes_ohlcv_only(tmp_path, monkeypatch):

    from gentrade.ingest import TIMEFRAMES

    bar_ms = TIMEFRAMES["15m"]
    fake_bars = [
        [1_640_995_200_000 + i * bar_ms, 100, 101, 99, 100.5, 5.0]
        for i in range(50)
    ]

    class FakeExchange:
        rateLimit = 0  # noqa: N815

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
            return list(fake_bars)

    from gentrade import ingest

    monkeypatch.setattr(
        ingest, "_default_exchange_factory", lambda exchange_id: lambda: FakeExchange()
    )

    out_path = tmp_path / "raw.parquet"
    rc = main([
        "ingest",
        "--exchange", "binance",
        "--asset", "BTC/USDT",
        "--interval", "15m",
        "--since", "2022-01-01",
        "--out", str(out_path),
        "--no-indicators",
    ])
    assert rc == 0

    from gentrade.ingest import load_bars

    df = load_bars(out_path)
    assert list(df.columns) == ["open_ts", "open", "high", "low", "close", "volume"]


def test_show_prints_headline_metrics_for_a_run(tmp_path):
    bars_csv = _write_bars_csv(tmp_path / "bars.csv")
    strats_json = _write_strategies_json(tmp_path / "strats.json")
    db_url = f"sqlite:///{tmp_path}/g.db"
    init_db(db_url)
    main([
        "run", "--data", bars_csv, "--strategies", strats_json,
        "--population-size", "4", "--generations", "2", "--seed", "42",
        "--db-url", db_url, "--allow-dirty",
    ])

    rows = list_runs(init_db(db_url))
    run_id = rows[0]["id"]

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["show", run_id, "--db-url", db_url])
    assert rc == 0
    out = buf.getvalue()
    assert run_id in out
    assert "reported" in out
    assert "generation" in out
    assert "train_max" in out


def test_show_unknown_id_returns_nonzero(tmp_path):
    db_url = f"sqlite:///{tmp_path}/g.db"
    init_db(db_url)
    rc = main(["show", "not-a-real-id", "--db-url", db_url])
    assert rc != 0


def test_run_resume_unknown_id_returns_nonzero(tmp_path):
    bars_csv = _write_bars_csv(tmp_path / "bars.csv")
    db_url = f"sqlite:///{tmp_path}/g.db"
    init_db(db_url)
    rc = main([
        "resume", "not-a-real-id",
        "--data", bars_csv,
        "--db-url", db_url,
    ])
    assert rc != 0
