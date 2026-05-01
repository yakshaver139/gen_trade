"""Tests for the ccxt-based ingest pipeline.

Tests inject a fake ccxt-style exchange via ``exchange_factory`` so no
network calls happen. The fake mirrors ccxt's actual ``fetch_ohlcv``
contract: returns a list of ``[ms, o, h, l, c, v]``, capped by ``limit``,
keyed off ``since``.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from gentrade.ingest import (
    OHLCV_COLS,
    TIMEFRAMES,
    compute_indicators,
    fetch_ohlcv,
    load_parquet,
    save_parquet,
)


def _bar_series(start_ms: int, n: int, bar_ms: int) -> list[list[float]]:
    rows = []
    for i in range(n):
        ts = start_ms + i * bar_ms
        # Plausible-shaped OHLCV; not real prices.
        close = 100.0 + (i % 10) * 0.2
        rows.append([ts, close, close + 0.1, close - 0.1, close, 5.0])
    return rows


class FakeExchange:
    """ccxt-shaped stub. ``pages`` is an iterator of pre-built bar lists."""

    def __init__(self, pages: Iterator[list[list[float]]]) -> None:
        self._pages = list(pages)
        self.calls: list[tuple] = []
        self.rateLimit = 0  # noqa: N815 — ccxt name

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls.append((symbol, timeframe, since, limit))
        if not self._pages:
            return []
        return self._pages.pop(0)


# ---------------------------------------------------------------------------
# fetch_ohlcv: paging + dedup + window
# ---------------------------------------------------------------------------

def test_fetch_ohlcv_returns_tidy_frame():
    bar_ms = TIMEFRAMES["15m"]
    page = _bar_series(start_ms=1_640_995_200_000, n=10, bar_ms=bar_ms)
    fake = FakeExchange(iter([page]))

    df = fetch_ohlcv(
        "binance", "BTC/USDT", interval="15m",
        page_limit=1000, exchange_factory=lambda: fake,
    )

    assert list(df.columns) == OHLCV_COLS
    assert len(df) == 10
    assert df["open_ts"].dt.tz is not None  # UTC-aware
    assert df["open_ts"].is_monotonic_increasing


def test_fetch_ohlcv_pages_until_short_response():
    bar_ms = TIMEFRAMES["15m"]
    full = _bar_series(start_ms=1_640_995_200_000, n=1000, bar_ms=bar_ms)
    short = _bar_series(start_ms=1_640_995_200_000 + 1000 * bar_ms, n=300, bar_ms=bar_ms)
    fake = FakeExchange(iter([full, short]))

    df = fetch_ohlcv(
        "binance", "BTC/USDT", interval="15m",
        page_limit=1000, exchange_factory=lambda: fake,
    )

    assert len(df) == 1300
    # Two calls — first gets 1000, second gets the short page (300 < limit, stop).
    assert len(fake.calls) == 2


def test_fetch_ohlcv_dedupes_and_sorts():
    bar_ms = TIMEFRAMES["15m"]
    page_a = _bar_series(start_ms=1_640_995_200_000, n=5, bar_ms=bar_ms)
    # Overlap: page_b's first bar duplicates page_a's last.
    page_b = _bar_series(start_ms=page_a[-1][0], n=5, bar_ms=bar_ms)
    fake = FakeExchange(iter([page_a, page_b]))

    df = fetch_ohlcv(
        "binance", "BTC/USDT", interval="15m",
        page_limit=5, exchange_factory=lambda: fake,
    )

    # 5 + 5 - 1 overlap = 9, plus possible additional dedup
    assert df["open_ts"].is_unique
    assert df["open_ts"].is_monotonic_increasing


def test_fetch_ohlcv_respects_until():
    bar_ms = TIMEFRAMES["15m"]
    page = _bar_series(start_ms=1_640_995_200_000, n=20, bar_ms=bar_ms)
    fake = FakeExchange(iter([page]))

    until = pd.Timestamp(1_640_995_200_000 + 10 * bar_ms, unit="ms", tz="UTC")
    df = fetch_ohlcv(
        "binance", "BTC/USDT", interval="15m",
        until=until, page_limit=1000, exchange_factory=lambda: fake,
    )

    # Only bars strictly before `until` are kept.
    assert len(df) == 10
    assert df["open_ts"].max() < until


def test_fetch_ohlcv_empty_response_returns_empty_frame():
    fake = FakeExchange(iter([[]]))
    df = fetch_ohlcv(
        "binance", "BTC/USDT", interval="15m",
        exchange_factory=lambda: fake,
    )
    assert len(df) == 0
    assert list(df.columns) == OHLCV_COLS


def test_fetch_ohlcv_unknown_interval_raises():
    with pytest.raises(ValueError, match="unsupported interval"):
        fetch_ohlcv("binance", "BTC/USDT", interval="2m")


# ---------------------------------------------------------------------------
# compute_indicators
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2022-01-01", tz="UTC")
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "open_ts": [base + pd.Timedelta(minutes=15 * i) for i in range(n)],
            "open": opens,
            "high": np.maximum(opens, closes) + 0.5,
            "low": np.minimum(opens, closes) - 0.5,
            "close": closes,
            "volume": rng.uniform(10, 50, n),
        }
    )


def test_compute_indicators_adds_catalogue_columns():
    df = _ohlcv()
    out = compute_indicators(df)
    # ta library prefixes by class
    assert any(c.startswith("momentum_") for c in out.columns)
    assert any(c.startswith("trend_") for c in out.columns)
    assert any(c.startswith("volatility_") for c in out.columns)
    assert any(c.startswith("volume_") for c in out.columns)
    assert "trend_direction" in out.columns
    assert len(out) == len(df)


def test_compute_indicators_does_not_mutate_input():
    df = _ohlcv()
    original_cols = list(df.columns)
    _ = compute_indicators(df)
    assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# parquet round-trip
# ---------------------------------------------------------------------------

def test_parquet_round_trip_preserves_tz_and_values(tmp_path):
    df = _ohlcv(n=20)
    path = tmp_path / "bars.parquet"
    save_parquet(df, path)

    loaded = load_parquet(path)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded.reset_index(drop=True))
    assert loaded["open_ts"].dt.tz is not None


def test_parquet_save_creates_parent_directory(tmp_path):
    df = _ohlcv(n=5)
    nested = tmp_path / "deep" / "nested" / "dir" / "bars.parquet"
    save_parquet(df, nested)
    assert nested.exists()
    pd.testing.assert_frame_equal(df, load_parquet(nested))
