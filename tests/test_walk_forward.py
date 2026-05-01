"""Tests for the walk-forward orchestrator (Phase 1).

Pins the contract for window slicing, per-window strategy evaluation,
buy-and-hold + random-entry baselines, and the end-to-end BacktestReport.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gentrade.backtest import NO_CLOSE, BacktestConfig
from gentrade.walk_forward import (
    BacktestReport,
    buy_and_hold_trades,
    compute_overfitting_gap,
    evaluate_strategy,
    produce_backtest_report,
    random_entry_trades,
    slice_window,
)


def make_bars(n: int, start_price: float = 100.0, drift: float = 0.0) -> pd.DataFrame:
    """OHLCV with optional per-bar drift, fixed timestamps, no noise."""
    base = pd.Timestamp("2022-01-01", tz="UTC")
    open_ts = [base + pd.Timedelta(minutes=15 * i) for i in range(n)]
    closes = [start_price * (1 + drift) ** i for i in range(n)]
    opens = ([start_price] + closes[:-1]) if n else []
    return pd.DataFrame(
        {
            "open_ts": open_ts,
            "open": opens,
            "high": [max(o, c) + 0.01 for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) - 0.01 for o, c in zip(opens, closes, strict=True)],
            "close": closes,
            # absolute-strategy target column for query-DSL tests
            "close_marker": closes,
        }
    )


CFG = BacktestConfig(
    target_pct=0.015,
    stop_loss_pct=0.0075,
    trade_window_bars=4,
    taker_fee_bps=10.0,
    slippage_bps=1.0,
)


# ---------------------------------------------------------------------------
# slice_window
# ---------------------------------------------------------------------------

def test_slice_window_returns_inclusive_range():
    bars = make_bars(10)
    start = bars.iloc[2]["open_ts"]
    end = bars.iloc[5]["open_ts"]

    out = slice_window(bars, start, end)

    assert len(out) == 4
    assert out.iloc[0]["open_ts"] == start
    assert out.iloc[-1]["open_ts"] == end


def test_slice_window_handles_empty_window():
    bars = make_bars(10)
    far_start = pd.Timestamp("2030-01-01", tz="UTC")
    far_end = pd.Timestamp("2030-01-02", tz="UTC")

    out = slice_window(bars, far_start, far_end)

    assert len(out) == 0


# ---------------------------------------------------------------------------
# evaluate_strategy (DSL → backtest)
# ---------------------------------------------------------------------------

def test_evaluate_strategy_runs_query_and_emits_trades():
    """An always-true absolute filter on a rising series should hit target on each cycle."""
    bars = make_bars(20, drift=0.005)  # +0.5% per bar → target reached fast
    strategy = {
        "id": "always-on",
        # close_marker >= 0 is always true
        "indicators": [
            {
                "absolute": True,
                "indicator": "close_marker",
                "op": ">=",
                "abs_value": 0,
            }
        ],
        "conjunctions": [],
    }

    trades = evaluate_strategy(bars, strategy, CFG)

    assert len(trades) >= 1
    # rising series → all closed trades are TARGET_HIT
    assert (trades["outcome"] != NO_CLOSE).any()
    # no two trades overlap
    for i in range(1, len(trades)):
        prev_exit = trades.iloc[i - 1]["exit_time"]
        curr_entry = trades.iloc[i]["entry_time"]
        assert curr_entry > prev_exit


def test_evaluate_strategy_swallows_missing_column_errors():
    """A strategy referencing columns the frame doesn't have → empty trades, no crash."""
    bars = make_bars(20)
    strategy = {
        "id": "missing-col",
        "indicators": [
            {
                "absolute": True,
                "indicator": "momentum_does_not_exist",
                "op": ">=",
                "abs_value": 50,
            }
        ],
        "conjunctions": [],
    }
    trades = evaluate_strategy(bars, strategy, CFG)
    assert len(trades) == 0


def test_evaluate_strategy_with_never_true_signal_returns_empty():
    bars = make_bars(20)
    strategy = {
        "id": "never-on",
        "indicators": [
            {
                "absolute": True,
                "indicator": "close_marker",
                "op": "<=",
                "abs_value": -1,
            }
        ],
        "conjunctions": [],
    }

    trades = evaluate_strategy(bars, strategy, CFG)

    assert len(trades) == 0


# ---------------------------------------------------------------------------
# buy_and_hold
# ---------------------------------------------------------------------------

def test_buy_and_hold_trades_emits_single_long_trade():
    bars = make_bars(20, drift=0.005)
    trades = buy_and_hold_trades(bars, CFG)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_time"] == bars.iloc[0]["open_ts"]
    assert t["exit_time"] == bars.iloc[-1]["open_ts"]
    # gross gain ~ (1.005^19 - 1) ≈ 9.96%; net subtracts 2 * 10bp + slippage
    assert t["return"] > 0


def test_buy_and_hold_trades_handles_empty_window():
    empty = make_bars(0)
    trades = buy_and_hold_trades(empty, CFG)
    assert len(trades) == 0


# ---------------------------------------------------------------------------
# random_entry
# ---------------------------------------------------------------------------

def test_random_entry_trades_seed_reproducible():
    bars = make_bars(50, drift=0.001)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    trades_a = random_entry_trades(bars, n_entries=5, config=CFG, rng=rng_a)
    trades_b = random_entry_trades(bars, n_entries=5, config=CFG, rng=rng_b)

    assert trades_a["entry_time"].tolist() == trades_b["entry_time"].tolist()
    assert trades_a["return"].tolist() == trades_b["return"].tolist()


def test_random_entry_trades_handles_zero_entries():
    bars = make_bars(20)
    trades = random_entry_trades(bars, n_entries=0, config=CFG, rng=np.random.default_rng(0))
    assert len(trades) == 0


def test_random_entry_trades_handles_empty_bars():
    empty = make_bars(0)
    trades = random_entry_trades(
        empty, n_entries=5, config=CFG, rng=np.random.default_rng(0)
    )
    assert len(trades) == 0


# ---------------------------------------------------------------------------
# overfitting gap
# ---------------------------------------------------------------------------

def test_overfitting_gap_zero_when_metrics_match():
    """Train and validation expectancies equal → no gap."""
    from gentrade.metrics import PerformanceMetrics

    m = PerformanceMetrics(
        n_trades=5,
        win_rate=0.6,
        profit_factor=2.0,
        expectancy=0.012,
        sharpe=1.5,
        sortino=2.0,
        calmar=1.0,
        max_drawdown=-0.05,
        avg_trade_duration=pd.Timedelta(hours=24),
    )

    assert compute_overfitting_gap(m, m) == pytest.approx(0.0)


def test_overfitting_gap_positive_when_validation_underperforms():
    from gentrade.metrics import PerformanceMetrics

    train = PerformanceMetrics(
        n_trades=5, win_rate=0.6, profit_factor=2.0, expectancy=0.020,
        sharpe=1.5, sortino=2.0, calmar=1.0, max_drawdown=-0.05,
        avg_trade_duration=pd.Timedelta(hours=24),
    )
    val = PerformanceMetrics(
        n_trades=5, win_rate=0.4, profit_factor=1.0, expectancy=0.010,
        sharpe=0.5, sortino=0.7, calmar=0.5, max_drawdown=-0.10,
        avg_trade_duration=pd.Timedelta(hours=24),
    )

    gap = compute_overfitting_gap(train, val)
    assert gap == pytest.approx(0.5, rel=1e-9)  # (0.02 - 0.01) / |0.02|


def test_overfitting_gap_handles_zero_train_expectancy():
    """No information from train → gap is NaN, not a divide-by-zero crash."""
    from gentrade.metrics import PerformanceMetrics

    z = PerformanceMetrics(
        n_trades=0, win_rate=0.0, profit_factor=0.0, expectancy=0.0,
        sharpe=math.nan, sortino=math.nan, calmar=math.nan, max_drawdown=0.0,
        avg_trade_duration=pd.Timedelta(0),
    )

    assert math.isnan(compute_overfitting_gap(z, z))


# ---------------------------------------------------------------------------
# end-to-end report
# ---------------------------------------------------------------------------

def test_produce_backtest_report_populates_all_windows():
    bars = make_bars(60, drift=0.002)
    strategy = {
        "id": "always-on",
        "indicators": [
            {
                "absolute": True,
                "indicator": "close_marker",
                "op": ">=",
                "abs_value": 0,
            }
        ],
        "conjunctions": [],
    }
    train = (bars.iloc[0]["open_ts"], bars.iloc[19]["open_ts"])
    val = (bars.iloc[20]["open_ts"], bars.iloc[39]["open_ts"])
    test = (bars.iloc[40]["open_ts"], bars.iloc[59]["open_ts"])

    report = produce_backtest_report(
        bars=bars,
        chosen_strategy=strategy,
        train_window=train,
        validation_window=val,
        test_window=test,
        config=CFG,
        seed=42,
    )

    assert isinstance(report, BacktestReport)
    assert report.chosen_strategy_id == "always-on"
    assert report.train_metrics.n_trades > 0
    assert report.validation_metrics.n_trades > 0
    assert report.test_metrics.n_trades > 0
    # baselines on the test window
    assert report.buy_and_hold_test.n_trades == 1
    # random entry uses the candidate's exposure
    assert report.random_entry_test.n_trades >= 0


def test_produce_backtest_report_random_baseline_seed_reproducible():
    bars = make_bars(60, drift=0.001)
    strategy = {
        "id": "always-on",
        "indicators": [
            {
                "absolute": True,
                "indicator": "close_marker",
                "op": ">=",
                "abs_value": 0,
            }
        ],
        "conjunctions": [],
    }
    train = (bars.iloc[0]["open_ts"], bars.iloc[19]["open_ts"])
    val = (bars.iloc[20]["open_ts"], bars.iloc[39]["open_ts"])
    test = (bars.iloc[40]["open_ts"], bars.iloc[59]["open_ts"])

    a = produce_backtest_report(bars, strategy, train, val, test, CFG, seed=42)
    b = produce_backtest_report(bars, strategy, train, val, test, CFG, seed=42)

    assert a.random_entry_test.expectancy == b.random_entry_test.expectancy
    assert a.random_entry_test.n_trades == b.random_entry_test.n_trades


def test_produce_backtest_report_train_metrics_only_see_train_window():
    """A strategy that fires only outside the train window has 0 train trades."""
    bars = make_bars(60, drift=0.002)
    # Set close_marker to a sentinel value only on bars >= 30
    bars.loc[bars.index >= 30, "close_marker"] = 9999
    bars.loc[bars.index < 30, "close_marker"] = -1

    strategy = {
        "id": "late-firing",
        "indicators": [
            {
                "absolute": True,
                "indicator": "close_marker",
                "op": ">=",
                "abs_value": 100,
            }
        ],
        "conjunctions": [],
    }
    train = (bars.iloc[0]["open_ts"], bars.iloc[19]["open_ts"])
    val = (bars.iloc[20]["open_ts"], bars.iloc[39]["open_ts"])
    test = (bars.iloc[40]["open_ts"], bars.iloc[59]["open_ts"])

    report = produce_backtest_report(bars, strategy, train, val, test, CFG, seed=0)

    assert report.train_metrics.n_trades == 0
    assert report.validation_metrics.n_trades >= 1
    assert report.test_metrics.n_trades >= 1
