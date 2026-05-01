"""Tests for the trading-performance metrics module (Phase 1).

Each metric is pinned against a hand-computed fixture so any later refactor
must explain its diff. Empty / degenerate fixtures (zero trades, zero
variance, all wins) are pinned too — those are the cases that break naive
implementations and silently produce NaN-laced reports.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from gentrade.metrics import (
    PerformanceMetrics,
    compute_avg_trade_duration,
    compute_calmar,
    compute_expectancy,
    compute_max_drawdown,
    compute_metrics,
    compute_profit_factor,
    compute_sharpe,
    compute_sortino,
    compute_win_rate,
)

PERIODS_PER_YEAR = 365  # crypto: 24/7 trading, 1-day trade window


def make_trades(returns: list[float], durations_hours: list[float] | None = None) -> pd.DataFrame:
    """Build a trades frame in the shape `compute_metrics` consumes."""
    if durations_hours is None:
        durations_hours = [24.0] * len(returns)
    base = pd.Timestamp("2022-01-01", tz="UTC")
    rows = []
    for i, (r, dh) in enumerate(zip(returns, durations_hours, strict=True)):
        entry = base + pd.Timedelta(days=i)
        exit_ = entry + pd.Timedelta(hours=dh)
        rows.append({"return": r, "entry_time": entry, "exit_time": exit_})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# win rate
# ---------------------------------------------------------------------------

def test_win_rate_simple():
    trades = make_trades([0.01, -0.02, 0.03, -0.01, 0.02])
    assert compute_win_rate(trades) == pytest.approx(3 / 5)


def test_win_rate_no_trades():
    assert compute_win_rate(make_trades([])) == 0.0


def test_win_rate_all_wins():
    assert compute_win_rate(make_trades([0.01, 0.02])) == 1.0


# ---------------------------------------------------------------------------
# profit factor
# ---------------------------------------------------------------------------

def test_profit_factor_simple():
    # gross_profit = 0.05 + 0.03 = 0.08; gross_loss = 0.02
    trades = make_trades([0.05, -0.02, 0.03])
    assert compute_profit_factor(trades) == pytest.approx(4.0)


def test_profit_factor_all_wins_is_inf():
    assert compute_profit_factor(make_trades([0.01, 0.02, 0.03])) == math.inf


def test_profit_factor_all_losses_is_zero():
    assert compute_profit_factor(make_trades([-0.01, -0.02])) == 0.0


def test_profit_factor_no_trades_is_zero():
    assert compute_profit_factor(make_trades([])) == 0.0


# ---------------------------------------------------------------------------
# expectancy
# ---------------------------------------------------------------------------

def test_expectancy_simple():
    trades = make_trades([0.05, -0.02, 0.03])
    assert compute_expectancy(trades) == pytest.approx(0.02)


def test_expectancy_no_trades():
    assert compute_expectancy(make_trades([])) == 0.0


# ---------------------------------------------------------------------------
# max drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_simple():
    # Equity from 1.0: 1.0, 1.10, 0.88, 0.924
    # Running peak: 1.0, 1.10, 1.10, 1.10
    # DDs: 0, 0, -0.20, -0.16 — max is -0.20
    trades = make_trades([0.10, -0.20, 0.05])
    assert compute_max_drawdown(trades) == pytest.approx(-0.20, abs=1e-9)


def test_max_drawdown_all_winning_is_zero():
    assert compute_max_drawdown(make_trades([0.01, 0.02, 0.03])) == 0.0


def test_max_drawdown_no_trades_is_zero():
    assert compute_max_drawdown(make_trades([])) == 0.0


def test_max_drawdown_recovers_then_drops_again():
    # 1, 1.5, 1.2, 1.4, 0.7
    # peak: 1, 1.5, 1.5, 1.5, 1.5
    # dd: 0, 0, -0.20, -0.0667, -0.5333 — max is the final
    trades = make_trades([0.5, -0.2, 1 / 6, -0.5])
    assert compute_max_drawdown(trades) == pytest.approx(-(1 - 0.7 / 1.5), abs=1e-9)


# ---------------------------------------------------------------------------
# sharpe
# ---------------------------------------------------------------------------

def test_sharpe_known_value():
    # returns: mean=0.012, sample-std=0.025884, sharpe_per_period=0.4636
    # annualized at 365 = 0.4636 * sqrt(365) ≈ 8.86
    trades = make_trades([0.02, -0.01, 0.03, -0.02, 0.04])
    sharpe = compute_sharpe(trades, periods_per_year=PERIODS_PER_YEAR)
    assert sharpe == pytest.approx(8.86, rel=5e-3)


def test_sharpe_constant_positive_returns_is_inf():
    # zero variance with a positive mean → infinite Sharpe
    sharpe = compute_sharpe(make_trades([0.01, 0.01, 0.01]), periods_per_year=PERIODS_PER_YEAR)
    assert sharpe == math.inf


def test_sharpe_zero_returns_is_nan():
    sharpe = compute_sharpe(make_trades([0.0, 0.0, 0.0]), periods_per_year=PERIODS_PER_YEAR)
    assert math.isnan(sharpe)


def test_sharpe_no_trades_is_nan():
    sharpe = compute_sharpe(make_trades([]), periods_per_year=PERIODS_PER_YEAR)
    assert math.isnan(sharpe)


# ---------------------------------------------------------------------------
# sortino
# ---------------------------------------------------------------------------

def test_sortino_uses_downside_only():
    """Asymmetric returns: large upside variance, small downside → sortino > sharpe."""
    trades = make_trades([-0.01, 0.05, 0.10, 0.01, 0.02])
    sharpe = compute_sharpe(trades, periods_per_year=PERIODS_PER_YEAR)
    sortino = compute_sortino(trades, periods_per_year=PERIODS_PER_YEAR)
    assert sortino > sharpe


def test_sortino_all_winners_is_inf():
    sortino = compute_sortino(make_trades([0.01, 0.02, 0.03]), periods_per_year=PERIODS_PER_YEAR)
    assert sortino == math.inf


def test_sortino_no_trades_is_nan():
    assert math.isnan(compute_sortino(make_trades([]), periods_per_year=PERIODS_PER_YEAR))


# ---------------------------------------------------------------------------
# calmar
# ---------------------------------------------------------------------------

def test_calmar_positive_when_returns_positive():
    trades = make_trades([0.02, -0.01, 0.03, 0.01, 0.02])
    calmar = compute_calmar(trades, periods_per_year=PERIODS_PER_YEAR)
    assert calmar > 0


def test_calmar_no_drawdown_is_inf():
    trades = make_trades([0.01, 0.02, 0.03])
    assert compute_calmar(trades, periods_per_year=PERIODS_PER_YEAR) == math.inf


def test_calmar_no_trades_is_nan():
    assert math.isnan(compute_calmar(make_trades([]), periods_per_year=PERIODS_PER_YEAR))


# ---------------------------------------------------------------------------
# trade duration
# ---------------------------------------------------------------------------

def test_avg_trade_duration_simple():
    trades = make_trades([0.01, 0.02, 0.03], durations_hours=[24, 48, 72])
    assert compute_avg_trade_duration(trades) == pd.Timedelta(hours=48)


def test_avg_trade_duration_no_trades_is_zero():
    assert compute_avg_trade_duration(make_trades([])) == pd.Timedelta(0)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def test_compute_metrics_returns_dataclass_with_all_fields():
    trades = make_trades([0.05, -0.02, 0.03], durations_hours=[24, 24, 48])
    m = compute_metrics(trades, periods_per_year=PERIODS_PER_YEAR)
    assert isinstance(m, PerformanceMetrics)
    assert m.n_trades == 3
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.profit_factor == pytest.approx(4.0)
    assert m.expectancy == pytest.approx(0.02)
    assert m.avg_trade_duration == pd.Timedelta(hours=32)


def test_compute_metrics_handles_empty():
    m = compute_metrics(make_trades([]), periods_per_year=PERIODS_PER_YEAR)
    assert m.n_trades == 0
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0
    assert m.expectancy == 0.0
    assert m.max_drawdown == 0.0
    assert m.avg_trade_duration == pd.Timedelta(0)
    # ratios of nothing are NaN — the report consumer must handle this
    assert math.isnan(m.sharpe)
    assert math.isnan(m.sortino)
    assert math.isnan(m.calmar)
