"""Trading-performance metrics: Sharpe, Sortino, Calmar, drawdown, etc.

Inputs are a per-trade DataFrame with columns ``return`` (decimal P&L per
trade, after fees and slippage), ``entry_time``, and ``exit_time``. Callers
are expected to assemble that frame from a Backtest's Trades.

Annualisation uses ``periods_per_year`` — for crypto with a 1-day trade
window the natural value is 365. For daily equities, 252.

Edge-case behaviour is deliberate:
- Empty trades → counts and sums collapse to 0; ratios that depend on
  variance return NaN (no information). Consumers must handle NaN.
- Zero-variance positive returns → Sharpe = +∞ (perfectly riskless gain).
- All-winning returns → Sortino = +∞, profit factor = +∞.
- All-losing → profit factor = 0, max drawdown reflects the equity collapse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_PERIODS_PER_YEAR = 365


@dataclass(frozen=True)
class PerformanceMetrics:
    """Container for the headline metrics; mirrors `PerformanceMetrics` in the spec."""

    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    avg_trade_duration: pd.Timedelta


def _returns(trades: pd.DataFrame) -> np.ndarray:
    if trades.empty or "return" not in trades.columns:
        return np.array([], dtype=float)
    return trades["return"].to_numpy(dtype=float)


def compute_win_rate(trades: pd.DataFrame) -> float:
    r = _returns(trades)
    if r.size == 0:
        return 0.0
    return float((r > 0).sum() / r.size)


def compute_profit_factor(trades: pd.DataFrame) -> float:
    r = _returns(trades)
    if r.size == 0:
        return 0.0
    gross_profit = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_expectancy(trades: pd.DataFrame) -> float:
    r = _returns(trades)
    if r.size == 0:
        return 0.0
    return float(r.mean())


def _equity_curve(returns: np.ndarray) -> np.ndarray:
    """Cumulative equity from 1.0 under multiplicative per-trade returns."""
    if returns.size == 0:
        return np.array([1.0])
    return np.concatenate([[1.0], np.cumprod(1.0 + returns)])


def compute_max_drawdown(trades: pd.DataFrame) -> float:
    """Most negative peak-to-trough drawdown of the equity curve.

    Returns a value in (-1, 0]. 0 means the curve never dipped below its
    running peak; -1 would mean total wipeout.
    """
    r = _returns(trades)
    eq = _equity_curve(r)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    return float(dd.min())


def compute_sharpe(
    trades: pd.DataFrame, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> float:
    r = _returns(trades)
    if r.size == 0:
        return math.nan
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if r.size > 1 else 0.0
    if std == 0:
        if mean == 0:
            return math.nan
        return math.inf if mean > 0 else -math.inf
    return (mean / std) * math.sqrt(periods_per_year)


def compute_sortino(
    trades: pd.DataFrame, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> float:
    r = _returns(trades)
    if r.size == 0:
        return math.nan
    mean = float(r.mean())
    downside = r[r < 0]
    if downside.size == 0:
        if mean > 0:
            return math.inf
        return 0.0 if mean == 0 else -math.inf
    # Downside deviation: RMS of negative returns relative to zero.
    dstd = float(np.sqrt((downside**2).mean()))
    if dstd == 0:
        return math.nan
    return (mean / dstd) * math.sqrt(periods_per_year)


def compute_calmar(
    trades: pd.DataFrame, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> float:
    r = _returns(trades)
    if r.size == 0:
        return math.nan
    eq = _equity_curve(r)
    total_return = eq[-1] / eq[0] - 1.0
    if total_return <= -1.0:
        # Total wipeout — annualised return is undefined.
        return -math.inf
    annualised = (1.0 + total_return) ** (periods_per_year / r.size) - 1.0
    mdd = abs(compute_max_drawdown(trades))
    if mdd == 0:
        return math.inf if annualised > 0 else 0.0 if annualised == 0 else -math.inf
    return annualised / mdd


def compute_avg_trade_duration(trades: pd.DataFrame) -> pd.Timedelta:
    if trades.empty:
        return pd.Timedelta(0)
    durations = trades["exit_time"] - trades["entry_time"]
    return durations.mean()


def compute_metrics(
    trades: pd.DataFrame, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> PerformanceMetrics:
    return PerformanceMetrics(
        n_trades=len(trades),
        win_rate=compute_win_rate(trades),
        profit_factor=compute_profit_factor(trades),
        expectancy=compute_expectancy(trades),
        sharpe=compute_sharpe(trades, periods_per_year),
        sortino=compute_sortino(trades, periods_per_year),
        calmar=compute_calmar(trades, periods_per_year),
        max_drawdown=compute_max_drawdown(trades),
        avg_trade_duration=compute_avg_trade_duration(trades),
    )
