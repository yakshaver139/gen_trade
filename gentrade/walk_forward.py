"""Walk-forward orchestration: produce a BacktestReport for a chosen strategy.

The flow is:

1. The caller hands us a single bars frame with all indicator + ``_previous``
   columns already computed (shift on the *full* frame, then slice — slicing
   first leaves the first bar of every window with NaN for ``_previous`` and
   silently breaks comparisons).
2. We slice into train / validation / test windows by ``open_ts``.
3. We re-evaluate the chosen strategy on each window using the realistic
   backtest engine (``simulate_trades``) with fees and slippage.
4. We compute baselines on the test window only — buy-and-hold (one
   trade, entry to last close) and random-entry with the same exposure as
   the candidate. If the chosen strategy can't beat random with the same
   number of entries, it isn't a strategy.
5. We pack the lot into a ``BacktestReport``.

The ``overfitting_gap`` field is computed on per-trade ``expectancy`` —
the most directly comparable single number between train and validation.
A positive gap means train outperformed validation; a near-zero gap is
the goal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gentrade.backtest import BacktestConfig, simulate_trades
from gentrade.load_strategy import load_from_object_parenthesised, query_strategy
from gentrade.metrics import (
    DEFAULT_PERIODS_PER_YEAR,
    PerformanceMetrics,
    compute_metrics,
)


@dataclass(frozen=True)
class BacktestReport:
    """Single-strategy report mirroring the spec's BacktestReport entity.

    `per_generation` carries the in-sample-vs-validation curves so the
    divergence can be plotted from a single artefact. It defaults to an
    empty list when the report is built from `produce_backtest_report`
    standalone (no GA loop ran), and is populated by `run_ga` from its
    accumulated per-generation snapshots.
    """

    chosen_strategy_id: str
    train_metrics: PerformanceMetrics
    validation_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    buy_and_hold_test: PerformanceMetrics
    random_entry_test: PerformanceMetrics
    overfitting_gap: float
    per_generation: list[Any] = field(default_factory=list)


def slice_window(
    bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Return rows whose ``open_ts`` falls inside ``[start, end]`` inclusive."""
    mask = (bars["open_ts"] >= start) & (bars["open_ts"] <= end)
    return bars.loc[mask].reset_index(drop=True)


def evaluate_strategy(
    bars: pd.DataFrame, strategy: dict, config: BacktestConfig
) -> pd.DataFrame:
    """Build the entry signal from the strategy DSL and run the backtest."""
    if bars.empty:
        return simulate_trades(bars, np.array([], dtype=bool), config)
    parsed = load_from_object_parenthesised(strategy)
    matched = query_strategy(bars, query=parsed)
    matched_ts = set(matched["open_ts"].tolist())
    sig = bars["open_ts"].isin(matched_ts).to_numpy()
    return simulate_trades(bars, sig, config)


def buy_and_hold_trades(bars: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Synthesise the trade frame for a single buy-at-first / sell-at-last position.

    Built directly rather than going through ``simulate_trades`` because the
    backtest engine enforces the trade-window cap, and buy-and-hold is by
    definition window-length.
    """
    if bars.empty:
        return pd.DataFrame(
            columns=[
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                "target_price",
                "stop_loss_price",
                "outcome",
                "return",
                "entry_fee_bps",
                "exit_fee_bps",
                "slippage_bps",
            ]
        )
    slip = config.slippage_bps / 10_000.0
    fee_pct = config.taker_fee_bps / 10_000.0

    first, last = bars.iloc[0], bars.iloc[-1]
    entry_price = float(first["open"]) * (1 + slip)
    exit_price = float(last["close"]) * (1 - slip)
    gross_return = (exit_price - entry_price) / entry_price
    net_return = gross_return - 2 * fee_pct

    return pd.DataFrame(
        [
            {
                "entry_time": first["open_ts"],
                "entry_price": entry_price,
                "exit_time": last["open_ts"],
                "exit_price": exit_price,
                "target_price": math.nan,
                "stop_loss_price": math.nan,
                "outcome": "BUY_AND_HOLD",
                "return": net_return,
                "entry_fee_bps": config.taker_fee_bps,
                "exit_fee_bps": config.taker_fee_bps,
                "slippage_bps": config.slippage_bps,
            }
        ]
    )


def random_entry_trades(
    bars: pd.DataFrame,
    n_entries: int,
    config: BacktestConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample ``n_entries`` random bars (without replacement) and run the engine.

    The position-state engine then discards any sampled bar that falls inside
    an open trade's window, so the realised trade count is ≤ ``n_entries``.
    That asymmetry is acceptable — the baseline matches *exposure*, not
    necessarily an exact trade count.
    """
    n = len(bars)
    if n == 0 or n_entries <= 0:
        return simulate_trades(bars, np.zeros(n, dtype=bool), config)

    k = min(n_entries, n)
    idx = rng.choice(n, size=k, replace=False)
    sig = np.zeros(n, dtype=bool)
    sig[idx] = True
    return simulate_trades(bars, sig, config)


def compute_overfitting_gap(
    train: PerformanceMetrics, validation: PerformanceMetrics
) -> float:
    """Normalised gap on per-trade expectancy: ``(train - val) / |train|``.

    Positive ⇒ validation underperforms train (overfit).
    NaN when train expectancy is zero (no signal to compare against).
    """
    if train.expectancy == 0:
        return math.nan
    return (train.expectancy - validation.expectancy) / abs(train.expectancy)


def produce_backtest_report(
    bars: pd.DataFrame,
    chosen_strategy: dict,
    train_window: tuple[pd.Timestamp, pd.Timestamp],
    validation_window: tuple[pd.Timestamp, pd.Timestamp],
    test_window: tuple[pd.Timestamp, pd.Timestamp],
    config: BacktestConfig | None = None,
    seed: int = 0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    per_generation: list[Any] | None = None,
    overfitting_gap: float | None = None,
) -> BacktestReport:
    """Produce a single-strategy walk-forward report.

    ``bars`` must have indicator + ``_previous`` columns already populated.
    ``seed`` drives the random-entry baseline RNG for reproducibility.

    ``per_generation`` and ``overfitting_gap`` are caller-supplied when the
    GA loop has population-level data (preferred for spec compliance —
    overfitting gap is defined on max_fitness across the final generation).
    When omitted, the report is computed from the chosen strategy alone:
    per_generation is empty and overfitting_gap is the expectancy delta.
    """
    cfg = config or BacktestConfig()

    train_bars = slice_window(bars, *train_window)
    val_bars = slice_window(bars, *validation_window)
    test_bars = slice_window(bars, *test_window)

    train_trades = evaluate_strategy(train_bars, chosen_strategy, cfg)
    val_trades = evaluate_strategy(val_bars, chosen_strategy, cfg)
    test_trades = evaluate_strategy(test_bars, chosen_strategy, cfg)

    train_m = compute_metrics(train_trades, periods_per_year=periods_per_year)
    val_m = compute_metrics(val_trades, periods_per_year=periods_per_year)
    test_m = compute_metrics(test_trades, periods_per_year=periods_per_year)

    bh_test = compute_metrics(
        buy_and_hold_trades(test_bars, cfg), periods_per_year=periods_per_year
    )

    rng = np.random.default_rng(seed)
    random_test = compute_metrics(
        random_entry_trades(test_bars, n_entries=test_m.n_trades, config=cfg, rng=rng),
        periods_per_year=periods_per_year,
    )

    gap = (
        overfitting_gap
        if overfitting_gap is not None
        else compute_overfitting_gap(train_m, val_m)
    )

    return BacktestReport(
        chosen_strategy_id=chosen_strategy["id"],
        train_metrics=train_m,
        validation_metrics=val_m,
        test_metrics=test_m,
        buy_and_hold_test=bh_test,
        random_entry_test=random_test,
        overfitting_gap=gap,
        per_generation=list(per_generation) if per_generation else [],
    )
