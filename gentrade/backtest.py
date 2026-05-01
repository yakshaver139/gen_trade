"""Realistic backtest engine: position state, no-overlap, fees, slippage.

Replaces the original `find_profit_in_window` evaluator, which evaluated
every entry signal independently — no concept of an open position, no fees,
and no slippage. The numbers it produced were not reachable by any real
account.

This engine walks the bars in order. When the entry signal fires *and* no
trade is open, a new long trade opens at the bar's `open` price (with
adverse slippage applied). The trade closes the first bar a forward `close`
crosses the target or stop level (target wins ties — they cannot occur with
strict inequalities), or at the end of the trade window if neither
threshold is reached. Exit prices have adverse slippage applied; per-side
fees are deducted from the realised return.

While a trade is open, every entry signal is ignored. After exit, the next
signal on or after the exit bar+1 may open a new trade.

Inputs
------
``bars``: an OHLCV DataFrame with at least ``open_ts``, ``open``, ``close``.
``entry_signal``: a boolean per-bar array/Series — True means "the strategy
fires on this bar". Length must match ``bars``.

Outputs
-------
A DataFrame with one row per trade and columns the metrics module
consumes — in particular ``return`` (net of fees + slippage), plus
diagnostic fields like ``outcome``, ``entry_price``, ``exit_price``,
``target_price``, and the bps figures used for the cost model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Outcome tokens. Match the spec's TradeOutcome enum.
TARGET_HIT = "TARGET_HIT"
STOPPED = "STOPPED_OUT"
NO_CLOSE = "NO_CLOSE_IN_WINDOW"


@dataclass(frozen=True)
class BacktestConfig:
    """Per-run cost & sizing parameters.

    Defaults are the spec's: ±1.5% / -0.75% on a 1-day window of 15-min bars,
    Binance taker fees (10 bps), majors-grade slippage (1 bp).
    """

    target_pct: float = 0.015
    stop_loss_pct: float = 0.0075
    trade_window_bars: int = 96
    taker_fee_bps: float = 10.0
    slippage_bps: float = 1.0


def simulate_trades(
    bars: pd.DataFrame,
    entry_signal: np.ndarray | pd.Series,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Simulate a long-only sequence of trades from per-bar entry signals."""
    cfg = config or BacktestConfig()
    bars = bars.reset_index(drop=True)
    n = len(bars)

    sig = np.asarray(entry_signal, dtype=bool)
    if sig.shape != (n,):
        raise ValueError(
            f"entry_signal length {sig.shape} does not match bars length ({n},)"
        )

    fee_pct = cfg.taker_fee_bps / 10_000.0
    slip = cfg.slippage_bps / 10_000.0
    rows: list[dict] = []

    i = 0
    while i < n:
        if not sig[i]:
            i += 1
            continue

        # No forward bars to evaluate target/stop in → cannot open this trade.
        if i + 1 >= n:
            break

        entry_bar = bars.iloc[i]
        entry_price_raw = float(entry_bar["open"])
        entry_price = entry_price_raw * (1 + slip)
        target = entry_price * (1 + cfg.target_pct)
        stop = entry_price * (1 - cfg.stop_loss_pct)

        last_idx = min(i + cfg.trade_window_bars, n - 1)

        outcome: str | None = None
        exit_idx = last_idx
        exit_price_raw = float(bars.iloc[last_idx]["close"])

        for j in range(i + 1, last_idx + 1):
            close_j = float(bars.iloc[j]["close"])
            # Target wins ties: a bar that crosses both within strict
            # inequalities cannot happen, but if data is degenerate we
            # prefer the optimistic interpretation only when target is
            # strictly cleared. Otherwise stop wins.
            if close_j >= target:
                outcome = TARGET_HIT
                exit_idx = j
                exit_price_raw = target
                break
            if close_j <= stop:
                outcome = STOPPED
                exit_idx = j
                exit_price_raw = stop
                break

        if outcome is None:
            outcome = NO_CLOSE

        exit_price = exit_price_raw * (1 - slip)
        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - 2 * fee_pct

        rows.append(
            {
                "entry_time": entry_bar["open_ts"],
                "entry_price": entry_price,
                "exit_time": bars.iloc[exit_idx]["open_ts"],
                "exit_price": exit_price,
                "target_price": target,
                "stop_loss_price": stop,
                "outcome": outcome,
                "return": net_return,
                "entry_fee_bps": cfg.taker_fee_bps,
                "exit_fee_bps": cfg.taker_fee_bps,
                "slippage_bps": cfg.slippage_bps,
            }
        )

        i = exit_idx + 1

    if not rows:
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
    return pd.DataFrame(rows)
