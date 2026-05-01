"""Tests for the realistic backtest engine (Phase 1).

The original engine evaluated every entry signal independently, with no
position state, no fees, and no slippage. This module replaces that with:

- Position state: at most one trade open at a time per strategy. Signals
  that fire while a trade is open are ignored.
- Fees: per-side taker fee in basis points, default 10 bps (Binance spot).
- Slippage: per-side, applied adversely (entry pays slightly more, exit
  receives slightly less).

Each test pins the contract against a small synthetic OHLC frame with
hand-computed expected outcomes. The fixtures use a tiny `trade_window`
(4 bars) so every test stays legible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gentrade.backtest import (
    NO_CLOSE,
    STOPPED,
    TARGET_HIT,
    BacktestConfig,
    simulate_trades,
)


def make_bars(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    """Build a tiny OHLC frame indexed by minute.

    `opens` defaults to lagging close-from-previous (or `closes[0]` for bar 0)
    so HLC are consistent enough for tests that don't probe high/low.
    """
    n = len(closes)
    if opens is None:
        opens = [closes[0]] + closes[:-1]
    base = pd.Timestamp("2022-01-01", tz="UTC")
    open_ts = [base + pd.Timedelta(minutes=15 * i) for i in range(n)]
    return pd.DataFrame(
        {
            "open_ts": open_ts,
            "open": opens,
            "high": [max(o, c) + 0.01 for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) - 0.01 for o, c in zip(opens, closes, strict=True)],
            "close": closes,
        }
    )


def signals_at(n: int, indices: list[int]) -> np.ndarray:
    sig = np.zeros(n, dtype=bool)
    sig[indices] = True
    return sig


CFG_BASIC = BacktestConfig(
    target_pct=0.015,
    stop_loss_pct=0.0075,
    trade_window_bars=4,
    taker_fee_bps=10.0,
    slippage_bps=1.0,
)
CFG_NOFRICTION = BacktestConfig(
    target_pct=0.015,
    stop_loss_pct=0.0075,
    trade_window_bars=4,
    taker_fee_bps=0.0,
    slippage_bps=0.0,
)


# ---------------------------------------------------------------------------
# headline outcomes
# ---------------------------------------------------------------------------

def test_target_hit_emits_one_trade_with_friction_adjusted_return():
    """Bar 0 entry at 100; bar 3 close clears target; HIT.

    Expected:
      raw target  = 100.01 * 1.015              = 101.51015
      entry_price = 100 * (1 + 1bp)             = 100.01
      exit_price  = 101.51015 * (1 - 1bp)       = 101.4986...
      gross       = (exit - entry) / entry      ≈ 0.014880
      net         = gross - 2 * 10bp (fees)     ≈ 0.012880
    """
    bars = make_bars([100, 100.5, 101.0, 102.0, 102.5])
    sig = signals_at(len(bars), [0])

    trades = simulate_trades(bars, sig, CFG_BASIC)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["outcome"] == TARGET_HIT
    expected_entry = 100 * (1 + 1e-4)
    expected_target = expected_entry * 1.015
    expected_exit = expected_target * (1 - 1e-4)
    expected_gross = (expected_exit - expected_entry) / expected_entry
    expected_net = expected_gross - 2 * 10e-4
    assert t["return"] == pytest.approx(expected_net, rel=1e-6)
    assert t["entry_price"] == pytest.approx(expected_entry, rel=1e-9)
    assert t["target_price"] == pytest.approx(expected_target, rel=1e-9)


def test_stopped_out_returns_negative_more_than_stop_loss():
    """Slippage + fees mean a stop-out costs more than the headline stop_loss_pct."""
    bars = make_bars([100, 99.5, 99.0, 98.0, 97.5])
    sig = signals_at(len(bars), [0])

    trades = simulate_trades(bars, sig, CFG_BASIC)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["outcome"] == STOPPED
    # net loss is more than 0.0075 (the headline stop)
    assert t["return"] < -0.0075


def test_no_close_in_window_exits_at_window_end_close():
    """Sideways bars: trade times out after trade_window_bars and exits at last close."""
    bars = make_bars([100, 100.1, 100.2, 100.3, 100.4, 100.5])
    sig = signals_at(len(bars), [0])

    trades = simulate_trades(bars, sig, CFG_BASIC)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["outcome"] == NO_CLOSE
    # exit time is at i + trade_window_bars (= 4)
    assert t["exit_time"] == bars.iloc[4]["open_ts"]


# ---------------------------------------------------------------------------
# position state / no-overlap
# ---------------------------------------------------------------------------

def test_signal_during_open_trade_is_ignored():
    """Signal at bar 0 opens; signal at bar 2 (still open) must be skipped;
    signal at bar 4 (after trade closed at bar 3) opens a second trade."""
    bars = make_bars(
        # 0    1    2    3       4    5    6    7
        [100, 100.5, 101.0, 102.0, 102.5, 103.0, 104.5, 105.0]
    )
    sig = signals_at(len(bars), [0, 2, 4])

    trades = simulate_trades(bars, sig, CFG_BASIC)

    # only the entry at bar 0 and the entry at bar 4 should fire
    assert len(trades) == 2
    assert trades.iloc[0]["entry_time"] == bars.iloc[0]["open_ts"]
    assert trades.iloc[1]["entry_time"] == bars.iloc[4]["open_ts"]


def test_no_overlap_within_window_then_re_entry_on_next_signal_after_exit():
    """After exit at bar 3 (target hit), the next signal we honour is on bar 4 onwards."""
    bars = make_bars([100, 100.5, 101.0, 102.0, 99.5, 99.0, 98.0])
    # signal fires every bar
    sig = np.ones(len(bars), dtype=bool)

    trades = simulate_trades(bars, sig, CFG_BASIC)

    # bar 0 opens (target hit at bar 3) → bar 4 reopens (stop hit at bar 6)
    assert len(trades) == 2
    assert trades.iloc[0]["outcome"] == TARGET_HIT
    assert trades.iloc[0]["entry_time"] == bars.iloc[0]["open_ts"]
    assert trades.iloc[1]["entry_time"] == bars.iloc[4]["open_ts"]


# ---------------------------------------------------------------------------
# resolution order
# ---------------------------------------------------------------------------

def test_first_threshold_to_close_wins():
    """If stop is hit on bar 1 and target on bar 2, the trade is STOPPED at bar 1."""
    bars = make_bars([100, 99.0, 102.0, 102.0, 102.0])  # bar 1 stops, bar 2 would target
    sig = signals_at(len(bars), [0])

    trades = simulate_trades(bars, sig, CFG_BASIC)

    assert len(trades) == 1
    assert trades.iloc[0]["outcome"] == STOPPED
    assert trades.iloc[0]["exit_time"] == bars.iloc[1]["open_ts"]


# ---------------------------------------------------------------------------
# friction model
# ---------------------------------------------------------------------------

def test_zero_friction_recovers_headline_target_pct():
    """With fees=0 and slippage=0, a HIT trade returns exactly target_pct."""
    bars = make_bars([100, 100.5, 101.0, 102.0])
    sig = signals_at(len(bars), [0])

    trades = simulate_trades(bars, sig, CFG_NOFRICTION)

    assert len(trades) == 1
    assert trades.iloc[0]["outcome"] == TARGET_HIT
    assert trades.iloc[0]["return"] == pytest.approx(CFG_NOFRICTION.target_pct, rel=1e-9)


def test_zero_friction_stopped_recovers_headline_stop_pct():
    bars = make_bars([100, 98.0, 98.0, 98.0])
    sig = signals_at(len(bars), [0])

    trades = simulate_trades(bars, sig, CFG_NOFRICTION)

    assert len(trades) == 1
    assert trades.iloc[0]["outcome"] == STOPPED
    assert trades.iloc[0]["return"] == pytest.approx(-CFG_NOFRICTION.stop_loss_pct, rel=1e-9)


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------

def test_no_signals_no_trades():
    bars = make_bars([100, 101, 102, 103])
    trades = simulate_trades(bars, signals_at(len(bars), []), CFG_BASIC)
    assert len(trades) == 0


def test_signal_at_last_bar_with_no_window_left_does_not_emit():
    """A signal at the very last bar has no forward bars to evaluate."""
    bars = make_bars([100, 100, 100, 100])
    sig = signals_at(len(bars), [3])
    trades = simulate_trades(bars, sig, CFG_BASIC)
    assert len(trades) == 0


def test_signal_near_end_uses_truncated_window():
    """A signal close to end-of-frame still runs, with a truncated window."""
    bars = make_bars([100, 100.1, 100.2])
    # window = 4 bars but only 2 forward bars are available
    sig = signals_at(len(bars), [0])
    trades = simulate_trades(bars, sig, CFG_BASIC)
    # No threshold reached in 2 bars of sideways → NO_CLOSE
    assert len(trades) == 1
    assert trades.iloc[0]["outcome"] == NO_CLOSE
    assert trades.iloc[0]["exit_time"] == bars.iloc[2]["open_ts"]


def test_returns_have_metrics_module_compatible_columns():
    """The trades frame must be consumable by gentrade.metrics.compute_metrics."""
    from gentrade.metrics import compute_metrics

    bars = make_bars([100, 100.5, 101.0, 102.0, 99.0, 98.0])
    sig = np.ones(len(bars), dtype=bool)
    trades = simulate_trades(bars, sig, CFG_BASIC)

    m = compute_metrics(trades, periods_per_year=365)
    assert m.n_trades == len(trades)
    # one HIT, one STOPPED → win rate is 1/2
    assert m.win_rate == pytest.approx(0.5)
