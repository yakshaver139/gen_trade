"""Phase 1 modelling regression suite.

A fixed seed + synthetic dataset + known strategy produces a fixed
BacktestReport. Any future change to selection, metrics, the backtest
engine, or the walk-forward orchestrator must explain its diff against
these pinned numbers.

Run ad-hoc with: ``uv run pytest --tb=line tests/regression``.

These tests are deliberately:
- Deterministic (no RNG outside the seeded baseline path).
- Cheap (under 1 second total).
- Synthetic (no exchange data dependency).
- Pinned at full precision so float drift is visible immediately.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gentrade.backtest import BacktestConfig
from gentrade.walk_forward import produce_backtest_report

CFG = BacktestConfig(
    target_pct=0.015,
    stop_loss_pct=0.0075,
    trade_window_bars=4,
    fee_bps=10.0,
    slippage_bps=1.0,
)
SEED = 1337


def _baseline_bars() -> pd.DataFrame:
    """Sawtooth price series: 100 → 101.8 over 10 bars, repeat 9 times = 90 bars.

    Predictable enough that hand-checking is feasible; varied enough that
    every regression metric carries information.
    """
    n = 90
    base = pd.Timestamp("2022-01-01", tz="UTC")
    open_ts = [base + pd.Timedelta(minutes=15 * i) for i in range(n)]
    closes = np.array([100.0 + (i % 10) * 0.2 for i in range(n)])
    opens = np.concatenate([[closes[0]], closes[:-1]])

    return pd.DataFrame(
        {
            "open_ts": open_ts,
            "open": opens,
            "high": np.maximum(opens, closes) + 0.01,
            "low": np.minimum(opens, closes) - 0.01,
            "close": closes,
        }
    )


_STRATEGY = {
    "id": "phase1-regression",
    # fires whenever close clears the sawtooth's mid-band
    "indicators": [
        {
            "absolute": True,
            "indicator": "close",
            "op": ">=",
            "abs_value": 101.0,
        }
    ],
    "conjunctions": [],
}


def _run_baseline_report():
    bars = _baseline_bars()
    train = (bars.iloc[0]["open_ts"], bars.iloc[29]["open_ts"])
    val = (bars.iloc[30]["open_ts"], bars.iloc[59]["open_ts"])
    test = (bars.iloc[60]["open_ts"], bars.iloc[89]["open_ts"])
    return produce_backtest_report(
        bars=bars,
        chosen_strategy=_STRATEGY,
        train_window=train,
        validation_window=val,
        test_window=test,
        config=CFG,
        seed=SEED,
    )


# ---------------------------------------------------------------------------
# Pinned report — exact-precision regression
# ---------------------------------------------------------------------------

def test_train_window_metrics_pinned():
    r = _run_baseline_report()
    m = r.train_metrics
    # Three entries (one per sawtooth cycle); a fourth signal at bar 29 has
    # no forward bars left in the train slice and is skipped.
    assert m.n_trades == 3
    assert m.win_rate == pytest.approx(1.0)
    # Each trade times out (NO_CLOSE) but with a positive close-to-open delta.
    assert m.expectancy == pytest.approx(0.007718670990043777, rel=1e-9)
    assert m.profit_factor == math.inf
    # All-winning equity curve never dips below its running peak.
    assert m.max_drawdown == pytest.approx(0.0)


def test_validation_window_metrics_pinned():
    r = _run_baseline_report()
    m = r.validation_metrics
    # Identical sawtooth shape across windows → identical metrics.
    assert m.n_trades == 3
    assert m.expectancy == pytest.approx(0.007718670990043777, rel=1e-9)


def test_test_window_metrics_pinned():
    r = _run_baseline_report()
    m = r.test_metrics
    assert m.n_trades == 3
    assert m.expectancy == pytest.approx(0.007718670990043777, rel=1e-9)


def test_buy_and_hold_baseline_on_test_window_pinned():
    r = _run_baseline_report()
    bh = r.buy_and_hold_test
    assert bh.n_trades == 1
    # Test window opens at peak of cycle (close=101.8) and closes at the same
    # level → the round trip is a pure friction loss (slippage + fees ≈ -22bp).
    assert bh.expectancy == pytest.approx(-0.002199980001999906, rel=1e-9)


def test_random_entry_baseline_seed_pinned():
    r = _run_baseline_report()
    rand = r.random_entry_test
    # Seed=1337 → 3 sampled bars but two collapse into one trade under
    # the no-overlap rule, so 2 realised trades.
    assert rand.n_trades == 2
    assert rand.expectancy == pytest.approx(0.007758345594011993, rel=1e-9)


def test_overfitting_gap_pinned():
    r = _run_baseline_report()
    # train and validation expectancies are equal on the sawtooth → gap = 0
    assert r.overfitting_gap == pytest.approx(0.0, abs=1e-12)


def test_chosen_strategy_id_passes_through():
    r = _run_baseline_report()
    assert r.chosen_strategy_id == "phase1-regression"
