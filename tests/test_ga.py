"""Tests for the new GA orchestrator (Phase 1).

This module pins the contract for `run_ga`: it consumes the realistic
backtest engine on the train window only, records validation metrics every
generation (no selection feedback), and produces a final BacktestReport
on the chosen elite strategy. The legacy `genetic.main` path is untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gentrade.backtest import BacktestConfig
from gentrade.ga import (
    GAConfig,
    GenerationSnapshot,
    RunResult,
    fitness_from_trades,
    run_ga,
    summarise_generation,
)
from gentrade.walk_forward import BacktestReport

CFG = GAConfig(
    population_size=4,
    max_generations=2,
    elitism_count=2,
    selection_pressure="tournament",
    min_trades_for_fitness=1,  # tiny fixtures don't have many trades
    no_entries_fitness=-1000.0,
)
BT_CFG = BacktestConfig(
    target_pct=0.015,
    stop_loss_pct=0.0075,
    trade_window_bars=4,
    fee_bps=10.0,
    slippage_bps=1.0,
)


def _bars(n: int = 90) -> pd.DataFrame:
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


def _strategies() -> list[dict]:
    """Four hand-built strategies whose queries fire at different intensities."""
    return [
        {
            "id": "s0",
            "indicators": [
                {"absolute": True, "indicator": "close", "op": ">=", "abs_value": 101.0}
            ],
            "conjunctions": [],
        },
        {
            "id": "s1",
            "indicators": [
                {"absolute": True, "indicator": "close", "op": ">=", "abs_value": 100.5}
            ],
            "conjunctions": [],
        },
        {
            "id": "s2",
            "indicators": [
                {"absolute": True, "indicator": "close", "op": "<=", "abs_value": 100.4}
            ],
            "conjunctions": [],
        },
        {
            "id": "s3",
            "indicators": [
                {"absolute": True, "indicator": "close", "op": ">=", "abs_value": 105.0}
            ],
            "conjunctions": [],
        },
    ]


def _windows(bars: pd.DataFrame):
    return (
        (bars.iloc[0]["open_ts"], bars.iloc[29]["open_ts"]),
        (bars.iloc[30]["open_ts"], bars.iloc[59]["open_ts"]),
        (bars.iloc[60]["open_ts"], bars.iloc[89]["open_ts"]),
    )


# ---------------------------------------------------------------------------
# fitness_from_trades
# ---------------------------------------------------------------------------

def test_fitness_from_trades_below_threshold_returns_floor():
    trades = pd.DataFrame({"return": [0.01]})
    f = fitness_from_trades(trades, min_trades=3, no_entries_fitness=-1000.0)
    assert f == -1000.0


def test_fitness_from_trades_returns_expectancy_above_threshold():
    trades = pd.DataFrame({"return": [0.01, 0.02, 0.03]})
    f = fitness_from_trades(trades, min_trades=3, no_entries_fitness=-1000.0)
    assert f == 0.02


def test_fitness_from_trades_handles_empty_frame():
    trades = pd.DataFrame({"return": []})
    f = fitness_from_trades(trades, min_trades=1, no_entries_fitness=-1000.0)
    assert f == -1000.0


# ---------------------------------------------------------------------------
# summarise_generation
# ---------------------------------------------------------------------------

def test_summarise_generation_records_distribution():
    snap = summarise_generation([0.01, 0.02, 0.03, 0.04], [1, 2, 3, 0])
    assert snap.max_fitness == 0.04
    assert snap.mean_fitness == 0.025
    assert snap.median_fitness == 0.025
    assert snap.n_strategies_with_trades == 3


def test_summarise_generation_handles_empty_population():
    snap = summarise_generation([], [])
    assert snap.max_fitness == 0.0
    assert snap.n_strategies_with_trades == 0


# ---------------------------------------------------------------------------
# run_ga: end-to-end
# ---------------------------------------------------------------------------

def test_run_ga_returns_run_result_shape():
    bars = _bars()
    train, val, test = _windows(bars)
    result = run_ga(
        bars=bars,
        initial_strategies=_strategies(),
        train_window=train,
        validation_window=val,
        test_window=test,
        config=CFG,
        backtest_config=BT_CFG,
        seed=42,
    )

    assert isinstance(result, RunResult)
    assert isinstance(result.backtest_report, BacktestReport)
    # one snapshot per generation
    assert len(result.per_generation) == CFG.max_generations
    assert all(isinstance(s, GenerationSnapshot) for s in result.per_generation)


def test_run_ga_per_generation_metrics_populated():
    bars = _bars()
    train, val, test = _windows(bars)
    result = run_ga(
        bars=bars,
        initial_strategies=_strategies(),
        train_window=train,
        validation_window=val,
        test_window=test,
        config=CFG,
        backtest_config=BT_CFG,
        seed=42,
    )
    for snap in result.per_generation:
        assert snap.train_metrics.n_strategies_with_trades >= 0
        assert snap.validation_metrics.n_strategies_with_trades >= 0


def test_run_ga_chosen_strategy_matches_top_of_final_ranking():
    bars = _bars()
    train, val, test = _windows(bars)
    result = run_ga(
        bars=bars,
        initial_strategies=_strategies(),
        train_window=train,
        validation_window=val,
        test_window=test,
        config=CFG,
        backtest_config=BT_CFG,
        seed=42,
    )
    top_id = result.final_ranking.iloc[0]["id"]
    assert result.backtest_report.chosen_strategy_id == top_id


def test_run_ga_population_size_preserved_across_generations():
    bars = _bars()
    train, val, test = _windows(bars)
    cfg = GAConfig(
        population_size=4,
        max_generations=3,
        elitism_count=2,
        selection_pressure="tournament",
        min_trades_for_fitness=1,
    )
    result = run_ga(
        bars=bars,
        initial_strategies=_strategies(),
        train_window=train,
        validation_window=val,
        test_window=test,
        config=cfg,
        backtest_config=BT_CFG,
        seed=42,
    )
    # final ranking should always have population_size rows
    assert len(result.final_ranking) == cfg.population_size


def test_run_ga_deterministic_with_same_seed():
    """Same seed → same fitness sequence and same per-generation diagnostics.

    Bred children get a fresh UUID per call (legacy `make_strategy_from_indicators`
    behaviour), so child IDs *do* differ across runs. Algorithmic determinism is
    asserted through fitness — same seed implies same content, same content
    implies same fitness — and through the per-generation summary metrics.
    """
    bars = _bars()
    train, val, test = _windows(bars)
    a = run_ga(
        bars, _strategies(), train, val, test,
        config=CFG, backtest_config=BT_CFG, seed=99,
    )
    b = run_ga(
        bars, _strategies(), train, val, test,
        config=CFG, backtest_config=BT_CFG, seed=99,
    )
    assert a.final_ranking["fitness"].tolist() == b.final_ranking["fitness"].tolist()
    a_summary = [
        (s.train_metrics.max_fitness, s.validation_metrics.max_fitness)
        for s in a.per_generation
    ]
    b_summary = [
        (s.train_metrics.max_fitness, s.validation_metrics.max_fitness)
        for s in b.per_generation
    ]
    assert a_summary == b_summary
    assert a.backtest_report.train_metrics.expectancy == b.backtest_report.train_metrics.expectancy


def test_run_ga_manifest_captures_seed_and_windows():
    bars = _bars()
    train, val, test = _windows(bars)
    result = run_ga(
        bars, _strategies(), train, val, test,
        config=CFG, backtest_config=BT_CFG, seed=7,
    )
    assert result.manifest.seed == 7
    assert result.manifest.train_window == train
    assert result.manifest.validation_window == val
    assert result.manifest.test_window == test
    assert "ga" in result.manifest.config_snapshot
    assert "backtest" in result.manifest.config_snapshot


def test_run_ga_validation_metrics_dont_feed_selection():
    """Sanity: validation metrics are recorded but the ranking ordering
    must reflect train fitness, not a mix."""
    bars = _bars()
    train, val, test = _windows(bars)
    result = run_ga(
        bars, _strategies(), train, val, test,
        config=CFG, backtest_config=BT_CFG, seed=42,
    )
    # final ranking is sorted descending by train fitness
    fitnesses = result.final_ranking["fitness"].tolist()
    assert fitnesses == sorted(fitnesses, reverse=True)
