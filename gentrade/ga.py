"""Genetic algorithm orchestrator built on the realistic backtest engine.

Replaces the ``genetic.main`` evaluation path. The legacy module stays in
place for the smoke test until Phase 2 deprecates it; this orchestrator
is the path Phases 2+ build on.

Per-generation flow:

1. Backtest every strategy on the train window via ``simulate_trades``
   (real fees, slippage, position state).
2. Score with ``fitness_from_trades`` (expectancy with a min-trades floor).
3. Independently re-evaluate every strategy on the validation window.
   Validation fitness is recorded in the per-generation snapshot but
   **never feeds selection**.
4. Rank by train fitness, build the next population using the configured
   selection pressure, PPX crossover, mutation, and elitism.

After the final generation, the elite strategy (top of the train ranking)
is run through ``produce_backtest_report`` so the test window is touched
exactly once and the report carries baselines + overfitting gap.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gentrade.backtest import BacktestConfig
from gentrade.genetic import generate_population
from gentrade.manifest import Manifest, capture_manifest
from gentrade.selection import compute_weights
from gentrade.walk_forward import (
    BacktestReport,
    evaluate_strategy,
    produce_backtest_report,
    slice_window,
)


@dataclass(frozen=True)
class GAConfig:
    """Knobs for the GA loop. Backtest costs are configured separately via BacktestConfig."""

    population_size: int = 10
    max_generations: int = 100
    elitism_count: int = 2
    selection_pressure: str = "tournament"
    # Strategies producing fewer than this many closed trades on a window
    # collapse to the no_entries floor — penalises noise / lottery-ticket
    # strategies that fire once and luck into a winner.
    min_trades_for_fitness: int = 3
    no_entries_fitness: float = -1000.0


@dataclass(frozen=True)
class GenerationMetrics:
    """Fitness distribution across one generation's population on one window."""

    max_fitness: float
    median_fitness: float
    mean_fitness: float
    n_strategies_with_trades: int


@dataclass(frozen=True)
class GenerationSnapshot:
    """Per-generation pair of (train, validation) metrics for the report curves."""

    generation: int
    train_metrics: GenerationMetrics
    validation_metrics: GenerationMetrics


@dataclass(frozen=True)
class RunResult:
    """Complete output of a GA run.

    Per-generation diagnostics live on ``backtest_report.per_generation`` to
    match the spec's BacktestReport entity. ``final_ranking`` is exposed
    here so callers can inspect the full final-generation population
    without unpacking the report.
    """

    manifest: Manifest
    final_ranking: pd.DataFrame
    backtest_report: BacktestReport

    @property
    def per_generation(self) -> list[GenerationSnapshot]:
        """Convenience accessor for the per-generation diagnostic curves."""
        return self.backtest_report.per_generation


def fitness_from_trades(
    trades: pd.DataFrame,
    min_trades: int,
    no_entries_fitness: float,
) -> float:
    """Per-trade expectancy with a noise floor.

    Strategies producing too few trades collapse to ``no_entries_fitness``
    so the GA can't lottery-ticket its way to a high score on a single
    fluke trade.
    """
    if len(trades) < min_trades:
        return no_entries_fitness
    return float(trades["return"].mean())


def summarise_generation(
    fitnesses: list[float], n_trades_per_strategy: list[int]
) -> GenerationMetrics:
    """Build a GenerationMetrics value from one population's evaluations."""
    if not fitnesses:
        return GenerationMetrics(
            max_fitness=0.0,
            median_fitness=0.0,
            mean_fitness=0.0,
            n_strategies_with_trades=0,
        )
    arr = np.asarray(fitnesses, dtype=float)
    return GenerationMetrics(
        max_fitness=float(arr.max()),
        median_fitness=float(np.median(arr)),
        mean_fitness=float(arr.mean()),
        n_strategies_with_trades=int(sum(1 for n in n_trades_per_strategy if n >= 1)),
    )


def run_ga(
    bars: pd.DataFrame,
    initial_strategies: list[dict],
    train_window: tuple[pd.Timestamp, pd.Timestamp],
    validation_window: tuple[pd.Timestamp, pd.Timestamp],
    test_window: tuple[pd.Timestamp, pd.Timestamp],
    config: GAConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    seed: int = 0,
) -> RunResult:
    """Run the GA end-to-end and return a fully-populated RunResult."""
    cfg = config or GAConfig()
    bt_cfg = backtest_config or BacktestConfig()

    train_bars = slice_window(bars, *train_window)
    val_bars = slice_window(bars, *validation_window)

    # Seed the RNGs the GA loop reads from. genetic.cross_over_ppx uses
    # `random.choice` and pandas `.sample` reads numpy's global RNG.
    random.seed(seed)
    np.random.seed(seed)

    manifest = capture_manifest(
        seed=seed,
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        config_snapshot={
            "ga": asdict(cfg),
            "backtest": asdict(bt_cfg),
        },
    )

    strategies: list[dict] = list(initial_strategies)
    per_generation: list[GenerationSnapshot] = []
    ranked: pd.DataFrame | None = None

    for gen_num in range(1, cfg.max_generations + 1):
        train_trades = [evaluate_strategy(train_bars, s, bt_cfg) for s in strategies]
        train_fitness = [
            fitness_from_trades(t, cfg.min_trades_for_fitness, cfg.no_entries_fitness)
            for t in train_trades
        ]
        train_metrics = summarise_generation(
            train_fitness, [len(t) for t in train_trades]
        )

        val_trades = [evaluate_strategy(val_bars, s, bt_cfg) for s in strategies]
        val_fitness = [
            fitness_from_trades(t, cfg.min_trades_for_fitness, cfg.no_entries_fitness)
            for t in val_trades
        ]
        val_metrics = summarise_generation(val_fitness, [len(t) for t in val_trades])

        per_generation.append(
            GenerationSnapshot(
                generation=gen_num,
                train_metrics=train_metrics,
                validation_metrics=val_metrics,
            )
        )

        ranked = pd.DataFrame(
            [
                {"strategy": s, "id": s["id"], "fitness": f}
                for s, f in zip(strategies, train_fitness, strict=True)
            ]
        )
        ranked = ranked.sort_values("fitness", ascending=False).reset_index(drop=True)

        if gen_num >= cfg.max_generations:
            break

        weights = compute_weights(ranked, cfg.selection_pressure)
        strategies = generate_population(
            ranked, weights, population_size=cfg.population_size
        )

    assert ranked is not None  # max_generations >= 1 by contract
    chosen = ranked.iloc[0]["strategy"]

    # Spec's overfitting gap is the population-level final-generation
    # divergence on max_fitness — the GA *is* a max_fitness optimiser, so
    # the gap there is the right signal. Single-strategy expectancy delta
    # is a fallback when the GA loop didn't run.
    final = per_generation[-1]
    train_max = final.train_metrics.max_fitness
    val_max = final.validation_metrics.max_fitness
    pop_overfitting_gap = (
        (train_max - val_max) / abs(train_max) if train_max != 0 else float("nan")
    )

    report = produce_backtest_report(
        bars=bars,
        chosen_strategy=chosen,
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
        config=bt_cfg,
        seed=seed,
        per_generation=per_generation,
        overfitting_gap=pop_overfitting_gap,
    )

    return RunResult(
        manifest=manifest,
        final_ranking=ranked,
        backtest_report=report,
    )
