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
from typing import Any

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


def train_metrics_zero():
    """Empty PerformanceMetrics placeholder for stop_after_generation results."""
    from gentrade.metrics import PerformanceMetrics

    return PerformanceMetrics(
        n_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        sharpe=float("nan"),
        sortino=float("nan"),
        calmar=float("nan"),
        max_drawdown=0.0,
        avg_trade_duration=pd.Timedelta(0),
    )


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
    initial_strategies: list[dict] | None = None,
    train_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    validation_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    test_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    config: GAConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    seed: int = 0,
    code_sha: str | None = None,
    data_hash: str | None = None,
    engine: Any | None = None,
    resume_run_id: str | None = None,
    stop_after_generation: int | None = None,
) -> RunResult:
    """Run the GA end-to-end and return a fully-populated RunResult.

    ``engine`` (a SQLAlchemy Engine): if provided, the run is persisted
    incrementally — a row per generation, with RNG state pickled at every
    checkpoint. Crashes lose at most the in-flight generation.

    ``resume_run_id``: continue a previously-checkpointed run. Loads
    manifest, snapshots, and the bred-but-unevaluated population for the
    next generation; restores Python+numpy RNG state. The resumed run
    produces byte-equivalent fitness and strategy content to a fresh run
    with the same seed (strategy UUIDs differ — they come from
    `uuid.uuid4()`, which is not seeded — but content matches).

    ``stop_after_generation``: simulate a clean kill — exit the loop at
    the end of this generation without producing the BacktestReport.
    The persisted run stays in `in_progress` state and can be resumed.
    Tests use this; production callers don't.

    ``code_sha`` and ``data_hash`` are pass-through kwargs for the manifest;
    callers that care about reproducibility (the CLI) compute them via
    ``manifest.current_git_sha`` / ``manifest.compute_data_hash`` and pass
    them in. Tests can leave them None.
    """
    if resume_run_id is not None and engine is None:
        raise ValueError("resume_run_id requires an engine")

    if resume_run_id is not None:
        from gentrade.persistence import restore_rng_states, resume_persisted_run

        state = resume_persisted_run(resume_run_id, engine)
        manifest = state.manifest
        train_window = manifest.train_window
        validation_window = manifest.validation_window
        test_window = manifest.test_window
        cfg = GAConfig(**manifest.config_snapshot["ga"])
        bt_cfg = BacktestConfig(**manifest.config_snapshot["backtest"])
        per_generation: list[GenerationSnapshot] = list(state.snapshots_so_far)
        strategies: list[dict] = state.next_population
        gen_start = state.next_generation
        run_id: str | None = state.run_id
        restore_rng_states(state.py_rng_state, state.np_rng_state)
    else:
        if initial_strategies is None or train_window is None or validation_window is None or test_window is None:
            raise ValueError(
                "fresh runs require initial_strategies, train_window, "
                "validation_window, and test_window"
            )
        cfg = config or GAConfig()
        bt_cfg = backtest_config or BacktestConfig()

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
            code_sha=code_sha,
            data_hash=data_hash,
        )

        strategies = list(initial_strategies)
        per_generation = []
        gen_start = 1
        run_id = None
        if engine is not None:
            from gentrade.persistence import start_persisted_run

            run_id = start_persisted_run(manifest, strategies, engine)

    train_bars = slice_window(bars, *train_window)
    val_bars = slice_window(bars, *validation_window)

    ranked: pd.DataFrame | None = None

    for gen_num in range(gen_start, cfg.max_generations + 1):
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

        snapshot = GenerationSnapshot(
            generation=gen_num,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
        )
        per_generation.append(snapshot)

        ranked = pd.DataFrame(
            [
                {"strategy": s, "id": s["id"], "fitness": f}
                for s, f in zip(strategies, train_fitness, strict=True)
            ]
        )
        ranked = ranked.sort_values("fitness", ascending=False).reset_index(drop=True)

        is_last_gen = gen_num >= cfg.max_generations
        if is_last_gen:
            next_strategies: list[dict] | None = None
        else:
            weights = compute_weights(ranked, cfg.selection_pressure)
            next_strategies = generate_population(
                ranked, weights, population_size=cfg.population_size
            )

        if engine is not None and run_id is not None:
            from gentrade.persistence import (
                checkpoint_generation,
                serialise_rng_states,
            )

            py_state, np_state = serialise_rng_states()
            checkpoint_generation(
                run_id=run_id,
                snapshot=snapshot,
                population=strategies,
                fitnesses=train_fitness,
                next_population=next_strategies,
                py_rng_state=py_state,
                np_rng_state=np_state,
                engine=engine,
            )

        if not is_last_gen:
            assert next_strategies is not None
            strategies = next_strategies

        if stop_after_generation is not None and gen_num >= stop_after_generation:
            return RunResult(
                manifest=manifest,
                final_ranking=ranked,
                backtest_report=BacktestReport(
                    chosen_strategy_id="",
                    train_metrics=train_metrics_zero(),
                    validation_metrics=train_metrics_zero(),
                    test_metrics=train_metrics_zero(),
                    buy_and_hold_test=train_metrics_zero(),
                    random_entry_test=train_metrics_zero(),
                    overfitting_gap=float("nan"),
                    per_generation=list(per_generation),
                ),
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

    if engine is not None and run_id is not None:
        from gentrade.persistence import finalize_persisted_run

        finalize_persisted_run(run_id, chosen["id"], report, engine)

    return RunResult(
        manifest=manifest,
        final_ranking=ranked,
        backtest_report=report,
    )
