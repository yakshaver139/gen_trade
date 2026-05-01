"""Tests for the SQLAlchemy persistence layer (Phase 2).

Pins the round-trip contract: a `RunResult` produced by `run_ga` must
serialise to a SQLite database and round-trip back to a structurally
equivalent `RunResult`. This is the foundation the API and resumability
in Phase 2 build on.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gentrade.backtest import BacktestConfig
from gentrade.ga import GAConfig, run_ga
from gentrade.persistence import (
    init_db,
    list_runs,
    load_run,
    save_run,
)

CFG = GAConfig(
    population_size=4,
    max_generations=2,
    elitism_count=2,
    selection_pressure="tournament",
    min_trades_for_fitness=1,
)
BT_CFG = BacktestConfig(
    target_pct=0.015,
    stop_loss_pct=0.0075,
    trade_window_bars=4,
    taker_fee_bps=10.0,
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
    return [
        {
            "id": f"s{i}",
            "indicators": [
                {"absolute": True, "indicator": "close", "op": ">=", "abs_value": 100.0 + i * 0.5}
            ],
            "conjunctions": [],
        }
        for i in range(4)
    ]


def _windows(bars):
    return (
        (bars.iloc[0]["open_ts"], bars.iloc[29]["open_ts"]),
        (bars.iloc[30]["open_ts"], bars.iloc[59]["open_ts"]),
        (bars.iloc[60]["open_ts"], bars.iloc[89]["open_ts"]),
    )


def _do_run(seed: int = 42):
    bars = _bars()
    train, val, test = _windows(bars)
    return run_ga(
        bars=bars,
        initial_strategies=_strategies(),
        train_window=train,
        validation_window=val,
        test_window=test,
        config=CFG,
        backtest_config=BT_CFG,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# init_db / empty state
# ---------------------------------------------------------------------------

def test_init_db_creates_schema(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = init_db(db_url)
    # listing on a fresh schema returns empty
    assert list_runs(engine) == []


# ---------------------------------------------------------------------------
# save_run / load_run round trip
# ---------------------------------------------------------------------------

def test_save_run_returns_run_id(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/t.db")
    result = _do_run(seed=42)
    run_id = save_run(result, engine)
    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_save_then_load_round_trips_manifest(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/t.db")
    result = _do_run(seed=42)
    run_id = save_run(result, engine)

    loaded = load_run(run_id, engine)

    assert loaded.manifest.seed == result.manifest.seed
    assert loaded.manifest.train_window == result.manifest.train_window
    assert loaded.manifest.validation_window == result.manifest.validation_window
    assert loaded.manifest.test_window == result.manifest.test_window
    assert loaded.manifest.config_snapshot == result.manifest.config_snapshot


def test_save_then_load_round_trips_per_generation(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/t.db")
    result = _do_run(seed=42)
    run_id = save_run(result, engine)

    loaded = load_run(run_id, engine)

    assert len(loaded.per_generation) == len(result.per_generation)
    for orig, got in zip(result.per_generation, loaded.per_generation, strict=True):
        assert orig.generation == got.generation
        assert orig.train_metrics.max_fitness == pytest.approx(got.train_metrics.max_fitness)
        assert orig.train_metrics.median_fitness == pytest.approx(got.train_metrics.median_fitness)
        assert orig.train_metrics.n_strategies_with_trades == got.train_metrics.n_strategies_with_trades
        assert orig.validation_metrics.max_fitness == pytest.approx(got.validation_metrics.max_fitness)


def test_save_then_load_round_trips_backtest_report(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/t.db")
    result = _do_run(seed=42)
    run_id = save_run(result, engine)

    loaded = load_run(run_id, engine)

    assert loaded.backtest_report.chosen_strategy_id == result.backtest_report.chosen_strategy_id

    def _eq_metrics(a, b):
        # NaN-safe float comparison for the ratio metrics
        for field in ("n_trades", "win_rate", "profit_factor", "expectancy",
                      "sharpe", "sortino", "calmar", "max_drawdown"):
            av, bv = getattr(a, field), getattr(b, field)
            if isinstance(av, float) and (math.isnan(av) or math.isinf(av)):
                assert (math.isnan(av) and math.isnan(bv)) or av == bv
            else:
                assert av == pytest.approx(bv)

    _eq_metrics(loaded.backtest_report.train_metrics, result.backtest_report.train_metrics)
    _eq_metrics(loaded.backtest_report.validation_metrics, result.backtest_report.validation_metrics)
    _eq_metrics(loaded.backtest_report.test_metrics, result.backtest_report.test_metrics)
    _eq_metrics(loaded.backtest_report.buy_and_hold_test, result.backtest_report.buy_and_hold_test)
    _eq_metrics(loaded.backtest_report.random_entry_test, result.backtest_report.random_entry_test)
    if math.isnan(result.backtest_report.overfitting_gap):
        assert math.isnan(loaded.backtest_report.overfitting_gap)
    else:
        assert loaded.backtest_report.overfitting_gap == pytest.approx(
            result.backtest_report.overfitting_gap
        )


def test_save_then_load_preserves_final_ranking(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/t.db")
    result = _do_run(seed=42)
    run_id = save_run(result, engine)

    loaded = load_run(run_id, engine)

    # ranking ids and fitnesses must round-trip in the same order
    assert loaded.final_ranking["id"].tolist() == result.final_ranking["id"].tolist()
    assert loaded.final_ranking["fitness"].tolist() == pytest.approx(
        result.final_ranking["fitness"].tolist()
    )
    # strategy dicts round-trip too (indicators + conjunctions only;
    # the dynamically-attached `parsed` query string isn't persisted)
    for orig, got in zip(
        result.final_ranking["strategy"].tolist(),
        loaded.final_ranking["strategy"].tolist(),
        strict=True,
    ):
        assert orig["id"] == got["id"]
        assert orig["indicators"] == got["indicators"]
        assert orig["conjunctions"] == got["conjunctions"]


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

def test_list_runs_summarises_each_run(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/t.db")
    a = _do_run(seed=1)
    b = _do_run(seed=2)
    run_a = save_run(a, engine)
    run_b = save_run(b, engine)

    rows = list_runs(engine)

    assert len(rows) == 2
    ids = {r["id"] for r in rows}
    assert ids == {run_a, run_b}
    # each row must carry seed + chosen_strategy_id at minimum
    for r in rows:
        assert "seed" in r
        assert "chosen_strategy_id" in r
        assert "started_at" in r


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------

def test_load_run_missing_id_raises(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/t.db")
    with pytest.raises(LookupError, match="not found"):
        load_run("does-not-exist", engine)


def test_save_run_with_code_sha_and_data_hash(tmp_path):
    """When the manifest carries reproducibility hashes, they round-trip."""
    engine = init_db(f"sqlite:///{tmp_path}/t.db")

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
        code_sha="deadbeef" * 5,
        data_hash="cafe" * 16,
    )
    run_id = save_run(result, engine)

    loaded = load_run(run_id, engine)
    assert loaded.manifest.code_sha == "deadbeef" * 5
    assert loaded.manifest.data_hash == "cafe" * 16


# ---------------------------------------------------------------------------
# resumability — the gold-standard determinism test
# ---------------------------------------------------------------------------

def test_incremental_save_persists_each_generation(tmp_path):
    """Running with engine= populates generations + strategies tables incrementally."""
    from sqlalchemy.orm import Session

    from gentrade.persistence import RunRow

    engine = init_db(f"sqlite:///{tmp_path}/t.db")
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
        engine=engine,
    )

    # The result returned in-memory is identical to a non-persisted run
    assert result.backtest_report is not None

    runs = list_runs(engine)
    assert len(runs) == 1
    assert runs[0]["status"] == "reported"
    assert runs[0]["current_generation"] == CFG.max_generations

    # Strategies for every evaluated generation are present.
    with Session(engine) as session:
        run_row = session.scalars(__import__("sqlalchemy").select(RunRow)).one()
        gen_numbers = sorted({s.generation_number for s in run_row.strategies})
        # gens 1..max are evaluated; the final gen has no successor population
        assert gen_numbers == list(range(1, CFG.max_generations + 1))


def test_resume_after_kill_produces_byte_equivalent_result(tmp_path):
    """Run for max_generations from scratch; run for half then resume; assert identical.

    'Identical' here means: same fitness sequence, same per-generation max_fitness,
    same chosen strategy content. Strategy IDs differ (uuid.uuid4 is not seeded by
    `random.seed`) — that's the same caveat as `test_run_ga_deterministic_with_same_seed`.
    """
    from gentrade.ga import run_ga as run

    fresh_db = init_db(f"sqlite:///{tmp_path}/fresh.db")
    full_cfg = GAConfig(
        population_size=4,
        max_generations=4,
        elitism_count=2,
        selection_pressure="tournament",
        min_trades_for_fitness=1,
    )
    bars = _bars()
    train, val, test = _windows(bars)
    fresh = run(
        bars=bars,
        initial_strategies=_strategies(),
        train_window=train,
        validation_window=val,
        test_window=test,
        config=full_cfg,
        backtest_config=BT_CFG,
        seed=99,
        engine=fresh_db,
    )

    # Now: a partial run that completes 2 of 4 generations and then
    # exits cleanly without finalizing — simulating a kill at end of gen 2.
    partial_db = init_db(f"sqlite:///{tmp_path}/partial.db")
    partial = run(
        bars=bars,
        initial_strategies=_strategies(),
        train_window=train,
        validation_window=val,
        test_window=test,
        config=full_cfg,
        backtest_config=BT_CFG,
        seed=99,
        engine=partial_db,
        stop_after_generation=2,
    )
    assert len(partial.per_generation) == 2

    # The run is still in progress and resumable.
    rows = list_runs(partial_db)
    assert rows[0]["status"] == "in_progress"
    assert rows[0]["current_generation"] == 2
    run_id = rows[0]["id"]

    # Resume the run.
    resumed = run(
        bars=bars,
        engine=partial_db,
        resume_run_id=run_id,
    )

    # Resume must produce 4 generations total.
    assert len(resumed.per_generation) == 4

    # Per-generation max fitness curves match.
    for fresh_snap, resumed_snap in zip(
        fresh.per_generation, resumed.per_generation, strict=True
    ):
        assert fresh_snap.train_metrics.max_fitness == pytest.approx(
            resumed_snap.train_metrics.max_fitness
        )
        assert fresh_snap.validation_metrics.max_fitness == pytest.approx(
            resumed_snap.validation_metrics.max_fitness
        )

    # Final-ranking fitnesses match exactly.
    assert fresh.final_ranking["fitness"].tolist() == resumed.final_ranking["fitness"].tolist()

    # Chosen strategy content (indicators + conjunctions) matches —
    # IDs differ because uuid.uuid4 isn't seeded by random.seed.
    fresh_chosen = fresh.final_ranking.iloc[0]["strategy"]
    resumed_chosen = resumed.final_ranking.iloc[0]["strategy"]
    assert fresh_chosen["indicators"] == resumed_chosen["indicators"]
    assert fresh_chosen["conjunctions"] == resumed_chosen["conjunctions"]


def test_resume_unknown_run_id_raises(tmp_path):
    engine = init_db(f"sqlite:///{tmp_path}/t.db")
    from gentrade.ga import run_ga as run

    with pytest.raises(LookupError, match="not found"):
        run(bars=_bars(), engine=engine, resume_run_id="not-a-real-id")


def test_resume_completed_run_raises(tmp_path):
    """A finished run is not resumable — it's already in 'reported' state."""
    engine = init_db(f"sqlite:///{tmp_path}/t.db")
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
        engine=engine,
    )
    # find the run id
    rows = list_runs(engine)
    assert len(rows) == 1
    run_id = rows[0]["id"]

    from gentrade.ga import run_ga as run

    with pytest.raises(ValueError, match="already reported"):
        run(bars=bars, engine=engine, resume_run_id=run_id)

    _ = result  # keep linters happy
