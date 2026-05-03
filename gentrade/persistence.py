"""SQLAlchemy persistence layer for GA runs.

Schema mirrors the spec's entity model: a `Run` owns `Generation`s and
`Strategy`s. A run carries the reproducibility manifest, the headline
BacktestReport metrics (one PerformanceMetrics value per window plus the
two test-window baselines), the chosen strategy id, and the overfitting
gap. Per-generation rows carry the train/validation diagnostic
distributions. Strategy rows store one chromosome per (run, generation)
slot — the same chromosome may appear in multiple generations via
elitism, so the primary key is `(run_id, generation_number, id)`.

PerformanceMetrics is stored as JSON inside the run row rather than
normalised to its own table — it's small, fixed-shape, and never queried
by individual field. Indicator + conjunction lists on strategies are
likewise JSON; their internal shape is the strategy DSL's concern, not
the persistence layer's.

Resumability is supported via per-generation checkpoints: after each
generation, the run row's `current_generation` is bumped, the snapshot
+ population are saved, and Python+numpy RNG state is pickled into
`py_rng_state`/`np_rng_state`. Resuming restores the RNG state and
continues — the resumed run produces byte-equivalent fitness and
strategy content to a fresh run with the same seed.

The default backend is SQLite (``sqlite:///gentrade.db``); set
``GENTRADE_DB_URL`` to point at a Postgres instance for shared runs.
"""
from __future__ import annotations

import json
import math
import os
import random
import uuid
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    delete,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

from gentrade.ga import (
    GenerationMetrics,
    GenerationSnapshot,
    RunResult,
)
from gentrade.manifest import Manifest
from gentrade.metrics import PerformanceMetrics
from gentrade.walk_forward import BacktestReport

DEFAULT_DB_URL = os.getenv("GENTRADE_DB_URL", "sqlite:///gentrade.db")

_METRIC_FIELDS = [f.name for f in fields(PerformanceMetrics)]


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    started_at: Mapped[datetime] = mapped_column()
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    seed: Mapped[int] = mapped_column(Integer)
    code_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    data_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    train_start: Mapped[datetime] = mapped_column()
    train_end: Mapped[datetime] = mapped_column()
    validation_start: Mapped[datetime] = mapped_column()
    validation_end: Mapped[datetime] = mapped_column()
    test_start: Mapped[datetime] = mapped_column()
    test_end: Mapped[datetime] = mapped_column()
    config_snapshot_json: Mapped[str] = mapped_column(Text)

    # Lifecycle state. "in_progress" → some generations evaluated but
    # report not produced. "reported" → terminal (BacktestReport persisted).
    # "failed" → an exception aborted the loop; the partial state is still
    # readable but should not be resumed without intent.
    status: Mapped[str] = mapped_column(String)
    current_generation: Mapped[int] = mapped_column(Integer, default=0)

    # Pickled RNG state captured at the end of each checkpoint. Both Python
    # `random` and numpy globals are checkpointed so a resumed run picks up
    # exactly where it left off. They are nullable until the first checkpoint.
    py_rng_state: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    np_rng_state: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    chosen_strategy_id: Mapped[str | None] = mapped_column(String, nullable=True)
    overfitting_gap: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    test_metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    buy_and_hold_test_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    random_entry_test_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    generations: Mapped[list[GenerationRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="GenerationRow.number"
    )
    strategies: Mapped[list[StrategyRow]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="(StrategyRow.generation_number, StrategyRow.rank)",
    )
    breeding_events: Mapped[list[BreedingEventRow]] = relationship(
        cascade="all, delete-orphan",
        order_by="(BreedingEventRow.generation_number, BreedingEventRow.id)",
        primaryjoin="RunRow.id == BreedingEventRow.run_id",
        foreign_keys="BreedingEventRow.run_id",
    )


class GenerationRow(Base):
    __tablename__ = "generations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    number: Mapped[int] = mapped_column(Integer)

    train_max_fitness: Mapped[float] = mapped_column(Float)
    train_median_fitness: Mapped[float] = mapped_column(Float)
    train_mean_fitness: Mapped[float] = mapped_column(Float)
    train_n_strategies_with_trades: Mapped[int] = mapped_column(Integer)

    validation_max_fitness: Mapped[float] = mapped_column(Float)
    validation_median_fitness: Mapped[float] = mapped_column(Float)
    validation_mean_fitness: Mapped[float] = mapped_column(Float)
    validation_n_strategies_with_trades: Mapped[int] = mapped_column(Integer)

    run: Mapped[RunRow] = relationship(back_populates="generations")


class StrategyRow(Base):
    __tablename__ = "strategies"

    # A chromosome can recur across generations (elitism keeps the top N
    # untouched), and the same chromosome may also appear in the
    # not-yet-evaluated `generation_number = current_generation + 1` slot
    # — the population that's been bred for the next gen. The composite
    # PK is therefore `(run_id, generation_number, id)`. `fitness` is null
    # for unevaluated generations (i.e. the "next" population).
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    generation_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    fitness: Mapped[float | None] = mapped_column(Float, nullable=True)
    indicators_json: Mapped[str] = mapped_column(Text)
    conjunctions_json: Mapped[str] = mapped_column(Text)

    run: Mapped[RunRow] = relationship(back_populates="strategies")


class BreedingEventRow(Base):
    """Per-child audit trail: which parents made it, and which mutation
    operator (if any) altered it. Drives the live "Breeding activity"
    visualisation on Run detail."""

    __tablename__ = "breeding_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    generation_number: Mapped[int] = mapped_column(Integer)
    child_id: Mapped[str] = mapped_column(String)
    parent_a_id: Mapped[str] = mapped_column(String)
    parent_b_id: Mapped[str] = mapped_column(String)
    operator: Mapped[str] = mapped_column(String)
    applied: Mapped[bool] = mapped_column()
    reason: Mapped[str | None] = mapped_column(String, nullable=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def init_db(db_url: str = DEFAULT_DB_URL) -> Engine:
    """Create the engine and ensure the schema exists."""
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    return engine


def _metrics_to_json(m: PerformanceMetrics) -> str:
    payload: dict[str, Any] = {}
    for f in _METRIC_FIELDS:
        v = getattr(m, f)
        if isinstance(v, pd.Timedelta):
            # store as nanoseconds — exact, language-agnostic
            payload[f] = {"__timedelta_ns__": int(v.value)}
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            payload[f] = {"__float__": "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")}
        else:
            payload[f] = v
    return json.dumps(payload)


def _metrics_from_json(s: str) -> PerformanceMetrics:
    payload = json.loads(s)
    kwargs: dict[str, Any] = {}
    for f in _METRIC_FIELDS:
        v = payload[f]
        if isinstance(v, dict) and "__timedelta_ns__" in v:
            kwargs[f] = pd.Timedelta(v["__timedelta_ns__"], unit="ns")
        elif isinstance(v, dict) and "__float__" in v:
            kwargs[f] = float(v["__float__"])
        else:
            kwargs[f] = v
    return PerformanceMetrics(**kwargs)


def _ts(value: pd.Timestamp | datetime) -> datetime:
    """SQLAlchemy datetime columns expect a stdlib datetime."""
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _utc(value: datetime | pd.Timestamp) -> pd.Timestamp:
    """Coerce a database-loaded datetime back to a UTC-aware pandas Timestamp.

    SQLite (and many drivers) drop tz info on the wire. The codebase only
    ever writes UTC, so it's safe to localise on read.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# ---------------------------------------------------------------------------
# resume state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResumeState:
    """Everything `run_ga` needs to pick up an interrupted run.

    `next_generation` is the generation number that still needs to be
    evaluated (1-indexed). `next_population` is the saved population for
    that generation — it was already bred at the end of the prior gen.
    `py_rng_state` and `np_rng_state` are pickle blobs ready for
    `random.setstate` / `np.random.set_state`.
    """

    run_id: str
    manifest: Manifest
    snapshots_so_far: list[GenerationSnapshot]
    next_generation: int
    next_population: list[dict]
    py_rng_state: bytes | None
    np_rng_state: bytes | None


# ---------------------------------------------------------------------------
# incremental save: start / checkpoint / finalize / resume
# ---------------------------------------------------------------------------

def start_persisted_run(
    manifest: Manifest,
    initial_strategies: list[dict],
    engine: Engine,
) -> str:
    """Create a new run row in `in_progress` status and save the initial population.

    The initial population is stored under `generation_number = 1` (the
    next generation to evaluate). RNG state is left null until the first
    checkpoint — by then the loop will have consumed its first slice of
    randomness and there's something meaningful to capture.
    """
    run_id = str(uuid.uuid4())
    run_row = RunRow(
        id=run_id,
        started_at=_ts(manifest.started_at),
        seed=manifest.seed,
        code_sha=manifest.code_sha,
        data_hash=manifest.data_hash,
        train_start=_ts(manifest.train_window[0]),
        train_end=_ts(manifest.train_window[1]),
        validation_start=_ts(manifest.validation_window[0]),
        validation_end=_ts(manifest.validation_window[1]),
        test_start=_ts(manifest.test_window[0]),
        test_end=_ts(manifest.test_window[1]),
        config_snapshot_json=json.dumps(manifest.config_snapshot),
        status="in_progress",
        current_generation=0,
    )
    for rank, strat in enumerate(initial_strategies):
        run_row.strategies.append(
            StrategyRow(
                generation_number=1,
                id=strat["id"],
                rank=rank,
                fitness=None,
                indicators_json=json.dumps(strat["indicators"]),
                conjunctions_json=json.dumps(strat["conjunctions"]),
            )
        )
    with Session(engine) as session:
        session.add(run_row)
        session.commit()
    return run_id


def checkpoint_generation(
    run_id: str,
    snapshot: GenerationSnapshot,
    population: list[dict],
    fitnesses: list[float],
    next_population: list[dict] | None,
    py_rng_state: bytes,
    np_rng_state: bytes,
    engine: Engine,
    breeding_events: list | None = None,
) -> None:
    """Persist the just-completed generation and (optionally) the next-gen population.

    Idempotent at the (run_id, generation_number) granularity: re-checkpointing
    the same generation overwrites prior strategy + breeding-event rows for
    the relevant generation_number.

    ``breeding_events`` are :class:`gentrade.mutation.BreedingEvent`
    instances for the *next* generation's children — recording how each
    chromosome in next_population came to exist (parents + mutation
    operator). Empty list / None for the final generation.
    """
    with Session(engine) as session:
        run_row = session.get(RunRow, run_id)
        if run_row is None:
            raise LookupError(f"run {run_id!r} not found")

        # Replace any existing rows for the just-evaluated gen and its
        # bred-but-unevaluated successor (defensive — a partial checkpoint
        # could otherwise leave stale rows behind).
        session.execute(
            delete(StrategyRow).where(
                StrategyRow.run_id == run_id,
                StrategyRow.generation_number == snapshot.generation,
            )
        )
        if next_population is not None:
            session.execute(
                delete(StrategyRow).where(
                    StrategyRow.run_id == run_id,
                    StrategyRow.generation_number == snapshot.generation + 1,
                )
            )

        # Write the just-evaluated generation's population with fitnesses.
        for rank, (strat, fit) in enumerate(zip(population, fitnesses, strict=True)):
            session.add(
                StrategyRow(
                    run_id=run_id,
                    generation_number=snapshot.generation,
                    id=strat["id"],
                    rank=rank,
                    fitness=float(fit),
                    indicators_json=json.dumps(strat["indicators"]),
                    conjunctions_json=json.dumps(strat["conjunctions"]),
                )
            )

        # Write the bred-but-unevaluated next population (if any).
        if next_population is not None:
            for rank, strat in enumerate(next_population):
                session.add(
                    StrategyRow(
                        run_id=run_id,
                        generation_number=snapshot.generation + 1,
                        id=strat["id"],
                        rank=rank,
                        fitness=None,
                        indicators_json=json.dumps(strat["indicators"]),
                        conjunctions_json=json.dumps(strat["conjunctions"]),
                    )
                )

        # Append the snapshot row.
        session.add(
            GenerationRow(
                run_id=run_id,
                number=snapshot.generation,
                train_max_fitness=snapshot.train_metrics.max_fitness,
                train_median_fitness=snapshot.train_metrics.median_fitness,
                train_mean_fitness=snapshot.train_metrics.mean_fitness,
                train_n_strategies_with_trades=snapshot.train_metrics.n_strategies_with_trades,
                validation_max_fitness=snapshot.validation_metrics.max_fitness,
                validation_median_fitness=snapshot.validation_metrics.median_fitness,
                validation_mean_fitness=snapshot.validation_metrics.mean_fitness,
                validation_n_strategies_with_trades=snapshot.validation_metrics.n_strategies_with_trades,
            )
        )

        # Persist per-child breeding events for the next-generation children.
        # Idempotent at (run_id, generation_number) — clear any existing
        # rows for the same generation before inserting.
        if breeding_events:
            target_gen = breeding_events[0].generation_number
            session.execute(
                delete(BreedingEventRow).where(
                    BreedingEventRow.run_id == run_id,
                    BreedingEventRow.generation_number == target_gen,
                )
            )
            for ev in breeding_events:
                session.add(
                    BreedingEventRow(
                        run_id=run_id,
                        generation_number=ev.generation_number,
                        child_id=ev.child_id,
                        parent_a_id=ev.parent_a_id,
                        parent_b_id=ev.parent_b_id,
                        operator=ev.operator,
                        applied=ev.applied,
                        reason=ev.reason,
                    )
                )

        run_row.current_generation = snapshot.generation
        run_row.py_rng_state = py_rng_state
        run_row.np_rng_state = np_rng_state
        session.commit()


def finalize_persisted_run(
    run_id: str,
    chosen_strategy_id: str,
    report: BacktestReport,
    engine: Engine,
) -> None:
    """Mark a run completed and store its BacktestReport headline metrics."""
    with Session(engine) as session:
        run_row = session.get(RunRow, run_id)
        if run_row is None:
            raise LookupError(f"run {run_id!r} not found")
        run_row.status = "reported"
        run_row.finished_at = _ts(pd.Timestamp.utcnow())
        run_row.chosen_strategy_id = chosen_strategy_id
        run_row.overfitting_gap = report.overfitting_gap
        run_row.train_metrics_json = _metrics_to_json(report.train_metrics)
        run_row.validation_metrics_json = _metrics_to_json(report.validation_metrics)
        run_row.test_metrics_json = _metrics_to_json(report.test_metrics)
        run_row.buy_and_hold_test_json = _metrics_to_json(report.buy_and_hold_test)
        run_row.random_entry_test_json = _metrics_to_json(report.random_entry_test)
        session.commit()


def resume_persisted_run(run_id: str, engine: Engine) -> ResumeState:
    """Load enough state to continue an in-progress run."""
    with Session(engine) as session:
        run_row = session.get(RunRow, run_id)
        if run_row is None:
            raise LookupError(f"run {run_id!r} not found")
        if run_row.status == "reported":
            raise ValueError(
                f"run {run_id!r} is already reported — "
                "load_run() instead of resume_persisted_run()"
            )

        manifest = Manifest(
            seed=run_row.seed,
            train_window=(_utc(run_row.train_start), _utc(run_row.train_end)),
            validation_window=(
                _utc(run_row.validation_start),
                _utc(run_row.validation_end),
            ),
            test_window=(_utc(run_row.test_start), _utc(run_row.test_end)),
            started_at=_utc(run_row.started_at),
            config_snapshot=json.loads(run_row.config_snapshot_json),
            code_sha=run_row.code_sha,
            data_hash=run_row.data_hash,
        )

        snapshots = [
            GenerationSnapshot(
                generation=gen.number,
                train_metrics=GenerationMetrics(
                    max_fitness=gen.train_max_fitness,
                    median_fitness=gen.train_median_fitness,
                    mean_fitness=gen.train_mean_fitness,
                    n_strategies_with_trades=gen.train_n_strategies_with_trades,
                ),
                validation_metrics=GenerationMetrics(
                    max_fitness=gen.validation_max_fitness,
                    median_fitness=gen.validation_median_fitness,
                    mean_fitness=gen.validation_mean_fitness,
                    n_strategies_with_trades=gen.validation_n_strategies_with_trades,
                ),
            )
            for gen in run_row.generations
        ]

        next_gen_number = run_row.current_generation + 1
        next_pop_rows = [
            s for s in run_row.strategies if s.generation_number == next_gen_number
        ]
        if not next_pop_rows:
            raise RuntimeError(
                f"run {run_id!r} has no saved population for gen {next_gen_number} — "
                "the checkpoint is incomplete and cannot be resumed"
            )
        next_pop_rows.sort(key=lambda s: s.rank)
        next_population = [
            {
                "id": s.id,
                "indicators": json.loads(s.indicators_json),
                "conjunctions": json.loads(s.conjunctions_json),
            }
            for s in next_pop_rows
        ]

        return ResumeState(
            run_id=run_id,
            manifest=manifest,
            snapshots_so_far=snapshots,
            next_generation=next_gen_number,
            next_population=next_population,
            py_rng_state=run_row.py_rng_state,
            np_rng_state=run_row.np_rng_state,
        )


def serialise_rng_states() -> tuple[bytes, bytes]:
    """JSON-encode the current Python and numpy global RNG states.

    Both states are tuples of (version-tag, list/array of ints, small ints).
    JSON gives us deterministic, language-portable, deserialisation-safe
    blobs — `pickle.loads` over DB content is an RCE gadget if an attacker
    ever gains write access to the database (CWE-502).
    """
    py = random.getstate()
    # Python's getstate returns (version_int, tuple_of_624_ints+1, None_or_float).
    py_payload = {
        "version": py[0],
        "state": list(py[1]),
        "gauss_next": py[2],
    }
    np_state = np.random.get_state()
    # numpy's get_state returns ('MT19937', ndarray uint32 624, int, int, float).
    np_payload = {
        "name": np_state[0],
        "state": [int(x) for x in np_state[1]],
        "pos": int(np_state[2]),
        "has_gauss": int(np_state[3]),
        "cached_gaussian": float(np_state[4]),
    }
    return json.dumps(py_payload).encode(), json.dumps(np_payload).encode()


def restore_rng_states(py_state: bytes | None, np_state: bytes | None) -> None:
    """Restore RNG states previously written by ``serialise_rng_states``."""
    if py_state is not None:
        py = json.loads(py_state.decode() if isinstance(py_state, bytes | bytearray) else py_state)
        random.setstate((py["version"], tuple(py["state"]), py["gauss_next"]))
    if np_state is not None:
        nps = json.loads(np_state.decode() if isinstance(np_state, bytes | bytearray) else np_state)
        np.random.set_state(
            (
                nps["name"],
                np.array(nps["state"], dtype=np.uint32),
                nps["pos"],
                nps["has_gauss"],
                nps["cached_gaussian"],
            )
        )


# ---------------------------------------------------------------------------
# one-shot save / load / list (Phase 1 surface preserved)
# ---------------------------------------------------------------------------

def save_run(result: RunResult, engine: Engine) -> str:
    """One-shot save: persist a finished `RunResult` start-to-finish.

    Equivalent to start + checkpoint(every gen) + finalize. Used by tests
    and by the in-memory path that builds a complete result before saving.
    Live runs prefer the incremental helpers so they survive crashes.
    """
    final_pop = [row["strategy"] for _, row in result.final_ranking.iterrows()]
    final_fitnesses = [float(f) for f in result.final_ranking["fitness"].tolist()]

    run_id = start_persisted_run(
        manifest=result.manifest,
        initial_strategies=final_pop,
        engine=engine,
    )

    py_state, np_state = serialise_rng_states()
    for snap in result.backtest_report.per_generation:
        is_last = snap.generation == max(
            s.generation for s in result.backtest_report.per_generation
        )
        # one-shot save doesn't have per-gen populations (we collapsed),
        # so we record the final population at every generation. That's
        # lossy for resumability but fine for this code path's purpose.
        checkpoint_generation(
            run_id=run_id,
            snapshot=snap,
            population=final_pop,
            fitnesses=final_fitnesses,
            next_population=None if is_last else final_pop,
            py_rng_state=py_state,
            np_rng_state=np_state,
            engine=engine,
        )

    finalize_persisted_run(
        run_id=run_id,
        chosen_strategy_id=result.backtest_report.chosen_strategy_id,
        report=result.backtest_report,
        engine=engine,
    )
    return run_id


def load_run(run_id: str, engine: Engine) -> RunResult:
    """Load a finished run. Raises if the run never reached `reported`."""
    with Session(engine) as session:
        run_row = session.get(RunRow, run_id)
        if run_row is None:
            raise LookupError(f"run {run_id!r} not found")
        if run_row.status != "reported":
            raise ValueError(
                f"run {run_id!r} status is {run_row.status!r}, not reported. "
                "Use resume_persisted_run() for in-progress runs."
            )

        manifest = Manifest(
            seed=run_row.seed,
            train_window=(_utc(run_row.train_start), _utc(run_row.train_end)),
            validation_window=(
                _utc(run_row.validation_start),
                _utc(run_row.validation_end),
            ),
            test_window=(_utc(run_row.test_start), _utc(run_row.test_end)),
            started_at=_utc(run_row.started_at),
            config_snapshot=json.loads(run_row.config_snapshot_json),
            code_sha=run_row.code_sha,
            data_hash=run_row.data_hash,
        )

        per_generation = [
            GenerationSnapshot(
                generation=gen.number,
                train_metrics=GenerationMetrics(
                    max_fitness=gen.train_max_fitness,
                    median_fitness=gen.train_median_fitness,
                    mean_fitness=gen.train_mean_fitness,
                    n_strategies_with_trades=gen.train_n_strategies_with_trades,
                ),
                validation_metrics=GenerationMetrics(
                    max_fitness=gen.validation_max_fitness,
                    median_fitness=gen.validation_median_fitness,
                    mean_fitness=gen.validation_mean_fitness,
                    n_strategies_with_trades=gen.validation_n_strategies_with_trades,
                ),
            )
            for gen in run_row.generations
        ]

        report = BacktestReport(
            chosen_strategy_id=run_row.chosen_strategy_id or "",
            train_metrics=_metrics_from_json(run_row.train_metrics_json),
            validation_metrics=_metrics_from_json(run_row.validation_metrics_json),
            test_metrics=_metrics_from_json(run_row.test_metrics_json),
            buy_and_hold_test=_metrics_from_json(run_row.buy_and_hold_test_json),
            random_entry_test=_metrics_from_json(run_row.random_entry_test_json),
            overfitting_gap=run_row.overfitting_gap if run_row.overfitting_gap is not None else math.nan,
            per_generation=per_generation,
        )

        # Final ranking = the strategies for the last evaluated generation.
        last_gen = run_row.current_generation
        final_rows = sorted(
            [s for s in run_row.strategies if s.generation_number == last_gen],
            key=lambda r: r.rank,
        )
        ranking = pd.DataFrame(
            [
                {
                    "strategy": {
                        "id": s.id,
                        "indicators": json.loads(s.indicators_json),
                        "conjunctions": json.loads(s.conjunctions_json),
                    },
                    "id": s.id,
                    "fitness": s.fitness,
                }
                for s in final_rows
            ]
        )

        return RunResult(
            manifest=manifest,
            final_ranking=ranking,
            backtest_report=report,
        )


def list_runs(engine: Engine) -> list[dict[str, Any]]:
    """Return a summary row per persisted run, newest first."""
    with Session(engine) as session:
        stmt = select(RunRow).order_by(RunRow.started_at.desc())
        return [
            {
                "id": r.id,
                "started_at": _utc(r.started_at),
                "finished_at": _utc(r.finished_at) if r.finished_at else None,
                "status": r.status,
                "current_generation": r.current_generation,
                "seed": r.seed,
                "chosen_strategy_id": r.chosen_strategy_id,
                "overfitting_gap": r.overfitting_gap,
                "code_sha": r.code_sha,
                "data_hash": r.data_hash,
            }
            for r in session.scalars(stmt)
        ]
