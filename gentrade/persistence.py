"""SQLAlchemy persistence layer for GA runs.

Schema mirrors the spec's entity model: a `Run` owns `Generation`s and
`Strategy`s. A run carries the reproducibility manifest, the headline
BacktestReport metrics (one PerformanceMetrics value per window plus the
two test-window baselines), the chosen strategy id, and the overfitting
gap. Per-generation rows carry the train/validation diagnostic
distributions. Strategy rows store the chromosome (indicators +
conjunctions) plus its final-generation fitness so the population is
fully reconstructable.

PerformanceMetrics is stored as JSON inside the run row rather than
normalised to its own table — it's small, fixed-shape, and never queried
by individual field. Indicator + conjunction lists on strategies are
likewise JSON; their internal shape is the strategy DSL's concern, not
the persistence layer's.

The default backend is SQLite (``sqlite:///gentrade.db``); set
``GENTRADE_DB_URL`` to point at a Postgres instance for shared runs.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import fields
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
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

    chosen_strategy_id: Mapped[str | None] = mapped_column(String, nullable=True)
    overfitting_gap: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_metrics_json: Mapped[str] = mapped_column(Text)
    validation_metrics_json: Mapped[str] = mapped_column(Text)
    test_metrics_json: Mapped[str] = mapped_column(Text)
    buy_and_hold_test_json: Mapped[str] = mapped_column(Text)
    random_entry_test_json: Mapped[str] = mapped_column(Text)

    generations: Mapped[list[GenerationRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="GenerationRow.number"
    )
    strategies: Mapped[list[StrategyRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="StrategyRow.rank"
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

    # The strategy id is unique within a run (see comments on UUID generation
    # in gentrade.generate_strategy); composite primary key (run_id, id)
    # keeps that scoping explicit.
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    fitness: Mapped[float] = mapped_column(Float)
    indicators_json: Mapped[str] = mapped_column(Text)
    conjunctions_json: Mapped[str] = mapped_column(Text)

    run: Mapped[RunRow] = relationship(back_populates="strategies")


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
# save / load / list
# ---------------------------------------------------------------------------

def save_run(result: RunResult, engine: Engine) -> str:
    """Serialise a RunResult to the database; returns the new run id."""
    run_id = str(uuid.uuid4())
    m = result.manifest
    report = result.backtest_report

    run_row = RunRow(
        id=run_id,
        started_at=_ts(m.started_at),
        seed=m.seed,
        code_sha=m.code_sha,
        data_hash=m.data_hash,
        train_start=_ts(m.train_window[0]),
        train_end=_ts(m.train_window[1]),
        validation_start=_ts(m.validation_window[0]),
        validation_end=_ts(m.validation_window[1]),
        test_start=_ts(m.test_window[0]),
        test_end=_ts(m.test_window[1]),
        config_snapshot_json=json.dumps(m.config_snapshot),
        chosen_strategy_id=report.chosen_strategy_id,
        overfitting_gap=report.overfitting_gap,
        train_metrics_json=_metrics_to_json(report.train_metrics),
        validation_metrics_json=_metrics_to_json(report.validation_metrics),
        test_metrics_json=_metrics_to_json(report.test_metrics),
        buy_and_hold_test_json=_metrics_to_json(report.buy_and_hold_test),
        random_entry_test_json=_metrics_to_json(report.random_entry_test),
    )

    for snap in report.per_generation:
        run_row.generations.append(
            GenerationRow(
                number=snap.generation,
                train_max_fitness=snap.train_metrics.max_fitness,
                train_median_fitness=snap.train_metrics.median_fitness,
                train_mean_fitness=snap.train_metrics.mean_fitness,
                train_n_strategies_with_trades=snap.train_metrics.n_strategies_with_trades,
                validation_max_fitness=snap.validation_metrics.max_fitness,
                validation_median_fitness=snap.validation_metrics.median_fitness,
                validation_mean_fitness=snap.validation_metrics.mean_fitness,
                validation_n_strategies_with_trades=snap.validation_metrics.n_strategies_with_trades,
            )
        )

    for rank, row in result.final_ranking.iterrows():
        strat = row["strategy"]
        run_row.strategies.append(
            StrategyRow(
                id=strat["id"],
                rank=int(rank),
                fitness=float(row["fitness"]),
                indicators_json=json.dumps(strat["indicators"]),
                conjunctions_json=json.dumps(strat["conjunctions"]),
            )
        )

    with Session(engine) as session:
        session.add(run_row)
        session.commit()

    return run_id


def load_run(run_id: str, engine: Engine) -> RunResult:
    with Session(engine) as session:
        run_row = session.get(RunRow, run_id)
        if run_row is None:
            raise LookupError(f"run {run_id!r} not found")

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
                for s in sorted(run_row.strategies, key=lambda r: r.rank)
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
                "seed": r.seed,
                "chosen_strategy_id": r.chosen_strategy_id,
                "overfitting_gap": r.overfitting_gap,
                "code_sha": r.code_sha,
                "data_hash": r.data_hash,
            }
            for r in session.scalars(stmt)
        ]
