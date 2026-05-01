"""Background job runner for the API.

POST /runs returns 202 immediately; the GA loop runs in a daemon thread
that calls ``run_ga`` with the SQLAlchemy engine. Each generation is
checkpointed to the DB before the next one starts, so a process crash
leaves the run resumable from CLI (`gentrade resume <id>`).

The job thread also writes a per-run log file at ``runs/<run_id>/run.log``
so post-mortem debugging doesn't have to grep stdout. The log file is
rotated nowhere — runs typically last hours, not days, and Phase 6 ops
work would handle log retention if we ever needed it.

This is deliberately a thread, not asyncio: the GA's CPU-bound work
would block the event loop, and starting subprocesses for each run is
overkill for a single-machine deployment. If/when we move to many-runs-
per-host we'll graduate to Celery + Redis (PLAN.md:120).
"""
from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from gentrade.api import assets as asset_registry
from gentrade.api.schemas import CreateRunRequest
from gentrade.backtest import BacktestConfig
from gentrade.ga import GAConfig, run_ga
from gentrade.manifest import compute_data_hash
from gentrade.persistence import RunRow, start_persisted_run


@dataclass
class RunSpec:
    """Per-job inputs assembled from the API request."""

    run_id: str
    bars: pd.DataFrame
    initial_strategies: list[dict]
    train_window: tuple[pd.Timestamp, pd.Timestamp]
    validation_window: tuple[pd.Timestamp, pd.Timestamp]
    test_window: tuple[pd.Timestamp, pd.Timestamp]
    ga_config: GAConfig
    backtest_config: BacktestConfig
    seed: int
    code_sha: str | None
    data_hash: str | None


def _setup_run_log(run_id: str, log_dir: Path) -> logging.Logger:
    """Configure (or reconfigure) a per-run logger writing to runs/<id>/run.log."""
    run_dir = log_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"gentrade.run.{run_id}")
    # Avoid duplicate handlers if the function is called twice for the same id
    # (prepare_run sets up; the worker thread re-sets up).
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    logger.setLevel(logging.INFO)
    h = logging.FileHandler(run_dir / "run.log")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


def _split_windows(
    bars: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[
    tuple[pd.Timestamp, pd.Timestamp],
    tuple[pd.Timestamp, pd.Timestamp],
    tuple[pd.Timestamp, pd.Timestamp],
]:
    n = len(bars)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return (
        (bars.iloc[0]["open_ts"], bars.iloc[train_end - 1]["open_ts"]),
        (bars.iloc[train_end]["open_ts"], bars.iloc[val_end - 1]["open_ts"]),
        (bars.iloc[val_end]["open_ts"], bars.iloc[n - 1]["open_ts"]),
    )


def _generate_strategies(population_size: int, seed: int) -> list[dict]:
    """Generate ``population_size`` random strategies from the trusted catalogue.

    Wrapped so tests can monkeypatch with a deterministic, smaller catalogue.
    """
    import random

    from gentrade.generate_strategy import main as gen_main

    random.seed(seed)
    return gen_main(population_size=population_size)


def _load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "open_ts" not in df.columns:
        raise ValueError(f"{path}: missing required 'open_ts' column")
    ts = df["open_ts"]
    if pd.api.types.is_numeric_dtype(ts):
        df["open_ts"] = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        df["open_ts"] = pd.to_datetime(ts, utc=True)
    return df


def prepare_run(
    body: CreateRunRequest, engine: Engine, log_dir: Path
) -> tuple[RunSpec, str]:
    """Resolve the asset, load bars, generate strategies, write the in_progress run row.

    Raises ``ValueError`` for unknown assets so the caller can return 400.
    """
    entry = asset_registry.resolve(body.asset)
    if entry is None:
        raise ValueError(f"unknown asset {body.asset!r}")
    bars = _load_bars(entry.path)
    train, val, test = _split_windows(bars)
    strategies = _generate_strategies(body.population_size, body.seed)
    cfg = GAConfig(
        population_size=body.population_size,
        max_generations=body.generations,
        elitism_count=min(body.elitism_count, body.population_size),
        selection_pressure=body.selection_pressure,
    )
    bt_cfg = BacktestConfig(
        target_pct=body.target_pct,
        stop_loss_pct=body.stop_loss_pct,
        trade_window_bars=body.trade_window_bars,
        taker_fee_bps=body.taker_fee_bps,
        slippage_bps=body.slippage_bps,
    )
    data_hash = compute_data_hash(bars)

    # Capture manifest + start the row so the run_id exists immediately and
    # the API can return it before the thread even starts.
    from dataclasses import asdict

    from gentrade.manifest import capture_manifest

    manifest = capture_manifest(
        seed=body.seed,
        train_window=train,
        validation_window=val,
        test_window=test,
        config_snapshot={
            "ga": asdict(cfg),
            "backtest": asdict(bt_cfg),
            # Persisted so POST /backtests can rerun the strategy on the
            # same asset without falling back to "any registered asset".
            "asset": body.asset,
        },
        code_sha=None,  # set by the worker; see note in start_run
        data_hash=data_hash,
    )
    run_id = start_persisted_run(
        manifest=manifest,
        initial_strategies=strategies,
        engine=engine,
    )
    logger = _setup_run_log(run_id, log_dir)
    logger.info(f"prepared run for asset={body.asset} pop={body.population_size} gens={body.generations}")

    spec = RunSpec(
        run_id=run_id,
        bars=bars,
        initial_strategies=strategies,
        train_window=train,
        validation_window=val,
        test_window=test,
        ga_config=cfg,
        backtest_config=bt_cfg,
        seed=body.seed,
        code_sha=None,
        data_hash=data_hash,
    )
    return spec, run_id


def run_in_background(spec: RunSpec, engine: Engine, log_dir: Path) -> threading.Thread:
    """Start a daemon thread that resumes the prepared run to completion."""

    def _target() -> None:
        logger = _setup_run_log(spec.run_id, log_dir)
        try:
            logger.info(f"starting run {spec.run_id}")
            run_ga(
                bars=spec.bars,
                engine=engine,
                resume_run_id=spec.run_id,
            )
            logger.info(f"finished run {spec.run_id}")
        except Exception:
            logger.error(
                f"run {spec.run_id} failed:\n{traceback.format_exc()}"
            )
            with Session(engine) as session:
                row = session.get(RunRow, spec.run_id)
                if row is not None:
                    row.status = "failed"
                    session.commit()

    t = threading.Thread(target=_target, name=f"gentrade-run-{spec.run_id}", daemon=True)
    t.start()
    return t
