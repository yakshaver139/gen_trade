"""FastAPI app exposing GA runs, generations, and ad-hoc backtests.

Auth: every non-public endpoint depends on `require_api_key`, which checks
`X-API-Key` against `GENTRADE_API_KEY` (env) in constant time.

Inputs are deliberately constrained:
- Data sources are looked up by asset name from a server-side registry —
  no user-supplied file paths.
- Strategies are not accepted from clients. POST /backtests refers to a
  saved strategy by `(run_id, strategy_id)`. The strategy DSL builds a
  `pandas.DataFrame.query()` string, and accepting arbitrary chromosomes
  from the wire would expose us to expression-injection-shaped bugs.

The app is built behind a factory (`create_app`) so tests can wire their
own engine, asset registry, and log directory cleanly without poking
module globals.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from gentrade.api import assets as asset_registry
from gentrade.api.auth import require_api_key
from gentrade.api.jobs import prepare_run, run_in_background
from gentrade.api.schemas import (
    AssetOut,
    BacktestRequest,
    BacktestResponse,
    CreateRunRequest,
    CreateRunResponse,
    GenerationDetailOut,
    GenerationMetricsOut,
    GenerationSummaryOut,
    ManifestOut,
    PerformanceMetricsOut,
    RunDetailOut,
    RunSummaryOut,
    StrategyOut,
    TradeOut,
    WindowOut,
)
from gentrade.backtest import BacktestConfig
from gentrade.ingest import load_bars as _load_bars_for_backtest
from gentrade.load_strategy import load_from_object_parenthesised
from gentrade.metrics import PerformanceMetrics
from gentrade.persistence import (
    RunRow,
    StrategyRow,
    list_runs,
)
from gentrade.walk_forward import produce_backtest_report


def _metrics_out(m: PerformanceMetrics | None) -> PerformanceMetricsOut | None:
    if m is None:
        return None
    return PerformanceMetricsOut(
        n_trades=m.n_trades,
        win_rate=m.win_rate,
        profit_factor=m.profit_factor,
        expectancy=m.expectancy,
        sharpe=m.sharpe,
        sortino=m.sortino,
        calmar=m.calmar,
        max_drawdown=m.max_drawdown,
        avg_trade_duration_ns=int(m.avg_trade_duration.value),
    )


def _metrics_from_json_or_none(s: str | None) -> PerformanceMetrics | None:
    if s is None:
        return None
    from gentrade.persistence import _metrics_from_json

    return _metrics_from_json(s)


def _utc(value) -> pd.Timestamp:
    """Database timestamps come back tz-naive; coerce to UTC."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def create_app(
    engine: Engine,
    log_dir: Path | str = "runs",
    title: str = "gentrade",
) -> FastAPI:
    log_dir_path = Path(log_dir)
    app = FastAPI(title=title, version="0.1.0")

    # ----- public endpoints -----

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # ----- protected endpoints -----

    auth = Depends(require_api_key)

    @app.get("/assets", response_model=list[AssetOut])
    def get_assets(_: None = auth) -> list[AssetOut]:
        return [
            AssetOut(asset=e.asset, exchange=e.exchange, interval=e.interval)
            for e in asset_registry.list_assets()
        ]

    @app.post(
        "/runs",
        response_model=CreateRunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def post_run(body: CreateRunRequest, _: None = auth) -> CreateRunResponse:
        try:
            spec, run_id = prepare_run(body, engine, log_dir_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        run_in_background(spec, engine, log_dir_path)
        return CreateRunResponse(
            run_id=run_id, status="in_progress", status_url=f"/runs/{run_id}"
        )

    @app.get("/runs", response_model=list[RunSummaryOut])
    def get_runs(_: None = auth) -> list[RunSummaryOut]:
        return [
            RunSummaryOut(
                id=r["id"],
                status=r["status"],
                current_generation=r["current_generation"],
                started_at=r["started_at"],
                finished_at=r["finished_at"],
                seed=r["seed"],
                chosen_strategy_id=r["chosen_strategy_id"],
                overfitting_gap=r["overfitting_gap"],
            )
            for r in list_runs(engine)
        ]

    @app.get("/runs/{run_id}", response_model=RunDetailOut)
    def get_run(run_id: str, _: None = auth) -> RunDetailOut:
        with Session(engine) as session:
            row = session.get(RunRow, run_id)
            if row is None:
                raise HTTPException(status_code=404, detail=f"run {run_id} not found")
            manifest = ManifestOut(
                seed=row.seed,
                code_sha=row.code_sha,
                data_hash=row.data_hash,
                started_at=_utc(row.started_at),
                train_window=WindowOut(
                    start=_utc(row.train_start), end=_utc(row.train_end)
                ),
                validation_window=WindowOut(
                    start=_utc(row.validation_start), end=_utc(row.validation_end)
                ),
                test_window=WindowOut(
                    start=_utc(row.test_start), end=_utc(row.test_end)
                ),
                config_snapshot=json.loads(row.config_snapshot_json),
            )
            generations = [
                GenerationSummaryOut(
                    number=g.number,
                    train_metrics=GenerationMetricsOut(
                        max_fitness=g.train_max_fitness,
                        median_fitness=g.train_median_fitness,
                        mean_fitness=g.train_mean_fitness,
                        n_strategies_with_trades=g.train_n_strategies_with_trades,
                    ),
                    validation_metrics=GenerationMetricsOut(
                        max_fitness=g.validation_max_fitness,
                        median_fitness=g.validation_median_fitness,
                        mean_fitness=g.validation_mean_fitness,
                        n_strategies_with_trades=g.validation_n_strategies_with_trades,
                    ),
                )
                for g in sorted(row.generations, key=lambda x: x.number)
            ]
            return RunDetailOut(
                id=row.id,
                status=row.status,
                current_generation=row.current_generation,
                manifest=manifest,
                generations=generations,
                chosen_strategy_id=row.chosen_strategy_id,
                overfitting_gap=row.overfitting_gap,
                train_metrics=_metrics_out(_metrics_from_json_or_none(row.train_metrics_json)),
                validation_metrics=_metrics_out(_metrics_from_json_or_none(row.validation_metrics_json)),
                test_metrics=_metrics_out(_metrics_from_json_or_none(row.test_metrics_json)),
                buy_and_hold_test=_metrics_out(_metrics_from_json_or_none(row.buy_and_hold_test_json)),
                random_entry_test=_metrics_out(_metrics_from_json_or_none(row.random_entry_test_json)),
            )

    @app.get("/runs/{run_id}/events")
    async def get_run_events(run_id: str, _: None = auth) -> StreamingResponse:
        """Server-sent events stream of progress updates for an in-progress run.

        Emits one ``progress`` event each time the run's
        ``current_generation`` or ``status`` changes, plus a final
        ``terminal`` event when the run reaches ``reported`` or
        ``failed``. Closes immediately if the run is already terminal at
        connection time, or if the run id is unknown.

        ``X-API-Key`` auth is required (same as every other endpoint);
        browsers consuming this from EventSource have to use a fetch +
        ReadableStream shim because EventSource cannot set custom headers.
        """
        # Verify the run exists up-front so we return 404 cleanly rather
        # than silently emitting an empty stream.
        with Session(engine) as session:
            row = session.get(RunRow, run_id)
            if row is None:
                raise HTTPException(status_code=404, detail=f"run {run_id} not found")

        async def event_stream():
            last_gen = -1
            last_status: str | None = None
            poll_interval = 1.0
            # Up to ~10 minutes of inactivity before we close to free the
            # connection. Live runs rarely sit idle for that long; a
            # crashed worker would otherwise hold this connection open.
            max_idle_polls = 600
            idle = 0
            while True:
                with Session(engine) as session:
                    row = session.get(RunRow, run_id)
                    if row is None:
                        yield "event: error\ndata: {\"detail\":\"run vanished\"}\n\n"
                        return
                    current_gen = row.current_generation
                    current_status = row.status

                changed = current_gen != last_gen or current_status != last_status
                if changed:
                    payload = json.dumps(
                        {"current_generation": current_gen, "status": current_status}
                    )
                    yield f"event: progress\ndata: {payload}\n\n"
                    last_gen = current_gen
                    last_status = current_status
                    idle = 0
                else:
                    idle += 1

                if current_status in ("reported", "failed"):
                    yield (
                        "event: terminal\n"
                        f"data: {json.dumps({'status': current_status})}\n\n"
                    )
                    return

                if idle >= max_idle_polls:
                    yield "event: timeout\ndata: {}\n\n"
                    return

                await asyncio.sleep(poll_interval)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get(
        "/runs/{run_id}/generations/{n}",
        response_model=GenerationDetailOut,
    )
    def get_generation(run_id: str, n: int, _: None = auth) -> GenerationDetailOut:
        with Session(engine) as session:
            run = session.get(RunRow, run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"run {run_id} not found")
            gen = next((g for g in run.generations if g.number == n), None)
            if gen is None:
                raise HTTPException(
                    status_code=404, detail=f"generation {n} not found in run {run_id}"
                )
            strategies = sorted(
                [s for s in run.strategies if s.generation_number == n],
                key=lambda s: s.rank,
            )
            return GenerationDetailOut(
                run_id=run_id,
                number=n,
                train_metrics=GenerationMetricsOut(
                    max_fitness=gen.train_max_fitness,
                    median_fitness=gen.train_median_fitness,
                    mean_fitness=gen.train_mean_fitness,
                    n_strategies_with_trades=gen.train_n_strategies_with_trades,
                ),
                validation_metrics=GenerationMetricsOut(
                    max_fitness=gen.validation_max_fitness,
                    median_fitness=gen.validation_median_fitness,
                    mean_fitness=gen.validation_mean_fitness,
                    n_strategies_with_trades=gen.validation_n_strategies_with_trades,
                ),
                strategies=[_strategy_out(run_id, s) for s in strategies],
            )

    @app.get(
        "/runs/{run_id}/strategies/{strategy_id}",
        response_model=StrategyOut,
    )
    def get_strategy(run_id: str, strategy_id: str, _: None = auth) -> StrategyOut:
        with Session(engine) as session:
            stmt = select(StrategyRow).where(
                StrategyRow.run_id == run_id, StrategyRow.id == strategy_id
            )
            rows = session.scalars(stmt).all()
            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail=f"strategy {strategy_id} not found in run {run_id}",
                )
            # Prefer the latest-evaluated occurrence (highest generation_number).
            row = max(rows, key=lambda s: s.generation_number)
            return _strategy_out(run_id, row)

    @app.post("/backtests", response_model=BacktestResponse)
    def post_backtest(body: BacktestRequest, _: None = auth) -> BacktestResponse:
        with Session(engine) as session:
            run = session.get(RunRow, body.run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"run {body.run_id} not found")
            stmt = select(StrategyRow).where(
                StrategyRow.run_id == body.run_id, StrategyRow.id == body.strategy_id
            )
            strategy_rows = session.scalars(stmt).all()
            if not strategy_rows:
                raise HTTPException(
                    status_code=404,
                    detail=f"strategy {body.strategy_id} not in run {body.run_id}",
                )
            strategy_row = max(strategy_rows, key=lambda s: s.generation_number)
            train = (_utc(run.train_start), _utc(run.train_end))
            val = (_utc(run.validation_start), _utc(run.validation_end))
            test = (_utc(run.test_start), _utc(run.test_end))
            cfg = json.loads(run.config_snapshot_json).get("backtest", {})

        bt_cfg = BacktestConfig(**cfg) if cfg else BacktestConfig()
        # Resolve the asset from the manifest's recorded path is not possible —
        # the path isn't persisted (only the data_hash is). Re-running ad-hoc
        # backtests requires the same asset to still be in the registry; the
        # caller is responsible for keeping bars on disk. Phase 5 will tighten this.
        asset_lookup = json.loads(run.config_snapshot_json).get("asset")
        entry = asset_registry.resolve(asset_lookup) if asset_lookup else None
        if entry is None:
            # Refuse rather than silently substitute a different dataset —
            # backtests on the wrong asset are misleading financial metrics,
            # which in a trading context is a real integrity issue.
            raise HTTPException(
                status_code=422,
                detail=(
                    f"cannot rerun backtest: asset {asset_lookup!r} for run "
                    f"{body.run_id} is not in the current registry"
                ),
            )
        bars = _load_bars_for_backtest(entry.path)

        strategy = {
            "id": strategy_row.id,
            "indicators": json.loads(strategy_row.indicators_json),
            "conjunctions": json.loads(strategy_row.conjunctions_json),
        }
        report = produce_backtest_report(
            bars=bars,
            chosen_strategy=strategy,
            train_window=train,
            validation_window=val,
            test_window=test,
            config=bt_cfg,
            seed=run.seed,
        )
        # Re-evaluate the chosen strategy on the test window to surface
        # individual trades for the UI's equity / drawdown / scatter charts.
        from gentrade.walk_forward import evaluate_strategy, slice_window

        test_bars = slice_window(bars, *test)
        test_trades_df = evaluate_strategy(test_bars, strategy, bt_cfg)
        test_trades = [
            _trade_out(row) for _, row in test_trades_df.iterrows()
        ]

        return BacktestResponse(
            chosen_strategy_id=report.chosen_strategy_id,
            train_metrics=_metrics_out(report.train_metrics),
            validation_metrics=_metrics_out(report.validation_metrics),
            test_metrics=_metrics_out(report.test_metrics),
            buy_and_hold_test=_metrics_out(report.buy_and_hold_test),
            random_entry_test=_metrics_out(report.random_entry_test),
            overfitting_gap=report.overfitting_gap if not _isnan(report.overfitting_gap) else 0.0,
            test_trades=test_trades,
        )

    return app


def _trade_out(row) -> TradeOut:
    """Map a row from `evaluate_strategy`'s trades frame into the response shape."""
    target = row.get("target_price")
    stop = row.get("stop_loss_price")
    return TradeOut(
        entry_time=pd.Timestamp(row["entry_time"]).to_pydatetime(),
        entry_price=float(row["entry_price"]),
        exit_time=pd.Timestamp(row["exit_time"]).to_pydatetime(),
        exit_price=float(row["exit_price"]),
        target_price=None if pd.isna(target) else float(target),
        stop_loss_price=None if pd.isna(stop) else float(stop),
        outcome=str(row["outcome"]),
        **{"return": float(row["return"])},
    )


def _strategy_out(run_id: str, row: StrategyRow) -> StrategyOut:
    indicators = json.loads(row.indicators_json)
    conjunctions = json.loads(row.conjunctions_json)
    strategy_dict = {
        "id": row.id,
        "indicators": indicators,
        "conjunctions": conjunctions,
    }
    parsed = load_from_object_parenthesised(strategy_dict)
    return StrategyOut(
        run_id=run_id,
        generation_number=row.generation_number,
        id=row.id,
        rank=row.rank,
        fitness=row.fitness,
        indicators=indicators,
        conjunctions=conjunctions,
        parsed_query=parsed,
    )


def _isnan(x: float) -> bool:
    return x != x  # NaN is the only float that's not equal to itself


# Default factory for `uvicorn gentrade.api.app:default_app --factory`.
# Reads GENTRADE_DB_URL and GENTRADE_LOG_DIR from env.
def default_app() -> FastAPI:
    import os

    from gentrade.persistence import DEFAULT_DB_URL, init_db

    db_url = os.getenv("GENTRADE_DB_URL", DEFAULT_DB_URL)
    log_dir = os.getenv("GENTRADE_LOG_DIR", "runs")
    return create_app(engine=init_db(db_url), log_dir=log_dir)
