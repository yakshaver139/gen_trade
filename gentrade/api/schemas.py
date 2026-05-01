"""Pydantic request/response schemas for the API.

These deliberately do not allow user-supplied strategy chromosomes — the
strategy DSL builds a `pandas.DataFrame.query()` string, and accepting
arbitrary indicator + threshold combinations from the wire would expose
us to expression-injection-shaped bugs (see PLAN.md:153). For now, all
strategies are server-generated. POST /backtests re-runs a strategy that
the server already knows about (looked up by `(run_id, strategy_id)`).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CreateRunRequest(BaseModel):
    """Body of POST /runs.

    ``asset`` must be a registered asset (see assets.py). ``population_size``
    is also the number of strategies the server generates.
    """

    asset: str
    population_size: int = Field(ge=2, le=200, default=10)
    generations: int = Field(ge=1, le=500, default=50)
    elitism_count: int = Field(ge=0, le=10, default=2)
    selection_pressure: str = "tournament"
    seed: int = 0
    target_pct: float = Field(gt=0, lt=1, default=0.015)
    stop_loss_pct: float = Field(gt=0, lt=1, default=0.0075)
    trade_window_bars: int = Field(ge=1, le=10_000, default=96)
    taker_fee_bps: float = Field(ge=0, le=1000, default=10.0)
    slippage_bps: float = Field(ge=0, le=1000, default=1.0)

    @field_validator("selection_pressure")
    @classmethod
    def _valid_pressure(cls, v: str) -> str:
        from gentrade.selection import SELECTION_PRESSURES
        if v not in SELECTION_PRESSURES:
            raise ValueError(f"selection_pressure must be one of {SELECTION_PRESSURES}")
        return v


class CreateRunResponse(BaseModel):
    run_id: str
    status: str
    status_url: str


class WindowOut(BaseModel):
    start: datetime
    end: datetime


class ManifestOut(BaseModel):
    seed: int
    code_sha: str | None
    data_hash: str | None
    started_at: datetime
    train_window: WindowOut
    validation_window: WindowOut
    test_window: WindowOut
    config_snapshot: dict[str, Any]


class GenerationMetricsOut(BaseModel):
    max_fitness: float
    median_fitness: float
    mean_fitness: float
    n_strategies_with_trades: int


class GenerationSummaryOut(BaseModel):
    number: int
    train_metrics: GenerationMetricsOut
    validation_metrics: GenerationMetricsOut


class PerformanceMetricsOut(BaseModel):
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    avg_trade_duration_ns: int


class RunSummaryOut(BaseModel):
    id: str
    status: str
    current_generation: int
    started_at: datetime
    finished_at: datetime | None
    seed: int
    chosen_strategy_id: str | None
    overfitting_gap: float | None


class RunDetailOut(BaseModel):
    id: str
    status: str
    current_generation: int
    manifest: ManifestOut
    generations: list[GenerationSummaryOut]
    chosen_strategy_id: str | None
    overfitting_gap: float | None
    train_metrics: PerformanceMetricsOut | None
    validation_metrics: PerformanceMetricsOut | None
    test_metrics: PerformanceMetricsOut | None
    buy_and_hold_test: PerformanceMetricsOut | None
    random_entry_test: PerformanceMetricsOut | None


class StrategyOut(BaseModel):
    run_id: str
    generation_number: int
    id: str
    rank: int
    fitness: float | None
    indicators: list[dict[str, Any]]
    conjunctions: list[str]
    parsed_query: str


class GenerationDetailOut(BaseModel):
    run_id: str
    number: int
    train_metrics: GenerationMetricsOut
    validation_metrics: GenerationMetricsOut
    strategies: list[StrategyOut]


class BacktestRequest(BaseModel):
    """Re-run a saved strategy on the run's original windows.

    ``train_window``/``validation_window``/``test_window`` are intentionally
    NOT exposed yet — Phase 5 will add cross-asset / cross-window
    re-evaluation, with whitelisted asset registration. For now, the only
    supported re-run is on the same windows the strategy came from.
    """

    run_id: str
    strategy_id: str


class BacktestResponse(BaseModel):
    chosen_strategy_id: str
    train_metrics: PerformanceMetricsOut
    validation_metrics: PerformanceMetricsOut
    test_metrics: PerformanceMetricsOut
    buy_and_hold_test: PerformanceMetricsOut
    random_entry_test: PerformanceMetricsOut
    overfitting_gap: float


class AssetOut(BaseModel):
    asset: str
    exchange: str
    interval: str
