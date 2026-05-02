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
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

_ALLOWED_OPS = (">=", "<=", ">", "<")


class IndicatorSpec(BaseModel):
    """One signal in a user-seeded strategy.

    Server-side validation against the trusted catalogue is what makes
    this safe — the indicator name is interpolated into a pandas query
    string. Only ``absolute=True`` strategies are accepted from the
    wire today; relative comparators (MA / PREVIOUS_PERIOD / sibling)
    will land in a follow-up.
    """

    indicator: str
    op: Literal[">=", "<=", ">", "<"]
    absolute: bool = True
    abs_value: float | None = None


class StrategySpec(BaseModel):
    """One user-seeded strategy. ``conjunctions`` joins adjacent indicators
    so ``len(conjunctions) == len(indicators) - 1``. Per the
    FirstConjunctionIsAnd spec invariant, ``conjunctions[0]`` (if any)
    must be ``"and"`` so the parsed query string parenthesises cleanly.
    """

    indicators: list[IndicatorSpec] = Field(min_length=1, max_length=8)
    conjunctions: list[Literal["and", "or"]] = Field(default_factory=list)


class CatalogueIndicatorOut(BaseModel):
    """One entry in the indicator catalogue exposed via GET /catalogue.

    The UI uses this to populate dropdowns. ``ops`` is the operators the
    legacy catalogue pairs this indicator with; ``absolute_thresholds``
    is the canonical list of numeric thresholds (helpful as defaults in
    the UI). ``type`` is the signal class (momentum / trend / volatility
    / volume).
    """

    indicator: str
    type: str
    ops: list[str]
    absolute_thresholds: list[float]
    relative_targets: list[str]


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

    # Optional seed strategies. Each is validated against the trusted
    # catalogue server-side; the rest of the population is auto-generated
    # to fill ``population_size``. None or empty list ⇒ pure auto-generate.
    seed_strategies: list[StrategySpec] | None = None

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


class BarOut(BaseModel):
    """A single OHLCV bar for UI candlestick rendering.

    Server-side downsampled to a manageable count via proper OHLC
    resampling (open=first, high=max, low=min, close=last) — sending
    the raw 16k-row test slice to a browser is wasteful and slow.
    """

    open_ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class TradeOut(BaseModel):
    """A single trade from a window's evaluation, surfaced for UI charts."""

    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    target_price: float | None = None
    stop_loss_price: float | None = None
    outcome: str
    return_: float = Field(alias="return")
    model_config = {"populate_by_name": True}


class BacktestResponse(BaseModel):
    chosen_strategy_id: str
    train_metrics: PerformanceMetricsOut
    validation_metrics: PerformanceMetricsOut
    test_metrics: PerformanceMetricsOut
    buy_and_hold_test: PerformanceMetricsOut
    random_entry_test: PerformanceMetricsOut
    overfitting_gap: float
    # Per-trade detail for the test window so the UI can render equity
    # curve, drawdown, and trade-scatter charts. Train/validation trades
    # are intentionally not exposed yet — too easy to read them as the
    # number to optimise against, which is the bug Phase 1 closed.
    test_trades: list[TradeOut] = Field(default_factory=list)
    # Test-window OHLC for the candlestick chart. Server-side
    # downsampled to ≤1000 bars (proper OHLC resampling) so the UI
    # can render without choking.
    test_bars: list[BarOut] = Field(default_factory=list)


class AssetOut(BaseModel):
    asset: str
    exchange: str
    interval: str


class CrossAssetRequest(BaseModel):
    """Re-evaluate a saved strategy on a list of registered assets.

    Each ``asset`` is a name from the server-side registry — same trust
    boundary as POST /runs. The strategy is loaded from the DB by
    ``(run_id, strategy_id)``; nothing is accepted from the wire that
    could feed the pandas query string.
    """

    run_id: str
    strategy_id: str
    assets: list[str] = Field(min_length=1, max_length=20)


class CrossAssetRowOut(BaseModel):
    asset: str
    n_bars: int
    metrics: PerformanceMetricsOut
    error: str | None = None


class CrossAssetResponse(BaseModel):
    chosen_strategy_id: str
    base_asset: str | None
    rows: list[CrossAssetRowOut]
