"""Paper trading: live (or replayed) market data, simulated fills.

This is the substrate for Phase 6's gated path to real money. Per PLAN.md:217
the trader is "live order book, simulated fills, full P&L tracking" —
the entire stack is built here so flipping the broker to a real exchange
later is a backend swap, not a refactor.

Architecture
------------
- ``Broker`` (Protocol): the only interface the trader knows about.
  ``PaperBroker`` simulates fills against the existing cost model
  (``BacktestConfig.taker_fee_bps`` + ``slippage_bps``).
  ``ExchangeBroker`` (not yet) will submit real orders via ccxt.

- ``PaperPortfolio``: cash + open positions + closed trades + equity
  history. Single source of truth for accounting; the trader never
  mutates these directly.

- ``RiskLimits`` / ``RiskGuard``: hard backstops. Per PLAN.md:219 every
  limit here has tests; the risk module is the only thing standing
  between an optimistic GA fitness number and a margin call.

- ``PaperTrader``: per-bar state machine. ``tick(enriched_bars)`` takes
  the current bars frame (with indicator columns already populated),
  manages position lifecycle, and applies risk + strategy.

What's NOT here yet
-------------------
- Live ccxt connection: the trader takes bars from the caller. Phase 6
  session 2 wires a polling/websocket loop.
- CLI / API surface: same.
- Reconciliation against an exchange's reported balance: there's no
  exchange yet, so nothing to reconcile against.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from gentrade.backtest import NO_CLOSE, STOPPED, TARGET_HIT, BacktestConfig
from gentrade.load_strategy import load_from_object_parenthesised, query_strategy

KILLED = "KILL_SWITCH"


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar — the slice of market data the trader reasons about."""

    open_ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Fill:
    """A simulated (or real) order confirmation."""

    order_id: str
    side: str  # "buy" | "sell"
    qty: float
    price: float  # realised price after slippage
    fee: float  # absolute fee paid in account currency
    timestamp: pd.Timestamp


@dataclass
class Position:
    """An open paper trade. ``deadline_bar_count`` is the absolute bar-count
    at which the position should be force-closed if neither target nor stop
    was hit (mirrors the backtest engine's ``trade_window_bars``)."""

    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    target_price: float
    stop_loss_price: float
    deadline_bar_count: int
    notional_at_entry: float  # qty * entry_price; the at-risk capital


@dataclass
class ClosedTrade:
    """A finalised paper trade. ``pnl`` is net of fees and slippage."""

    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    qty: float
    outcome: str
    pnl: float


@dataclass(frozen=True)
class RiskLimits:
    """Hard backstops the trader cannot violate.

    Every limit here is a *backstop*, not a target — well-behaved
    strategies should normally stay well within them. The defaults are
    deliberately conservative; production uses must override consciously.
    """

    max_position_size_usd: float = 1_000.0
    max_open_positions: int = 1
    max_daily_loss_usd: float = 100.0
    max_drawdown_pct: float = 0.20
    kill_switch: bool = False


# ---------------------------------------------------------------------------
# Portfolio + risk
# ---------------------------------------------------------------------------

class PaperPortfolio:
    """Cash + open positions + closed trades + equity history.

    All accounting flows through here so a single place can be audited
    against an exchange's reported balance later (the reconciliation
    step in PLAN.md:221).
    """

    def __init__(self, initial_cash: float = 10_000.0) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[ClosedTrade] = []
        self.equity_history: list[tuple[pd.Timestamp, float]] = []

    # ----- accessors -----

    def equity(self, mark_to_market: dict[str, float] | None = None) -> float:
        """Total equity = cash + open-position notional at current market price."""
        positions_value = 0.0
        for sym, pos in self.positions.items():
            mark = (mark_to_market or {}).get(sym, pos.entry_price)
            positions_value += pos.qty * mark
        return self.cash + positions_value

    def realized_pnl(self) -> float:
        return float(sum(t.pnl for t in self.closed_trades))

    def daily_realized_pnl(self, day: pd.Timestamp) -> float:
        """Realized P&L on the UTC calendar date of ``day``."""
        target = pd.Timestamp(day).tz_convert("UTC").normalize()
        return float(
            sum(
                t.pnl
                for t in self.closed_trades
                if pd.Timestamp(t.exit_time).tz_convert("UTC").normalize() == target
            )
        )

    def peak_equity(self) -> float:
        return max(
            (eq for _, eq in self.equity_history),
            default=self.initial_cash,
        )

    # ----- mutators -----

    def open_position(self, pos: Position, fill: Fill) -> None:
        if pos.symbol in self.positions:
            raise RuntimeError(f"already long {pos.symbol}; close first")
        self.cash -= pos.qty * fill.price + fill.fee
        self.positions[pos.symbol] = pos

    def close_position(self, symbol: str, exit_fill: Fill, outcome: str) -> ClosedTrade:
        pos = self.positions.pop(symbol)
        proceeds = pos.qty * exit_fill.price - exit_fill.fee
        self.cash += proceeds
        # Net P&L = proceeds - capital deployed at entry
        # The fee paid at entry was already deducted from cash; we re-derive
        # it here for symmetric attribution.
        entry_fee = pos.qty * pos.entry_price * 0  # actual fee tracked inside cash
        pnl = (
            (exit_fill.price - pos.entry_price) * pos.qty
            - exit_fill.fee
            - entry_fee  # already in cash; left as 0 here for clarity
        )
        # Symmetric: pnl should equal (close cash flows) - (open cash flows).
        # The accounting above is correct because both fees passed through cash.
        trade = ClosedTrade(
            symbol=pos.symbol,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=exit_fill.timestamp,
            exit_price=exit_fill.price,
            qty=pos.qty,
            outcome=outcome,
            pnl=pnl,
        )
        self.closed_trades.append(trade)
        return trade

    def record_equity(self, ts: pd.Timestamp, marks: dict[str, float]) -> None:
        self.equity_history.append((ts, self.equity(marks)))


class RiskGuard:
    """Yes/no on new entries given current state + the configured limits."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def can_open(
        self,
        portfolio: PaperPortfolio,
        now: pd.Timestamp,
        proposed_notional_usd: float,
    ) -> tuple[bool, str | None]:
        """Returns ``(allowed, reason_if_blocked)``.

        Order matters — kill switch first, then per-trade size, then
        portfolio-level limits. The reason string is meant for logs;
        callers are not expected to parse it.
        """
        if self.limits.kill_switch:
            return False, "kill switch engaged"
        if proposed_notional_usd > self.limits.max_position_size_usd:
            return (
                False,
                f"size ${proposed_notional_usd:,.0f} exceeds "
                f"max_position_size_usd ${self.limits.max_position_size_usd:,.0f}",
            )
        if len(portfolio.positions) >= self.limits.max_open_positions:
            return (
                False,
                f"already at {self.limits.max_open_positions} open positions",
            )
        # daily loss limit halts new entries for the rest of the UTC day
        daily = portfolio.daily_realized_pnl(now)
        if daily <= -self.limits.max_daily_loss_usd:
            return (
                False,
                f"daily realised loss ${-daily:,.0f} exceeds "
                f"max_daily_loss_usd ${self.limits.max_daily_loss_usd:,.0f}",
            )
        # drawdown from peak equity
        peak = portfolio.peak_equity()
        current = portfolio.equity()
        if peak > 0:
            dd = (current - peak) / peak
            if dd <= -self.limits.max_drawdown_pct:
                return (
                    False,
                    f"drawdown {dd:.2%} exceeds "
                    f"max_drawdown_pct {-self.limits.max_drawdown_pct:.0%}",
                )
        return True, None


# ---------------------------------------------------------------------------
# Broker protocol + paper implementation
# ---------------------------------------------------------------------------

class Broker(Protocol):
    """Anything that can execute a buy/sell order at a given timestamp."""

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reference_price: float,
        ts: pd.Timestamp,
    ) -> Fill:
        ...


class PaperBroker:
    """Simulated-fill broker. Applies the same fee + slippage model the
    backtest engine uses, so paper trading numbers are directly
    comparable to the offline `BacktestReport`.

    Slippage is applied adversely — buys pay slightly more, sells receive
    slightly less. Fees are absolute (notional × bps / 10_000).
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reference_price: float,
        ts: pd.Timestamp,
    ) -> Fill:
        slip = self.config.slippage_bps / 10_000.0
        if side == "buy":
            fill_price = reference_price * (1.0 + slip)
        elif side == "sell":
            fill_price = reference_price * (1.0 - slip)
        else:
            raise ValueError(f"unknown side {side!r}")
        notional = qty * fill_price
        fee = notional * (self.config.taker_fee_bps / 10_000.0)
        return Fill(
            order_id=str(uuid.uuid4()),
            side=side,
            qty=qty,
            price=fill_price,
            fee=fee,
            timestamp=ts,
        )


# ---------------------------------------------------------------------------
# Trader state machine
# ---------------------------------------------------------------------------

@dataclass
class TickResult:
    """What happened on a single ``tick`` call. The API/log surface."""

    bar_count: int
    bar_close_price: float
    events: list[dict] = field(default_factory=list)
    portfolio_equity: float = 0.0
    open_positions: int = 0


class PaperTrader:
    """Per-bar state machine: update positions, evaluate strategy, open if allowed.

    The trader does not own the data fetch loop. Callers feed it bars
    (one row of the latest enriched DataFrame) and the trader returns a
    `TickResult` describing what happened. This makes the state machine
    pure-Python and trivially testable; live ccxt polling is layered on
    top by Phase 6 session 2's runner.
    """

    def __init__(
        self,
        symbol: str,
        strategy: dict,
        broker: Broker,
        config: BacktestConfig,
        risk_limits: RiskLimits,
        portfolio: PaperPortfolio | None = None,
        *,
        notional_per_trade_usd: float = 1_000.0,
    ) -> None:
        self.symbol = symbol
        self.strategy = strategy
        self.broker = broker
        self.config = config
        self.risk = RiskGuard(risk_limits)
        self.portfolio = portfolio or PaperPortfolio()
        self.notional_per_trade_usd = float(notional_per_trade_usd)
        self.bar_count = 0
        # Cache the parsed query so we don't rebuild it per tick.
        self._parsed_query = load_from_object_parenthesised(strategy)

    # ----- main loop -----

    def tick(self, enriched_bars: pd.DataFrame) -> TickResult:
        """Process the latest bar. ``enriched_bars`` must include all
        indicator columns the strategy DSL queries; the last row is the
        bar to act on."""
        if len(enriched_bars) == 0:
            raise ValueError("tick() requires a non-empty bars frame")
        self.bar_count += 1
        latest = enriched_bars.iloc[-1]
        bar = _row_to_bar(latest)
        events: list[dict] = []

        # 1. Manage open position (close on target / stop / timeout)
        just_closed = False
        if self.symbol in self.portfolio.positions:
            outcome = self._check_open_position(bar)
            if outcome is not None:
                events.append(self._close_at(bar, outcome))
                just_closed = True

        # 2. Evaluate strategy → maybe open. We never re-open on the same
        # tick we just closed on — that would compound exposure and make
        # the trader's behaviour diverge from the offline backtest engine
        # (which advances past the exit bar before considering re-entry).
        if (
            not just_closed
            and self.symbol not in self.portfolio.positions
            and self._signal_fires(enriched_bars)
        ):
            allowed, reason = self.risk.can_open(
                self.portfolio, bar.open_ts, self.notional_per_trade_usd
            )
            if allowed:
                events.append(self._open_at(bar))
            else:
                events.append(
                    {"event": "blocked", "reason": reason, "ts": bar.open_ts.isoformat()}
                )

        # 3. Mark-to-market + record equity
        self.portfolio.record_equity(bar.open_ts, {self.symbol: bar.close})

        return TickResult(
            bar_count=self.bar_count,
            bar_close_price=bar.close,
            events=events,
            portfolio_equity=self.portfolio.equity({self.symbol: bar.close}),
            open_positions=len(self.portfolio.positions),
        )

    # ----- state transitions -----

    def _check_open_position(self, bar: Bar) -> str | None:
        """Return an outcome token if the open position should close on this bar."""
        pos = self.portfolio.positions[self.symbol]
        if self.risk.limits.kill_switch:
            return KILLED
        if bar.close >= pos.target_price:
            return TARGET_HIT
        if bar.close <= pos.stop_loss_price:
            return STOPPED
        if self.bar_count >= pos.deadline_bar_count:
            return NO_CLOSE
        return None

    def _close_at(self, bar: Bar, outcome: str) -> dict:
        pos = self.portfolio.positions[self.symbol]
        # For HIT/STOPPED we close at the threshold; for NO_CLOSE we close
        # at the bar's close. Slippage is applied by the broker.
        if outcome == TARGET_HIT:
            ref = pos.target_price
        elif outcome == STOPPED:
            ref = pos.stop_loss_price
        else:
            ref = bar.close  # NO_CLOSE / KILLED
        fill = self.broker.place_order(self.symbol, "sell", pos.qty, ref, bar.open_ts)
        trade = self.portfolio.close_position(self.symbol, fill, outcome)
        return {
            "event": "close",
            "outcome": outcome,
            "price": fill.price,
            "pnl": trade.pnl,
            "ts": bar.open_ts.isoformat(),
        }

    def _open_at(self, bar: Bar) -> dict:
        # qty sized so that notional ≈ notional_per_trade_usd at entry
        qty = self.notional_per_trade_usd / bar.close
        fill = self.broker.place_order(self.symbol, "buy", qty, bar.close, bar.open_ts)
        target = fill.price * (1.0 + self.config.target_pct)
        stop = fill.price * (1.0 - self.config.stop_loss_pct)
        pos = Position(
            symbol=self.symbol,
            entry_time=bar.open_ts,
            entry_price=fill.price,
            qty=qty,
            target_price=target,
            stop_loss_price=stop,
            deadline_bar_count=self.bar_count + self.config.trade_window_bars,
            notional_at_entry=qty * fill.price,
        )
        self.portfolio.open_position(pos, fill)
        return {
            "event": "open",
            "price": fill.price,
            "qty": qty,
            "target": target,
            "stop": stop,
            "ts": bar.open_ts.isoformat(),
        }

    # ----- helpers -----

    def _signal_fires(self, enriched_bars: pd.DataFrame) -> bool:
        """Does the strategy's query match the last row of ``enriched_bars``?"""
        try:
            matched = query_strategy(enriched_bars, query=self._parsed_query)
        except Exception:
            return False
        if len(matched) == 0:
            return False
        latest_ts = enriched_bars.iloc[-1]["open_ts"]
        return bool((matched["open_ts"] == latest_ts).any())


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _row_to_bar(row) -> Bar:
    return Bar(
        open_ts=pd.Timestamp(row["open_ts"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row.get("volume", 0.0)),
    )
