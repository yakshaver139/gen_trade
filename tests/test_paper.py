"""Tests for the paper-trading state machine + risk module.

Per PLAN.md:219, every limit on the risk module is tested. The state
machine tests pin every transition the trader can make: open, target_hit,
stopped_out, no_close (timeout), kill_switch close, blocked_by_size,
blocked_by_max_open, blocked_by_daily_loss, blocked_by_drawdown,
blocked_by_kill_switch.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gentrade.backtest import NO_CLOSE, STOPPED, TARGET_HIT, BacktestConfig
from gentrade.paper import (
    KILLED,
    ClosedTrade,
    PaperBroker,
    PaperPortfolio,
    PaperTrader,
    Position,
    RiskGuard,
    RiskLimits,
)

CFG = BacktestConfig(
    target_pct=0.015,
    stop_loss_pct=0.0075,
    trade_window_bars=4,
    taker_fee_bps=10.0,
    slippage_bps=1.0,
)
ALWAYS_ON = {
    "id": "always-on",
    "indicators": [
        {"absolute": True, "indicator": "close", "op": ">=", "abs_value": 0}
    ],
    "conjunctions": [],
}


def _bars(closes: list[float], start_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    base = start_ts or pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    prev = closes[0]
    for i, c in enumerate(closes):
        rows.append({
            "open_ts": base + pd.Timedelta(minutes=15 * i),
            "open": prev,
            "high": max(prev, c) + 0.01,
            "low": min(prev, c) - 0.01,
            "close": c,
            "volume": 10.0,
        })
        prev = c
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PaperBroker
# ---------------------------------------------------------------------------

def test_paper_broker_buy_applies_slippage_up_and_fee():
    br = PaperBroker(CFG)
    fill = br.place_order("BTC", "buy", qty=0.1, reference_price=100.0,
                          ts=pd.Timestamp("2024-01-01", tz="UTC"))
    # buy: price * (1 + 1bp) = 100.01
    assert fill.price == pytest.approx(100.0 * (1 + 1e-4))
    # fee = qty * fill_price * 10bp
    assert fill.fee == pytest.approx(0.1 * fill.price * 1e-3)


def test_paper_broker_sell_applies_slippage_down():
    br = PaperBroker(CFG)
    fill = br.place_order("BTC", "sell", qty=0.1, reference_price=100.0,
                          ts=pd.Timestamp("2024-01-01", tz="UTC"))
    assert fill.price == pytest.approx(100.0 * (1 - 1e-4))


def test_paper_broker_unknown_side_raises():
    br = PaperBroker(CFG)
    with pytest.raises(ValueError):
        br.place_order("BTC", "short", qty=1, reference_price=100,
                       ts=pd.Timestamp("2024-01-01", tz="UTC"))


# ---------------------------------------------------------------------------
# RiskGuard — every limit gets a test
# ---------------------------------------------------------------------------

def _portfolio_at_initial(cash: float = 10_000.0) -> PaperPortfolio:
    p = PaperPortfolio(initial_cash=cash)
    p.equity_history.append((pd.Timestamp("2024-01-01", tz="UTC"), cash))
    return p


def test_risk_kill_switch_blocks_immediately():
    guard = RiskGuard(RiskLimits(kill_switch=True))
    ok, reason = guard.can_open(_portfolio_at_initial(), pd.Timestamp.utcnow(), 100)
    assert ok is False
    assert "kill switch" in reason


def test_risk_max_position_size_blocks_oversized_orders():
    guard = RiskGuard(RiskLimits(max_position_size_usd=500))
    ok, reason = guard.can_open(_portfolio_at_initial(), pd.Timestamp.utcnow(), 600)
    assert ok is False
    assert "exceeds" in reason


def test_risk_max_open_positions_blocks_when_already_full():
    guard = RiskGuard(RiskLimits(max_open_positions=1))
    p = _portfolio_at_initial()
    p.positions["BTC"] = Position(
        symbol="BTC", entry_time=pd.Timestamp.utcnow(), entry_price=100,
        qty=1, target_price=101, stop_loss_price=99,
        deadline_bar_count=4, notional_at_entry=100,
    )
    ok, reason = guard.can_open(p, pd.Timestamp.utcnow(), 100)
    assert ok is False
    assert "open positions" in reason


def test_risk_daily_loss_limit_blocks_after_breach():
    guard = RiskGuard(RiskLimits(max_daily_loss_usd=50))
    p = _portfolio_at_initial()
    today = pd.Timestamp("2024-01-01", tz="UTC")
    p.closed_trades.append(
        ClosedTrade(symbol="BTC", entry_time=today, entry_price=100,
                    exit_time=today + pd.Timedelta(hours=1),
                    exit_price=99, qty=1, outcome=STOPPED, pnl=-60)
    )
    ok, reason = guard.can_open(p, today, 100)
    assert ok is False
    assert "daily" in reason


def test_risk_daily_loss_limit_resets_next_day():
    """A loss yesterday must not block today's entries."""
    guard = RiskGuard(RiskLimits(max_daily_loss_usd=50))
    p = _portfolio_at_initial()
    yesterday = pd.Timestamp("2024-01-01", tz="UTC")
    today = pd.Timestamp("2024-01-02", tz="UTC")
    p.closed_trades.append(
        ClosedTrade(symbol="BTC", entry_time=yesterday, entry_price=100,
                    exit_time=yesterday + pd.Timedelta(hours=1),
                    exit_price=99, qty=1, outcome=STOPPED, pnl=-60)
    )
    ok, _ = guard.can_open(p, today, 100)
    assert ok is True


def test_risk_drawdown_limit_blocks_when_breached():
    guard = RiskGuard(RiskLimits(max_drawdown_pct=0.10))
    p = _portfolio_at_initial(cash=10_000)
    # equity history: peaked at 10_000, now at 8_500 → 15% drawdown
    p.equity_history = [
        (pd.Timestamp("2024-01-01", tz="UTC"), 10_000.0),
        (pd.Timestamp("2024-01-02", tz="UTC"), 8_500.0),
    ]
    p.cash = 8_500
    ok, reason = guard.can_open(p, pd.Timestamp.utcnow(), 100)
    assert ok is False
    assert "drawdown" in reason


def test_risk_allows_when_all_limits_clear():
    guard = RiskGuard(RiskLimits())
    ok, reason = guard.can_open(_portfolio_at_initial(), pd.Timestamp.utcnow(), 100)
    assert ok is True
    assert reason is None


# ---------------------------------------------------------------------------
# Portfolio accounting
# ---------------------------------------------------------------------------

def test_portfolio_open_then_close_zero_friction_returns_to_initial_cash():
    """Open at p, close at p with zero fees / slippage → cash unchanged."""
    cfg = BacktestConfig(taker_fee_bps=0, slippage_bps=0,
                         target_pct=0.015, stop_loss_pct=0.0075,
                         trade_window_bars=4)
    br = PaperBroker(cfg)
    p = PaperPortfolio(initial_cash=10_000)

    ts = pd.Timestamp("2024-01-01", tz="UTC")
    qty = 1.0
    entry = br.place_order("BTC", "buy", qty=qty, reference_price=100, ts=ts)
    pos = Position(
        symbol="BTC", entry_time=ts, entry_price=entry.price, qty=qty,
        target_price=101, stop_loss_price=99,
        deadline_bar_count=4, notional_at_entry=qty * entry.price,
    )
    p.open_position(pos, entry)
    exit_ = br.place_order("BTC", "sell", qty=qty, reference_price=100,
                           ts=ts + pd.Timedelta(minutes=15))
    p.close_position("BTC", exit_, NO_CLOSE)

    assert p.cash == pytest.approx(10_000.0)


def test_portfolio_winning_trade_increases_cash():
    cfg = BacktestConfig(taker_fee_bps=0, slippage_bps=0,
                         target_pct=0.015, stop_loss_pct=0.0075,
                         trade_window_bars=4)
    br = PaperBroker(cfg)
    p = PaperPortfolio(initial_cash=10_000)

    ts = pd.Timestamp("2024-01-01", tz="UTC")
    entry = br.place_order("BTC", "buy", qty=1.0, reference_price=100, ts=ts)
    pos = Position("BTC", ts, entry.price, 1.0, 101, 99, 4, 100)
    p.open_position(pos, entry)
    exit_ = br.place_order("BTC", "sell", qty=1.0, reference_price=110,
                           ts=ts + pd.Timedelta(minutes=15))
    p.close_position("BTC", exit_, TARGET_HIT)

    assert p.cash == pytest.approx(10_010.0)
    assert p.realized_pnl() == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# PaperTrader: state transitions
# ---------------------------------------------------------------------------

def _trader(cfg: BacktestConfig = CFG, limits: RiskLimits | None = None,
            cash: float = 10_000.0) -> PaperTrader:
    return PaperTrader(
        symbol="BTC",
        strategy=ALWAYS_ON,
        broker=PaperBroker(cfg),
        config=cfg,
        risk_limits=limits or RiskLimits(max_open_positions=1, max_position_size_usd=10_000),
        portfolio=PaperPortfolio(initial_cash=cash),
        notional_per_trade_usd=1_000,
    )


def test_trader_opens_on_first_signal():
    t = _trader()
    bars = _bars([100.0, 100.5])
    out = t.tick(bars)

    assert out.open_positions == 1
    assert any(e["event"] == "open" for e in out.events)


def test_trader_closes_on_target_hit():
    t = _trader()
    # Bar 1 opens at ~100. Target ≈ 100.01 * 1.015 = 101.51. Bar 2 close 102 clears it.
    out1 = t.tick(_bars([100.0]))
    assert out1.open_positions == 1
    out2 = t.tick(_bars([100.0, 102.0]))
    closes = [e for e in out2.events if e["event"] == "close"]
    assert closes and closes[0]["outcome"] == TARGET_HIT
    assert out2.open_positions == 0
    # PnL is positive (target was crossed)
    assert closes[0]["pnl"] > 0


def test_trader_closes_on_stop_hit():
    t = _trader()
    t.tick(_bars([100.0]))
    out = t.tick(_bars([100.0, 98.0]))  # cross stop at ~99.25
    closes = [e for e in out.events if e["event"] == "close"]
    assert closes and closes[0]["outcome"] == STOPPED
    assert closes[0]["pnl"] < 0


def test_trader_closes_on_window_timeout():
    """trade_window_bars=4 → the position times out before either threshold."""
    t = _trader()
    sideways = [100.0, 100.05, 100.10, 100.05, 100.0, 100.05]
    for i in range(1, len(sideways) + 1):
        t.tick(_bars(sideways[:i]))
    # at least one closed trade has the timeout outcome
    no_close_trades = [
        tr for tr in t.portfolio.closed_trades if tr.outcome == NO_CLOSE
    ]
    assert no_close_trades, "expected a NO_CLOSE timeout among closed trades"


def test_trader_kill_switch_force_closes_open_position():
    """Flipping the kill switch on a tick after entry must close the position."""
    t = _trader()
    t.tick(_bars([100.0]))
    assert t.portfolio.positions  # we are long
    # Now flip the kill switch.
    t.risk = RiskGuard(RiskLimits(kill_switch=True, max_open_positions=1))
    out = t.tick(_bars([100.0, 100.05]))
    closes = [e for e in out.events if e["event"] == "close"]
    assert closes and closes[0]["outcome"] == KILLED


def test_trader_blocked_by_size_emits_blocked_event():
    """notional > max_position_size_usd → entry is refused, no position opens."""
    t = _trader(limits=RiskLimits(
        max_position_size_usd=100,  # < 1_000
        max_open_positions=1,
    ))
    out = t.tick(_bars([100.0]))
    assert out.open_positions == 0
    blocked = [e for e in out.events if e["event"] == "blocked"]
    assert blocked and "exceeds" in blocked[0]["reason"]


def test_trader_blocked_by_kill_switch_does_not_open():
    t = _trader(limits=RiskLimits(kill_switch=True, max_position_size_usd=10_000))
    out = t.tick(_bars([100.0]))
    assert out.open_positions == 0
    assert any(e["event"] == "blocked" and "kill" in e["reason"] for e in out.events)


def test_trader_does_not_double_open_while_position_is_open():
    """Signal fires every bar but only one position can be open."""
    t = _trader()
    t.tick(_bars([100.0]))
    # next bar — signal still fires, but we already hold one
    out = t.tick(_bars([100.0, 100.05]))
    assert out.open_positions == 1
    # no second open on this tick
    assert sum(1 for e in out.events if e["event"] == "open") == 0


def test_trader_reopens_on_next_signal_after_close():
    """After a target hit, the next signal should be honoured."""
    t = _trader()
    t.tick(_bars([100.0]))
    t.tick(_bars([100.0, 102.0]))  # target hit, close
    assert t.portfolio.positions == {}
    # next bar fires again
    out = t.tick(_bars([100.0, 102.0, 100.0]))
    assert out.open_positions == 1


def test_trader_records_equity_history_per_tick():
    t = _trader()
    for i in range(1, 6):
        t.tick(_bars([100.0 + j * 0.05 for j in range(i)]))
    # one equity snapshot per tick
    assert len(t.portfolio.equity_history) == 5


def test_trader_with_never_true_strategy_never_opens():
    never = {
        "id": "never",
        "indicators": [
            {"absolute": True, "indicator": "close", "op": "<=", "abs_value": -1}
        ],
        "conjunctions": [],
    }
    t = PaperTrader(
        symbol="BTC", strategy=never, broker=PaperBroker(CFG),
        config=CFG, risk_limits=RiskLimits(),
        portfolio=PaperPortfolio(),
    )
    for i in range(1, 6):
        out = t.tick(_bars([100.0 + j * 0.5 for j in range(i)]))
    assert out.open_positions == 0
    assert t.portfolio.closed_trades == []
