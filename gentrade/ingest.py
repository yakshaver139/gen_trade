"""Multi-asset OHLCV ingest via ccxt + indicator computation + Parquet caching.

Replaces the ``gentrade.binance_download`` (python-binance) path with a
ccxt-based pipeline that works against any exchange ccxt knows about
(Binance, Coinbase, Kraken, Bybit, OKX, …). Output is a Parquet file
with OHLCV + a standard set of TA indicator columns from the ``ta``
library, ready for the GA's strategy DSL.

The ingest is page-aware: ccxt caps per-call rows (typically 500–1000),
so we page until ``until`` (default: now) is reached or the exchange
returns short pages. Each page respects ``exchange.rateLimit`` so the
ingest is a polite citizen.

Indicator catalogue: ``ta.add_all_ta_features`` produces ~80 columns
named ``momentum_*``, ``trend_*``, ``volatility_*``, ``volume_*`` —
matching the strategy catalogue's expected column names.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

# Standard ccxt timeframe strings → milliseconds-per-bar. Mapping kept
# narrow on purpose: anything outside this set probably needs domain
# input (different trade-window assumptions, equities, etc).
TIMEFRAMES: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

OHLCV_COLS = ["open_ts", "open", "high", "low", "close", "volume"]


def _to_ms(value: pd.Timestamp | datetime | str | int) -> int:
    """Coerce a since/until value to UTC epoch milliseconds."""
    if isinstance(value, int | float):
        return int(value)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


def _default_exchange_factory(exchange_id: str) -> Callable[[], object]:
    """Build a ccxt exchange instance lazily (so tests can monkeypatch)."""

    def _factory() -> object:
        import ccxt  # local import — keeps test runs without ccxt-network

        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"unknown exchange {exchange_id!r} (not in ccxt)")
        cls = getattr(ccxt, exchange_id)
        return cls({"enableRateLimit": True})

    return _factory


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    interval: str = "15m",
    since: pd.Timestamp | datetime | str | int | None = None,
    until: pd.Timestamp | datetime | str | int | None = None,
    *,
    page_limit: int = 1000,
    exchange_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
    """Page through OHLCV from a ccxt exchange and return a tidy DataFrame.

    Returns a frame with columns ``open_ts`` (tz-aware UTC), ``open``,
    ``high``, ``low``, ``close``, ``volume`` — one row per bar, sorted
    chronologically, no duplicates.
    """
    if interval not in TIMEFRAMES:
        raise ValueError(
            f"unsupported interval {interval!r}; choose from {list(TIMEFRAMES)}"
        )
    if exchange_factory is None:
        exchange_factory = _default_exchange_factory(exchange_id)
    exchange = exchange_factory()

    since_ms = _to_ms(since) if since is not None else None
    until_ms = _to_ms(until) if until is not None else None
    bar_ms = TIMEFRAMES[interval]

    all_bars: list[list] = []
    next_since = since_ms
    while True:
        page = exchange.fetch_ohlcv(symbol, interval, since=next_since, limit=page_limit)
        if not page:
            break
        all_bars.extend(page)
        last_ts = page[-1][0]
        # Stop once we've covered the whole requested window.
        if until_ms is not None and last_ts >= until_ms:
            break
        # Short page → exchange has nothing else for now.
        if len(page) < page_limit:
            break
        next_since = last_ts + bar_ms

    if not all_bars:
        return pd.DataFrame(columns=OHLCV_COLS).astype(
            {"open_ts": "datetime64[ns, UTC]", "open": float, "high": float,
             "low": float, "close": float, "volume": float}
        )

    df = pd.DataFrame(all_bars, columns=OHLCV_COLS)
    df["open_ts"] = pd.to_datetime(df["open_ts"], unit="ms", utc=True)
    if until_ms is not None:
        cutoff = pd.Timestamp(until_ms, unit="ms", tz="UTC")
        df = df[df["open_ts"] < cutoff]
    df = (
        df.drop_duplicates(subset=["open_ts"])
        .sort_values("open_ts")
        .reset_index(drop=True)
    )
    return df


YFINANCE_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def fetch_yfinance(
    symbol: str,
    interval: str = "1d",
    since: pd.Timestamp | datetime | str | None = None,
    until: pd.Timestamp | datetime | str | None = None,
    *,
    yf_module=None,  # injected for tests
) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance via yfinance.

    yfinance's intraday history is bounded (typically ~60 days for 1m,
    ~730 days for 1h); for daily and longer-window backtests this is fine.
    The function normalises the response to the same OHLCV schema as the
    ccxt path so downstream code is asset-class-agnostic.

    Be aware of two structural caveats with equities (vs crypto):
    - **Market hours**: equities are closed nights / weekends, so the
      "trade window = 1 day" assumption may need revisiting in
      `BacktestConfig.trade_window_bars`.
    - **Survivorship bias**: yfinance only returns currently-listed
      symbols. Train on a universe that includes dead names where
      possible; we do not yet provide one.
    """
    if interval not in YFINANCE_INTERVALS:
        raise ValueError(
            f"unsupported yfinance interval {interval!r}; choose from "
            f"{list(YFINANCE_INTERVALS)}"
        )

    if yf_module is None:
        import yfinance as yf  # local import; only paid when fetch_yfinance runs

        yf_module = yf

    start = pd.Timestamp(since) if since is not None else None
    end = pd.Timestamp(until) if until is not None else None

    raw = yf_module.download(
        tickers=symbol,
        start=start.tz_convert("UTC").tz_localize(None) if start is not None and start.tzinfo else start,
        end=end.tz_convert("UTC").tz_localize(None) if end is not None and end.tzinfo else end,
        interval=YFINANCE_INTERVALS[interval],
        auto_adjust=True,
        progress=False,
    )
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=OHLCV_COLS).astype(
            {"open_ts": "datetime64[ns, UTC]", "open": float, "high": float,
             "low": float, "close": float, "volume": float}
        )

    # yfinance returns a DataFrame indexed by Timestamp with columns
    # {Open, High, Low, Close, Volume} (and possibly multi-index when
    # multiple tickers — we always pass a single ticker).
    df = raw.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Index column is "Date" (daily) or "Datetime" (intraday).
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(
        columns={
            ts_col: "open_ts",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["open_ts"] = pd.to_datetime(df["open_ts"], utc=True)
    df = df[OHLCV_COLS].sort_values("open_ts").reset_index(drop=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``ta`` library's standard indicator set in place-safe fashion.

    Produces ``momentum_*``, ``trend_*``, ``volatility_*``, ``volume_*``
    columns matching the strategy catalogue. Adds a placeholder
    ``trend_direction`` column (UP / DOWN / NOTREND) — the GA's signals
    don't currently read it but it's part of the legacy schema and
    cheap to keep present.
    """
    import ta  # local import — keeps the rest of the package import-cheap

    out = df.copy()
    out = ta.add_all_ta_features(
        out,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True,
    )
    if "trend_direction" not in out.columns:
        out["trend_direction"] = "NOTREND"
    return out


def load_bars(path: str | Path) -> pd.DataFrame:
    """Load bars from CSV or Parquet (decided by suffix). UTC-aware ``open_ts``.

    The CSV path additionally accepts ``open_ts`` as either ISO 8601 or
    epoch milliseconds; Parquet preserves whatever the writer wrote.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = load_parquet(p)
    else:
        df = pd.read_csv(p)
        if "open_ts" not in df.columns:
            raise ValueError(f"{path}: missing required 'open_ts' column")
        ts = df["open_ts"]
        if pd.api.types.is_numeric_dtype(ts):
            df["open_ts"] = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            df["open_ts"] = pd.to_datetime(ts, utc=True)
    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write the bars frame to Parquet via pyarrow. Tz-aware datetimes survive."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet bars frame previously written by ``save_parquet``."""
    return pd.read_parquet(path, engine="pyarrow")
