"""Generate a tiny synthetic OHLCV+indicators CSV for local dev / smoke runs.

Used by scripts/dev_server.sh. The series is a noisy random walk so
generated strategies can produce a meaningful spread of fitnesses,
plus the technical-indicator columns the strategy DSL queries.

This is not real data — it exists so the API smoke path runs end-to-end
without Binance credentials. Indicator values that don't have a clean
synthetic analogue (RSI, MFI, etc.) are filled with plausibly-shaped
random values; the GA happens to optimise against this noise, so don't
read fitness numbers from a dev run as anything but a smoke signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=1).mean().to_numpy()


def main(out_path: str, n_bars: int = 1500, seed: int = 42) -> int:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2022-01-01", tz="UTC")
    open_ts = [(base + pd.Timedelta(minutes=15 * i)).isoformat() for i in range(n_bars)]

    # Random walk with mild drift — varied enough to produce both winning
    # and losing strategies, deterministic given the seed.
    base_price = 30_000 + np.cumsum(rng.normal(0, 50, n_bars))
    high = base_price + rng.uniform(0, 200, n_bars)
    low = base_price - rng.uniform(0, 200, n_bars)
    close = base_price + rng.normal(0, 30, n_bars)
    volume = rng.uniform(10, 100, n_bars)

    df = pd.DataFrame(
        {
            "open_ts": open_ts,
            "open": base_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            # The columns the strategy DSL references. Mirrors
            # gentrade.smoke.synthetic_indicators_df — kept in sync by hand.
            "trend_sma_slow": _rolling_mean(close, 50),
            "trend_sma_fast": _rolling_mean(close, 20),
            "trend_psar_up": base_price + rng.uniform(-50, 50, n_bars),
            "trend_ema_slow": _rolling_mean(close, 40),
            "trend_adx_pos": rng.uniform(10, 40, n_bars),
            "trend_adx_neg": rng.uniform(10, 40, n_bars),
            "momentum_rsi": rng.uniform(20, 80, n_bars),
            "momentum_roc": rng.normal(0, 1, n_bars),
            "volatility_bbm": _rolling_mean(close, 20),
            "volatility_dcm": rng.uniform(low.min(), high.max(), n_bars),
            "volatility_kchi": rng.uniform(low.min(), high.max(), n_bars),
            "volume_cmf": rng.uniform(-1, 1, n_bars),
            "volume_mfi": rng.uniform(0, 100, n_bars),
            "volume_obv": np.cumsum(rng.normal(0, 100, n_bars)),
            "trend_direction": rng.choice(["UP", "DOWN", "NOTREND"], n_bars),
        }
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {n_bars} bars + indicators to {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: dev_bars.py <out.csv>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
