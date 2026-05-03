"""End-to-end smoke run of the GA on a synthetic OHLCV+indicators dataframe.

No network, no Binance, no S3. Used to prove the pipeline is wired up.
Run as: ``uv run python -m gentrade.smoke``.
"""

import random

import numpy as np
import pandas as pd

from gentrade.fitness_functions import fitness_function_original
from gentrade.generate_strategy import LOADED_INDICATORS
from gentrade.generate_strategy import main as generate_strategies
from gentrade.genetic import add_previous_window_values, apply_ranking, generate_population
from gentrade.run_strategy import run_strategy

SEED = 42


def synthetic_indicators_df(n_bars: int = 1500, seed: int = SEED) -> pd.DataFrame:
    """Build a tiny OHLCV+indicators dataframe with the columns the GA expects.

    The GA's strategy DSL queries columns whose names start with one of the
    four signal classes (``momentum_``, ``trend_``, ``volatility_``,
    ``volume_``) plus ``trend_sma_slow`` for the moving-average comparator.
    We synthesise plausible indicator values; this dataframe is *not* meant
    to be predictive.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2022-01-01", tz="UTC")
    open_ts = (start + pd.to_timedelta(np.arange(n_bars) * 15, unit="m")).astype("int64") // 10**6

    base_price = 30_000 + np.cumsum(rng.normal(0, 50, n_bars))
    high = base_price + rng.uniform(0, 200, n_bars)
    low = base_price - rng.uniform(0, 200, n_bars)
    close = base_price + rng.normal(0, 30, n_bars)
    volume = rng.uniform(10, 100, n_bars)

    df = pd.DataFrame({
        "open_ts": open_ts,
        "open": base_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "trend_sma_slow": pd.Series(close).rolling(50, min_periods=1).mean().to_numpy(),
        "trend_sma_fast": pd.Series(close).rolling(20, min_periods=1).mean().to_numpy(),
        "trend_psar_up": base_price + rng.uniform(-50, 50, n_bars),
        "trend_ema_slow": pd.Series(close).rolling(40, min_periods=1).mean().to_numpy(),
        "trend_adx_pos": rng.uniform(10, 40, n_bars),
        "trend_adx_neg": rng.uniform(10, 40, n_bars),
        "momentum_rsi": rng.uniform(20, 80, n_bars),
        "momentum_roc": rng.normal(0, 1, n_bars),
        "volatility_bbm": pd.Series(close).rolling(20, min_periods=1).mean().to_numpy(),
        "volatility_dcm": rng.uniform(low.min(), high.max(), n_bars),
        "volatility_kchi": rng.uniform(low.min(), high.max(), n_bars),
        "volume_cmf": rng.uniform(-1, 1, n_bars),
        "volume_mfi": rng.uniform(0, 100, n_bars),
        "volume_obv": np.cumsum(rng.normal(0, 100, n_bars)),
        "trend_direction": rng.choice(["UP", "DOWN", "NOTREND"], n_bars),
    })
    df["converted_open_ts"] = pd.to_datetime(df["open_ts"], unit="ms")
    df.index = df["open_ts"]
    return df


def _filter_catalogue_to_df(df: pd.DataFrame) -> list[dict]:
    """Restrict the signal catalogue to indicators whose columns exist in ``df``.

    The full catalogue references ~80 indicators that the BTC indicator CSV
    has but our synthetic frame does not. Generating strategies from the full
    catalogue and then querying the smaller frame produces ``UndefinedVariable``
    errors. We constrain the catalogue to what we synthesised.
    """
    cols = set(df.columns)
    return [s for s in LOADED_INDICATORS if s["indicator"] in cols]


def run_one_generation(population_size: int = 4, seed: int = SEED) -> pd.DataFrame:
    """Generate strategies, backtest, score, breed once; return the ranking."""
    random.seed(seed)
    np.random.seed(seed)

    df = synthetic_indicators_df(seed=seed)
    df = add_previous_window_values(df)

    catalogue = _filter_catalogue_to_df(df)
    if len(catalogue) < 4:
        raise RuntimeError(
            f"synthetic frame only matches {len(catalogue)} catalogue indicators; "
            "add more columns to synthetic_indicators_df"
        )
    strategies = generate_strategies(indicators=catalogue, population_size=population_size)
    backtests = [run_strategy(df, [], strat) for strat in strategies]
    fitness = [fitness_function_original(b) for b in backtests]
    ranking, weights = apply_ranking(fitness)

    # exercise the breeding step too
    next_population, _events = generate_population(ranking, weights, population_size=population_size)
    assert len(next_population) == population_size
    return ranking


def main() -> None:
    ranking = run_one_generation()
    print(f"smoke OK — population ranked, top fitness = {ranking.iloc[0].fitness}")


if __name__ == "__main__":
    main()
