from collections.abc import Iterable
from math import log

import pandas as pd

from gentrade.async_caller import process_future_caller
from gentrade.df_adapter import dfa
from gentrade.env import HIT
from gentrade.profit_calculator import calculate_simple_profit


def fitness_metadata(result: pd.DataFrame) -> tuple[int, float, float]:
    try:
        n_trades = len(result)
        win_percent = (len(result.loc[result.result == HIT]) / n_trades)
        avg_gain = result.performance.mean()

    except (AttributeError, ZeroDivisionError):
        n_trades = 0
        win_percent = 0
        avg_gain = 0
    return n_trades, win_percent, avg_gain


def fitness_simple_profit(results: Iterable, serial_debug: bool) -> pd.DataFrame:
    if serial_debug:
        df = pd.concat([fitness_function_ha_and_moon(x) for x in results])
    else:
        df = pd.concat(process_future_caller(fitness_function_ha_and_moon, results))
    fitness = calculate_simple_profit(df=df)
    df["fitness"] = fitness.values()
    fitness = [df]
    return fitness


def fitness_function_ha_and_moon(result: dfa.DataFrame) -> pd.DataFrame:
    """Fitness function based on the Ha & Moon study.

    g(r)i,j = log((pc(i, j) + k) / pc(i, j))
    """
    # handle recursive calls already made which already have fitness data
    if "fitness" in result:
        return result

    n_trades, win_percent, avg_gain = fitness_metadata(result)
    if not n_trades:
        fitness = -1000
    else:
        result["fitness"] = result.apply(lambda x: log_increase(x, n_trades), axis=1)
        fitness = result.fitness.mean()
    return transform_fitness_results(result, n_trades, win_percent, avg_gain, fitness)


def fitness_function_original(result: dfa.DataFrame) -> pd.DataFrame:
    """Win percentage * number of trades gives a performance coefficient.
    That is the higher the WP / NT, the bigger the coeff.
    Returns gain on account * performance coeff.
    """
    # handle recursive calls already made which already have fitness data
    if "fitness" in result:
        return result
    n_trades, win_percent, avg_gain = fitness_metadata(result)
    fitness = -1000 if n_trades <= 1 else avg_gain * (win_percent * n_trades)
    return transform_fitness_results(result, n_trades, win_percent, avg_gain, fitness)


def transform_fitness_results(
    result: pd.DataFrame,
    n_trades: int | None = None,
    win_percent: float | None = None,
    avg_gain: float | None = None,
    fitness: float | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": result.iloc[0].strategy["id"],
                "avg_gain": avg_gain,
                "fitness": fitness,
                "n_trades": n_trades,
                "win_percent": win_percent * 100,
                "target": result.target.to_dict(),
                "stop_loss": result.stop_loss.to_dict(),
                "result": result.result.to_dict(),
                "strategy": result.iloc[0].strategy,
                "performance": result.performance.to_dict(),
                "open_ts": result.open_timestamp.to_dict(),
                "trend": result.trend.to_dict(),
            }
        ]
    )


def log_increase(x, n_trades):
    price = x.close or 0
    try:
        return log((price + n_trades) / price)
    except ZeroDivisionError:
        return -1000
