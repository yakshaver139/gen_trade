import json
import os
import random
from collections.abc import Callable
from copy import deepcopy

import arrow
import pandas as pd

from gentrade.async_caller import process_future_caller
from gentrade.df_adapter import dfa
from gentrade.env import MAX_GENERATIONS, POPULATION_SIZE
from gentrade.fitness_functions import (
    fitness_function_ha_and_moon,
    fitness_function_original,
    fitness_simple_profit,
)
from gentrade.generate_blank_signals import indicators_from_df
from gentrade.generate_strategy import (
    main as generate_main,
)
from gentrade.generate_strategy import (
    make_strategy_from_indicators,
)
from gentrade.helpers import base_arg_parser, make_pandas_df, write_df_to_s3
from gentrade.logger import get_logger
from gentrade.run_strategy import run_strategy
from gentrade.selection import DEFAULT_PRESSURE, compute_weights


def load_df(path: str = "BTCUSDC_indicators.csv"):
    df = dfa.read_csv(path)
    return add_previous_window_values(df)


def add_previous_window_values(df):
    """For each indicator column, add a ``<col>_previous`` column shifted by one bar."""
    shifted = df.shift(1)
    for ind in indicators_from_df(df):
        df[f"{ind}_previous"] = shifted[ind]
    return df


def main(
    trading_data,
    strategies: list[dict] | None = None,
    fitness_function: Callable | None = None,
    generation: int = 1,
    max_generations: int = MAX_GENERATIONS,
    ranked_results: list | None = None,
    serial_debug: bool = False,
    population_size: int = POPULATION_SIZE,
    save_options: dict | None = None,
) -> list:
    """Main Genetic Algorithm. Logic:

    1. Subset the dataframe by the strategy (for each strategy)
    2. For each strategy:
        2.a Set the trade period to one day.
        2.b For each of these windows find the highest point (profit)
    3. calculate the average of these profits points
    4. calculate the fitness function
    5. selection pair from initial population
    6.a apply crossover operation on pair with cross over probability
    6.b apply mutation on offspring using mutation probability
    7. replace old population and repeat for MAX_POP steps
    8. rank the solutions and return the best
    """
    ranked_results = ranked_results or []
    save_options = save_options or {}
    logger = get_logger(__name__)

    if generation <= max_generations:
        logger.info(f"Running generation {generation}")

        # Special case as this one requires passing in all results together
        # (i.e. not iteratively per strategy)
        if fitness_function.__name__ == "fitness_simple_profit":
            results = process_future_caller(
                run_strategy, strategies, trading_data, ranked_results
            )
            fitness = fitness_function(results, serial_debug)
        else:
            # Serial invocation - for debugging
            if serial_debug:
                results = [
                    run_strategy(trading_data, ranked_results, strat)
                    for strat in strategies
                ]
                fitness = [fitness_function(x) for x in results]
            else:
                results = process_future_caller(
                    run_strategy, strategies, trading_data, ranked_results
                )
                fitness = process_future_caller(fitness_function, results)

        ranking, weights = apply_ranking(fitness)
        ranked_results.append(ranking)

        if save_options.get('incremental_saves'):
            logger.info(f"save incremental data for generation {generation}")
            dt = arrow.utcnow()
            fname = (f"{dt.isoformat()}_{fitness_function.__name__}"
                     f"_{population_size}_{generation}.csv")
            save_data(fname, ranking, **save_options)

        logger.info("Recursing into next generation")

        population = generate_population(ranking, weights, population_size)
        main(
            trading_data,
            population,
            fitness_function,
            generation=generation + 1,
            max_generations=max_generations,
            ranked_results=ranked_results,
            serial_debug=serial_debug,
            population_size=population_size,
            save_options=save_options,
        )
    return ranked_results


def apply_ranking(
    results, pressure: str = DEFAULT_PRESSURE
) -> tuple[pd.DataFrame, dict]:
    df = pd.concat(results)
    df.sort_values(by="fitness", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    weights = compute_weights(df, pressure=pressure)
    return df, weights


def generate_population(
    ranked: pd.DataFrame, weights: dict, population_size: int = POPULATION_SIZE
) -> list[dict]:
    """Applies ranking, cross over and mutation to create a new population"""
    # Elitism - keep the two best solutions from the previous population

    ranked = ranked.reset_index()
    # elitism
    population = [ranked.iloc[0].strategy, ranked.iloc[1].strategy]
    logger = get_logger(__name__)
    logger.info(f"generating new population of length {population_size}")

    _weights = deepcopy(weights)

    while len(population) < population_size:
        x, y = select_parents(ranked, _weights)
        offspring = cross_over_ppx(x.strategy, y.strategy)
        population.append(mutate(offspring))
    logger.info("Population created")
    return population


def select_parents(
    ranked: pd.DataFrame, _weights: dict, parents: dict | None = None
) -> tuple:
    """Randomly sample two distinct parents (rank-weighted) for offspring generation."""
    if parents is None:
        parents = {}
    sampled = ranked.sample(2, weights=list(_weights.values()))

    x, y = sampled.iloc[0], sampled.iloc[1]
    if (x.id, y.id) in parents:
        return select_parents(ranked, _weights, parents)
    parents[(x.id, y.id)] = 1

    return x, y


def cross_over_ppx(strat_x: dict, strat_y: dict) -> dict:
    """Precedence preservative crossover"""
    x_ind = strat_x["indicators"]
    y_ind = strat_y["indicators"]

    child_len = max(len(x_ind), len(y_ind))
    mapping = [random.choice([0, 1]) for _ in range(child_len)]
    mapped_inds = {0: (x_ind, y_ind), 1: (y_ind, x_ind)}

    offspring = []

    for ix, chromo in enumerate(mapping):
        primary, secondary = mapped_inds[chromo]
        try:
            p_ind = primary[ix]
        except IndexError:
            p_ind = secondary[ix]
        finally:
            if p_ind not in offspring:
                offspring.append(p_ind)
    return make_strategy_from_indicators(offspring)


def mutate(strategy: dict) -> dict:
    _strat = deepcopy(strategy)
    abs_strats = [x for x in _strat["indicators"] if x["absolute"]]

    if abs_strats:
        mutate_ix = random.choice(range(len(abs_strats)))
        mutate_ind = abs_strats[mutate_ix]
        mutate_percent = random.choice(range(-20, 20)) / 100
        mutate_ind["abs_value"] = mutate_ind["abs_value"] * (1 + mutate_percent)

    return _strat


def load_trading_data():
    df = load_df()
    df["converted_open_ts"] = dfa.to_datetime(df["open_ts"], unit="ms")
    df.index = df["open_ts"]
    return df


def load_strategies(
    path: str | None = None,
    max_indicators: int | None = None,
    max_same_class: int | None = None,
    population_size: int = POPULATION_SIZE,
) -> list[dict]:
    if path:
        with open(path) as fi:
            return json.loads(fi.read())

    return generate_main(
        max_indicators=max_indicators,
        max_same_class=max_same_class,
        population_size=population_size,
    )


FITNESS_MAP = dict(
    o=fitness_function_original,
    h=fitness_function_ha_and_moon,
    p=fitness_simple_profit,
)


def save_data(
    fname,
    df,
    incremental_saves=False,
    write_local=False,
    write_s3=False,
    output_path=None,
    bucket=None,
) -> None:
    if output_path:
        fname = os.path.join(output_path, fname)
    if write_local:
        df.to_csv(fname)
    if write_s3:
        write_df_to_s3(df, bucket, fname)


if __name__ == "__main__":
    parser = base_arg_parser("Main genetic algorithm.")
    parser.add_argument("--write_s3", type=bool, help="exports data to s3.")
    parser.add_argument("--write_local", type=bool, help="exports data to local FS.")
    parser.add_argument(
        "--s3_bucket", type=str, help="bucket name to which data is written."
    )
    parser.add_argument("--generations", type=int, help="N generations to run.")
    parser.add_argument(
        "--serial_debug", type=bool, help="run without async for debugging"
    )
    parser.add_argument(
        "--strategies_path",
        type=str,
        help="load strategies from this path rather than generating on the fly",
    )
    parser.add_argument(
        "--fitness_function",
        type=str,
        help="fitness function use (h=ha_and_moon, o=original, p=profit)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to save outputs",
    )
    parser.add_argument(
        "--incremental_saves",
        type=bool,
        help="when true saves the output for every 10 strategies tested",
    )

    parser.set_defaults(
        write_s3=bool(os.getenv("WRITE_S3", False)),
        write_local=bool(os.getenv("WRITE_LOCAL", True)),
        generations=int(os.getenv("GENERATIONS", MAX_GENERATIONS)),
        s3_bucket=os.getenv("BUCKET"),
        serial_debug=os.getenv("SERIAL_DEBUG", False),
        strategies_path=os.getenv("STRATEGY_PATH"),
        fitness_function=os.getenv("FITNESS_FUNCTION", "p"),
        incremental_saves=bool(os.getenv("INCREMENTAL_SAVES", False))
    )
    args = parser.parse_args()
    gens = args.generations
    pop_size = args.population_size

    strategies = load_strategies(
        args.strategies_path,
        args.max_indicators,
        args.max_same_class,
        pop_size,
    )
    start = arrow.utcnow()

    save_options = dict(
        incremental_saves=args.incremental_saves,
        write_local=args.write_local,
        write_s3=args.write_s3,
        output_path=args.output_path,
        bucket=args.s3_bucket,
    )

    ff = FITNESS_MAP[args.fitness_function]
    results = main(
        load_trading_data(),
        strategies,
        ff,
        max_generations=gens,
        serial_debug=args.serial_debug,
        population_size=pop_size,
        save_options=save_options,
    )
    stop = arrow.utcnow()

    df = dfa.concat(results)

    fname = f"{start.isoformat()}_results_{ff.__name__}_{pop_size}_{gens}.csv"
    save_data(fname, df, **save_options)
    df = make_pandas_df(df)

    logger = get_logger(__name__)
    logger.info(f"Finished in {(stop - start).seconds} seconds")
