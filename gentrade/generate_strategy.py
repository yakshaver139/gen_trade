"""Random strategy (chromosome) generation. See dissertation Algorithm 1."""

import json
import logging
import random
import sys
import uuid
from collections.abc import Sequence
from importlib.resources import files
from string import ascii_lowercase

from gentrade.env import (
    CONJUNCTIONS as _ALL_CONJUNCTIONS,
)
from gentrade.env import (
    MAX_SAME_CLASS_INDICATORS,
    MAX_STRATEGY_INDICATORS,
    MIN_INDICATORS,
    POPULATION_SIZE,
)
from gentrade.helpers import base_arg_parser

# Negations are not used by the live algorithm; only `and` and `or`.
CONJUNCTIONS = _ALL_CONJUNCTIONS[:2]


def _load_signals() -> list[dict]:
    """Load the absolute + relative signal catalogues from the package data."""
    base = files("gentrade").joinpath("signals")
    with base.joinpath("absolute_signals.json").open("r") as fi:
        signals = json.load(fi)
    with base.joinpath("relative_signals.json").open("r") as fi:
        signals.extend(json.load(fi))
    return signals


LOADED_INDICATORS = _load_signals()


def choose_indicator(indicators, max_same_class, same_class_indicators):
    indicator = random.choice(indicators)
    if same_class_indicators.get(indicator["type"], -1) + 1 > max_same_class:
        # Subset to avoid hitting max recursion depth.
        _indicators = [x for x in indicators if x["type"] != indicator["type"]]
        return choose_indicator(_indicators, max_same_class, same_class_indicators)
    return indicator


def pop_indicator(to_pop, indicators):
    return [x for x in indicators if x != to_pop]


def choose_num_indicators(max_indicators: int) -> int:
    return random.choice(range(MIN_INDICATORS, max_indicators + 1))


def make_strategy_from_indicators(indicators: Sequence, conjunctions=CONJUNCTIONS) -> dict:
    """Wrap an iterable of indicators into a well-formed strategy dict.

    The first conjunction is always ``"and"`` to satisfy the spec's
    `FirstConjunctionIsAnd` invariant — the parsed query parenthesises
    cleanly only if disjunctions never lead. Without this, every
    crossover child had a ~50% chance of silently violating the
    invariant (``generate`` honours it; ``cross_over_ppx`` calls this
    helper which previously did not).
    """
    conj = [random.choice(conjunctions) for _ in range(len(indicators) - 1)]
    if conj:
        conj[0] = "and"
    return dict(
        id=str(uuid.uuid4()),
        indicators=list(indicators),
        conjunctions=conj,
    )


def generate(
    base_indicator,
    indicators,
    conjunctions=CONJUNCTIONS,
    max_indicators=MAX_STRATEGY_INDICATORS,
    max_same_class=MAX_SAME_CLASS_INDICATORS,
) -> dict:
    """Generate a strategy with ``base_indicator`` as the first indicator."""
    strategy = dict(id=str(uuid.uuid4()), indicators=[base_indicator], conjunctions=[])
    same_class_indicators = {base_indicator["type"]: 1}
    indicators = pop_indicator(base_indicator, indicators)

    num_indicators = choose_num_indicators(max_indicators)

    for _ in range(num_indicators - 1):
        strategy["conjunctions"].append(random.choice(conjunctions))
        # FirstConjunctionIsAnd invariant: keeps parenthesisation well-formed.
        strategy["conjunctions"][0] = "and"
        ind = choose_indicator(indicators, max_same_class, same_class_indicators)
        same_class_indicators[ind["type"]] = same_class_indicators.get(ind["type"], 0) + 1
        strategy["indicators"].append(ind)
        indicators = pop_indicator(ind, indicators)
    return strategy


def main(
    indicators=None,
    max_indicators=MAX_STRATEGY_INDICATORS,
    max_same_class=MAX_SAME_CLASS_INDICATORS,
    population_size=POPULATION_SIZE,
) -> list[dict]:
    if indicators is None:
        indicators = LOADED_INDICATORS
    strategies = []
    for _ in range(population_size):
        base_indicator = random.choice(indicators)
        strategies.append(
            generate(base_indicator, indicators, CONJUNCTIONS, max_indicators, max_same_class)
        )
    return strategies


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s%(levelname)s:%(message)s",
        stream=sys.stderr,
        level=logging.INFO,
    )
    parser = base_arg_parser("Generate an initial population of random strategies")
    parser.add_argument("--test", type=bool, help="whether to run in testmode")

    args = parser.parse_args()
    indicators = list(ascii_lowercase) if args.test else LOADED_INDICATORS
    main(indicators, args.max_indicators, args.max_same_class, args.population_size)
