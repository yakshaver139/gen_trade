"""Genotype → phenotype: turn a strategy dict into a pandas-query string."""

from copy import copy

from gentrade.df_adapter import DFAdapter
from gentrade.logger import get_logger

dfa = DFAdapter()


def get_callback(indicator: dict) -> str:
    """Build the relative comparison fragment for a non-absolute indicator."""
    rel = indicator["rel_value"]
    if rel == "MA":
        return f"{indicator['indicator']} {indicator['op']} trend_sma_slow"
    elif rel == "PREVIOUS_PERIOD":
        return f"{indicator['indicator']} {indicator['op']} {indicator['indicator']}_previous"
    else:
        return f"{indicator['indicator']} {indicator['op']} {rel.lower()}"


def _validate_conjunctions(strategy: dict) -> list[str]:
    conjunctions = strategy.get("conjunctions", [])
    if len(conjunctions) != len(strategy["indicators"]) - 1:
        raise RuntimeError(
            f"Strategy does not have correct number of conjunctions! {strategy}"
        )
    return conjunctions


def _parse_signal(indicator: dict) -> str:
    if indicator["absolute"]:
        return f"{indicator['indicator']} {indicator['op']} {indicator['abs_value']}"
    return get_callback(indicator)


def load_from_object(strategy: dict) -> str:
    """Original un-parenthesised version (legacy; kept for tests)."""
    conjunctions = _validate_conjunctions(strategy)
    parsed = [_parse_signal(ind) for ind in strategy["indicators"]]

    if not conjunctions:
        return parsed[0]

    result = ""
    for ix, strat in enumerate(parsed):
        result += strat
        if ix < len(conjunctions):
            result += f" {conjunctions[ix]} "
    return result


def load_from_object_parenthesised(strategy: dict) -> str:
    """Parenthesise disjunctions to keep evaluation order well-defined."""
    conjunctions = _validate_conjunctions(strategy)
    parsed = [_parse_signal(ind) for ind in strategy["indicators"]]

    if not conjunctions:
        return parsed[0]

    result = ""
    open_paren_count = 0
    for ix, strat in enumerate(parsed):
        _strat = copy(strat)
        if ix < len(conjunctions):
            conj = conjunctions[ix]
            _strat = f"({_strat} {conj}" if "or" in conj.lower() else f"{_strat} {conj}"
            open_paren_count += int(_strat[0] == "(" and _strat[-1] != ")")
            result = f"{result} {_strat}"
        else:
            result = f"{result} {_strat}"
            if open_paren_count:
                result += ")" * open_paren_count

    res = result[1:]
    get_logger(__name__).info(f"Generated strategy: {res}")
    return res


def query_strategy(df, strategy: dict | None = None, query: str | None = None):
    if query is None:
        query = load_from_object_parenthesised(strategy)
    try:
        return df.query(query).compute()
    except AttributeError:
        return df.query(query)
