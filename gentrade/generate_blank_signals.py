"""Build the relative-signal catalogue from a price dataframe's indicator columns.

The previous version read ``BTCUSDC_indicators.csv`` at import time and
exposed ``INDICATORS`` as a generator (a known footgun: any caller that
iterated it first left an empty generator for everyone else). Now the
catalogue is derived per-dataframe at call time.
"""

import json
from collections.abc import Iterable

import pandas as pd

CATS = ("momentum", "trend", "volatility", "volume")
RELATIVES = ("PREVIOUS_PERIOD", "MA")


def indicators_from_df(df: pd.DataFrame) -> list[str]:
    """Return the indicator column names from ``df`` (any column whose name
    starts with one of the four signal classes)."""
    return [c for c in df.columns if any(c.startswith(prefix) for prefix in CATS)]


def gte_signal(ind: str, _type: str, rel_value: str) -> dict:
    return _gen_signal(ind, _type, rel_value, ">=")


def lte_signal(ind: str, _type: str, rel_value: str) -> dict:
    return _gen_signal(ind, _type, rel_value, "<=")


def _gen_signal(ind: str, _type: str, rel_value: str, op: str) -> dict:
    return {
        "indicator": ind,
        "type": _type,
        "name": f"{ind}_gt",
        "absolute": False,
        "op": op,
        "abs_value": None,
        "rel_value": rel_value,
    }


def build_relative_signals(indicators: Iterable[str]) -> list[dict]:
    signals: list[dict] = []
    for ind in indicators:
        _type = ind.split("_")[0]
        for rel in RELATIVES:
            signals.extend([gte_signal(ind, _type, rel), lte_signal(ind, _type, rel)])
    return signals


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Generate relative_signals.json from a CSV.")
    parser.add_argument("csv", help="Path to OHLCV+indicators CSV")
    parser.add_argument(
        "--out", default="gentrade/signals/relative_signals.json", help="Output path"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    signals = build_relative_signals(indicators_from_df(df))
    with open(args.out, "w") as fi:
        json.dump(signals, fi, indent=2)
