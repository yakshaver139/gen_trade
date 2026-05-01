import argparse
import os
from io import StringIO

import pandas as pd

from gentrade.env import (
    MAX_SAME_CLASS_INDICATORS,
    MAX_STRATEGY_INDICATORS,
    POPULATION_SIZE,
    get_s3_resource,
)


def make_pandas_df(df) -> pd.DataFrame:
    # convert dask dd back to pandas; pandas dfs pass through.
    try:
        return df.compute()
    except AttributeError:
        return df


def get_latest_path(path: str, suffix: str | None = None) -> str:
    paths = []
    for entry in os.scandir(path):
        if entry.is_dir():
            continue
        if suffix and entry.name.endswith(suffix):
            paths.append(entry)
    return max(paths, key=os.path.getctime).path


def write_df_to_s3(df: pd.DataFrame, bucket: str, name: str) -> None:
    """write csv to s3 bucket. Overwrites file if already exists."""
    buffer = StringIO()
    df.to_csv(buffer)
    get_s3_resource().Object(bucket, name).put(Body=buffer.getvalue())


def base_arg_parser(help_message: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(help_message)
    parser.add_argument("--population_size", type=int, required=False,
                        help="number of strategies to generate")
    parser.add_argument("--max_indicators", type=int, required=False,
                        help="max number of indicators in a strategy")
    parser.add_argument("--max_same_class", type=int, required=False,
                        help="max number of same class indicators in a strategy")
    parser.set_defaults(
        population_size=POPULATION_SIZE,
        max_indicators=MAX_STRATEGY_INDICATORS,
        max_same_class=MAX_SAME_CLASS_INDICATORS,
    )
    return parser
