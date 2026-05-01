import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
from binance.client import Client
from ta import add_all_ta_features
from ta.utils import dropna

from gentrade.ta_trends import trend_direction

DF_COLUMNS = [
    'open_ts', 'open', 'high', 'low', 'close', 'volume', 'close_ts', 'qav',
    'num_trades'
]


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Converts raw string to floats and adds all TA indicators to the dataframe.
    see: https://technical-analysis-library-in-python.readthedocs.io/en/latest/index.html#examples
    NB for indicators built from historical data there there are nans for the where not enough prior data exists
    i.e. rows 0-18 are all nans as by default indicators are built from 20 periods.

    """
    # convert string to floats
    for column in DF_COLUMNS:
        df[column] = pd.to_numeric(df[column])

    # remove empty data
    df = dropna(df)
    df = add_all_ta_features(df,
                             open='open',
                             high='high',
                             low='low',
                             close='close',
                             volume='volume')
    return df


def write_data(klines: list, csv_path: str) -> pd.DataFrame:
    # Drop some of the columns we're not interested in
    _klines = [k[:-3] for k in klines]
    df = pd.DataFrame(_klines)
    df.columns = DF_COLUMNS

    df = prepare_df(df)
    df['trend_direction'] = df.apply(trend_direction, axis=1)
    df.to_csv(csv_path)


def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # nb timestamps are unix time, could optionally convert by * 1000, but maybe not necessary?
    for y in [df.open_ts, df.close_ts]:
        for x in y:
            df.index = [datetime.fromtimestamp(x / 1000.0)]
    return df


def main(start_date: str, end_date: str, symbol_a: str, symbol_b: str,
         csv_path: str) -> None:
    client = Client(os.environ["BINANCE_API"], os.environ["BINANCE_SECRET"])
    klines = client.get_historical_klines(f"{symbol_a}{symbol_b}",
                                          client.KLINE_INTERVAL_15MINUTE,
                                          start_date, end_date)
    write_data(klines, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download historical Binance data.")
    parser.add_argument(
        '--start_date',
        type=str,
        help=
        "Start date - human readable (see Binance API docs) - default `1 Jan, 2018`",
        required=False)
    parser.add_argument(
        '--end_date',
        type=str,
        help="- human readable (see Binance API docs) default `1 day ago UTC`",
        required=False)
    parser.add_argument('--symbol_a',
                        type=str,
                        help='First ticker in the trade - default BTC',
                        required=False)
    parser.add_argument('--symbol_b',
                        type=str,
                        help='Secondary ticker in the trade - default USDC',
                        required=False)
    parser.add_argument(
        '--csv_path',
        type=str,
        help=
        'Output path/filename for CSV. If None uses the concat of the symbols.',
        required=False)

    parser.set_defaults(start_date="1 Jan, 2018",
                        end_date="1 day ago UTC",
                        symbol_a="BTC",
                        symbol_b="USDC")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s%(levelname)s:%(message)s',
                        stream=sys.stderr,
                        level=logging.ERROR)

    csv_path = args.csv_path or f"{args.symbol_a}{args.symbol_b}.csv"
    main(args.start_date, args.end_date, args.symbol_a, args.symbol_b,
         csv_path)
