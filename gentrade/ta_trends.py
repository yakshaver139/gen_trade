import pandas as pd

"""Used for mapping trend directions to the raw data produced from `binance_download.py`
These helpers have been seperated into their own module as they are used during the GP
to get the next window of data.
"""

def trend_direction(adx_row: pd.Series) -> str:
    """Use as part of a df.apply to map trend direction from ADX data

    NB that ADX is undirectional (provides strength of trend) between 0-100, hence use in combination with directional indicators.
    see: https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#trend-indicators and
    https://www.investopedia.com/articles/trading/07/adx-trend-indicator.asp

    """
    if adx_row.trend_adx_pos > adx_row.trend_adx_neg:
        return 'UP'
    elif adx_row.trend_adx_pos < adx_row.trend_adx_neg:
        return 'DOWN'
    else:
        return 'NOTREND'

def filter_trends(adx_df: pd.DataFrame, direction='UP') -> pd.DataFrame:
    return adx_df.loc[adx_df['trend_direction'] == direction]

def _filter_trend_strength(adx_df: pd.DataFrame, limit=25, op='ge') -> pd.DataFrame:
    """Helper to filter out all values >= / <= / > / < or == to the limit"""
    op = f"__{op}__"
    comparitor = getattr(pd.Series, op)
    return adx_df.loc[comparitor(adx_df.trend_adx, limit )]

def get_weak_trends(adx_df: pd.DataFrame) -> pd.DataFrame:
    return _filter_trend_strength(adx_df, op='lt')

def get_strong_trends(adx_df: pd.DataFrame) -> pd.DataFrame:
    return _filter_trend_strength(_filter_trend_strength(adx_df, op='ge'), limit=50, op='lt')

def get_vstrong_trends(adx_df: pd.DataFrame) -> pd.DataFrame:
    return _filter_trend_strength(_filter_trend_strength(adx_df, limit=50, op='ge'),limit=75, op='lt')

def get_exstrong_trends(adx_df: pd.DataFrame) -> pd.DataFrame:
    return _filter_trend_strength(adx_df, limit=75, op='ge')

def generate_trends(df: pd.DataFrame, periods: int=96):
    """Yield data from the trend window. NB 96 = one day of 15 minute periods"""
    yield from df[:periods]

def main(in_path: str = "BTCUSDC.csv", out_path: str = "BTCUSDC_indicators.csv") -> None:
    df = pd.read_csv(in_path)
    df["trend_direction"] = df.apply(trend_direction, axis=1)
    df.to_csv(out_path)

if __name__ == "__main__":
    main()