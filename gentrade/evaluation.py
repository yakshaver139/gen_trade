"""Post-hoc analysis. Optional dependency group: ``pip install gentrade[analysis]``."""

import ast
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud

from gentrade.env import BUY_AMOUNT
from gentrade.genetic import load_trading_data


def plot_multiple(dfs):
    _dfs = {f"gen_{ix}": df for ix, df in enumerate(dfs)}

    # plot the data
    fig = go.Figure()

    for k, v in _dfs.items():
        fig = fig.add_trace(go.Scatter(x=v.id, y=v.fitness, name=k))
    return fig


def plot_price_increase(df):
    plt.plot("converted_open_ts", "close", data=df)

    plt.title("Bitcoin Price by Year")
    plt.ylabel("Price (USD)")
    plt.xlabel("Date")
    plt.show()


def percent_increase(df):
    earliest_close = df.iloc[0].close
    last_close = df.iloc[-1].close
    increase = (last_close - earliest_close) / last_close
    buy_hold = 10000 * increase
    return increase, buy_hold


def cumulative_returns(total):
    return total / BUY_AMOUNT


def annualised_returns(roi_percent, n_years=3.75):
    return (1 + roi_percent) ** (365 / (365 * n_years)) - 1


def sharpe_ratio(annualised_ret, sd_annualised_ret):
    # rate of returns of risk free investments, e.g. bonds
    risk_free_rate = 0.01 / 365
    return (annualised_ret - risk_free_rate) / sd_annualised_ret


def plotly_bubble(dfs, maxes, x):
    win_p = [x.win_percent.max() for x in dfs]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=maxes,
                mode="markers",
                marker=dict(
                    size=win_p,
                ),
            )
        ]
    )
    return fig

def collect_strats(dfs):
    top_15 = [x.iloc[:15][['strategy', 'fitness']] for x in dfs]
    out = []
    for df in top_15:
        for ix in range(len(df)):
            row = df.iloc[ix]
            parsed = ast.literal_eval(row.strategy)['parsed']
            out.append(dict(parsed_strategy=parsed, fitness=row.fitness))
    return top_15, out


def group_summary_results(summary_results):
    parsed = summary_results.parsed_strategy
    volatility = [x for x in parsed if 'volatility' in x]
    volume = [x for x in parsed if 'volume' in x]
    trend = [x for x in parsed if 'trend' in x]
    mo = [x for x in parsed if 'momentum' in x]
    return volatility, volume, trend, mo

def plot_wc(summary_results):
    wc = WordCloud().generate(summary_results.parsed.to_json())
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def plot_maxes(x, maxes):
    plt.plot(x, maxes, ".")
    plt.title("Max Fitness function across generations")
    plt.show()


def get_metadata(dfs):
    maxes = [x.fitness.max() for x in dfs]
    x = [_ for _ in range(1, len(dfs) + 1)]
    return maxes, x


def get_returns(dfs):
    top_15 = [x.iloc[:15] for x in dfs]
    maxes, x_axis = get_metadata(top_15)
    c_returns = [cumulative_returns(x) for x in maxes]
    a_returns = [annualised_returns(x) for x in c_returns]
    return c_returns, a_returns


if __name__ == "__main__":
    paths = sorted(os.listdir("outputs/ec2_output/"))
    dfs = [pd.read_csv(f"outputs/ec2_output/{p}") for p in paths]
    conc = pd.concat(dfs)
    price_data = load_trading_data()
