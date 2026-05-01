import ast
import logging

import pandas as pd

from gentrade.env import BUY_AMOUNT, HIT, NA, STOP_LOSS, STOPPED, TARGET
from gentrade.helpers import get_latest_path
from gentrade.run_strategy import loss, win

AMOUNT = 1000


def main(path):
    df = pd.read_csv(path)
    logger = logging.getLogger(__name__)
    res = dict()
    account = 0
    for ix in range(len(df)):
        row = df.iloc[ix]
        logger.info(f"Evaluating {row.id}")
        results = ast.literal_eval(row.result).values()
        for x in results:
            if x == HIT:
                account += win(AMOUNT)
                logger.info(f"target HIT setting account total to {account}")
            elif x == STOPPED:
                account -= loss(AMOUNT)
                logger.info(f"target MISSED setting account total to {account}")
            res[row.id] = account
    return res


def calculate_cumulative_profit(
    path: str | None = None, df: pd.DataFrame | None = None
) -> dict:
    """groups consecutive results to calculate the cumulative profit and loss.

    NB this would be useful for working out the compounded P&L, but not in use.
    """

    df = df or pd.read_csv(path)
    output = dict()

    for ix in range(len(df)):
        wins = []
        losses = []
        holds = []
        row = df.iloc[ix]
        results = list(ast.literal_eval(row.result).values())
        streak = 1
        for ix, res in enumerate(results):
            win = res == HIT
            loss = res == STOPPED
            hold = res == NA
            switched = False
            try:
                if res == results[ix + 1]:
                    streak += 1
                else:
                    switched = True
                    wins, losses, holds = increment_counters(
                        streak, win, loss, hold, wins, losses, holds
                    )
                    streak = 1
            except IndexError:
                if not switched:
                    wins, losses, holds = increment_counters(
                        streak, win, loss, hold, wins, losses, holds
                    )
                profit = get_profit(wins, losses)
                output[row.id] = dict(
                    wins=wins, losses=losses, holds=holds, profit=profit
                )

    return output


def calculate_simple_profit(
    path: str | None = None, df: pd.DataFrame | None = None
) -> dict:
    try:
        if len(df):
            df = df
    except (TypeError, ValueError):
        df = pd.read_csv(path) if path else pd.DataFrame(columns=["id", "results"])
    output = dict()

    for ix in range(len(df)):
        total = 10000
        traded = []
        row = df.iloc[ix]
        try:
            results = list(ast.literal_eval(row.result).values())
        except ValueError:
            results = list(row.result.values())
        # wins = results.count(HIT)
        # losses = results.count(STOPPED)
        # output[row.id] = (wins * AMOUNT * TARGET) - (losses * AMOUNT * STOP_LOSS)
        for res in results:
            traded.append(res in (HIT, STOPPED))
            if res == HIT:
                total += BUY_AMOUNT * TARGET
            elif res == STOPPED:
                total -= BUY_AMOUNT * STOP_LOSS

        # take away the starting ammount
        total -= 10000

        output[row.id] = total

    return output


def increment_counters(
    streak: int,
    win: bool,
    loss: bool,
    hold: bool,
    wins: list,
    losses: list,
    holds: list,
) -> tuple:
    if win:
        wins.append(streak)
        losses.append(0)
        holds.append(0)
    elif loss:
        wins.append(0)
        losses.append(streak)
        holds.append(0)
    elif hold:
        wins.append(0)
        losses.append(0)
        holds.append(streak)
    return wins, losses, holds


def get_profit(wins: list, losses: list) -> int:
    zipper = zip(wins, losses, strict=True)
    profit = 0
    for n_wins, n_losses in zipper:
        if n_wins > n_losses:
            profit += sum(win(AMOUNT) for x in range(n_wins))
        elif n_losses > n_wins:
            profit -= sum(loss(AMOUNT) for x in range(n_losses))
        else:
            profit += 0
    return profit


if __name__ == "__main__":
    path = get_latest_path(".", "csv")
    # logging.basicConfig(
    #     filename=f"{path.split('+')[0]}_trade_log",
    #     filemode="w",
    #     format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    #     datefmt="%H:%M:%S",
    #     level=logging.INFO,
    # )
    results = calculate_simple_profit(path)
