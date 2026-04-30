"""End-to-end smoke test: generate, backtest, score, breed on synthetic data."""

import pandas as pd

from gentrade.smoke import run_one_generation


def test_smoke_one_generation_produces_ranking():
    ranking = run_one_generation(population_size=4, seed=42)
    assert isinstance(ranking, pd.DataFrame)
    assert len(ranking) == 4
    assert "fitness" in ranking.columns
    # the GA must produce comparable fitness values
    assert ranking["fitness"].notna().all()
    # ranking is sorted descending
    fitnesses = ranking["fitness"].tolist()
    assert fitnesses == sorted(fitnesses, reverse=True)
