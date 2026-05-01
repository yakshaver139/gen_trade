"""Tests for parent-selection weighting (Phase 1).

The previous `apply_ranking` formula `weight_i = n_items + 1 / (i + 1)`
collapses to nearly uniform — selection pressure was effectively random.
Phase 1 replaces it with deliberate, configurable schemes:

  - tournament (default, k=3): weight_i = (N-i)^3 - (N-i-1)^3
  - rank_linear:                weight_i = N - i
  - fitness_proportional:       weight_i ∝ fitness shifted to non-negative

These tests pin the contract before implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gentrade.selection import SELECTION_PRESSURES, compute_weights


def make_ranked(fitnesses: list[float]) -> pd.DataFrame:
    """Build a fitness-sorted (descending) frame in the shape `apply_ranking` produces."""
    sorted_f = sorted(fitnesses, reverse=True)
    rows = [
        {"id": f"s{i}", "strategy": {"id": f"s{i}"}, "fitness": f}
        for i, f in enumerate(sorted_f)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# rank_linear
# ---------------------------------------------------------------------------

def test_rank_linear_weights_strictly_decreasing():
    df = make_ranked([10.0, 5.0, 0.0, -5.0, -10.0])
    weights = compute_weights(df, "rank_linear")
    values = list(weights.values())
    assert values == [5, 4, 3, 2, 1]


def test_rank_linear_best_sampled_more_than_median():
    """rank_linear gives the best 2× the sampling probability of the median."""
    n = 10
    df = make_ranked([float(x) for x in range(n, 0, -1)])
    weights = compute_weights(df, "rank_linear")

    rng = np.random.default_rng(42)
    ids = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    probs /= probs.sum()

    draws = rng.choice(ids, size=200_000, p=probs)
    counts = pd.Series(draws).value_counts()
    best_id, median_id = ids[0], ids[n // 2]

    ratio = counts[best_id] / counts[median_id]
    assert ratio >= 1.8, f"best/median sampling ratio {ratio:.2f} < 1.8"


def test_default_pressure_meets_phase1_3x_target():
    """The PLAN target: best is sampled >=3x more often than median.

    The default pressure (tournament k=3) clears this with margin; rank_linear
    on its own is mathematically capped at 2x for N=10 so the default carries
    the deliverable.
    """
    from gentrade.selection import DEFAULT_PRESSURE

    n = 10
    df = make_ranked([float(x) for x in range(n, 0, -1)])
    weights = compute_weights(df, DEFAULT_PRESSURE)

    rng = np.random.default_rng(42)
    ids = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    probs /= probs.sum()

    draws = rng.choice(ids, size=200_000, p=probs)
    counts = pd.Series(draws).value_counts()
    best_id, median_id = ids[0], ids[n // 2]

    ratio = counts[best_id] / counts[median_id]
    assert ratio >= 3.0, f"default pressure best/median ratio {ratio:.2f} < 3.0"


def test_rank_linear_weights_keyed_by_strategy_id_in_row_order():
    """`pandas.DataFrame.sample(weights=...)` requires order alignment with the frame."""
    df = make_ranked([7.0, 3.0, 1.0])
    weights = compute_weights(df, "rank_linear")
    assert list(weights.keys()) == ["s0", "s1", "s2"]


# ---------------------------------------------------------------------------
# tournament
# ---------------------------------------------------------------------------

def test_tournament_weights_strictly_decreasing():
    df = make_ranked([10.0, 5.0, 0.0, -5.0, -10.0])
    weights = compute_weights(df, "tournament")
    values = list(weights.values())
    # k=3, N=5: (N-i)^3 - (N-i-1)^3 → 61, 37, 19, 7, 1
    assert values == [61, 37, 19, 7, 1]


def test_tournament_concentrates_more_pressure_than_rank_linear():
    df = make_ranked([float(x) for x in range(10, 0, -1)])
    rl = compute_weights(df, "rank_linear")
    tn = compute_weights(df, "tournament")
    rl_top = list(rl.values())[0] / sum(rl.values())
    tn_top = list(tn.values())[0] / sum(tn.values())
    assert tn_top > rl_top


# ---------------------------------------------------------------------------
# fitness_proportional
# ---------------------------------------------------------------------------

def test_fitness_proportional_orders_by_fitness():
    df = make_ranked([10.0, 5.0, 0.0, -5.0])
    weights = compute_weights(df, "fitness_proportional")
    values = list(weights.values())
    assert values[0] > values[1] > values[2] > values[3]


def test_fitness_proportional_all_weights_non_negative():
    """Negative fitness values must not produce negative weights — pandas would reject."""
    df = make_ranked([5.0, 0.0, -10.0, -50.0])
    weights = compute_weights(df, "fitness_proportional")
    assert all(w >= 0 for w in weights.values())
    # the weakest must still have a strictly positive weight so pandas can sample it
    assert min(weights.values()) > 0


def test_fitness_proportional_constant_fitness_falls_back_to_uniform():
    """When all fitnesses are equal, weights should be equal and positive."""
    df = make_ranked([2.0, 2.0, 2.0, 2.0])
    weights = compute_weights(df, "fitness_proportional")
    values = list(weights.values())
    assert min(values) > 0
    assert all(v == values[0] for v in values)


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------

def test_unknown_pressure_raises():
    df = make_ranked([1.0, 0.0])
    with pytest.raises(ValueError, match="Unknown selection pressure"):
        compute_weights(df, "not_a_pressure")


def test_supported_pressures_match_spec():
    assert set(SELECTION_PRESSURES) == {"rank_linear", "tournament", "fitness_proportional"}


def test_apply_ranking_uses_configured_pressure():
    """`apply_ranking` must propagate the pressure choice through to the weights."""
    from gentrade.fitness_functions import transform_fitness_results

    # Build two fake fitness-report frames the GA shape expects.
    def fake_report(strat_id: str, fitness: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "id": strat_id,
                    "strategy": {"id": strat_id},
                    "fitness": fitness,
                    "n_trades": 5,
                    "win_percent": 50.0,
                    "avg_gain": 0.01,
                    "performance": {},
                    "target": {},
                    "stop_loss": {},
                    "result": {},
                    "open_ts": {},
                    "trend": {},
                }
            ]
        )

    _ = transform_fitness_results  # imported to ensure the module loads

    from gentrade.genetic import apply_ranking

    results = [fake_report(f"s{i}", float(10 - i)) for i in range(5)]
    df_rl, w_rl = apply_ranking(results, pressure="rank_linear")
    df_tn, w_tn = apply_ranking(results, pressure="tournament")

    assert list(w_rl.values()) == [5, 4, 3, 2, 1]
    assert list(w_tn.values()) == [61, 37, 19, 7, 1]
    # ranking shape unchanged
    assert df_rl["fitness"].tolist() == sorted(df_rl["fitness"].tolist(), reverse=True)
    assert df_tn["fitness"].tolist() == sorted(df_tn["fitness"].tolist(), reverse=True)
