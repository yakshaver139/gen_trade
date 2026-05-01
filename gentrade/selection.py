"""Parent-selection weighting for the genetic algorithm.

The previous formula `weight_i = n_items + 1 / (i + 1)` collapses to a near-
uniform distribution: a population of 10 has weights all in [10, 11], so the
GA effectively sampled parents at random. Phase 1 replaces that with three
deliberate, configurable schemes.

Schemes
-------
- ``tournament`` (default, k=3): weights equivalent to a size-3 tournament,
  ``weight_i = (N − i)^3 − (N − i − 1)^3``. Best is sampled ≈4.4× more often
  than median for N=10, which clears the Phase 1 "best ≥ 3× median" target.
- ``rank_linear``: ``weight_i = N − i``. Best is sampled 2× more often than
  median; gentler pressure, less likely to converge prematurely.
- ``fitness_proportional``: weights proportional to (fitness − min_fitness),
  shifted so the worst still has a strictly positive weight so it can be
  sampled. Falls back to uniform when all fitnesses are equal.

The chosen scheme is read from ``SELECTION_PRESSURE`` env var at import time
and passed through ``apply_ranking`` so it is captured in the run manifest.
"""
from __future__ import annotations

import os

import pandas as pd

SELECTION_PRESSURES: tuple[str, ...] = (
    "rank_linear",
    "tournament",
    "fitness_proportional",
)
DEFAULT_PRESSURE: str = os.getenv("SELECTION_PRESSURE", "tournament")
TOURNAMENT_SIZE: int = 3


def compute_weights(ranked: pd.DataFrame, pressure: str = DEFAULT_PRESSURE) -> dict:
    """Return parent-sampling weights keyed by strategy id, in row order.

    ``ranked`` must already be sorted by fitness descending (as
    ``apply_ranking`` produces). The dict's insertion order matches the
    frame's row order so ``list(weights.values())`` aligns with
    ``DataFrame.sample(weights=...)``.
    """
    if pressure not in SELECTION_PRESSURES:
        raise ValueError(
            f"Unknown selection pressure {pressure!r}; "
            f"choose from {SELECTION_PRESSURES}"
        )

    n = len(ranked)
    if n == 0:
        return {}

    if pressure == "rank_linear":
        raw = [n - i for i in range(n)]
    elif pressure == "tournament":
        # Equivalent to size-k tournament selection without replacement:
        # P(rank i) ∝ (N - i)^k - (N - i - 1)^k.
        k = TOURNAMENT_SIZE
        raw = [(n - i) ** k - (n - i - 1) ** k for i in range(n)]
    else:  # fitness_proportional
        fitness = ranked["fitness"].astype(float).to_numpy()
        f_min, f_max = float(fitness.min()), float(fitness.max())
        spread = f_max - f_min
        if spread == 0:
            raw = [1.0] * n
        else:
            # Shift to non-negative; add a small epsilon so the worst still
            # has a strictly positive weight (pandas drops zero-weight rows
            # and a fully-zero `weights` arg is rejected).
            eps = spread * 1e-3
            raw = [float(f - f_min + eps) for f in fitness]

    return {ranked.iloc[i].strategy["id"]: raw[i] for i in range(n)}
