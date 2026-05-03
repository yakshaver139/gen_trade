"""Tests for invariants enforced by the chromosome-construction helpers
in ``gentrade.generate_strategy`` and the legacy genetic loop.
"""
from __future__ import annotations

import random

from gentrade.generate_strategy import (
    LOADED_INDICATORS,
    make_strategy_from_indicators,
)


def _abs_signals(n: int = 4) -> list[dict]:
    """Pick the first n absolute signals from the catalogue."""
    abs_only = [s for s in LOADED_INDICATORS if s.get("absolute")]
    assert len(abs_only) >= n
    return abs_only[:n]


def test_first_conjunction_is_always_and():
    """FirstConjunctionIsAnd spec invariant: every chromosome built via
    make_strategy_from_indicators must have conjunctions[0] == 'and'.

    Run many seeded trials so the random.choice is exercised over the
    full {and, or, and not, or not} domain — without the fix, an "or"
    or "or not" wins ~half the time at index 0.
    """
    inds = _abs_signals(4)
    for seed in range(200):
        random.seed(seed)
        strat = make_strategy_from_indicators(inds)
        assert len(strat["conjunctions"]) == len(strat["indicators"]) - 1
        if strat["conjunctions"]:
            assert strat["conjunctions"][0] == "and", (
                f"seed={seed}: first conjunction was "
                f"{strat['conjunctions'][0]!r}, expected 'and'"
            )


def test_make_strategy_from_indicators_single_indicator_has_no_conjunctions():
    """A 1-signal chromosome has zero conjunctions; the invariant is vacuous."""
    strat = make_strategy_from_indicators(_abs_signals(1))
    assert strat["conjunctions"] == []


def test_make_strategy_from_indicators_preserves_indicator_order():
    inds = _abs_signals(3)
    strat = make_strategy_from_indicators(inds)
    assert [i["indicator"] for i in strat["indicators"]] == [
        i["indicator"] for i in inds
    ]
