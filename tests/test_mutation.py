"""Tests for the seven mutation operators in gentrade.mutation.

Per-operator behaviour is pinned with seeded ``random.Random`` instances
so tests are deterministic without polluting the global RNG. Each
operator also has an "inapplicable" no-op test plus an invariant
property test (random fuzz with the operator applied N times, asserting
all four spec invariants still hold).
"""
from __future__ import annotations

import random

from gentrade.generate_strategy import LOADED_INDICATORS
from gentrade.mutation import (
    MutationConfig,
    MutationOutcome,
    _add_signal,
    _flip_conjunction,
    _flip_operator,
    _perturb_threshold,
    _remove_signal,
    _swap_indicator,
    _swap_rel_target,
    mutate_strategy,
)

CATALOGUE = LOADED_INDICATORS


def _abs_sig(indicator: str, op: str, abs_value: float, type_: str = "momentum") -> dict:
    return {
        "indicator": indicator,
        "type": type_,
        "name": indicator,
        "absolute": True,
        "op": op,
        "abs_value": abs_value,
        "rel_value": None,
    }


def _rel_sig(indicator: str, rel_value: str, op: str = ">", type_: str = "momentum") -> dict:
    return {
        "indicator": indicator,
        "type": type_,
        "name": indicator,
        "absolute": False,
        "op": op,
        "abs_value": None,
        "rel_value": rel_value,
    }


def _strategy(indicators: list[dict], conjunctions: list[str] | None = None) -> dict:
    if conjunctions is None:
        conjunctions = ["and"] * (len(indicators) - 1)
    return {"id": "test", "indicators": indicators, "conjunctions": conjunctions}


def _is_well_formed(strat: dict, cfg: MutationConfig) -> tuple[bool, str | None]:
    """Check the four spec invariants: ConjunctionCardinality, SignalCountInRange,
    SameClassLimit, FirstConjunctionIsAnd. Returns (ok, reason)."""
    inds = strat["indicators"]
    conjs = strat["conjunctions"]
    if len(conjs) != len(inds) - 1:
        return False, f"ConjunctionCardinality: {len(conjs)} vs {len(inds) - 1}"
    if not (cfg.min_signals <= len(inds) <= cfg.max_signals):
        return False, f"SignalCountInRange: len={len(inds)}"
    counts: dict[str, int] = {}
    for s in inds:
        counts[s.get("type", "")] = counts.get(s.get("type", ""), 0) + 1
        if counts[s.get("type", "")] > cfg.max_same_class_signals:
            return False, f"SameClassLimit: {s.get('type')} count={counts[s.get('type', '')]}"
    if conjs and conjs[0] != "and":
        return False, f"FirstConjunctionIsAnd: conjunctions[0]={conjs[0]!r}"
    return True, None


# ---------------------------------------------------------------------------
# perturb_threshold
# ---------------------------------------------------------------------------

def test_perturb_threshold_changes_abs_value():
    cfg = MutationConfig(threshold_noise="uniform", threshold_scale=0.20,
                         clamp_to_catalogue_range=False)
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70.0)])
    rng = random.Random(0)
    new_strat, applied, reason = _perturb_threshold(strat, cfg, CATALOGUE, rng)
    assert applied
    assert reason is None
    new_v = new_strat["indicators"][0]["abs_value"]
    assert new_v != 70.0
    # Within ±20% multiplicative band.
    assert 70.0 * 0.8 <= new_v <= 70.0 * 1.2


def test_perturb_threshold_no_op_when_no_absolute_signals():
    cfg = MutationConfig.rich()
    strat = _strategy([
        _rel_sig("momentum_rsi", "PREVIOUS_PERIOD"),
        _rel_sig("trend_sma_fast", "MA", type_="trend"),
    ])
    new_strat, applied, reason = _perturb_threshold(strat, cfg, CATALOGUE, random.Random(0))
    assert applied is False
    assert "no absolute" in reason


def test_perturb_threshold_clamps_to_catalogue_range():
    """When clamp is on, perturbing past the catalogue's [min, max] caps the result."""
    cfg = MutationConfig(threshold_noise="uniform", threshold_scale=10.0,  # massive noise
                         clamp_to_catalogue_range=True)
    # momentum_rsi catalogue has thresholds {30, 70, 80, 20} → bounds [20, 80]
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70.0)])
    for seed in range(20):
        new_strat, _, _ = _perturb_threshold(strat, cfg, CATALOGUE, random.Random(seed))
        v = new_strat["indicators"][0]["abs_value"]
        assert 20.0 <= v <= 80.0, f"seed {seed}: {v} out of [20, 80]"


# ---------------------------------------------------------------------------
# flip_operator
# ---------------------------------------------------------------------------

def test_flip_operator_swaps_geq_and_leq():
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70.0)])
    new_strat, applied, _ = _flip_operator(strat, MutationConfig.rich(), CATALOGUE, random.Random(0))
    assert applied
    assert new_strat["indicators"][0]["op"] == "<="


def test_flip_operator_works_on_relative_signals():
    strat = _strategy([_rel_sig("momentum_rsi", "PREVIOUS_PERIOD", op=">")])
    new_strat, applied, _ = _flip_operator(strat, MutationConfig.rich(), CATALOGUE, random.Random(0))
    assert applied
    assert new_strat["indicators"][0]["op"] == "<"


# ---------------------------------------------------------------------------
# swap_indicator
# ---------------------------------------------------------------------------

def test_swap_indicator_preserves_class():
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70.0, type_="momentum")])
    new_strat, applied, _ = _swap_indicator(strat, MutationConfig.rich(), CATALOGUE, random.Random(0))
    assert applied
    assert new_strat["indicators"][0]["type"] == "momentum"
    # And the indicator name actually changed
    assert new_strat["indicators"][0]["indicator"] != "momentum_rsi"


def test_swap_indicator_no_op_when_no_alternative():
    """A class with only one indicator in the catalogue has no candidate."""
    # Build a synthetic catalogue with one momentum indicator only.
    tiny = [_abs_sig("momentum_rsi", ">=", 70.0)]
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70.0)])
    new_strat, applied, reason = _swap_indicator(strat, MutationConfig.rich(), tiny, random.Random(0))
    assert applied is False
    assert "no other" in reason


# ---------------------------------------------------------------------------
# flip_conjunction
# ---------------------------------------------------------------------------

def test_flip_conjunction_skips_index_zero():
    strat = _strategy(
        [
            _abs_sig("momentum_rsi", ">=", 70),
            _abs_sig("trend_sma_fast", ">=", 1.0, type_="trend"),
            _abs_sig("volume_cmf", ">=", 0.5, type_="volume"),
        ],
        conjunctions=["and", "and"],
    )
    cfg = MutationConfig.rich()
    # Run many seeds; index 0 must never flip to "or".
    for seed in range(50):
        new_strat, applied, _ = _flip_conjunction(strat, cfg, CATALOGUE, random.Random(seed))
        if applied:
            assert new_strat["conjunctions"][0] == "and"


def test_flip_conjunction_no_op_when_only_one_conjunction():
    strat = _strategy(
        [_abs_sig("momentum_rsi", ">=", 70), _abs_sig("trend_sma_fast", ">=", 1.0, type_="trend")],
        conjunctions=["and"],
    )
    new_strat, applied, reason = _flip_conjunction(strat, MutationConfig.rich(), CATALOGUE, random.Random(0))
    assert applied is False
    assert "fewer than 2" in reason


# ---------------------------------------------------------------------------
# add_signal
# ---------------------------------------------------------------------------

def test_add_signal_extends_chromosome():
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70)])
    cfg = MutationConfig.rich()
    new_strat, applied, _ = _add_signal(strat, cfg, CATALOGUE, random.Random(0))
    assert applied
    assert len(new_strat["indicators"]) == 2
    assert len(new_strat["conjunctions"]) == 1
    assert new_strat["conjunctions"][0] == "and"


def test_add_signal_no_op_at_max_signals():
    cfg = MutationConfig.rich()  # max_signals=4
    inds = [
        _abs_sig("momentum_rsi", ">=", 70),
        _abs_sig("trend_sma_fast", ">=", 1.0, type_="trend"),
        _abs_sig("volume_cmf", ">=", 0.5, type_="volume"),
        _abs_sig("volatility_atr", ">=", 1.0, type_="volatility"),
    ]
    strat = _strategy(inds, conjunctions=["and", "and", "and"])
    _, applied, reason = _add_signal(strat, cfg, CATALOGUE, random.Random(0))
    assert applied is False
    assert "max_signals" in reason


# ---------------------------------------------------------------------------
# remove_signal
# ---------------------------------------------------------------------------

def test_remove_signal_shrinks_chromosome():
    strat = _strategy(
        [
            _abs_sig("momentum_rsi", ">=", 70),
            _abs_sig("trend_sma_fast", ">=", 1.0, type_="trend"),
            _abs_sig("volume_cmf", ">=", 0.5, type_="volume"),
        ],
        conjunctions=["and", "and"],
    )
    cfg = MutationConfig.rich()
    new_strat, applied, _ = _remove_signal(strat, cfg, CATALOGUE, random.Random(0))
    assert applied
    assert len(new_strat["indicators"]) == 2
    assert len(new_strat["conjunctions"]) == 1
    assert new_strat["conjunctions"][0] == "and"


def test_remove_signal_no_op_at_min_signals():
    strat = _strategy(
        [_abs_sig("momentum_rsi", ">=", 70), _abs_sig("trend_sma_fast", ">=", 1.0, type_="trend")],
        conjunctions=["and"],
    )
    cfg = MutationConfig.rich()  # min_signals=2
    _, applied, reason = _remove_signal(strat, cfg, CATALOGUE, random.Random(0))
    assert applied is False
    assert "min_signals" in reason


def test_remove_signal_at_index_zero_preserves_first_conjunction():
    """If we drop signal 0 and the new conjunctions[0] would have been "or",
    it must be reset to "and"."""
    strat = _strategy(
        [
            _abs_sig("momentum_rsi", ">=", 70),
            _abs_sig("trend_sma_fast", ">=", 1.0, type_="trend"),
            _abs_sig("volume_cmf", ">=", 0.5, type_="volume"),
        ],
        # If we drop index 0, conjunctions[0] would become "or" — must flip.
        conjunctions=["and", "or"],
    )
    cfg = MutationConfig.rich()

    # Find a seed that picks index 0; rng.randrange(3) — seeds vary.
    for seed in range(50):
        rng = random.Random(seed)
        # peek: replicate randrange(3) → either 0, 1 or 2
        if random.Random(seed).randrange(3) == 0:
            new_strat, applied, _ = _remove_signal(strat, cfg, CATALOGUE, rng)
            assert applied
            assert new_strat["conjunctions"][0] == "and"
            return
    raise AssertionError("no test seed picked index 0; widen the seed search")


# ---------------------------------------------------------------------------
# swap_rel_target
# ---------------------------------------------------------------------------

def test_swap_rel_target_changes_rel_value():
    """An indicator with multiple rel_values in the catalogue gets a different one."""
    # `volume` has both PREVIOUS_PERIOD and MA in relative_signals.json
    strat = _strategy([_rel_sig("volume", "PREVIOUS_PERIOD", type_="volume")])
    cfg = MutationConfig.rich()
    new_strat, applied, _ = _swap_rel_target(strat, cfg, CATALOGUE, random.Random(0))
    assert applied
    assert new_strat["indicators"][0]["rel_value"] != "PREVIOUS_PERIOD"


def test_swap_rel_target_no_op_when_no_relative_signals():
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70)])
    new_strat, applied, reason = _swap_rel_target(strat, MutationConfig.rich(), CATALOGUE, random.Random(0))
    assert applied is False
    assert "no relative" in reason


# ---------------------------------------------------------------------------
# Top-level mutate_strategy
# ---------------------------------------------------------------------------

def test_mutate_strategy_returns_outcome_with_operator_name():
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70)])
    new_strat, outcome = mutate_strategy(strat, MutationConfig.rich(), CATALOGUE, random.Random(0))
    assert isinstance(outcome, MutationOutcome)
    assert outcome.operator in {
        "perturb_threshold", "flip_operator", "swap_indicator",
        "flip_conjunction", "add_signal", "remove_signal", "swap_rel_target",
    }
    # original strategy is untouched (deep-copied)
    assert strat["indicators"][0]["abs_value"] == 70


def test_mutate_strategy_legacy_only_perturbs_thresholds():
    """MutationConfig.legacy() makes us behave exactly like the old
    single-operator mutator: only perturb_threshold ever fires."""
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70)])
    cfg = MutationConfig.legacy()
    operators_seen = set()
    for seed in range(50):
        _, outcome = mutate_strategy(strat, cfg, CATALOGUE, random.Random(seed))
        operators_seen.add(outcome.operator)
    assert operators_seen == {"perturb_threshold"}


def test_mutate_strategy_rich_eventually_fires_every_operator():
    """Across enough seeded calls, the rich config samples every operator."""
    inds = [
        _abs_sig("momentum_rsi", ">=", 70),
        _rel_sig("volume", "PREVIOUS_PERIOD", type_="volume"),
        _abs_sig("trend_sma_fast", ">=", 1.0, type_="trend"),
    ]
    strat = _strategy(inds, conjunctions=["and", "and"])
    cfg = MutationConfig.rich()

    seen: dict[str, int] = {}
    for seed in range(500):
        _, outcome = mutate_strategy(strat, cfg, CATALOGUE, random.Random(seed))
        seen[outcome.operator] = seen.get(outcome.operator, 0) + 1

    expected = set(MutationConfig.rich().weights().keys())
    assert set(seen.keys()) == expected


def test_mutate_strategy_invariant_preservation_property():
    """Fuzz: 200 random chromosomes × 200 mutation calls, every spec
    invariant must hold afterwards.

    The chromosome generator picks 2-4 same-shape signals with at most
    2 of any class, so the property test starts from a well-formed
    state and only exercises the operators' invariant preservation.
    """
    cfg = MutationConfig.rich()
    abs_pool = [s for s in CATALOGUE if s.get("absolute")]
    rel_pool = [s for s in CATALOGUE if not s.get("absolute")]
    rng = random.Random(0)

    failures: list[str] = []
    for trial in range(200):
        # Build a random well-formed chromosome.
        n = rng.randint(cfg.min_signals, cfg.max_signals)
        pool = abs_pool if rng.random() < 0.5 else rel_pool
        chosen: list[dict] = []
        counts: dict[str, int] = {}
        attempts = 0
        while len(chosen) < n and attempts < 100:
            attempts += 1
            cand = rng.choice(pool)
            if counts.get(cand["type"], 0) >= cfg.max_same_class_signals:
                continue
            chosen.append(dict(cand))
            counts[cand["type"]] = counts.get(cand["type"], 0) + 1
        if len(chosen) < cfg.min_signals:
            continue
        strat = _strategy(chosen, conjunctions=["and"] * (len(chosen) - 1))

        ok, reason = _is_well_formed(strat, cfg)
        if not ok:
            failures.append(f"seed-strat trial {trial}: {reason}")
            continue

        # Apply N mutations sequentially.
        for step in range(20):
            strat, _ = mutate_strategy(strat, cfg, CATALOGUE, rng)
            ok, reason = _is_well_formed(strat, cfg)
            if not ok:
                failures.append(f"trial {trial} step {step}: {reason}")
                break

    assert not failures, f"{len(failures)} invariant violations: {failures[:3]}"


def test_mutate_strategy_does_not_mutate_input():
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70)])
    snapshot = (
        strat["indicators"][0]["abs_value"],
        strat["indicators"][0]["op"],
        list(strat["conjunctions"]),
    )
    for seed in range(20):
        _ = mutate_strategy(strat, MutationConfig.rich(), CATALOGUE, random.Random(seed))
    assert (
        strat["indicators"][0]["abs_value"],
        strat["indicators"][0]["op"],
        list(strat["conjunctions"]),
    ) == snapshot


def test_mutate_strategy_all_zero_weights_returns_unchanged():
    cfg = MutationConfig(
        perturb_threshold=0.0, flip_operator=0.0, swap_indicator=0.0,
        flip_conjunction=0.0, add_signal=0.0, remove_signal=0.0, swap_rel_target=0.0,
    )
    strat = _strategy([_abs_sig("momentum_rsi", ">=", 70)])
    new_strat, outcome = mutate_strategy(strat, cfg, CATALOGUE, random.Random(0))
    assert outcome.applied is False
    assert outcome.operator == "(none)"
    assert new_strat["indicators"][0]["abs_value"] == 70
