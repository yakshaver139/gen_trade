"""Mutation operators for the genetic algorithm.

Replaces the original ``genetic.mutate`` — which only ever perturbed
one absolute threshold by a uniform ±20% — with seven probabilistic
operators that span both parameter and structural mutation. The
dissertation's "simplistic mutation" critique is faithful to the old
behaviour; this module is the answer.

Operators (per call, one is sampled by `MutationConfig` weights):

  perturb_threshold  multiply abs_value by (1 ± noise), noise = N(0, σ) or U(-σ, σ)
                     clamps to per-indicator catalogue [min, max] when configured
  flip_operator      swap >= ↔ <= or > ↔ <  (class-agnostic, works on relative too)
  swap_indicator     replace one signal with another of the same class
  flip_conjunction   and ↔ or at index ≥ 1 (preserves FirstConjunctionIsAnd)
  add_signal         append a signal of an under-budget class (≤ max_signals)
  remove_signal      drop a signal (≥ min_signals); forces conjunctions[0]="and"
  swap_rel_target    on a relative signal, swap rel_value to another catalogue
                     entry for the same indicator. Lets the GA actually mutate
                     relative-only chromosomes (the old mutate skipped them).

Determinism: every operator takes an explicit ``rng`` (a
``random.Random`` instance) so unit tests can pass ``random.Random(0)``
without polluting the global. Production callers pass the ``random``
module itself, preserving byte-equivalence on resume (the global RNG
state is what `gentrade.persistence.serialise_rng_states` snapshots).

Spec invariants preserved by construction:
- ConjunctionCardinality (len(conjunctions) == len(indicators) - 1)
- SignalCountInRange      (min_signals ≤ len(indicators) ≤ max_signals)
- SameClassLimit          (≤ max_same_class_signals per type)
- FirstConjunctionIsAnd   (conjunctions[0] == "and" if any)

Operators that find themselves inapplicable to a chromosome (e.g.
``remove_signal`` on a min-length one) record a no-op in
`MutationOutcome` rather than retrying — keeps wall-clock bounded
and the per-operator hit rate observable.
"""
from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Config + outcome
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationConfig:
    """Per-operator weights and bounded behaviour for the mutator.

    Weights need not sum to 1 — they're normalised at sampling time.
    A weight of 0 disables an operator entirely.
    """

    perturb_threshold: float = 0.30
    flip_operator: float = 0.15
    swap_indicator: float = 0.15
    flip_conjunction: float = 0.10
    add_signal: float = 0.10
    remove_signal: float = 0.10
    swap_rel_target: float = 0.10

    threshold_noise: Literal["uniform", "gaussian"] = "gaussian"
    threshold_scale: float = 0.20  # σ for gaussian, half-range for uniform
    clamp_to_catalogue_range: bool = True

    min_signals: int = 2
    max_signals: int = 4
    max_same_class_signals: int = 2

    @classmethod
    def legacy(cls) -> MutationConfig:
        """Pre-improvement: perturb_threshold only, uniform [-0.20, +0.20], no clamp.

        Used by tests that pin the old determinism guarantees.
        """
        return cls(
            perturb_threshold=1.0,
            flip_operator=0.0,
            swap_indicator=0.0,
            flip_conjunction=0.0,
            add_signal=0.0,
            remove_signal=0.0,
            swap_rel_target=0.0,
            threshold_noise="uniform",
            threshold_scale=0.20,
            clamp_to_catalogue_range=False,
        )

    @classmethod
    def rich(cls) -> MutationConfig:
        """Default for new runs — the rates above as a named constructor."""
        return cls()

    def weights(self) -> dict[str, float]:
        """Operator → weight, in the order operators are sampled."""
        return {
            "perturb_threshold": self.perturb_threshold,
            "flip_operator": self.flip_operator,
            "swap_indicator": self.swap_indicator,
            "flip_conjunction": self.flip_conjunction,
            "add_signal": self.add_signal,
            "remove_signal": self.remove_signal,
            "swap_rel_target": self.swap_rel_target,
        }


@dataclass(frozen=True)
class MutationOutcome:
    """What happened in one ``mutate_strategy`` call."""

    operator: str
    applied: bool
    reason: str | None = None


# ---------------------------------------------------------------------------
# Catalogue helpers (pure)
# ---------------------------------------------------------------------------
# The catalogue is small (~80 entries) so it's cheap to recompute these
# per call. Callers needing performance can pass precomputed lookups in
# via a richer config later; for now keep it simple.

def _threshold_bounds(catalogue: list[dict]) -> dict[str, tuple[float, float]]:
    """Per-indicator [min, max] of catalogue absolute thresholds."""
    out: dict[str, list[float]] = {}
    for s in catalogue:
        if s.get("absolute") and s.get("abs_value") is not None:
            out.setdefault(s["indicator"], []).append(float(s["abs_value"]))
    return {k: (min(v), max(v)) for k, v in out.items()}


def _indicators_by_class(catalogue: list[dict]) -> dict[str, list[dict]]:
    """Catalogue entries grouped by ``type`` (momentum / trend / volatility / volume)."""
    out: dict[str, list[dict]] = {}
    for s in catalogue:
        out.setdefault(s.get("type", ""), []).append(s)
    return out


def _rel_targets_for(catalogue: list[dict], indicator: str) -> list[str]:
    """The set of `rel_value` strings the catalogue pairs with this indicator."""
    seen: set[str] = set()
    for s in catalogue:
        if s["indicator"] == indicator and not s.get("absolute"):
            rv = s.get("rel_value")
            if rv:
                seen.add(rv)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OP_FLIP = {">=": "<=", "<=": ">=", ">": "<", "<": ">"}
_CONJ_FLIP = {"and": "or", "or": "and"}


def _class_counts(strategy: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in strategy["indicators"]:
        out[s.get("type", "")] = out.get(s.get("type", ""), 0) + 1
    return out


def _gauss_or_uniform(rng: random.Random, kind: str, scale: float) -> float:
    if kind == "gaussian":
        return rng.gauss(0.0, scale)
    return rng.uniform(-scale, scale)


# ---------------------------------------------------------------------------
# Operators — each returns (new_strategy, applied: bool, reason: str | None)
# ---------------------------------------------------------------------------

def _perturb_threshold(
    strat: dict, config: MutationConfig, catalogue: list[dict], rng: random.Random
) -> tuple[dict, bool, str | None]:
    abs_idx = [i for i, s in enumerate(strat["indicators"]) if s.get("absolute")]
    if not abs_idx:
        return strat, False, "no absolute signals"

    i = rng.choice(abs_idx)
    sig = strat["indicators"][i]
    noise = _gauss_or_uniform(rng, config.threshold_noise, config.threshold_scale)
    new_value = float(sig["abs_value"]) * (1.0 + noise)

    if config.clamp_to_catalogue_range:
        bounds = _threshold_bounds(catalogue)
        if sig["indicator"] in bounds:
            lo, hi = bounds[sig["indicator"]]
            # The catalogue may only have one threshold per indicator
            # (lo == hi); widen by 50% so perturb still has somewhere to
            # land. Otherwise clamp tight to the observed range.
            if lo == hi:
                span = max(abs(lo) * 0.5, 1.0)
                lo, hi = lo - span, hi + span
            new_value = max(lo, min(hi, new_value))

    sig["abs_value"] = new_value
    return strat, True, None


def _flip_operator(
    strat: dict, config: MutationConfig, catalogue: list[dict], rng: random.Random
) -> tuple[dict, bool, str | None]:
    if not strat["indicators"]:
        return strat, False, "no signals"
    i = rng.randrange(len(strat["indicators"]))
    sig = strat["indicators"][i]
    new_op = _OP_FLIP.get(sig.get("op"))
    if new_op is None:
        return strat, False, f"unknown op {sig.get('op')!r}"
    sig["op"] = new_op
    return strat, True, None


def _swap_indicator(
    strat: dict, config: MutationConfig, catalogue: list[dict], rng: random.Random
) -> tuple[dict, bool, str | None]:
    if not strat["indicators"]:
        return strat, False, "no signals"
    i = rng.randrange(len(strat["indicators"]))
    sig = strat["indicators"][i]
    sig_class = sig.get("type", "")

    by_class = _indicators_by_class(catalogue)
    pool = [s for s in by_class.get(sig_class, []) if s["indicator"] != sig["indicator"]]
    # Match shape: absolute → only absolute candidates; relative → only relative.
    pool = [s for s in pool if bool(s.get("absolute")) == bool(sig.get("absolute"))]
    if not pool:
        return strat, False, f"no other {sig_class!r} indicator of matching shape"

    pick = rng.choice(pool)
    # Replace fields wholesale so the new signal is well-formed.
    strat["indicators"][i] = dict(pick)
    return strat, True, None


def _flip_conjunction(
    strat: dict, config: MutationConfig, catalogue: list[dict], rng: random.Random
) -> tuple[dict, bool, str | None]:
    conjs = strat["conjunctions"]
    if len(conjs) < 2:
        return strat, False, "fewer than 2 conjunctions"
    # Index 0 is locked to "and" (FirstConjunctionIsAnd).
    i = rng.randrange(1, len(conjs))
    new = _CONJ_FLIP.get(conjs[i])
    if new is None:
        return strat, False, f"unknown conjunction {conjs[i]!r}"
    conjs[i] = new
    return strat, True, None


def _add_signal(
    strat: dict, config: MutationConfig, catalogue: list[dict], rng: random.Random
) -> tuple[dict, bool, str | None]:
    if len(strat["indicators"]) >= config.max_signals:
        return strat, False, f"at max_signals={config.max_signals}"
    counts = _class_counts(strat)
    by_class = _indicators_by_class(catalogue)
    eligible_classes = [
        c for c in by_class
        if counts.get(c, 0) < config.max_same_class_signals
    ]
    if not eligible_classes:
        return strat, False, "no class has same-class budget"
    cls = rng.choice(eligible_classes)
    pick = rng.choice(by_class[cls])
    strat["indicators"].append(dict(pick))
    # Conjunction at slot len-1 is "and" — the safest choice and never
    # touches index 0. ConjunctionCardinality preserved.
    strat["conjunctions"].append("and")
    if strat["conjunctions"]:
        strat["conjunctions"][0] = "and"
    return strat, True, None


def _remove_signal(
    strat: dict, config: MutationConfig, catalogue: list[dict], rng: random.Random
) -> tuple[dict, bool, str | None]:
    if len(strat["indicators"]) <= config.min_signals:
        return strat, False, f"at min_signals={config.min_signals}"
    i = rng.randrange(len(strat["indicators"]))
    strat["indicators"].pop(i)
    # Drop the matching conjunction. For i == 0, drop conjunction 0; else i-1.
    if strat["conjunctions"]:
        strat["conjunctions"].pop(max(0, i - 1) if i > 0 else 0)
        if strat["conjunctions"]:
            strat["conjunctions"][0] = "and"
    return strat, True, None


def _swap_rel_target(
    strat: dict, config: MutationConfig, catalogue: list[dict], rng: random.Random
) -> tuple[dict, bool, str | None]:
    rel_idx = [i for i, s in enumerate(strat["indicators"]) if not s.get("absolute")]
    if not rel_idx:
        return strat, False, "no relative signals"
    i = rng.choice(rel_idx)
    sig = strat["indicators"][i]
    targets = _rel_targets_for(catalogue, sig["indicator"])
    alts = [t for t in targets if t != sig.get("rel_value")]
    if not alts:
        return strat, False, f"no alternative rel_value for {sig['indicator']!r}"
    sig["rel_value"] = rng.choice(alts)
    return strat, True, None


_OPERATORS = {
    "perturb_threshold": _perturb_threshold,
    "flip_operator": _flip_operator,
    "swap_indicator": _swap_indicator,
    "flip_conjunction": _flip_conjunction,
    "add_signal": _add_signal,
    "remove_signal": _remove_signal,
    "swap_rel_target": _swap_rel_target,
}


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def mutate_strategy(
    strategy: dict,
    config: MutationConfig,
    catalogue: list[dict],
    rng: random.Random | Any = random,
) -> tuple[dict, MutationOutcome]:
    """Pick one operator weighted by ``config`` and apply it.

    Always deep-copies before mutating so the input strategy is left
    intact. Returns the (possibly unchanged) child plus a
    :class:`MutationOutcome` describing what happened — useful for
    per-generation telemetry.
    """
    weights = config.weights()
    names = [n for n, w in weights.items() if w > 0]
    if not names:
        return deepcopy(strategy), MutationOutcome(
            operator="(none)", applied=False, reason="all operator weights zero"
        )
    ws = [weights[n] for n in names]
    op_name = rng.choices(names, weights=ws, k=1)[0]
    op = _OPERATORS[op_name]
    child = deepcopy(strategy)
    new_child, applied, reason = op(child, config, catalogue, rng)
    return new_child, MutationOutcome(operator=op_name, applied=applied, reason=reason)
