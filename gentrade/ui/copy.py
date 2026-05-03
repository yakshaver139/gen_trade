"""Single source of truth for metric tooltips + chart explanations.

Centralised so the Runs / Run detail / Strategy detail pages all
explain the same metric the same way. Each entry is intentionally
short — Streamlit ``help`` shows it as a small `?` next to the field
title, and that surface punishes wordiness.
"""
from __future__ import annotations

# ----- per-metric tooltips (used as st.metric(help=...)) -----

METRIC_HELP: dict[str, str] = {
    "n_trades": (
        "Number of trades that closed (target_hit / stopped_out / "
        "no_close_in_window) during the window."
    ),
    "win_rate": (
        "Fraction of closed trades whose return was positive *after* "
        "fees and slippage."
    ),
    "profit_factor": (
        "Gross winning P&L divided by gross losing P&L. >1 means winners "
        "outweigh losers; <1 means net loser; ∞ when there are no losses."
    ),
    "expectancy": (
        "Mean per-trade return after fees and slippage — the expected "
        "P&L per round trip. The fitness function uses this with a "
        "min-trades floor so a single lucky trade can't game the GA."
    ),
    "sharpe": (
        "Annualised Sharpe ratio: mean per-trade return / sample std "
        "dev, ×√365 (crypto trades 24/7). >1 is good, >2 is rare and "
        "should be treated with suspicion. NaN when fewer than 2 trades."
    ),
    "sortino": (
        "Like Sharpe but only the downside std dev (returns < 0) is "
        "the denominator. Higher = more upside per unit of downside "
        "volatility. ∞ when there are no losing trades."
    ),
    "calmar": (
        "Annualised return divided by |max drawdown|. Captures "
        "risk-adjusted growth. Higher is better; very small max "
        "drawdowns make this number explode."
    ),
    "max_drawdown": (
        "Worst peak-to-trough drop in equity, as a negative fraction. "
        "−0.20 means equity once fell 20% from a prior peak. "
        "Drawdown depth is what determines whether you can stay in a "
        "trade emotionally, regardless of long-run expectancy."
    ),
    "avg_trade_duration": (
        "Mean time from entry to exit across closed trades."
    ),
    "overfitting_gap": (
        "(final-gen train_max_fitness − validation_max_fitness) / "
        "|train_max_fitness|. Positive = the GA fit train better than "
        "it generalises. ~0 is the goal; >0.5 means the chosen "
        "strategy may be a curve fit."
    ),
    "current_generation": (
        "Generations completed so far. Bumps as the GA loop checkpoints."
    ),
    "rank": (
        "Position in the final-generation ranking (0 = best). "
        "Tournament selection samples lower ranks more aggressively."
    ),
    "fitness": (
        "Per-trade expectancy on the train window, after fees and "
        "slippage. Strategies producing fewer than the min-trades "
        "floor collapse to −1000."
    ),
    "status": (
        "in_progress → checkpointed every generation; reported → final "
        "BacktestReport persisted; failed → exception aborted the "
        "loop, partial state is resumable."
    ),
    "chosen_strategy": (
        "The top-ranked strategy from the final generation — the one "
        "the BacktestReport's headline metrics are computed against. "
        "Click to see the equity curve / drawdown / trade scatter."
    ),
}


# ----- chart-level captions (used under st.plotly_chart) -----

CHART_HELP: dict[str, str] = {
    "fitness_curves": (
        "Solid lines = max fitness; dotted = median. Blue = train (drives "
        "selection); orange = validation (diagnostic only — never feeds "
        "selection). When the orange max curve flattens or peels away "
        "from the blue, the GA is over-fitting the train window."
    ),
    "equity_curve": (
        "Per-trade compounded return on the test window, starting at 1.0. "
        "Each marker is a trade exit; a flat segment means no new trade "
        "between exits. Slope ≠ price chart — this is the strategy's "
        "equity, not the asset's."
    ),
    "drawdown_curve": (
        "Equity's distance from its running peak, as a fraction. 0 = at "
        "all-time high; the most negative point is the max drawdown. "
        "A strategy that recovers quickly has a sharp V; one that "
        "compounds losses has a long, deep U."
    ),
    "trade_scatter": (
        "Each dot is one closed trade. X = entry time, Y = net return. "
        "Green = win, red = loss. Hover for outcome "
        "(TARGET_HIT / STOPPED_OUT / NO_CLOSE_IN_WINDOW). Clusters of "
        "red mean the strategy is exposed to a regime it doesn't handle."
    ),
    "lineage_tree": (
        "Strategy DAG across generations. Each dot is a strategy "
        "(x = generation, y = rank within that generation; rank 0 sits "
        "at the top). Lines connect every child to its two PPX-crossover "
        "parents in the previous generation. Edge colour matches the "
        "operator chart above — pick a colour from the legend to spot "
        "where each operator carried genetic material forward. Node "
        "colour scales with train-window fitness on a viridis ramp; "
        "hover any node for its id, rank, and fitness."
    ),
    "operator_counts": (
        "Per-generation count of each mutation operator that fired while "
        "breeding the next population. Stacked bars: structural operators "
        "(swap_indicator / add_signal / remove_signal / flip_conjunction / "
        "swap_rel_target) move the GA into new chromosome shapes; "
        "parameter operators (perturb_threshold / flip_operator) tune the "
        "shape it's already in. (elite) bars are the top strategies copied "
        "unchanged. A run dominated by perturb_threshold is the old "
        "mutator's signature — sign of insufficient operator diversity."
    ),
    "breeding_events_table": (
        "One row per child in the population. Operator is the mutation "
        "applied after crossover; applied=False means the operator was "
        "inapplicable (e.g. remove_signal at min_signals) so the child "
        "is whatever crossover produced. Parents are the two strategies "
        "(by id) that contributed via PPX crossover. Elites carry "
        "self-as-both-parents and operator='(elite)'."
    ),
    "candlestick": (
        "Test-window OHLC candles, server-side downsampled to ≤1000 bars "
        "via OHLC aggregation. Blue triangles ▲ = entries; coloured "
        "triangles ▼ = exits — green for target_hit, red for stopped_out, "
        "grey for no_close_in_window. Hover any marker for the outcome + "
        "P&L. The candlestick is the asset's price; the equity curve "
        "below is the strategy's account."
    ),
    "cross_asset_table": (
        "Each row is the same chromosome re-run on a different asset's "
        "test slice. Big drops vs. the row for the run's training asset "
        "= the strategy was a curve fit. Same fee + slippage model "
        "across all assets."
    ),
}
