"""Gentrade CLI: run, list, resume.

Subcommands:

  gentrade run --data BARS.csv --strategies STRATS.json
               --population-size N --generations N --seed S
               [--db-url URL] [--allow-dirty] [--stop-after-generation K]
  gentrade list  [--db-url URL]
  gentrade resume RUN_ID --data BARS.csv [--db-url URL]

The default database is ``sqlite:///gentrade.db`` (overridable via the
``GENTRADE_DB_URL`` env var or ``--db-url`` on each command). Bars CSV is
loaded with pandas and must carry at least ``open_ts``, ``open``, ``high``,
``low``, ``close``; ``volume`` is optional but read if present. The
``open_ts`` column may be either an ISO 8601 string or epoch milliseconds.

The CLI does not yet generate strategies on the fly; pass a JSON file
listing the initial population. A future revision will compose with
``gentrade.generate_strategy.main`` for ``--auto`` generation.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from gentrade.backtest import BacktestConfig
from gentrade.ga import GAConfig, run_ga
from gentrade.manifest import compute_data_hash, current_git_sha
from gentrade.persistence import (
    DEFAULT_DB_URL,
    init_db,
    list_runs,
)

log = logging.getLogger(__name__)


def _load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "open_ts" not in df.columns:
        raise ValueError(f"{path}: missing required 'open_ts' column")
    # Accept either ISO strings or epoch ms.
    ts = df["open_ts"]
    if pd.api.types.is_numeric_dtype(ts):
        df["open_ts"] = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        df["open_ts"] = pd.to_datetime(ts, utc=True)
    return df


def _split_windows(
    bars: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[
    tuple[pd.Timestamp, pd.Timestamp],
    tuple[pd.Timestamp, pd.Timestamp],
    tuple[pd.Timestamp, pd.Timestamp],
]:
    """Chronological 60/20/20 split by row count by default."""
    n = len(bars)
    if n < 6:
        raise ValueError(f"need ≥6 bars, got {n}")
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return (
        (bars.iloc[0]["open_ts"], bars.iloc[train_end - 1]["open_ts"]),
        (bars.iloc[train_end]["open_ts"], bars.iloc[val_end - 1]["open_ts"]),
        (bars.iloc[val_end]["open_ts"], bars.iloc[n - 1]["open_ts"]),
    )


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    bars = _load_bars(args.data)
    strategies = json.loads(Path(args.strategies).read_text())
    train, val, test = _split_windows(bars)

    engine = init_db(args.db_url)

    code_sha: str | None = None
    try:
        code_sha = current_git_sha(allow_dirty=args.allow_dirty)
    except RuntimeError as e:
        if not args.allow_dirty:
            print(f"refusing to start: {e}", file=sys.stderr)
            return 2
    data_hash = compute_data_hash(bars)

    cfg = GAConfig(
        population_size=args.population_size,
        max_generations=args.generations,
        elitism_count=min(args.elitism_count, args.population_size),
        selection_pressure=args.selection_pressure,
        min_trades_for_fitness=args.min_trades_for_fitness,
    )
    bt_cfg = BacktestConfig(
        target_pct=args.target_pct,
        stop_loss_pct=args.stop_loss_pct,
        trade_window_bars=args.trade_window_bars,
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
    )

    result = run_ga(
        bars=bars,
        initial_strategies=strategies,
        train_window=train,
        validation_window=val,
        test_window=test,
        config=cfg,
        backtest_config=bt_cfg,
        seed=args.seed,
        code_sha=code_sha,
        data_hash=data_hash,
        engine=engine,
        stop_after_generation=args.stop_after_generation,
    )

    rows = list_runs(engine)
    run_id = rows[0]["id"]  # newest first
    chosen = result.backtest_report.chosen_strategy_id or "(in progress)"
    print(f"run_id={run_id} status={rows[0]['status']} chosen_strategy={chosen}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    engine = init_db(args.db_url)
    rows = list_runs(engine)
    if not rows:
        print("no runs persisted yet")
        return 0
    # tab-separated; easy to pipe to column / awk / jq
    print("\t".join(["id", "status", "gen", "started_at", "seed", "chosen", "gap"]))
    for r in rows:
        print(
            "\t".join(
                [
                    r["id"],
                    r["status"],
                    str(r["current_generation"]),
                    r["started_at"].isoformat() if r["started_at"] else "",
                    str(r["seed"]),
                    r["chosen_strategy_id"] or "-",
                    f"{r['overfitting_gap']:.4f}" if r["overfitting_gap"] is not None else "-",
                ]
            )
        )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    bars = _load_bars(args.data)
    engine = init_db(args.db_url)
    try:
        result = run_ga(
            bars=bars,
            engine=engine,
            resume_run_id=args.run_id,
        )
    except LookupError as e:
        print(f"resume failed: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"resume failed: {e}", file=sys.stderr)
        return 4
    chosen = result.backtest_report.chosen_strategy_id or "(in progress)"
    print(f"resumed run_id={args.run_id} status=reported chosen_strategy={chosen}")
    return 0


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gentrade")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="start a new GA run")
    run_p.add_argument("--data", required=True, help="path to bars CSV")
    run_p.add_argument("--strategies", required=True, help="path to initial strategies JSON")
    run_p.add_argument("--population-size", type=int, default=10)
    run_p.add_argument("--generations", type=int, default=50)
    run_p.add_argument("--elitism-count", type=int, default=2)
    run_p.add_argument("--selection-pressure", default="tournament",
                       choices=("tournament", "rank_linear", "fitness_proportional"))
    run_p.add_argument("--min-trades-for-fitness", type=int, default=3)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--db-url", default=DEFAULT_DB_URL)
    run_p.add_argument("--allow-dirty", action="store_true",
                       help="allow runs from a dirty git tree (won't be byte-reproducible)")
    run_p.add_argument("--target-pct", type=float, default=0.015)
    run_p.add_argument("--stop-loss-pct", type=float, default=0.0075)
    run_p.add_argument("--trade-window-bars", type=int, default=96)
    run_p.add_argument("--taker-fee-bps", type=float, default=10.0)
    run_p.add_argument("--slippage-bps", type=float, default=1.0)
    run_p.add_argument("--stop-after-generation", type=int, default=None,
                       help=argparse.SUPPRESS)  # internal: simulates a kill
    run_p.set_defaults(func=cmd_run)

    list_p = sub.add_parser("list", help="list persisted runs")
    list_p.add_argument("--db-url", default=DEFAULT_DB_URL)
    list_p.set_defaults(func=cmd_list)

    resume_p = sub.add_parser("resume", help="resume an interrupted run")
    resume_p.add_argument("run_id")
    resume_p.add_argument("--data", required=True, help="path to bars CSV (must match the run)")
    resume_p.add_argument("--db-url", default=DEFAULT_DB_URL)
    resume_p.set_defaults(func=cmd_resume)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
