"""Gentrade CLI: ingest, run, list, show, resume.

Subcommands:

  gentrade ingest --exchange binance --asset BTC/USDT --interval 15m
                  --since 2022-01-01 [--until ...] [--out PATH]
  gentrade run    --data BARS.{csv,parquet} --strategies STRATS.json
                  --population-size N --generations N --seed S
                  [--db-url URL] [--allow-dirty] [--stop-after-generation K]
  gentrade list   [--db-url URL]
  gentrade show   RUN_ID [--db-url URL]
  gentrade resume RUN_ID --data BARS.{csv,parquet} [--db-url URL]

The default database is ``sqlite:///gentrade.db`` (overridable via the
``GENTRADE_DB_URL`` env var or ``--db-url`` on each command). Bars files
are loaded by extension — ``.parquet`` (the ingest output) or ``.csv``
(legacy). CSV ``open_ts`` may be ISO 8601 or epoch milliseconds.

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
from gentrade.ingest import (
    compute_indicators,
    fetch_ohlcv,
    fetch_yfinance,
    save_parquet,
)
from gentrade.ingest import (
    load_bars as _load_bars,
)
from gentrade.manifest import compute_data_hash, current_git_sha
from gentrade.persistence import (
    DEFAULT_DB_URL,
    init_db,
    list_runs,
)

log = logging.getLogger(__name__)


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


def cmd_ingest(args: argparse.Namespace) -> int:
    """Download OHLCV from ccxt or yfinance, add TA indicators, save Parquet."""
    if args.source == "ccxt" and not args.exchange:
        print("--exchange is required when --source=ccxt", file=sys.stderr)
        return 7
    if args.source == "yfinance":
        print(
            f"fetching {args.asset} from yfinance ({args.interval}) "
            f"from {args.since}..."
        )
        df = fetch_yfinance(
            symbol=args.asset,
            interval=args.interval,
            since=args.since,
            until=args.until,
        )
        source_label = "yfinance"
    else:
        print(
            f"fetching {args.asset} on {args.exchange} ({args.interval}) "
            f"from {args.since}..."
        )
        df = fetch_ohlcv(
            exchange_id=args.exchange,
            symbol=args.asset,
            interval=args.interval,
            since=args.since,
            until=args.until,
        )
        source_label = args.exchange

    if len(df) == 0:
        print("source returned no bars; check symbol/interval/since.", file=sys.stderr)
        return 6
    print(f"  fetched {len(df):,} bars; computing indicators...")
    if not args.no_indicators:
        df = compute_indicators(df)
    save_parquet(df, args.out)
    print(f"  wrote {args.out}")

    # Print a copy-paste-ready assets.json snippet so the operator can
    # register the file with the API.
    asset_id = (
        args.asset_id
        or f"{args.asset.replace('/', '').replace(':', '_')}-{args.interval}"
    )
    print()
    print("Add this to your assets.json:")
    print(
        json.dumps(
            {
                "asset": asset_id,
                "exchange": source_label,
                "interval": args.interval,
                "path": str(args.out),
            },
            indent=2,
        )
    )
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


def cmd_show(args: argparse.Namespace) -> int:
    """Print the headline metrics + per-generation curves for a persisted run."""
    from sqlalchemy.orm import Session

    from gentrade.persistence import RunRow

    engine = init_db(args.db_url)
    with Session(engine) as session:
        row = session.get(RunRow, args.run_id)
        if row is None:
            print(f"run {args.run_id!r} not found", file=sys.stderr)
            return 5
        print(f"id              : {row.id}")
        print(f"status          : {row.status}")
        print(f"seed            : {row.seed}")
        print(f"started_at      : {row.started_at}")
        print(f"finished_at     : {row.finished_at or '-'}")
        print(f"gen             : {row.current_generation}")
        print(f"chosen_strategy : {row.chosen_strategy_id or '-'}")
        print(
            f"overfitting_gap : "
            f"{row.overfitting_gap:.4f}" if row.overfitting_gap is not None else "-"
        )
        print(f"code_sha        : {row.code_sha or '-'}")
        print(f"data_hash       : {row.data_hash or '-'}")
        print()
        if row.generations:
            print("generation\ttrain_max\tval_max\ttrain_n_with_trades\tval_n_with_trades")
            for g in sorted(row.generations, key=lambda x: x.number):
                print(
                    f"{g.number}\t"
                    f"{g.train_max_fitness:+.4f}\t"
                    f"{g.validation_max_fitness:+.4f}\t"
                    f"{g.train_n_strategies_with_trades}\t"
                    f"{g.validation_n_strategies_with_trades}"
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

    ingest_p = sub.add_parser(
        "ingest",
        help="download OHLCV via ccxt or yfinance, compute indicators, save Parquet",
    )
    ingest_p.add_argument("--source", default="ccxt", choices=("ccxt", "yfinance"),
                          help="data source backend (default: ccxt)")
    ingest_p.add_argument("--exchange", default=None,
                          help="ccxt exchange id (binance, coinbase, kraken, …); "
                          "ignored when --source=yfinance")
    ingest_p.add_argument("--asset", required=True,
                          help="ccxt symbol like BTC/USDT, or yfinance ticker like SPY")
    ingest_p.add_argument("--interval", default="15m",
                          help="bar interval (1m / 5m / 15m / 30m / 1h / 4h / 1d)")
    ingest_p.add_argument("--since", required=True,
                          help="start date / time (ISO 8601 or epoch ms)")
    ingest_p.add_argument("--until", default=None,
                          help="end date / time (default: now)")
    ingest_p.add_argument("--out", required=True,
                          help="output Parquet path")
    ingest_p.add_argument("--asset-id", default=None,
                          help="asset name for the registry snippet (default: derived)")
    ingest_p.add_argument("--no-indicators", action="store_true",
                          help="skip TA indicator computation (just OHLCV)")
    ingest_p.set_defaults(func=cmd_ingest)

    run_p = sub.add_parser("run", help="start a new GA run")
    run_p.add_argument("--data", required=True, help="path to bars CSV or Parquet")
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

    show_p = sub.add_parser("show", help="print headline metrics + per-gen curves for one run")
    show_p.add_argument("run_id")
    show_p.add_argument("--db-url", default=DEFAULT_DB_URL)
    show_p.set_defaults(func=cmd_show)
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
