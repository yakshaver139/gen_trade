# Gen Trade

Algorithmic trading strategy generation for crypto assets using a genetic algorithm.

**Quickstart**: `uv sync && ./scripts/dev_server.sh` (terminal 1) →
`GENTRADE_API_KEY=<from-terminal-1> ./scripts/dev_ui.sh` (terminal 2) →
browse http://127.0.0.1:8501. No Binance credentials needed; the dev path
runs on a deterministic synthetic dataset.

## Background

This was a dissertation project undertaken as part of the [Software Engineering Master MSc program at the University of Oxford](https://www.cs.ox.ac.uk/softeng/programme/index.html), as such it should be considered an academic excercise only and is not recommended for actual trading! The project received a mark of 68%, which is a high merit (distinction = 70%).

## Project Aims

1. Explore if technical indicator-based trading is a viable option for algorithmic trading of Bitcoin

The research was cautiously optimistic that this object was satisfied. The strategies generated
did outperform the buy and hold strategy, and there were no clear indications that successful
strategies were constrained to favourable market conditions, (e.g. an up trend) hints at the
greater transferability of the TI-based approach over DNN methods. However, whilst the
results are suggestive of the generalisability of technical indicator-led strategies, the trading
scenarios followed in this study were artificial as they didn’t assess how much to invest on a
given trade but rather treated all entry signals as equal. In this sense the trading component
was a blunt instrument but they do serve to prove the concept that technical indicators can
not be ruled out. Finally, the alarming positivity of the results should be taken with a pinch of
salt and hints that there could be a calculation error that would require further investigation.

2. Determine if genetic algorithms offer a means to the discovery of novel trading strategies.

Whilst we have to be cautious about the suitability of TI-based strategies, the experimental
results show strong suitability for the application of genetic algorithms in this domain. We
can conclude that the GA was successful as the results showed an improvement in the fitness
function of the top strategies in every generation. Not only this but the lesser strategies also
showed signs of improvement compared to previous generations. Although 50 generations of
50 populations were run in the final genetic algorithm, this is too shallow to say with certainty
that the genetic process reached a global maximum. Running the algorithm across greater
generations would allow for a greater evolutionary process.

3. Produce a fully functioning end-to-end genetic algorithm software system for this specific context.

This goal was achieved as the final version of the code was containerised into a Docker ap-
plication and deployed and run in AWS via infrastructure as code tools. This means that
without significant uplift the service could be horizontally scaled to run on multiple machines
with greater parallelism. Additionally, the underlying Python codebase works as a stan-
dalone Python package and embeds many of the interesting implementation challenges such
as multiprocessing, modularity and extensibility.

## Usage

This project uses [`uv`](https://docs.astral.sh/uv/) and Python 3.11.

```sh
# install
uv sync

# smoke test (synthetic data, no network, no API key needed)
uv run python -m gentrade.smoke

# fetch real data — any ccxt exchange, no Binance creds required for
# public OHLCV. Output is a Parquet file with TA indicators precomputed.
uv run gentrade ingest --exchange binance --asset BTC/USDT --interval 15m \
                       --since 2022-01-01 --out ./data/BTCUSDT-15m.parquet
```

The legacy `python -m gentrade.binance_download` + `gentrade.ta_trends` pair
still works for backward compatibility but isn't the recommended path —
`gentrade ingest` covers more exchanges and writes a single self-contained
Parquet file in one step.

There are two ways to drive a run after that: the **CLI** for one-off
runs, or the **API + UI** for a hands-off, browse-able experience.

## CLI

The `gentrade` console script (defined in `pyproject.toml`) wraps the
GA orchestrator with persistent runs, resumable state, and a multi-asset
ingest pipeline.

```sh
# download OHLCV via ccxt + compute TA indicators + save as Parquet
gentrade ingest --exchange binance --asset BTC/USDT --interval 15m \
                --since 2022-01-01 --out ./data/BTCUSDT-15m.parquet
# (prints a ready-to-paste assets.json snippet)

# start a fresh run; persists incrementally to sqlite:///gentrade.db by default
gentrade run --data ./data/BTCUSDT-15m.parquet \
             --strategies ./signals/initial_population.json \
             --population-size 50 --generations 100 --seed 42

# list persisted runs (newest first)
gentrade list

# print headline metrics + per-generation curves for one run
gentrade show <run_id>

# resume an in-progress run after a crash / Ctrl-C
gentrade resume <run_id> --data ./data/BTCUSDT-15m.parquet
```

`gentrade ingest` works against any [ccxt](https://github.com/ccxt/ccxt)-supported
exchange — Binance, Coinbase, Kraken, Bybit, OKX, etc. Pass `--source yfinance`
for equities (e.g. `--asset SPY --interval 1d`); be aware of the market-hours
mismatch — the default `trade_window_bars=96` (one day at 15-min crypto) may
need to be tightened for daily equity bars. Both CSV and Parquet inputs are
accepted by `--data`; Parquet is much faster for the inner loop.

```sh
# crypto (ccxt)
gentrade ingest --exchange binance --asset BTC/USDT --interval 15m \
                --since 2022-01-01 --out ./data/BTCUSDT-15m.parquet

# equity (yfinance)
gentrade ingest --source yfinance --asset SPY --interval 1d \
                --since 2018-01-01 --out ./data/SPY-1d.parquet
```

### Cross-asset robustness check

Once a run has finished, you can re-run its chosen strategy against any
other registered asset to spot curve-fits:

```sh
curl -X POST http://127.0.0.1:8000/backtests/cross_asset \
  -H "X-API-Key: $GENTRADE_API_KEY" -H 'content-type: application/json' \
  -d '{"run_id":"<id>","strategy_id":"<id>","assets":["ETH-15m","SOL-15m"]}'
```

The Strategy detail UI page exposes the same comparison via a
multiselect + table — pick assets, click **Compare**, see one row per
asset with `n_trades / win_rate / expectancy / sharpe / max_dd`. A
strategy that wins on `BTC-15m` and craters on `ETH-15m` was a curve fit.

Behaviour worth knowing:

- Every generation is checkpointed (snapshot + RNG state) before the
  next one starts. A killed run resumes byte-equivalently — the
  determinism test in `tests/test_persistence.py` pins this.
- `gentrade run` refuses dirty git trees by default. Pass `--allow-dirty`
  to override; the run won't be byte-reproducible from its `code_sha`.
- 60/20/20 chronological train/validation/test split by default.
- All flags: `gentrade run --help` etc.

For the legacy `genetic.py` CLI (no friction model, no walk-forward
windowing), see `gentrade/genetic.py`. It still drives the smoke test
but its numbers should not be trusted.

## API server

The Phase 3 FastAPI service exposes runs, generations, and ad-hoc backtests over HTTP.

### Quickest path: zero-config dev server

```sh
./scripts/dev_server.sh
```

This synthesises a deterministic OHLCV+indicators frame in `.dev/`, registers it as the
`SAW-15m` asset, generates a fresh API key, and execs `uvicorn`. Everything (db,
per-run logs, bars, asset registry) lives in `.dev/` (git-ignored).

```sh
# different port
PORT=9000 ./scripts/dev_server.sh

# pin the API key across restarts
GENTRADE_API_KEY=mykey ./scripts/dev_server.sh
```

The script prints the API key + ready-to-run curl examples on startup. Swagger UI
is at `http://127.0.0.1:8000/docs` — click **Authorize** and paste the key to
exercise endpoints from the browser.

### Manual setup against real data

```sh
# 1. required: API key. Server fails closed (503) without one.
export GENTRADE_API_KEY=$(openssl rand -hex 32)

# 2. register at least one asset. Clients refer to it by name; the server
#    resolves to the CSV. Asset paths must resolve under GENTRADE_DATA_ROOT.
mkdir -p data
cat > data/assets.json <<'JSON'
[
  {"asset": "BTCUSDC-15m", "exchange": "binance", "interval": "15m",
   "path": "/abs/path/to/data/BTCUSDC_indicators.csv"}
]
JSON
export GENTRADE_ASSETS_PATH=$(pwd)/data/assets.json
export GENTRADE_DATA_ROOT=$(pwd)/data

# 3. (optional) DB and log dir
export GENTRADE_DB_URL=sqlite:///$(pwd)/gentrade.db
export GENTRADE_LOG_DIR=$(pwd)/runs

# 4. start the server
uv run uvicorn gentrade.api.app:default_app --factory --host 127.0.0.1 --port 8000
```

### Hitting the API

```sh
# health (unauthenticated)
curl http://127.0.0.1:8000/healthz

# list registered assets
curl -H "X-API-Key: $GENTRADE_API_KEY" http://127.0.0.1:8000/assets

# kick off a run (returns 202 + run_id; runs in a background thread)
curl -X POST http://127.0.0.1:8000/runs \
  -H "X-API-Key: $GENTRADE_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"asset":"SAW-15m","population_size":10,"generations":5,"seed":42}'

# poll for status
curl -H "X-API-Key: $GENTRADE_API_KEY" http://127.0.0.1:8000/runs/<run_id>

# rerun a saved strategy on its original windows
curl -X POST http://127.0.0.1:8000/backtests \
  -H "X-API-Key: $GENTRADE_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"run_id":"<run_id>","strategy_id":"<strategy_id>"}'
```

Tips:

- **Process restarts kill in-flight runs.** Their state is checkpointed every generation,
  so you can recover them with `gentrade resume <run_id>` from the CLI.
- **`--reload` is not safe** with the daemon-thread job runner — it kills running jobs
  without finalising. Use it only when no runs are in flight.
- **Per-run logs** stream to `<GENTRADE_LOG_DIR>/<run_id>/run.log` so you can
  `tail -f` without grepping uvicorn stdout.
- **Adding endpoints?** Regenerate `openapi.json` with
  `uv run python -m gentrade.api.export_openapi > openapi.json` and review the diff —
  the snapshot is asserted against by `tests/test_api.py`.

## UI

A Streamlit frontend lives in `gentrade/ui/`. It talks to the API over HTTP — start
the API (`./scripts/dev_server.sh`) first, then the UI:

```sh
# Terminal 1 — API. Note the printed api key.
./scripts/dev_server.sh

# Terminal 2 — UI, with the API key from terminal 1.
GENTRADE_API_KEY=<paste_from_terminal_1> ./scripts/dev_ui.sh
# UI on http://127.0.0.1:8501
```

Pages (left sidebar):

- **Runs** — list of persisted runs, status + headline metrics
- **New Run** — form that POSTs `/runs` (asset, population size, generations, seed,
  selection pressure, costs)
- **Run detail** — fitness curves (train max + median, validation max), terminal
  metrics, manifest. Paste a run id at the top
- **Strategy detail** — chromosome + parsed pandas query, equity curve, drawdown,
  per-trade scatter. Both run id and strategy id required (the Run detail page
  prints them ready to copy)

Auto-refresh is intentionally not wired in v1 — every page has a **Refresh** button.
For a longer-running run, click Refresh every few seconds; the per-generation curve
is the live progress signal.

## Docker

`docker build -t gen-trade .`
`docker run gen-trade`

Note, the `ecr_push.sh` script can be used for pushing the docker image to ecr to start to run as an ecs service.

## Development

```sh
uv run pytest
uv run ruff check gentrade tests
uv run ruff format gentrade tests
```

## What this does and doesn't claim

The original 2022 dissertation reported that GA-evolved strategies outperformed buy-and-hold
on Bitcoin, with the candid caveat that "the alarming positivity of the results... hints that
there could be a calculation error." The Phase 1 work is the audit of that claim.

**What the current code does honestly:**

- Backtests trades with realistic friction: per-side taker fees (10 bps default, Binance spot)
  and slippage (1 bp default; bump for thin alts). See `gentrade/backtest.py`.
- Enforces a single open position per strategy. Signals that fire while a trade is open are
  ignored — the original code evaluated every signal independently, which double-counted
  exposure and inflated fitness on clustered signals.
- Splits market history into chronologically ordered train / validation / test windows.
  Selection only ever sees train. Validation is re-evaluated every generation but never feeds
  selection — it's there to detect when in-sample fitness diverges from out-of-sample.
  Test is touched exactly once, at the end, to produce the final report.
- Produces a `BacktestReport` with per-window metrics (Sharpe, Sortino, Calmar, max drawdown,
  profit factor, win rate, expectancy, avg trade duration, n_trades) plus two test-window
  baselines (buy-and-hold; random-entry with the same exposure) so the reader can judge
  whether the strategy is doing meaningful work.
- Tournament-of-3 selection by default (best-vs-median sampling ratio ≈ 4.4× for N=10),
  replacing the original near-uniform `n + 1/(i+1)` formula that effectively randomised parent
  selection.
- Pins a regression suite (`tests/regression/`): a fixed seed + synthetic dataset produces a
  fully-pinned report. Any change to selection / metrics / backtest / walk-forward must explain
  its diff against the pinned numbers.
- Multi-asset ingest via [ccxt](https://github.com/ccxt/ccxt): any of the major spot exchanges
  (Binance / Coinbase / Kraken / Bybit / OKX / …) can populate a Parquet bars file consumed by
  the same GA / API / UI without code changes. `BTC/USDT`, `ETH/USDT`, `SOL/USDT` etc. are all
  one `gentrade ingest` away.

**What the current code does NOT yet claim:**

- The GA loop with the new engine has not yet been run on real Binance data end-to-end, so
  the dissertation's outperformance claim has not yet been re-tested under realistic costs and
  walk-forward windowing. Until it has, treat the original conclusion as unverified.
- Indicator pre-computation has not been audited for look-ahead. Most `ta` library indicators
  are causal (rolling-window, no centred mean), but the audit has not been completed and any
  centred-window or full-series-normalised indicator would silently leak the future across the
  train/validation boundary.
- Strategy selection on the population is currently single-objective (per-trade expectancy with
  a min-trades floor). The plan calls for multi-objective fitness (Sharpe + drawdown penalty +
  trade count); not yet implemented.
- No statistical-significance test (bootstrap p-values vs. random-entry / buy-and-hold) is
  produced yet — the random-entry baseline ships, but a single seed isn't a confidence interval.
- The legacy `genetic.main` and `run_strategy` paths are still present and still drive the smoke
  test. They use the old no-friction, no-overlap-rule evaluator. Don't trust their numbers.
- **Cross-asset robustness check is wired up but not yet automatic.** The endpoint
  (`POST /backtests/cross_asset`) and the Strategy detail UI section let you re-run a
  strategy on other registered assets and read the per-asset metrics; making it part
  of the headline `BacktestReport` so you can't ship a curve-fit by accident is the
  next step.
- **yfinance equity support has the obvious caveats.** Survivorship bias: yfinance
  only returns currently-listed tickers, so any backtest that uses it overestimates
  the universe's returns. Market hours: equities trade ~6.5h/day on weekdays, so
  the default `trade_window_bars` (calibrated for 24/7 crypto) may need tightening.
  Both are documented at the call site in `gentrade/ingest.py::fetch_yfinance`.
- Per-asset signal thresholds are still global (a strategy that uses
  `momentum_rsi >= 70` interprets that the same way for BTC and SOL); the GA already
  mutates absolute thresholds, but cross-asset retraining isn't automatic.
- No paper trading. No live trading. No risk module.

If you point money at this code in its current state, that's on you.

See [`PLAN.md`](PLAN.md) for the phased plan. Phases 0 (revival), 1 (modelling rigor),
2 (persistence + job model), 3 (FastAPI + auth + security review), 4 (Streamlit UI),
and 5 (multi-asset ingest via ccxt + yfinance + cross-asset robustness check) are
complete. Phase 6 (paper + live trading, gated) is not yet started.