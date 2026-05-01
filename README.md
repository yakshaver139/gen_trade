# Gen Trade

Algorithmic trading strategy generation for crypto assets using a genetic algorithm.

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

# Usage

This project uses [`uv`](https://docs.astral.sh/uv/) and Python 3.11.

```sh
# 1. install
uv sync

# 2. run the smoke test (no Binance, no S3 — synthetic data)
uv run python -m gentrade.smoke

# 3. download real data (requires BINANCE_API and BINANCE_SECRET env vars)
uv run python -m gentrade.binance_download

# 4. preprocess (adds trend_direction column)
uv run python -m gentrade.ta_trends

# 5. run the genetic algorithm
uv run python -m gentrade.genetic --write_local=True --generations=10 --population_size=100 --fitness_function=p
```

To run on previously-generated strategies:

```sh
uv run python -m gentrade.genetic --generations=1 --strategies_path=best_strategy.json --output_path=results.csv
```

## CLI

Command line interface menu for
genetic.py

```
poetry run python genetic.py --help
usage: Main genetic algorithm. [-h] [--population_size POPULATION_SIZE]
[--max_indicators MAX_INDICATORS]
[--max_same_class MAX_SAME_CLASS]
[--write_s3 WRITE_S3]
[--write_local WRITE_LOCAL]
[--s3_bucket S3_BUCKET]
[--generations GENERATIONS]
[--serial_debug SERIAL_DEBUG]
[--strategies_path STRATEGIES_PATH]
[--fitness_function FITNESS_FUNCTION]
[--output_path OUTPUT_PATH]
[--incremental_saves INCREMENTAL_SAVES]
options:
-h, --help show this help message and exit
--population_size POPULATION_SIZE
number of strategies to generate
--max_indicators MAX_INDICATORS
max number of indicators in a strategy
--max_same_class MAX_SAME_CLASS
max number of same class indicators in a strategy
--write_s3 WRITE_S3 exports data to s3.
--write_local WRITE_LOCAL
exports data to local FS.
--s3_bucket S3_BUCKET
bucket name to which data is written.
--generations GENERATIONS
N generations to run.
--serial_debug SERIAL_DEBUG
run without async Future - for debugging
--strategies_path STRATEGIES_PATH
load strategies from this path rather than generating on the fly
--fitness_function FITNESS_FUNCTION
fitness function use (h=ha_and_moon, o=original, p=profit)
--output_path OUTPUT_PATH
path to save outputs
--incremental_saves INCREMENTAL_SAVES
when true, saves the output for every 10 strategies tested
```

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
- No paper trading. No live trading. No risk module.

If you point money at this code in its current state, that's on you.

See [`PLAN.md`](PLAN.md) for the phased plan. Phases 0 (revival) and 1 (modelling rigor) are
complete; Phase 2 (persistence + job model) is in progress.