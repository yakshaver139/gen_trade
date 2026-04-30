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

## Status

This is a 2022 dissertation codebase under active revival. See [`PLAN.md`](PLAN.md) for the
phased plan to take this to production. Phase 0 (revival) is complete; Phase 1 (modelling
rigor) is the priority next.