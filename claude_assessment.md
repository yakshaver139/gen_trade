  Is it any good?

  Genuinely — yes, for a 2022 dissertation. The shape is sound:

  - The Allium spec you drafted maps cleanly to the code, which means the domain you reverse-engineered
   is coherent: Run → Generation → Strategy → Backtest → FitnessReport, with selection / PPX crossover
  / mutation / elitism. That's a real GA, not a toy.
  - Separation of concerns is mostly there: generate_strategy (chromosome), load_strategy (genotype →
  phenotype string), run_strategy (backtest), fitness_functions (selection pressure), genetic (loop).
  Three pluggable fitness functions is a nice touch.
  - The PPX crossover, rank-weighted selection, elitism=2, mutation perturbing only absolute
  thresholds, the entry_cutoff_percent noise filter — these are reasonable choices, not academic mush.
  - The DataFrame adapter (df_adapter.dfa) so you can swap pandas/dask at runtime is a clever pragmatic
   hack for the OOM problem you hit.
  - IaC + Docker + ECR + EC2 user-data bootstrapping is more end-to-end than most dissertations get.

  68% feels about right — the idea is solid; the execution has bugs and rough edges that I'd expect a
  marker to dock you for.

  Bugs / correctness concerns (worth knowing about even if you don't fix)

  1. INDICATORS is a generator, not a list (generate_blank_signals.py:18). It's imported by
  genetic.py:14 and used in add_previous_window_values. After it's consumed once at module load (or by
  anything else that touches it), subsequent uses get nothing. Likely silently truncating which
  _previous columns get added.
  2. evaluation.py:11 — from word_cloud import WordCloud. The library is wordcloud, single word.
  ImportError.
  3. evaluation.py:14 — _dfs = {"gen_{ix}": df for ix, df in enumerate(dfs)} — missing the f prefix, so
   every key collapses to the literal string "gen_{ix}". The dict has one entry.
  4. Mutable default arg — select_parents(..., parents: Dict = {}) (genetic.py:188). Persists across
  all calls in the process; unintended sharing.
  5. Rank weights are nearly uniform — n_items + 1 / (i + 1) with operator precedence n_items +
  (1/(i+1)). With pop=10, top-ranked weight is 11, worst is 10.1. So selection pressure from
  "rank-weighted" sampling is almost flat. Probably wanted (n_items + 1) / (i + 1). The Allium spec
  encodes the implemented formula, but it's almost certainly a precedence bug.
  6. Two run_strategy modules — run_strategy_async.py is half-finished: assess_strategy_window ends
  with bare res instead of return res (line 122), so its results are all None. The live path is
  run_strategy.py, which works.
  7. get_trade_result mixes -1 sentinel with timestamps in comparisons. Works because open_ts is an int
   (ms), but fragile.
  8. env.py initialises boto3 at import — S3_RESOURCE = boto3.resource("s3", ...) runs every time
  anything imports env. Hard requirement on AWS creds being resolvable, even for tests.
  9. fitness_simple_profit calls fitness_function_ha_and_moon first (line 27 of fitness_functions.py) —
   meaning the simple-profit fitness depends on log-return scoring as a side-effect to populate
  metadata. The two are entangled in a way the spec says they aren't.
  10. cumulative_returns(total) = total / BUY_AMOUNT in evaluation.py divides by 1000, but
  calculate_simple_profit is operating on a 10000 starting balance. Returns will be 10× off.
  11. Tests don't match the implementation. test_generate asserts strategy == ['ble', 'AND', 'foo',
  'AND', 'bar'] — but generate returns a dict(id=..., indicators=..., conjunctions=...). They've
  drifted.
  12. mock in deps but unittest.mock is stdlib — only one test still uses it.

  None of these are catastrophic. (5) is the one that probably mattered most for the dissertation
  results — your selection was effectively random, which complicates the "GA improved across
  generations" claim. Worth a footnote if you ever revisit it.

  What it takes to revive it

  The fastest "make it run again" path:

  1. Bump Python. pyproject.toml says python = "^3.9", your machine is on 3.13. Most of the locked deps
   (pandas-ta 0.3.14-beta, dask 2022.8.0, ta 0.10.1, wordcloud) won't build on 3.13 — drop the lock,
  pin Python to ~=3.11, regenerate. Or move the project to uv and modern pins.
  2. Replace pandas-ta. It's been abandoned and is broken on modern numpy/pandas. ta (which you also
  use) still works; or switch to pandas_ta_classic/finta. This is the single biggest dep risk.
  3. Lazy boto3. Move the S3_RESOURCE = boto3.resource(...) out of env.py import-time into a function.
  Otherwise pytest and any local invocation needs AWS creds.
  4. Fix INDICATORS — change the generator to a list(...) or rederive from the loaded CSV at call time.
  5. Re-download data. BTCUSDC_indicators.csv isn't in the repo (gitignored). You'll need a Binance API
   key + run binance_download.py then ta_trends.py. Note python-binance ^1.0.16 is from 2022; the v3
  API changed — likely needs a bump and possibly code changes.
  6. Pick one run_strategy. Delete run_strategy_async.py (it's broken anyway).
  7. Remove the IaC tier or quarantine it. terraform.tfstate is checked in at the repo root and in iac/
   — that's a small leak (not creds, but resource IDs). Delete from git history if you ever push
  5. Re-download data. BTCUSDC_indicators.csv isn't in the repo (gitignored). You'll need a Binance API key + run binance_download.py then
  ta_trends.py. Note python-binance ^1.0.16 is from 2022; the v3 API changed — likely needs a bump and possibly code changes.
  6. Pick one run_strategy. Delete run_strategy_async.py (it's broken anyway).
  7. Remove the IaC tier or quarantine it. terraform.tfstate is checked in at the repo root and in iac/ — that's a small leak (not creds, but
  resource IDs). Delete from git history if you ever push public, and add *.tfstate* to .gitignore. Honestly, the IaC tier is more "demo I'm
  cloud-savvy" than "I need to run this in 2026" — for a personal revival, drop it.
  8. Fix or delete the test suite. Three test files; the strategy ones are stale. Easier to rewrite a small smoke test of cross_over_ppx, mutate,
  load_from_object_parenthesised than salvage these.

  Realistically, ~half a day to get a population of 10 running locally once you have a fresh CSV.

  Cleanup opportunities (independent of revival)

  - Delete strategy.py — superseded class-based shape, your spec marks it excluded, nothing imports it.
  - Delete cross_over_pmx — defined but unused; spec confirms.
  - Delete run_strategy_async.py — broken and unused.
  - Delete is_profitable / is_loss_making in genetic.py — defined, unused.
  - Delete get_window_performance — defined in two files, unused.
  - Strip the giant comment block at top of genetic.py (run-time logs from August 2022). It's noise.
  - Remove checked-in result files — best_strategy.json, ec2_strategies.json (77KB), input_strategies.json, terraform.tfstate files. These are
  outputs, not source.
  - Consolidate env.py constants with the Allium config block — currently duplicated (TARGET=0.015 vs target_profit: 0.015, MAX_GENERATIONS=100
  etc.).
  - logger.py calls logging.basicConfig inside get_logger — every call reconfigures. Move basicConfig once at startup.
  - mock dep — drop, use unittest.mock.
  - memory-profiler, dill, dask, wordcloud, matplotlib, plotly, ipdb, ipython — all heavy and only used in evaluation.py (which is post-hoc
  8. Fix or delete the test suite. Three test files; the strategy ones are stale. Easier to rewrite a small smoke test of cross_over_ppx, mutate,
  load_from_object_parenthesised than salvage these.

  Realistically, ~half a day to get a population of 10 running locally once you have a fresh CSV.

  Cleanup opportunities (independent of revival)

  - Delete strategy.py — superseded class-based shape, your spec marks it excluded, nothing imports it.
  - Delete cross_over_pmx — defined but unused; spec confirms.
  - Delete run_strategy_async.py — broken and unused.
  - Delete is_profitable / is_loss_making in genetic.py — defined, unused.
  - Delete get_window_performance — defined in two files, unused.
  - Strip the giant comment block at top of genetic.py (run-time logs from August 2022). It's noise.
  - Remove checked-in result files — best_strategy.json, ec2_strategies.json (77KB), input_strategies.json, terraform.tfstate files. These are
  outputs, not source.
  - Consolidate env.py constants with the Allium config block — currently duplicated (TARGET=0.015 vs target_profit: 0.015, MAX_GENERATIONS=100
  etc.).
  - logger.py calls logging.basicConfig inside get_logger — every call reconfigures. Move basicConfig once at startup.
  - mock dep — drop, use unittest.mock.
  - memory-profiler, dill, dask, wordcloud, matplotlib, plotly, ipdb, ipython — all heavy and only used in evaluation.py (which is post-hoc

  Realistically, ~half a day to get a population of 10 running locally once you have a fresh CSV.

  Cleanup opportunities (independent of revival)

  - Delete strategy.py — superseded class-based shape, your spec marks it excluded, nothing imports it.
  - Delete cross_over_pmx — defined but unused; spec confirms.
  - Delete run_strategy_async.py — broken and unused.
  - Delete is_profitable / is_loss_making in genetic.py — defined, unused.
  - Delete get_window_performance — defined in two files, unused.
  - Strip the giant comment block at top of genetic.py (run-time logs from August 2022). It's noise.
  - Remove checked-in result files — best_strategy.json, ec2_strategies.json (77KB), input_strategies.json, terraform.tfstate files. These are
  outputs, not source.
  - Consolidate env.py constants with the Allium config block — currently duplicated (TARGET=0.015 vs target_profit: 0.015, MAX_GENERATIONS=100
  etc.).
  - logger.py calls logging.basicConfig inside get_logger — every call reconfigures. Move basicConfig once at startup.
  - mock dep — drop, use unittest.mock.
  - memory-profiler, dill, dask, wordcloud, matplotlib, plotly, ipdb, ipython — all heavy and only used in evaluation.py (which is post-hoc

  Cleanup opportunities (independent of revival)

  - Delete strategy.py — superseded class-based shape, your spec marks it excluded, nothing imports it.
  - Delete cross_over_pmx — defined but unused; spec confirms.
  - Delete run_strategy_async.py — broken and unused.
  - Delete is_profitable / is_loss_making in genetic.py — defined, unused.
  - Delete get_window_performance — defined in two files, unused.
  - Strip the giant comment block at top of genetic.py (run-time logs from August 2022). It's noise.
  - Remove checked-in result files — best_strategy.json, ec2_strategies.json (77KB), input_strategies.json, terraform.tfstate files. These are
  outputs, not source.
  - Consolidate env.py constants with the Allium config block — currently duplicated (TARGET=0.015 vs target_profit: 0.015, MAX_GENERATIONS=100
  etc.).
  - logger.py calls logging.basicConfig inside get_logger — every call reconfigures. Move basicConfig once at startup.
  - mock dep — drop, use unittest.mock.
  - memory-profiler, dill, dask, wordcloud, matplotlib, plotly, ipdb, ipython — all heavy and only used in evaluation.py (which is post-hoc
  - Delete cross_over_pmx — defined but unused; spec confirms.
  - Delete run_strategy_async.py — broken and unused.
  - Delete is_profitable / is_loss_making in genetic.py — defined, unused.
  - Delete get_window_performance — defined in two files, unused.
  - Strip the giant comment block at top of genetic.py (run-time logs from August 2022). It's noise.
  - Remove checked-in result files — best_strategy.json, ec2_strategies.json (77KB), input_strategies.json, terraform.tfstate files. These are
  outputs, not source.
  - Consolidate env.py constants with the Allium config block — currently duplicated (TARGET=0.015 vs target_profit: 0.015, MAX_GENERATIONS=100
  etc.).
  - logger.py calls logging.basicConfig inside get_logger — every call reconfigures. Move basicConfig once at startup.
  - mock dep — drop, use unittest.mock.
  - memory-profiler, dill, dask, wordcloud, matplotlib, plotly, ipdb, ipython — all heavy and only used in evaluation.py (which is post-hoc
  - Remove checked-in result files — best_strategy.json, ec2_strategies.json (77KB), input_strategies.json, terraform.tfstate files. These are
  outputs, not source.
  - Consolidate env.py constants with the Allium config block — currently duplicated (TARGET=0.015 vs target_profit: 0.015, MAX_GENERATIONS=100
  etc.).
  - logger.py calls logging.basicConfig inside get_logger — every call reconfigures. Move basicConfig once at startup.
  - mock dep — drop, use unittest.mock.
  - memory-profiler, dill, dask, wordcloud, matplotlib, plotly, ipdb, ipython — all heavy and only used in evaluation.py (which is post-hoc
  etc.).
  - logger.py calls logging.basicConfig inside get_logger — every call reconfigures. Move basicConfig once at startup.
  - mock dep — drop, use unittest.mock.
  - memory-profiler, dill, dask, wordcloud, matplotlib, plotly, ipdb, ipython — all heavy and only used in evaluation.py (which is post-hoc
  - memory-profiler, dill, dask, wordcloud, matplotlib, plotly, ipdb, ipython — all heavy and only used in evaluation.py (which is post-hoc
  analysis). Move them to an optional analysis extra.

  Bottom line

  Your Allium spec is a more honest description of the system than the code is — the spec captures the design as intended, while the code has the
  usual academic-codebase scars: dead branches, duplicated constants, an aspirational async path that was never finished, tests that drifted. None
   of this is unsalvageable.

  If the goal is "I want to play with this again," budget half a day: bump Python, replace pandas-ta, lazy-load boto, fix the INDICATORS
  generator, regenerate the CSV. The GA core itself doesn't need to change.

  If the goal is "show this off as a portfolio piece in 2026," I'd also fix the rank-weight precedence
  Realistically, ~half a day to get a population of 10 running locally once you have a fresh CSV.

  Cleanup opportunities (independent of revival)

  - Delete strategy.py — superseded class-based shape, your spec marks it excluded, nothing imports it.
  - Delete cross_over_pmx — defined but unused; spec confirms.
  - Delete run_strategy_async.py — broken and unused.
  - Delete is_profitable / is_loss_making in genetic.py — defined, unused.
  - Delete get_window_performance — defined in two files, unused.
  - Strip the giant comment block at top of genetic.py (run-time logs from August 2022). It's noise.
  - Remove checked-in result files — best_strategy.json, ec2_strategies.json (77KB),
  input_strategies.json, terraform.tfstate files. These are outputs, not source.
  - Consolidate env.py constants with the Allium config block — currently duplicated (TARGET=0.015 vs
  target_profit: 0.015, MAX_GENERATIONS=100 etc.).
  - logger.py calls logging.basicConfig inside get_logger — every call reconfigures. Move basicConfig
  once at startup.
  - mock dep — drop, use unittest.mock.
  - memory-profiler, dill, dask, wordcloud, matplotlib, plotly, ipdb, ipython — all heavy and only used
   in evaluation.py (which is post-hoc analysis). Move them to an optional analysis extra.

  Bottom line

  Your Allium spec is a more honest description of the system than the code is — the spec captures the
  design as intended, while the code has the usual academic-codebase scars: dead branches, duplicated
  constants, an aspirational async path that was never finished, tests that drifted. None of this is

  Your Allium spec is a more honest description of the system than the code is — the spec captures the design as intended, while the code has the
  usual academic-codebase scars: dead branches, duplicated constants, an aspirational async path that was never finished, tests that drifted. None
   of this is unsalvageable.

  If the goal is "I want to play with this again," budget half a day: bump Python, replace pandas-ta, lazy-load boto, fix the INDICATORS
  generator, regenerate the CSV. The GA core itself doesn't need to change.

  If the goal is "show this off as a portfolio piece in 2026," I'd also fix the rank-weight precedence bug, kill the dead code, and write a
  50-line pytest smoke that actually runs end-to-end on a tiny synthetic dataframe — that's the missing artefact.