# Gen Trade — Revival & Production Plan

A phased build plan for taking the dissertation codebase from "ran once on EC2 in 2022" to "I trust it enough to point money at it." Written to be executed *agentically* — the work is decomposed so that subagents (Allium, TDD, security review, etc.) can do most of the heavy lifting under your direction.

---

## Guiding principles

1. **The Allium spec is the source of truth for domain.** Code drifts; the spec is what we keep honest. Every phase ends with `allium:weed` to surface divergence and `allium:tend` to grow the spec for new behaviour.
2. **Walk-forward, never look-ahead.** The single biggest risk in this project is *backtest overfitting* — a GA is an overfitting machine by construction. Every modelling decision is judged against out-of-sample performance, not in-sample fitness.
3. **TDD for anything that matters.** Pricing, risk, P&L, signal evaluation — write the tests first using `tdd-test-writer`, implement with `tdd-implementation-writer`. Skip TDD for plumbing (scripts, CLI, CI), keep it for the things where a silent bug = wrong number on a chart = wrong trading decision.
4. **Paper-trade before money.** Phase 6 is gated behind a 30-day forward-paper-trading run with documented risk limits.
5. **Each phase ships something runnable.** No "we'll wire it up at the end."

---

## Agent roster (what we'll lean on, and when)

| Agent | Used for |
|---|---|
| `allium:tend` | Growing the spec when new domain behaviour is introduced |
| `allium:weed` | Auditing spec ↔ code divergence at end of each phase |
| `Plan` | Designing non-trivial implementation strategies before coding |
| `tdd-test-writer` / `tdd-implementation-writer` | Test-first work on modelling, risk, and trading logic |
| `Explore` | Codebase research that would otherwise burn main context |
| `secure-code-reviewer` | Mandatory pass on the API, auth, and any path that touches exchange credentials |
| `lint-fixer` | Final pre-PR pass once a phase is done |
| Background agents (`run_in_background: true`) | Long-running backtests, GA runs, dependency upgrades |
| `schedule` | Recurring jobs once we're in production (nightly backtests, drift checks) |

We'll also run a `claude.md` at the project root after Phase 0 to capture conventions (Python version, where data lives, how to run a smoke test) so future sessions don't re-derive it.

---

## Phase 0 — Revive (target: 1 session, ~half a day)

**Goal:** `python -m gentrade.smoke` runs a 1-generation, pop-of-3 GA on a tiny synthetic dataframe and prints a ranking. Nothing else.

### Tasks

- [ ] Migrate to `uv` (or modern Poetry); pin Python to `~=3.11` (3.13 is too aggressive given `pandas-ta` constraints).
- [ ] Replace `pandas-ta` (abandoned) with `ta` (already in deps) + a small custom shim for any indicators only `pandas-ta` provided. Audit which indicators we actually use against `signals/*.json`.
- [ ] Lazy-load boto3 — move `S3_RESOURCE = boto3.resource(...)` out of `env.py` import-time into a `get_s3()` helper. Currently any import of `env` requires AWS creds.
- [ ] Fix the `INDICATORS` generator bug (`generate_blank_signals.py:18`) — make it a list, otherwise `_previous` columns are silently dropped.
- [ ] Fix `evaluation.py:11` (`word_cloud` → `wordcloud`) and `evaluation.py:14` (missing `f` prefix in dict comprehension).
- [ ] Delete `run_strategy_async.py` (broken — `assess_strategy_window` returns nothing), `strategy.py` (superseded), `cross_over_pmx` (unused).
- [ ] Delete checked-in artefacts: `best_strategy.json`, `ec2_strategies.json`, `input_strategies.json`, `terraform.tfstate*` at root. Add to `.gitignore`.
- [ ] Replace mutable default arg in `select_parents`.
- [ ] Reorganise: move all source under `gentrade/` (proper package). Modules-at-root makes it impossible to ship as a library or install for the API.
- [ ] Add a `tests/test_smoke.py` that runs the GA end-to-end on a tiny synthetic frame (no Binance, no S3).
- [ ] CI: GitHub Actions workflow running `pytest -q` and `ruff check`.

### Out of scope this phase

Behaviour changes. We're getting it green, not improving it.

### Agentic mechanics

- One pass with `allium:weed` against the existing spec to confirm what we *think* exists actually exists — already done partially in conversation; formalise as a worktree run.
- `lint-fixer` at the end before PR.

### Done when

- `pytest -q` passes locally and in CI.
- `python -m gentrade.smoke` produces a deterministic ranking on a fixed-seed synthetic frame.
- A README "quickstart" runs cleanly on a fresh clone with `uv sync`.

---

## Phase 1 — Modelling rigor (target: 2–3 sessions)

**Goal:** Make the GA's outputs *trustworthy*. Right now we have three fitness functions, a flat selection pressure, and no separation between in-sample and out-of-sample. Fix that before we build anything else on top.

### Hard problems to solve (in priority order)

1. **Train/validation/test split.** Walk-forward windows: e.g. evolve on `[2018–2020]`, validate on `[2021]`, never touch `[2022–]` until the end. Currently the GA fits the entire dataset and reports fitness on the same data.
2. **Selection-pressure bug.** `apply_ranking` weights collapse to nearly uniform (`n_items + 1/(i+1)`). Decide what we *want* — fitness-proportional, rank-linear, or tournament — and implement it deliberately. Add a config knob.
3. **Realistic trade simulation.** Currently entries are evaluated independently with no notion of an open position. Add: position state, no overlapping trades on the same strategy, transaction costs (Binance taker is ~10bps), realistic slippage (≥1bp on majors, more on alts).
4. **Position sizing.** Right now every trade is "all in" on a notional `BUY_AMOUNT=1000`. Add fractional Kelly or a fixed-fractional sizing rule as a config; backtest both.
5. **Multi-objective fitness.** Single-objective profit is gameable by lottery-ticket strategies. Combine: Sharpe + max drawdown penalty + minimum trade count. Or use NSGA-II for proper multi-objective GA.
6. **Statistical significance.** Bootstrap or block-bootstrap the trade returns; report p-values vs. buy-and-hold and vs. random-entry baselines.

### Tasks

- [ ] **Plan agent first.** Hand `Plan` the question: "design a walk-forward backtesting scheme for a GA where each generation needs to be re-evaluated against a held-out window." Get a written plan before touching code.
- [ ] Refactor `Run` in the spec to carry a `train_window`, `validation_window`, `test_window`. `allium:tend`.
- [ ] TDD the new `Backtest` module: realistic position state, no-overlap rule, fees, slippage. Test against hand-computed expected results on a 20-bar synthetic frame.
- [ ] TDD the rank-weight rewrite — write tests that assert "best strategy is sampled ≥3× more often than median" before implementing.
- [ ] TDD a `Metrics` module: Sharpe, Sortino, Calmar, max drawdown, profit factor, win rate, expectancy, average trade duration. These are the columns we'll need everywhere downstream.
- [ ] Add a baseline comparator: random-entry strategies with the same trade count as the candidate. If a strategy can't beat random with the same exposure, it isn't a strategy.
- [ ] Build a "modelling regression suite": a fixed seed + dataset combination that produces a fixed report. Any future change must explain the diff. (The spec ships with a `pytest --tb=line tests/regression` entrypoint.)

### Risks to flag

- **Overfitting.** If the GA improves out-of-sample test fitness as we scale generations and population, we're learning. If it improves in-sample but not out-of-sample, we're memorising. Watch this gap religiously.
- **The dissertation result might evaporate.** With proper costs, slippage, and walk-forward, the "outperforms buy-and-hold" claim may not hold. That's fine — better to know.

### Agentic mechanics

- This phase wants `xruns-data-scientist`-style critique. We don't have that exact agent for trading, but `secure-code-reviewer` + a manually scoped `Plan` invocation ("audit this fitness function for ways a GA could exploit it") is a reasonable substitute.
- `tdd-test-writer` first on every modelling change.
- Long backtest sweeps go to background agents; don't block the main session on them.

### Done when

- A run produces a `BacktestReport` with in-sample, validation, and test metrics — and they don't all agree.
- The regression suite is green and pinned in CI.
- An honest README section on "what this does and doesn't claim."

---

## Phase 2 — Persistence & job model (target: 1–2 sessions)

**Goal:** GA runs become first-class objects — stored, queryable, resumable. This is the substrate the API needs.

### Tasks

- [ ] Pick a database. SQLite for local; Postgres optional via env var. Schema mirrors the Allium entities: `runs`, `generations`, `strategies`, `backtests`, `fitness_reports`, `trades`. Use SQLAlchemy 2.x with typed models.
- [ ] Write a migration tool (Alembic).
- [ ] Job queue: start with a simple `BackgroundTasks`-style executor backed by a `runs.status` column. Don't reach for Celery/Redis until we have multiple machines.
- [ ] Make GA runs *resumable*: each generation persisted at completion; restart picks up at last completed generation. (Important — Phase 1 backtests may now take hours.)
- [ ] CLI: `gentrade run --asset BTCUSDC --population 50 --generations 100 --resume`.
- [ ] Optional: structured logs to stdout + a per-run log file in `runs/<run_id>/run.log`.

### Done when

- Killing a run mid-way and restarting it produces identical results to a clean run (same seed).
- `sqlite3 gentrade.db "select * from runs"` shows history.
- `allium:weed` reports clean.

---

## Phase 3 — API (target: 1 session)

**Goal:** A small FastAPI service that exposes runs, strategies, and backtests over HTTP, so the UI in Phase 4 has something to talk to.

### Tasks

- [ ] FastAPI app under `gentrade/api/`. Endpoints:
  - `POST /runs` — start a new run (returns 202, run_id)
  - `GET /runs` / `GET /runs/{id}` — list / detail with progress
  - `GET /runs/{id}/generations/{n}` — generation detail with ranked strategies
  - `GET /strategies/{id}` — strategy detail + parsed expression
  - `POST /backtests` — backtest a saved strategy on a different asset/window
  - `GET /assets` — supported asset list (Phase 5 fills this in)
  - `GET /healthz`
- [ ] Auth: API-key header at minimum. Don't deploy this to the public internet without it.
- [ ] OpenAPI spec auto-generated; commit a snapshot for diff review.
- [ ] **`secure-code-reviewer` mandatory pass** before this phase merges. Things to specifically audit: query parameter injection into pandas `.query()` (the strategy DSL is *literally* a pandas query string — this is a SQLi-shaped problem), API-key storage, any path that reads files based on user input.

### Risks to flag

- The strategy DSL is a `pandas.DataFrame.query()` string built from user-controlled signal definitions. If we ever let users define strategies via the API (vs. only generating them), we are one step away from arbitrary code execution via `eval`-backed query parsing. Constrain inputs to a strict whitelist *before* this is exposed.

### Done when

- `curl localhost:8000/runs -X POST -H 'x-api-key: …' -d '…'` kicks off a run.
- Security review approves.

---

## Phase 4 — UI (target: 1–2 sessions)

**Goal:** A frontend that lets you start a run, watch it evolve, browse top strategies, and view a backtest report.

### Tasks

- [ ] Pick the stack. Two paths:
  - **Streamlit** — fast, ugly, perfect for an analyst-facing tool. ~1 session total.
  - **Next.js + shadcn/ui** — slow, pretty, future-proof. ~2 sessions and ongoing maintenance.
  - Recommendation: Streamlit for v1 (Phase 4); Next.js only if/when Phase 6 happens and we need a real ops dashboard.
- [ ] Pages: Runs list → Run detail (live fitness curve across generations) → Strategy detail (parsed expression, equity curve, per-trade table, metrics) → New Run form.
- [ ] Live progress: poll `GET /runs/{id}` every 5s; if we're feeling fancy, server-sent events.
- [ ] Visualisations: fitness max/median per generation, equity curve, drawdown chart, trade scatter (entry time vs. P&L).

### Done when

- You can start a run, watch it tick through generations, and click into the top strategy without touching the CLI.

---

## Phase 5 — Multi-asset (target: 1–2 sessions)

**Goal:** The system isn't BTCUSDC-shaped any more.

### Tasks

- [ ] Replace `python-binance` direct usage with `ccxt` — a unified exchange API covering Binance, Coinbase, Kraken, Bybit, etc. Worth the ~1-day rewrite.
- [ ] Generalise data ingest: `gentrade ingest --asset ETHUSDT --exchange binance --interval 15m --since 2022-01-01`.
- [ ] Cache OHLCV in Parquet (much faster reload than CSV for the inner loop).
- [ ] Per-asset signal catalogue: same indicators, but absolute thresholds may need to differ (a "cheap" RSI level for BTC isn't necessarily cheap for SOL). Decide whether the GA should auto-tune thresholds per asset (probably yes — it already mutates them).
- [ ] Cross-asset robustness check: a strategy that wins on BTCUSDC should be re-evaluated on ETHUSDT, SOLUSDT, etc. If it falls apart, it was a curve-fit. Add this as a standard report.
- [ ] Add equities support via `yfinance` for free daily data (good enough for a multi-asset robustness check; not for trading).

### Risks to flag

- **Survivorship bias.** Crypto pairs get delisted. Equities get acquired. Train on a universe that includes dead names where possible.
- **Time-zone & calendar gotchas.** Crypto trades 24/7; equities don't. The "trade window = 1 day" assumption needs revisiting per asset class.

### Done when

- Same GA, same code, same CLI flags can produce a backtest for `BTCUSDC`, `ETHUSDT`, and `SPY` from a single command.

---

## Phase 6 — Live trading (target: ongoing — gated, not timed)

**Goal:** A small amount of real capital trading the best strategy under tight risk controls.

This phase **does not start** until:
1. Phase 5 is complete and we have ≥3 strategies that survive cross-asset robustness checks.
2. A 30-day **paper trading** run (live data, simulated execution, the API recording fills) has run end-to-end without a single bug surfacing.
3. Risk limits are documented, tested, and reviewed.

### Tasks (in order)

- [ ] **Paper trading mode first.** Live order book, simulated fills, full P&L tracking. Build the entire stack here — don't add "is this real?" as a flag later.
- [ ] Order management system: position state, stop-loss management, take-profit, trailing stops. Idempotent — a network glitch shouldn't double-submit an order.
- [ ] Risk module: max position size, max daily loss, max open positions, circuit breaker that flattens everything if drawdown exceeds X%. **Tests for the risk module are non-negotiable.**
- [ ] Reconciliation: every minute, compare our ledger to the exchange's reported balance. Alert on mismatch.
- [ ] Monitoring: Grafana or a simple status page showing P&L, open positions, last fill, last reconciliation.
- [ ] Kill switch — single endpoint / single command that flattens all positions and disables new entries.
- [ ] **Mandatory `secure-code-reviewer` pass on every PR that touches the trading path.** No exceptions, even for typos.
- [ ] API-key storage: never in git, never in env in plaintext on a shared machine. Use OS keychain (`keyring` library) or AWS Secrets Manager.
- [ ] Start at the smallest unit the exchange allows. Trade for 30 days. Compare actual P&L to the paper trading P&L from the same period. If they diverge by more than 10%, stop and investigate slippage / fee modelling.

### Risks to flag (non-exhaustive)

- The dissertation note: *"the alarming positivity of the results... hints that there could be a calculation error."* Take this seriously. The whole point of Phases 1–5 is to check whether that calculation error exists.
- Real exchanges have rate limits, partial fills, maintenance windows, withdrawals frozen during congestion. The simulator pretends none of this happens.
- One bad fitness function + one absent risk limit + a multi-day connection blip = a margin call. This is the only phase where bugs cost real money.

### Done when

This phase has no "done." It has "trading" or "paused." Treat it as ops, not a project.

---

## Operating model (how the agents collaborate session-to-session)

### Standing rituals

- **Start of every phase**: `Plan` agent writes the implementation plan; we critique it; then we execute.
- **End of every phase**: `allium:weed` for spec drift, `lint-fixer`, `secure-code-reviewer` (mandatory from Phase 3 onward).
- **Pre-PR**: regression suite green, smoke test green, CI green.
- **Once Phase 6 is live**: `schedule` a recurring agent for nightly backtest-vs-live reconciliation reports.

### What main-session Claude does vs. what subagents do

- **Main session**: design discussion, code review, spec edits, decisions about trade-offs.
- **Subagents** (background where possible):
  - Long GA runs (`run_in_background: true`) — fire and forget, notified when complete.
  - Codebase exploration (`Explore`) — keeps grep/find chatter out of main context.
  - TDD writer/implementation pairs — handed off in sequence, each reading the other's output.
  - Security review — stateless, parallelisable, run on every PR touching auth/exchange paths.

### What we do *not* delegate

- Decisions about modelling (which fitness function, which selection pressure, what "production grade" means in Phase 1).
- Decisions about real money (Phase 6 anything).
- Spec edits — `allium:tend` proposes, you approve.

---

## Cross-cutting concerns

### Data

- One canonical source of truth: a Parquet store at `data/ohlcv/<exchange>/<symbol>/<interval>.parquet`.
- All ingests are idempotent and resumable.
- Schema versioned; breaking changes require a migration script.

### Reproducibility

- Every run captures: code git SHA, data file hashes, RNG seed, config, all CLI args. Persisted in the `runs` table. A run from 6 months ago must be reproducible byte-for-byte.

### Performance

- Don't optimise until Phase 1 forces it. Then profile. Then decide whether the win is in: (a) caching backtests for unchanged strategies (elite are re-evaluated every generation today — wasteful), (b) vectorising `find_profit_in_window`, or (c) parallelism across generations.

### Cost

- Local-first. Don't go back to AWS/EC2 until we have a reason. The Phase 0–5 stack should run comfortably on a laptop.

---

## Milestones at a glance

| Phase | Output | Trust threshold |
|---|---|---|
| 0 | Smoke-tests on `main` | "It runs." |
| 1 | Walk-forward backtest report with realistic costs | "The numbers aren't lying to me." |
| 2 | Resumable persistent runs | "I can crash it without losing work." |
| 3 | API + security review | "I could let someone else hit this." |
| 4 | UI | "I can show this to a non-engineer." |
| 5 | Multi-asset | "It's not just one curve fit." |
| 6 | Live trading (gated) | "I trust this with money I can afford to lose." |

---

## What we do next

Phase 0, today. Concrete first-task list:

1. Spin up a worktree for the revival.
2. Migrate `pyproject.toml` to `uv` with Python 3.11.
3. Reorganise modules into a `gentrade/` package.
4. Lazy-load boto3.
5. Fix the four bugs called out in Phase 0.
6. Write the smoke test.
7. CI workflow.
8. `allium:weed` to confirm we haven't drifted further from the spec.
9. PR.

If you give the word, I'll kick this off.
