#!/usr/bin/env bash
#
# Spin up the gentrade API. Always registers the synthetic SAW-15m asset
# (deterministic, no network); also registers any real Parquet bars it
# finds under data/ so a `gentrade ingest` output shows up automatically.
#
# Usage:
#   ./scripts/dev_server.sh             # run on http://127.0.0.1:8000
#   PORT=9000 ./scripts/dev_server.sh   # different port
#
# The API key is generated fresh per run unless GENTRADE_API_KEY is
# already set; it's printed once at startup so you can curl with it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_DIR="${ROOT}/.dev"
DATA_DIR="${ROOT}/data"
BARS_CSV="${DEV_DIR}/SAW-15m.csv"
ASSETS_JSON="${DEV_DIR}/assets.json"
DB_PATH="${DEV_DIR}/gentrade.db"
LOG_DIR="${DEV_DIR}/runs"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

mkdir -p "${DEV_DIR}" "${LOG_DIR}" "${DATA_DIR}"

# 1. Synthesise the bars (idempotent — overwrites each run, the synthetic
#    series is deterministic anyway).
uv run python "${ROOT}/scripts/dev_bars.py" "${BARS_CSV}"

# 2. Write the assets JSON. Always include the synthetic SAW-15m; auto-
#    register any *.parquet found under data/ alongside it. Asset ids are
#    derived from the filename (stripped of ".parquet"). Paths must
#    resolve under GENTRADE_DATA_ROOT — set below to ROOT so both
#    .dev/ and data/ are inside the boundary.
uv run python - "${ROOT}" "${BARS_CSV}" "${ASSETS_JSON}" "${DATA_DIR}" <<'PY'
import json, sys
from pathlib import Path

root, saw_csv, out_json, data_dir = sys.argv[1:]
data_dir_path = Path(data_dir)

entries = [
    {"asset": "SAW-15m", "exchange": "synthetic", "interval": "15m", "path": saw_csv}
]

for parquet in sorted(data_dir_path.glob("*.parquet")):
    asset_id = parquet.stem  # e.g. BTCUSDT-15m
    # Best-effort split of "<symbol>-<interval>".
    if "-" in asset_id:
        _symbol, interval = asset_id.rsplit("-", 1)
    else:
        interval = "15m"
    entries.append({
        "asset": asset_id,
        "exchange": "ingested",
        "interval": interval,
        "path": str(parquet.resolve()),
    })

Path(out_json).write_text(json.dumps(entries, indent=2))
print(f"wrote asset registry with {len(entries)} entries to {out_json}")
for e in entries:
    print(f"  - {e['asset']:<20} → {e['path']}")
PY

# 3. API key. Use what's in the env if the caller already set it; otherwise
#    generate a fresh one. Printed once for `curl -H "X-API-Key: ..."`.
if [[ -z "${GENTRADE_API_KEY:-}" ]]; then
  export GENTRADE_API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(16))")
fi

# 4. Wire env vars and exec uvicorn. `--factory` invokes
#    gentrade.api.app.default_app() which reads GENTRADE_DB_URL etc.
#    GENTRADE_DATA_ROOT covers both .dev/ and data/.
export GENTRADE_ASSETS_PATH="${ASSETS_JSON}"
export GENTRADE_DATA_ROOT="${ROOT}"
export GENTRADE_DB_URL="sqlite:///${DB_PATH}"
export GENTRADE_LOG_DIR="${LOG_DIR}"

cat <<INFO

────────────────────────────────────────────────────────────────────
  gentrade dev server
  url            http://${HOST}:${PORT}
  docs           http://${HOST}:${PORT}/docs
  api key        ${GENTRADE_API_KEY}
  assets         see ${ASSETS_JSON}
  db             ${DB_PATH}
  per-run logs   ${LOG_DIR}/<run_id>/run.log
  data root      ${ROOT}

  try:
    curl -H "X-API-Key: \${GENTRADE_API_KEY}" http://${HOST}:${PORT}/assets
    curl -X POST http://${HOST}:${PORT}/runs \\
      -H "X-API-Key: \${GENTRADE_API_KEY}" \\
      -H 'content-type: application/json' \\
      -d '{"asset":"BTCUSDT-15m","population_size":6,"generations":5,"seed":42}'
────────────────────────────────────────────────────────────────────

INFO

exec uv run uvicorn gentrade.api.app:default_app --factory --host "${HOST}" --port "${PORT}"
