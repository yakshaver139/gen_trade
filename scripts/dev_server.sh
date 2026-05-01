#!/usr/bin/env bash
#
# Spin up the gentrade API against a synthetic dataset — no Binance creds,
# no manual setup. Drops bars + an assets JSON + a fresh SQLite DB into
# .dev/ alongside the repo.
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
BARS_CSV="${DEV_DIR}/SAW-15m.csv"
ASSETS_JSON="${DEV_DIR}/assets.json"
DB_PATH="${DEV_DIR}/gentrade.db"
LOG_DIR="${DEV_DIR}/runs"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

mkdir -p "${DEV_DIR}" "${LOG_DIR}"

# 1. Synthesise the bars (idempotent — overwrites each run, the synthetic
#    series is deterministic anyway).
uv run python "${ROOT}/scripts/dev_bars.py" "${BARS_CSV}"

# 2. Write the assets JSON. The path inside it is absolute and lives under
#    GENTRADE_DATA_ROOT so the registry-load boundary check passes.
cat > "${ASSETS_JSON}" <<JSON
[
  {"asset": "SAW-15m", "exchange": "synthetic", "interval": "15m", "path": "${BARS_CSV}"}
]
JSON
echo "wrote asset registry to ${ASSETS_JSON}"

# 3. API key. Use what's in the env if the caller already set it; otherwise
#    generate a fresh one. Printed once for `curl -H "X-API-Key: ..."`.
if [[ -z "${GENTRADE_API_KEY:-}" ]]; then
  export GENTRADE_API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(16))")
fi

# 4. Wire env vars and exec uvicorn. `--factory` invokes
#    gentrade.api.app.default_app() which reads GENTRADE_DB_URL etc.
export GENTRADE_ASSETS_PATH="${ASSETS_JSON}"
export GENTRADE_DATA_ROOT="${DEV_DIR}"
export GENTRADE_DB_URL="sqlite:///${DB_PATH}"
export GENTRADE_LOG_DIR="${LOG_DIR}"

cat <<INFO

────────────────────────────────────────────────────────────────────
  gentrade dev server
  url            http://${HOST}:${PORT}
  docs           http://${HOST}:${PORT}/docs
  api key        ${GENTRADE_API_KEY}
  asset          SAW-15m  →  ${BARS_CSV}
  db             ${DB_PATH}
  per-run logs   ${LOG_DIR}/<run_id>/run.log

  try:
    curl -H "X-API-Key: \${GENTRADE_API_KEY}" http://${HOST}:${PORT}/assets
    curl -X POST http://${HOST}:${PORT}/runs \\
      -H "X-API-Key: \${GENTRADE_API_KEY}" \\
      -H 'content-type: application/json' \\
      -d '{"asset":"SAW-15m","population_size":4,"generations":5,"seed":42}'
────────────────────────────────────────────────────────────────────

INFO

exec uv run uvicorn gentrade.api.app:default_app --factory --host "${HOST}" --port "${PORT}"
