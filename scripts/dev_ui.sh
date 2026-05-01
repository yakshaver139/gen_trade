#!/usr/bin/env bash
#
# Spin up the Streamlit UI pointing at a running dev API server.
#
# Usage: in one terminal run ./scripts/dev_server.sh; copy the printed
# api key into this command:
#   GENTRADE_API_KEY=<copy from dev_server> ./scripts/dev_ui.sh
#
# Or set GENTRADE_API_URL / GENTRADE_API_KEY in your shell rc.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${UI_PORT:-8501}"
HOST="${UI_HOST:-127.0.0.1}"

: "${GENTRADE_API_URL:=http://127.0.0.1:8000}"
export GENTRADE_API_URL

if [[ -z "${GENTRADE_API_KEY:-}" ]]; then
  cat <<WARN >&2
warning: GENTRADE_API_KEY is not set. Authenticated calls will fail.
         Copy the key printed by ./scripts/dev_server.sh and re-run:
           GENTRADE_API_KEY=<key> ./scripts/dev_ui.sh
WARN
fi

if [[ -n "${GENTRADE_API_KEY:-}" ]]; then
  KEY_STATE="set (${#GENTRADE_API_KEY} chars)"
else
  KEY_STATE="NOT SET"
fi

cat <<INFO

────────────────────────────────────────────────────────────────────
  gentrade UI
  url          http://${HOST}:${PORT}
  api          ${GENTRADE_API_URL}
  api key      ${KEY_STATE}
────────────────────────────────────────────────────────────────────

INFO

exec uv run streamlit run "${ROOT}/gentrade/ui/main.py" \
  --server.address "${HOST}" \
  --server.port "${PORT}" \
  --server.headless true \
  --browser.gatherUsageStats false
