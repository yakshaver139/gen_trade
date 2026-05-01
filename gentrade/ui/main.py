"""Gentrade UI entrypoint.

Run with: ``uv run streamlit run gentrade/ui/main.py``.

This file lives at the package root; per-page Streamlit scripts live in
``gentrade/ui/pages/`` and are auto-discovered by Streamlit's multi-page
router. Configuration (API URL, API key) is read from the
``GENTRADE_API_URL`` and ``GENTRADE_API_KEY`` env vars.
"""
from __future__ import annotations

import streamlit as st

from gentrade.ui.api_client import ApiClient, ApiError, config_from_env

st.set_page_config(page_title="gentrade", layout="wide")

st.title("gentrade")
st.caption("Genetic algorithm runs over technical-indicator strategies.")

cfg = config_from_env()
client = ApiClient(base_url=cfg.base_url, api_key=cfg.api_key)

st.markdown(
    f"""
**API**: `{cfg.base_url}`

Use the navigation in the left sidebar:

- **Runs** — list of persisted runs, status + headline metrics
- **New Run** — start a GA run on a registered asset
- **Run detail** — drill into one run, watch fitness curves
- **Strategy detail** — equity curve + drawdown + per-trade scatter for one strategy
"""
)

with st.expander("server health"):
    if not cfg.api_key:
        st.warning(
            "GENTRADE_API_KEY is not set in this shell. "
            "Authenticated calls will return 401."
        )
    try:
        body = client.healthz()
        st.success(f"healthz → {body}")
    except ApiError as e:
        st.error(f"healthz failed: {e}")
    except Exception as e:  # noqa: BLE001 — connection errors etc, surface verbatim
        st.error(f"can't reach API at {cfg.base_url}: {e}")

st.caption(
    "Phase 4 v1: the UI does not auto-refresh. Use the **Refresh** button on "
    "each page to repoll the API. A live-tail mode will land in a follow-up."
)
