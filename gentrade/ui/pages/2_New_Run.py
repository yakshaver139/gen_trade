"""New Run page: form that POSTs /runs."""
from __future__ import annotations

import streamlit as st

from gentrade.ui.api_client import ApiClient, ApiError

st.set_page_config(page_title="gentrade — new run", layout="wide")
st.title("New Run")

client = ApiClient()

try:
    assets = client.list_assets()
except ApiError as e:
    st.error(f"failed to load assets: {e}")
    st.stop()

if not assets:
    st.warning(
        "No assets registered on the server. Set GENTRADE_ASSETS_PATH to a JSON "
        "file describing one before starting a run."
    )
    st.stop()

with st.form("new_run"):
    asset = st.selectbox(
        "Asset", options=[a["asset"] for a in assets], help="Server-side registered asset name."
    )
    cols = st.columns(3)
    with cols[0]:
        population_size = st.number_input("Population size", min_value=2, max_value=200, value=10)
    with cols[1]:
        generations = st.number_input("Generations", min_value=1, max_value=500, value=20)
    with cols[2]:
        seed = st.number_input("Seed", min_value=0, value=42)

    pressure = st.selectbox(
        "Selection pressure",
        ("tournament", "rank_linear", "fitness_proportional"),
        index=0,
    )
    elitism = st.number_input("Elitism count", min_value=0, max_value=10, value=2)

    st.markdown("**Backtest costs (per side)**")
    cols2 = st.columns(4)
    with cols2[0]:
        target_pct = st.number_input("Target %", value=0.015, format="%.4f", min_value=0.0001, max_value=0.5)
    with cols2[1]:
        stop_pct = st.number_input("Stop loss %", value=0.0075, format="%.4f", min_value=0.0001, max_value=0.5)
    with cols2[2]:
        fee_bps = st.number_input("Fee bps", value=10.0, min_value=0.0, max_value=500.0)
    with cols2[3]:
        slip_bps = st.number_input("Slippage bps", value=1.0, min_value=0.0, max_value=500.0)

    submitted = st.form_submit_button("Start run", type="primary")

if submitted:
    body = {
        "asset": asset,
        "population_size": int(population_size),
        "generations": int(generations),
        "seed": int(seed),
        "selection_pressure": pressure,
        "elitism_count": int(elitism),
        "target_pct": float(target_pct),
        "stop_loss_pct": float(stop_pct),
        "taker_fee_bps": float(fee_bps),
        "slippage_bps": float(slip_bps),
    }
    try:
        resp = client.create_run(body)
    except ApiError as e:
        st.error(f"start failed: {e}")
        st.stop()

    st.success(f"Started run {resp['run_id']} (status: {resp['status']}).")
    st.code(resp["run_id"], language="text")
    st.caption("Paste this id into **Run detail** in the sidebar to track progress.")
