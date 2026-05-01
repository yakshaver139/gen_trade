"""Runs list page: every persisted run, newest first."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from gentrade.ui.api_client import ApiClient, ApiError
from gentrade.ui.format import fmt

st.set_page_config(page_title="gentrade — runs", layout="wide")
st.title("Runs")

client = ApiClient()

if st.button("Refresh", type="primary"):
    st.rerun()

try:
    rows = client.list_runs()
except ApiError as e:
    st.error(str(e))
    st.stop()

if not rows:
    st.info("No runs persisted yet. Use **New Run** to start one.")
    st.stop()

df = pd.DataFrame(rows)
# Display columns; drop noisy ones for the table.
display_cols = [
    "id",
    "status",
    "current_generation",
    "started_at",
    "finished_at",
    "seed",
    "chosen_strategy_id",
    "overfitting_gap",
]
df = df[[c for c in display_cols if c in df.columns]]
df["overfitting_gap"] = df["overfitting_gap"].apply(lambda x: fmt(x, "+.4f"))

st.dataframe(df, use_container_width=True, hide_index=True)

st.caption(
    "Pick a run and head to **Run detail** in the sidebar — paste the id "
    "into the input there."
)
