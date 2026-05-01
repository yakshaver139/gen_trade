"""Runs list page: every persisted run, newest first.

The `id` column links to the Run detail page with `?run_id=...` baked
in, and `chosen_strategy_id` links to the Strategy detail page with
both `run_id` and `strategy_id` populated. No copy-paste required.
"""
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
df["overfitting_gap"] = df["overfitting_gap"].apply(lambda x: fmt(x, "+.4f"))

# Build deep-link URLs per row.
df["run_link"] = df["id"].apply(lambda x: f"/Run_detail?run_id={x}")
df["strategy_link"] = df.apply(
    lambda row: (
        f"/Strategy_detail?run_id={row['id']}"
        f"&strategy_id={row['chosen_strategy_id']}"
    )
    if pd.notna(row.get("chosen_strategy_id")) and row.get("chosen_strategy_id")
    else None,
    axis=1,
)

display_cols = [
    "run_link",
    "id",
    "status",
    "current_generation",
    "started_at",
    "finished_at",
    "seed",
    "chosen_strategy_id",
    "strategy_link",
    "overfitting_gap",
]
df_show = df[[c for c in display_cols if c in df.columns]]

st.dataframe(
    df_show,
    use_container_width=True,
    hide_index=True,
    column_config={
        "run_link": st.column_config.LinkColumn(
            "Open",
            display_text="→ run detail",
            help="open this run's detail page",
        ),
        "id": st.column_config.TextColumn("Run id"),
        "status": "Status",
        "current_generation": st.column_config.NumberColumn("Gen", width="small"),
        "started_at": "Started",
        "finished_at": "Finished",
        "seed": st.column_config.NumberColumn("Seed", width="small"),
        "chosen_strategy_id": st.column_config.TextColumn("Chosen strategy"),
        "strategy_link": st.column_config.LinkColumn(
            "Open strategy",
            display_text="→ strategy detail",
            help="open the chosen strategy's detail page",
        ),
        "overfitting_gap": st.column_config.TextColumn("Overfitting gap"),
    },
)
