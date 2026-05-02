"""Runs list page: every persisted run, newest first.

Rendered as an HTML table (rather than ``st.dataframe`` with
``LinkColumn``) so the deep-link anchors can carry ``target='_self'``
— ``LinkColumn`` opens links in a new tab with no public override.
"""
from __future__ import annotations

import streamlit as st

from gentrade.ui.api_client import ApiClient, ApiError
from gentrade.ui.format import fmt, link, render_html_table

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

columns = [
    {"key": "run_link", "label": "Open"},
    {"key": "id", "label": "Run id"},
    {"key": "status", "label": "Status"},
    {"key": "current_generation", "label": "Gen", "align": "right"},
    {"key": "started_at", "label": "Started"},
    {"key": "finished_at", "label": "Finished"},
    {"key": "seed", "label": "Seed", "align": "right"},
    {"key": "chosen_strategy_id", "label": "Chosen strategy"},
    {"key": "strategy_link", "label": "Open strategy"},
    {"key": "overfitting_gap", "label": "Overfitting gap", "align": "right"},
]


def _render_row(r: dict) -> dict:
    run_id = r["id"]
    chosen = r.get("chosen_strategy_id")
    started = r.get("started_at")
    finished = r.get("finished_at")
    return {
        "run_link": link(f"/Run_detail?run_id={run_id}", "→ run detail"),
        "id": run_id[:8] + "…",
        "status": r.get("status", ""),
        "current_generation": r.get("current_generation", ""),
        "started_at": started.isoformat(timespec="seconds") if started else "",
        "finished_at": finished.isoformat(timespec="seconds") if finished else "",
        "seed": r.get("seed", ""),
        "chosen_strategy_id": (chosen[:8] + "…") if chosen else "—",
        "strategy_link": (
            link(
                f"/Strategy_detail?run_id={run_id}&strategy_id={chosen}",
                "→ strategy detail",
            )
            if chosen
            else "—"
        ),
        "overfitting_gap": fmt(r.get("overfitting_gap"), "+.4f"),
    }


st.markdown(
    render_html_table(columns, [_render_row(r) for r in rows]),
    unsafe_allow_html=True,
)
