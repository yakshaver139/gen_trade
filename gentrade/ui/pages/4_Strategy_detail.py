"""Strategy detail: parsed expression + equity curve + drawdown + trade scatter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gentrade.ui.api_client import ApiClient, ApiError
from gentrade.ui.copy import CHART_HELP, METRIC_HELP
from gentrade.ui.format import fmt

st.set_page_config(page_title="gentrade — strategy detail", layout="wide")
st.title("Strategy detail")

client = ApiClient()

cols = st.columns(2)
with cols[0]:
    run_id = st.text_input("Run id", value=st.query_params.get("run_id", ""))
with cols[1]:
    strategy_id = st.text_input("Strategy id", value=st.query_params.get("strategy_id", ""))

if not run_id or not strategy_id:
    st.info("Enter both ids (or use the **Run detail** page's link to populate them).")
    st.stop()

st.markdown(f"← [back to run detail](/Run_detail?run_id={run_id})")

if st.button("Refresh"):
    st.rerun()

# ---------------- chromosome ----------------
try:
    strat = client.get_strategy(run_id, strategy_id)
except ApiError as e:
    st.error(str(e))
    st.stop()

st.subheader("Chromosome")
st.metric("rank", strat["rank"], help=METRIC_HELP["rank"])
st.metric(
    "fitness (train)",
    fmt(strat.get("fitness"), "+.4f"),
    help=METRIC_HELP["fitness"],
)
st.markdown("**Parsed pandas query**")
st.code(strat["parsed_query"], language="python")

with st.expander("indicators / conjunctions"):
    st.json({"indicators": strat["indicators"], "conjunctions": strat["conjunctions"]})

# ---------------- backtest report (with trades) ----------------
st.subheader("Test-window backtest")
try:
    bt = client.post_backtest(run_id, strategy_id)
except ApiError as e:
    st.error(str(e))
    st.stop()

m = bt["test_metrics"] or {}
metric_cols = st.columns(5)
metric_cols[0].metric("trades", m.get("n_trades", 0), help=METRIC_HELP["n_trades"])
metric_cols[1].metric(
    "win rate", fmt(m.get("win_rate"), ".2%"), help=METRIC_HELP["win_rate"]
)
metric_cols[2].metric(
    "expectancy",
    fmt(m.get("expectancy"), "+.4f"),
    help=METRIC_HELP["expectancy"],
)
metric_cols[3].metric(
    "sharpe", fmt(m.get("sharpe"), "+.2f"), help=METRIC_HELP["sharpe"]
)
metric_cols[4].metric(
    "max DD", fmt(m.get("max_drawdown"), "+.4f"), help=METRIC_HELP["max_drawdown"]
)

trades = bt.get("test_trades") or []
if not trades:
    st.warning("No test-window trades for this strategy — nothing to chart.")
    st.stop()

tdf = pd.DataFrame(trades)
tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
tdf["exit_time"] = pd.to_datetime(tdf["exit_time"])
tdf = tdf.sort_values("entry_time").reset_index(drop=True)
tdf["equity"] = (1.0 + tdf["return"]).cumprod()
tdf["running_peak"] = tdf["equity"].cummax()
tdf["drawdown"] = (tdf["equity"] - tdf["running_peak"]) / tdf["running_peak"]

# ---------------- equity curve ----------------
st.markdown("**Equity curve** (per-trade compounded, starts at 1.0)")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=tdf["exit_time"], y=tdf["equity"], mode="lines+markers",
    name="equity",
))
fig.update_layout(
    xaxis_title="exit time",
    yaxis_title="equity multiple",
    margin={"t": 20},
    height=320,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(CHART_HELP["equity_curve"])

# ---------------- drawdown ----------------
st.markdown("**Drawdown**")
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=tdf["exit_time"], y=tdf["drawdown"], mode="lines",
    fill="tozeroy", line={"color": "#d62728"}, name="drawdown",
))
fig_dd.update_layout(
    xaxis_title="exit time",
    yaxis_title="drawdown",
    yaxis={"tickformat": ".0%"},
    margin={"t": 20},
    height=240,
)
st.plotly_chart(fig_dd, use_container_width=True)
st.caption(CHART_HELP["drawdown_curve"])

# ---------------- trade scatter ----------------
st.markdown("**Trade scatter** (entry time vs P&L)")
fig_sc = go.Figure()
colours = np.where(tdf["return"] >= 0, "#2ca02c", "#d62728")
fig_sc.add_trace(go.Scatter(
    x=tdf["entry_time"],
    y=tdf["return"],
    mode="markers",
    marker={"color": colours, "size": 8},
    text=tdf["outcome"],
))
fig_sc.add_hline(y=0, line_dash="dot", line_color="#888")
fig_sc.update_layout(
    xaxis_title="entry time",
    yaxis_title="per-trade return",
    margin={"t": 20},
    height=320,
)
st.plotly_chart(fig_sc, use_container_width=True)
st.caption(CHART_HELP["trade_scatter"])

# ---------------- raw table ----------------
with st.expander("trades"):
    st.dataframe(tdf, use_container_width=True, hide_index=True)

# ---------------- cross-asset robustness ----------------
st.subheader("Cross-asset robustness")
st.caption(
    "Pick other registered assets to re-run this strategy on. If the metrics "
    "fall apart on assets it wasn't trained on, the strategy was a curve fit."
)
try:
    available = client.list_assets()
except ApiError as e:
    st.error(f"failed to load assets: {e}")
    available = []

asset_names = [a["asset"] for a in available]
default_targets = [a for a in asset_names if a != strat.get("base_asset")]
choice = st.multiselect(
    "Assets to compare",
    options=asset_names,
    default=default_targets[:3],
)
if st.button("Compare", disabled=not choice):
    try:
        cmp_resp = client.post_cross_asset(run_id, strategy_id, choice)
    except ApiError as e:
        st.error(str(e))
    else:
        rows = cmp_resp.get("rows", [])
        if not rows:
            st.info("No rows returned.")
        else:
            cmp_rows = []
            for r in rows:
                m = r.get("metrics") or {}
                cmp_rows.append({
                    "asset": r["asset"],
                    "n_bars": r["n_bars"],
                    "n_trades": m.get("n_trades", 0),
                    "win_rate": fmt(m.get("win_rate"), ".2%"),
                    "expectancy": fmt(m.get("expectancy"), "+.4f"),
                    "sharpe": fmt(m.get("sharpe"), "+.2f"),
                    "max_dd": fmt(m.get("max_drawdown"), "+.4f"),
                    "error": r.get("error") or "",
                })
            cmp_df = pd.DataFrame(cmp_rows)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
            st.caption(CHART_HELP["cross_asset_table"])
            base = cmp_resp.get("base_asset")
            if base:
                st.caption(
                    f"This strategy was trained on `{base}`. Compare its "
                    f"out-of-sample test metrics there to the per-asset "
                    f"rows above — large drops on other assets are the "
                    f"curve-fit signal."
                )
