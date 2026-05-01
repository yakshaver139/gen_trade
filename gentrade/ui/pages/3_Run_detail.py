"""Run detail: per-generation curves + headline metrics + chosen strategy."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gentrade.ui.api_client import ApiClient, ApiError

st.set_page_config(page_title="gentrade — run detail", layout="wide")
st.title("Run detail")

client = ApiClient()

run_id = st.text_input("Run id", value=st.query_params.get("run_id", ""))
if not run_id:
    st.info("Enter a run id (or pass `?run_id=…` in the URL).")
    st.stop()

if st.button("Refresh"):
    st.rerun()

try:
    run = client.get_run(run_id)
except ApiError as e:
    st.error(str(e))
    st.stop()

# ---------------- headline ----------------
top = st.columns(4)
top[0].metric("Status", run["status"])
top[1].metric("Generations done", run["current_generation"])
top[2].metric(
    "Overfitting gap",
    f"{run['overfitting_gap']:+.4f}" if run["overfitting_gap"] is not None else "—",
)
top[3].metric("Chosen strategy", run["chosen_strategy_id"] or "—")

with st.expander("manifest"):
    st.json(run["manifest"])

# ---------------- per-generation curves ----------------
gens = run.get("generations", [])
if gens:
    df = pd.DataFrame(
        [
            {
                "generation": g["number"],
                "train_max": g["train_metrics"]["max_fitness"],
                "train_median": g["train_metrics"]["median_fitness"],
                "validation_max": g["validation_metrics"]["max_fitness"],
                "validation_median": g["validation_metrics"]["median_fitness"],
            }
            for g in gens
        ]
    )

    st.subheader("Fitness across generations")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["train_max"], name="train max",
        mode="lines+markers", line={"color": "#1f77b4"},
    ))
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["validation_max"], name="validation max",
        mode="lines+markers", line={"color": "#ff7f0e", "dash": "dash"},
    ))
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["train_median"], name="train median",
        mode="lines", line={"color": "#1f77b4", "dash": "dot"},
    ))
    fig.update_layout(
        xaxis_title="generation",
        yaxis_title="fitness (per-trade expectancy)",
        legend={"orientation": "h"},
        margin={"t": 20},
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "When the orange (validation) line peels away from the blue (train) "
        "line, the GA is over-fitting — every additional generation makes "
        "out-of-sample worse."
    )
else:
    st.info("No generations evaluated yet. Refresh after a few seconds.")

# ---------------- terminal report metrics ----------------
if run["status"] == "reported":
    st.subheader("Headline metrics")
    cols = st.columns(5)
    for i, (label, key) in enumerate([
        ("Train", "train_metrics"),
        ("Validation", "validation_metrics"),
        ("Test", "test_metrics"),
        ("Buy & hold", "buy_and_hold_test"),
        ("Random entry", "random_entry_test"),
    ]):
        m = run.get(key) or {}
        with cols[i]:
            st.markdown(f"**{label}**")
            st.metric("n_trades", m.get("n_trades", "—"))
            st.metric("expectancy", f"{m.get('expectancy', 0):+.4f}")
            st.metric("sharpe", f"{m.get('sharpe', 0):+.2f}")
            st.metric("max_dd", f"{m.get('max_drawdown', 0):+.4f}")

    chosen = run.get("chosen_strategy_id")
    if chosen:
        st.markdown(
            f"➡️ Open **Strategy detail** in the sidebar with run_id "
            f"`{run_id}` and strategy_id `{chosen}` to see the test-window "
            "equity curve and per-trade scatter."
        )
