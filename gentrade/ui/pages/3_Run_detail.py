"""Run detail: per-generation curves + headline metrics + chosen strategy.

Reads ``run_id`` from the URL query string when present so the Runs list
can deep-link into here. Both the chosen strategy and the final-generation
population table link onward to the Strategy detail page with both ids
already populated.

Live progress: while the run is ``in_progress`` the body re-renders
every 5 seconds via ``st.fragment(run_every=5)``. Once the status flips
to ``reported`` or ``failed`` the auto-refresh stops and the page
becomes static. The server also exposes
``GET /runs/{id}/events`` (server-sent events) for clients that want
push semantics; the Streamlit page polls because it's the simplest
fit for the top-down script model.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gentrade.ui.api_client import ApiClient, ApiError
from gentrade.ui.format import fmt

st.set_page_config(page_title="gentrade — run detail", layout="wide")
st.title("Run detail")

client = ApiClient()

run_id = st.text_input("Run id", value=st.query_params.get("run_id", ""))
if not run_id:
    st.info("Enter a run id (or pass `?run_id=…` in the URL).")
    st.stop()

# Initial fetch decides whether to wire up the auto-refreshing fragment.
try:
    initial = client.get_run(run_id)
except ApiError as e:
    st.error(str(e))
    st.stop()

is_live = initial["status"] == "in_progress"


def _render(run: dict) -> None:
    """Render the body of the page for the given run snapshot."""
    # ---------------- headline ----------------
    top = st.columns(4)
    top[0].metric("Status", run["status"])
    top[1].metric("Generations done", run["current_generation"])
    top[2].metric("Overfitting gap", fmt(run.get("overfitting_gap"), "+.4f"))
    chosen = run.get("chosen_strategy_id")
    with top[3]:
        if chosen:
            chosen_url = f"/Strategy_detail?run_id={run_id}&strategy_id={chosen}"
            st.markdown(
                f"<div style='font-size:0.875rem;color:#888;'>Chosen strategy</div>"
                f"<div style='font-size:1.5rem;'><a href='{chosen_url}' "
                f"target='_self'>{chosen}</a></div>",
                unsafe_allow_html=True,
            )
        else:
            st.metric("Chosen strategy", "—")

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
        fig.add_trace(go.Scatter(
            x=df["generation"], y=df["validation_median"], name="validation median",
            mode="lines", line={"color": "#ff7f0e", "dash": "dot"},
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
            "Solid = max fitness, dotted = median. When the orange "
            "(validation) lines peel away from the blue (train) lines, "
            "the GA is over-fitting — every additional generation makes "
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
                st.metric(
                    "n_trades",
                    m.get("n_trades") if m.get("n_trades") is not None else "—",
                )
                st.metric("expectancy", fmt(m.get("expectancy"), "+.4f"))
                st.metric("sharpe", fmt(m.get("sharpe"), "+.2f"))
                st.metric("max_dd", fmt(m.get("max_drawdown"), "+.4f"))

        if chosen:
            st.markdown(
                f"➡️ [open chosen strategy → Strategy detail]"
                f"(/Strategy_detail?run_id={run_id}&strategy_id={chosen})"
            )

    # ---------------- final generation population ----------------
    final_n = run.get("current_generation") or 0
    if final_n > 0:
        st.subheader(f"Generation {final_n} population")
        try:
            gen_detail = client.get_generation(run_id, final_n)
        except ApiError as e:
            st.error(f"failed to load generation {final_n}: {e}")
        else:
            strategies = gen_detail.get("strategies", [])
            if strategies:
                df_pop = pd.DataFrame(strategies)
                df_pop["fitness"] = df_pop["fitness"].apply(lambda x: fmt(x, "+.4f"))
                df_pop["link"] = df_pop["id"].apply(
                    lambda sid: f"/Strategy_detail?run_id={run_id}&strategy_id={sid}"
                )
                st.dataframe(
                    df_pop[["link", "rank", "id", "fitness", "parsed_query"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "link": st.column_config.LinkColumn(
                            "Open", display_text="→ strategy detail",
                        ),
                        "rank": st.column_config.NumberColumn("Rank", width="small"),
                        "id": st.column_config.TextColumn("Strategy id"),
                        "fitness": st.column_config.TextColumn("Train fitness"),
                        "parsed_query": st.column_config.TextColumn(
                            "Parsed pandas query", width="large"
                        ),
                    },
                )


if is_live:
    st.caption("🔴 Live — auto-refreshing every 5s while status is `in_progress`.")

    @st.fragment(run_every=5)
    def live_block() -> None:
        try:
            current = client.get_run(run_id)
        except ApiError as e:
            st.error(str(e))
            return
        _render(current)

    live_block()
else:
    if st.button("Refresh"):
        st.rerun()
    _render(initial)
