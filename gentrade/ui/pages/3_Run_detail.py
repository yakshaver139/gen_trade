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
from gentrade.ui.copy import CHART_HELP, METRIC_HELP
from gentrade.ui.format import fmt, link, render_html_table

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
    top[0].metric("Status", run["status"], help=METRIC_HELP["status"])
    top[1].metric(
        "Generations done",
        run["current_generation"],
        help=METRIC_HELP["current_generation"],
    )
    top[2].metric(
        "Overfitting gap",
        fmt(run.get("overfitting_gap"), "+.4f"),
        help=METRIC_HELP["overfitting_gap"],
    )
    chosen = run.get("chosen_strategy_id")
    with top[3]:
        if chosen:
            chosen_url = f"/Strategy_detail?run_id={run_id}&strategy_id={chosen}"
            st.markdown(
                f"<div style='font-size:0.875rem;color:#888;' "
                f"title='{METRIC_HELP['chosen_strategy']}'>"
                f"Chosen strategy</div>"
                f"<div style='font-size:1.5rem;'><a href='{chosen_url}' "
                f"target='_self'>{chosen}</a></div>",
                unsafe_allow_html=True,
            )
        else:
            st.metric(
                "Chosen strategy", "—", help=METRIC_HELP["chosen_strategy"]
            )

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
        st.caption(CHART_HELP["fitness_curves"])
    else:
        st.info("No generations evaluated yet. Refresh after a few seconds.")

    # ---------------- terminal report metrics ----------------
    if run["status"] == "reported":
        st.subheader("Headline metrics")
        windows = [
            ("train_metrics", "Train"),
            ("validation_metrics", "Validation"),
            ("test_metrics", "Test"),
            ("buy_and_hold_test", "Buy & hold"),
            ("random_entry_test", "Random entry"),
        ]

        def _cell(value: float | int | None, kind: str) -> dict:
            """Render one metric value with sign-based colouring where it's meaningful."""
            if kind == "n_trades":
                if value is None:
                    return "—"
                return str(int(value))
            text = fmt(
                value, "+.4f" if kind != "sharpe" else "+.2f", default="—"
            )
            if value is None or text == "—":
                return text
            try:
                v = float(value)
            except (TypeError, ValueError):
                return text
            if kind in ("expectancy", "sharpe"):
                colour = "#2ca02c" if v > 0 else "#d62728" if v < 0 else "#888"
                return {"html": f'<span style="color:{colour};font-weight:600;">{text}</span>'}
            if kind == "max_drawdown":
                # Always ≤ 0; deeper drawdowns get a redder background.
                magnitude = min(1.0, abs(v) / 0.30)  # cap at 30% drawdown
                alpha = 0.12 + 0.40 * magnitude
                return {
                    "html": (
                        f'<span style="background-color:rgba(214,39,40,{alpha:.2f});'
                        f'padding:0.1rem 0.4rem;border-radius:3px;'
                        f'color:#d62728;font-weight:600;">{text}</span>'
                    )
                }
            return text

        rows_html: list[dict] = []
        metric_specs = [
            ("n_trades", "n_trades", "n_trades"),
            ("expectancy", "expectancy", "expectancy"),
            ("sharpe", "sharpe", "sharpe"),
            ("max_drawdown", "max_dd", "max_drawdown"),
        ]
        for metric_key, label_text, kind in metric_specs:
            row: dict = {
                "metric": {
                    "html": (
                        f'<span title="{METRIC_HELP[metric_key]}">'
                        f'<b>{label_text}</b></span>'
                    )
                }
            }
            for win_key, _ in windows:
                m = run.get(win_key) or {}
                row[win_key] = _cell(m.get(metric_key), kind)
            rows_html.append(row)

        columns = [{"key": "metric", "label": "", "align": "left"}] + [
            {"key": k, "label": label, "align": "right"} for k, label in windows
        ]
        st.markdown(
            render_html_table(columns, rows_html),
            unsafe_allow_html=True,
        )

        if chosen:
            st.markdown(
                f'➡️ <a href="/Strategy_detail?run_id={run_id}&strategy_id={chosen}" '
                f'target="_self">open chosen strategy → Strategy detail</a>',
                unsafe_allow_html=True,
            )

    # ---------------- breeding activity ----------------
    try:
        events = client.list_breeding_events(run_id)
    except ApiError as e:
        st.error(f"failed to load breeding events: {e}")
        events = []

    if events:
        st.subheader("Breeding activity")
        ev_df = pd.DataFrame(events)
        # Operator counts per generation → stacked bar.
        # The (elite) carry-over is shown alongside the seven mutation operators.
        op_counts = (
            ev_df.groupby(["generation_number", "operator"])
            .size()
            .reset_index(name="count")
        )

        operator_palette = {
            "(elite)": "#888888",
            "perturb_threshold": "#1f77b4",
            "flip_operator": "#ff7f0e",
            "swap_indicator": "#2ca02c",
            "flip_conjunction": "#d62728",
            "add_signal": "#9467bd",
            "remove_signal": "#8c564b",
            "swap_rel_target": "#e377c2",
        }

        fig_op = go.Figure()
        # One bar trace per operator so they stack.
        seen_ops = list(op_counts["operator"].unique())
        # Plot in the palette order so the legend reads sensibly.
        for op in operator_palette:
            if op not in seen_ops:
                continue
            sub = op_counts[op_counts["operator"] == op]
            fig_op.add_trace(go.Bar(
                x=sub["generation_number"],
                y=sub["count"],
                name=op,
                marker={"color": operator_palette.get(op, "#444")},
            ))
        # Any operator the run produced that wasn't in our palette (future-proof).
        for op in seen_ops:
            if op in operator_palette:
                continue
            sub = op_counts[op_counts["operator"] == op]
            fig_op.add_trace(go.Bar(
                x=sub["generation_number"], y=sub["count"], name=op,
            ))

        fig_op.update_layout(
            barmode="stack",
            xaxis_title="generation",
            yaxis_title="children",
            margin={"t": 20},
            height=320,
            legend={"orientation": "h", "y": 1.05},
        )
        st.plotly_chart(fig_op, use_container_width=True)
        st.caption(CHART_HELP["operator_counts"])

        with st.expander(f"event log ({len(events)} events)", expanded=False):
            # Tint each row by operator. Elites get a soft grey;
            # parameter mutations a blue tint; structural ones a green
            # tint; failures a red tint.
            def _row_tint(op: str, applied: bool) -> str:
                if not applied and op != "(elite)":
                    return "rgba(214, 39, 40, 0.10)"
                if op == "(elite)":
                    return "rgba(136, 136, 136, 0.10)"
                if op in {"perturb_threshold", "flip_operator", "swap_rel_target"}:
                    return "rgba(31, 119, 180, 0.10)"  # parameter
                return "rgba(46, 160, 67, 0.10)"  # structural

            log_columns = [
                {"key": "gen", "label": "Gen", "align": "right"},
                {"key": "child", "label": "Child"},
                {"key": "parents", "label": "Parents (a / b)"},
                {"key": "operator", "label": "Operator"},
                {"key": "applied", "label": "Applied", "align": "center"},
                {"key": "reason", "label": "Reason"},
            ]
            log_rows = [
                {
                    "gen": e["generation_number"],
                    "child": e["child_id"][:8] + "…",
                    "parents": (
                        e["parent_a_id"][:8] + "… / " + e["parent_b_id"][:8] + "…"
                    ),
                    "operator": e["operator"],
                    "applied": "✓" if e["applied"] else "—",
                    "reason": e.get("reason") or "",
                    "_row_style": (
                        f"background-color: {_row_tint(e['operator'], e['applied'])};"
                    ),
                }
                # newest generation first; within a gen keep the order
                for e in reversed(events)
            ]
            st.markdown(
                render_html_table(log_columns, log_rows),
                unsafe_allow_html=True,
            )
            st.caption(CHART_HELP["breeding_events_table"])

    # ---------------- lineage tree ----------------
    if events:
        try:
            all_strategies = client.list_strategies(run_id)
        except ApiError as e:
            st.error(f"failed to load strategies for lineage: {e}")
            all_strategies = []

        if all_strategies:
            with st.expander("Lineage tree", expanded=False):
                # Index strategies by (gen, id) → (rank, fitness) for lookup.
                strat_idx: dict[tuple[int, str], dict] = {}
                for s in all_strategies:
                    strat_idx[(s["generation_number"], s["id"])] = s

                # Build edge segments per operator. Each segment runs
                # from a parent at gen-1 to the child at gen; we use
                # NaN separators so a single trace handles many edges.
                edge_segments: dict[str, tuple[list, list]] = {}
                for ev in events:
                    gen = ev["generation_number"]
                    child_pos = strat_idx.get((gen, ev["child_id"]))
                    if not child_pos:
                        continue
                    for parent_id in (ev["parent_a_id"], ev["parent_b_id"]):
                        parent_pos = strat_idx.get((gen - 1, parent_id))
                        if not parent_pos:
                            continue
                        xs, ys = edge_segments.setdefault(
                            ev["operator"], ([], [])
                        )
                        xs.extend([
                            parent_pos["generation_number"],
                            child_pos["generation_number"],
                            None,
                        ])
                        ys.extend([
                            -parent_pos["rank"],
                            -child_pos["rank"],
                            None,
                        ])

                fig_lineage = go.Figure()
                # Plot edges in the same palette as the operator chart.
                for op in operator_palette:
                    if op not in edge_segments:
                        continue
                    xs, ys = edge_segments[op]
                    fig_lineage.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines",
                        line={
                            "color": operator_palette.get(op, "#444"),
                            "width": 1,
                        },
                        opacity=0.55 if op == "(elite)" else 0.85,
                        name=op,
                        hoverinfo="skip",
                        legendgroup=op,
                    ))
                # Catch-all for unexpected operators.
                for op, (xs, ys) in edge_segments.items():
                    if op in operator_palette:
                        continue
                    fig_lineage.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines",
                        line={"color": "#444", "width": 1},
                        opacity=0.6, name=op, hoverinfo="skip",
                    ))

                # Nodes: one marker per strategy, colour by fitness.
                node_x = [s["generation_number"] for s in all_strategies]
                node_y = [-s["rank"] for s in all_strategies]
                node_text = [
                    (
                        f"gen {s['generation_number']}<br>"
                        f"{s['id'][:8]}…<br>rank {s['rank']}<br>"
                        f"fitness {fmt(s.get('fitness'), '+.4f')}"
                    )
                    for s in all_strategies
                ]
                node_color = [
                    s["fitness"] if s.get("fitness") is not None else 0.0
                    for s in all_strategies
                ]
                fig_lineage.add_trace(go.Scatter(
                    x=node_x, y=node_y, mode="markers",
                    marker={
                        "color": node_color,
                        "colorscale": "Viridis",
                        "size": 9,
                        "line": {"width": 0.5, "color": "rgba(0,0,0,0.3)"},
                        "colorbar": {"title": "fitness", "thickness": 12},
                    },
                    text=node_text,
                    hovertemplate="%{text}<extra></extra>",
                    name="strategies",
                    showlegend=False,
                ))

                fig_lineage.update_layout(
                    xaxis_title="generation",
                    yaxis_title="rank (top = best)",
                    yaxis={"autorange": "reversed", "tickmode": "array",
                           "tickvals": [], "showticklabels": False},
                    margin={"t": 20},
                    height=480,
                    legend={"orientation": "h", "y": 1.05},
                )
                st.plotly_chart(fig_lineage, use_container_width=True)
                st.caption(CHART_HELP["lineage_tree"])

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
                pop_columns = [
                    {"key": "open", "label": "Open"},
                    {"key": "rank", "label": "Rank", "align": "right"},
                    {"key": "id", "label": "Strategy id"},
                    {"key": "fitness", "label": "Train fitness", "align": "right"},
                    {"key": "parsed_query", "label": "Parsed pandas query"},
                ]
                pop_rows = [
                    {
                        "open": link(
                            f"/Strategy_detail?run_id={run_id}&strategy_id={s['id']}",
                            "→ strategy detail",
                        ),
                        "rank": s.get("rank", ""),
                        "id": s.get("id", "")[:8] + "…",
                        "fitness": fmt(s.get("fitness"), "+.4f"),
                        "parsed_query": s.get("parsed_query", ""),
                    }
                    for s in strategies
                ]
                st.markdown(
                    render_html_table(pop_columns, pop_rows),
                    unsafe_allow_html=True,
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
