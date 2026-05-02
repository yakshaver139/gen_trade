"""New Run page: form that POSTs /runs.

Two ways to populate the initial population:
- **Auto-generate** — server samples from the trusted catalogue
  (default; matches the legacy CLI behaviour).
- **Seed a strategy** — the user builds a single chromosome from
  dropdowns over the catalogue. The GA prepends it to the random
  pool (so the first member of generation 1 is the seeded one) and
  evolves from there.

Validation is enforced server-side against the same catalogue the UI
reads — anything not in the dropdown is rejected by `POST /runs`. The
indicator name is interpolated into a pandas query string downstream,
so this allowlist is a security boundary, not a UX nicety.
"""
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
        "No assets registered on the server. Set GENTRADE_ASSETS_PATH to a "
        "JSON file describing one before starting a run."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Initial-population mode: auto-generate or seed a strategy
# ---------------------------------------------------------------------------
OPS = (">=", "<=", ">", "<")
MODES = ("Auto-generate", "Seed a strategy")

if "seed_rows" not in st.session_state:
    st.session_state.seed_rows = []

mode = st.radio(
    "Initial population",
    options=MODES,
    horizontal=True,
    help=(
        "Auto-generate: server samples from the catalogue. Seed a strategy: "
        "you hand-build the first chromosome and the GA fills the rest of "
        "the population randomly + evolves from there."
    ),
)

# When in seed mode, fetch the catalogue once. Cached across reruns.
catalogue: list[dict] = []
indicator_names: list[str] = []
default_thresholds: dict[str, float] = {}
if mode == "Seed a strategy":
    try:
        catalogue = client.list_catalogue()
    except ApiError as e:
        st.error(f"failed to load catalogue: {e}")
        st.stop()
    indicator_names = sorted(c["indicator"] for c in catalogue if c["absolute_thresholds"])
    if not indicator_names:
        st.warning(
            "No indicators in the catalogue support absolute thresholds. "
            "Seed mode is unavailable; switch to Auto-generate."
        )
        st.stop()
    for c in catalogue:
        if c["absolute_thresholds"]:
            mid = c["absolute_thresholds"][len(c["absolute_thresholds"]) // 2]
            default_thresholds[c["indicator"]] = float(mid)

    # Initialise with one row if none yet.
    if not st.session_state.seed_rows:
        first = indicator_names[0]
        st.session_state.seed_rows = [
            {
                "indicator": first,
                "op": ">=",
                "abs_value": default_thresholds.get(first, 0.5),
                "conjunction": "and",  # ignored on first row
            }
        ]

    st.markdown("**Seeded strategy** — joined left-to-right by the conjunction at the start of each row.")
    rows = st.session_state.seed_rows
    for i, row in enumerate(rows):
        cols = st.columns([1.0, 3.0, 1.0, 2.0, 0.5])
        with cols[0]:
            if i == 0:
                st.markdown("&nbsp;<br><strong>WHEN</strong>", unsafe_allow_html=True)
            else:
                row["conjunction"] = st.selectbox(
                    "conj",
                    options=("and", "or"),
                    index=("and", "or").index(row.get("conjunction", "and")),
                    key=f"conj_{i}",
                    label_visibility="collapsed",
                )
        with cols[1]:
            ind_idx = indicator_names.index(row["indicator"]) if row["indicator"] in indicator_names else 0
            row["indicator"] = st.selectbox(
                "indicator",
                options=indicator_names,
                index=ind_idx,
                key=f"ind_{i}",
                label_visibility="collapsed",
            )
        with cols[2]:
            row["op"] = st.selectbox(
                "op",
                options=OPS,
                index=OPS.index(row.get("op", ">=")),
                key=f"op_{i}",
                label_visibility="collapsed",
            )
        with cols[3]:
            row["abs_value"] = float(
                st.number_input(
                    "abs_value",
                    value=float(row.get("abs_value", 0.0)),
                    format="%.4f",
                    key=f"val_{i}",
                    label_visibility="collapsed",
                )
            )
        with cols[4]:
            if i > 0 and st.button("✕", key=f"rm_{i}", help="remove this row"):
                rows.pop(i)
                st.rerun()

    add_col, _, preview_col = st.columns([1.5, 0.5, 6])
    with add_col:
        if st.button("+ Add indicator row"):
            last = rows[-1] if rows else None
            seed_indicator = (last["indicator"] if last else indicator_names[0])
            rows.append(
                {
                    "indicator": seed_indicator,
                    "op": ">=",
                    "abs_value": default_thresholds.get(seed_indicator, 0.5),
                    "conjunction": "and",
                }
            )
            st.rerun()

    # Live preview of the parsed query string the GA will run.
    fragments = []
    for i, r in enumerate(rows):
        if i > 0:
            fragments.append(r.get("conjunction", "and"))
        fragments.append(f"{r['indicator']} {r['op']} {r['abs_value']:g}")
    with preview_col:
        st.caption("preview")
        st.code(" ".join(fragments), language="python")
else:
    # Reset rows so toggling back doesn't leak old state.
    st.session_state.seed_rows = []

# ---------------------------------------------------------------------------
# Run config form
# ---------------------------------------------------------------------------

with st.form("new_run"):
    asset = st.selectbox(
        "Asset", options=[a["asset"] for a in assets],
        help="Server-side registered asset name.",
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
    body: dict = {
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

    if mode == "Seed a strategy" and st.session_state.seed_rows:
        rows = st.session_state.seed_rows
        body["seed_strategies"] = [
            {
                "indicators": [
                    {
                        "indicator": r["indicator"],
                        "op": r["op"],
                        "absolute": True,
                        "abs_value": float(r["abs_value"]),
                    }
                    for r in rows
                ],
                # Skip the first row's conjunction (it's a placeholder for
                # the WHEN cell). The remaining ones join adjacent rows.
                "conjunctions": [r.get("conjunction", "and") for r in rows[1:]],
            }
        ]

    try:
        resp = client.create_run(body)
    except ApiError as e:
        st.error(f"start failed: {e}")
        st.stop()

    detail_url = f"/Run_detail?run_id={resp['run_id']}"
    st.success(f"Started run `{resp['run_id']}` (status: {resp['status']}).")
    st.link_button("→ Watch live progress", detail_url, type="primary")
    st.caption(
        "The Run detail page auto-refreshes every 5 seconds while the run is "
        "in progress. Once it reaches `reported`, click into the chosen "
        "strategy from there to see the equity curve, drawdown, and trade "
        "scatter."
    )
