"""
Cascading Impact Explorer — an interactive Streamlit front end for the
demand-driven Leontief cascading-impact model on a Physical Input-Output
Table (PIOT).

Slide the final-demand increase (% of each commodity's baseline final demand)
and watch the cascading output impact, the tier-by-tier decomposition, and the
waste cascade update live.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import glob
import io
import os

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import model as m

# --------------------------------------------------------------------------- #
# Page config & light styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Cascading Impact Explorer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      div[data-testid="stMetricValue"] {font-size: 1.5rem;}
      .small-note {color: #6b7280; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# The bundled PIOT. Falls back to the first CSV in the folder if the exact
# name changes, so a renamed data file in the repo still loads.
_HERE = os.path.dirname(__file__)
_PREFERRED = os.path.join(_HERE, "PIOT APAP Model Cascading 1.csv")


def _default_csv_path() -> str:
    if os.path.exists(_PREFERRED):
        return _PREFERRED
    candidates = sorted(glob.glob(os.path.join(_HERE, "*.csv")))
    return candidates[0] if candidates else _PREFERRED


DEFAULT_CSV = _default_csv_path()


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_default_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@st.cache_data(show_spinner=False)
def load_and_build(file_bytes: bytes):
    """Load a PIOT from raw bytes and build the network. Cached on the bytes."""
    df, industries = m.load_piot(io.BytesIO(file_bytes))
    net = m.build_network(df, industries)
    return df, industries, net


def fmt_kg(v: float) -> str:
    """Human-friendly kg formatting with metric-ish suffixes."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    a = abs(v)
    for div, suf in [(1e12, " Tg"), (1e9, " Gg"), (1e6, " Mg"), (1e3, " t")]:
        if a >= div:
            return f"{v / div:,.2f}{suf}"
    return f"{v:,.0f} kg"


# --------------------------------------------------------------------------- #
# Sidebar — data source & shock definition
# --------------------------------------------------------------------------- #
st.sidebar.title("🔗 Cascading Impact")
st.sidebar.caption("Demand-driven Leontief model on a Physical Input-Output Table")

with st.sidebar.expander("1 · Data source", expanded=False):
    uploaded = st.file_uploader(
        "Upload a custom PIOT CSV (optional)",
        type=["csv"],
        help=(
            "Industries as both rows and columns, plus meta columns "
            "ROE, EXPORTS, FINAL_DEMAND, WASTE. Leave empty to use the bundled "
            "Acetaminophen PIOT."
        ),
    )
    st.caption(f"Bundled file: **{os.path.basename(DEFAULT_CSV)}**")

file_bytes = uploaded.getvalue() if uploaded is not None else _load_default_bytes(DEFAULT_CSV)

try:
    df, industries, net = load_and_build(file_bytes)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not build the network from this PIOT: {exc}")
    st.stop()

x0 = net["x0"]
d0 = net["d0"]

# Shock mode is fixed: % increase in each commodity's baseline final demand.
MODE = "pct_final_demand"

st.sidebar.markdown("### 2 · Shock definition")
st.sidebar.caption(
    "Each slider is a **% increase in that commodity's baseline final demand**. "
    "Δdᵢ = d₀ᵢ × (slider ÷ 100)."
)

n_tiers = st.sidebar.slider(
    "Explicit supply-chain tiers",
    min_value=2,
    max_value=6,
    value=3,
    help="Tiers 0…(n-1) are shown explicitly; everything deeper is grouped as 'tier n+'.",
)

colq1, colq2 = st.sidebar.columns(2)
if colq1.button("Reset sliders", use_container_width=True):
    for name in industries:
        st.session_state[f"shock_{name}"] = 0.0
if colq2.button("APAP +15%", use_container_width=True,
                help="Quick preset: 15% increase in Acetaminophen final demand."):
    for name in industries:
        st.session_state[f"shock_{name}"] = 0.0
    if "Acetaminophen" in industries:
        st.session_state["shock_Acetaminophen"] = 15.0

st.sidebar.markdown("### 3 · Final-demand sliders")
st.sidebar.caption("Drag a commodity to raise its final demand. The cascade recomputes instantly.")

# One % slider per commodity (% of baseline final demand).
shocks: dict[str, float] = {}
for i, name in enumerate(industries):
    key = f"shock_{name}"
    st.session_state.setdefault(key, 0.0)
    has_fd = d0[i] > 0
    label = f"{name}" + ("" if has_fd else "  (no baseline FD)")
    if not has_fd:
        st.session_state[key] = 0.0
    val = st.sidebar.slider(
        label, min_value=0.0, max_value=200.0,
        step=1.0, key=key, format="%.0f%%",
        disabled=not has_fd,
    )
    shocks[name] = val

# --------------------------------------------------------------------------- #
# Compute everything
# --------------------------------------------------------------------------- #
delta_d = m.build_delta_d(net, shocks, mode=MODE)
casc = m.cascading_impact(net, delta_d)
tiers = m.tier_decomposition(net, delta_d, n_tiers=n_tiers)
waste = m.waste_cascade(net, casc["delta_x"].values)

total_dx = float(casc["delta_x"].sum())
total_dd = float(delta_d.sum())
total_dw = float(waste["delta_waste"].sum())
multiplier = (total_dx / total_dd) if total_dd else float("nan")
any_shock = np.any(delta_d != 0)

# --------------------------------------------------------------------------- #
# Header + KPI row
# --------------------------------------------------------------------------- #
st.title("Cascading Impact Explorer")
st.markdown(
    "A change in final demand for one commodity ripples upstream through the "
    "whole supply chain. Move the sliders on the left to see the total cascading "
    "output impact, how it splits across supply-chain tiers, and the waste it drags along."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Injected final demand  (Σ Δd)", fmt_kg(total_dd))
k2.metric("Total cascading output  (Σ Δx)", fmt_kg(total_dx))
k3.metric("Output multiplier  (Σ Δx / Σ Δd)",
          f"{multiplier:,.2f}×" if total_dd else "—")
k4.metric("Total waste cascade  (Σ ΔW)", fmt_kg(total_dw))

with st.expander("Network diagnostics"):
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Industries", f"{len(industries)}")
    dc2.metric("Spectral radius of A", f"{net['spectral_radius']:.4f}",
               help="Must be < 1 for the Leontief inverse to exist.")
    active = [industries[i] for i in range(len(industries)) if delta_d[i] != 0]
    dc3.metric("Commodities shocked", f"{len(active)}")
    if active:
        st.caption("Active shocks: " + ", ".join(active))

if not any_shock:
    st.info(
        "No shock applied yet. Use the **Final-demand sliders** in the sidebar "
        "(or the **APAP +15%** preset) to inject a demand increase.",
        icon="👈",
    )

st.divider()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_casc, tab_tier, tab_waste, tab_data = st.tabs(
    ["📈 Cascading impact", "🪜 Tier cascade", "♻️ Waste cascade", "🧾 Data & downloads"]
)

TOP_N_DEFAULT = 12


def top_frame(frame: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    return frame.reindex(frame[col].abs().sort_values(ascending=False).index).head(n)


# ----- Cascading impact (LOG SCALE) --------------------------------------- #
with tab_casc:
    st.subheader("Cascading output impact, Δx = L · Δd")
    st.caption("Incremental gross output forced across the network, ranked by magnitude.")
    top_n = st.slider("Show top N commodities", 3, len(industries),
                      min(TOP_N_DEFAULT, len(industries)), key="topn_casc")
    pos = casc[casc["delta_x"] != 0]
    top = top_frame(pos, "delta_x", top_n).reset_index()

    if len(top):
        chart = (
            alt.Chart(top)
            .mark_bar(color="#1F6F5C")
            .encode(
                x=alt.X("delta_x:Q",
                        title="Incremental output Δx (kg)",
                        axis=alt.Axis(format="~s")),
                y=alt.Y("industry:N", sort="-x", title=None),
                tooltip=[
                    "industry",
                    alt.Tooltip("delta_x:Q", title="Δx (kg)", format=",.0f"),
                    alt.Tooltip("pct_change:Q", title="% of baseline output", format=".3f"),
                    alt.Tooltip("x0_baseline:Q", title="baseline output (kg)", format=",.0f"),
                ],
            )
            .properties(height=28 * len(top) + 40)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("Apply a shock to see the cascade.")

    st.dataframe(
        casc.sort_values("delta_x", key=lambda s: s.abs(), ascending=False)
        .style.format({
            "x0_baseline": "{:,.0f}", "delta_d": "{:,.0f}", "delta_x": "{:,.0f}",
            "pct_change": "{:.4f}%", "x_scaled": "{:,.0f}",
        }),
        use_container_width=True,
    )

# ----- Tier cascade -------------------------------------------------------- #
with tab_tier:
    st.subheader("Supply-chain tier decomposition, Δx = Σₖ Aᵏ·Δd")
    st.caption(
        "Tier 0 = own-demand effect · Tier 1 = direct suppliers · "
        "Tier 2 = indirect · deeper indirect grouped as the last tier. "
        "Tiers sum exactly to the Leontief total."
    )
    tier_cols = [c for c in tiers.columns if c != "total"]
    top_n_t = st.slider("Show top N commodities", 3, len(industries),
                        min(8, len(industries)), key="topn_tier")
    top_t = top_frame(tiers[tiers["total"] != 0], "total", top_n_t)

    if len(top_t):
        long = (
            top_t[tier_cols]
            .reset_index()
            .melt(id_vars="industry", var_name="tier", value_name="value")
        )
        order = {c: i for i, c in enumerate(tier_cols)}
        long["order"] = long["tier"].map(order)
        pretty = {}
        for c in tier_cols:
            if c.startswith("tier0"):
                pretty[c] = "Tier 0 · own demand"
            elif c.startswith("tier1"):
                pretty[c] = "Tier 1 · direct"
            elif "plus" in c:
                pretty[c] = f"{c.split('_')[0].replace('tier','Tier ')}+ · deeper"
            else:
                pretty[c] = c.split("_")[0].replace("tier", "Tier ") + " · indirect"
        long["tier_label"] = long["tier"].map(pretty)

        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("value:Q", title="Contribution to Δx (kg)", stack="zero"),
                y=alt.Y("industry:N", sort="-x", title=None),
                color=alt.Color("tier_label:N", title="Tier",
                                sort=[pretty[c] for c in tier_cols],
                                scale=alt.Scale(scheme="viridis")),
                order=alt.Order("order:Q"),
                tooltip=["industry", "tier_label",
                         alt.Tooltip("value:Q", title="kg", format=",.0f")],
            )
            .properties(height=32 * len(top_t) + 40)
        )
        st.altair_chart(chart, use_container_width=True)

        sys_tot = tiers[tier_cols].sum()
        sys_df = pd.DataFrame({
            "tier": [pretty[c] for c in tier_cols],
            "total_kg": sys_tot.values,
        })
        sys_df["share"] = sys_df["total_kg"] / sys_df["total_kg"].sum() * 100
        st.markdown("**System-wide split by tier**")
        st.dataframe(
            sys_df.set_index("tier").style.format(
                {"total_kg": "{:,.0f}", "share": "{:.1f}%"}),
            use_container_width=True,
        )
    else:
        st.caption("Apply a shock to see the tier decomposition.")

    with st.expander("Full tier table"):
        st.dataframe(
            tiers.sort_values("total", key=lambda s: s.abs(), ascending=False)
            .style.format("{:,.0f}"),
            use_container_width=True,
        )

# ----- Waste cascade ------------------------------------------------------- #
with tab_waste:
    st.subheader("Waste cascade, ΔWⱼ = gⱼ · Δxⱼ")
    st.caption("Each commodity's output cascade weighted by its baseline waste "
               "intensity gⱼ = W₀ⱼ / x₀ⱼ (kg waste per kg output).")
    top_n_w = st.slider("Show top N commodities", 3, len(industries),
                        min(10, len(industries)), key="topn_waste")
    wpos = waste[waste["delta_waste"] > 0]
    top_w = top_frame(wpos, "delta_waste", top_n_w).reset_index()

    if len(top_w):
        chart = (
            alt.Chart(top_w)
            .mark_bar(color="#C4622D")
            .encode(
                x=alt.X("delta_waste:Q",
                        title="Incremental waste ΔW (kg)",
                        axis=alt.Axis(format="~s")),
                y=alt.Y("industry:N", sort="-x", title=None),
                tooltip=[
                    "industry",
                    alt.Tooltip("delta_waste:Q", title="ΔW (kg)", format=",.0f"),
                    alt.Tooltip("waste_intensity:Q", title="intensity (kg/kg)", format=".4f"),
                ],
            )
            .properties(height=28 * len(top_w) + 40)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("Apply a shock to see the waste cascade.")

    st.dataframe(
        waste.sort_values("delta_waste", key=lambda s: s.abs(), ascending=False)
        .style.format({
            "baseline_waste": "{:,.0f}", "waste_intensity": "{:.5f}",
            "delta_waste": "{:,.0f}", "waste_scaled": "{:,.0f}",
        }),
        use_container_width=True,
    )

# ----- Data & downloads ---------------------------------------------------- #
with tab_data:
    st.subheader("Download results")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Cascading impact (CSV)", casc.to_csv().encode(),
                       "cascading_results.csv", "text/csv", use_container_width=True)
    c2.download_button("Tier decomposition (CSV)", tiers.to_csv().encode(),
                       "tier_decomposition.csv", "text/csv", use_container_width=True)
    c3.download_button("Waste cascade (CSV)", waste.to_csv().encode(),
                       "waste_cascade.csv", "text/csv", use_container_width=True)

    st.markdown("#### Current shock vector (Δd)")
    dd_df = pd.DataFrame({"industry": industries, "delta_d_kg": delta_d})
    dd_df = dd_df[dd_df["delta_d_kg"] != 0].set_index("industry")
    if len(dd_df):
        st.dataframe(dd_df.style.format("{:,.0f}"), use_container_width=True)
    else:
        st.caption("No active shock.")

    with st.expander("Baseline PIOT (as loaded)"):
        st.dataframe(df, use_container_width=True)

    st.markdown(
        "<p class='small-note'>Method: A = Z·x̂⁻¹, "
        "L = (I−A)⁻¹, Δx = L·Δd, Δx = Σₖ Aᵏ·Δd, ΔW = g·Δx. "
        "Reproduces the NSF workshop notebook.</p>",
        unsafe_allow_html=True,
    )
