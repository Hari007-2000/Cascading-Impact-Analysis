"""
Cascading Impact Explorer — an interactive Streamlit front end for the
demand-driven Leontief cascading-impact model on a Physical Input-Output
Table (PIOT).

Slide the final-demand increase for any commodity and watch the cascading
output impact, the tier-by-tier decomposition, and the waste cascade update
live.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

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

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "PIOT_ModelD_APAP_workshop.csv")

SHOCK_MODES = {
    "% of baseline final demand": "pct_final_demand",
    "% of baseline gross output": "pct_gross_output",
    "Absolute increase (kg)": "absolute_kg",
}


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_default_bytes() -> bytes:
    with open(DEFAULT_CSV, "rb") as fh:
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
            "Acetaminophen (Model D) workshop table."
        ),
    )

file_bytes = uploaded.getvalue() if uploaded is not None else _load_default_bytes()

try:
    df, industries, net = load_and_build(file_bytes)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not build the network from this PIOT: {exc}")
    st.stop()

x0 = net["x0"]
d0 = net["d0"]

st.sidebar.markdown("### 2 · Shock definition")
mode_label = st.sidebar.radio(
    "How should the sliders be interpreted?",
    list(SHOCK_MODES.keys()),
    index=1,
    help=(
        "In this PIOT only Acetaminophen has a non-zero baseline *final demand*, "
        "so '% of baseline final demand' reproduces the original notebook but only "
        "moves when you shock Acetaminophen. '% of baseline gross output' or "
        "'Absolute increase (kg)' let you shock any commodity."
    ),
)
mode = SHOCK_MODES[mode_label]

n_tiers = st.sidebar.slider(
    "Explicit supply-chain tiers",
    min_value=2,
    max_value=6,
    value=3,
    help="Tiers 0…(n-1) are shown explicitly; everything deeper is grouped as 'tier n+'.",
)

# Slider state is namespaced by mode so switching modes can never leave a
# stored value outside the new mode's range.
def skey(name: str) -> str:
    return f"shock_{mode}_{name}"


colq1, colq2 = st.sidebar.columns(2)
if colq1.button("Reset sliders", use_container_width=True):
    for name in industries:
        st.session_state[skey(name)] = 0.0
if colq2.button("APAP +15%", use_container_width=True,
                help="Quick preset: 15% increase, matching the notebook example."):
    for name in industries:
        st.session_state[skey(name)] = 0.0
    idx = industries.index("Acetaminophen") if "Acetaminophen" in industries else 0
    apap = industries[idx]
    if mode == "pct_final_demand":
        st.session_state[skey(apap)] = 15.0
    elif mode == "pct_gross_output":
        # 15% of APAP final demand expressed against gross output
        st.session_state[skey(apap)] = float(100.0 * d0[idx] * 0.15 / x0[idx])
    else:
        st.session_state[skey(apap)] = float(d0[idx] * 0.15)

st.sidebar.markdown("### 3 · Final-demand sliders")
st.sidebar.caption(
    "Drag a commodity to inject a final-demand increase. "
    "The cascade recomputes instantly."
)

# Build per-commodity sliders. Ranges adapt to the chosen mode. The widget
# value is driven purely by session_state (seeded with setdefault) rather than
# a `value=` argument, which avoids the "default value + session state" warning
# and lets the Reset / preset buttons work cleanly.
shocks: dict[str, float] = {}
for i, name in enumerate(industries):
    key = skey(name)
    st.session_state.setdefault(key, 0.0)
    if mode == "pct_final_demand":
        has_fd = d0[i] > 0
        label = f"{name}" + ("" if has_fd else "  (no baseline FD)")
        if not has_fd:
            st.session_state[key] = 0.0
        val = st.sidebar.slider(
            label, min_value=0.0, max_value=200.0,
            step=1.0, key=key, format="%.0f%%",
            disabled=not has_fd,
        )
    elif mode == "pct_gross_output":
        val = st.sidebar.slider(
            name, min_value=0.0, max_value=100.0,
            step=0.5, key=key, format="%.1f%%",
        )
    else:  # absolute_kg
        # cap the slider at the industry's own baseline gross output for a sane range
        cap = max(float(x0[i]), 1.0)
        # keep any seeded value within the current cap
        if st.session_state[key] > cap:
            st.session_state[key] = cap
        val = st.sidebar.slider(
            f"{name}  (max {fmt_kg(cap)})",
            min_value=0.0, max_value=cap,
            step=cap / 200.0, key=key,
        )
    shocks[name] = val

# --------------------------------------------------------------------------- #
# Compute everything
# --------------------------------------------------------------------------- #
delta_d = m.build_delta_d(net, shocks, mode=mode)
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
tab_casc, tab_tier, tab_waste, tab_struct, tab_data = st.tabs(
    ["📈 Cascading impact", "🪜 Tier cascade", "♻️ Waste cascade",
     "🧭 Structural multipliers", "🧾 Data & downloads"]
)

TOP_N_DEFAULT = 12


def top_frame(frame: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    return frame.reindex(frame[col].abs().sort_values(ascending=False).index).head(n)


# ----- Cascading impact ---------------------------------------------------- #
with tab_casc:
    st.subheader("Cascading output impact, Δx = L · Δd")
    top_n = st.slider("Show top N commodities", 3, len(industries),
                      min(TOP_N_DEFAULT, len(industries)), key="topn_casc")
    top = top_frame(casc[casc["delta_x"] != 0], "delta_x", top_n).reset_index()

    if len(top):
        chart = (
            alt.Chart(top)
            .mark_bar(color="#1F6F5C")
            .encode(
                x=alt.X("delta_x:Q", title="Incremental output Δx (kg)"),
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
        # nice tier ordering & labels
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

        # System-wide tier totals
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
                x=alt.X("delta_waste:Q", title="Incremental waste ΔW (kg)"),
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

# ----- Structural multipliers --------------------------------------------- #
with tab_struct:
    st.subheader("Scale-independent structural multipliers")
    st.caption(
        "From a unit final-demand shock — independent of the sliders. "
        "Backward linkage = total output pulled per unit of final demand "
        "(column sum of L)."
    )
    mult = m.all_industry_multipliers(net).reset_index()
    chart = (
        alt.Chart(mult)
        .mark_bar(color="#4C72B0")
        .encode(
            x=alt.X("backward_linkage_raw:Q", title="Total output multiplier (backward linkage)"),
            y=alt.Y("industry:N", sort="-x", title=None),
            tooltip=[
                "industry",
                alt.Tooltip("backward_linkage_raw:Q", title="backward linkage", format=".3f"),
                alt.Tooltip("forward_sensitivity_norm:Q", title="forward (norm)", format=".3f"),
                alt.Tooltip("self_loop_coef:Q", title="self-loop a_jj", format=".4f"),
            ],
        )
        .properties(height=28 * len(mult) + 40)
    )
    rule = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(
        color="#333", strokeDash=[4, 3]).encode(x="x:Q")
    st.altair_chart(chart + rule, use_container_width=True)
    st.dataframe(
        mult.set_index("industry").style.format({
            "backward_linkage_raw": "{:.4f}", "backward_linkage_norm": "{:.3f}",
            "forward_sensitivity_norm": "{:.3f}", "self_loop_coef": "{:.4f}",
        }),
        use_container_width=True,
    )

# ----- Data & downloads ---------------------------------------------------- #
with tab_data:
    st.subheader("Download results")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Cascading impact (CSV)", casc.to_csv().encode(),
                       "cascading_results.csv", "text/csv", use_container_width=True)
    c2.download_button("Tier decomposition (CSV)", tiers.to_csv().encode(),
                       "tier_decomposition.csv", "text/csv", use_container_width=True)
    c3.download_button("Waste cascade (CSV)", waste.to_csv().encode(),
                       "waste_cascade.csv", "text/csv", use_container_width=True)
    c4.download_button("Structural multipliers (CSV)",
                       m.all_industry_multipliers(net).to_csv().encode(),
                       "all_industry_multipliers.csv", "text/csv", use_container_width=True)

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
