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
# Bundled default final-demand file, used when an uploaded PIOT has no
# final-demand column of its own.
_FD_FALLBACK_CSV = os.path.join(_HERE, "APAP_Exports_16.csv")


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_default_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@st.cache_data(show_spinner=False)
def _load_fd_fallback():
    """Load the bundled default final-demand table, or None if absent."""
    if os.path.exists(_FD_FALLBACK_CSV):
        try:
            return m.load_fd_fallback(_FD_FALLBACK_CSV)
        except Exception:  # noqa: BLE001
            return None
    return None


@st.cache_data(show_spinner=False)
def load_and_build(file_bytes: bytes):
    """Load a PIOT from raw bytes and build the network. Cached on the bytes."""
    df, industries = m.load_piot(io.BytesIO(file_bytes))
    net = m.build_network(df, industries, fd_fallback=_load_fd_fallback())
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


def fmt_value(v: float) -> str:
    """Compact numeric formatting for indicator values (unit shown separately)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    a = abs(v)
    for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")]:
        if a >= div:
            return f"{v / div:,.2f}{suf}"
    if a >= 1 or v == 0:
        return f"{v:,.2f}"
    return f"{v:,.4g}"


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
            "Any PIOT works: industries just need to appear as both rows and "
            "columns (the square inter-industry block). The network is built "
            "from that Z-matrix alone — ROE / Imports / Exports / Final-demand / "
            "Waste columns are optional. Leave empty to use the bundled "
            "Acetaminophen PIOT."
        ),
    )
    st.caption(f"Bundled PIOT: **{os.path.basename(DEFAULT_CSV)}**")
    if os.path.exists(_FD_FALLBACK_CSV):
        st.caption("Default final demand (for PIOTs without a final-demand "
                   f"column): **{os.path.basename(_FD_FALLBACK_CSV)}**")

file_bytes = uploaded.getvalue() if uploaded is not None else _load_default_bytes(DEFAULT_CSV)

try:
    df, industries, net = load_and_build(file_bytes)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not build the network from this PIOT: {exc}")
    st.stop()

x0 = np.asarray(net["x0"], dtype=float)
d0 = np.asarray(net["d0"], dtype=float)

# Per-commodity reference used to size each slider's injection. Where a commodity
# has a positive baseline final demand we shock a % of THAT (notebook-faithful);
# where it has none, we shock a % of its gross output instead, so every slider is
# always usable and the user can always increase final demand.
shock_base = np.where(d0 > 0, d0, x0)
uses_output = d0 <= 0  # which commodities fall back to the gross-output basis

st.sidebar.markdown("### 2 · Shock definition")
st.sidebar.caption(
    "Each slider injects extra **final demand** for a commodity. For a commodity "
    "with a baseline final demand the slider is a **% of that baseline**; for one "
    "with none it is a **% of its gross output** (marked *× output*). "
    "Δdᵢ = baseᵢ × (slider ÷ 100)."
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

# One % slider per commodity — always enabled.
shocks: dict[str, float] = {}
for i, name in enumerate(industries):
    key = f"shock_{name}"
    st.session_state.setdefault(key, 0.0)
    label = f"{name}" + ("  (× output)" if uses_output[i] else "")
    val = st.sidebar.slider(
        label, min_value=0.0, max_value=200.0,
        step=1.0, key=key, format="%.0f%%",
    )
    shocks[name] = val

# --------------------------------------------------------------------------- #
# Compute everything
# --------------------------------------------------------------------------- #
# Δdᵢ = shock_baseᵢ × (slider% ÷ 100), where shock_base is the baseline final
# demand when positive, else gross output. This never depends on a non-zero FD.
delta_d = np.array(
    [shock_base[i] * (float(shocks[name]) / 100.0)
     for i, name in enumerate(industries)],
    dtype=float,
)
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
    st.caption(f"Gross output x from column totals (Z-matrix). "
               f"Baseline final demand: {net.get('d0_source','')}. "
               f"Waste: {net.get('w0_source','')}.")

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
tab_about, tab_casc, tab_tier, tab_waste, tab_data = st.tabs(
    ["📖 Overview", "📈 Cascading impact", "🪜 Tier cascade",
     "♻️ Waste cascade", "🧾 Data & downloads"]
)

# --------------------------------------------------------------------------- #
# Overview / About (first page)
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("Why cascading-impact analysis?")
    st.markdown(
        "A modern chemical is made in **tiers**. Natural resources feed the "
        "upstream bulk-chemical plants (Tier-4), whose outputs feed the "
        "intermediates (Tier-3, Tier-2), which feed the key starting materials "
        "(Tier-1), which finally feed the active pharmaceutical ingredient "
        "(Tier-0) — here **Acetaminophen (APAP)**. Because every tier draws on "
        "the ones above it, a change in demand for the *final* product does not "
        "stay local: it **cascades upstream**, pulling extra output — and extra "
        "waste and resource use — from suppliers, their suppliers, and so on."
    )

    _fig = os.path.join(_HERE, "piot_tiers.png")
    if os.path.exists(_fig):
        lc, mc, rc = st.columns([1, 8, 1])
        mc.image(_fig, use_container_width=True,
                 caption="Tiered supply chain of the Acetaminophen manufacturing network "
                         "(Model D), from natural resources down to the API.")

    st.markdown(
        "Looking at one plant in isolation badly **under-counts** its true "
        "footprint, because most of the material and waste is generated upstream. "
        "Cascading-impact analysis makes that hidden burden visible: it answers "
        "*if final demand for a commodity rises by X%, how much total output does "
        "the whole network have to produce, where does that fall across the "
        "supply-chain tiers, and how much extra waste comes with it?* That is "
        "exactly what the tabs to the right let you explore interactively."
    )

    st.divider()
    st.subheader("The basic maths")
    st.markdown(
        "The model is the classic **Leontief demand-pull**, applied to a "
        "**Physical** Input–Output Table (flows in kg/yr instead of currency)."
    )

    st.markdown("**1 · Technical-coefficient matrix.** Normalise each "
                "inter-industry flow $Z_{ij}$ by the receiving industry's gross "
                "output $x_j$:")
    st.latex(r"A = Z\,\hat{x}^{-1} \qquad\Longleftrightarrow\qquad "
             r"a_{ij} = \frac{Z_{ij}}{x_j}")
    st.caption("$a_{ij}$ = kg of $i$ needed per kg of $j$ produced. Gross output "
               "$x_j$ is read straight down each industry column (total input = "
               "total output), so **only the Z-matrix is required** — the Imports, "
               "Exports and Final-demand columns are optional.")

    st.markdown("**2 · Leontief inverse.** Sum the direct requirement, its "
                "own upstream requirement, and so on to infinity:")
    st.latex(r"L = (I - A)^{-1} = I + A + A^{2} + A^{3} + \cdots")
    st.caption("$l_{ij}$ = total output of $i$ (direct **plus** all indirect "
               "rounds) needed per unit of final demand for $j$.")

    st.markdown("**3 · Cascading output.** A final-demand shock $\\Delta d$ "
                "propagates to a total output change:")
    st.latex(r"\Delta x = L\,\Delta d")

    st.markdown("**4 · Tier decomposition.** The same total, split by how many "
                "supply-chain steps upstream each part sits:")
    st.latex(r"\Delta x = \underbrace{\Delta d}_{\text{tier }0} "
             r"+ \underbrace{A\,\Delta d}_{\text{tier }1} "
             r"+ \underbrace{A^{2}\,\Delta d}_{\text{tier }2} + \cdots")

    st.markdown("**5 · Waste cascade.** Weight the extra output by each "
                "industry's baseline waste intensity $g_j = W_j / x_j$:")
    st.latex(r"\Delta W_j = g_j\,\Delta x_j")

    st.info("Everything is built generically: the industries are auto-detected "
            "as the square inter-industry block of whatever PIOT you upload, so "
            "the same tool runs on any network — not just Acetaminophen.",
            icon="🧩")

TOP_N_DEFAULT = 12


def top_frame(frame: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    return frame.reindex(frame[col].abs().sort_values(ascending=False).index).head(n)


def top_n_slider(label: str, key: str, default: int = TOP_N_DEFAULT) -> int:
    """A 'show top N' slider that is safe for tiny networks.

    Streamlit rejects a slider whose min == max, so when there are 3 or fewer
    industries we simply show them all instead of drawing a slider.
    """
    n = len(industries)
    if n <= 3:
        st.caption(f"Showing all {n} commodities.")
        return n
    return st.slider(label, 3, n, min(default, n), key=key)


# ----- Cascading impact --------------------------------------------------- #
with tab_casc:
    st.subheader("Cascading output impact, Δx = L · Δd")
    st.caption("Incremental gross output forced across the network, ranked by magnitude.")
    top_n = top_n_slider("Show top N commodities", key="topn_casc")
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
    top_n_t = top_n_slider("Show top N commodities", key="topn_tier", default=8)
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
    if not net.get("has_waste", True):
        st.info("This PIOT has no Waste column, so waste intensities are zero and "
                "the waste cascade is unavailable. Add a 'WASTE' column to enable it.",
                icon="♻️")
    top_n_w = top_n_slider("Show top N commodities", key="topn_waste", default=10)
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
