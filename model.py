"""
Core Leontief cascading-impact model for a Physical Input-Output Table (PIOT).

This module reproduces the methodology from the NSF workshop notebook
`piot_cascading_impact_nsf_workshop_final.py`:

    A = Z x_hat^-1            (physical technical-coefficient matrix)
    L = (I - A)^-1            (Leontief inverse)
    delta_x = L @ delta_d     (cascading output impact of a final-demand shock)
    delta_x = sum_k A^k delta_d   (tier decomposition, k = 0,1,2,...)
    delta_W_j = g_j * delta_x_j    (waste cascade, g_j = W0_j / x0_j)

It is deliberately free of any Streamlit / plotting code so it can be reused,
unit-tested, and imported by the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Meta columns that are NOT industries (they live to the right of the Z block).
META_COLS = ["ROE", "EXPORTS", "FINAL_DEMAND", "WASTE"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_piot(source):
    """
    Load a raw PIOT CSV.

    `source` may be a path string or any file-like object (e.g. a Streamlit
    UploadedFile). Industries are every column that is not a meta column and
    not an unnamed/spare column.

    Returns
    -------
    (df, industries) : (pandas.DataFrame, list[str])
    """
    df = pd.read_csv(source, index_col=0).fillna(0.0)
    industries = [
        c for c in df.columns
        if c not in META_COLS and not str(c).startswith("Unnamed:")
    ]
    return df, industries


# --------------------------------------------------------------------------- #
# Network construction
# --------------------------------------------------------------------------- #
def build_network(df: pd.DataFrame, industries: list[str],
                  remove_self_loops: bool = False) -> dict:
    """
    Build Z, x0, A and L from a loaded PIOT dataframe.

    x_j (gross output of industry j) is the row sum across all industry cells
    plus the meta columns ROE + EXPORTS + FINAL_DEMAND + WASTE.

    Raises
    ------
    ValueError
        If the spectral radius of A is >= 1 (network not productive, so the
        Leontief inverse does not exist).
    """
    Z = df.loc[industries, industries].values.astype(float)

    x0 = df.loc[industries, industries + META_COLS].sum(axis=1).values.astype(float)
    x0 = np.where(x0 == 0, 1e-9, x0)

    A = Z / x0[np.newaxis, :]
    self_loop_coef = np.diag(A).copy()  # retained as a diagnostic

    if remove_self_loops:
        np.fill_diagonal(A, 0.0)

    n = len(industries)
    identity = np.eye(n)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if spectral_radius >= 1.0:
        raise ValueError(
            f"Spectral radius of A is {spectral_radius:.4f} >= 1: the Leontief "
            "inverse does not exist (network not productive). This usually means "
            "a cell is a genuine data error rather than real recycling."
        )
    L = np.linalg.inv(identity - A)

    d0 = df.loc[industries, "FINAL_DEMAND"].values.astype(float)
    w0 = df.loc[industries, "WASTE"].values.astype(float)

    return {
        "Z": Z,
        "x0": x0,
        "A": A,
        "L": L,
        "d0": d0,
        "w0": w0,
        "industries": industries,
        "spectral_radius": spectral_radius,
        "self_loop_coef": self_loop_coef,
    }


# --------------------------------------------------------------------------- #
# Final-demand shock vector
# --------------------------------------------------------------------------- #
def build_delta_d(net: dict, shocks: dict, mode: str = "pct_final_demand") -> np.ndarray:
    """
    Turn a {industry_name: value} dict of user shocks into a delta_d vector.

    mode:
      "pct_final_demand" -> value is a % increase in that industry's baseline
                            FINAL DEMAND.  delta_d_i = d0_i * (value/100)
      "pct_gross_output" -> value is a % of that industry's baseline GROSS
                            OUTPUT.        delta_d_i = x0_i * (value/100)
      "absolute_kg"      -> value is an absolute increase in kg.
                            delta_d_i = value
    """
    industries = net["industries"]
    delta_d = np.zeros(len(industries), dtype=float)
    for i, name in enumerate(industries):
        v = float(shocks.get(name, 0.0) or 0.0)
        if v == 0.0:
            continue
        if mode == "pct_final_demand":
            delta_d[i] = net["d0"][i] * (v / 100.0)
        elif mode == "pct_gross_output":
            delta_d[i] = net["x0"][i] * (v / 100.0)
        elif mode == "absolute_kg":
            delta_d[i] = v
        else:
            raise ValueError(f"Unknown shock mode: {mode}")
    return delta_d


# --------------------------------------------------------------------------- #
# Cascading impact
# --------------------------------------------------------------------------- #
def cascading_impact(net: dict, delta_d: np.ndarray) -> pd.DataFrame:
    """delta_x = L @ delta_d, plus baseline output and % change per industry."""
    industries = net["industries"]
    delta_x = net["L"] @ delta_d
    out = pd.DataFrame(
        {
            "industry": industries,
            "x0_baseline": net["x0"],
            "delta_d": delta_d,
            "delta_x": delta_x,
        }
    ).set_index("industry")
    out["pct_change"] = 100.0 * out["delta_x"] / out["x0_baseline"]
    out["x_scaled"] = out["x0_baseline"] + out["delta_x"]
    return out


# --------------------------------------------------------------------------- #
# Tier decomposition
# --------------------------------------------------------------------------- #
def tier_decomposition(net: dict, delta_d: np.ndarray, n_tiers: int = 3) -> pd.DataFrame:
    """
    Split delta_x by supply-chain depth:

        t0 = delta_d             (own-demand effect)
        t1 = A delta_d           (direct suppliers)
        t2 = A^2 delta_d         (indirect)
        ...
        t_{n}+ = delta_x - sum(t0..t_{n-1})   (everything deeper, as a remainder)

    n_tiers is the number of *explicit* tiers beyond tier 0 (default 3, i.e.
    columns tier0, tier1, tier2, tier3plus to match the original notebook).
    """
    A, L = net["A"], net["L"]
    industries = net["industries"]
    delta_x = L @ delta_d

    cols = {}
    running = np.zeros_like(delta_d)
    term = delta_d.copy()  # A^0 delta_d
    cols["tier0_own_demand"] = term.copy()
    running += term
    for k in range(1, n_tiers):
        term = A @ term  # A^k delta_d
        cols[f"tier{k}_{'direct' if k == 1 else 'indirect'}"] = term.copy()
        running += term
    cols[f"tier{n_tiers}plus_indirect"] = delta_x - running

    frame = pd.DataFrame({"industry": industries, **cols}).set_index("industry")
    frame["total"] = delta_x
    return frame


# --------------------------------------------------------------------------- #
# Waste cascade
# --------------------------------------------------------------------------- #
def waste_cascade(net: dict, delta_x: np.ndarray) -> pd.DataFrame:
    """
    Weight the output cascade by each industry's baseline waste intensity
    g_j = W0_j / x0_j  ->  delta_W_j = g_j * delta_x_j.
    """
    industries = net["industries"]
    x0, w0 = net["x0"], net["w0"]
    waste_intensity = np.divide(w0, x0, out=np.zeros_like(w0), where=x0 > 0)
    delta_waste = delta_x * waste_intensity
    frame = pd.DataFrame(
        {
            "industry": industries,
            "baseline_waste": w0,
            "waste_intensity": waste_intensity,
            "delta_waste": delta_waste,
        }
    ).set_index("industry")
    frame["waste_scaled"] = frame["baseline_waste"] + frame["delta_waste"]
    return frame


# --------------------------------------------------------------------------- #
# Structural multipliers (unit shock, scale-independent)
# --------------------------------------------------------------------------- #
def all_industry_multipliers(net: dict) -> pd.DataFrame:
    """Total output multiplier (backward linkage) etc. from a unit demand shock."""
    L = net["L"]
    industries = net["industries"]
    backward_raw = L.sum(axis=0)
    forward_raw = L.sum(axis=1)
    return pd.DataFrame(
        {
            "industry": industries,
            "backward_linkage_raw": backward_raw,
            "backward_linkage_norm": backward_raw / backward_raw.mean(),
            "forward_sensitivity_norm": forward_raw / forward_raw.mean(),
            "self_loop_coef": net["self_loop_coef"],
        }
    ).set_index("industry").sort_values("backward_linkage_raw", ascending=False)
