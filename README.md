# Cascading Impact Explorer

An interactive **Streamlit** app for the demand-driven Leontief cascading-impact
model on a Physical Input–Output Table (PIOT). Slide the final-demand increase
for any commodity and see, live:

- **Cascading output impact** — `Δx = L · Δd`
- **Tier-level cascade** — `Δx = Σₖ Aᵏ · Δd` (own demand → direct suppliers → indirect → deeper)
- **Waste cascade** — `ΔWⱼ = gⱼ · Δxⱼ`, where `gⱼ = W₀ⱼ / x₀ⱼ`
- **Structural multipliers** — scale-independent backward/forward linkages from a unit shock

It reproduces the methodology in `piot_cascading_impact_nsf_workshop_final.py`
(the NSF workshop notebook) and ships with the bundled Acetaminophen "Model D"
PIOT. Verified: a 15% Acetaminophen final-demand shock gives total Δx ≈
1,870,223 kg and total ΔW ≈ 825,881 kg — identical to the notebook.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (sliders, charts, tabs, downloads) |
| `model.py` | Pure NumPy/pandas model — load, build A & L, shocks, tiers, waste. No UI, easy to unit-test. |
| `PIOT_ModelD_APAP_workshop.csv` | Bundled baseline PIOT (16 commodities) |
| `requirements.txt` | Python dependencies |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Using the app

**Shock modes** (sidebar → *Shock definition*) — the sliders can be read three ways:

1. **% of baseline final demand** — reproduces the original notebook exactly.
   Note: in this PIOT only *Acetaminophen* has a non-zero baseline final demand,
   so in this mode only its slider moves the cascade.
2. **% of baseline gross output** *(default)* — lets you shock **any** of the 16
   commodities, since every commodity has a gross output.
3. **Absolute increase (kg)** — inject a raw kg increase in final demand per commodity.

You can shock **several commodities at once** — the shocks combine into a single
`Δd` vector. Use **Reset sliders** to clear, or **APAP +15%** for the notebook
example. The number of explicit supply-chain tiers is adjustable (2–6).

**Tabs**

- **📈 Cascading impact** — Δx per commodity, % of baseline output, scaled output.
- **🪜 Tier cascade** — stacked contribution of each tier per commodity, plus the
  system-wide split. Tiers sum exactly to the Leontief total.
- **♻️ Waste cascade** — incremental waste per commodity from baseline waste intensity.
- **🧭 Structural multipliers** — backward-linkage multipliers (unit shock, slider-independent).
- **🧾 Data & downloads** — export every result table as CSV, inspect the loaded PIOT.

## Bring your own PIOT

Sidebar → *Data source* → upload a CSV shaped like the bundled one:
industries as both rows and columns, plus meta columns `ROE`, `EXPORTS`,
`FINAL_DEMAND`, `WASTE`. Gross output `xⱼ` is the row sum across all industry
cells plus those meta columns. The app rebuilds A, L, tiers and waste, and warns
if the spectral radius of A ≥ 1 (network not productive → no Leontief inverse).

## Method in one line

`A = Z·x̂⁻¹` → `L = (I − A)⁻¹` → `Δx = L·Δd` → tiers `Aᵏ·Δd` → waste `g·Δx`.
