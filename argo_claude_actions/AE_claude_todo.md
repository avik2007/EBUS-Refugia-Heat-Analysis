# Claude TODO — ArgoEBUSAnalysis

Last updated: 2026-03-27

Human Notes:
 - I want to work out the moving window, in such a way that I can study
   time correlations and spatial correlations. This will hopefully allow
   me to bring down the RMSRE.
 - Both RBF and Matern(nu=0.5) should be usable as a parameter (kernel_type='rbf'
   or kernel_type='matern0.5'). The goal is modular comparison: same call, different
   kernel, compare results_df directly. Do NOT implement Matern(nu=1.5).
 - To that end, analyze_rolling_correlations has been extended with mode='3D' and
   kernel_type parameters. See AE_plan_3d_gpr_matern.md for the full design.

---

## Priority 1: RMSRE Optimization

- [ ] **Adopt C2 config (window=45) as the new standard for Skin Layer**
  - Update `03_ae_inspect_data.py` to use `window_size_days=45, time_ls_bounds_days=(2.0, 45.0)`
  - Re-run Script 03 to regenerate the canonical Skin Layer audit CSV with the new config
  - Update `ae_file_structure.txt` and any hardcoded window references

- [ ] **Investigate 3D Matern Std Z underconfidence (min 0.74)**
  - Days 6090–6105 (late Sep / early Oct 2015): RMSRE is actually great (3.0%) but
    Std Z = 0.74, meaning the model's error bars are too large relative to actual errors.
  - Confirmed persistent: C2 (window=45) shows the same 0.74 at the same windows.
    Widening the window does not resolve the underconfidence.
  - Defer to Source Layer run — if Std Z is still < 0.85 there, investigate the
    auto-calibration noise adjustment bounds as the likely cause.

---

## Priority 2: Cloud Runs (Require AWS)

- [ ] **Source Layer cloud run** — Script 02 with `depth_range=(150, 400)`
  - This is the Ekman upwelling source water layer
  - Expect: Anisotropy Ratio to increase (ratio > Skin Layer) — deeper = less wind shredding
  - Then run Script 03 on the result

- [ ] **Background Layer cloud run** — Script 02 with `depth_range=(500, 1000)`
  - This is the deep ocean control/baseline
  - Needed to establish the "background" warming rate for comparison

---

## Priority 3: Analysis and Comparison

- [ ] **Vertical Delta Comparison Script** (new script, e.g., `04_ae_vertical_compare.py`)
  - Load audit CSVs from all three depth layers
  - Plot: OHC trend for each layer on same axes
  - Plot: Anisotropy Ratio by depth layer over time
  - Key question: Is Source Layer (150–400m) warming faster than Background (500–1000m)?

- [ ] **Seasonal Anisotropy Report**
  - From the Skin Layer audit, compare Jan vs. Aug Anisotropy Ratios
  - Already partially done: confirmed ratio ~0.36 Jan, ~0.49 Aug from 2015 logs
  - Formalize this into a plot showing ratio vs. month for a full year

