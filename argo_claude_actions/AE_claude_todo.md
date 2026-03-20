# Claude TODO — ArgoEBUSAnalysis

Last updated: 2026-03-20

---

## Priority 1: RMSRE Optimization (If Needed)

- [ ] **Evaluate Matérn 3/2 kernel** as alternative to Squared Exponential (RBF)
  - Matérn handles mesoscale eddies better in heterogeneous summer ocean
  - Compare RMSRE between RBF and Matérn 3/2 in the rolling analysis
  - Only switch if it moves RMSRE from ~8% toward target of <5%
  - User preference: avoid nonstationary noise kernel

- [ ] **Test tighter temporal window** (20 days vs. 30 days)
  - Motivation: reduce RMSRE by ensuring GP fits same physics, not bridging seasonal transitions
  - Expected tradeoff: fewer data points per window → lower data density

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

---

## Completed Tasks

- [x] Cloud pipeline working (Scripts 01–03)
- [x] 2015 Skin Layer (0–100m) full year processed
- [x] Kriging storyboard PNGs saved (23 snapshots)
- [x] Audit CSV and CV pickle saved to `AEResults/aelogs/`
- [x] Anisotropy Ratio tracking added to `analyze_rolling_correlations()`
- [x] Visual inspection of summer vs. winter snapshots confirmed "eddy collapse"
- [x] CLAUDE.md created
- [x] Claude reports folder initialized
- [x] `plot_physics_history()` run on 2015 Skin Layer — `03b_ae_plot_physics.py` exists
- [x] `get_float_history()` added to `ae_utils.py` — raw per-dive ERDDAP data access layer
- [x] `03_ae_plot_float_paths.py` created — float trajectory spaghetti map diagnostic
