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

