# Claude TODO — ArgoEBUSAnalysis

Last updated: 2026-03-20

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

- [x] **Run RMSRE comparison: 2D-RBF vs. 3D-Matern(nu=0.5)** — DONE
  - 2D RBF:      median 4.87%, max 8.34%, 11/23 windows failing (<5% target)
  - 3D Matern05: median 3.82%, max 6.50%,  5/23 windows failing
  - Improvement: 52% → 78% of windows meet target. Clear win for 3D Exponential.
  - Remaining failures concentrated in two problem clusters (see below).

- [ ] **Reduce RMSRE in the two remaining problem clusters**

  **Cluster 1 — Early January window (day ~5835, ~50 obs only):**
  - Both models fail here (~7% RMSRE). Root cause: only 50 spatial bins vs. 136–155
    in all other windows. The GP has too little data to constrain the kernel well.
  - Options to try: (a) raise the minimum-bin threshold to skip this window entirely,
    (b) widen the first window to 45 days to pull in more early-January floats.

  **Cluster 2 — Summer windows (days ~6030–6045, ~July 2015):**
  - Both models fail here (~8.3% 2D, ~6.5% 3D). Anisotropy ratio ~0.26–0.36,
    indicating maximum eddy activity and zonal atmospheric chaos.
  - Std Z is slightly high (1.08–1.11) — the model is mildly overconfident in summer.
  - Options to try: (a) shorter window (15–20 days) to avoid bridging eddy lifecycles,
    (b) raise noise floor in the summer band, (c) accept ~6% for the eddy season as
    physically irreducible and focus budget on Source/Background layers instead.

- [ ] **Investigate 3D Matern Std Z underconfidence (min 0.74)**
  - Days 6090–6105 (late Sep / early Oct 2015): RMSRE is actually great (3.0%) but
    Std Z = 0.74, meaning the model's error bars are too large relative to actual errors.
  - This is the "paranoid" direction — safe but wasteful. May self-correct in deeper
    layers where the field is smoother. Flag for review after Source Layer run.

- [ ] **Test tighter temporal window** (15–20 days vs. 30 days)
  - Primary motivation: summer Cluster 2 windows. Shorter window means fewer
    cross-eddy observations contaminating the covariance estimate.
  - Expected tradeoff: fewer data points per window → noisier kernel fits.

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

