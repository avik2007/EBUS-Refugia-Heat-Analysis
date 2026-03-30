# Claude TODO — ArgoEBUSAnalysis

Last updated: 2026-03-30

**[Gemini Diagnosis]:** The time persistence oscillation in the Skin Layer is likely a **sampling aliasing effect**. The 10-day Argo float cycle and 15-day window step create a 30-day (2-window) repetition in the sampling time distribution relative to the window center. In the low-coherence Skin Layer, windows with larger gaps between float surfacings and the center lose temporal information, causing the GP to "pin" to the upper bound.

Human Notes:
 - **[For Gemini]** The 3D Matern(nu=0.5) GP is showing a striking alternating pattern
   in its time length scale (scale_time_days): every other rolling window pins hard to
   the upper bound (45d or 30d depending on the bound set), while the intervening windows
   find a free value. This is not a bounds issue — tightening from tb=45 to tb=30 just
   moves the ceiling without fixing the alternation. The windows themselves are overlapping
   15-day steps through 2015 CCS data. Is there a physical or statistical explanation for
   why alternate windows would have fundamentally different temporal correlation structure?
   Could this reflect a real oceanographic signal (e.g., spring/neap tidal aliasing, eddy
   phase alternation), or is it a symptom of the 3D GP mode being ill-suited to the Skin
   Layer where atmospheric forcing dominates and temporal coherence is low?

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


- [ ] **Time persistence oscillation — pending Gemini input (see Human Notes)**
  - C5 (w45, tb=30) confirmed the alternation is structural, not a bounds artifact.
    Tightening tb=45 → tb=30 moved the ceiling but the every-other-window pin persists.
  - C2 (w45, tb=45, 83% pass) accepted as canonical pending Gemini's diagnosis.
  - Do not pursue further bounds adjustments without a physical hypothesis.

---

## Priority 2: Cloud Runs (Require AWS)

- [ ] **Source Layer cloud run** — Script 02 with `depth_range=(150, 400)`
  - This is the Ekman upwelling source water layer
  - Expect: Anisotropy Ratio to increase (ratio > Skin Layer) — deeper = less wind shredding
  - Then run Script 03 on the result

- [ ] **Background Layer cloud run** — Script 02 with `depth_range=(500, 1000)`
  - This is the deep ocean control/baseline
  - Needed to establish the "background" warming rate for comparison

- [ ] **Investigate 3D Matern Std Z underconfidence (min 0.74)**
  - Days 6090–6105 (late Sep / early Oct 2015): RMSRE is actually great (3.0%) but
    Std Z = 0.74, meaning the model's error bars are too large relative to actual errors.
  - Confirmed persistent: C2 (window=45) shows the same 0.74 at the same windows.
  - RMSRE is small in all affected windows — this is a conservative/safe failure mode,
    not a false-positive risk. Defer until after Source Layer analysis.
  - If Std Z is still < 0.85 in the Source Layer, investigate auto-calibration noise
    adjustment bounds as the likely cause.

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

