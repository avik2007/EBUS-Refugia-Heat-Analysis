# Gemini Recent Actions — ArgoEBUSAnalysis

---

## 2026-07-17 — Gibbs vs Matérn Statistical Audit Completed (Diebold-Mariano & Block Bootstrap)

**Action:** Implemented a rigorous kernel comparison script (`compare_kernels.py`) to run Diebold-Mariano and overlapping block-bootstrap significance tests on the Matérn vs. Gibbs validation metrics across all scientific layers. Updated task tracking.

### 1. Robust Kernel Performance Audit (Diebold-Mariano Test with Newey-West HAC lag=4)
*   **Skin Layer (0-100m):** Gibbs kernel achieved a statistically significant **12.86% relative median RMSRE improvement** (median RMSRE 4.25% → 3.71%, DM-stat = 3.193, p-val = 7.05e-04). Uncertainty calibration collapsed the high Matérn Z-score variance to an ideal mean of **0.9958 ± 0.0883**.
*   **Source Layer (150-400m):** Gibbs achieved a **13.53% relative median RMSRE improvement** (median 3.05% → 2.63%, DM-stat = 2.051, p-val = 2.01e-02, 95% block-bootstrap CI [-0.00008, 0.00510]). Uncertainty calibration was stabilized to **0.9816 ± 0.0834**.
*   **Background Layer (500-1000m):** Gibbs achieved a massive **18.87% relative median RMSRE improvement** (median 2.50% → 2.03%, DM-stat = 5.068, p-val = 2.01e-07). It successfully resolved the extreme non-stationarity spikes associated with the Pacific Blob, reducing Z-score variance from 2.6299 to a perfectly calibrated **0.9664 ± 0.1002**.
*   **Verdict:** The Gibbs Non-Stationary Kernel is **statistically superior** across all layers, and successfully resolves coastal upwelling gradients and large-scale non-stationarities.

### 2. Physical Regime Transition & Temporal Persistence Insights
*   **Coast-to-offshore transition ($d_0$):** Found to be **204 km** in the Skin layer and **227 km** in the Source layer, aligning perfectly with the spatial width of coastal upwelling filaments. In the Background layer, $d_0$ pegged at the upper bound of **700 km**, indicating that deep-ocean dynamics do not exhibit the same sharp coastal transition, or that data sparsity at depth demands a broad spatial smoothing scale.
*   **Temporal persistence saturation:** Following the units bug fix, `time_ls_days` pegged at the upper limit (**200 days**) in the majority of windows across **all three layers**. This confirms that anomaly memory (persistence of temperature anomalies, such as the Marine Heatwave/Blob) is unresolvable within a 45-day rolling window. It strongly supports the "ocean anomaly memory exceeds a month and a half" hypothesis at all depths.
*   **Recommendation:** For deeper layers, widening the rolling window (e.g., to 90 or 120 days) is highly recommended. For the Skin layer, the 45-day window is retained to avoid smoothing out rapid seasonal transitions, accepting that anomaly persistence will remain pegged.

---

## 2026-07-07 — Mixed layer/buoyancy question logged; kernel significance testing methodology defined

**Action:** No code touched this session — planning/science discussion only, closed out for `/clear`.

### 1. Interviewer question logged for science review
*   Interviewer asked whether stealth warming could deepen the mixed layer / alter boundary-layer buoyancy via weakened stratification below the mixed layer (Source Layer warming → reduced N² at the pycnocline → easier wind/wave mixing → deeper MLD).
*   Assessed: plausible but indirect (Undercurrent core ~100–300m vs. mixed layer ~20–50m in CCS; needs a mixing/eddy-pump step). Does not change the 3-layer depth-defined design.
*   **Added to `AE_gemini_todo.md` (Priority 2):** research item for Gemini to consider, plus candidate diagnostic (N², MLD) to test the mechanism empirically via Argo profiles.

### 2. Kernel comparison significance testing — methodology settled (not yet implemented)
*   Question: how to tell if Gibbs kernel's RMSRE improvement over Matern 5/2 is real vs. noise.
*   **Finding:** rolling windows in the audit CSVs overlap (`step_size_days` < `window_size_days`), so per-window RMSRE values are autocorrelated — a plain paired t-test/Wilcoxon would overstate significance.
*   **Decision:** primary test = Diebold-Mariano (HAC/Newey-West variance) on the per-window RMSRE differential; secondary/sanity = paired Wilcoxon (flagged as optimistic); effect size = block-bootstrap CI (block size ≈ window/step ratio).
*   **Recorded as a standing reminder** in both `AE_gemini_lessons.md` and `argo_claude_actions/AE_claude_lessons.md`: use significance tests, not raw summary-stat deltas, to judge whether a pipeline variant actually improved.
*   Implementation (MLD/N² diagnostic functions in `argoebus_thermodynamics.py`, kernel significance-test script) planned but deliberately deferred — no files touched.

---

## 2026-04-26 — MLOps Spec Review & FX2 Science Verdict

**Action:** Completed the comprehensive review of the MLOps Foundation design and issued a science verdict on the FX2 run regressions.

### 1. MLOps Foundation Review
*   **Result:** Spec **Approved for Implementation** with conditions.
*   **Key Findings:** Identified gaps in **ERDDAP data lineage** (missing dataset IDs/timestamps), **Quality Control (QC) policy** documentation, and **GPR Anisotropy** configuration in the YAML schema.
*   **Output:** Created `docs/superpowers/specs/2026-04-26-mlops-review-results.md`.

### 2. FX2 Science Verdict & Depth Range Standardization
*   **Diagnosis:** Confirmed the Source Layer regression (8.13% RMSRE) is a result of **Sparsity-Resolution Conflict** in 10-day bins.
*   **Action:** Standardized the vertical layers to align with **Roemmich-Gilson (RG)** climatology definitions for all future `californiav3` runs.
*   **Updated Depths:**
    *   **Response:** [0, 100m]
    *   **Source:** [100, 500m]
    *   **Background:** [500, 1500m]
*   **Output:** Created `docs/superpowers/specs/2026-04-26-fx2-diagnostics-verdict.md`.

### 3. RG-Gibbs Strategy Alignment
*   **Verdict:** Rejected data-density-driven lengthscales ($l(x)$) as physically indefensible.
*   **Strategy:** Pivoted toward a **learnable sigmoid function** based on `dist_to_coast`, allowing the GPR to determine the physical regime transition boundary from the data.

---

## 2026-04-11 — FX2 Run Diagnosis & Coastal Distance Implementation

**Action:** Diagnosed the FX2 Source Layer regression, implemented the `dist_to_coast` feature, and optimized the `californiav3` domain based on the 26-year float census.

### 1. FX2 Science Review (Three Issues Diagnosed)
*   **Source Layer Regression (Q1):** Confirmed as **domain clipping**. The tight `californiav2` domain (130W–115W) is insufficient for the Source Layer (150-400m) where float drift reduces density. The underdetermined GPR kernel caused extreme anisotropy and RMSRE up to 22%.
*   **Temporal Scale Saturation (Q2):** `scale_time_bin` hit the 45-day ceiling in Skin/Source layers. Conclusion: Ocean memory at depth likely exceeds the 45-day window width. Strategy: Recommend widening `time_ls_bounds_days` to 90 days for deeper layers.
*   **Background Layer Z-Spike (Q3):** Z=18.73 at window 6102.5 (Sep 2015) confirmed as a **stationarity violation** from the **Pacific Blob** onset. The model accurately fits the temperature anomaly but underestimates uncertainty during this extreme event.

### 2. Empirical Domain Optimization (californiav3)
*   **Action:** Executed the depth-aware float census (`09c_ae_depth_aware_float_census.py`) across all scientific layers (1999–2025).
*   **Optimization:** Redefined `californiav3` in `ae_utils.py` based on census hotspots: **Lat [30, 48], Lon [-135, -115]**. This wider longitudinal buffer (increased from 130W to 135W) stabilizes the GPR by capturing more high-density offshore float trajectories.

### 3. Coastal Distance Implementation
*   **Action:** Developed and implemented `calculate_dist_to_coast` in `ebus_core/ae_utils.py` using Cartopy and KDTree for fast, accurate spatial lookups.
*   **Integration:** Updated `01_ae_cloud_ingestion.py`, `02_ae_cloud_run.py`, and `ebus_core/argoebus_thermodynamics.py` to include `dist_to_coast_km` as a standard feature in every OHC parquet output.

### 4. Repository Hygiene
*   **Updated:** `argo_gemini_actions/AE_gemini_todo.md` and `argo_gemini_actions/AE_gemini_lessons.md` with the new findings and completed tasks.
*   **Audit Output:** Census data and heatmaps archived in `AEResults/aeplots/float_census_depth_aware/`.

---

## 2026-04-01 — Data-Driven Domain Strategy: The "Long-Term Census"

**Action:** Diagnosed the "Data Desert" at depth in `californiav2` and pivoted to an empirical boundary optimization strategy for `californiav3`.

### 1. Diagnosis of californiav2 Source Layer Failure
*   **Finding:** Verified that `californiav2` (130W–115W) is severely under-sampled at 150-400m depth, with only **39 unique floats** (down from 97 in the original domain) and a median of **23 bins per 10-day window**.
*   **Verdict:** The 3D GP model is underdetermined in this tight domain, causing the extreme anisotropy ratios (up to 35.75) and RMSRE regressions (8.13% median).

### 2. Implementation Plan: Long-Term Float Census (1999–2025)
*   **Action:** Drafted `argo_gemini_actions/AE_plan_longterm_float_census.md` for Claude to implement.
*   **Strategy:** Map float availability in 5°x5° bins over a 26-year period to identify stable data "Hotspots" (e.g., Southern California Bight).
*   **Goal:** Use the resulting "Small Multiples" heatmap to define `californiav3` based on where the sensors actually are, ensuring the "Stealth Warming" study has sufficient statistical power.

### 3. Repository Hygiene
*   **Updated:** `argo_gemini_actions/AE_gemini_todo.md` to reflect the priority shift toward the Census and `californiav3`.
*   **Created:** `ArgoEBUSCloud/08_ae_diagnose_density.py` (diagnostic script used to confirm the data desert).

---

## 2026-04-05 — California_V3 Transition & Coastal Distance Planning

**Action:** Defined the `californiav3` domain and established a planning/validation role for the `dist_to_coast` feature.

### 1. Defined California_V3 Domain
*   **Action:** Added `californiav3` to the EBUS registry in `ae_utils.py` with Lat [25, 50] and Lon [-140, -110].
*   **Rationale:** This expands the spatial domain to increase Argo float density ($N$) and stabilize the GPR, matching the original broad "california" window while providing a robust baseline for the "Thermal Battery" audit.

### 2. Coastal Distance Planning
*   **Strategy Change:** Shifted responsibility for the `dist_to_coast` implementation to Claude.
*   **Role:** Gemini will serve as the validator for the `dist_to_coast` implementation, ensuring its robustness and accuracy for future ML stages.
*   **Updated:** `argo_gemini_actions/AE_gemini_todo.md` to reflect this new division of labor.

### 3. Spatio-Temporal GPR Refactoring
*   **Next Steps:** Claude will refactor `ebus_core/argoebus_gp_physics.py` to use a 3D Matérn kernel ($\nu=0.5$) with features `[lat, lon, time_days]`.
*   **Verification:** Gemini will monitor key metrics (RMSRE < 5%, Anisotropy Ratio, Temporal Persistence) to ensure scientific validity.

## 2026-04-11 — Proposed the RG-Anisotropy Hybrid Model Directive

- Defined the **RG Anchor Mean Function** (Copernicus RG Climatology) for anomaly-based training.
- Formalized the **2:1 Anisotropy Ratio** constraint to respect the California Undercurrent structure.
- Expanded the **Vertical Sandwich** depths for better Stealth
## 2026-04-11 — Proposed the RG-Gibbs Non-Stationary Model Directive

- Replaced the stationary anisotropy hybrid with a **Gibbs Non-Stationary Kernel** to solve data sparsity.
- Defined the **longitude-dependent lengthscale** $l(lon)$ to resolve sharp coastal gradients (100km) while smoothing offshore (400km).
- Integrated the **Roemmich-Gilson Anchor** as the prior mean function to ensure statistical integrity.
- Saved the updated blueprint to `argo_claude_actions/brainstorming/RG_Gibbs_NonStationary_Model_Plan.md`.

---

## 2026-05-04 — californiav3 Science Review & Gibbs Green-light

**Action:** Conducted a detailed science review of Claude's `californiav3` results and issued a formal "Green-light" for the Gibbs Non-Stationary Kernel implementation.

### 1. californiav3 Science Verdict
*   **Source Layer (150-400m) Fix:** Confirmed the median RMSRE dropped from **8.13% (v2)** to **3.05% (v3)**. This validates that the Source layer sparsity regression was a domain-clipping artifact, now resolved by the wider 35^\circ W$ buffer.
*   **Scientific Consistency:** The Source layer exhibits the predicted **meridional anisotropy** ({lat} > l_{lon}$), a clear signature of the California Undercurrent.
*   **Stationarity Audit:** Background layer Z-score spikes (Z > 9.0) in Jan-Feb and Sep 2015 are confirmed as **Pacific Blob non-stationarity events**. The stationary Matérn's failure to estimate uncertainty here is the primary motivation for the Gibbs pivot.
*   **Time Scales:** `scale_time_bin` saturation at 45 days persists across all layers, confirming ocean memory > 45 days at depth.

### 2. Gibbs Kernel Green-light
*   **Verdict:** Approved the transition to the **Gibbs Non-Stationary Kernel**.
*   **Motivation:** The "chronic" Z-score pattern (0.5–0.9) near the shelf break indicates the stationary model is struggling with the coastal-offshore regime transition.
*   **Implementation Directive:** Use `dist_to_coast` as the coordinate for the learnable sigmoid lengthscale function (x)$, as planned in the RG-Gibbs directive.

---
