# Gemini Recent Actions — ArgoEBUSAnalysis

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
