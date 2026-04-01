# Gemini TODO — ArgoEBUSAnalysis

Last updated: 2026-04-01

---

## Priority 1: Empirical Domain Optimization (californiav3)

**Objective:** Use the Long-Term Float Census to define a high-density sub-region for the stealth warming analysis.

- [ ] **[For Claude] Execute Long-Term Census (1999-2025)**
  - Implement `09_ae_longterm_float_census.py` based on the design in `argo_gemini_actions/AE_plan_longterm_float_census.md`.
  - Output: 26-year "Small Multiples" heatmap of unique floats per 5°x5° bin.

- [ ] **[For Gemini] Interpret Census and Define californiav3**
  - Review the census heatmap to identify the "Golden Age" (years with max density) and "Hotspots" (bins with >10 floats consistently).
  - Select final Lat/Lon bounds for `californiav3`.

- [ ] **[For Claude] Implement californiav3 in ae_utils.py**
  - Add the new region definition to the registry.

---

## Priority 2: Consolidated "Clean Slate" Runs (on californiav3)

**Objective:** Once the domain is optimized, run the full 3-layer analysis.

- [ ] **Cloud Ingestion (Script 02)**
  - `californiav3` all three layers: `time_step=10.0`, high-res temporal bins.

- [ ] **GPR Analysis (Script 05/07)**
  - Execute 3D Matern pipeline with canonical FX2/T3 settings:
    - `step_size_days=10`
    - `time_ls_bounds_days=(15.0, 45.0)`
    - `spatial_ls_upper_bound=10`

---

## Priority 3: Vertical Heat Content Comparison

- [ ] **Vertical Delta Analysis Script**
  - Compare warming rates between Source (150-400m) and Background (500-1000m) layers.
  - Test the hypothesis: Is the California Undercurrent corridor warming faster than the deep ocean?

- [ ] **Anisotropy Fingerprinting**
  - Document the meridional/zonal dominance shifts across the three layers to isolate the jet signature.

---

## Priority 4: Background & Maintenance (On Hold)

- [ ] **May 2015 Background failure (Z=18.73)**
  - Re-examine on the new `californiav3` domain once available.
- [ ] **SST Cross-Validation (OISST)**
  - Compare Argo Skin Layer results with satellite SST for ground-truthing.
