# Gemini TODO — ArgoEBUSAnalysis

Last updated: 2026-03-31

---

## Priority 1: californiav2 Migration and Vertical Comparison

- [ ] **Coordinate californiav2 Clean Slate Runs**
  - Directive to Claude: Execute Script 02 for all layers using `region='californiav2'`, `time_step=10.0`, and `time_ls_bounds_days=(15.0, 45.0)`.
  - Objective: Align data bins with the 10-day Argo cycle to resolve structural aliasing at the source.
  - Audit GPR results with `step_size_days=10.0`.

- [ ] **Synthesize Vertical Delta Report (`04_ae_vertical_compare.py`)**
  - Develop a comparison script to load audit CSVs from the `californiav2` runs for all three layers.
  - Quantify: Is the **Source Layer (150–400m)** warming faster than the **Background (500–1000m)**?
  - Verify: Does the Source Layer maintain a stable meridional anisotropy ratio (Undercurrent signature) compared to the Skin layer?

- [ ] **Deep-Dive: May 2015 Blob Onset**
  - Analyze the 500m window 5977.5 in the high-resolution run.
  - Determine if higher temporal resolution allows the GP to capture the heat surge without overconfidence (Z > 2.0).

---

## Priority 2: Multi-Layer Verification and Physics Review (Complete - 2015 baseline)

- [x] **Analyze Background Layer Failure (May 2015)**
  - Investigated window 5955–6000: Anisotropy=0.28, Z=2.02. Identified as a zonally elongated front event at depth.
- [x] **Evaluate Source Layer Anisotropy (Aug–Sep 2015)**
  - Confirmed meridional dominance (>1.0) in Skin/Source, switching to Zonal (<0.9) in Background.
- [x] **Assess Spatial Length Scale Bounds**
  - Confirmed saturation at 5.0 sigma. Recommended widening to 10.0 for depth > 500m.
- [x] **Cross-Layer Persistence Comparison (The Sampling Beat)**
  - Disproved hypothesis: oscillation persists in Source layer due to sampling aliasing (now known as structural binning aliasing).

---

## Priority 2: Branch Merging & Stability

- [ ] **Extend `plot_kriging_snapshot` for 3D**
  - Update the snapshot logic to handle the 3rd (time) dimension (likely by slicing at the window center) to allow visual inspection of 3D fits.

---

## Priority 3: Cross-EBUS Expansion

- [ ] **Test 3D Pipeline on `humboldt` or `canary`**
  - Validate that the 15-day temporal persistence and 30-day window size are globally applicable or require per-region tuning.

---

## Priority 4: External Validation

- [ ] **SST Comparison Design**
  - Review the plan to use OISST or MUR SST to cross-validate the Argo Skin Layer (0–100m) results.
  - Focus on ensuring the 0.5° binning logic is robust before trusting deep-layer deltas.

---
**Human Note (2026-03-25):**
I want to begin writing detailed, readable documentation, both for coders (implementation details, API, scaling) and scientists (physical significance, hypothesis testing, metric interpretation).
