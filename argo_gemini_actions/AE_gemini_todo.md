# Gemini TODO — ArgoEBUSAnalysis

Last updated: 2026-03-25

## Priority 1: Multi-Layer Verification (The Stealth Signal)

- [ ] **Review Source Layer (150-400m) Results**
  - Run `04_ae_testmatern_and_3dwindow.py` for the Source Layer.
  - Compare Anisotropy Ratio and Temporal Persistence with Skin Layer (0-100m).
  - *Expectation*: Higher anisotropy and longer persistence at depth due to Undercurrent dominance.

- [ ] **Review Background Layer (500-1000m) Control**
  - Establish the baseline warming rate and variance for the deep ocean.
  - Confirm the 3D GP remains stable in low-variance deep water.

## Priority 2: Branch Merging & Stability

- [ ] **Extend `plot_kriging_snapshot` for 3D**
  - Update the snapshot logic to handle the 3rd (time) dimension (likely by slicing at the window center) to allow visual inspection of 3D fits.

## Priority 3: Cross-EBUS Expansion

- [ ] **Test 3D Pipeline on `humboldt` or `canary`**
  - Validate that the 15-day temporal persistence and 30-day window size are globally applicable or require per-region tuning.

---
**Human Note (2026-03-25):**
I want to begin writing detailed, readable documentation, both for coders (implementation details, API, scaling) and scientists (physical significance, hypothesis testing, metric interpretation).

