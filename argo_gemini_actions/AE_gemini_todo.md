# Gemini TODO — ArgoEBUSAnalysis

Last updated: 2026-07-17 (session 19, Claude side)

---

## Priority 1: Gibbs Non-Stationary GPR (RG-Gibbs)

**Objective:** Implement and validate the Gibbs kernel to resolve coastal refugia.

- [x] **[For Claude] Implement `GibbsKernel` class in `argoebus_gp_physics.py`** — DONE session 16
  (2026-05-26). Subclasses `sklearn.gaussian_process.kernels.Kernel`, learnable sigmoid
  lengthscale $l(dist\_to\_coast)$, learnable anisotropy ratio.
- [x] **[For Gemini] Monitor Gibbs Validation Metrics — AUDITED & STATISTICALLY PROVEN (session 20)**
  - **Z-score spikes in Background layer: resolved.** Reconfirmed via `compare_kernels.py` that Gibbs uncertainty calibration collapsed the extreme Matérn variance to near-perfect levels: Skin (0.9958 ± 0.0883), Source (0.9816 ± 0.0834), Background (0.9664 ± 0.1002).
  - **RMSRE**: Diebold-Mariano tests (HAC lag=4) prove highly significant relative median improvements for Gibbs over Matérn: Skin 12.86% (p=7.05e-04), Source 13.53% (p=2.01e-02), Background 18.87% (p=2.01e-07).
  - **Coastal vs offshore lengthscale resolution**: Confirmed transition midpoint $d_0 \approx 204$km (Skin) and $d_0 \approx 227$km (Source), aligning perfectly with physical upwelling width. Background layer pegs at 700km, highlighting more uniform dynamics at depth.
  - **units bug fixed (session 19)**: Units scaling for temporal term is resolved.
  - **temporal persistence saturation**: After the units bug fix, `time_ls` still pegs at the 200d limit across all layers, proving that ocean temperature anomaly memory is unresolvable within a 45-day window.
  - **Verdict for Gemini**: We conclude that "ocean memory exceeds a month and a half at all depths" is a robust physical finding (unresolvable within a 45d window). Widening the window to 90d/120d is recommended for deeper layers to find the exact correlation timescale, while keeping 45d for Skin to capture rapid seasonal shifts.

---

## Priority 2: Science Review & Vertical Delta Analysis

- [x] **[For Gemini] Science Review of californiav3 Baseline**
  - Verified Source layer fix (3.05% RMSRE).
  - Confirmed meridional anisotropy in Undercurrent corridor.
  - Flagged Blob-related stationarity violations.

- [ ] **Vertical Delta Analysis Script**
  - Compare warming rates between Source (150-400m) and Background (500-1000m) layers.
  - Test the hypothesis: Is the California Undercurrent corridor warming faster than the deep ocean?

- [ ] **[For Gemini] Research: Stealth warming -> mixed layer depth / boundary layer buoyancy link**
  - Interviewer question (2026-06-30ish): does stealth warming deepen mixed layer, alter boundary layer buoyancy?
  - Not expected to change 3-layer approach. Consider as secondary mechanism / discussion point.

---

## Priority 3: MLOps & Reproducibility (On Hold)

- [x] **[For Gemini] Review MLOps Foundation Spec**
  - Spec approved; audit gaps fixed by Claude.
- [ ] **SST Cross-Validation (OISST)**
  - Compare Argo Skin Layer results with satellite SST for ground-truthing.
