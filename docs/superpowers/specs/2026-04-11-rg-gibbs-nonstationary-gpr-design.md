# RG-Gibbs Non-Stationary GPR — Draft Design Spec

**Date:** 2026-04-11  
**Status:** DRAFT — Brainstorming in progress. Open question on l(x) functional form. Do NOT implement until resolved.  
**Author:** Claude Code (brainstorming session with Avik)

---

## Objective

Replace the current stationary Matern/RBF GPR with a non-stationary Gibbs kernel GPR anchored to the Roemmich-Gilson (RG) climatology as a mean function. Goal: resolve fine-scale coastal "Refugia" features while maintaining statistical integrity in data-sparse offshore regions.

---

## Decisions Made

### Framework
- **Stay in sklearn.** Custom `GibbsKernel` subclasses `sklearn.gaussian_process.kernels.Kernel`. No TF/GPflow dependency — avoids Coiled environment bloat and pipeline rewrites.

### Integration Pattern
- **Approach B: New standalone `validate_moving_window_nonstationary()`.**
  - Existing `validate_moving_window` untouched — stationary runs remain valid for A/B comparison.
  - New function shares the same audit CSV output format.
  - Unification deferred until Gibbs model validates (RMSRE < 5%).

### RG Climatology
- **Source:** Copernicus Marine `INSITU_GLO_PHY_TS_OA_MY_013_052`
- **Ingestion:** New script `00_ae_rg_climatology_ingest.py` fetches via `copernicusmarine` client, clips to `californiav3` bounds, converts to Zarr.
- **Storage:** `s3://argo-ebus-project-data-abm/rg_climatology/californiav3_rg.zarr`
- **Usage:** `load_rg_mean()` helper interpolates Zarr to query lat/lon/time for each window. GPR trains on residuals `(T_obs - T_RG)`. RG mean added back at prediction time.

### Cloud Ingestion
- Full new run required with `californiav3` domain [30–48°N, 135–115°W].
- New depth ranges: `d0_100`, `d100_500`, `d500_1500` (Roemmich-defined layers).
- Scripts 01 + 02 unchanged structurally — new run_ids only.

### Vertical Layers
- `get_vertical_layers()` updated to:
  - `Response:    [0, 100]`
  - `Source:      [100, 500]`
  - `Background:  [500, 1500]`
- Aligned with Roemmich-Gilson natural layer definitions.

### Focus Slider Demo
- Scoped separately. Not part of this implementation phase.
- See `AE_claude_todo.md` for LinkedIn demo task.

---

## File Changes

| File | Change |
|------|--------|
| `ArgoEBUSCloud/00_ae_rg_climatology_ingest.py` | **NEW** — Copernicus fetch → S3 Zarr |
| `ArgoEBUSCloud/01_ae_cloud_ingestion.py` | Run with californiav3 + new depth ranges |
| `ArgoEBUSCloud/02_ae_cloud_run.py` | Run with californiav3 + new depth ranges |
| `ArgoEBUSCloud/ebus_core/ae_utils.py` | Update `get_vertical_layers()` |
| `ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py` | Add `GibbsKernel`, `load_rg_mean()`, `validate_moving_window_nonstationary()` |

---

## Open Question — GibbsKernel: l(x) Functional Form

**Status: BLOCKED — awaiting literature review + Gemini science input.**

The Gibbs kernel requires a position-dependent lengthscale function `l(x)`. The design discussion surfaced a fundamental question:

**What we rejected:**
- `l(lon)` with fixed `lon_mid = -125°W` — arbitrary; no physical justification for the midpoint
- `l(dist_to_coast)` with fixed `d_transition = 300km` — shelf break varies from ~50km (Oregon) to ~150km (central CA); fixing a transition distance is also an assumption
- Linear ramp from coast to offshore — still prescribes a profile shape

**The deeper issue:** Any fixed functional form for `l(x)` is an assumption about where and how the lengthscale transitions. The whole motivation for the Gibbs kernel is to let `l` vary — so prescribing its shape undermines the design.

**Candidate approaches under consideration:**
1. **Data-density-driven `l(x)`:** For each GPR window, compute local Argo float density per spatial cell. Normalize to [0,1]. Set `l(x) = l_max - (l_max - l_min) * normalized_density(x)`. No functional form assumed — l follows data availability. Only `l_min` and `l_max` are user-set bounds.
2. **Fully learnable parametric form:** Expose `l_min`, `l_max`, and a "rate" parameter α to sklearn's optimizer. Let data determine the transition shape.
3. **Literature-guided form:** Identify established practice from GP oceanography papers and adopt that functional form with physical justification.

**Action items before resuming:**
- [ ] Avik: review GP oceanography literature for established l(x) functional forms
- [ ] Gemini: science input on whether data-density-driven l(x) is physically defensible for the CCS
- [ ] Resume brainstorming session to finalize Section 3 (GibbsKernel design)

---

## Remaining Design Sections (Not Yet Written)

- **Section 3 (cont.):** GibbsKernel — full class interface, hyperparameter bounds, anisotropy handling
- **Section 4:** `load_rg_mean()` — S3 Zarr read, interpolation strategy, handling missing climatology cells
- **Section 5:** `validate_moving_window_nonstationary()` — full function signature, residual workflow, audit CSV format
- **Section 6:** Vertical layer update details + backward compatibility notes
- **Section 7:** Testing strategy — how to validate Gibbs vs stationary on same data

---

## Resume Checkpoint

Next brainstorm session starts at: **Section 3 — GibbsKernel design, after l(x) question is resolved.**
Reference this file and `AE_claude_todo.md` checkpoint section to reconstruct context.
