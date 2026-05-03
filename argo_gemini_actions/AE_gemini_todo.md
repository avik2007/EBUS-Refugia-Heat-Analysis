# Gemini TODO — ArgoEBUSAnalysis

Last updated: 2026-04-25

---

## [Note from Claude — 2026-04-25] Review MLOps Foundation Spec

Avik approved an MLOps foundation design today. Please review the spec and flag
any gaps from a science / reproducibility / engineering-practitioner perspective
before implementation begins.

**Spec:** `docs/superpowers/specs/2026-04-25-mlops-foundation-design.md`

**Scope:** config-driven runs (YAML per region/layer/experiment) + reproducibility
manifests (config hash, git SHA, conda env, S3 lineage) + thin `aebus` CLI. Additive
layer atop existing scripts; no refactor. Deferred: MLflow/W&B (B-tier), pip package
+ Docker (C-tier).

**Specific things to look at:**
- §3 Config schema — are the captured fields enough to reproduce a science run?
  Anything science-relevant we'd lose if we only had the YAML?
- §4 Manifest — provenance fields adequate? Missing anything (e.g., ERDDAP query
  URL/timestamp, gsw library version, TEOS-10 conventions)?
- §6 Backfill — risk of mis-recovering old run params from the run_id alone?
- §9 — confirm RG-Gibbs (`2026-04-11-rg-gibbs-nonstationary-gpr-design.md`) will
  cleanly slot a `kernel_type: gibbs` entry into the new AnalysisConfig schema.

**Related companion spec (already in your queue):**
`docs/superpowers/specs/2026-04-11-rg-gibbs-nonstationary-gpr-design.md` — still
blocked on l(x) functional form (see note below).

---

## [Note from Claude — 2026-04-11] RG-Gibbs Brainstorm: Science Input Needed on l(x)

Brainstorming the Gibbs kernel design hit a fundamental question that needs Gemini's 
oceanographic perspective before implementation can proceed.

**Context:** We are implementing a Gibbs non-stationary kernel where the GPR lengthscale
`l` varies with position — small (~100km) near the coast, large (~400km) offshore.
The question is: what should determine `l(x)` at each point?

**Candidate approaches:**
1. **Data-density-driven:** `l(x) = l_max − (l_max−l_min) × normalized_float_density(x)`.
   Dense float coverage (coast) → fine scale. Sparse (offshore) → smooth. No assumed profile.
2. **Fully learnable parametric:** Expose l_min, l_max, and a rate parameter α to the
   sklearn optimizer. Let data determine transition shape via log-likelihood maximization.
3. **Literature-guided:** Adopt an established functional form from GP oceanography papers.

**The core tension:** Any fixed profile (sigmoid of dist_to_coast, linear ramp, etc.)
prescribes where the transition happens — which contradicts the Gibbs motivation of
letting l vary freely. But sklearn needs hyperparameters to be fixed scalars.

**Gemini question:** 
- Is a data-density-driven l(x) physically defensible for the CCS? 
  (i.e., does float density actually trace the physical regime boundary between coastal 
  upwelling dynamics and offshore gyre dynamics?)
- Is there a standard approach in the GP oceanography literature that you're aware of?
- Avik is reviewing papers — please discuss when he brings findings to you.

**Draft spec:** `docs/superpowers/specs/2026-04-11-rg-gibbs-nonstationary-gpr-design.md`

---

## [Note from Claude — 2026-04-11] FX2 Run Diagnostics: Three Science Questions

Results from the californiav2 FX2 high-res temporal run (`t10_0`) are in. Three issues need Gemini science review before proceeding.

### Q1: Source Layer RMSRE Regression — Root Cause?
- Median RMSRE degraded from ~4.2% (t30 baseline) → 8.13% (t10 run).
- Only 8/34 windows pass 5% threshold. Max RMSRE 22.09%.
- Extreme anisotropy ratios (up to 35.75) — non-physical.
- Worst windows: day centers 5952, 6032, 6072, 6082, 6132, 6142, 6152, 6172, 6182, 6192.
- Z spike: window 6022 std_z=15.63; window 6172 std_z=4.48.
- Audit CSV: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45/audit_californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45.csv`
- **Question:** Is the regression from (a) californiav2 domain clipping float trajectories at depth, (b) 10d bins exposing genuine sparsity that 30d bins masked, or (c) a GPR configuration issue?

### Q2: `scale_time_bin` Saturates at 45d Ceiling (Skin + Source)
- Every window in Skin and Source layers hits the `time_ls_bounds_days` upper limit (45d).
- No aliasing oscillation, but still pegged to ceiling.
- Background layer healthy: scale_time_bin varies 26–45d mid-year.
- **Question:** Widen `time_ls_bounds_days` upper bound for Skin/Source? Or is 45d saturation physically meaningful (ocean memory > window width)?

### Q3: Background Layer Z=18.73 Spike at Window 6102.5 (~Sep 2015)
- RMSRE only 2.67% but std_z=18.73. Likely Pacific Blob peak non-stationarity.
- Prior Gemini verdict: genuine physical event, flag if Z > 2.0 persists.
- Audit CSV: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45/audit_californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45.csv`
- **Question:** Confirm Z=18.73 is Blob onset. Mark as stationarity violation?

---

## Priority 1: Empirical Domain Optimization (californiav3)

**Objective:** Use the Long-Term Float Census to define a high-density sub-region for the stealth warming analysis.

- [x] **[For Claude] Execute Long-Term Census (1999-2025)**
  - Executed by Gemini using `09c_ae_depth_aware_float_census.py`.
  - Output: 26-year census data and mean heatmaps in `AEResults/aeplots/float_census_depth_aware/`.

- [x] **[For Gemini] Interpret Census and Define californiav3**
  - Identified "Golden Age" (2015-2025) and "Hotspots" (Lat 30-48, Lon -135 to -115).
  - Selected final Lat/Lon bounds for `californiav3` in `ae_utils.py`.

- [x] **[For Claude] Implement californiav3 in ae_utils.py**
  - Updated by Gemini.

---

## Priority 2: Coastal Distance & Spatio-Temporal GPR (California_V3)

**Objective:** Expand the spatial domain to $140^\circ W$ to $115^\circ W$ (California_V3) and introduce a `dist_to_coast` feature.

- [x] **[For Gemini] Define California_V3 Bounds**
  - Lat [30, 48], Lon [-135, -115] (Optimized based on census).
  - Added to `ae_utils.py` registry.

- [x] **[For Claude] Implement `dist_to_coast` Metric**
  - Implemented by Gemini in `ebus_core/ae_utils.py` and integrated into `01_ae_cloud_ingestion.py`, `02_ae_cloud_run.py`, and `ebus_core/argoebus_thermodynamics.py`.

- [x] **[For Gemini] Validate Claude's `dist_to_coast` Implementation**
  - Verified logic using Haversine distance and KDTree for fast lookup.

- [ ] **[For Claude] Refactor `argoebus_gp_physics.py` for 3D Matérn**
  - Ensure features include `[lat, lon, time_days]`.
  - Implement "Centric-Snapshot" logic in the predict step.

- [ ] **[For Claude] Refactor `argoebus_gp_physics.py` for 3D Matérn**
  - Ensure features include `[lat, lon, time_days]`.
  - Implement "Centric-Snapshot" logic in the predict step.

- [ ] **[For Gemini] Monitor GPR Validation Metrics**
  - Target RMSRE < 5%.
  - Audit Anisotropy Ratio ($l_{lat}/l_{lon}$) and Temporal Persistence ($l_{time}$).

---

## Priority 3: Consolidated "Clean Slate" Runs (on California_V3)

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
