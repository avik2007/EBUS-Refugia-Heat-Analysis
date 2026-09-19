# MLE Interview Notes — ArgoEBUSAnalysis
*Assembled 2026-05-10. Framework A (Principled Model Choice) with Framework C (Failure-Driven) elements. Target: Senior MLE (L6/E5+) at industry.*

---

## The Problem

**Hypothesis:** The California Current System has a hidden subsurface heat reservoir — warming transported by the California Undercurrent that hasn't reached the surface yet ("stealth warming" / "ocean refugia"). Coastal ecosystems currently sheltering in cold upwelled water may be living on borrowed time.

**Experimental design — three depth layers:**
- Skin (0–100m): atmospheric forcing dominates, volatile
- Source (150–400m): Ekman upwelling source water — where stealth heat hides
- Background (500–1000m): deep ocean baseline, control

**Signal:** Source layer warming faster than Background, with spatial pattern consistent with current flow (meridional), not atmosphere (zonal).

**Data:** Argo profiling floats — autonomous robots that drift with ocean currents, surfacing every 10 days to transmit T/P profiles. Sparse (~100s of profiles per year in domain), spatially irregular, physically smooth.

---

## Why GPR (Model Choice Justification)

- **Sparse + irregular data:** with ~100s of points per rolling window, fitting a NN means millions of params to hundreds of samples. GPR's inductive bias (smoothness, stationarity) matches the physical domain.
- **Calibrated uncertainty required:** science needs confidence intervals, not just predictions. GPR gives you a posterior distribution; you can ask whether the model is over- or underconfident and use that as a diagnostic.
- **Interpretable structure:** the lengthscale IS the measurement. It encodes how spatially correlated OHC is in lat vs. lon — which is how you distinguish current-driven vs. atmosphere-driven structure.
- **Rolling windows handle temporal non-stationarity:** 45-day windows, each fit independently. No assumption of a single stationary GP across 20 years.

---

## Hyperparameters (Things You Control and Tune)

These are the design choices you set before fitting, tuned based on results:

- **Kernel type:** model selection decision. RBF → Matern-0.5 → Gibbs (see Kernel Evolution below). Each transition was motivated by a specific failure mode in the data.

- **Window size (45 days):** how much data each GP fit sees. Too small → not enough float profiles to identify the GP. Too large → temporal non-stationarity bleeds in (the 2015 Pacific Blob lasted >1 year; you don't want it contaminating a summer window).

- **Step size (10 days):** temporal resolution of the rolling output. Must be ≥ time_step (the bin width of the input data) or you get duplicate data in consecutive windows — see Mistake #2 below.

- **Lengthscale bounds:** you don't set the lengthscales — the optimizer learns them via marginal likelihood — but you constrain the search space. Time bounds: 15–45 days (mesoscale eddy timescale). Spatial bounds: informed by expected current structure. If the optimizer saturates at a bound, that's a signal the bound is wrong, not the model.

- **Grid resolution (0.5° × 0.5°):** output kriging grid. Matches Argo float density in the California Current. Coarser = faster but smears upwelling filaments; finer = sparse coverage artifacts.

- **min_bins:** minimum float profiles required in a window to fit at all. Below this the GP is underidentified — no fit attempted.

- **Gibbs-specific (fixed by physics, not learned):** l_min_km = 100 (coastal filament width), l_max_km = 400 (open-ocean correlation length), anisotropy ratio = 2:1 lat:lon (CCS current is meridional). These are domain constraints, not optimizer knobs.

- **RMSRE threshold (< 5%):** acceptance criterion for whether a hyperparameter configuration is working, applied per rolling window.

---

## The Model Output as Scientific Observable

The anisotropy ratio = lat_lengthscale / lon_lengthscale is not a tuning artifact — it IS the finding:
- < 1.0: zonal/atmospheric forcing dominates
- > 1.0: meridional current flow dominates

**2015 results:**
- Skin layer: ratio 0.36–0.49 year-round → zonal dominance confirmed (atmosphere)
- Source layer: ratio rises to 1.07–1.15 in Aug–Sep → meridional dominance, California Undercurrent signature
- Background layer: remains zonal → no current forcing at 500–1000m

The vertical profile of the anisotropy ratio is the primary scientific deliverable.

---

## Kernel Evolution: RBF → Matern-0.5 → Gibbs

### RBF (Squared Exponential)
Assumes the field is infinitely differentiable — you can take derivatives to any order. Wrong for ocean heat content. OHC has sharp gradients: upwelling filaments, the shelf break, mesoscale eddies. RBF was smoothing over physically real structure. The script is still named `05_ae_update_tomatern0.5.py` — the name is a record of this transition.

### Matern-0.5 (Exponential Kernel)
k(r) = exp(−r/l). Only continuous, not differentiable — allows sharp local structure. RMSRE came down substantially: Source 3.05%, Background 2.50%. The anisotropy signal emerged cleanly.

But the stationary assumption remained: one global lengthscale across the entire domain. That treats a point 20km from the coast identically to a point 400km offshore. In the CCS these are physically distinct regimes — coastal upwelling filaments cohere at ~100km; open-ocean OHC anomalies at ~400km. The shelf break is the transition.

**Symptom:** Chronic Z-score 0.5–0.9 near the shelf break, across all three layers, all windows. Z < 1 means the model is overconfident — uncertainty estimates too small. Spatially specific, systematic — only visible by mapping the residuals, not from aggregate RMSRE.

### Gibbs Kernel (Non-Stationary)
Replaces the single global lengthscale with a function: l(x) = sigmoid(dist_to_coast_km).

```
l(d) = l_min + (l_max - l_min) / (1 + exp(-k * (d - d_0)))
```

Near the coast: l → 100km. Far offshore: l → 400km. Sigmoid midpoint d_0 and steepness k are learned per window. The 2:1 lat:lon anisotropy is fixed.

Kernel matrix entry (Gibbs 1997, Paciorek & Schervish 2004):
```
k(x_i, x_j) = sqrt(2·l(x_i)·l(x_j) / (l(x_i)² + l(x_j)²)) · exp(−(Δx)² / (l(x_i)² + l(x_j)²))
```
When l(x_i) = l(x_j), prefactor = 1 and you recover stationary Matern. When they differ — as they do near the shelf break — the prefactor shrinks, correctly inflating uncertainty between points in different regimes.

**Implementation wrinkle:** sklearn's default GPR optimizer calls `kernel(X, eval_gradient=True)`. Gibbs doesn't have closed-form analytic gradients. Solution: custom L-BFGS-B optimizer using scipy finite-difference gradients. Slower, but correct, and bounded per-window so overhead is manageable.

**Prerequisite bug (fixed first):** `dist_to_coast_km` was computed with a KDTree in raw degree space. At ~40°N, 1° lon ≈ 85 km vs 1° lat ≈ 111 km — 30% distortion. Fixed by converting to 3D ECEF unit vectors before building the tree (see Mistake #4).

---

## MLOps Story

**Before:** numbered scripts (00–09), config baked into `__main__` blocks, no reproducibility, no registry. Reproducing a run meant reading the script and hoping the defaults hadn't changed.

**What was built (Tier A):**
- YAML configs in `configs/<region>/` with Pydantic v2 schema validation
- CLI: `aebus_cli.py validate / analyze / ingest / list / show`
- Run registry: `AEResults/run_registry.jsonl` — append-only JSONL, every run uniquely ID'd
- Reproducibility manifests: config hash (SHA256), git SHA, conda env snapshot, S3 data lineage, wall-clock duration
- Collision detection: if you re-run the same config, it checks hashes and aborts with a diff if the config changed
- 54+ tests, TDD throughout — every feature started with a failing test

**Key framing:** `kernel_type: matern0.5 → kernel_type: gibbs` in a YAML, and you get a completely reproducible comparison on identical data and windows. The config system is what makes the model comparison credible.

**Roadmap:**
- Tier B (next): MLflow/W&B experiment tracking, per-window metrics in a tracked store
- Tier C (long-term): pip-installable package, Docker workers, public dashboard

---

## Mistakes — With Dates

### 1. Temporal aliasing — oscillating time lengthscale (2026-04-01)
Skin layer time lengthscale alternated between ~2d and ~45d across consecutive windows. First hypothesis: Argo 10-day resurface cycle beating against 15-day window step. Ran Experiment T1 (`step_size_days=10`) — no effect (std 14.87 vs baseline 14.99). The real cause: `step_size_days=15` was shorter than `time_step=30d`. Consecutive windows contained literally identical data bins. The GP was fitting the same dataset and returning different results due to optimization stochasticity. Experiment T2 (`step_size_days=30`) confirmed it: zero variance in time lengthscale, all windows saturating at 45d. What looked like a physical oscillation signal was 100% a data structure artifact.

### 2. Source Layer RMSRE collapse from 4.2% to 8.13% (2026-04-01)
Migrated from `california` to `californiav2` (tighter domain based on float density data) and moved from `time_step=30d` to `10d`. Source layer went from 4.2% to 8.13% median RMSRE; only 8/34 windows passed; some extreme anisotropy ratios (up to 35.75, non-physical). Cause: Argo floats drift with surface currents, so surface positions fall within the domain — but at 150–400m, their positions at measurement time may lie outside the tighter bounding box. The tighter domain was clipping float trajectories at depth. Simultaneously, 10d bins exposed sparsity that 30d bins had masked by averaging. Neither effect was visible in the Skin layer (measured near surface) or Background (deep floats uniformly distributed). Diagnosed from spatial coverage plots, not aggregate RMSRE.

### 3. Spatial bounds saturation misunderstood (2026-03-31/2026-04-01)
Assumed `spatial_ls_upper_bound=5` in scaled units meant ~23.5° — a reasonable open-ocean correlation length. But StandardScaler `scale_` is computed from the actual data spread per rolling window. When the domain is wide and float coverage is sparse, `scaler.scale_` is large, so `5 × scaler.scale_` was already much larger than 23.5° in most windows. Experiment S1 (widening to `spatial_ls_upper_bound=10`) had almost no effect on 20/23 Background windows. Only 3 late-season windows showed any change. The constraint wasn't binding where assumed. Bounds in scaled space are meaningless without checking what the scaler is doing per window.

### 4. KDTree bias in `calculate_dist_to_coast` (2026-04-11)
Gemini implemented the nearest-coastline calculation with a KDTree built on raw (lat, lon) degree coordinates. At CCS latitudes (~35–48°N), 1° lon ≈ 85 km vs 1° lat ≈ 111 km — a 30% difference. A degree-space KDTree treats them as equal, so it systematically underestimates east-west distances. For a feature encoding distance from a north-south coastline, this means profiles due west of the coast look artificially closer than profiles equidistant to the north or south. Fixed by converting to 3D ECEF unit vectors before building the tree, then one exact Haversine pass on the nearest candidate.

### 5. Silent kwargs drift — lat/lon bounds never forwarded to GPR (2026-05-02)
The runner's `dispatch_kwargs` assembled arguments to pass to the GPR script. After the config schema was redesigned to split `spatial_ls_upper_bound` into `lat_ls_bounds` and `lon_ls_bounds`, the runner was updated to compute the merged bound — but never inserted it into `dispatch_kwargs`. Every run since the schema change had been running GPR with the default spatial bound from the function signature, ignoring the config. No error, no warning. Caught by Gemini's post-implementation audit, not by any test. The tests verified manifest and registry output, not that every config field reached its destination.

### 6. Merge order bug in `build_manifest` (2026-04-30)
`build_manifest` merged config-derived fields (source, s3_path, ingestion_run_id) with caller-supplied `inputs_extra`. The merge was `{**config_fields, **inputs_extra}` — `inputs_extra` spread last, silently overwriting any matching config field. If a caller passed an `inputs_extra` with a `source` key, the manifest would record wrong data provenance. For a reproducibility system whose entire purpose is accurate lineage, this would have been silent and catastrophic. Caught during second-pass code review by a separate reviewer agent. Fix: reverse merge order so config fields always win.

### 7. Schema validation passed, runtime crashed (2026-05-03)
`time_ls_bounds_days` in `GPRBlock` was declared `Optional` to support legacy backfilled configs. The new californiav3 analysis YAMLs were written with `time_ls_bounds_days: null` — valid Pydantic schema, passes `aebus validate` cleanly. But `analyze_rolling_correlations` immediately subscripts it: `time_ls_bounds_days[0]` → `TypeError: NoneType is not subscriptable`. The validation boundary didn't enforce that non-legacy configs must have the field populated. The `_non_legacy_complete` validator existed but was checking a different field set — this combination slipped through. Fix: set `time_ls_bounds_days: [15.0, 45.0]` in all three californiav3 YAMLs.

---

## What's Next

### Short term (active)
Implement the Gibbs kernel. Plan written, Gemini green-lit, schema wiring exists. Needs per-session approval before code starts. 9 TDD tasks: kernel skeleton → `__call__` → sklearn API → custom optimizer → wire into `analyze_rolling_correlations` → Script 05 → runner → first YAML → smoke run on Source layer.

### After Gibbs validates on Source
Scale to Skin + Background. Validation test: Gibbs should reduce chronic Z-score 0.5–0.9 near shelf break. It should NOT improve Pacific Blob Z-spikes (those are genuine temporal non-stationarity, not coastal-regime confusion — if Gibbs fixes them, something is wrong).

### Medium term
1. **Vertical delta comparison script** (`04_ae_vertical_compare.py`): load audit CSVs from all three layers, plot OHC trend and anisotropy ratio by depth on same axes. This is the primary deliverable — is Source warming faster than Background?
2. **SST cross-validation**: collocate Skin layer OHC against OISST satellite data to validate pipeline before trusting the deeper-layer comparisons. OISST (0.25°/daily, same ERDDAP infrastructure) recommended.

### Long term
- **Tier B:** MLflow/W&B experiment tracking with per-window metrics in a tracked store
- **Tier C:** pip package, Docker workers, public dashboard
- **LinkedIn demo:** interactive focus slider (Gibbs resolution vs. stationary smoothing), once 3-layer comparison is published
