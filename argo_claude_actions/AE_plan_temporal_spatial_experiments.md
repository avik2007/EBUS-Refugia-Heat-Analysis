# Experiment Plan: Temporal Aliasing & Spatial Bound Saturation

**Date:** 2026-04-01
**Prompted by:** Gemini multi-layer analysis `2026-04-01_2015_CCS_MultiLayer_Analysis.md`
**Status:** T3 and S1 complete (2026-04-01). T1 queued. See results below.

---

## Problem 1: Temporal Aliasing ("The Sampling Beat")

### Root Cause

`scale_time_days` oscillates with a ~30-day period across ALL depth layers,
alternating between saturation at the 45-day upper bound and collapse to <10 days.

Beat frequency arithmetic:
- Argo resurface cycle: ~10 days
- Window step: 15 days
- Beat period = 1 / |1/15 − 1/10| = **30 days** ← matches observed oscillation

The GP optimizer alternately sees a well-populated window and a half-empty one.
This is a sampling artifact; Gemini's hypothesis that deeper water would suppress
it was **disproven** — the oscillation is identical across Skin, Source, and Background.

### Experiments

| ID | Change | File | Risk | Status |
|----|--------|------|------|--------|
| **T3** | `time_ls_bounds_days=(15.0, 45.0)` (lower bound 2→15d) | `05_ae_update_tomatern0.5.py` | Low | **Done** — partial (std 14.99→11.82, min 2.1→16.3d) |
| **T1** | `step_size_days=10` (align to Argo cycle) | `05_ae_update_tomatern0.5.py` | Medium | **Done** — no improvement (std 14.87 ≈ baseline 14.99). Root cause was wrong — see note. |
| **T2** | `step_size_days=30` (sub-harmonic) | `05_ae_update_tomatern0.5.py` | Medium | **Done** — oscillation eliminated. scale_time=45.0d in all 12 windows (std=0.00). Confirmed artifact. |
| **T4** | `window_size_days=90` (wider window) | `05_ae_update_tomatern0.5.py` | High | Deprioritized |

**T3 rationale:** Lowest-risk first. Prevents optimizer collapse to sub-15d scales
that are unphysical for the ocean (fastest internal adjustment timescales ~15-30d).
If T3 does not fully suppress the oscillation, T1 (step=10d) is the definitive fix.

**Expected T3 outcome:** `scale_time_days` stays in [15, 45] range in all windows.
Oscillation amplitude shrinks. RMSRE may rise slightly in sparse windows.

**Watch for:** Std Z > 1.1 in any window → model overconstrained on a truly fast event.

**Output run_id suffix:** `_3dmatern_w45_t3lb15`

---

### ROOT CAUSE REVISION (after T1 result, 2026-04-01)

**The aliasing is NOT caused by the Argo 10-day resurface cycle. It is caused by the 30-day time bin width in the input data.**

Evidence: with step=10d (T1), 11/33 consecutive window pairs contain **identical data** (same RMSRE, same n_bins to 8 decimal places). This happens because the data is pre-binned into 30-day time bins (`time_step=30.0` in Script 02). Any window step shorter than 30 days will produce consecutive windows that differ only in which 30-day bins fall inside — and if both windows straddle the same bin boundaries, they see identical observations.

The aliasing period of ~30 days is the bin width, not the Argo cycle. Aligning the step to 10d (or any value not divisible by 30d) cannot solve this — you get duplicate windows instead of aliased ones.

**Structural fix options (require Gemini decision before implementation):**

| Option | Action | Cloud Run Needed? |
|--------|--------|-------------------|
| **FX1** | Re-run Script 02 with `time_step=15` → finer bins, 2 bins per window | Yes |
| **FX2** | Re-run Script 02 with `time_step=10` → 4-5 bins per window (matches Argo cycle) | Yes |
| **FX3** | Accept: use `step_size_days=30` so windows never share the same bin set | No (T2) |
| **FX4** | Accept the oscillation as a data artifact; use T3 lower bound as a physical floor | No |

FX3 (T2) is the no-cloud-run path: a 30-day step means each window advances by exactly one bin, so there are no duplicate windows. The oscillation would become a genuine signal (or disappear) rather than an aliasing artifact.

**Recommend FX3 as next immediate step, then Gemini review of FX1/FX2.**

---

## Problem 2: Spatial Bound Saturation at Depth

### Root Cause

`spatial_l_bounds = (1e-2, 5)` in `argoebus_gp_physics.py:1140` (scaled units).
Background layer (500–1000m) spatial scales cluster at ~23.5°, which is the
physical equivalent of the upper bound. The optimizer saturates before finding
the true correlation scale.

Consequence: anisotropy ratios in the Background layer are artificially
compressed (both lat and lon hit the same wall → ratio → 1.0 artifact).
The May 2015 RMSRE=5.72%, Std Z=2.02 failure may be partly due to this.

Gemini recommendation: widen to 10× standard deviations for depth > 500m.

### Experiments

| ID | Change | File(s) | Risk | Status |
|----|--------|---------|------|--------|
| **S1** | Add `spatial_ls_upper_bound` param; pass `10` for Background | `argoebus_gp_physics.py`, `05_ae_update_tomatern0.5.py`, `07_ae_deeper_layers.py` | Low-Med | **Done** — minimal effect (3/23 windows changed; scales already exceeded 23.5°) |
| **S2** | Domain-adaptive bounds from data spread | `argoebus_gp_physics.py` | Medium | Low priority — Background scales already unconstrained in most windows |

**S1 implementation:**
1. `argoebus_gp_physics.py:998` — add `spatial_ls_upper_bound=5` to `analyze_rolling_correlations()` signature
2. `argoebus_gp_physics.py:1140` — `spatial_l_bounds = (1e-2, spatial_ls_upper_bound)`
3. `05_ae_update_tomatern0.5.py:40` — add `spatial_ls_upper_bound=5`, `run_suffix=""`, `time_ls_bounds_days=(2.0, 45.0)` parameters; thread through to `analyze_rolling_correlations()` call
4. `07_ae_deeper_layers.py:58` — Background call with `spatial_ls_upper_bound=10`, `run_suffix="_s1ub10"`

**Expected S1 outcome:** Background layer scales no longer cluster at ~23.5°.
Anisotropy ratios spread to reflect true spatial structure (not a wall artifact).
May 2015 window Std Z may improve if the misfit was bound-driven.

**Output run_id suffix:** `_3dmatern_w45_s1ub10`

---

## Verification Protocol

For each experiment:
1. Run with `conda run -n ebus-cloud-env python <script>` — confirm no optimizer `ConvergenceWarning`
2. Open new audit CSV
3. **T experiments:** check `scale_time_days` range and plot `temporal_persistence_*.png`
4. **S experiments:** check `scale_lat_bin` / `scale_lon_bin` max values and `lat_lon_evolution_*.png`
5. Compare median RMSRE and Std Z range to baseline (`_3dmatern_w45`)
6. Record outcome in this file (update Status column) and in `AE_claude_recentactions.md`

---

## Baseline Reference

| Layer | Run ID suffix | Median RMSRE | Std Z range | Time scale range |
|-------|--------------|--------------|-------------|-----------------|
| Skin | `_3dmatern_w45` | ~3.4% | 0.73–1.11 | 2–45d (oscillating) |
| Source | `_3dmatern_w45` | ~4.2% | 0.68–1.18 | 2–45d (oscillating) |
| Background | `_3dmatern_w45` | ~4.8% | 0.53–2.02 | 2–45d (oscillating) |

---

## Final Outcome (2026-04-01)

### Gemini Verdicts

**Temporal aliasing:** Adopt FX2 (`time_step=10.0`). The structural aliasing at
30d bins is unacceptable for heat-transport fingerprinting. High-res 10d bins will
resolve the true physical coherence of the Undercurrent.

**Anisotropy vertical profile:** Meridional dominance in Skin Layer (Aug–Sep) is
physically consistent with the southward CC jet, not an artifact. The vertical
fingerprint is confirmed: meridionality persists into the Source layer (CUC
signature), zonal dominance only emerges below 500m (Background Layer).

**May 2015 Background failure:** Confirmed genuine non-stationarity (Pacific Blob
onset). Expect Z > 2.0 to persist in the new high-res run. If so, flag as a
physical violation of GP stationarity at that window.

### Canonical Configuration (Permanent)

All experiment guardrails are now baked into `run_diagnostic_inspection()` defaults:

| Parameter | Value | Source |
|---|---|---|
| `region` | `californiav2` | Prior todo item |
| `time_step` (Script 02) | `10.0` | FX2 |
| `step_size_days` | `10` | Matches bin width |
| `time_ls_bounds_days` | `(15.0, 45.0)` | T3 permanent |
| `spatial_ls_upper_bound` | `10` | S1 permanent (all layers) |

New run_id prefix: `californiav2_20150101_20151231_res0_5x0_5_t10_0_d{range}`

### Experiment Summary Table

| ID | Change | Result | Disposition |
|---|---|---|---|
| T3 | temporal lower bound 2→15d | Partial: min 2.1→16.3d, std 14.99→11.82 | Permanent |
| S1 | spatial upper bound 5→10 (Background) | Minimal: 3/23 windows changed | Permanent, all layers |
| T1 | step_size 15→10d | No improvement (std 14.87≈14.99) | Wrong hypothesis |
| T2 | step_size 15→30d | Oscillation eliminated (std=0.00) | Confirmed artifact; FX2 chosen instead |
| FX2 | time_step=10 (cloud re-run) | Pending cloud run | Priority 1 |
