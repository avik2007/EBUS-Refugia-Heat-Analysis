# ⚠️ GEMINI MLOPS AUDIT — 5 GAPS TO FIX (2026-04-30) [PLANNED, NOT STARTED]
**Spec:** `docs/superpowers/specs/2026-04-30-mlops-gemini-audit-verdict.md`
**Do NOT run new science experiments until Gap 1 is fixed.**
**Plan is below — resume here next session.**

---

## [NEXT] MLOps Audit Fixes — 5 Gaps (Priority Order)

### Gap 1 — CRITICAL: Kwarg mismatch in runner.py (spatial bounds ignored)

**File:** `ArgoEBUSCloud/ebus_core/runner.py:run_analysis():dispatch_kwargs`
**File:** `ArgoEBUSCloud/05_ae_update_tomatern0.5.py:run_diagnostic_inspection()`

**Problem:** `run_diagnostic_inspection` expects `spatial_ls_upper_bound` (single scalar upper
bound, default=10). `runner.py:dispatch_kwargs` never passes it. So `lat_ls_bounds` and
`lon_ls_bounds` from the config are silently ignored; the physics engine always uses 10.

**Fix:**
```python
# In run_analysis():dispatch_kwargs, add:
"spatial_ls_upper_bound": (
    max(cfg.gpr.lat_ls_bounds[1], cfg.gpr.lon_ls_bounds[1])
    if cfg.gpr.lat_ls_bounds is not None and cfg.gpr.lon_ls_bounds is not None
    else None  # omit = script default (10) applies
),
```
Then update `_call_run_diagnostic_inspection` shim to filter out `None`-valued keys before
forwarding to the real function (or pass only when not None).

**Test:** Add test asserting that when config has `lat_ls_bounds=(1.0, 8.0)` and
`lon_ls_bounds=(1.0, 6.0)`, the shim receives `spatial_ls_upper_bound=8.0`.

---

### Gap 2 — MEDIUM: Ghost successes in registry

**File:** `ArgoEBUSCloud/ebus_core/runner.py:run_analysis()`
**File:** `ArgoEBUSCloud/ebus_core/manifest.py:append_registry()`

**Problem:** `append_registry` fires after `write_manifest` but neither checks that the
science outputs (audit CSV, snapshots) actually landed on disk. A Dask crash that returns
a partial result would register as a successful run.

**Fix:** After `dispatch_result`, verify the audit CSV exists on disk before writing the
manifest. Add a `status` field to registry entries: `"finalized"` = outputs verified,
`"incomplete"` = manifest written but outputs missing.

```python
# In run_analysis(), after dispatch:
if audit_csv and not Path(audit_csv).exists():
    status = "incomplete"
else:
    status = "finalized"
# Pass status into build_manifest → outputs block → registry line
```

**Test:** Monkeypatch dispatch to return a path that doesn't exist; assert registry `status="incomplete"`.

---

### Gap 3 — LOW-MEDIUM: Incomplete depth validation

**File:** `ArgoEBUSCloud/ebus_core/config_schema.py`

**Problem:** `IngestionConfig` and `AnalysisConfig` both have `_depth_range_ordered`, but
`PhysicsParamsBlock.ohc_depth_top_m` / `ohc_depth_bot_m` are not cross-validated against
`depth_range`. Non-physical bounds (e.g., ohc_top > depth_range[1]) can be written silently.

**Fix:** Add an `AnalysisConfig` model-level validator that checks:
```python
if cfg.physics.ohc_depth_top_m is not None:
    assert cfg.physics.ohc_depth_top_m >= cfg.depth_range[0]
if cfg.physics.ohc_depth_bot_m is not None:
    assert cfg.physics.ohc_depth_bot_m <= cfg.depth_range[1]
```

**Test:** Verify config with `ohc_depth_bot_m > depth_range[1]` raises ValidationError.

---

### Gap 4 — MEDIUM: _fmt_dec / derive_run_id not centralized

**Files:** `ArgoEBUSCloud/ebus_core/runner.py`, `ArgoEBUSCloud/ebus_core/ae_utils.py`

**Problem:** `_fmt_dec` (float → filesystem-safe string) is private to `runner.py`. Same
logic exists in legacy scripts. Drift between them would silently break the config↔result
naming link.

**Fix:** Move `_fmt_dec` and `derive_run_id` to `ebus_core/ae_utils.py`. Re-import in
`runner.py` (no behavior change). Legacy scripts can optionally import from there too.

**Test:** Existing `test_derive_run_id_*` tests cover behavior; just update import path.

---

### Gap 5 — LOW: Backfill metadata not structured

**Files:** `configs/california/*.yaml`, `configs/californiav2/*.yaml`

**Problem:** Null fields in backfilled YAMLs are explained only in the `description` string.
No structured `metadata` block distinguishes "recovered" fields from "assumed defaults."

**Fix:** Add to each backfilled YAML:
```yaml
metadata:
  recovered_fields: [region, date_start, date_end, lat_step, lon_step, time_step,
                     depth_range, run_suffix, noise_vals_audit]
  assumed_fields: [noise_val, time_ls_bounds_days, lat_ls_bounds, lon_ls_bounds]
  backfill_date: "2026-04-30"
  source_run_id: "<run_id>"
```
Update `backfill.py:reconstruct_analysis_config()` to emit this block.
Update `config_schema.py` to accept optional `metadata: dict` (extra="ignore" or typed).

**Test:** Verify round-trip: backfill YAML parses cleanly and `metadata.assumed_fields`
contains the expected keys.

---

**Execution notes:**
- Gap 1 must land before any new analysis runs on californiav3.
- Gaps 2–5 can follow in order; each is independent.
- TDD: write failing test first for each gap, then fix.
- Commit each gap separately (clear audit trail on the PR).

Last updated: 2026-04-30 (session 8)

---

## 2026-04-30 (session 8) — [DONE] MLOps Phase 5 Complete — Ready to Commit

**Status:** Phase 5 COMPLETE. All docs written. 44 tests still passing.

Phase 4 + Phase 5 changes are uncommitted. Commit when ready:
- `ArgoEBUSCloud/ebus_core/backfill.py` (new)
- `ArgoEBUSCloud/10_ae_backfill_configs.py` (new)
- `ArgoEBUSCloud/aebus_cli.py` (new)
- `ArgoEBUSCloud/ebus_core/config_schema.py` (M)
- `ArgoEBUSCloud/test_mlops_foundation.py` (M)
- `configs/california/` (15 YAMLs, new)
- `configs/californiav2/` (3 YAMLs, new)
- `configs/README.md` (new)
- `README.md` (M — MLOps section added)
- `CLAUDE.md` (M — Config-Driven Workflow section added)
- `ae_file_structure.txt` (M — 8 new files documented)

Next: create commit on `feat/mlops-phase2` and optionally open PR to `main`.

Last updated: 2026-04-30 (session 8)

---

## 2026-04-26 — [HISTORICAL] MLOps Foundation: Phase 1+ resume (superseded by session-3 entry above)

---

## 2026-04-25 — [SUPERSEDED 2026-04-26] Execute MLOps Foundation Plan

**Status:** Spec + plan written, committed (commit `d119377`). Implementation NOT started.
Brainstorm phase from 2026-04-24 entry below is COMPLETE and superseded by this entry.

**Spec:** `docs/superpowers/specs/2026-04-25-mlops-foundation-design.md`
**Plan:** `docs/superpowers/plans/2026-04-25-mlops-foundation.md`

**What this delivers:** A-tier of the MLOps showcase — config-driven runs (YAML per
region/layer/experiment) + reproducibility manifests (config hash, git SHA, conda env,
S3 lineage) + thin `aebus` CLI. Additive layer atop existing scripts; no refactor.
B-tier (MLflow/W&B) and C-tier (pip pkg + Docker + dashboard) deferred to future specs.

**How to resume next session:**
1. Read the plan top-to-bottom before touching anything.
2. Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` skill.
3. Per project CLAUDE.md hard stop: do NOT start implementation until Avik approves the plan in the new session.

**Phase order (16 tasks total, ~6-7 days solo):**
- Phase 0: install pydantic + pytest into `ebus-cloud-env`, update env spec files (1 task)
- Phase 1: config_schema.py + manifest.py (8 tasks: schema, validators, YAML loader, hash, git/env capture, manifest IO, collision detector, registry)
- Phase 2: runner.py wrappers around existing GPR + ingestion fns (5 tasks)
- Phase 3: aebus CLI — validate / analyze / ingest / list / show (3 tasks)
- Phase 4: backfill `configs/` from existing AEResults/aelogs/ (2 tasks)
- Phase 5: docs (README + CLAUDE.md + ae_file_structure.txt + recentactions) (3 tasks)

**Open dependency for Task 2.5:** verify `02_ae_cloud_run.py` exposes a top-level
`run_ingestion_pipeline(**kwargs)` callable. If not, surgical extract from `__main__`
required as a separate commit before runner can dispatch.

**Pending Gemini review:** spec entry at top of `argo_gemini_actions/AE_gemini_todo.md`
asks Gemini to flag any science / reproducibility gaps before implementation begins.
Worth checking Gemini's response before starting Phase 1.

**Hard stop reminder (project CLAUDE.md):** even within an approved plan, every
sub-step must be announced verbosely before execution. Per-task TDD discipline
(write failing test → run → implement → run → commit) is non-negotiable.

Last updated: 2026-04-25

---

## 2026-04-24 — [SUPERSEDED 2026-04-25] Engineering & MLOps Showcase Brainstorm

**Status: COMPLETE.** Brainstorm session 2026-04-25 produced spec + plan
above. This entry retained for historical context only.

Original goal: project must double as MLOps demo. Brainstorm scoped sequenced
practitioner-first → hiring-target showcase, identified region scaling +
reproducibility as top pains, decomposed into A/B/C tiers (A = this spec,
B = MLflow/W&B next, C = pkg + Docker + dashboard long-term).

Last updated: 2026-04-25

---

## 2026-04-11 — [NEW DIRECTIVE] The RG-Gibbs Non-Stationary Model (Approved)

**Draft spec:** `docs/superpowers/specs/2026-04-11-rg-gibbs-nonstationary-gpr-design.md`

### Decisions locked
- [x] Stay in sklearn — custom `GibbsKernel` subclass, no GPflow/TF
- [x] Approach B — new `validate_moving_window_nonstationary()`, existing function untouched
- [x] RG climatology → S3 Zarr via `00_ae_rg_climatology_ingest.py`
- [x] New cloud ingestion run with `californiav3`, depths `d0_100` / `d100_500` / `d500_1500`
- [x] `get_vertical_layers()` → Response [0,100], Source [100,500], Background [500,1500]
- [x] Interactive focus slider scoped separately (see LinkedIn demo task below)

### [BLOCKED] Open question — GibbsKernel: l(x) functional form
Brainstorming paused here. Avik reviewing GP oceanography literature.
Gemini science input also requested (see AE_gemini_todo.md).

**The issue:** Any fixed functional form (sigmoid, linear ramp, dist_to_coast profile) 
prescribes where and how the lengthscale transitions — which contradicts the Gibbs 
motivation. Candidate approaches:
1. Data-density-driven l(x): l = l_max − (l_max−l_min) × normalized_float_density(x)
2. Fully learnable parametric: expose l_min, l_max, rate α to sklearn optimizer
3. Literature-guided: adopt established GP oceanography practice

**Resume point:** Section 3 of brainstorm — GibbsKernel class design.
Run `/brainstorm` and reference `docs/superpowers/specs/2026-04-11-rg-gibbs-nonstationary-gpr-design.md`.

### Remaining implementation tasks (do NOT start until l(x) resolved)
- [ ] **`00_ae_rg_climatology_ingest.py`** — Copernicus fetch → S3 Zarr
- [ ] **Cloud ingestion run** — californiav3 + new layer bounds
- [ ] **`GibbsKernel` class** — in `argoebus_gp_physics.py`
- [ ] **`load_rg_mean()` helper** — S3 Zarr read + interpolation
- [ ] **`validate_moving_window_nonstationary()`** — full Gibbs + RG mean GPR engine
- [ ] **`get_vertical_layers()` update** — ae_utils.py

### LinkedIn Demo (scope separately, after GPR is validated)
- [ ] **Interactive Focus Slider** — browser demo showing Gibbs kernel resolution vs.
  standard global smoothing. Publish to LinkedIn once RG-Gibbs model is validated.

Last updated: 2026-04-11

---


## Priority 1: Diagnose FX2 GPR Results — Gemini Review Required

Cloud run and GPR analysis are complete (2026-04-01). Results are mixed and require
Gemini science review before proceeding. See `AE_claude_recentactions.md` for full
output files and per-window tables.

- [x] **Re-run Cloud Ingestion (Script 02) with FX2 High-Res Temporal Resolution** — DONE
  - `californiav2_20150101_20151231_res0_5x0_5_t10_0_d{0_100, 150_400, 500_1000}.parquet` in S3

- [x] **Execute GPR Analysis (Script 05/07)** — DONE (results problematic, see below)

- [ ] **[For Gemini] Source Layer regression — diagnose root cause**
  - Source Layer median RMSRE degraded from ~4.2% (t30 baseline) to 8.13% (t10 run).
  - Only 8/34 windows pass 5% threshold. Max RMSRE 22.09%. Extreme anisotropy ratios
    (up to 35.75) are non-physical.
  - Worst windows (day centers): 5952, 6032, 6072, 6082, 6132, 6142, 6152, 6172, 6182, 6192.
  - Z spike: window 6022 std_z=15.63. Window 6172 std_z=4.48.
  - Key audit: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45/audit_californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45.csv`
  - **Gemini question:** Is the Source Layer degradation from (a) the tighter californiav2
    domain clipping float trajectories at depth, (b) 10d bins exposing genuine sparsity
    that 30d bins masked, or (c) a GPR configuration issue?

- [ ] **[For Gemini] scale_time_bin saturates at 45d in all Skin + Source windows**
  - Every window in Skin and Source hits the `time_ls_bounds_days` upper limit.
  - No aliasing oscillation (FX2 worked), but still pegged to 45d.
  - Background layer is healthy: scale_time_bin varies 26–45d in mid-year.
  - **Gemini question:** Should we widen `time_ls_bounds_days` upper bound for Skin/Source?
    Or is 45d saturation physically meaningful (ocean memory > window width)?

- [ ] **[For Gemini] Background Layer Z=18.73 spike at window 6102.5 (~Sep 2015)**
  - RMSRE only 2.67% but std_z=18.73. Likely Pacific Blob peak non-stationarity.
  - Prior Gemini verdict: genuine physical event, flag if Z > 2.0 persists.
  - Key audit: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45/audit_californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45.csv`
  - **Gemini question:** Confirm Z=18.73 is the Blob onset. Mark as stationarity violation?

---

## Priority 2: Experiments — Temporal Aliasing & Spatial Bounds (Resolved)

- [x] **[For Gemini] Temporal persistence architecture decision**
  - **Gemini Verdict:** Adopt **FX2 (`time_step=10.0`)**. Structural aliasing at 30d bins is unacceptable for heat-transport fingerprinting. High-res temporal bins will allow us to see the true physical coherence of the Undercurrent.
- [x] **[For Gemini] Anisotropy vertical profile — flag for science review**
  - **Gemini Verdict:** Meridional dominance in Skin (Aug-Sep) is physically consistent with the southward CC jet. The vertical fingerprint is confirmed: meridionality persists at depth (Source layer) while zonal dominance only emerges below 500m (Background).

- [ ] **Background Layer window 5955–6000 (May 2015) — Case Study**
  - Gemini confirms this is a **genuine non-stationarity event** (Pacific Blob onset).
  - Task: Compare Z-score in the new `t10_0` high-res run; if Z > 2.0 persists, mark as physical violation of stationarity.

---

## Priority 3: Analysis and Comparison

- [ ] **Vertical Delta Comparison Script** (new script, e.g., `04_ae_vertical_compare.py`)
  - Load audit CSVs from all three depth layers
  - Plot: OHC trend for each layer on same axes
  - Plot: Anisotropy Ratio by depth layer over time
  - Key question: Is Source Layer (150–400m) warming faster than Background (500–1000m)?

- [ ] **Seasonal Anisotropy Report**
  - From the Skin Layer audit, compare Jan vs. Aug Anisotropy Ratios
  - Already partially done: confirmed ratio ~0.36 Jan, ~0.49 Aug from 2015 logs
  - Formalize this into a plot showing ratio vs. month for a full year

- [ ] **SST Cross-Validation: Argo Surface vs. Satellite SST**
  - Collocate Argo Skin Layer (0–100m) binned temperature against OISST or MUR SST
    for the California region, 2015.
  - **OISST** (NOAA OI, 0.25°/daily, 1981–present): available via ERDDAP at
    `https://coastwatch.pfeg.noaa.gov/erddap/`. Coarser but long record — good match
    to the 0.5° Argo grid.
  - **MUR** (NASA, 0.01°/daily, 2002–present): available via NASA PODAAC ERDDAP.
    Finer resolution but more processing overhead.
  - Compute: bias, RMSE, and Pearson r between collocated pairs, by month.
  - Purpose: builds confidence that the Argo binning + OHC pipeline is capturing
    the correct SST signal before the deeper-layer stealth warming comparison is trusted.
  - Recommended start: OISST (resolution matches Argo grid; same ERDDAP infrastructure
    already used for float trajectories).

