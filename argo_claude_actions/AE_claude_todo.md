## 2026-05-02 — [DONE] MLOps Foundation + All Gemini Audit Gaps — COMPLETE

**Status:** COMPLETE. main has 54 tests. All 5 Gemini audit gaps fixed and merged.

- PR #1 (`feat/mlops-phase2`): Phases 1–5 + Gap 1 fix — merged 2026-05-02
- PR #2 (`fix/mlops-audit-gaps-2-5`): Gaps 2–5 — merged 2026-05-02
- Plan: `docs/superpowers/plans/2026-05-02-mlops-audit-gaps-2-5.md`

### Next: Science — californiav3 domain definition
Before any new experiment runs, complete float census analysis:
1. Run `09_ae_longterm_float_census.py` — builds density maps 1999–2024 (untracked, needs commit first)
2. Run `09b_ae_analyze_float_census.py` — surfaces domain-recommendation stats
3. Share output with Gemini to define `californiav3` domain bounds
4. New run must use `aebus analyze` + a proper YAML config in `configs/californiav3/`

Last updated: 2026-05-02 (session 9)

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

