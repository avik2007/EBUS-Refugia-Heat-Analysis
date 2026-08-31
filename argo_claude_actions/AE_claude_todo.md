## 2026-08-30 — [ACTIVE #-2] Add test coverage for the physics/GPR engine (TOP PRIORITY)

**Priority:** Highest. Do before further kernel tuning or any external-facing claim.
The whole GPR + thermodynamics core has zero unit coverage — only the MLOps wrapper
(`test_mlops_foundation.py`, ~66 tests) is tested. `test_pipeline.py` is a live-network
Dask smoke script, not pytest, and is not wired to CI.

**Why now:** history review (session 2026-08-30) found every serious silent bug landed in
the untested engine. The `GibbsKernel` `time_ls` units bug (056c34f) corrupted a published
LinkedIn/interview claim (44d→54d→58d depth trend, since retracted). `lat_ls_bounds`/
`lon_ls_bounds` were silently ignored for weeks (be013be, Gap 1) — engine used a hardcoded
default. Neither had a test.

**Priority order for new suites:**
1. **`GibbsKernel` unit tests** (`argoebus_gp_physics.py`). Pure function, trivial to test.
   - `k(Δt = half-window)` == `exp(-half_window_days / time_ls)` — the exact units bug (fix
     added `test_gibbs_kernel_time_ls_converts_normalized_dt_to_days`; extend it).
   - Gibbs vs Matérn path parity: near-equal `K` when lengthscales set equal; `np.isfinite(K).all()`.
   - `get_params` / `clone_with_theta` round-trip preserves every constructor arg (incl. `window_size_days`).
   - PSD check: `K` symmetric, eigenvalues ≥ -1e-8 on a small random `X`.
2. **`argoebus_thermodynamics.py` golden values.** Highest blast radius, currently unguarded.
   - Hand-computed T/S/P profile → known OHC in J/m² (assert to tolerance).
   - Unit check (`units == 'J/m^2'`), monotonicity (warmer water → more OHC).
   - Depth clipping: `depth_min`/`depth_max` actually respected (the "CRITICAL: Respecting
     chosen depth" comments in Script 02 mark the anxiety).
3. **config → dispatch contract tests** (`runner.py`). "Is every config field actually
   threaded to the engine?" — the Gap 1 class of bug. Bounds present → correct kwarg;
   null bounds → key absent (be013be already added a version of this — generalise it).
4. **ERDDAP URL-builder unit test** (`02_ae_cloud_run.py`). Offline, no network.
   - Assert `>`/`<` encoded as `%3E`/`%3C` (fsspec treats bare ones as glob).
   - Assert host is `erddap.ifremer.fr` (www→erddap redirect not followed with comparison ops).
5. **Repo-layout test.** Resolved output dir `== repo_root/AEResults` (lesson #2, commit 74d20c2
   — silent writes into `ArgoEBUSCloud/AEResults/`).
6. **Signature-drift test.** Each pipeline entry fn matches `(region, lat_step, lon_step,
   time_step, depth_range)` (lesson #3).

**Also:** wire `test_pipeline.py` (or a trimmed offline version) into the same pytest run,
or delete it and replace with a mocked-ERDDAP integration test.

Last updated: 2026-08-30

---

## 2026-07-17 — [ACTIVE #-1] Finish Diebold-Mariano significance test (HLN correction missing)

**Priority:** Resume first next session — pending correctness check before the
significance claim is used anywhere external-facing (LinkedIn, Gemini briefing, interview prep).

Full findings: `argo_claude_actions/dm_test_review_2026-07-17.md`.

`compare_kernels.py` (Gemini, this session) runs clean and its numbers check out, but it's
missing half of the methodology agreed in session 18: no small-sample Harvey-Leybourne-Newbold
(HLN) correction (p-values use standard normal instead of t(N-1), likely overstating
significance at N=34), and `lag=4` is hardcoded rather than derived from
`window_size_days/step_size_days − 1` (=3.5 currently).

Next step: add HLN correction + derive lag from config, re-run, and check whether the
"Gibbs statistically superior on all 3 layers" verdict survives — **Source** layer is the
one most at risk (closest p-value to 0.05, bootstrap CI already nearly crossing zero).

Last updated: 2026-07-17

---

## 2026-07-17 — [ACTIVE #0] Resume LinkedIn post on Gibbs kernel results

**Status:** Paused mid-session 19, pending the time_ls investigation (now resolved, see #1 below).

Post drafted (RMSRE + Z-std calibration charts built, real-fitted-kernel field/uncertainty
illustration built — see session 19 recentactions for file paths, all in scratchpad, not yet
moved into the repo). Core claims (RMSRE down 18–27%, calibration spread ~10x tighter) are
still valid post-fix. Drop the time-persistence angle entirely — do not resurrect the
"44d→54d→58d, increasing with depth" framing (see #1 below, session 19 lesson #6).

Next step: decide whether to keep the post as RMSRE + calibration only (2 charts + field
illustration), or fold in a short "found and fixed a units bug mid-illustration" angle as
its own point of engineering credibility. Re-generate any charts/images since prior session's
scratchpad files are ephemeral (session-scoped tmp dir, will not persist).

Last updated: 2026-07-17 (session 19)

---

## 2026-07-17 — [ACTIVE #1] Brief Gemini on Gibbs 3-layer results (time_ls claim CORRECTED session 19)

**Priority:** Do this first next session before any further tuning.

Gemini needs to see the full Gibbs vs Matérn comparison and weigh in on:
1. **Z-score calibration improvement**: Gibbs collapses std_Z to mean~0.98, std~0.07–0.10 across all layers.
   Matérn had mean 1.13–1.72, std up to 2.63, max 11.35 (Background Blob windows). Unaffected by
   session 19's fix — reconfirmed post-fix (see recentactions 2026-07-17).
2. **RMSRE gains**: Skin 4.25%→3.49%, Source 3.05%→2.54%, Background 2.50%→1.84%. Unaffected by
   session 19's fix — reconfirmed post-fix (3.71% / 2.63% / 2.03%, same ballpark).
3. **SUPERSEDED — do NOT brief Gemini on this as stated**: "Time persistence now learnable... Skin
   44d, Source 54d, Background 58d — increasing with depth" was based on a units bug in `GibbsKernel`
   (dt normalized vs time_ls in days — see lesson #6 in `AE_claude_lessons.md`). Fixed in session 19.
   Post-fix, `time_ls` pegs at whatever bound is given (tested to 200d) for the large majority of
   windows in all 3 layers — it is **not currently resolvable** with a 45-day rolling window. If
   briefing Gemini on temporal persistence, report it as "≥200d, unresolvable at this window width"
   for all three layers, not a graded depth trend. Open question for Gemini: is widening
   `window_size_days` itself (a bigger methodological change, deferred in session 19) worth pursuing
   to actually resolve this, or is "unresolvable at 45d" itself a usable/interesting finding?
4. **Remaining convergence warnings** (bounds still being hit):
   - `d_transition_bounds_km` upper bound 700km saturating on some windows → widen to 1000–1500km?
   - `anisotropy_lat_lon_ratio` lower bound 1.0 hit on ~40% of Source windows → allow down to 0.5?
5. **Science verdict**: Is Gibbs ready to be called the canonical kernel? Or more tuning first?

Audit CSVs (original, pre-fix): `AEResults/aelogs/californiav3_..._d{layer}_3dgibbs_w45/audit_*.csv`.
Audit CSVs (post-fix, session 19): `AEResults/aelogs/californiav3_..._d{layer}_3dgibbs_w45_timelsfix/audit_*.csv`.

Last updated: 2026-07-17 (session 19)

---

## 2026-05-26 — [ACTIVE #2] Update github.io portfolio with Gibbs results

After Gemini sign-off, update `docs/index.html`:
- Replace Matérn baseline metrics with Gibbs numbers in the Results section
- Add 3-layer Gibbs vs Matérn comparison table (RMSRE + Z-score)
- Add new kriging heat map snapshots (Gibbs versions) if visually cleaner
- Update "What's Next" Gibbs v2 card to reflect implementation complete

Last updated: 2026-05-26 (session 17)

---

## 2026-05-26 — [DONE] Presentation Slides

**Status:** COMPLETE (done before session 17).

Last updated: 2026-05-26 (session 17)

---

## 2026-05-26 — [DONE] Gibbs Post-Implementation: Validate, Scale, Tune

**Status:** COMPLETE (session 17, 2026-05-26).
- Temporal persistence plot fix: `scale_time_bin` was NaN on gibbs path → now stores `time_ls_days`
- `--force-overwrite` bug fixed in `runner.py` (verdict unbound, collision raised before delete)
- `time_ls_bounds_days` widened 45→90d in all 3 gibbs configs
- Skin + Background gibbs configs created and run
- Full 3-layer comparison: Gibbs beats Matérn on RMSRE and Z-calibration across all layers

Last updated: 2026-05-26 (session 17)

---

## 2026-05-04 — [DONE] RG-Gibbs Kernel Implementation

**Status:** COMPLETE. All 9 tasks done (session 16, 2026-05-26).
- `GibbsKernel` in `argoebus_gp_physics.py`: sigmoid l(x), learnable `[d_0, k, time_ls, anisotropy_ratio]`
- 12 new TDD tests (60 total, 5 pre-existing CLI failures unchanged)
- `configs/californiav3/californiav3_d150_400_gibbs.yaml` + smoke run: 32/34 pass, RMSRE 2.54%
- Kriging NaN bug fixed (effective scale at median dist_to_coast stored in `scale_lat_bin/lon_bin`)
- anisotropy_ratio made learnable (bounds 1.0–4.0); `_gibbs_optimizer` uses scipy L-BFGS-B + jac='2-point'

Last updated: 2026-05-26 (session 16)

---

## 2026-05-03 — [DONE] californiav3 Matérn Baseline Run (Path A)

**Context:** Float census done (09c, committed). californiav3 bounds confirmed in `ae_utils.py`
(Lat [30,48], Lon [-135,-115]).

**Steps:**
1. [x] Write 3 analysis YAMLs + 3 ingestion YAMLs in `configs/californiav3/` — all validate clean.
2. [x] Run ingestion — all 3 parquets on S3, registered in run_registry.jsonl.
3. [x] Run GPR — all 3 layers complete (session 11, 2026-05-03).
       YAML fix applied: `time_ls_bounds_days: [15.0, 45.0]` in all 3 analysis configs.
4. [x] Review results:
       Skin  0-100m:   median RMSRE 4.25%, max 6.45%, 27/35 pass. Z chronic 0.5-0.9 + spikes 5.77, 4.35.
       Source 150-400m: median RMSRE 3.05%, max 5.38%, 32/35 pass. Z chronic 0.5-0.9 + spikes 5.14, 4.94 (Aug-Sep).
       Background 500-1000m: median RMSRE 2.50%, max 3.92%, 35/35 pass. Z chronic low + extreme spikes 11.35, 9.28, 9.11 (Blob onset).
       Domain fix validated: Source improved from 8.13% (californiav2) → 3.05%.
       Cross-layer Z pattern: stationary Matérn cannot adapt near shelf-break — dist_to_coast Gibbs motivated.
5. [x] Share results with Gemini for science verdict — DONE 2026-05-04. Verdict: Gibbs green-lit
       with dist_to_coast as l(x) coordinate. Background Z-spikes confirmed as Pacific Blob.
       See `argo_gemini_actions/AE_gemini_recentactions.md` 2026-05-04 entry.

**Pipeline fixes landed in session 10 (54 tests still passing):**
- `02_ae_cloud_run.py`: `run_ingestion_pipeline()` wrapper (absorbs runner extras)
- `02_ae_cloud_run.py`: ERDDAP URL — encode `>/%3E`, `</%3C`; point to `erddap.ifremer.fr`
- `02_ae_cloud_run.py`: try/finally cluster cleanup wraps all post-cluster code
- `02_ae_cloud_run.py`: `client.run(_warm_cartopy_cache)` pre-warms coastline on workers
- `05_ae_update_tomatern0.5.py`: `**_` absorbs unknown runner kwargs (mode, kernel_type, etc.)

Last updated: 2026-05-04 (session 12)

---

## 2026-05-02 — [DONE] MLOps Foundation + All Gemini Audit Gaps — COMPLETE

**Status:** COMPLETE. main has 54 tests. All 5 Gemini audit gaps fixed and merged.

- PR #1 (`feat/mlops-phase2`): Phases 1–5 + Gap 1 fix — merged 2026-05-02
- PR #2 (`fix/mlops-audit-gaps-2-5`): Gaps 2–5 — merged 2026-05-02
- Plan: `docs/superpowers/plans/2026-05-02-mlops-audit-gaps-2-5.md`

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

