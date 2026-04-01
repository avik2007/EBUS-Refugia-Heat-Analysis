# Claude TODO — ArgoEBUSAnalysis

Last updated: 2026-04-01

---

## Next Up: Implement Script 09 — Long-Term Argo Float Census

**Plan file:** `argo_claude_actions/2026-04-01_float_census_plan.md`

Build `09_ae_longterm_float_census.py` to map Argo float density across the broad
`california` domain (lat [25,50], lon [-140,-110]) for 1999–2025 on a 5°×5° grid.
Output: small-multiples PNG + CSV hotspot table. Purpose: empirically define
`californiav3` bounds based on where floats actually are at depth, fixing the Source
Layer sparsity problem that caused the FX2 GPR regression.

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

