# Claude TODO — ArgoEBUSAnalysis

Last updated: 2026-04-01

---

## Priority 1: Switch to `californiav2` Domain (Consolidated FX2 Strategy)

- [ ] **Re-run Cloud Ingestion (Script 02) with FX2 High-Res Temporal Resolution**
  - **Action:** Re-run Script 02 for all three layers (Skin, Source, Background) using:
    - `region='californiav2'` (lat [30, 45], lon [-130, -115])
    - **`time_step=10.0`** (Aligns data bins with Argo 10-day heartbeat to resolve aliasing)
  - New run_id prefix: `californiav2_20150101_20151231_res0_5x0_5_t10_0_d{range}`

- [ ] **Execute GPR Analysis (Script 05/07)**
  - **Action:** Once `t10_0` parquets are in S3, run:
    - `conda run -n ebus-cloud-env python 05_ae_update_tomatern0.5.py` (Skin)
    - `conda run -n ebus-cloud-env python 07_ae_deeper_layers.py` (Source + Background)
  - No parameter overrides needed — all canonical guardrails are baked into defaults.
  - Targets: Confirm Undercurrent meridional signature in Source layer; assess May 2015 Blob onset at depth.

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

