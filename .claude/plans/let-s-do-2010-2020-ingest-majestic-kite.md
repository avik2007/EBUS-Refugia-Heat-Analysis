# Plan: 2010-2020 ingest + d_0 LML sweep + d_0 distribution plots

## Context
`d_transition_km` (d_0) in the GibbsKernel has CV 0.6-0.74 across 34 rolling windows in all 3 layers (2015 only). User wants (1) that variance reported visibly: d_0 distribution plots annotated with CV, paths logged in `AE_claude_recentactions.md`; (2) an LML-vs-d_0 sweep over 2010-2020 to test whether the likelihood is flat in d_0 (identifiability). Goal downstream: a constant d_0 across layers as a structural prior for better OHC fit, not a reported metric.

## Verified blockers (read-only exploration)
- **Ingest drops dates.** `02_ae_cloud_run.py:154-163` `run_ingestion_pipeline` ignores `date_start/date_end`; `get_ae_config` falls back to registry `californiav3` time = 2015 (`ebus_core/ae_utils.py:40`). A 2012 config would silently re-ingest 2015.
- **Analysis same gap.** `05_ae_update_tomatern0.5.py:run_diagnostic_inspection` (l.54) calls `get_ae_config` without dates and swallows kwargs via `**_`; `runner.run_analysis` (l.174) doesn't pass dates.
- **LML not recorded.** Per-window record (gp_physics.py ~l.1700-1719) has `d_transition_km` but no LML. `cv_details` pkl is empty.
- `analyze_d0_distribution()` (`vertical_delta_analysis.py:311`) writes fixed filenames (would overwrite per year), layer keys hard-coded, no CV annotation on plot.
- Ingest is per layer per year: 3 layers x 11 years = 33 Coiled runs, S3 writes, needs Coiled + AWS creds + ERDDAP network. Cost/time unmeasured (agent guess: 5-15 min, cents each).

## Steps (each announced before execution; stop and report on surprises)
1. **Fix date plumbing** (minimal): forward `date_start/date_end` ISO strings from `run_ingestion_pipeline` -> `run_cloud_pipeline` -> `get_ae_config(start_date, end_date)` in `02_ae_cloud_run.py`; same for `run_diagnostic_inspection` in `05_ae_update_tomatern0.5.py` and `runner.run_analysis`. Add a test to `test_mlops_foundation.py` first (tests-first): non-2015 config reaches `get_ae_config` with the right dates.
2. **Cost/time probe:** ingest ONE layer-year (2010, d0_100) via `aebus_cli.py validate` then `ingest`; measure runtime, S3 size, row count. Report before launching the rest.
3. **Generate + run 33 ingest configs** (2010-2020 x d0_100 / d150_400 / d500_1000) by copying `configs/californiav3/californiav3_20150101_20151231_res0_5x0_5_t10_0_d0_100_ingest.yaml`, changing only dates/description.
4. **Add `record['lml'] = gp.log_marginal_likelihood_value_`** in `analyze_rolling_correlations` record dict (gp_physics.py ~l.1719). Test-first.
5. **33 analysis configs** from the `_gibbs_timelsfix` templates (500-1000 uses 50-1500 km bound), run free-fit Gibbs per layer-year -> audit CSVs now carry d_0 + LML.
6. **LML sweep script `d0_lml_sweep.py`** (repo root; signature `(region, lat_step, lon_step, time_step, depth_range, year)`): per window, clamp d_0 on grid 50,100,200,400,800,1500 km via narrow band (d0 +/- 0.5 km, NOT equal bounds, init inside band), re-optimise k/time_ls/anisotropy/noise, record LML. Outputs: CSV `AEResults/aelogs/d0_lml_sweep_<layer>_<year>.csv`, normalized LML-vs-d_0 curves in `AEResults/aeplots/d0_sweep/`. Sanity check: grid-max LML ~ free-fit LML. Time one window first.
7. **d_0 distribution plots with CV annotation:** generalise `analyze_d0_distribution()` with a `tag/year` param (no overwrite), annotate each layer's box with `CV = x.xx`. Produce per-year plots and one pooled 2010-2020 plot (layers side by side, CV in labels). Outputs under `AEResults/aeplots/vertical_delta/`, stats CSV under `AEResults/aelogs/`.
8. **Report:** add entry to `argo_claude_actions/AE_claude_recentactions.md` with exact plot/CSV paths and CV values; fix stale Gibbs-tests item in `AE_claude_todo.md` (tests already exist at test_mlops_foundation.py:1381-1563); correct the "AUDITED & STATISTICALLY PROVEN" d_0 line in `AE_gemini_todo.md`.

## Critical files
`ArgoEBUSCloud/02_ae_cloud_run.py`, `05_ae_update_tomatern0.5.py`, `ebus_core/runner.py`, `ebus_core/argoebus_gp_physics.py`, `vertical_delta_analysis.py`, new `d0_lml_sweep.py`, `test_mlops_foundation.py`.

## Verification
- `conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py` green after steps 1 and 4.
- Ingest probe: manifest date range + parquet time min/max match requested year (not 2015).
- Sweep reproduces free-fit LML at fitted d_0.
- Plots exist at logged paths; CV values printed match stats CSV.

## Uncommitted work already in tree
`californiav3_d500_1000_gibbs_timelsfix.yaml` and `vertical_delta_analysis.py` changes stay uncommitted; no branches, deletions or commits without per-instance permission.
