# Plan: Finish remaining engine test-coverage gaps + commit/push discipline

## Context

Top todo priority (`argo_claude_actions/AE_claude_todo.md`, "[ACTIVE #-2]") is closing the test-coverage
gap on the physics/GPR/pipeline engine — historically every serious silent bug (units bug 056c34f,
Gap 1 be013be) landed in untested code. The todo listed 6 target suites. Three-agent survey of the
current codebase (this session) found the todo list is **stale**: GibbsKernel already has 12 solid
tests, and the runner dispatch contract already has partial coverage. Verified against current code
(not memory) what's actually still missing.

**Explicit scope cut, per this session's instruction:** kernel-related work ("the kernels and such")
is being handed to Antigravity for review — this plan does **not** touch `GibbsKernel`,
`argoebus_gp_physics.py`, `compare_kernels.py`, or `vertical_delta_analysis.py`. It also does not touch
`compute_ohc_layer`/`calculate_thermodynamics` (confirmed DORMANT — no pipeline stage imports them,
kept intentionally for a future gridded-xarray path — not worth testing dead-but-intentionally-kept code).

**Second instruction this session:** treat committing/pushing completed work as high priority. Each
task below ends with its own commit; the plan pushes after each commit rather than batching everything
into one push at the end.

## Task 0 — Commit + push already-completed, unrelated work first

The working tree has pre-existing uncommitted changes from a prior session: type hints added to all 44
previously-untyped defs across `ebus_core/*.py` (`ae_utils.py`, `argoebus_gp_physics.py`,
`argoebus_plotting.py`, `argoebus_thermodynamics.py`, `config_schema.py`, `runner.py`) — no logic
changes, 81 tests already confirmed passing. This is finished work just sitting uncommitted.

- Stage only: the 6 `ebus_core/*.py` files + `CLAUDE.md` + `argo_claude_actions/AE_claude_todo.md` +
  `argo_gemini_actions/AE_gemini_{recentactions,todo}.md` (the tracked files that already reflect this
  work per `git status`).
- Commit message: `refactor(types): add type hints to ebus_core modules`.
- Push to `main`.
- **Left alone on purpose** (not part of this commit, not part of this plan): `compare_kernels.py`,
  `vertical_delta_analysis.py`, `summarize_audits.py`, `argo_probe_for_wherobots_int.md`,
  `docs/presentations/`, `docs/images/`, `References/` — these are pending Antigravity's review or are
  unrelated exploratory artifacts, not "new work" from this test-coverage task.

## Task 1 — Runner config→dispatch contract completeness

File: `ArgoEBUSCloud/test_mlops_foundation.py` (append near existing `test_run_analysis_*`/
`test_run_ingestion_dispatches` tests, ~line 744).

Current gap (confirmed via `runner.py:196-236` for `run_analysis`, `runner.py:322-333` for
`run_ingestion`): existing tests only spot-check 3-4 of the ~13 unconditional dispatch keys per
function, and the `gibbs_params` dict-building path inside `runner.py:225-236` is untested at the
runner level (only exercised indirectly, deep inside `analyze_rolling_correlations`).

New tests:
- `test_run_analysis_dispatch_kwargs_full_unconditional_set` — build an `AnalysisConfig`, monkeypatch
  `_call_run_diagnostic_inspection` to capture kwargs, assert **all** of `mode`, `step_size_days`,
  `min_bins`, `noise_val`, `time_ls_bounds_days`, `run_suffix` (in addition to the already-tested
  `region`/`depth_range`/`kernel_type`/`window_size_days`) reach dispatch with the exact config values.
- `test_run_ingestion_dispatch_kwargs_full_unconditional_set` — same pattern for `run_ingestion`:
  assert `date_start`, `date_end`, `n_workers`, `worker_region`, `s3_bucket` all reach
  `_call_run_ingestion` (in addition to already-tested `region`/`depth_range`).
- `test_run_analysis_builds_gibbs_params_dict` — `AnalysisConfig` with `kernel_type="gibbs"` and a
  `kernel_gibbs` block with distinct non-default values for all 8 `KernelGibbsBlock` fields; assert
  `dispatch_kwargs["gibbs_params"]` is a dict containing exactly those 8 keys/values.
- `test_run_analysis_omits_gibbs_params_when_not_gibbs` — `kernel_type="matern0.5"` → no
  `gibbs_params` key in dispatch_kwargs at all.

Commit: `test(runner): full dispatch_kwargs contract coverage for run_analysis/run_ingestion`. Push.

## Task 2 — ERDDAP URL builder test (offline, mocked)

Confirmed (`02_ae_cloud_run.py:27-76`): there is no standalone URL-builder function — the URL is an
f-string built inline inside `run_cloud_pipeline`, after `coiled.Cluster`/`Client` setup and before
`dd.read_csv`. No production code changes — test entirely via monkeypatching, so `run_cloud_pipeline`
runs fully offline:

File: `ArgoEBUSCloud/test_mlops_foundation.py` (new section, or a new
`ArgoEBUSCloud/test_ingestion_url.py` if it reads cleaner standalone — decide during implementation,
default to appending to `test_mlops_foundation.py` to avoid proliferating test files).

- Monkeypatch `coiled.Cluster`, `distributed.Client` (or whatever `02_ae_cloud_run.py` imports them
  as), `client.run`, and `dask.dataframe.read_csv` (capture the `erddap_url` argument via a
  `side_effect`/mock, then raise a sentinel exception or return an empty synthetic DataFrame so the
  rest of the function's Dask/S3 logic doesn't need to run for real).
- `test_erddap_url_uses_correct_host_and_encoding` — call `run_cloud_pipeline(region=..., ...)` with
  known lat/lon/date bounds, assert the captured URL: host is `erddap.ifremer.fr`, contains
  `%3E`/`%3C` (not bare `>`/`<`), and correctly substitutes the config's lat/lon/date bounds.
- Note in the test file (short comment) that `test_pipeline.py` remains the separate, intentionally
  non-pytest, live-network smoke script — this new test is the offline substitute the todo item asked
  for ("wire test_pipeline.py in, or delete and replace with a mocked-ERDDAP integration test"); we
  keep `test_pipeline.py` as-is (still useful for manual smoke checks) rather than deleting it.

Commit: `test(ingestion): offline ERDDAP URL-encoding test via mocked Dask/Coiled`. Push.

## Task 3 — Repo-layout test

Confirmed (`ae_utils.py:152-184`): `get_project_paths()` and `ensure_ae_dirs()` each independently
re-derive the two-levels-up traversal from `ebus_core/` to repo root; `05_ae_update_tomatern0.5.py:111-112`
does a *third*, independent one-level-up traversal (correct because that file lives one level shallower).
No test currently protects these three independent computations from silently diverging (this is exactly
the shape of the historical lesson-#2 bug: silent writes into the wrong `AEResults/` location).

File: `ArgoEBUSCloud/test_mlops_foundation.py` (new section, near other `ae_utils` tests like
`test_fmt_dec_importable_from_ae_utils`).

- `test_get_project_paths_root_is_repo_root_not_cloud_dir` — assert `paths['root']` does NOT end in
  `ArgoEBUSCloud`, and `paths['results'] == os.path.join(paths['root'], 'AEResults')`.
- `test_ensure_ae_dirs_creates_subdirs_under_correct_root` — call `ensure_ae_dirs()`, assert
  `aeplots`/`aedata`/`aelogs` exist under the same root `get_project_paths()` reports (not under
  `ArgoEBUSCloud/AEResults`). Non-destructive: only asserts existence, doesn't delete anything
  (these directories already exist in the real repo).
- `test_script05_plot_dir_matches_get_project_paths` — import `05_ae_update_tomatern0.5.py`, compute
  its independent `plot_dir`, assert it's identical to `get_project_paths()['plots']`. This is the
  actual regression guard: if either traversal is edited independently in the future, this test catches
  the divergence immediately instead of silently writing to the wrong place.

Commit: `test(ae_utils): pin AEResults path resolution against ArgoEBUSCloud/ divergence`. Push.

## Task 4 — Pipeline entry-point signature-drift test

Confirmed signatures: `run_diagnostic_inspection(region="california", lat_step=0.5, lon_step=0.5,
time_step=10.0, depth_range=(0,100), ...)` (`05_ae_update_tomatern0.5.py:54-62`) and
`run_cloud_pipeline(region="california", lat_step=0.5, lon_step=0.5, time_step=30.0,
depth_range=(0,100), n_workers=3)` (`02_ae_cloud_run.py:27-28`) — same first-5 names/order, different
defaults (expected, not a bug). `07_ae_deeper_layers.py` defines no entry point of its own (only
re-imports `05`'s function) and has sys.argv-gated module-level execution — **not imported** by this
test to avoid tripping that gate.

File: `ArgoEBUSCloud/test_mlops_foundation.py` (new section).

- `test_pipeline_entrypoints_share_first_five_param_contract` — use `inspect.signature` on
  `run_diagnostic_inspection` and `run_cloud_pipeline`; assert the first 5 parameter names, in order,
  are exactly `region, lat_step, lon_step, time_step, depth_range` for both (per the CLAUDE.md
  "common signature" hard-won rule). Does not assert equal defaults (intentionally differ).

Commit: `test(pipeline): guard (region, lat_step, lon_step, time_step, depth_range) entrypoint contract`. Push.

## Task 5 — Thermodynamics: depth_min/depth_max parameterization

Confirmed (`argoebus_thermodynamics.py:126-215`): `estimate_ohc_from_raw_bins`'s existing
`test_out_of_window_points_dropped` only exercises the **default** `depth_min=0, depth_max=2000`.
The `z_grid_edges`/`pd.cut` clipping logic is driven directly by these two params — no test currently
varies them, despite the CLAUDE.md-flagged "CRITICAL: Respecting chosen depth" anxiety in the pipeline
comments.

File: `ArgoEBUSCloud/test_thermodynamics.py` (append near `test_out_of_window_points_dropped`).

- `test_custom_depth_range_is_respected` — call `estimate_ohc_from_raw_bins` with a non-default
  `depth_min=150, depth_max=400` on synthetic data containing points both inside and outside that
  window; assert points outside `[150,400]` are excluded (mirroring the existing default-range test)
  and `ohc_per_m` divides by `depth_max - depth_min = 250`, not the default 2000.

Commit: `test(thermo): parameterize depth_min/depth_max coverage beyond the default window`. Push.

## Verification (after each task, before its commit)

```
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/ -q
```
Full suite must go from 81 → 81+N passing after each task, zero regressions, before that task's commit.

## Explicitly out of scope for this plan

- Any change to `GibbsKernel` / `argoebus_gp_physics.py` / kernel test coverage — deferred to
  Antigravity per this session's instruction.
- `compare_kernels.py`, `vertical_delta_analysis.py` review/run — deferred to Antigravity.
- `compute_ohc_layer` / `calculate_thermodynamics` — confirmed dormant-but-intentionally-kept, not
  worth testing.
- No production code is modified anywhere in this plan — every task is test-only (Task 2 uses
  monkeypatching instead of touching `02_ae_cloud_run.py`).
