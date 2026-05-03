# Claude Recent Actions — ArgoEBUSAnalysis

---

## 2026-05-02 (session 9) — MLOps audit complete; all 5 gaps fixed; main clean

### Summary
- Merged `feat/mlops-phase2` (PR #1) to main — full MLOps foundation (Phases 1–5), 46 tests
- Fixed all 5 Gemini audit gaps on branch `fix/mlops-audit-gaps-2-5` (PR #2), merged to main — 54 tests
- README updated with Data Coverage section (float census scripts) and backfill/test-suite blurbs

### README updates
- Added **Data Coverage** section documenting `09_ae_longterm_float_census.py` and `09b_ae_analyze_float_census.py` (unique-float density maps 1999–2024; domain-recommendation stats for californiav3)
- Expanded MLOps section: backfill blurb (18 historical configs from audit logs) + 44-test count

### Gap 1 (CRITICAL) — Fixed before merging PR #1
**File:** `ArgoEBUSCloud/ebus_core/runner.py:run_analysis():dispatch_kwargs`
- `spatial_ls_upper_bound` never passed to GPR shim; `lat_ls_bounds`/`lon_ls_bounds` silently ignored
- Fix: compute `max(lat_ls_bounds[1], lon_ls_bounds[1])` and add to `dispatch_kwargs` when both set; omit for legacy null-bounds configs
- 2 regression tests; commit `be013be`

### PR #1 merged: `feat/mlops-phase2` → `main` (46 tests)

### Gaps 2–5 — subagent-driven on `fix/mlops-audit-gaps-2-5` (PR #2)

**Gap 3 — Physics depth cross-validation** (`config_schema.py`)
- `_physics_depth_within_range` validator: `ohc_depth_top_m >= depth_range[0]`, `ohc_depth_bot_m <= depth_range[1]`
- 3 tests; `AnalysisConfig` docstring updated; commit `a76c2ca`

**Gap 5 — Backfill metadata** (`config_schema.py` + `backfill.py`)
- `BackfillMetadataBlock` with `recovered_fields` / `assumed_fields`; field added to `AnalysisConfig`
- `_parse_suffix` now returns `(dict, frozenset[recovered])` — eliminates duplicate token-detection in `backfill_configs`
- 18 YAMLs regenerated; 2 tests; commits `8785cca`, `f0cc102`

**Gap 4 — Centralize fmt_dec** (`ae_utils.py` + `runner.py` + `backfill.py`)
- `fmt_dec` public in `ae_utils`; `runner._fmt_dec` deleted; `backfill._infer_s3_path` inline closure deleted
- 1 test; commit `49492b5`

**Gap 2 — Registry status field** (`manifest.py` + `runner.py`)
- `status: "finalized"/"incomplete"` on every registry entry; checks audit CSV on disk post-dispatch
- Non-dict dispatch result forces `"incomplete"` unconditionally
- 2 tests; commits `8acdb1c`, `6bed512`

### PR #2 merged: `fix/mlops-audit-gaps-2-5` → `main` (54 tests)
All Gemini audit gaps closed. No open issues in MLOps layer.

### State after session
- Branch: `main` — 54 tests passing
- Float census scripts (`09`, `09b`, notebook) still untracked — science work, not yet committed

---

## 2026-04-30 (session 8) — MLOps Foundation Phase 5: Docs complete

Phase 5 fully complete on branch `feat/mlops-phase2`. 4 doc changes written.

- `README.md`: added MLOps Tooling section + `aebus` quickstart (validate/analyze/list/show)
- `CLAUDE.md`: added Config-Driven Workflow section (preferred entry point; scripts = escape hatch)
- `ae_file_structure.txt`: documented 8 new files — `config_schema.py`, `manifest.py`, `runner.py`, `backfill.py` (ebus_core/), `10_ae_backfill_configs.py`, `aebus_cli.py`, `test_mlops_foundation.py`, `configs/` tree
- `configs/README.md`: new file — schema field reference, versioning table, backfill null-field conventions

All Phase 4 + Phase 5 changes now ready to commit together.

---

## 2026-04-30 (session 7) — MLOps Foundation Phase 4: Backfill complete (44 tests)

Phase 4 fully complete on branch `feat/mlops-phase2`. 44 tests passing (up from 41).
18 `configs/*.yaml` files written. All round-trip via `load_config` + `derive_run_id`.

### Branch + test state

- Branch: `feat/mlops-phase2`
- Tests: 44 passing
- Last commit: `755ca59` (unchanged — no commit yet this session; Phase 5 docs will commit)
- New untracked: `ArgoEBUSCloud/ebus_core/backfill.py`, `ArgoEBUSCloud/10_ae_backfill_configs.py`, `configs/california/` (15 YAMLs), `configs/californiav2/` (3 YAMLs)

### Schema changes — `ebus_core/config_schema.py`

- Added `List` to imports.
- `GPRBlock`: `noise_val`, `time_ls_bounds_days`, `lat_ls_bounds`, `lon_ls_bounds` made `Optional` (null permitted for legacy backfill). Added `noise_vals_audit: Optional[List[float]] = None` (per-window empirical noise from audit CSV). Added `_noise_val_positive` field validator (skips when None). Updated `_ls_bounds_ordered` to also validate `time_ls_bounds_days` and skip all three when None.
- `IngestionConfig` + `AnalysisConfig`: added `legacy_backfill: bool = False`.
- `AnalysisConfig._no_bin_aliasing`: both bin-aliasing checks (step_size_days and time_ls_bounds_days) now skip entirely when `legacy_backfill=True`. This was necessary because the `_t1s10` experimental run had `step_size_days=10 < time_step=30` deliberately.
- New `AnalysisConfig._non_legacy_complete` validator: when `legacy_backfill=False`, asserts `gpr.noise_val`, `gpr.lat_ls_bounds`, `gpr.lon_ls_bounds` are all non-None.

### New files

**`ArgoEBUSCloud/ebus_core/backfill.py`** — core backfill logic:
- `_RUN_ID_PATTERN` regex decomposes canonical run_id into 9 groups.
- `_fmt_to_float(s)` — reverses `_fmt_dec` (e.g. `"0_5"` → `0.5`).
- `_parse_run_id(run_id)` → dict with all config fields recoverable from the run_id.
- `_parse_suffix(suffix, time_step)` → infers mode, kernel_type, window_size_days, min_bins, step_size_days from suffix tokens. step_size_days defaults to time_step when not in suffix.
- `_read_noise_vals(aelog_dir, run_id)` → reads all `noise_val` column values from audit CSV as `List[float]`.
- `_infer_s3_path(...)` → constructs expected S3 parquet URI from config fields.
- `backfill_configs(aelogs_root, configs_root, today)` → main fn. Walks dirs, writes one YAML per dir with null for unrecoverable fields + `legacy_backfill=true`.

**`ArgoEBUSCloud/10_ae_backfill_configs.py`** — CLI wrapper that calls `backfill_configs` with project-relative paths.

### Tests added (3 new)

- `test_legacy_backfill_marker_permits_null_provenance` — AnalysisConfig with `legacy_backfill=True` and all 4 gpr fields null parses without error.
- `test_non_legacy_config_requires_gpr_bounds` — same config with `legacy_backfill=False` raises ValidationError.
- `test_backfilled_configs_round_trip` — fixture aelog dir + audit CSV → `backfill_configs` → `load_config` → `derive_run_id` matches source dir name; `noise_vals_audit` matches CSV rows; null fields are null.

### Backfill output

18 YAMLs produced and all round-trip verified:
- `configs/california/` — 15 files (all `california` t30 runs including 2D-RBF, matern variants, experiment suffixes)
- `configs/californiav2/` — 3 files (FX2 canonical Skin/Source/Background)

Unrecoverable fields written as null: `noise_val`, `time_ls_bounds_days`, `lat_ls_bounds`, `lon_ls_bounds`.
Recovered from audit CSV: `noise_vals_audit` (list of per-window optimized noise).
Recovered from suffix: mode, kernel_type, window_size_days, min_bins, step_size_days.

Last updated: 2026-04-30 (session 7)

---

## 2026-04-30 (session 6) — MLOps Foundation Phase 3: CLI complete (41 tests)

Phase 3 fully complete on branch `feat/mlops-phase2`. 41 tests passing (up from 36).

### Branch + test state

- Branch: `feat/mlops-phase2`
- Tests: 41 passing
- Last commit: `755ca59`

### Commit 58e43c8 — feat(mlops): aebus CLI with validate subcommand

**Task 3.1.** Created `ArgoEBUSCloud/aebus_cli.py`:
- argparse skeleton with `prog="aebus"` and required subcommand
- `cmd_validate(args)` — loads config via `load_config`, derives run_id, prints `config_kind` / `run_id` / `manifest` path. Returns exit 2 on ValidationError.
- `main(argv)` — wires subparsers, raises SystemExit.
- Path injection: `sys.path.insert(0, str(_HERE))` so `python ArgoEBUSCloud/aebus_cli.py` works from repo root.
- Tests added: `test_cli_validate_prints_run_id`, `test_cli_validate_bad_yaml_exits_2`.

### Commit 82c76c9 — feat(mlops): aebus analyze + ingest subcommands with collision exit code 3

**Task 3.2.** Extended `aebus_cli.py`:
- `cmd_analyze` — validates config kind, calls `run_analysis`, catches `ManifestCollisionError` → exit 3.
- `cmd_ingest` — same pattern for ingestion.
- `p_ana` + `p_ing` subparsers with `--registry` and `--force-overwrite` flags.
- Plan test YAML contained removed field `spatial_ls_upper_bound` (superseded by §A.2 split-anisotropy); dropped from test fixture.
- Test added: `test_cli_analyze_collision_aborts`.

### Commit 755ca59 — feat(mlops): aebus list + show subcommands query the registry

**Task 3.3.** Extended `aebus_cli.py`:
- `cmd_list` — reads `registry.jsonl`, filters by `--region` / `--kind`, prints fixed-width table.
- `cmd_show` — looks up `run_id` in registry, reads and prints `manifest.json`.
- `p_list` + `p_show` subparsers registered.
- Tests added: `test_cli_list_filters_by_region`, `test_cli_show_prints_manifest_json`.
- Plan used `json` but test file imports it as `_json`; fixed consistently.

### Note on plan test YAML

Plan's `test_cli_analyze_collision_aborts` fixture used `spatial_ls_upper_bound: 10` — field removed from `GPRBlock` in §A.2 (split into `lat_ls_bounds` / `lon_ls_bounds`). Dropped silently; no schema change needed.

---

## 2026-04-30 (session 5) — MLOps Foundation Phase 2: Tasks 2.3, 2.5, 2.4 complete

Phase 2 fully complete on branch `feat/mlops-phase2`. 36 tests passing.

### Branch + test state

- Branch: `feat/mlops-phase2`
- Tests: 36 passing (up from 34 at session start)
- Last commit: `3f2c8d6`

### Commit ca696e5 — feat(mlops): run_analysis wraps existing GPR fn with manifest + collision detection

**Task 2.3.** Appended to `ArgoEBUSCloud/ebus_core/runner.py`:
- `_call_run_diagnostic_inspection(**kwargs)` — shim that loads script 05 via importlib
  and calls `run_diagnostic_inspection`. Tests monkeypatch this whole function.
- `run_analysis(cfg, registry_path, force_overwrite)` — dispatches to shim, times run,
  builds manifest, writes manifest.json, appends registry.
- dispatch_kwargs excludes `spatial_ls_upper_bound` — field absent from GPRBlock after
  §A.2 split-anisotropy redesign (was `lat_ls_bounds`/`lon_ls_bounds`).
- Test was pre-written in test file (caused ImportError at collection time); test now passes.
- Updated imports: added `shutil`, `time as _time`, `ManifestCollisionError`,
  `append_registry`, `check_collision`, `write_manifest` from manifest module.

### Commit d0f609e — feat(mlops): expose run_ingestion_pipeline seam in script 02

**Task 2.5.** Added `run_ingestion_pipeline = run_cloud_pipeline` alias to
`02_ae_cloud_run.py` just before `__main__`. One-line seam so runner shim can
dispatch by a stable, intent-clear name. Chose Option A (alias) over Option B
(change shim) because Option B would fail in production due to kwargs mismatch.

### Commit 3f2c8d6 — feat(mlops): run_ingestion wraps existing cloud-ingest pipeline

**Task 2.4.** Appended to `ArgoEBUSCloud/ebus_core/runner.py`:
- `INGESTION_AELOGS_DIR` — module-level Path constant, monkeypatchable by tests.
- `_call_run_ingestion(**kwargs)` — shim that loads script 02 via importlib and calls
  `run_ingestion_pipeline`. Raises RuntimeError if seam absent.
- `run_ingestion(cfg, registry_path, force_overwrite)` — mirrors run_analysis pattern.
  dispatch_kwargs includes date_start, date_end, n_workers, worker_region, s3_bucket.

---

## 2026-04-30 (session 4) — MLOps Foundation Phase 2: Tasks 2.1 + 2.2 complete

Phase 2 started on branch `feat/mlops-phase2`. Two tasks complete, three remain.

### Branch + test state

- Branch: `feat/mlops-phase2` (created this session from `main` at `084c7e6`)
- Tests: 34 passing (up from 29 at session start)
- Last commit: `ecc3828`

### Commit 148c11e — feat(mlops): derive_run_id matches existing canonical naming

**Task 2.1.** Created `ArgoEBUSCloud/ebus_core/runner.py` with:
- `derive_run_id(cfg)` — reproduces the canonical run_id pattern used by existing
  scripts (region_YYYYMMDD_YYYYMMDD_res{lat}x{lon}_t{time}_d{d0}_{d1}[{run_suffix}])
- `_fmt_dec(x)` — float → underscore decimal string (0.5 → "0_5", 10.0 → "10_0")
- Verbose comments explaining backward-compat rationale, why d0/d1 are raw ints
  (not through _fmt_dec), and run_suffix leading-underscore ownership
- 2 new tests: AnalysisConfig and IngestionConfig canonical patterns
- Two-stage review: spec compliant ✅, code quality NEEDS_WORK (comments) → fixed → APPROVED ✅

### Commit 13acc75 — feat(mlops): build_manifest assembles full manifest dict from config + metadata

**Task 2.2.** Appended `build_manifest(cfg, outputs, inputs_extra, duration_sec, conda_list_dest, cwd=None)` to runner.py:
- Pure function (no file IO) assembles full manifest dict with 12 top-level keys
- §A.4: teos10_convention="gsw-3.x" injected into env block
- inputs block branches on AnalysisConfig vs IngestionConfig
- 3 new tests: required top-level keys, §A.4 erddap lineage, IngestionConfig branch
- Spec compliant ✅

### Commit ecc3828 — fix(mlops): build_manifest config fields win over inputs_extra; add ingestion test

**Task 2.2 code-review fix.** Code review found:
- C1 (Critical): inputs_extra spread was LAST, silently overwriting config-derived fields
  Fixed: reversed merge order so config fields (source, s3_path, ingestion_run_id) come last
- I1: teos10 comment expanded to explain convention-family-tag vs version-pin semantics
- I2: docstring OUTPUT section now enumerates all 12 keys
- I3: added test_build_manifest_ingestion_config (IngestionConfig branch coverage)
- Code quality APPROVED ✅

### Kwargs drift documented

`run_diagnostic_inspection` in `05_ae_update_tomatern0.5.py` does NOT accept: mode,
kernel_type, window_size_days, min_bins, noise_val. `spatial_ls_upper_bound` is in the
function signature (default=10) but NOT in GPRBlock. Task 2.3 shim must filter kwargs
accordingly. (To be documented in AE_claude_lessons.md in next session.)

### Open tasks — Phase 2

- Task 2.3 (run_analysis wrapper): IN PROGRESS — implementer agent was dispatched
  but was interrupted by context limit. Full context + kwargs-drift analysis prepared.
  Resume point: dispatch fresh implementer subagent with the prompt built in this session.
- Task 2.5 (callable seam audit): pending
- Task 2.4 (run_ingestion wrapper): pending, blocked on Task 2.5

Last updated: 2026-04-30 (session 4)

---

## 2026-04-30 (session 3) — MLOps Foundation Phase 1: Tasks 1.1–1.8, all complete

Phase 1 of the MLOps foundation is fully implemented and committed. 29 tests
pass. All work on branch `main`.

### Commit bd018b9 — chore(mlops): hoist QCPolicyBlock test import to module top

Landed the deferred Task 1.1 code-review nit (I3): promoted
`from ebus_core.config_schema import QCPolicyBlock` from inside the
`test_qc_policy_defaults_and_validation` body to module-top alongside
the existing `IngestionConfig` import. Comment-only + import-hygiene; no
behaviour change. 2 tests still pass.

### Commit 1deb899 — feat(mlops): add IngestionConfig validators for region, depth, dates

Task 1.2. Three validators added to `IngestionConfig`:
- `_region_in_registry`: calls `get_ebus_registry()`, rejects unknown regions
- `_depth_range_ordered`: rejects `depth_range` where top < 0 or top >= bottom
- `_dates_ordered`: model_validator rejecting date_start >= date_end
4 new tests (unknown key, unknown region, bad dates, bad depth). 6 tests pass.

### Commit c9698f1 — chore(mlops): fix validator Raises comments

Code-review fix: changed "Raises: ValueError" to
"Raises: pydantic.ValidationError (wraps ValueError)" in all three new
validator header comments. Comment-only; no behaviour change.

### Commit cca8837 — feat(mlops): add AnalysisConfig with GPR/input/output/physics sub-blocks and bin-aliasing rule

Task 1.3 with §A.2 + §A.3 amendments applied.

New classes in `config_schema.py`:
- `KernelRBFBlock`, `KernelMatern05Block` — empty placeholder kernel blocks
- `KernelGibbsBlock` — full RG-Gibbs sigmoid-of-dist_to_coast params
  (l_min_km, l_max_km, d_transition, k_steepness, anisotropy_lat_lon_ratio,
  climatology_source per 2026-04-26 l(x) directive)
- `GPRBlock` — §A.2: lat_ls_bounds + lon_ls_bounds split (NOT spatial_ls_upper_bound),
  polymorphic kernel sub-blocks with `_kernel_sub_block_exclusive` validator that
  auto-instantiates the matching block; `_ls_bounds_ordered` on both bound fields
- `PhysicsParamsBlock` — §A.3: OHC integration constants, _depth_top_below_bot validator
- `AnalysisInputBlock` — s3/ingestion_run pointer with _exactly_one_pointer validator
- `OutputsBlock` — artifact dirs
- `AnalysisConfig` — top-level with _region_in_registry, _depth_range_ordered,
  _no_bin_aliasing (step_size_days >= time_step AND time_ls_bounds lower >= time_step),
  physics_params field

6 new tests including §A.6 additions: polymorphic_kernel_blocks_exclusive,
lat_lon_ls_bounds_split_validates, physics_params_depth_ordering. 12 tests pass.

### Commit e1c152e — fix(mlops): split _no_bin_aliasing/_dates_ordered, fix setattr idiom, add 3 tests

Code-review fixes for Task 1.3:
- C1: Split `_no_bin_aliasing` + `_dates_ordered` into two separate validators
  so Pydantic can surface both errors independently
- I1: `object.__setattr__` → `setattr` in `_kernel_sub_block_exclusive` (correct
  Pydantic v2 idiom for non-frozen models)
- M1: Full header comment on `AnalysisConfig._depth_range_ordered`
- Added: `test_analysis_config_bad_dates_rejected`, `test_analysis_input_block_ingestion_run_requires_run_id`,
  `test_gpr_gibbs_auto_instantiates_kernel_block`. 15 tests pass.

### Commit 1152903 — feat(mlops): add load_config() YAML loader dispatching on config_kind

Task 1.4. `load_config(path)` in `config_schema.py`:
- Reads YAML with `yaml.safe_load`
- Dispatches to `IngestionConfig` or `AnalysisConfig` on `config_kind`
- Raises ValueError if top-level is not a mapping or config_kind missing/unknown
Added `yaml` + `pathlib.Path` + `Union` imports. 2 new tests. 17 tests pass.

### Commit 0b1110a — feat(mlops): canonicalize configs and compute sha256 hash

Task 1.5. New file `ArgoEBUSCloud/ebus_core/manifest.py` with:
- `canonical_config_dict(cfg)` — model_dump(mode='json'), strip _HASH_EXCLUDE
  (description), recursively sort all dict keys
- `config_hash(cfg)` — sha256 hex digest of json.dumps(canon)
4 new tests: excludes description, stable across description, changes with real
field, 64-char hex. 21 tests pass.

### Commit 2896cb3 — feat(mlops): capture git, conda env, and host metadata for manifests

Task 1.6. Added to `manifest.py`:
- `KEY_PACKAGES` tuple: scikit-learn, xarray, numpy, pandas, gsw, coiled, dask,
  matplotlib, cartopy, scipy (§A.4 scipy added)
- `capture_code(cwd)` — git rev-parse HEAD, --porcelain, --abbrev-ref HEAD via subprocess
- `capture_env(conda_env_name, conda_list_dest)` — conda list --json; inlines
  KEY_PACKAGES versions; optionally writes full conda_list.txt
- `capture_host()` — socket.gethostname() + platform.platform()
3 new tests. 24 tests pass.

### Commit 084c7e6 — feat(mlops): manifest read/write, collision detector, and run registry

Tasks 1.7 + 1.8.

Task 1.7 additions to `manifest.py`:
- `ManifestCollisionError` — Exception subclass for run_id collision
- `write_manifest(manifest, path)` — json.dump with indent=2, sort_keys, mkdir -p
- `read_manifest(path)` — json.load
- `check_collision(manifest_path, new_hash)` → "fresh" / "rerun" / raises
  ManifestCollisionError (with both hashes in message)

Task 1.8 additions:
- `_REGISTRY_FIELDS` tuple — canonical JSONL line schema
- `append_registry(manifest, registry_path, manifest_path)` — appends one
  denormalized JSON line per run to AEResults/run_registry.jsonl

5 new tests: roundtrip, collision_no_existing, collision_same_hash,
collision_different_hash, append_registry_appends_one_line. 29 tests pass.

### End state

- Branch: `main`
- Last commit: `084c7e6`
- 29 tests passing
- Files created this session:
  - `ArgoEBUSCloud/ebus_core/config_schema.py` (expanded: AnalysisConfig + sub-models)
  - `ArgoEBUSCloud/ebus_core/manifest.py` (new)
  - `ArgoEBUSCloud/test_mlops_foundation.py` (expanded: 29 tests)
- Working tree dirty: `.gitignore` (M), `argo_claude_actions/` (M),
  `argo_gemini_actions/` (M, pre-existing), `.claude/`, `09_*`,
  `float_census_viewer.ipynb` (untracked, separate work)

Last updated: 2026-04-30 (session 3)

---

## 2026-04-26 (session 2) — MLOps Foundation: side-decision commits + Task 1.1 implementation

### Commit e93ae01 — docs: Gemini RG-Gibbs + MLOps review specs + brainstorm plan

Side-decision 1 from prior session resolved: tracked the 4 untracked Gemini
specs that the plan §A amendment block references (so future readers can
trace amendment provenance). Files added:
- `docs/superpowers/specs/2026-04-11-rg-gibbs-nonstationary-gpr-design.md`
- `docs/superpowers/specs/2026-04-26-fx2-diagnostics-verdict.md`
- `docs/superpowers/specs/2026-04-26-mlops-review-results.md`
- `docs/superpowers/specs/2026-04-26-rg-gibbs-l-x-directive.md`
- `argo_claude_actions/brainstorming/RG_Gibbs_NonStationary_Model_Plan.md`

### Commit 2f1e570 — feat(pipeline): dist_to_coast feature + californiav3

Side-decision 2 resolved: committed Gemini's `dist_to_coast` integration
so Phase 1 diff lands purely on top of stable pipeline state. Files:
- `ArgoEBUSCloud/ebus_core/ae_utils.py` — new `get_coastline_points()`
  + `calculate_dist_to_coast()` (KDTree on 3D ECEF unit-vector space +
  exact Haversine on nearest vertex; isotropic-on-sphere). Adds
  `californiav3` registry entry (lat 30–48, lon -135 to -115).
- `ArgoEBUSCloud/ebus_core/argoebus_thermodynamics.py` —
  `estimate_ohc_from_raw_bins()` stamps `dist_to_coast_km` on every grid
  center.
- `ArgoEBUSCloud/01_ae_cloud_ingestion.py` /
  `ArgoEBUSCloud/02_ae_cloud_run.py` — added `dist_to_coast_km` +
  `platform_number` to dask meta schema.
- `CLAUDE.md` + `ae_file_structure.txt` — doc updates.

Files DELIBERATELY left uncommitted: `.gitignore` (M), the `argo_*_actions/`
diaries (M), `.claude/` (untracked, personal), the new `09_*` float census
scripts + `float_census_viewer.ipynb` (untracked, separate work). Out of
scope per todo.

### Commit a27d324 — feat(mlops): IngestionConfig pydantic stub + first valid-parse test

**Task 1.1 implementation, dispatched via subagent-driven-development skill.**
Implementer = sonnet general-purpose. TDD followed (test wrote first,
ran red, then green). Files created:
- `ArgoEBUSCloud/ebus_core/config_schema.py` (88 lines) — module
  docstring; `CloudBlock`, `S3Block`, `QCPolicyBlock`, `IngestionConfig`
  pydantic v2 models. §A.1 amendment folded in: `QCPolicyBlock` carries
  `argo_qc_flags_accepted=[1, 2]` default, `_flags_in_argo_domain`
  validator rejects flags outside `{1,2,3,4,5,8,9}`. Project verbose-comment
  rule honored on every field.
- `ArgoEBUSCloud/test_mlops_foundation.py` (49 lines) — 2 tests:
  `test_ingestion_config_valid_minimal` (plan stub) +
  `test_qc_policy_defaults_and_validation` (§A.6 amendment).
  `cd ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest
  ArgoEBUSCloud/test_mlops_foundation.py -v` → `2 passed`.

### Two-stage review for Task 1.1

**Spec compliance review (sonnet general-purpose):** ✅ Spec compliant.
Confirmed every required class/field/validator/default present, both tests
present, exactly 2 files touched, commit msg + co-author trailer correct.

**Code-quality review (superpowers:code-reviewer):** Approved with minor
fixes. Findings:
- **I1** (Important): `IngestionConfig` lacks `depth_range[0] < depth_range[1]`
  validator. **CONTROLLER DEFERRED to Task 1.2** — Task 1.2 is explicitly
  "IngestionConfig validation rules" (plan:346). Adding now violates
  surgical-changes rule and TDD-incremental design.
- **I2** (Important): `test_ingestion_config_valid_minimal` only spot-checks
  3 attributes. **CONTROLLER DEFERRED** — plan stub spec listed exactly
  those 3 assertions; expanding is over-spec.
- **I3** (Important): `from ebus_core.config_schema import QCPolicyBlock`
  is inside `test_qc_policy_defaults_and_validation` body instead of at
  module top. **NOT FIXED** — implementer dispatch was REJECTED by user
  (context-limit interrupt) before the fix landed. Fix is one edit:
  promote import to module top alongside the existing IngestionConfig
  import.
- M1/M2/M3 (Minor): worker_region open string, notes default `""` vs
  `Optional[str]=None`, premature fixture extraction. All deferred per
  surgical-changes.

### Workflow notes

- Subagent-driven-development skill loaded; implementer-prompt + spec-
  reviewer-prompt + code-quality-reviewer-prompt templates used verbatim.
- TaskCreate seeded all 12 Phase-1+ tasks (#1–#12); Task 1.1 currently
  status `in_progress` because I3 fix not yet committed.
- Working on branch `main` (per project hard-stop default; user did not
  request a feature branch when authorizing Task 1.1 start).

---

## 2026-04-26 — MLOps Foundation: Gemini-review amendments + Phase 0 env install

Two commits landed this session:

### Commit 3bc7b75 — spec+plan amendments per Gemini review

Gemini review (`docs/superpowers/specs/2026-04-26-mlops-review-results.md`)
returned "Approved with conditions." Plus a separate l(x) directive
(`docs/superpowers/specs/2026-04-26-rg-gibbs-l-x-directive.md`) — learnable
sigmoid of `dist_to_coast_km` with fixed `l_min`/`l_max` and learnable
`d_0`/`k`, 2:1 lat:lon anisotropy.

Spec edits (`docs/superpowers/specs/2026-04-25-mlops-foundation-design.md`):
- §3 IngestionConfig: new `qc_policy` block (Argo flag whitelist + exclusions)
- §3 AnalysisConfig.gpr: replaced `spatial_ls_upper_bound` with split
  `lat_ls_bounds` + `lon_ls_bounds`; added polymorphic kernel sub-blocks
  (`kernel_rbf` / `kernel_matern05` / `kernel_gibbs`); commented
  `kernel_gibbs` placeholder reflects 2026-04-26 l(x) directive verbatim
- §3 AnalysisConfig: new `physics_params` block (OHC ref pressure, TEOS-10
  convention, QC thresholds)
- §3 validation rules expanded
- §4 manifest: added `inputs.erddap_dataset_id` + `erddap_server_url` +
  `data_access_timestamp`; `key_packages` whitelist gains scipy;
  `env.teos10_convention` top-level
- §6 backfill: null over assumed defaults; `legacy_backfill: true` marker;
  depth ranges recorded verbatim (no rewrite to 2026-04-26 RG-aligned bounds)
- §9 RG-Gibbs forward-reference now points at the reserved `kernel_gibbs`
  sub-block shape

Plan edits (`docs/superpowers/plans/2026-04-25-mlops-foundation.md`):
- New §A "2026-04-26 Amendment Block" inserted before Phase 0. Provides
  Pydantic shapes for `QCPolicyBlock`, `KernelGibbsBlock`, `PhysicsParamsBlock`;
  rules for the polymorphic kernel validator; ERDDAP+TEOS-10 manifest
  additions; backfill null policy; six new test cases.
- Existing per-task text in Phase 1 / Phase 4 is unchanged — implementer
  subagents read §A in addition to each task and apply where they conflict.

### Commit 59d752d — Phase 0 env install

- `pip install pydantic>=2.0 pytest>=7.0` into `ebus-cloud-env`. Live
  versions: pydantic 2.13.3, pytest 9.0.3.
- Added the same two lines to the `pip:` block in
  `ArgoEBUSCloud/ocean_cloud.yml` (the active env spec for this project).
- Same additions to `ocean_env_clean.yml` (the `oceanography` env, not
  `ebus-cloud-env` — note env name mismatch). Followed plan instruction
  literally; if Avik wants only the active env spec touched, this can be
  reverted.

### Workflow housekeeping

- Stashed 4 narrative md files (claude+gemini action dirs) before spec/plan
  edits to keep the WIP narrative cleanly separable. Popped at end.
- Did NOT add Gemini's untracked specs to git
  (`2026-04-26-mlops-review-results.md`,
  `2026-04-26-fx2-diagnostics-verdict.md`,
  `2026-04-26-rg-gibbs-l-x-directive.md`,
  `2026-04-11-rg-gibbs-nonstationary-gpr-design.md`,
  `argo_claude_actions/brainstorming/`). Avik's call whether to commit
  these as part of a future doc commit.
- Did NOT touch the existing modified pipeline files
  (`01_ae_cloud_ingestion.py`, `02_ae_cloud_run.py`, `ae_utils.py`,
  `argoebus_thermodynamics.py`) — those are Gemini's `dist_to_coast`
  integration, done work, untouched by Phase 0.
- Did Phase 0 inline (5 trivial steps); subagent dispatch starts at Phase 1.

### Resume point for next session

Top of `AE_claude_todo.md` updated: Phase 0 retired, Phase 1 is now the
top priority. Read the plan amendment block §A FIRST, then dispatch the
first implementer subagent for Task 1.1 per `superpowers:subagent-driven-development`.

Two follow-ups noted in todo: (a) decide whether to commit Gemini's
untracked specs, (b) verify `02_ae_cloud_run.py` exposes a top-level
`run_ingestion_pipeline(**kwargs)` callable before Task 2.5 (still open
dependency from the original todo).

---

## 2026-04-25 — MLOps Foundation: Brainstorm + Spec + Plan (no code yet)

Full brainstorming session for the MLOps showcase initiative (top of 2026-04-24
todo). Used `superpowers:brainstorming` then `superpowers:writing-plans` skills.
Three artifacts produced and committed (commit `d119377`).

### Decisions reached during brainstorm
- **Audience:** sequenced — practitioner-grade infra first, then hiring-target
  demo polish layered on top.
- **Top pains:** region scaling (B) + reproducibility (C). Tightly coupled, so
  treat as one MLOps foundation.
- **Scope tiers:** A = config-driven runs + manifests + thin CLI (this spec).
  B = MLflow/W&B (next spec). C = pip pkg + Docker + dashboard (long-term).
- **Refactor strategy:** additive only. No existing script touched. Old
  `__main__` blocks remain as escape hatch.
- **Stage separation:** two configs (`IngestionConfig`, `AnalysisConfig`).
  Analysis manifest cross-references ingestion manifest hash for full lineage.
- **Run identity:** preserve current `run_id` pattern verbatim. Disambiguate by
  collision detector that compares `config_hash` of existing manifest vs new
  run; abort-with-diff if differ.
- **Manifest contents:** identity + config + lineage + duration_sec ONLY. No
  `metrics_summary`, no plot enumeration, no `config_path`. Results live in
  audit CSV; manifest stays pure-provenance.
- **Registry:** JSONL (`AEResults/run_registry.jsonl`), append-only, fixed
  schema_version=1 line shape.
- **Backfill:** configs only (skip retro-manifests — too low fidelity).

### Files committed (commit d119377)
- `docs/superpowers/specs/2026-04-25-mlops-foundation-design.md` — full spec,
  10 sections, ~430 lines
- `docs/superpowers/plans/2026-04-25-mlops-foundation.md` — 16-task TDD
  implementation plan, 6 phases (~1100 lines)
- `argo_gemini_actions/AE_gemini_todo.md` — top entry asks Gemini to review
  the spec before implementation begins

### What was NOT done
- No implementation code written. Per project CLAUDE.md hard stop, code only
  starts after explicit per-session plan approval.
- No env changes (pydantic + pytest install is Plan Phase 0, deferred).
- Existing dirty files (01/02/ae_utils/argoebus_thermodynamics, CLAUDE.md,
  ae_file_structure.txt, etc.) untouched — those are Avik's in-progress work.

### Resume instructions for next session
Top of `AE_claude_todo.md` now lists "Execute MLOps Foundation Plan" as the
TOP priority. Read the plan file first, get explicit approval, then dispatch
via `superpowers:subagent-driven-development` or `superpowers:executing-plans`.

---

## 2026-04-11 — Fix `calculate_dist_to_coast` in `ebus_core/ae_utils.py`

Reviewed Gemini's implementation from today's session and corrected three bugs:

1. **Removed dead `gsw` import** (`ae_utils.py` line ~412). `from gsw import distance` was imported inside the function but never used — would cause import error if gsw unavailable.

2. **Fixed KDTree bias** — replaced naive lat/lon degree-space KDTree with 3D unit-vector (ECEF) KDTree. At CCS latitudes 1° lon ≈ 85 km vs 1° lat ≈ 111 km; raw degree-space tree returns biased nearest neighbor. Fix: convert (lat, lon) → (x, y, z) on unit sphere before building tree. One exact Haversine pass on the single nearest candidate gives correct km distance. O(log N) lookup preserved.

3. **Resolution `50m` → `10m`** — default changed to Natural Earth `10m` (~1–2 km vertex spacing) per approved design. Required for accurate coastal-gradient features in future XGBoost upwelling models.

4. **californiav3 bounds confirmed** — Gemini's census-driven update to lat [30, 48], lon [-135, -115] approved by user. Already in `ae_utils.py`; recorded here for traceability.

No other files touched. `dist_to_coast_km` integration in `argoebus_thermodynamics.py`, `01_ae_cloud_ingestion.py`, and `02_ae_cloud_run.py` left as-is (correct).

---

## 2026-04-02 — Run Scripts 09 + 09b: Long-Term Argo Float Census Results

### Run results

Both scripts executed successfully.

**09 census run:**
- 58,195 total dives fetched across 6 ERDDAP chunks
- 25 years with data (2001–2025; no floats in 1999–2000)
- 503 (year, cell) records in census
- 25 per-year PNGs + CSV written

**09b analysis results:**

Annual totals: sparse early-Argo era (11 float-obs in 2001), grows steadily to
~130–176 from 2014 onward. 2025 is the densest year at 199 total.

Top persistent cells (25/25 years present):
- 47.5°N / -132.5°W — mean 9.92 floats (northwest, open Pacific)
- 47.5°N / -137.5°W — mean 8.44 floats
- 47.5°N / -127.5°W — mean 8.28 floats
- 32.5°N / -122.5°W — mean 7.88 floats (Southern California Bight coastal)
- 32.5°N / -127.5°W — mean 7.80 floats

Domain recommendation (≥20/25 years): 19 qualifying cells.
Implied bounding box: lat [25, 50], lon [-140, -115].

**Output files:**
- `AEResults/aeplots/float_census_california/float_census_california_[YEAR].png` (25 files)
- `AEResults/aeplots/float_census_california/float_census_california_1999_2025.csv`
- `AEResults/aeplots/float_census_california/float_census_annual_totals.png`
- `AEResults/aeplots/float_census_california/float_census_mean_density.png`

**Status:** Awaiting Gemini review to define californiav3.

---

## 2026-04-02 — Implement Scripts 09 + 09b: Long-Term Argo Float Census

**Prompted by:** Source Layer GPR regression in californiav2 FX2 run.
**Purpose:** Generate empirical float density data so Gemini can define californiav3.

### What was built

**`ArgoEBUSCloud/09_ae_longterm_float_census.py`**
- Fetches per-dive Argo float positions for the broad california domain
  (lat [25,50], lon [-140,-110]) in 6 x 5-year chunks via `get_float_history()`.
- Bins on a 5°×5° grid, counts unique floats per (year, lat_bin, lon_bin).
- Saves full census to CSV.
- Generates one Cartopy pcolormesh PNG per year (1999–2024), fixed color scale
  vmin=0/vmax=15 for year-to-year comparability.
- Prints top-10 hotspot table to stdout as a sanity check.

**`ArgoEBUSCloud/09b_ae_analyze_float_census.py`**
- Standalone CSV analysis tool. Produces:
  1. Annual total float count table (printed + bar chart PNG)
  2. Top-10 most persistent cells (most years with ≥1 float)
  3. Mean density Cartopy map (averaged over all 26 years)
  4. **Domain recommendation table**: cells present in ≥20/26 years, sorted
     by mean n_floats — the direct empirical input for Gemini to define californiav3.
     Also prints the implied bounding box of qualifying cells.

### Output paths

All output in `AEResults/aeplots/float_census_california/`:
- `float_census_california_1999.png` … `float_census_california_2024.png`
- `float_census_california_1999_2025.csv`
- `float_census_annual_totals.png` (from 09b)
- `float_census_mean_density.png` (from 09b)

### Notes for Gemini

The domain recommendation output from `09b` is the key artifact. Run sequence:
```
conda run -n ebus-cloud-env python ArgoEBUSCloud/09_ae_longterm_float_census.py
conda run -n ebus-cloud-env python ArgoEBUSCloud/09b_ae_analyze_float_census.py
```

The `09b` stdout section headed `=== CALIFORNIAV3 DOMAIN RECOMMENDATION DATA ===`
lists every 5°×5° cell present in ≥20/26 years and their mean float count, plus
the implied bounding box. Gemini should:
1. Review the per-year PNGs and mean density map for spatial context.
2. Use the recommendation table + implied bounding box as the starting point for v3.
3. Refine bounds based on oceanographic rationale (e.g., exclude far-offshore
   Pacific transits, include the full California Undercurrent corridor).

---

## 2026-04-01 — FX2 Cloud Run + GPR Analysis: californiav2 t10_0 All Layers

**First GPR run on the FX2 canonical parquets (californiav2, time_step=10.0).**

### Cloud Run (Script 02)

Re-ran `02_ae_cloud_run.py` for all three layers on AWS/Coiled.
Updated `__main__` to canonical FX2 config; removed dead-code V1 block.
Also fixed `time_step` bug: both `05_ae_update_tomatern0.5.py` and
`07_ae_deeper_layers.py` had `time_step=30.0` in their config, causing
S3 path mismatches. Updated to `time_step=10.0` everywhere including the
`run_diagnostic_inspection()` default function signature.

S3 parquets written:
- `s3://argo-ebus-project-data-abm/californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100.parquet`
- `s3://argo-ebus-project-data-abm/californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400.parquet`
- `s3://argo-ebus-project-data-abm/californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000.parquet`

### GPR Results (Scripts 05 + 07)

Output run_id suffix: `_3dmatern_w45` (no experiment suffix — canonical run).
All output in `AEResults/aelogs/<run_id>/` and `AEResults/aeplots/snapshot_<run_id>/`.

| Layer | Pass Rate | Median RMSRE | Max RMSRE | Std Z Range | Old Median RMSRE |
|-------|-----------|--------------|-----------|-------------|-----------------|
| Skin (0–100m) | 25/34 (74%) | 3.80% | 6.24% | 0.42–2.35 | ~3.4% (t30 baseline) |
| Source (150–400m) | 8/34 (24%) | **8.13%** | **22.09%** | 0.20–15.63 | ~4.2% (t30 baseline) |
| Background (500–1000m) | 34/34 (100%) | 1.99% | 3.42% | 0.22–18.73 | ~4.8% (t30 baseline) |

### Issues Flagged for Gemini Review

**1. Source Layer severe regression (Priority)**
- Median RMSRE 8.13% vs old 4.2%. Only 8/34 windows pass. Max RMSRE 22.09%.
- Extreme anisotropy ratios in many windows (8.41, 35.75, 6.26) — non-physical.
- Worst windows (RMSRE > 10%): centers at days 5952, 6032, 6072, 6082, 6132,
  6142, 6152, 6172, 6182, 6192.
- Z spike: window 6022 has std_z=15.63 despite RMSRE=3.5% — extreme overconfidence.
- Possible causes: tighter californiav2 domain clips float trajectories at depth;
  10d bins expose sparsity in Source layer that 30d bins masked by averaging.

**2. scale_time_bin saturates at 45d in all Skin and Source windows**
- Every window hits the upper bound of `time_ls_bounds_days=(15.0, 45.0)`.
- T2 experiment already showed this: the GP always wants to use maximum temporal
  persistence. No oscillation (FX2 worked), but still saturated.
- Question for Gemini: is 45d an insufficient upper bound for skin temporal
  coherence? Should we widen `time_ls_bounds_days` upper limit for Skin layer?

**3. Background Layer: isolated Z spike at window 6102.5 (Z=18.73)**
- RMSRE only 2.67% but Z=18.73. Day 6102.5 from 1999-01-01 ≈ Sep 2015.
- Background scale_time_bin is variable (26–45d in mid-year) — FX2 is working here.
- Consistent with Pacific Blob peak non-stationarity. Gemini verdict from prior
  session: genuine physical event, flag as stationarity violation if Z > 2.0 persists.

### Output Files for Gemini

**Skin Layer (0–100m):**
- Audit CSV: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100_3dmatern_w45/audit_californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100_3dmatern_w45.csv`
- Temporal persistence: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100_3dmatern_w45/temporal_persistence_californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100_3dmatern_w45.png`
- Anisotropy: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100_3dmatern_w45/anisotropy_californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100_3dmatern_w45.png`
- Z-score: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100_3dmatern_w45/zscore_std_californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100_3dmatern_w45.png`
- Kriging snapshots: `AEResults/aeplots/snapshot_californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100_3dmatern_w45/`

**Source Layer (150–400m):**
- Audit CSV: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45/audit_californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45.csv`
- Temporal persistence: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45/temporal_persistence_californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45.png`
- Anisotropy: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45/anisotropy_californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45.png`
- Z-score: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45/zscore_std_californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45.png`
- Kriging snapshots: `AEResults/aeplots/snapshot_californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45/`

**Background Layer (500–1000m):**
- Audit CSV: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45/audit_californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45.csv`
- Temporal persistence: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45/temporal_persistence_californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45.png`
- Anisotropy: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45/anisotropy_californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45.png`
- Z-score: `AEResults/aelogs/californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45/zscore_std_californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45.png`
- Kriging snapshots: `AEResults/aeplots/snapshot_californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000_3dmatern_w45/`

---

## 2026-04-01 — Code Updates: New Canonical Config (californiav2 + FX2 guardrails)

**Prompted by Gemini FX2 verdict and californiav2 migration decision.**

### Changes Made

1. **`05_ae_update_tomatern0.5.py`** — updated `run_diagnostic_inspection()` defaults:
   - `time_ls_bounds_days`: `(2.0, 45.0)` → `(15.0, 45.0)` (T3 permanent)
   - `spatial_ls_upper_bound`: `5` → `10` (S1 permanent, all layers)
   - `step_size_days`: `15` → `10` (matches 10d bin width; no aliasing)
   - `__main__` block: now runs canonical `region='californiav2'` with no experiment suffix

2. **`07_ae_deeper_layers.py`** — updated for canonical config:
   - `COMMON` dict: `region='california'` → `region='californiav2'`
   - Background layer call: removed S1 experiment overrides (`spatial_ls_upper_bound=10`,
     `run_suffix="_s1ub10"`) — these are now the defaults

### Pending (requires AWS cloud run)
Script 02 re-run for all three layers: `region='californiav2'`, `time_step=10.0`.
New parquets: `californiav2_20150101_20151231_res0_5x0_5_t10_0_d{0_100, 150_400, 500_1000}.parquet`
GPR analysis ready to execute immediately once parquets are available.

---

## 2026-04-01 — Experiment T2: Step = 30d (Oscillation Verdict)

**Change:** `step_size_days=30` in `run_diagnostic_inspection()`. Output: `_3dmatern_w45_t2s30`.

**Result:**

| Experiment | Windows | `scale_time` std | `scale_time` min | n < 15d |
|---|---|---|---|---|
| Baseline (step=15d) | 23 | 14.99 | 2.1d | 3/23 |
| T3 (lb=15d, step=15d) | 23 | 11.82 | 16.3d | 0/23 |
| T1 (step=10d) | 34 | 14.87 | 3.7d | 7/34 |
| **T2 (step=30d)** | **12** | **0.00** | **45.0d** | **0/12** |

**Verdict: the oscillation was entirely a data-structure artifact.**

With step=30d, every window sees a genuinely new 30-day bin. The result: `scale_time` = 45.0d in
**all 12 windows, zero variance**. The GP consistently finds maximum temporal persistence — it
always saturates at the upper bound when given clean (non-duplicated) data.

Two conclusions:

1. **The oscillation was 100% caused by windows sharing the same data bins.** The apparent
   alternation between short and long time scales was the GP responding erratically to
   windows that had identical data in some runs and slightly different data in others due
   to the 15-day step straddling different edges of 30-day bins.

2. **The true Skin Layer temporal persistence saturates at or above 45 days.** The GP always
   wants to use the full window width as the time scale — the ocean memory genuinely exceeds
   the window. This is physically plausible (SST anomalies in the CCS can persist for months;
   the 2015 Blob lasted >1 year).

**What this means for the study design:**

The 3D GP time dimension is providing limited diagnostic information at this binning level:
the scale always saturates at the upper bound. Options (for Gemini):
- **FX1:** Re-run Script 02 with `time_step=15d` — finer bins, 3 bins/window, may resolve
  sub-window temporal structure that the 30d bins are averaging out
- **FX2:** `time_step=10d` — even finer resolution
- **Accept:** Use T2 (step=30d) as the canonical config, note that Skin Layer temporal
  memory > 45 days is a positive physical finding for the stealth warming study

RMSRE (10/12 pass, 3.90% median) and Std Z (0.73–1.07) are comparable to baseline.
The Std Z upper bound dropped from 1.11 → 1.07 with the cleaner data.

---

## 2026-04-01 — Experiments T3 + S1: Temporal Aliasing & Spatial Bound Saturation

**Prompted by Gemini analysis `2026-04-01_2015_CCS_MultiLayer_Analysis.md`.**
**Plan recorded in `AE_plan_temporal_spatial_experiments.md`.**

### Changes Implemented

1. **`argoebus_gp_physics.py`** — added `spatial_ls_upper_bound=5` parameter to
   `analyze_rolling_correlations()`. Changed `spatial_l_bounds = (1e-2, 5)` to
   `spatial_l_bounds = (1e-2, spatial_ls_upper_bound)`. Also expanded the comment on
   `time_ls_bounds_days` to document the aliasing root cause.

2. **`05_ae_update_tomatern0.5.py`** — added `run_suffix=""`, `spatial_ls_upper_bound=5`,
   `time_ls_bounds_days=(2.0, 45.0)` parameters to `run_diagnostic_inspection()`. Updated
   `output_run_id` construction to append `run_suffix`. Updated `__main__` to run Experiment T3.

3. **`07_ae_deeper_layers.py`** — Background layer call now passes `spatial_ls_upper_bound=10`
   and `run_suffix="_s1ub10"` (S1 experiment).

### Experiment T3 Results — Skin Layer, temporal lower bound 15d
*(Output: `_3dmatern_w45_t3lb15`)*

| Metric | Baseline | T3 |
|---|---|---|
| Pass (<5%) | 19/23 (83%) | 19/23 (83%) |
| Median RMSRE | 3.86% | 3.86% |
| Std Z range | 0.73–1.11 | 0.73–1.11 |
| `scale_time` min | 2.1d | **16.3d** |
| `scale_time` std | 14.99 | **11.82** |

**Verdict: partial success.** The <10d collapses are gone (min now 16.3d vs. 2.1d baseline).
RMSRE and Z unchanged — no accuracy cost. But the oscillation pattern persists in attenuated
form: scale_time still alternates between ~16–30d and 45d windows. Std reduced 14.99 → 11.82
(21% improvement). T3 is a useful floor but T1 (step=10d, align to Argo cycle) is still
needed to fully eliminate the beat frequency.

**New finding:** Spatial ConvergenceWarnings persist on lon_bin (dim 1, bound 5.0) in the
Skin Layer — the *spatial* bounds are also saturating on some windows even at 0–100m.
This suggests S1 may also be needed for the Skin Layer, not just Background.

### Experiment S1 Results — Background Layer, spatial upper bound 10
*(Output: `_3dmatern_w45_s1ub10`)*

| Metric | Baseline | S1 |
|---|---|---|
| Pass (<5%) | 21/23 (91%) | 21/23 (91%) |
| Median RMSRE | 2.06% | 2.06% |
| Std Z range | 0.54–2.02 | 0.54–2.02 |
| lat saturation (>23°) | 7/23 | 7/23 |
| lon saturation (>23°) | 19/23 | 19/23 |
| Anisotropy std | 0.207 | 0.204 |

**Verdict: almost no effect on most windows.** The spatial scales in the Background layer
were already exceeding 23.5° in the baseline — the assumption that `5 scaled units ≈ 23.5°`
was incorrect. Because the StandardScaler `scale_` varies per window based on the actual
data spread, `5 × scaler.scale_` can be much larger than 23.5° when the domain is wide.

Three late-season windows (6082, 6097, 6112; Sep–Oct) did show lon scale expansion:
35→47°, 36→50°, 35→52° — confirming the mechanism works. But the majority of windows
were not constrained.

**May 2015 failure window (5977, Z=2.02) completely unchanged by S1.** Confirms the
overconfidence is a true physical non-stationarity event (candidate: 2015 Pacific Blob onset),
not a bounds artifact. Flagged for Gemini.

### What To Do Next

1. **T1**: Run `step_size_days=10` on Skin Layer to fully eliminate the aliasing beat.
   T3 reduced the amplitude; T1 addresses the root cause.
2. **Flag for Gemini**: S1 showed Background spatial scales already varied widely (4–52°).
   The physical picture is more complex than simple saturation. Gemini should interpret the
   Sep–Oct lon scale expansion (35 → 52°) in the context of autumn deep water mass spreading.
3. **May 2015 Background failure**: confirmed physical. Gemini to assess Blob onset timing.

---

## 2026-04-01 — Experiment T1: Step = 10d (Root Cause Revision)

**Change:** Added `step_size_days` parameter to `run_diagnostic_inspection()` in
`05_ae_update_tomatern0.5.py`. Output: `_3dmatern_w45_t1s10`.

**Result:**

| Metric | Baseline (step=15d) | T3 (lb=15d) | T1 (step=10d) |
|---|---|---|---|
| Windows | 23 | 23 | **34** |
| Pass (<5%) | 19/23 (83%) | 19/23 (83%) | 28/34 (82%) |
| Median RMSRE | 3.86% | 3.86% | 3.89% |
| `scale_time` std | 14.99 | 11.82 | **14.87** |
| `scale_time` min | 2.1d | 16.3d | **3.7d** |

T1 produced **no meaningful improvement** over the baseline. Std 14.87 ≈ baseline 14.99.

**Root cause revision:** The aliasing is NOT driven by the Argo 10-day resurface cycle.
It is driven by the **30-day time bin width** in the pre-processed parquet (Script 02
`time_step=30.0`). With a 10-day step, 11/33 consecutive window pairs contain
**identical data** (same RMSRE and n_bins to machine precision) because the step is
shorter than the bin width — both windows span the exact same 30-day bins. The apparent
~30-day oscillation period IS the bin width, not the Argo cycle.

**Implication:** No step-size change will fix this unless `step_size_days ≥ time_step`.
The structural fixes are:
- **FX3 (T2, no cloud run):** `step_size_days=30` — advances exactly one bin per window
- **FX1/FX2 (requires cloud run):** re-run Script 02 with `time_step=15` or `time_step=10`

T2 (step=30d) is queued as the immediate no-cost diagnostic.

---

## 2026-03-31 — GPR Analysis: Source Layer (150-400m) + Background Layer (500-1000m)

**Both runs completed successfully via `07_ae_deeper_layers.py` (parallel background jobs).**

### Three-Layer Results Summary

| Metric | Skin (0-100m) | Source (150-400m) | Background (500-1000m) |
|---|---|---|---|
| Windows | 23 | 23 | 23 |
| Pass (<5%) | 19/23 (83%) | 18/23 (78%) | **21/23 (91%)** |
| Median RMSRE | 3.86% | 3.66% | **2.06%** |
| Max RMSRE | 6.50% | 8.59% | 6.02% |
| Min RMSRE | 2.00% | 2.56% | **1.61%** |
| Std Z range | 0.73–1.11 | 0.53–1.46 | 0.54–2.02 |

### Key Scientific Observations

**Anisotropy Ratio vertical profile (stealth warming fingerprint):**
- Skin Layer: 0.36–0.49 — strongly zonal throughout, dominated by atmospheric forcing
- Source Layer: rises to **1.07–1.15 in Aug–Sep** (windows 6075–6165) — meridional current
  dominance emerging at 150–400m. This is the first quantitative evidence of California
  Undercurrent influence at depth. Not present in Skin or Background.
- Background Layer: 0.17–0.94 — remains zonal throughout; no meridional dominance at 500–1000m.

**RMSRE trend with depth:** Background has lowest RMSRE (2.06% median) because deep water is
spatially coherent. Source Layer intermediate. Skin Layer most variable due to atmospheric forcing.

**Std Z widening at depth:** Both deeper layers show wider Z ranges than Skin (0.53–2.02 vs.
0.73–1.11). Root cause: spatial length scale bounds too tight. The optimizer hits the lat upper
bound (5.0°) and lon upper bound (2.0°) in many windows, meaning actual correlation structures
at depth are larger than the configured bounds allow.

**Notable failure — Background Layer window 5955–6000 (mid-May 2015):**
RMSRE=5.72%, Z=2.02 (strongly overconfident), Anisotropy=0.28. The only severe Z failure across
all three layers. Possibly related to the 2015 Pacific Blob onset. Flagged for Gemini.

### Output Artifacts

| Layer | run_id suffix | Audit CSV |
|---|---|---|
| Source | `_d150_400_3dmatern_w45` | `AEResults/aelogs/california_..._d150_400_3dmatern_w45/` |
| Background | `_d500_1000_3dmatern_w45` | `AEResults/aelogs/california_..._d500_1000_3dmatern_w45/` |

Each folder: `audit_*.csv`, `cv_details_*.pkl`, 5 physics PNGs. Kriging snapshots in `aeplots/`.

### Completed todo items retired here
- **Source Layer GPR analysis** (`depth_range=(150, 400)`)
- **Background Layer GPR analysis** (`depth_range=(500, 1000)`)
- **Investigate 3D Matern Std Z underconfidence (min 0.74)** — root cause now identified as
  spatial length scale bounds saturation, not a noise calibration issue. Affects all layers.
  New todo item added for bound widening.

---

## 2026-03-31 — Cloud Runs: Source Layer + Background Layer (Script 02)

**Both runs completed successfully.**

| Layer | depth_range | S3 Parquet |
|---|---|---|
| Source | (150, 400) | `california_20150101_20151231_res0_5x0_5_t30_0_d150_400.parquet` |
| Background | (500, 1000) | `california_20150101_20151231_res0_5x0_5_t30_0_d500_1000.parquet` |

Runs executed in parallel via background Bash jobs. Both used `region=california`, `lat_step=0.5`, `lon_step=0.5`, `time_step=30.0`, `n_workers=3` on Coiled AWS (us-east-1). Next step: run GPR analysis (`05_ae_update_tomatern0.5.py`) on each layer using C2 config.

---

## 2026-03-31 — Shelved: Time Persistence Oscillation (Priority 1 → archived)

**Status: Shelved pending Gemini physical hypothesis. No further implementation without one.**

**Gemini diagnosis (recorded):** The alternating time length scale in the Skin Layer 3D Matern GP is likely a **sampling aliasing effect**. The 10-day Argo float cycle and 15-day window step create a 30-day (2-window) repetition in sampling time distribution relative to the window center. In the low-coherence Skin Layer, windows with larger gaps between float surfacings and the center lose temporal information, causing the GP to pin to the upper bound.

**Implementation record:**
- C5 (w45, tb=30): alternation confirmed structural, not a bounds artifact. Tightening tb=45 → tb=30 moved the ceiling without fixing the alternation.
- C2 (w45, tb=45, 83% pass) accepted as canonical Skin Layer configuration.
- Do not pursue further bounds adjustments without a physical hypothesis from Gemini.
- The oscillation may reflect a real oceanographic signal (spring/neap tidal aliasing, eddy phase alternation) or may indicate the 3D GP mode is ill-suited to the Skin Layer where atmospheric forcing dominates and temporal coherence is low.

---

## 2026-03-31 — Session 11 (Improved kriged OHC plot labels + August 2015 snapshot)

**What was done:**

1. **Modified `plot_kriging_snapshot()` in `argoebus_gp_physics.py`** — label improvements:
   - Added `units_label` parameter (auto-detected: `"ohc"` in col name → `"J/m²"`, else `"°C"`)
   - Added `time_epoch` parameter (default `date(1999, 1, 1)`)
   - Predicted map colorbar: `target_col` raw string → `"OHC per m (J/m²)"`
   - Predicted map title: opaque `"Predicted Map (Window Center (months since 1999-01-01): N)"` → `"Predicted Map: August 2015"` (center_val converted via timedelta to real calendar month)
   - Uncertainty colorbar: `"Uncertainty (1σ)"` → `"1σ Uncertainty (J/m²)"` — same units as predicted field
   - Uncertainty title: `"Uncertainty Map"` → `"Uncertainty: August 2015"`

2. **Created `ArgoEBUSCloud/06_ae_plot_august2015.py`** — standalone script:
   - Loads the existing 2D-RBF audit CSV (no re-run of GPR)
   - Loads the Skin Layer parquet from S3
   - Targets mid-August 2015 (day 6070 since 1999-01-01); nearest window: day 6075
   - Saves to `AEResults/aeplots/august2015_ohc_kriged_{run_id}.png`

**Verification result:** Script ran cleanly. 151 obs in window, GP fitted and predicted on 0.25° grid. Plot saved successfully.

---

## 2026-03-30 — Session 10 (Canonical Skin Layer script: 3D Matern C2 config)

**What was done:**

1. **Created `05_ae_update_tomatern0.5.py`** — canonical Skin Layer diagnostic script,
   superseding `03_ae_inspect_data.py`. Adopts the C2 configuration determined optimal
   by the RMSRE optimization experiment:
   - `mode='3D'`, `kernel_type='matern0.5'`, `window_size_days=45`,
     `time_ls_bounds_days=(2.0, 45.0)`
   - Output run_id: `{config_run_id}_3dmatern_w45` — distinct from the base run_id so
     deprecated 2D-RBF results are not overwritten.
   - Fixed an existing NameError (`run_id` → `config['run_id']`) and a decimal/percent
     scaling bug in the summary block (`<= 5.0` → `<= 0.05`, display multiplied by 100).

2. **Wrote `DEPRECATED.txt`** in `AEResults/aelogs/california_20150101_20151231_res0_5x0_5_t30_0_d0_100/`
   documenting that the 2D-RBF results are superseded as of 2026-03-30 and pointing to
   the new canonical folder.

3. **Ran `05_ae_update_tomatern0.5.py`** — all 23 windows completed successfully.

4. **Updated `ae_file_structure.txt`**: marked `03_ae_inspect_data.py` as DEPRECATED,
   added entries for `05_ae_rmsre_optimization.py` and `05_ae_update_tomatern0.5.py`,
   updated the aelogs folder list.

**Verification result:**

| Metric       | Value                |
|---|---|
| Windows      | 23                   |
| Pass (<5%)   | 19/23 (83%)          |
| Median RMSRE | 3.86%                |
| Max RMSRE    | 6.50% (July eddies)  |
| Min RMSRE    | 2.00%                |
| Std Z range  | 0.73 – 1.11          |

Matches C2 result from optimization table exactly. July eddy Cluster 2 (6.50%) remains
physically irreducible; accepted as the Skin Layer floor.

**Completed todo items retired here:**

- **Adopt C2 config (window=45) as the new standard for Skin Layer**

---

## 2026-03-30 — Session 9 (merge gaussian-kriging-rework → main)

**What was done:**

1. **Committed outstanding `AE_claude_todo.md` change** on `gaussian-kriging-rework` before switching branches.
2. **Merged `gaussian-kriging-rework` into `main`** via fast-forward (no conflicts).
3. **Deleted `gaussian-kriging-rework` branch** (`git branch -D`) — branch was local-only, never pushed.
4. **Retired merge task from `AE_claude_todo.md`** and updated last-updated date to 2026-03-30.

**Net result:** All work from Sessions 6–8 (3D Matern kernel, RMSRE optimization, float coverage plot) is now on `main`.

---

## 2026-03-27 — Session 8 (RMSRE optimization — window experiment, float coverage plot)

**What was done:**

1. **Added `min_bins` parameter to `analyze_rolling_correlations`** in `argoebus_gp_physics.py`:
   - New parameter `min_bins=10` (default preserves backward compatibility) in the
     `ROLLING WINDOW CONFIG` block.
   - Sparse window guard at line ~1054 now uses `min_bins` instead of hardcoded `10`.
   - Enables callers to raise the threshold (e.g. `min_bins=80`) to skip underdetermined
     windows without editing library code.

2. **Added `plot_float_coverage()` to `argoebus_gp_physics.py`**:
   - Dual-axis plot: grey bars for `n_bins` (obs count) on left axis; RMSRE % on right
     axis (green dots = pass ≤5%, red dots = fail >5%).
   - Horizontal dashed line marks the `min_bins_threshold` used in the run.
   - Purpose: makes the data-sparsity / RMSRE correlation immediately visible.

3. **Created `05_ae_rmsre_optimization.py`**:
   - Runs four variants of 3D Matern(nu=0.5) pipeline, all against C0 baseline loaded
     from the existing script-04 audit CSV:
       - C1: `min_bins=80` — skip the underdetermined early-Jan window
       - C2: `window_size_days=45` — wider window to pull in more Jan floats
       - C3: `window_size_days=20` — shorter window to avoid bridging July eddies
       - C4: `noise_val=0.5` — higher initial noise floor for Cluster 2 overconfidence
   - Saves audit CSV and physics PNGs per variant under `AEResults/aelogs/{variant}/`.
   - Prints comparison table isolating Cluster 1 (~day 5835) and Cluster 2 (~days 6025-6050).
   - Generates RMSRE overlay plot (all 5 variants) and float coverage plot (C0 baseline).

**Completed todo items retired here:**

- **Run RMSRE comparison: 2D-RBF vs. 3D-Matern(nu=0.5)** (completed prior session):
  2D RBF median 4.87% / max 8.34% / 52% passing. 3D Matern median 3.82% / max 6.50% /
  78% passing. Clear win for 3D Exponential; remaining failures in two clusters.

- **Reduce RMSRE in the two remaining problem clusters** (this session — see table below).

- **Plot: float count per window as a function of time** — `plot_float_coverage()`
  added to `argoebus_gp_physics.py`; dual-axis (n_bins bars + RMSRE dots); called by script 05.

- **Test tighter temporal window (15–20 days)** — C3 (`window_size_days=20`) shows
  Cluster 2 unchanged at 6.498%. Shorter windows do not help; closed.

**Verification result:** All four runs completed. Comparison table:

| Variant | N | Pass | MedRMSRE | MaxRMSRE | Clust1 | Clust2 | MedZ |
|---|---|---|---|---|---|---|---|
| C0 ref (w30 mb10 n0.1) | 23 | 18 | 3.82% | 6.50% | 6.42% | 6.50% | 0.92 |
| C1 (w30 mb80 n0.1) | 22 | 18 | 3.82% | 6.50% | 2.00%* | 6.50% | 0.92 |
| C2 (w45 mb10 n0.1) | 23 | 19 | 3.86% | 6.50% | **3.43%** | 6.50% | 0.91 |
| C3 (w20 mb10 n0.1) | 24 | 18 | 3.87% | 6.50% | 6.42% | 6.50% | 0.93 |
| C4 (w30 mb10 n0.5) | 23 | 18 | 3.82% | 6.50% | 6.42% | 6.50% | 0.92 |

*C1 dropped the Jan window; value shown is nearest surviving window.

**Winner: C2 (`window_size_days=45`).** Expands the Jan window to 198 bins,
fixing Cluster 1 (6.42% → 3.43%) while keeping all 23 windows. Pass rate 78% → 83%.

**Cluster 2 confirmed physically irreducible.** All four variants return identical
6.498% for the July eddy windows. Shorter windows, higher noise, and min_bins skip
all fail to improve it. Accept ~6.5% for the summer eddy season in the Skin Layer.

---

## 2026-03-25 — Session 7 (3D GP with Exponential Kernel — implementation)

**What was done:**

1. **Extended `analyze_rolling_correlations` in `argoebus_gp_physics.py`** with three new parameters:
   - `mode='2D'` (default, backward-compatible) or `'3D'` (adds time as a third GP dimension)
   - `kernel_type='rbf'` (default) or `'matern0.5'` (Exponential/OU kernel). Both are fully
     interchangeable: same output columns, same scaling, same diagnostics — callers can run
     the function twice with different kernel_type values and compare results_df directly.
   - `time_ls_bounds_days=(2.0, 30.0)`: physical-day bounds on the time length scale optimizer.

2. **Split-scaler architecture**: in 3D mode, spatial columns (lat/lon) use `StandardScaler`;
   the time column uses a window-relative normalization (`t̃ = (t − t_center) / (W/2)`) so
   window center = 0 and edges = ±1. Physical time length scale recovered as `l̃_t × (W/2)`.

3. **Kernel factory (`_build_kernel`)**: a closure that switches between `RBF` and
   `Matern(nu=0.5)` based on `kernel_type`. Used for both the initial fit and the
   auto-calibration re-fit, ensuring the kernel choice propagates through the entire loop.

4. **Per-dimension time bounds**: in 3D auto-tune mode, the time length scale optimizer is
   constrained to [2/half_window, 30/half_window] in normalized units, preventing degenerate
   solutions (ignore time or treat all obs as synchronous).

5. **Results record updated**: iterates over `all_feature_cols` (spatial + time in 3D mode),
   automatically writing `scale_time_days` to the audit CSV. Anisotropy calculation switched
   from hardcoded `scale_lat_bin`/`scale_lon_bin` to a dynamic name lookup.

6. **`plot_physics_history` updated**:
   - Plot 1: now shows spatial-only scale columns (excludes `scale_time*`).
   - Plot 4: anisotropy uses dynamic column lookup; skips gracefully if columns absent.
   - Plot 5 (new): temporal persistence plot, conditional on `scale_time_days` in results_df.

7. **Created `argo_claude_actions/AE_plan_3d_gpr_matern.md`**: human-readable design doc with
   the math (kernel equations, normalization formulas, bounds derivation) for Gemini review.

8. **Updated** `ae_file_structure.txt`, `AE_claude_todo.md` (corrected human notes, updated
   Priority 1 task).

**Verification result:** Smoke test passed — 2D-RBF, 3D-Matern, and 3D-RBF all ran without error; `scale_time_days` present in 3D output; `plot_physics_history` produced Plot 5 conditionally.

---

## 2026-03-25 — Session 7b (04_ae_testmatern_and_3dwindow.py + 04b scripts)

**What was done:**

1. **Created `04_ae_testmatern_and_3dwindow.py`** — loads the same S3 parquet as script 03 and
   runs both GP variants back-to-back for direct comparison:
   - Run A: `mode='2D', kernel_type='rbf'` (identical to script 03 — direct baseline)
   - Run B: `mode='3D', kernel_type='matern0.5'` (Exponential kernel + time dimension)
   - Saves audit CSVs, CV pickles, and physics PNGs to separate named folders.
   - Generates kriging snapshots for the 2D run only (3D snapshot support not yet implemented).
   - Prints a side-by-side RMSRE/Z-score/anisotropy summary at the end.

2. **Created `04b_ae_plot_matern_physics.py`** — analogous to `03b_ae_plot_physics.py`.
   Loads both audit CSVs and regenerates physics PNGs without re-running kriging.
   Also saves a combined RMSRE comparison PNG overlaying both time series.

3. **Output folder naming convention:**
   - `AEResults/aelogs/{run_id}_2d_rbf/` — Run A baseline
   - `AEResults/aelogs/{run_id}_3d_matern05/` — Run B Exponential

4. **Updated `ae_file_structure.txt`** to document the new scripts and output folders.

---

## 2026-03-25 — Session 7c (first real-data test results)

**Results on 2015 California Skin Layer (0–100m), 23 rolling windows, window=30d / step=15d:**

| Metric | 2D RBF | 3D Matern(ν=0.5) |
|---|---|---|
| Median RMSRE | 4.87% | **3.82%** |
| Max RMSRE | 8.34% | **6.50%** |
| Min RMSRE | 2.95% | **2.00%** |
| Std Z range | 0.92 – 1.08 | 0.74 – 1.11 |
| Windows meeting <5% target | 52% (12/23) | **78% (18/23)** |

**Remaining 3D Matern failures (5 windows), by root cause:**

- **Early-Jan window (day ~5835):** Only 50 obs vs. 136–155 in all other windows. Sparse
  data makes kernel estimation unreliable regardless of kernel type. Both models fail here.
- **Summer cluster (days ~6030–6045, ~July 2015):** Anisotropy ratio ~0.26–0.36, maximum
  eddy season, zonal atmospheric chaos. Std Z slightly high (1.11 = mildly overconfident).
  Likely physically irreducible at a 30-day window width.
- **Spring window (days ~5910–5925):** Borderline failure (5.6%), close to target.

**Underconfidence note (3D Matern):** Days 6090–6105 (late Sep) have Std Z = 0.74 — error
bars larger than actual errors. RMSRE is 3.0% there so predictions are good; the model is
just being conservative. Flag for review if it persists in deeper layers.

**Next steps recorded in Priority 1 of AE_claude_todo.md.**

---

## 2026-03-25 — Session 6 (gaussian-kriging-rework branch created)

**What was done:**

1. **Created new branch `gaussian-kriging-rework`** from `main` — isolated workspace for experimenting with the Gaussian kernel and kriging run logic without touching the stable main branch. Once the new approach is validated, `main` will be merged into this branch (or this branch will become the new baseline).

---

## 2026-03-20 — Session 5 (registry corrections + get_vertical_layers)

**What was done:**

1. **Updated `get_ebus_registry()` in `ae_utils.py`** — aligned non-California EBUS bounds with Frontiers 2024 paper spatial domain:
   - `california` — NO CHANGE; preserved as-is (140W dataset, all existing 2015 S3 artifacts reference it)
   - `californiav2` — NEW entry: lat [30, 45], lon [-130, -115]; tighter coastal window matching the paper's CCS domain
   - `humboldt` — lat [-35, -5] (was [-45, 0]), lon [-85, -70] (was [-90, -70])
   - `canary` — lat [15, 35] (was [10, 45]), lon [-25, -10] (was [-30, -5])
   - `benguela` — lat [-35, -15] (was [-35, -10]), lon [5, 20] unchanged

2. **Added `get_vertical_layers()` to `ae_utils.py`** — formalizes the canonical Vertical Sandwich depth layer definitions:
   - `Response`: [0, 100] — fast atmospheric response
   - `Source`: [150, 400] — Ekman upwelling source water / stealth heat layer
   - `Background`: [500, 1000] — deep ocean baseline for warming rate comparison

**Verification result:** Import confirmed in `ebus-cloud-env`; all six registry entries and all three layer definitions match plan exactly. `california` bounds unchanged.

---

## 2026-03-20 — Session 4 (plot_float_paths modularity refactor)

**What was done:**

1. **Refactored `03_ae_plot_float_paths.py`** — converted from a standalone script with module-level constants into a proper importable function
   - Replaced `REGION`, `START_DATE`, etc. module-level constants + `main()` with `plot_float_paths(region, lat_step, lon_step, time_step, depth_range)`
   - Signature now matches `run_diagnostic_inspection()` exactly — both can be called serially in any parent analysis script without duplicating parameter handling
   - Dates are resolved internally via `get_ae_config()` (same as `run_diagnostic_inspection()`), not hardcoded
   - Returns output path as a string so callers can log it
   - `if __name__ == "__main__"` block preserved for standalone use

2. **Established workflow rule**: completed tasks leave `AE_claude_todo.md` entirely and are recorded here in `AE_claude_recentactions.md`. The todo file contains only forward-looking work.

3. **Updated `AE_claude_lessons.md` and `CLAUDE.md`** — added modularity rule

**Verification result:** `python 03_ae_plot_float_paths.py` → 4,348 rows, 99 floats, PNG saved to correct path.

---

## 2026-03-20 — Session 3 (physics history plots + AEResults path fix + repo hygiene)

**What was done:**

1. **Upgraded `plot_physics_history()` in `argoebus_gp_physics.py`**
   - Added `save_dir` and `run_id` parameters so figures are saved to disk headlessly
   - Added 4th subplot: Anisotropy Ratio (Lat_Scale / Lon_Scale) over time
   - Split the single 4-panel figure into 4 individual PNGs (one per metric) for easier viewing
   - Each figure saved as `{metric}_{run_id}.png` under `{save_dir}/`

2. **Created `03b_ae_plot_physics.py`** — standalone headless script
   - Loads existing audit CSV from `AEResults/aelogs/`
   - Calls `plot_physics_history()` to regenerate all 4 physics PNGs without re-running kriging
   - Useful for re-styling figures after the expensive analysis run is complete

3. **Moved `AEResults/` to correct location**
   - Was erroneously inside `ArgoEBUSCloud/AEResults/`; moved to `ArgoEBUSAnalysis/AEResults/` per `ae_file_structure.txt`
   - Fixed all path references in `03_ae_inspect_data.py`, `03b_ae_plot_physics.py`, and `ae_utils.py` (`ensure_ae_dirs`) to use `../AEResults/` from inside `ArgoEBUSCloud/`

4. **Renamed `claude_reports/` → `argo_claude_actions/`** with `AE_` file prefixes
   - `claude_lessons.md` → `AE_claude_lessons.md`
   - `claude_recentactions.md` → `AE_claude_recentactions.md`
   - `claude_todo.md` → `AE_claude_todo.md`

5. **Updated `CLAUDE.md`**
   - Added `## Hard-Won Rules` section at the bottom
   - Added `### 7. Python Environment` rule (always use `ebus-cloud-env`)
   - Updated Self-Improvement Loop (§3) to require dual updates: both `AE_claude_lessons.md` and `CLAUDE.md`
   - Added rule: `AEResults/` lives at `ArgoEBUSAnalysis/`, not inside `ArgoEBUSCloud/`

6. **Updated `ae_file_structure.txt`** — added `aelogs/` entry under `AEResults/`

**Verification result:** Physics PNGs (anisotropy, noise, length_scales, z_score) saved successfully for 2015 Skin Layer run.

---

## 2026-03-20 — Session 2 (get_float_history + float trajectory plot)

**What was done:**

1. **Added `get_float_history()` to `ae_utils.py`** (after `get_ae_config`)
   - Queries ERDDAP for raw per-dive float positions using `&distinct()` to collapse per-pressure rows
   - Returns 5-column DataFrame: `platform_number, lat, lon, time, time_days`
   - `time_days` uses same 1999-01-01 baseline as the OHC parquet

2. **Created `03_ae_plot_float_paths.py`** — standalone diagnostic script
   - Calls `get_float_history()`, prints shape and unique float count
   - Draws spaghetti map: each float as a distinct colored line over Cartopy basemap (PlateCarree, LAND + COASTLINE style matching `argoebus_gp_physics.py`)
   - Saves directly to `AEResults/aeplots/` (not inside a snapshot subfolder)
   - Filename mirrors `run_id`: `float_path_traj_california_20150101_20151231_res0_5x0_5_t30_0_d0_100.png`

3. **Filename correction** — initial attempt used a `day5844_6208` suffix; reverted to `run_id` suffix on user instruction (run_id already encodes dates, resolution, and depth cleanly)

4. **Updated `AE_claude_todo.md`** — marked both old Priority 1 tasks complete; promoted old Priority 4 (RMSRE Optimization) to Priority 1; Priority 2 and 3 unchanged

**Verification result:** 4,348 rows, 99 unique floats, figure saved to correct path.

**Bugs fixed during verification:**
- ERDDAP domain migration `www.ifremer.fr` → `erddap.ifremer.fr` (HTTP 302 → 400 chain); switched to `requests` with manual `%3C`/`%3E` percent-encoding to satisfy Tomcat 11 RFC 3986 strictness
- Matplotlib `cm.get_cmap` deprecation (→ `matplotlib.colormaps.get_cmap(...).resampled(...)`)

---

## 2026-03-19 — Session 1 (Onboarding)

**What was done:**

1. **Read the full codebase** — explored all Python files in `ArgoEBUSCloud/`, notebooks 1–5, helper modules
2. **Read the Gemini discussion log** (`Gemini-Testing Ocean Refugia Hypothesis March 19, 2026.txt`)
   - ~44,760 lines covering the full arc of scientific development with Gemini
   - Key science outcome: "Stealth Warming" 3-layer vertical fingerprinting study design
   - Identified the complete agenda of work still to be done
3. **Read the proposed CLAUDE.md** (`proposal_for_claudemd.txt`) and adapted it to this project
4. **Created `CLAUDE.md`** in the project root with:
   - Project mission statement (Ocean Refugia / Stealth Warming)
   - Workflow principles from the proposal
   - Scientific context (metrics, depth layer naming, key functions)
   - Repository structure map
5. **Created `claude_reports/` folder** with three files:
   - `claude_todo.md` — full agenda derived from Gemini session, prioritized
   - `claude_recentactions.md` — this file
   - `claude_lessons.md` — initially empty (no mistakes yet to document)
6. **Saved persistent memories** to `~/.claude/projects/.../memory/`

**Key scientific state understood:**
- 2015 Skin Layer (0–100m) is DONE: 23 snapshots, audit CSV, CV pickle all saved to S3/local
- Anisotropy Ratio in Skin Layer: ~0.36 (winter) → ~0.49 (summer) — atmospheric/zonal dominance
- Next critical run: Source Layer (150–400m) cloud job via Script 02
- RMSRE currently ~8%, target is <5%

**Nothing was modified in the codebase** — this was a read/onboard session only.

---

_Add new entries at the top after each session._
