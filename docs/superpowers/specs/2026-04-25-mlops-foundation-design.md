# MLOps Foundation — Config-Driven Runs + Reproducibility Manifests

**Date:** 2026-04-25
**Status:** Approved (design)
**Author:** Avik Mondal (with Claude)
**Audience target (sequenced):** practitioner-grade infra first, then hiring-target demo polish layered on top.

---

## 1. Motivation

The ArgoEBUSAnalysis pipeline currently runs via hand-edited `__main__` blocks in
`ArgoEBUSCloud/02_ae_cloud_run.py`, `05_ae_update_tomatern0.5.py`, `07_ae_deeper_layers.py`.
Two pains block scaling:

1. **Region scaling (B from brainstorm):** adding humboldt / canary / benguela means
   editing scripts and `__main__` blocks. No one-shot CLI like
   `aebus analyze configs/humboldt/d150_400_3dmatern_w45.yaml`.
2. **Reproducibility (C from brainstorm):** no record of dependency versions, git SHA,
   ERDDAP query timestamps, or which config produced which audit CSV. Re-running a
   month later may give different results with no way to detect drift.

This spec addresses **both** with one foundation, since configs without manifests
drift and manifests without configs are uninterpretable.

Out of scope (deferred to later specs):
- MLflow / W&B / dedicated experiment tracker (next phase, "B")
- pip-installable package, Docker, public dashboard (long-term, "C")
- New science features — RG-Gibbs has its own spec
  (`2026-04-11-rg-gibbs-nonstationary-gpr-design.md`)
- Migration of legacy scripts — additive only, old scripts keep working

---

## 2. Architecture Overview

Additive layer atop existing pipeline. No existing script is modified.

**New components:**

1. **`ebus_core/config_schema.py`** — Pydantic models `IngestionConfig` and
   `AnalysisConfig`. One source of truth for valid YAML.
2. **`ebus_core/manifest.py`** — config canonicalization + sha256 hash, git/env
   capture, manifest read/write, registry append.
3. **`ebus_core/runner.py`** — `run_ingestion(cfg)` and `run_analysis(cfg)`. Each
   validates config, computes run_id, calls existing pipeline functions, writes
   manifest, performs collision check.
4. **`ArgoEBUSCloud/aebus_cli.py`** — argparse-based CLI exposing
   `ingest | analyze | validate | list | show`.
5. **`configs/`** — versioned YAML configs at repo root, organized by region.
6. **`AEResults/run_registry.jsonl`** — append-only one-line-per-run index.

**Existing components untouched:** `01_ae_cloud_ingestion.py`, `02_ae_cloud_run.py`,
`05_ae_update_tomatern0.5.py`, `07_ae_deeper_layers.py`, `09_*`, `ebus_core/ae_utils.py`,
`ebus_core/argoebus_thermodynamics.py`, `ebus_core/argoebus_gp_physics.py`,
`ebus_core/argoebus_plotting.py`. `__main__` blocks continue to work as escape hatch.

### Data flow

```
configs/californiav3/ingest_d100_500.yaml
  → aebus ingest configs/californiav3/ingest_d100_500.yaml
  → runner.run_ingestion(cfg)
    → calls existing ingestion fn (script 02 logic)
    → writes parquet to S3
    → writes AEResults/aelogs/ingestion/{run_id}/manifest.json
    → appends one line to AEResults/run_registry.jsonl

configs/californiav3/analyze_d100_500_3dmatern_w45.yaml
  (references the ingestion run via S3 path or ingestion run_id)
  → aebus analyze configs/californiav3/analyze_d100_500_3dmatern_w45.yaml
  → runner.run_analysis(cfg)
    → loads parquet
    → calls run_diagnostic_inspection() with parsed args
    → writes audit CSV + plots to AEResults/aelogs/{run_id}/
    → writes manifest.json with link to ingestion manifest hash
    → appends one line to run_registry.jsonl
```

---

## 3. Config Schema

Pydantic models live in `ebus_core/config_schema.py`. YAML files under `configs/`.

### 3.1 IngestionConfig

```yaml
# configs/californiav3/ingest_d100_500.yaml
schema_version: 1
config_kind: ingestion

region: californiav3              # must match get_ebus_registry() entry
date_start: "2015-01-01"          # ISO-8601
date_end:   "2015-12-31"
lat_step: 0.5
lon_step: 0.5
time_step: 10.0                   # days per bin
depth_range: [100, 500]           # [top_m, bottom_m]

cloud:
  backend: coiled                 # coiled | local
  n_workers: 3
  worker_region: us-east-1

s3:
  bucket: argo-ebus-project-data-abm
  # parquet name auto-derived from above fields (current naming convention preserved)

description: "californiav3 Source Layer ingestion, FX2 t10"  # free-form, in manifest
```

### 3.2 AnalysisConfig

```yaml
# configs/californiav3/analyze_d100_500_3dmatern_w45.yaml
schema_version: 1
config_kind: analysis

# Pointer to ingestion — either explicit S3 path or ingestion run_id
input:
  source: s3                       # s3 | ingestion_run
  s3_path: s3://argo-ebus-project-data-abm/californiav3_20150101_20151231_res0_5x0_5_t10_0_d100_500.parquet
  # OR
  # ingestion_run_id: "<run_id of an ingestion manifest>"

# Mirrors ingestion grid (validated against parquet schema on load)
region: californiav3
date_start: "2015-01-01"
date_end:   "2015-12-31"
lat_step: 0.5
lon_step: 0.5
time_step: 10.0
depth_range: [100, 500]

gpr:
  mode: "3D"                       # 2D | 3D
  kernel_type: matern0.5           # rbf | matern0.5 | gibbs (future)
  window_size_days: 45
  step_size_days: 10
  min_bins: 10
  noise_val: 0.1
  time_ls_bounds_days: [15.0, 45.0]
  spatial_ls_upper_bound: 10
  run_suffix: "_3dmatern_w45"      # appended to auto run_id

outputs:
  aelogs_dir: AEResults/aelogs     # default; override for experiments
  aeplots_dir: AEResults/aeplots
  generate_snapshots: true
  generate_physics_plots: true

description: "californiav3 Source Layer canonical 3D Matern w45"
```

### 3.3 Validation rules

- `region` must exist in `get_ebus_registry()`
- `depth_range` must be a positive `[top, bottom]` pair with `top < bottom`. Warn (not error) if it doesn't match a `get_vertical_layers()` canonical entry — custom layers permitted.
- `time_step ≤ step_size_days` (FX bin-aliasing rule, hard-won)
- `time_ls_bounds_days[0] ≥ time_step` (lower bound respects bin width)
- `date_start < date_end`, parsed to `datetime.date`
- Strict mode: unknown YAML keys → ERROR (catches typos)

### 3.4 Schema versioning

`schema_version: 1` at top of every config. Future-proofs against schema breaks.
Schema bumps come with explicit migration notes in `configs/README.md`.

### 3.5 Config canonicalization & hash

- YAML loaded → Python dict (with sorted keys, recursively) → JSON-serialized.
- sha256 of canonical JSON = `config_hash`.
- `description` field excluded from hash (free-form annotation, no behavioral impact).
- Comments don't affect hash (they don't survive YAML parsing).

---

## 4. Run Identity + Manifest

### 4.1 Run ID

Current human-readable pattern preserved verbatim:

```
{region}_{date_start_compact}_{date_end_compact}_res{lat}x{lon}_t{time_step}_d{depth0}_{depth1}{run_suffix}
```

Example:
```
californiav3_20150101_20151231_res0_5x0_5_t10_0_d100_500_3dmatern_w45
```

No hash suffix in the path. Disambiguation enforced by collision detector (§4.3).

### 4.2 Manifest schema

`AEResults/aelogs/{run_id}/manifest.json` (analysis) or
`AEResults/aelogs/ingestion/{run_id}/manifest.json` (ingestion):

```json
{
  "schema_version": 1,
  "kind": "analysis",
  "run_id": "californiav3_..._3dmatern_w45",
  "config_hash": "a3f9b27c4d8e1f02...",
  "created_at": "2026-04-25T14:32:11Z",
  "duration_sec": 412.7,
  "config": { ...full canonicalized config dict... },

  "code": {
    "git_sha": "6bf0e52cc572",
    "git_dirty": false,
    "git_branch": "main",
    "repo_root": "/home/avik2007/ArgoEBUSAnalysis"
  },

  "env": {
    "conda_env_name": "ebus-cloud-env",
    "python_version": "3.11.7",
    "key_packages": {
      "scikit-learn": "1.3.2",
      "xarray": "2024.1.1",
      "numpy": "1.26.3",
      "pandas": "2.1.4",
      "gsw": "3.6.17",
      "coiled": "1.4.5",
      "dask": "2024.1.0",
      "matplotlib": "3.8.2",
      "cartopy": "0.22.0"
    },
    "conda_list_full": "AEResults/aelogs/{run_id}/conda_list.txt"
  },

  "inputs": {
    "source": "s3",
    "s3_path": "s3://argo-ebus-project-data-abm/californiav3_..._d100_500.parquet",
    "ingestion_manifest_hash": "f72ed1c8...",
    "parquet_etag": "<S3 ETag>",
    "parquet_size_bytes": 18234567
  },

  "outputs": {
    "aelogs_dir": "AEResults/aelogs/{run_id}/",
    "audit_csv": "AEResults/aelogs/{run_id}/audit_{run_id}.csv",
    "snapshots_dir": "AEResults/aeplots/snapshot_{run_id}/"
  },

  "host": {
    "hostname": "wsl-ubuntu",
    "platform": "Linux-6.6.87-microsoft-standard-WSL2"
  }
}
```

**Excluded by design:**
- No `metrics_summary` — RMSRE, Z-score, anisotropy live in audit CSV. No duplication.
- No `snapshot_count` or plot enumeration — output dir is enumerable on disk; manifest doesn't drift as you regenerate plots.
- No `config_path` — config content is in `config` field, hash identifies it; path is incidental and rots on rename.

**Cross-reference:** analysis manifest's `inputs.ingestion_manifest_hash` points back
to the ingestion manifest that produced its parquet → full lineage from any plot back
to the ERDDAP fetch.

**Conda env capture:** runner shells out to `conda list --json -n ebus-cloud-env`,
persists full output to `conda_list.txt` next to manifest, plus inlines a curated
`key_packages` dict for quick eyeball.

**`key_packages` whitelist is part of `schema_version: 1`:**
`scikit-learn, xarray, numpy, pandas, gsw, coiled, dask, matplotlib, cartopy`.
Adding/removing entries requires a schema bump. The full `conda_list.txt` is the
authoritative source — the inline dict is a convenience surface.

### 4.3 Collision detector

Before writing, runner checks:
- Does `AEResults/aelogs/{run_id}/manifest.json` already exist?
- If yes, compare `config_hash` of existing manifest vs current run's hash.
- **Hashes match** → exact re-run of identical config. Proceed; emit warning
  ("overwriting prior identical run").
- **Hashes differ** → ABORT with diff message:
  ```
  run_id collision: AEResults/aelogs/californiav3_..._3dmatern_w45/ already exists
    existing manifest config_hash: f72ed1c8
    new run config_hash:           a3f9b27c
  Configs differ. Either:
    - bump run_suffix in your analysis YAML (e.g., _3dmatern_w45_v2)
    - OR pass --force-overwrite (deletes prior outputs)
  ```
- `--force-overwrite` flag bypasses, deletes prior dir, proceeds.

### 4.4 Run registry

`AEResults/run_registry.jsonl` — JSON-Lines, append-only, one line per successful run:

```json
{"run_id": "californiav3_..._3dmatern_w45", "kind": "analysis", "config_hash": "a3f9b27c", "created_at": "2026-04-25T14:32:11Z", "region": "californiav3", "depth_range": [100, 500], "manifest_path": "AEResults/aelogs/californiav3_..._3dmatern_w45/manifest.json"}
```

Scan with `jq` or `pd.read_json("AEResults/run_registry.jsonl", lines=True)`. Per-run
detail still lives in the per-run manifest; registry is just an index.

**Registry line schema (denormalized identity index, fixed at `schema_version: 1`):**
`run_id, kind, config_hash, created_at, region, depth_range, manifest_path`.
Adding fields requires schema bump. For any field not in this set, look up via
`manifest_path` → load full manifest JSON.

---

## 5. CLI Surface

`ArgoEBUSCloud/aebus_cli.py`, argparse-based, no new dependencies.

```
aebus ingest <config.yaml>          # run ingestion stage
aebus analyze <config.yaml>         # run analysis stage
aebus validate <config.yaml>        # parse + schema-check + show resolved run_id; no execution
aebus list [--region R] [--kind K]  # query run_registry.jsonl, print runs as table
aebus show <run_id>                 # cat manifest.json for that run_id
```

**Common flags:**

```
--force-overwrite      # bypass collision detector
--dry-run              # validate + log what would happen, write nothing
--config-dir DIR       # override default `configs/` lookup root
-v / --verbose         # echo manifest fields as they're collected
```

**Invocation:** `python ArgoEBUSCloud/aebus_cli.py <cmd> <args>` directly, OR add a
shell shim if Avik wants the bare `aebus` command (out of scope for additive layer).

**Logging:** runner writes `AEResults/aelogs/{run_id}/run.log` with stdout + stderr
captured via `logging.FileHandler`. Manifest links to it.

### 5.1 Error handling (boundary-only)

| Failure | Behavior |
|---|---|
| YAML syntax error | Pydantic raises, CLI prints location + line number, exit 2 |
| Schema violation (unknown key, bad type, registry mismatch) | Pydantic ValidationError, pretty-printed, exit 2 |
| Run-id collision with differing config_hash | Abort with diff message, exit 3 |
| S3 write failure | Bubble boto3 exception, exit 1, no partial manifest written |
| ERDDAP fetch failure (ingestion) | Bubble exception, exit 1, no parquet written, no manifest |
| Audit CSV write succeeds, manifest write fails | Audit kept, manifest write retried once, then loud error — flag in registry as "incomplete" |

No retries, no fallbacks for impossible scenarios. Trust internal pipeline guarantees.

---

## 6. Backfill (Existing Runs)

**Scope: configs only.** Skip retro-manifests (low fidelity — `git_sha`,
`created_at`, `env` would be unknown / null and would mislead).

For every existing `AEResults/aelogs/*/` directory:
1. Parse the run_id to recover `region`, dates, grid resolution, depth range,
   `run_suffix`.
2. Inspect audit CSV columns to confirm GPR mode (2D/3D) and kernel type.
3. Read script defaults from the version of `05_ae_update_tomatern0.5.py` /
   `07_ae_deeper_layers.py` that produced the run (use git blame if needed).
4. Write `configs/<region>/<derived_filename>.yaml` with `schema_version: 1` and a
   `description` noting "backfilled from {aelogs_dir} on {date}".

Existing dirs without `manifest.json` are flagged "pre-manifest era" by registry
tooling. Useful as templates and as schema test fixtures (every backfilled config
must round-trip through the parser and reproduce its source dir's run_id).

---

## 7. Testing Strategy

`ArgoEBUSCloud/test_mlops_foundation.py`, additive to existing `test_pipeline.py`.

### 7.1 Unit tests

1. **Config schema validation** — for each Pydantic model: valid config parses;
   unknown key rejected; bad region rejected; FX bin-aliasing rule rejected;
   date string parsing.
2. **Hash stability** — same config → same hash. Reorder YAML keys → same hash.
   Add comment → same hash. Change `description` → same hash. Change `noise_val`
   → different hash.
3. **Manifest schema** — manifest dict round-trips through JSON without loss.
   Required fields present. `git_sha` resolves on a clean repo.
4. **Collision detector** — write a manifest, attempt second write with same
   run_id and (a) identical hash → succeeds with overwrite warning,
   (b) different hash → raises with diff message.
5. **Backfilled configs round-trip** — every backfilled `configs/*.yaml` parses
   to a valid config object whose derived `run_id` matches the existing
   `AEResults/aelogs/` dir name.

### 7.2 Smoke test (opt-in)

`AEBUS_RUN_SMOKE=1 pytest -k smoke` — one end-to-end `aebus analyze` against an
existing small parquet (e.g., 2015 Skin Layer). Asserts manifest written, audit
CSV present, run_id derivation correct.

### 7.3 Out of scope

- No tests for existing `run_diagnostic_inspection()` internals (additive layer).
- No tests for Coiled cloud workers (real AWS).
- No tests for plot pixel content.
- No tests for ERDDAP API.

---

## 8. Implementation Phases

Each phase is independently shippable and reviewable as a PR.

### Phase 1 — Schema + manifest + hash (~2 days)
- `ebus_core/config_schema.py`: `IngestionConfig`, `AnalysisConfig` Pydantic models
- `ebus_core/manifest.py`: hash, git/env capture, write/read functions
- Unit tests #1, #2, #3
- No CLI yet, no runner — pure library

### Phase 2 — Runner + collision detector (~2 days)
- `ebus_core/runner.py`: `run_ingestion()`, `run_analysis()` wrappers
- Calls existing pipeline fns, adds manifest write + collision check
- Unit test #4
- Manual smoke from a Python REPL with a backfilled config

### Phase 3 — CLI + registry (~1 day)
- `ArgoEBUSCloud/aebus_cli.py`: `ingest`, `analyze`, `validate`, `list`, `show`
- `AEResults/run_registry.jsonl` append on every successful run
- Smoke test from §7.2

### Phase 4 — Backfill configs (~1 day)
- Enumerate existing `AEResults/aelogs/*/` dirs
- Reverse-engineer config from run_id + audit CSV + script defaults
- Write `configs/<region>/<file>.yaml` for each
- Unit test #5 validates schema against reality

### Phase 5 — Documentation (~half day)
- `README.md`: brief MLOps section + `aebus` quickstart
- `CLAUDE.md`: config-driven workflow as preferred entry point; old scripts kept
  as escape hatch
- `ae_file_structure.txt`: new files
- `configs/README.md`: schema, versioning, backfill conventions

**Total: ~6–7 working days solo**, sequenced.

**Dependencies:** Phase 1 → 2 → 3 → 4. Phase 5 piggybacks on whatever is done.

**Risk:** Phase 4 may surface schema edge cases that force minor revisions. If so,
bump `schema_version: 1 → 2` and document migration in `configs/README.md`.

---

## 9. Open Items / Future Work

These are explicitly **not** in this spec but unlocked by it:

- **B-tier (next spec):** MLflow or W&B integration. Per-window metrics
  (RMSRE, Z, anisotropy) auto-logged; UI to compare runs across regions/kernels.
- **C-tier (long-term):** pip-installable `argo-ebus` package, Docker cloud workers,
  region template generator for new EBUS regions, public dashboard.
- **RG-Gibbs integration:** the pending `2026-04-11-rg-gibbs-nonstationary-gpr-design.md`
  spec lands its new GPR engine in `ebus_core/argoebus_gp_physics.py`. Once the
  MLOps foundation exists, the RG-Gibbs config gets a `kernel_type: gibbs` entry
  in `AnalysisConfig.gpr` and inherits all manifest + registry plumbing for free.

---

## 10. References

- Brainstorm conversation: 2026-04-25 session (this spec is its output).
- Project mission: `CLAUDE.md` — Ocean Refugia / Stealth Warming hypothesis.
- Pipeline overview: `ae_file_structure.txt`.
- Pending RG-Gibbs spec: `docs/superpowers/specs/2026-04-11-rg-gibbs-nonstationary-gpr-design.md`.
- Float census results that motivated californiav3:
  `argo_claude_actions/AE_claude_recentactions.md` 2026-04-02 entry.
