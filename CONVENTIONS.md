# CONVENTIONS.md — ArgoEBUSAnalysis

Pre-digested project context for AI coding assistants. Read this before editing.

## What this project does

Tests the "Ocean Refugia / Stealth Warming" hypothesis: subsurface warming in the
California Current System, carried by the California Undercurrent, that has not yet
surfaced. Argo float profiles are converted to Ocean Heat Content (TEOS-10 / GSW),
then Gaussian Process Regression (kriging) fills spatial gaps in three depth layers.
Signal of interest: the Source Layer warming faster than the Background Layer.

Depth layers: Skin `0–100m`, Source `150–400m`, Background `500–1000m`.

## Directory structure

- `ArgoEBUSCloud/` — the production pipeline.
  - `NN_ae_*.py` — numbered pipeline stages (00–10): ingestion, kriging, plotting,
    diagnostics, backfill. Canonical: `02_ae_cloud_run.py`, `05_ae_update_tomatern0.5.py`,
    `07_ae_deeper_layers.py` — never delete/refactor. `03`, `04`, `05_ae_rmsre_optimization.py`
    are superseded but kept as reference. No script has an in-code DEPRECATED marker.
  - `aebus_cli.py` — config-driven CLI (`validate` / `analyze` / `ingest` / `list` / `show`).
  - `ebus_core/` — importable core library:
    - `ae_utils.py` — region registry (lat/lon bounds, time window, S3 bucket).
    - `argoebus_thermodynamics.py` — T/S/P → heat density → OHC (J/m²).
    - `argoebus_gp_physics.py` — GPR / kriging engine, CV hyperparameter tuning.
    - `argoebus_plotting.py` — Cartopy maps and physics-history plots.
    - `config_schema.py` — Pydantic configs (`schema_version: 1`).
    - `manifest.py` / `runner.py` / `backfill.py` — MLOps layer (hash, provenance, dispatch).
  - `test_mlops_foundation.py` — pytest suite (~66 tests: config/manifest/runner/CLI).
    `test_pipeline.py` — standalone Dask/ERDDAP smoke script, NOT pytest.
- `configs/<region>/` — one YAML per run. For `california/` (15) and `californiav2/` (3) the
  filename stem equals the canonical `run_id`; `californiav3/` (12: `*_gibbs`, `*_ingest`) uses
  short ad-hoc names. See `configs/README.md`.
- `AEResults/` — outputs (`aeplots/`, `aedata/`, `aelogs/`, `run_registry.jsonl`). Lives at repo root.
- `*.ipynb` (root) — exploratory notebooks, not primary edit targets.
- `ArgoGPR.py`, `ArgoHeatContentDataCollater.py` — legacy root scripts, superseded by `ebus_core/`.
- `argo_claude_actions/`, `argo_gemini_actions/`, `argo_qwen_actions/` — per-agent
  session logs, todos, lessons. `argo_qwen_actions/` also holds dated task briefs
  handed to Qwen (see **Multi-agent task handoff** below).

## Conventions

- **Env**: always `conda run -n ebus-cloud-env python <script>`. Base env lacks packages.
  Pipeline env = `ArgoEBUSCloud/ocean_cloud.yml` (Python 3.10). Root `ocean_env_clean.yml`
  (`oceanography`, 3.11) is a different, unrelated env.
- **run_id**: `california_20150101_20151231_res0_5x0_5_t30_0_d0_100` — region, dates,
  `res<lat>x<lon>`, `t<time_step>`, `d<d0>_<d1>`. `d0_100`=Skin, `d150_400`=Source, `d500_1000`=Background.
- **Common signature**: the reusable analysis/diagnostic functions (`run_diagnostic_inspection`,
  `diagnose_density`) take all inputs as params: `(region, lat_step, lon_step, time_step, depth_range)`.
  Individual driver scripts (`06`, `07`, `09*`, `10`) still define their own module-level config constants.
- **Preferred workflow**: create/copy a YAML in `configs/<region>/`, then
  `aebus_cli.py validate` → `analyze` / `ingest`. Direct script calls are an escape hatch only.
- **Comments**: verbose header comment on every function you add/modify (what, why, input
  physical meaning, output). Never add comments/docstrings/annotations to code you didn't change.
- **Style**: surgical changes only. Minimum lines. No reformatting adjacent code.

## Multi-agent task handoff

Three assistants share this repo: **Claude** and **Antigravity** plan/design/review;
**Qwen** (run via aider) is the implementation executor. Work is passed to Qwen as
a file, not a chat message.

- **Writing a brief (Claude / Antigravity):** drop a Markdown file in
  `argo_qwen_actions/` named `YYYY-MM-DD_HHMM_<slug>.md` (24h local time, e.g.
  `2026-08-30_2236_thermo-tests.md`). Date-first so the directory sorts
  chronologically. Start the file with a header block:
  ```
  - **Author:** Claude | Antigravity
  - **Created:** YYYY-MM-DD HH:MM
  - **Status:** OPEN
  - **Target files:** <files Qwen may create/modify>
  - **Run to verify:** <exact command + expected result>
  ```
  Then: goal, behavior contract, required fixtures/isolation, enumerated test or
  implementation cases, acceptance criteria. Be explicit — Qwen executes literally.
- **Executing a brief (Qwen):** work the newest file whose `Status:` is `OPEN`.
  Touch only the files the brief names. On finish: set `Status:` to `DONE` (or
  `BLOCKED — <reason>`), append a dated entry to
  `argo_qwen_actions/AE_qwen_recentactions.md`, drop the item from
  `AE_qwen_todo.md`, and record any correction in `AE_qwen_lessons.md`.
- **Ambiguity:** Qwen does not guess. Add a `## Questions` section to the brief,
  set `Status: BLOCKED`, and hand back.

## Gotchas

- `AEResults/` is at the repo root, not inside `ArgoEBUSCloud/`. Paths must traverse up:
  `os.path.join(base_dir, "..", "AEResults", ...)`.
- Anisotropy ratio = `Lat_Scale / Lon_Scale`. Should increase with depth. Target RMSRE < 5%.
- Configs with `legacy_backfill: true` have `null` GPR fields on purpose — do NOT fill them
  with defaults; it corrupts the collision detector's hash identity.
- `schema_version` is in the config hash. Changing schema fields requires a version bump +
  migration note in `configs/README.md`.
