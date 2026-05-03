"""
Runner module for the ArgoEBUS MLOps foundation.

Wraps the existing pipeline functions (script 02 ingestion, scripts 05/07
analysis) with a uniform interface that:
- accepts a validated pydantic config
- derives the canonical run_id (matching existing AEResults/aelogs/ naming)
- runs the collision detector
- captures wall-clock duration
- writes a manifest.json + appends to the JSONL registry
"""
import datetime as _dt
import shutil
import time as _time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ebus_core.ae_utils import fmt_dec
from ebus_core.config_schema import AnalysisConfig, IngestionConfig
from ebus_core.manifest import (
    ManifestCollisionError, append_registry, canonical_config_dict,
    capture_code, capture_env, capture_host, check_collision, config_hash,
    write_manifest,
)


def derive_run_id(cfg: Union[IngestionConfig, AnalysisConfig]) -> str:
    """
    Reproduce the canonical run_id naming used by existing scripts.

    Output format:
        {region}_{date_start_compact}_{date_end_compact}
        _res{lat}x{lon}_t{time_step}_d{depth0}_{depth1}{run_suffix}

    Example: california_20150101_20151231_res0_5x0_5_t30_0_d150_400_3dmatern_w45

    WHY THIS FUNCTION EXISTS — BACKWARD COMPATIBILITY:
        Existing AEResults/aelogs/ directory trees, CV pickle filenames, and
        the run registry (runs.jsonl) all use this exact naming scheme. Any
        deviation would break collision detection, manifest lookups, and
        manual result inspection. This function is the single authoritative
        source of truth for the naming rule.

    INPUTS:
        cfg — a fully-validated pipeline config (IngestionConfig or
              AnalysisConfig). All fields have already passed pydantic
              validation; no further sanitisation is needed here.
              - cfg.region: string region tag, e.g. "california"
              - cfg.date_start/date_end: datetime objects, formatted YYYYMMDD
              - cfg.lat_step/lon_step/time_step: floats; the decimal point is
                replaced with underscore via fmt_dec (filesystem-safe)
              - cfg.depth_range: Tuple[int, int] bounding the depth layer
              - cfg.gpr.run_suffix (AnalysisConfig only): kernel/window tag
                such as "_3dmatern_w45". The leading underscore is part of
                the tag and is owned by the config — do NOT add one here.

    OUTPUT:
        A string used as:
          - the run_id field written into manifest.json
          - the collision-detection key (duplicate runs share the same id)
          - the output directory / log-file name under AEResults/aelogs/

    Where:
      - dates are YYYYMMDD (no separators)
      - lat/lon/time floats use underscore for the decimal point
        (matches existing convention: 0.5 -> "0_5", 10.0 -> "10_0")
      - depth bounds are plain integers and need no decimal conversion
      - run_suffix only applies to AnalysisConfig and is appended verbatim
    """
    region = cfg.region
    ds = cfg.date_start.strftime("%Y%m%d")
    de = cfg.date_end.strftime("%Y%m%d")
    lat = fmt_dec(cfg.lat_step)
    lon = fmt_dec(cfg.lon_step)
    t = fmt_dec(cfg.time_step)
    d0, d1 = cfg.depth_range
    # depth_range is Tuple[int, int] — integers format cleanly as "150", "400"
    # without needing decimal substitution, unlike lat/lon/time which are
    # floats and would produce "150.0" without _fmt_dec treatment.
    base = f"{region}_{ds}_{de}_res{lat}x{lon}_t{t}_d{d0}_{d1}"
    if isinstance(cfg, AnalysisConfig):
        return base + cfg.gpr.run_suffix
    return base



def build_manifest(
    cfg: Union[IngestionConfig, AnalysisConfig],
    outputs: Dict[str, Any],
    inputs_extra: Dict[str, Any],
    duration_sec: float,
    conda_list_dest: Optional[Path],
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Assemble a complete manifest dict from a validated config + execution metadata.

    WHY: Pure function — no IO besides read-only subprocess calls in capture_*.
    Separating assembly from writing lets tests verify manifest structure without
    touching the filesystem.

    INPUTS:
      cfg            — validated pipeline config (IngestionConfig or AnalysisConfig)
      outputs        — dict of output paths produced by the run (caller fills this)
      inputs_extra   — caller-supplied provenance: parquet_etag, erddap lineage, etc.
      duration_sec   — wall-clock seconds for the pipeline run
      conda_list_dest— where to save conda list output (None = skip)
      cwd            — working directory for git capture (None = cwd of process)

    OUTPUT: dict with these top-level keys:
      schema_version, kind, run_id, config_hash, created_at, duration_sec,
      config, code, env, inputs, outputs, host
      Ready to pass to write_manifest().
    """
    kind = "ingestion" if isinstance(cfg, IngestionConfig) else "analysis"

    # Build inputs block. Analysis carries the S3 parquet pointer + any caller
    # extras (etag, erddap lineage). Ingestion has no upstream S3 input.
    if isinstance(cfg, AnalysisConfig):
        # inputs_extra first so config-derived fields (source, s3_path, ingestion_run_id)
        # always take precedence — callers cannot accidentally overwrite provenance fields.
        inputs: Dict[str, Any] = {
            **inputs_extra,
            "source": cfg.input.source,
            "s3_path": cfg.input.s3_path,
            "ingestion_run_id": cfg.input.ingestion_run_id,
        }
    else:
        inputs = {**inputs_extra, "source": "erddap"}

    env_block = capture_env(
        conda_env_name="ebus-cloud-env",
        conda_list_dest=conda_list_dest,
    )
    # §A.4: teos10_convention is a convention family tag ("gsw-3.x"), not a pinned
    # version. GSW 3.x changed unit conventions and function signatures vs pre-3.0,
    # so runs with different major versions are not directly comparable. This is
    # deliberately NOT read from the installed gsw version — it names the API family.
    env_block["teos10_convention"] = "gsw-3.x"

    return {
        "schema_version": 1,
        "kind": kind,
        "run_id": derive_run_id(cfg),
        "config_hash": config_hash(cfg),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "duration_sec": duration_sec,
        "config": canonical_config_dict(cfg),
        "code": capture_code(cwd=cwd),
        "env": env_block,
        "inputs": inputs,
        "outputs": outputs,
        "host": capture_host(),
    }


def _call_run_diagnostic_inspection(**kwargs):
    # Thin shim: defers import of the GPR script so runner.py stays cheap to import.
    # Tests monkeypatch this whole function instead of the real script.
    import importlib.util
    import sys
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "05_ae_update_tomatern0.5.py"
    spec = importlib.util.spec_from_file_location("script_05", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_05"] = mod
    spec.loader.exec_module(mod)
    return mod.run_diagnostic_inspection(**kwargs)


def run_analysis(
    cfg: AnalysisConfig,
    registry_path: Optional[Path] = None,
    force_overwrite: bool = False,
) -> Dict[str, Any]:
    # Execute one GPR analysis run end-to-end with manifest + collision detection.
    # Steps: derive run_id → collision check → dispatch → build manifest → write → registry.
    run_id = derive_run_id(cfg)
    aelogs_dir = Path(cfg.outputs.aelogs_dir) / run_id
    manifest_path = aelogs_dir / "manifest.json"
    cfg_hash = config_hash(cfg)

    verdict = check_collision(manifest_path, new_hash=cfg_hash)
    if force_overwrite and aelogs_dir.exists():
        shutil.rmtree(aelogs_dir)

    aelogs_dir.mkdir(parents=True, exist_ok=True)

    # Pass ALL GPR config fields; the real shim filters to accepted params.
    # Tests verify kernel_type + window_size_days reach the shim.
    dispatch_kwargs = {
        "region": cfg.region,
        "lat_step": cfg.lat_step,
        "lon_step": cfg.lon_step,
        "time_step": cfg.time_step,
        "depth_range": cfg.depth_range,
        "mode": cfg.gpr.mode,
        "kernel_type": cfg.gpr.kernel_type,
        "window_size_days": cfg.gpr.window_size_days,
        "step_size_days": cfg.gpr.step_size_days,
        "min_bins": cfg.gpr.min_bins,
        "noise_val": cfg.gpr.noise_val,
        "time_ls_bounds_days": cfg.gpr.time_ls_bounds_days,
        "run_suffix": cfg.gpr.run_suffix,
    }
    # Derive the single spatial_ls_upper_bound the GPR function expects from the
    # per-axis bounds stored in config. The physics engine takes one scalar upper
    # bound for the spatial length-scale search; we use the looser (larger) of the
    # two axes so neither lat nor lon is over-constrained. Omitting the key when
    # bounds are null lets the script default (10.0) apply — correct for legacy
    # configs that predate split-anisotropy bounds.
    if cfg.gpr.lat_ls_bounds is not None and cfg.gpr.lon_ls_bounds is not None:
        dispatch_kwargs["spatial_ls_upper_bound"] = max(
            cfg.gpr.lat_ls_bounds[1], cfg.gpr.lon_ls_bounds[1]
        )

    t0 = _time.time()
    dispatch_result = _call_run_diagnostic_inspection(**dispatch_kwargs)
    duration = _time.time() - t0

    audit_csv = (
        dispatch_result.get("audit_csv")
        if isinstance(dispatch_result, dict)
        else str(aelogs_dir / f"audit_{run_id}.csv")
    )

    # Determine run completeness: "finalized" if the audit CSV landed on disk,
    # "incomplete" if the dispatch returned but outputs are missing. The latter
    # can happen when a Dask job crashes after returning a partial result dict.
    # Writing "incomplete" to the registry prevents silent ghost-success entries.
    status = (
        "finalized"
        if audit_csv and Path(audit_csv).exists()
        else "incomplete"
    )

    snapshots_dir = str(Path(cfg.outputs.aeplots_dir) / f"snapshot_{run_id}")

    manifest = build_manifest(
        cfg,
        outputs={
            "aelogs_dir": str(aelogs_dir),
            "audit_csv": audit_csv,
            "snapshots_dir": snapshots_dir,
        },
        inputs_extra={},
        duration_sec=duration,
        conda_list_dest=aelogs_dir / "conda_list.txt",
    )
    write_manifest(manifest, manifest_path)
    if registry_path is not None:
        append_registry(manifest, registry_path, manifest_path, status=status)

    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "duration_sec": duration,
        "verdict": verdict,
    }


# Default manifest output root for ingestion runs. Module-level so tests can
# monkeypatch via symbol name without touching the filesystem.
INGESTION_AELOGS_DIR = Path("AEResults/aelogs/ingestion")


def _call_run_ingestion(**kwargs):
    # Thin shim around script 02's run_ingestion_pipeline seam.
    # Tests monkeypatch this entire function; production loads the real script.
    import importlib.util
    import sys
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "02_ae_cloud_run.py"
    spec = importlib.util.spec_from_file_location("script_02", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_02"] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "run_ingestion_pipeline"):
        return mod.run_ingestion_pipeline(**kwargs)
    raise RuntimeError(
        "02_ae_cloud_run.py must expose run_ingestion_pipeline(**kwargs); "
        "this is the MLOps runner seam."
    )


def run_ingestion(
    cfg: IngestionConfig,
    registry_path: Optional[Path] = None,
    force_overwrite: bool = False,
) -> Dict[str, Any]:
    # Execute one cloud-ingestion run end-to-end with manifest + collision detection.
    # Mirrors run_analysis() but writes under INGESTION_AELOGS_DIR.
    run_id = derive_run_id(cfg)
    aelogs_dir = INGESTION_AELOGS_DIR / run_id
    manifest_path = aelogs_dir / "manifest.json"
    cfg_hash = config_hash(cfg)

    verdict = check_collision(manifest_path, new_hash=cfg_hash)
    if force_overwrite and aelogs_dir.exists():
        shutil.rmtree(aelogs_dir)
    aelogs_dir.mkdir(parents=True, exist_ok=True)

    dispatch_kwargs = {
        "region": cfg.region,
        "lat_step": cfg.lat_step,
        "lon_step": cfg.lon_step,
        "time_step": cfg.time_step,
        "depth_range": cfg.depth_range,
        "date_start": cfg.date_start,
        "date_end": cfg.date_end,
        "n_workers": cfg.cloud.n_workers,
        "worker_region": cfg.cloud.worker_region,
        "s3_bucket": cfg.s3.bucket,
    }

    t0 = _time.time()
    result = _call_run_ingestion(**dispatch_kwargs)
    duration = _time.time() - t0

    inputs_extra = {
        "parquet_etag": result.get("etag"),
        "parquet_size_bytes": result.get("size_bytes"),
    }
    outputs = {
        "aelogs_dir": str(aelogs_dir),
        "s3_path": result.get("s3_path"),
    }

    manifest = build_manifest(
        cfg, outputs=outputs, inputs_extra=inputs_extra,
        duration_sec=duration,
        conda_list_dest=aelogs_dir / "conda_list.txt",
    )
    write_manifest(manifest, manifest_path)
    if registry_path is not None:
        append_registry(manifest, registry_path, manifest_path)

    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "s3_path": result.get("s3_path"),
        "duration_sec": duration,
        "verdict": verdict,
    }
