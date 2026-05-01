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
from typing import Union

from ebus_core.config_schema import AnalysisConfig, IngestionConfig


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
                replaced with underscore via _fmt_dec (filesystem-safe)
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
    lat = _fmt_dec(cfg.lat_step)
    lon = _fmt_dec(cfg.lon_step)
    t = _fmt_dec(cfg.time_step)
    d0, d1 = cfg.depth_range
    # depth_range is Tuple[int, int] — integers format cleanly as "150", "400"
    # without needing decimal substitution, unlike lat/lon/time which are
    # floats and would produce "150.0" without _fmt_dec treatment.
    base = f"{region}_{ds}_{de}_res{lat}x{lon}_t{t}_d{d0}_{d1}"
    if isinstance(cfg, AnalysisConfig):
        return base + cfg.gpr.run_suffix
    return base


def _fmt_dec(x: float) -> str:
    """
    Convert a float to a filesystem-safe string by replacing '.' with '_'.

    Examples: 0.5 -> '0_5', 10.0 -> '10_0', 0.25 -> '0_25'

    WHY UNDERSCORE SUBSTITUTION:
        Dots in directory or filename components confuse shell glob patterns
        and some path-parsing utilities. The existing pipeline scripts (02, 05,
        07) have always used this convention, so all AEResults/ paths and
        run_id strings embed underscored floats. This helper centralises the
        rule so every caller stays consistent with that legacy convention.
    """
    s = f"{x:g}"  # canonical short form: '0.5', '10', '0.25'
    if "." not in s:
        s = s + ".0"
    return s.replace(".", "_")
