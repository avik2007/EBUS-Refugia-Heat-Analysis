"""
Backfill legacy analysis runs into the configs/ schema.

Walks AEResults/aelogs/ dirs and reconstructs an AnalysisConfig YAML per run.
Fields recoverable from the run_id and audit CSV are written verbatim.
Fields that cannot be recovered (noise_val, time_ls_bounds_days, lat_ls_bounds,
lon_ls_bounds) are written as null. Every backfill YAML carries legacy_backfill=true
so the schema validator skips completeness checks on those fields.

WHY THIS EXISTS:
    Pre-manifest runs have no provenance record. Backfill gives every existing
    aelogs dir a matching config YAML so aebus list/show and the collision detector
    can treat them consistently with new runs. The null markers are intentional —
    writing script defaults would misrepresent what was actually used.
"""
import csv
import re
from datetime import date
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Run-id parsing
# ---------------------------------------------------------------------------

# Pattern decomposes the canonical run_id naming scheme used by existing scripts.
# Groups: (region, date_start, date_end, lat_fmt, lon_fmt, time_fmt, d0, d1, suffix)
# Example: california_20150101_20151231_res0_5x0_5_t30_0_d0_100_3dmatern_w45
_RUN_ID_PATTERN = re.compile(
    r"^(.+)_(\d{8})_(\d{8})_res([\d_]+)x([\d_]+)_t([\d_]+)_d(\d+)_(\d+)(.*?)$"
)


def _fmt_to_float(s: str) -> float:
    # Convert a filesystem-safe float string back to float.
    # The naming scheme replaces '.' with '_' so 0.5 becomes "0_5" and 30.0
    # becomes "30_0". Replace only the first underscore that looks like a decimal
    # point (preceded and followed by digits).
    # Input: s — formatted string, e.g. "0_5", "30_0", "10_0"
    # Output: float value, e.g. 0.5, 30.0, 10.0
    return float(s.replace("_", "."))


def _parse_run_id(run_id: str) -> dict:
    # Parse the run_id string into its component config fields.
    # Input: run_id — canonical name of an aelogs dir
    # Output: dict with keys: region, date_start, date_end, lat_step, lon_step,
    #         time_step, depth_range (tuple), run_suffix (str)
    # Raises: ValueError if run_id does not match the expected pattern.
    m = _RUN_ID_PATTERN.match(run_id)
    if not m:
        raise ValueError(f"run_id does not match expected pattern: {run_id!r}")

    region = m.group(1)
    date_start = date(int(m.group(2)[:4]), int(m.group(2)[4:6]), int(m.group(2)[6:8]))
    date_end = date(int(m.group(3)[:4]), int(m.group(3)[4:6]), int(m.group(3)[6:8]))
    lat_step = _fmt_to_float(m.group(4))
    lon_step = _fmt_to_float(m.group(5))
    time_step = _fmt_to_float(m.group(6))
    depth_range = (int(m.group(7)), int(m.group(8)))
    run_suffix = m.group(9)  # includes leading underscore if non-empty

    return {
        "region": region,
        "date_start": date_start,
        "date_end": date_end,
        "lat_step": lat_step,
        "lon_step": lon_step,
        "time_step": time_step,
        "depth_range": depth_range,
        "run_suffix": run_suffix,
    }


# ---------------------------------------------------------------------------
# Suffix parsing for GPR mode, kernel, and window settings
# ---------------------------------------------------------------------------

def _parse_suffix(suffix: str, time_step: float) -> dict:
    # Infer GPR settings from the run_suffix where possible.
    # All inferences are best-effort: fields absent from the suffix fall back
    # to pipeline-default values documented here. When a field is genuinely
    # unrecoverable it is NOT set here — the caller writes null explicitly.
    #
    # Suffix conventions used by existing scripts:
    #   "2d" in suffix → mode 2D, else 3D
    #   "rbf" → kernel_type rbf; anything else → matern0.5 (the project default)
    #   "w{N}" → window_size_days = N (e.g. w45 → 45)
    #   "minbins{N}" → min_bins = N
    #   "t{X}s{N}" → step_size_days = N (e.g. t1s10 → step=10)
    #   absent → step_size_days = time_step (user-confirmed safe default)
    #
    # Input: suffix — raw run_suffix string (may be empty); time_step — days per bin
    # Output: dict with keys mode, kernel_type, window_size_days, min_bins, step_size_days
    sfx = suffix.lower()

    # mode: explicit "2d" tag overrides default of 3D
    mode = "2D" if "2d" in sfx else "3D"

    # kernel_type: only rbf is distinguishable from the suffix; all others are matern0.5
    kernel_type = "rbf" if "rbf" in sfx else "matern0.5"

    # window_size_days: w{N} tag; default 45 (the canonical production window)
    m_win = re.search(r"w(\d+)", sfx)
    window_size_days = int(m_win.group(1)) if m_win else 45

    # min_bins: minbins{N} tag; default 10 (pipeline default)
    m_bins = re.search(r"minbins(\d+)", sfx)
    min_bins = int(m_bins.group(1)) if m_bins else 10

    # step_size_days: t{X}s{N} tag (experiment notation); default = time_step
    # Using time_step ensures step_size_days >= time_step (bin-aliasing guard).
    m_step = re.search(r"t\d+s(\d+)", sfx)
    step_size_days = int(m_step.group(1)) if m_step else int(time_step)

    return {
        "mode": mode,
        "kernel_type": kernel_type,
        "window_size_days": window_size_days,
        "min_bins": min_bins,
        "step_size_days": step_size_days,
    }


# ---------------------------------------------------------------------------
# Audit CSV reader
# ---------------------------------------------------------------------------

def _read_noise_vals(aelog_dir: Path, run_id: str) -> Optional[List[float]]:
    # Read all per-window noise_val values from the run's audit CSV.
    # These are the OPTIMIZED noise values per sliding window, not the
    # regularization input — recorded here for forensic review only.
    # Returns None if the CSV is missing or has no noise_val column.
    # Input: aelog_dir — Path to the aelogs run directory
    #        run_id — run identifier used to find the CSV by naming convention
    # Output: list of floats (one per window row), or None
    csv_path = aelog_dir / f"audit_{run_id}.csv"
    if not csv_path.exists():
        return None
    values = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "noise_val" not in reader.fieldnames:
            return None
        for row in reader:
            try:
                values.append(float(row["noise_val"]))
            except (ValueError, KeyError):
                pass
    return values if values else None


# ---------------------------------------------------------------------------
# S3 path inference
# ---------------------------------------------------------------------------

def _infer_s3_path(region: str, date_start: date, date_end: date,
                   lat_step: float, lon_step: float, time_step: float,
                   depth_range: tuple) -> str:
    # Construct the expected S3 parquet path for the ingestion run that fed
    # this analysis. The ingestion run_id omits the analysis suffix.
    # Input: config fields (region, dates, grid, depth)
    # Output: s3:// URI string following the existing naming convention
    def fmt(v: float) -> str:
        return str(v).replace(".", "_")

    d0, d1 = depth_range
    ds = date_start.strftime("%Y%m%d")
    de = date_end.strftime("%Y%m%d")
    ingest_id = (
        f"{region}_{ds}_{de}"
        f"_res{fmt(lat_step)}x{fmt(lon_step)}"
        f"_t{fmt(time_step)}_d{d0}_{d1}"
    )
    return f"s3://argo-ebus-project-data-abm/{ingest_id}.parquet"


# ---------------------------------------------------------------------------
# Core backfill function
# ---------------------------------------------------------------------------

def backfill_configs(
    aelogs_root: Path,
    configs_root: Path,
    today: str = "2026-04-30",
) -> List[Path]:
    # Enumerate existing aelogs dirs and write one AnalysisConfig YAML per dir.
    # Fields recovered from run_id and audit CSV are written verbatim.
    # Fields unrecoverable from pre-manifest runs are written as null:
    #   - noise_val (regularization input, not the per-window output)
    #   - time_ls_bounds_days, lat_ls_bounds, lon_ls_bounds (GPR search bounds)
    # All backfill YAMLs carry legacy_backfill=true so the schema validator
    # skips completeness checks on the null fields.
    #
    # Input: aelogs_root — Path to AEResults/aelogs/ directory
    #        configs_root — Path to configs/ root (will be created if absent)
    #        today — ISO date string used in the description field
    # Output: list of Paths to written YAML files
    aelogs_root = Path(aelogs_root)
    configs_root = Path(configs_root)

    written = []
    for aelog_dir in sorted(aelogs_root.iterdir()):
        if not aelog_dir.is_dir():
            continue

        run_id = aelog_dir.name
        try:
            parsed = _parse_run_id(run_id)
        except ValueError:
            # Dir doesn't match expected naming — skip silently
            continue

        region = parsed["region"]
        date_start: date = parsed["date_start"]
        date_end: date = parsed["date_end"]
        lat_step: float = parsed["lat_step"]
        lon_step: float = parsed["lon_step"]
        time_step: float = parsed["time_step"]
        depth_range: tuple = parsed["depth_range"]
        run_suffix: str = parsed["run_suffix"]

        gpr_recoverable = _parse_suffix(run_suffix, time_step)

        # Determine which GPR fields were explicitly recovered from the suffix
        # vs which fell back to pipeline defaults. This feeds backfill_metadata.
        sfx_lower = run_suffix.lower()
        gpr_recovered = []
        gpr_assumed = []

        # mode: explicit only if "2d" is present (otherwise 3D is assumed)
        if "2d" in sfx_lower:
            gpr_recovered.append("gpr.mode")
        else:
            gpr_assumed.append("gpr.mode")

        # kernel_type: explicit only if "rbf" is present
        if "rbf" in sfx_lower:
            gpr_recovered.append("gpr.kernel_type")
        else:
            gpr_assumed.append("gpr.kernel_type")

        # window_size_days: explicit only if w{N} pattern present
        if re.search(r"w\d+", sfx_lower):
            gpr_recovered.append("gpr.window_size_days")
        else:
            gpr_assumed.append("gpr.window_size_days")

        # min_bins: explicit only if minbins{N} pattern present
        if re.search(r"minbins\d+", sfx_lower):
            gpr_recovered.append("gpr.min_bins")
        else:
            gpr_assumed.append("gpr.min_bins")

        # step_size_days: explicit only if t{X}s{N} pattern present
        if re.search(r"t\d+s\d+", sfx_lower):
            gpr_recovered.append("gpr.step_size_days")
        else:
            gpr_assumed.append("gpr.step_size_days")

        noise_vals = _read_noise_vals(aelog_dir, run_id)

        # noise_vals_audit: recovered if audit CSV existed and had noise_val column
        if noise_vals is not None:
            gpr_recovered.append("gpr.noise_vals_audit")

        # All run_id-derived fields are always recovered
        run_id_recovered = [
            "region", "date_start", "date_end",
            "lat_step", "lon_step", "time_step", "depth_range",
        ]
        all_recovered = run_id_recovered + gpr_recovered
        s3_path = _infer_s3_path(
            region, date_start, date_end, lat_step, lon_step, time_step, depth_range
        )

        yaml_dict = {
            "schema_version": 1,
            "config_kind": "analysis",
            "legacy_backfill": True,
            "input": {"source": "s3", "s3_path": s3_path},
            "region": region,
            "date_start": date_start.isoformat(),
            "date_end": date_end.isoformat(),
            "lat_step": lat_step,
            "lon_step": lon_step,
            "time_step": time_step,
            "depth_range": list(depth_range),
            "gpr": {
                "mode": gpr_recoverable["mode"],
                "kernel_type": gpr_recoverable["kernel_type"],
                "window_size_days": gpr_recoverable["window_size_days"],
                "step_size_days": gpr_recoverable["step_size_days"],
                "min_bins": gpr_recoverable["min_bins"],
                # Regularization input — unrecoverable from pre-manifest run
                "noise_val": None,
                # Optimiser search bounds — unrecoverable
                "time_ls_bounds_days": None,
                "lat_ls_bounds": None,
                "lon_ls_bounds": None,
                # Per-window empirical noise from audit CSV (output, not input)
                "noise_vals_audit": noise_vals,
                "run_suffix": run_suffix,
            },
            "description": (
                f"backfilled from {run_id} on {today}; "
                "noise_val / time_ls_bounds_days / lat_ls_bounds / lon_ls_bounds "
                "unrecoverable from pre-manifest run, treat as legacy"
            ),
            "backfill_metadata": {
                "recovered_fields": all_recovered,
                "assumed_fields": gpr_assumed,
            },
        }

        # Write to configs/{region}/{run_id}.yaml
        out_dir = configs_root / region
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.yaml"
        with out_path.open("w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

        written.append(out_path)

    return written
