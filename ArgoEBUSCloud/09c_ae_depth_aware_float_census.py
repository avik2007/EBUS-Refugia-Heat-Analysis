"""
09c_ae_depth_aware_float_census.py

PURPOSE:
    Depth-aware Argo float census for the California domain (lat [25,50], lon [-140,-110]).
    Builds and plots float coverage separately for three scientific layers:
        Skin       (  0– 100 dbar): atmosphere-forced surface layer
        Source     (150– 400 dbar): Ekman upwelling source water, "stealth heat" reservoir
        Background (500–1000 dbar): deep ocean baseline

    Motivation: the 2015 Source Layer GPR run (californiav2, t10_0) showed severe regression
    (RMSRE up to 22%, pass rate 24%). The tighter californiav2 domain (lon [-130,-115]) is
    suspected to clip float trajectories at depth, leaving too few observations for stable
    kriging. This script makes that clipping effect visible by overlaying the californiav2
    domain boundary ("CCS Analysis Bounds") on every figure.

OUTPUT (all in AEResults/aeplots/float_census_depth_aware/):
    Per-year all-depths PNGs:    float_census_alldepths_{year}.png        (26 files)
    Per-year skin PNGs:          float_census_skin_{year}.png             (26 files)
    Per-year source PNGs:        float_census_source_{year}.png           (26 files)
    Per-year background PNGs:    float_census_background_{year}.png       (26 files)
    Per-layer mean PNGs:         float_census_{layer}_mean.png            ( 3 files)
    CSVs:                        float_census_depth_aware_{layer}_1999_2025.csv (4 files)

USAGE:
    conda run -n ebus-cloud-env python ArgoEBUSCloud/09c_ae_depth_aware_float_census.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")    # Headless — no display required
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Add the ArgoEBUSCloud package root to sys.path so ebus_core is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ebus_core.ae_utils import (
    get_ebus_registry,
    get_project_paths,
    ensure_ae_dirs,
    get_float_history,
    get_float_history_by_layer,
)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# The three scientific depth layers plus an all-depths pass for baseline comparison.
# Layer name -> (pres_min_dbar, pres_max_dbar, display_label)
# "alldepths" uses get_float_history() (no pressure filter) and is included for
# visual parity with script 09 — lets us see whether deep-layer sparsity
# is real or a domain artifact.
LAYERS = {
    "alldepths":  (None, None,  "All Depths"),
    "skin":       (0,    100,   "Skin (0–100m)"),
    "source":     (150,  400,   "Source (150–400m)"),
    "background": (500,  1000,  "Background (500–1000m)"),
}

# ERDDAP fetch chunks — same 5-year splits as script 09, within ERDDAP row limits
FETCH_CHUNKS = [
    ("1999-01-01", "2003-12-31"),
    ("2004-01-01", "2008-12-31"),
    ("2009-01-01", "2013-12-31"),
    ("2014-01-01", "2018-12-31"),
    ("2019-01-01", "2023-12-31"),
    ("2024-01-01", "2025-12-31"),
]

# Broad california domain: all figures use this extent so year-to-year
# and layer-to-layer comparisons are geometrically consistent
LON_MIN, LON_MAX = -140.0, -110.0
LAT_MIN, LAT_MAX =   25.0,   50.0

# Uniform color scale across all layers and years.
# vmax=8: source/background layers are sparser than all-depths so a lower
# ceiling distinguishes the sparse years that would wash out at vmax=15.
VMIN, VMAX = 0, 8


# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

def build_output_dir():
    """
    Constructs and returns the output subfolder path:
        AEResults/aeplots/float_census_depth_aware/

    ensure_ae_dirs() creates the three top-level AEResults subdirectories
    only; the census subfolder is created explicitly here.
    """
    ensure_ae_dirs()
    paths = get_project_paths()
    subfolder = os.path.join(paths["plots"], "float_census_depth_aware")
    os.makedirs(subfolder, exist_ok=True)
    print(f"[census] Output directory: {subfolder}")
    return subfolder


def get_ccs_bounds():
    """
    Reads californiav2 lat/lon bounds from the registry and returns them
    as a dict for use by the plot overlay.

    Returns:
        {"lat": [lat_min, lat_max], "lon": [lon_min, lon_max]}

    Why registry instead of hardcoding? If californiav2 bounds change in the
    future, the overlay automatically stays in sync without touching this script.
    The californiav2 domain (lon [-130,-115], lat [30,45]) is the tighter GPR
    domain whose clipping effect this script is designed to visualize.
    """
    reg = get_ebus_registry()["californiav2"]
    return {"lat": reg["lat"], "lon": reg["lon"]}


# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def fetch_layer_data(layer_name, pres_min, pres_max):
    # Fetch raw per-dive Argo float positions for one depth layer across all
    # years 1999-2025, using the broad "california" domain.
    #
    # For the "alldepths" layer (pres_min is None), calls get_float_history()
    # with no pressure filter — identical behavior to script 09. For all other
    # layers, calls get_float_history_by_layer() which restricts to dives that
    # had at least one measurement in [pres_min, pres_max] dbar.
    #
    # Uses the same 5-year FETCH_CHUNKS as script 09 to stay within ERDDAP row
    # limits per request (~14k rows/chunk for all-depths; fewer for deep layers).
    # Chunk failures are caught individually — a single ERDDAP timeout does not
    # abort the whole run. Missing chunks will appear as zero-float years in the
    # output, clearly distinguishable from sparse-but-real coverage.
    #
    # Inputs:
    #   layer_name - Short string key (e.g., "source") used in log messages.
    #   pres_min   - Lower pressure bound in dbar, or None for no filter.
    #   pres_max   - Upper pressure bound in dbar, or None for no filter.
    #
    # Returns a single concatenated DataFrame with columns:
    #   platform_number, lat, lon, time, time_days, year (int)
    frames = []
    for start_date, end_date in FETCH_CHUNKS:
        print(f"[census/{layer_name}] Fetching {start_date} → {end_date} ...", flush=True)
        try:
            if pres_min is None:
                # All-depths: no pressure filter, identical to script 09
                chunk = get_float_history(
                    region="california",
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                chunk = get_float_history_by_layer(
                    region="california",
                    pres_min=pres_min,
                    pres_max=pres_max,
                    start_date=start_date,
                    end_date=end_date,
                )
            print(f"[census/{layer_name}]   Got {len(chunk):,} dives.")
            frames.append(chunk)
        except Exception as exc:
            print(f"[census/{layer_name}]   WARNING: chunk failed — {exc}")

    if not frames:
        raise RuntimeError(f"[census/{layer_name}] All ERDDAP chunks failed. Cannot continue.")

    df = pd.concat(frames, ignore_index=True)
    df["year"] = df["time"].dt.year
    print(f"[census/{layer_name}] Total: {len(df):,} dives, {df['year'].nunique()} years.")
    return df


# ---------------------------------------------------------------------------
# CENSUS BUILDING
# ---------------------------------------------------------------------------

def build_census(df):
    # Bins dive positions onto a 5x5 degree grid and counts unique float WMO IDs
    # (platform_number) per (year, lat_bin, lon_bin) cell.
    #
    # Bin centers are placed at the midpoint of each 5-degree cell:
    #   lat_bin = floor(lat / 5) * 5 + 2.5   e.g. lat=32.1 → bin 32.5
    #   lon_bin = floor(lon / 5) * 5 + 2.5   e.g. lon=-122.7 → bin -122.5
    #
    # Why unique floats, not dive count?
    #   One float making 30 dives in a cell represents one independent spatial
    #   sensor for the Gaussian Process. Using unique floats gives a true picture
    #   of independent GP support points, not ping volume.
    #
    # Inputs:
    #   df - DataFrame from fetch_layer_data() with columns:
    #        platform_number, lat, lon, time, time_days, year
    #
    # Returns DataFrame with columns: year, lat_bin, lon_bin, n_floats
    df = df.copy()
    df["lat_bin"] = (np.floor(df["lat"] / 5.0) * 5.0) + 2.5
    df["lon_bin"] = (np.floor(df["lon"] / 5.0) * 5.0) + 2.5

    census = (
        df.groupby(["year", "lat_bin", "lon_bin"])["platform_number"]
        .nunique()
        .reset_index()
        .rename(columns={"platform_number": "n_floats"})
    )
    print(f"[census] {len(census):,} (year, cell) records built.")
    return census


# ---------------------------------------------------------------------------
# CSV SAVING
# ---------------------------------------------------------------------------

def save_census_csv(census, layer_name, out_dir):
    # Saves the census DataFrame for one layer to CSV.
    #
    # Filename pattern: float_census_depth_aware_{layer_name}_1999_2025.csv
    #
    # This is the primary archival output — the PNGs are derived from these CSVs
    # and can be regenerated. The CSVs are the ground truth and are suitable for
    # downstream analysis (e.g., 09b-style persistence tables).
    #
    # Inputs:
    #   census     - DataFrame with columns: year, lat_bin, lon_bin, n_floats
    #   layer_name - Short key (e.g., "source") used in the filename
    #   out_dir    - Absolute path to the output subfolder
    csv_name = f"float_census_depth_aware_{layer_name}_1999_2025.csv"
    csv_path = os.path.join(out_dir, csv_name)
    census.to_csv(csv_path, index=False)
    print(f"[census/{layer_name}] CSV saved → {csv_path}")
    return csv_path
