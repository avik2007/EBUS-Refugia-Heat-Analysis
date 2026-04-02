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
