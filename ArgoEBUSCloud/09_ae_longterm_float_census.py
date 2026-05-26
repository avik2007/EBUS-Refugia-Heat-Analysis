"""
09_ae_longterm_float_census.py

PURPOSE:
    Build an empirical map of Argo float data density across the broad California
    domain (lat [25,50], lon [-140,-110]) for every year from 1999 through 2024.
    This is a diagnostic prerequisite to defining californiav3: we need to know
    WHERE floats actually are at depth before we can choose domain bounds that
    won't clip float trajectories and destabilize the GPR.

WHAT IT DOES:
    1. Fetches per-dive Argo float positions from ERDDAP in 5-year chunks.
    2. Bins on a 5°x5° grid and counts UNIQUE floats per (year, cell) — one float
       with 30 dives still counts as 1 sensor, which is the right metric for GP
       data density.
    3. Saves the full census to a CSV.
    4. Produces one Cartopy heatmap PNG per year, all using the same color scale
       so visual comparisons across years are valid.

OUTPUT (all in AEResults/aeplots/float_census_california/):
    float_census_california_1999.png ... float_census_california_2024.png
    float_census_california_1999_2025.csv

USAGE:
    conda run -n ebus-cloud-env python ArgoEBUSCloud/09_ae_longterm_float_census.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Headless — no display required
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Add the ArgoEBUSCloud package root to sys.path so ebus_core is importable
# regardless of working directory. This file lives at ArgoEBUSCloud/, so
# dirname(__file__) gives us that directory directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ebus_core.ae_utils import (
    get_ebus_registry,
    get_project_paths,
    ensure_ae_dirs,
    get_float_history,
)


# ---------------------------------------------------------------------------
# STEP 1 — Config and output directory
# ---------------------------------------------------------------------------

def build_output_dir():
    """
    Constructs the output subfolder path and ensures it exists.

    Returns the absolute path to:
        AEResults/aeplots/float_census_california/

    Note: ensure_ae_dirs() only creates the three top-level AEResults
    subdirectories (aeplots, aedata, aelogs). The census subfolder must be
    created explicitly here.
    """
    ensure_ae_dirs()
    paths = get_project_paths()
    subfolder = os.path.join(paths["plots"], "float_census_california")
    os.makedirs(subfolder, exist_ok=True)
    print(f"[census] Output directory: {subfolder}")
    return subfolder


# ---------------------------------------------------------------------------
# STEP 2 — Fetch float history in 5-year chunks
# ---------------------------------------------------------------------------

# Chunks chosen to stay within ERDDAP's row-limit per request. Each chunk
# covers 5 years of the full broad california domain. The last chunk covers
# only 2 years (2024-2025) to reach the present.
FETCH_CHUNKS = [
    ("1999-01-01", "2003-12-31"),
    ("2004-01-01", "2008-12-31"),
    ("2009-01-01", "2013-12-31"),
    ("2014-01-01", "2018-12-31"),
    ("2019-01-01", "2023-12-31"),
    ("2024-01-01", "2025-12-31"),
]

def fetch_all_float_history():
    """
    Fetches raw per-dive Argo float positions for all years 1999-2025 from
    ERDDAP, using the broad "california" domain (lat [25,50], lon [-140,-110]).

    We use the "california" registry entry for its spatial bounds only.
    The date range is overridden per chunk so we can span all 26 years.

    Wraps each chunk in try/except so a single ERDDAP timeout doesn't abort
    the whole run — partial results are still useful.

    Returns a single concatenated DataFrame with all dives and an added
    'year' column (integer) derived from the dive timestamp.
    """
    frames = []

    for start_date, end_date in FETCH_CHUNKS:
        print(f"[census] Fetching {start_date} → {end_date} ...", flush=True)
        try:
            chunk = get_float_history(
                region="california",
                start_date=start_date,
                end_date=end_date,
            )
            print(f"[census]   Got {len(chunk):,} dives.")
            frames.append(chunk)
        except Exception as exc:
            # Log the failure but keep going — partial coverage is still
            # useful for the census. Missing chunks will show up as zero-
            # float years in the output, which is clearly distinguishable
            # from sparse-but-real coverage.
            print(f"[census]   WARNING: chunk failed — {exc}")

    if not frames:
        raise RuntimeError("[census] All ERDDAP chunks failed. Cannot continue.")

    df = pd.concat(frames, ignore_index=True)

    # Add integer year column for groupby and per-year PNG filenames
    df["year"] = df["time"].dt.year

    print(f"[census] Total dives fetched: {len(df):,} across {df['year'].nunique()} years.")
    return df


# ---------------------------------------------------------------------------
# STEP 3 — Bin on 5°x5° grid, count unique floats
# ---------------------------------------------------------------------------

def build_census(df):
    """
    Bins dive positions onto a 5°x5° grid and counts unique float WMO IDs
    (platform_number) per (year, lat_bin, lon_bin) cell.

    Bin centers are placed at the midpoint of each 5° cell:
        lat_bin = floor(lat / 5) * 5 + 2.5   e.g. lat=32.1 → bin 32.5
        lon_bin = floor(lon / 5) * 5 + 2.5   e.g. lon=-122.7 → bin -122.5

    Why unique floats, not dive count?
        One float making 30 dives in a cell still represents one independent
        spatial sensor for the Gaussian Process. Using unique floats gives a
        true picture of independent GP support points, not ping volume.

    Returns DataFrame with columns: year, lat_bin, lon_bin, n_floats
    """
    df = df.copy()

    # 5° bin centers via floor division
    df["lat_bin"] = (np.floor(df["lat"] / 5.0) * 5.0) + 2.5
    df["lon_bin"] = (np.floor(df["lon"] / 5.0) * 5.0) + 2.5

    census = (
        df.groupby(["year", "lat_bin", "lon_bin"])["platform_number"]
        .nunique()
        .reset_index()
        .rename(columns={"platform_number": "n_floats"})
    )

    print(f"[census] Census built: {len(census):,} (year, cell) records.")
    return census


# ---------------------------------------------------------------------------
# STEP 4 — Save CSV
# ---------------------------------------------------------------------------

def save_census_csv(census, out_dir):
    """
    Saves the full (year, lat_bin, lon_bin, n_floats) census to CSV.

    This file is the primary input for 09b_ae_analyze_float_census.py, which
    reads it to produce summary statistics and domain-recommendation tables
    for Gemini to define californiav3.
    """
    csv_path = os.path.join(out_dir, "float_census_california_1999_2025.csv")
    census.to_csv(csv_path, index=False)
    print(f"[census] CSV saved → {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# STEP 5 — Per-year PNG generation
# ---------------------------------------------------------------------------

# Domain extent for all plots. Matches the broad "california" registry entry.
LON_MIN, LON_MAX = -140.0, -110.0
LAT_MIN, LAT_MAX =   25.0,   50.0

# Color scale fixed across all years so year-to-year comparisons are valid.
# 15 chosen as the upper bound: a well-sampled 5°x5° CCS cell typically has
# 5-12 unique floats in peak years; 15 gives headroom without washing out
# the sparse early-Argo era (1999-2004).
VMIN, VMAX = 0, 15


def plot_year(year_census, year, out_dir):
    """
    Produces a single Cartopy pcolormesh heatmap for one year's float density.

    Inputs:
        year_census  - DataFrame filtered to a single year:
                       columns [lat_bin, lon_bin, n_floats]
        year         - Integer year, used in title and filename
        out_dir      - Absolute path to the output subfolder

    The pivot step converts the tidy (lat_bin, lon_bin, n_floats) records into
    a 2D array on the regular 5°x5° grid. Cells with no floats are filled with
    zero so the full domain renders correctly (no blank tiles).

    pcolormesh expects bin EDGES, not centers, so we derive edges from the
    sorted center values by adding/subtracting half the bin width (2.5°).
    """
    # Pivot to 2D: rows = lat bins (ascending), cols = lon bins (ascending)
    pivot = (
        year_census
        .pivot(index="lat_bin", columns="lon_bin", values="n_floats")
        .sort_index()                    # lat ascending (south → north)
        .sort_index(axis=1)             # lon ascending (west → east)
        .fillna(0.0)
    )

    lat_centers = np.array(pivot.index)
    lon_centers = np.array(pivot.columns)

    # Convert bin centers to edges for pcolormesh.
    # Each center is the midpoint of a 5° cell, so edges are ±2.5° from center.
    lat_edges = np.concatenate([[lat_centers[0] - 2.5], lat_centers + 2.5])
    lon_edges = np.concatenate([[lon_centers[0] - 2.5], lon_centers + 2.5])

    # Build figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    # Plot the float density grid. zorder=1 so land overlay sits above it.
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, pivot.values,
        vmin=VMIN, vmax=VMAX,
        cmap="YlOrRd",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    # Geographic features — land on top (zorder 100) so coastline is crisp
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor="k", facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, zorder=101, linewidth=0.7)

    # Gridlines with degree labels
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5, zorder=102)
    gl.top_labels = False
    gl.right_labels = False

    # Colorbar and title
    cbar = plt.colorbar(mesh, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Unique Floats per 5°×5° Cell", fontsize=11)

    total_floats = int(year_census["n_floats"].sum())
    ax.set_title(
        f"Argo Float Density — {year}   (total unique floats: {total_floats})",
        fontsize=13, pad=10,
    )

    # Save and close — plt.close() is critical in a loop; without it each
    # figure accumulates in memory and the process will OOM after ~20 years.
    out_path = os.path.join(out_dir, f"float_census_california_{year}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_all_years(census, out_dir):
    """
    Iterates over every year present in the census and calls plot_year().
    Prints one progress line per year.
    """
    years = sorted(census["year"].unique())
    print(f"[census] Generating {len(years)} per-year PNGs ...")

    for year in years:
        year_df = census[census["year"] == year][["lat_bin", "lon_bin", "n_floats"]].copy()
        path = plot_year(year_df, year, out_dir)
        print(f"[census]   {year} → {os.path.basename(path)}")

    print(f"[census] All PNGs saved to {out_dir}")


# ---------------------------------------------------------------------------
# STEP 6 — Print top-10 hotspot table
# ---------------------------------------------------------------------------

def print_top_hotspots(census, n=10):
    """
    Prints the top-N (year, lat_bin, lon_bin) records by n_floats to stdout.
    This is a quick sanity check: Southern California Bight (30-35°N, ~120°W)
    should appear consistently in the top rows from ~2010 onward.
    """
    top = census.nlargest(n, "n_floats")
    print(f"\n[census] Top {n} float-dense cells:")
    print(top.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir  = build_output_dir()
    raw      = fetch_all_float_history()
    census   = build_census(raw)
    save_census_csv(census, out_dir)
    plot_all_years(census, out_dir)
    print_top_hotspots(census)
    print("[census] Done.")
