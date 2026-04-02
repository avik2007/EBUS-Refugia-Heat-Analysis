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


# ---------------------------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------------------------

def draw_ccs_bounds(ax, ccs_bounds):
    # Draws the californiav2 domain boundary as a dashed red rectangle on a
    # Cartopy axes. Labels it "CCS Analysis Bounds" in red near the top-left
    # corner of the rectangle.
    #
    # Why this overlay matters: the californiav2 domain (lon [-130,-115],
    # lat [30,45]) is tighter than the broad census domain (lon [-140,-110],
    # lat [25,50]). Floats outside this rectangle are excluded from GPR runs.
    # Overlaying this boundary makes visible exactly which census cells are
    # inside vs. outside the GPR domain — the key diagnostic for the 2015
    # source layer regression.
    #
    # Inputs:
    #   ax         - Cartopy GeoAxes (must be in PlateCarree projection)
    #   ccs_bounds - dict {"lat": [lat_min, lat_max], "lon": [lon_min, lon_max]}
    #                from get_ccs_bounds()
    lat_min, lat_max = ccs_bounds["lat"]
    lon_min, lon_max = ccs_bounds["lon"]

    # Draw the four sides of the rectangle as a closed polygon.
    # ax.plot() with transform=PlateCarree keeps the line on the map.
    box_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    box_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(
        box_lons, box_lats,
        color="red", linewidth=1.5, linestyle="--",
        transform=ccrs.PlateCarree(),
        zorder=200,   # Above land (zorder 100) and coastline (101)
    )

    # Label near the top-left interior corner, offset slightly inward so
    # the text sits inside the box and is legible against the ocean color.
    ax.text(
        lon_min + 0.5, lat_max - 1.5,
        "CCS Analysis Bounds",
        color="red", fontsize=8, fontweight="bold",
        transform=ccrs.PlateCarree(),
        zorder=201,
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
    )


def plot_year(year_census, year, layer_name, display_label, out_dir, ccs_bounds):
    # Produces a Cartopy pcolormesh heatmap for one year and one depth layer.
    #
    # The pivot step converts the tidy (lat_bin, lon_bin, n_floats) records
    # into a 2D array on the regular 5-degree grid. Cells with no floats are
    # filled with 0.0 so the full domain renders without blank tiles.
    #
    # pcolormesh expects bin EDGES, not centers. We derive edges from the
    # sorted center values by adding/subtracting 2.5 degrees (half the bin width).
    #
    # Inputs:
    #   year_census   - DataFrame for one year: columns [lat_bin, lon_bin, n_floats]
    #   year          - Integer year for title and filename
    #   layer_name    - Short key (e.g., "source") for filename
    #   display_label - Human-readable label (e.g., "Source (150-400m)") for title
    #   out_dir       - Absolute path to output subfolder
    #   ccs_bounds    - Dict from get_ccs_bounds(), passed to draw_ccs_bounds()
    #
    # Saves: float_census_{layer_name}_{year}.png
    # Returns: absolute path to the saved PNG
    pivot = (
        year_census
        .pivot(index="lat_bin", columns="lon_bin", values="n_floats")
        .sort_index()
        .sort_index(axis=1)
        .fillna(0.0)
    )

    lat_centers = np.array(pivot.index)
    lon_centers = np.array(pivot.columns)
    # Bin edges: each center is the midpoint of a 5-degree cell, so edges are ±2.5°
    lat_edges = np.concatenate([[lat_centers[0] - 2.5], lat_centers + 2.5])
    lon_edges = np.concatenate([[lon_centers[0] - 2.5], lon_centers + 2.5])

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon_edges, lat_edges, pivot.values,
        vmin=VMIN, vmax=VMAX,
        cmap="YlOrRd",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    ax.add_feature(cfeature.LAND, zorder=100, edgecolor="k", facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, zorder=101, linewidth=0.7)

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5, zorder=102)
    gl.top_labels   = False
    gl.right_labels = False

    # Overlay the CCS Analysis Bounds — the core diagnostic of this script
    draw_ccs_bounds(ax, ccs_bounds)

    cbar = plt.colorbar(mesh, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Unique Floats per 5°×5° Cell", fontsize=11)

    total_floats = int(year_census["n_floats"].sum())
    ax.set_title(
        f"Argo Float Density ({display_label}) — {year}   "
        f"(total unique floats: {total_floats})",
        fontsize=13, pad=10,
    )

    out_path = os.path.join(out_dir, f"float_census_{layer_name}_{year}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)   # Critical: prevents memory accumulation across 100+ iterations
    return out_path


def plot_layer_mean(census, layer_name, display_label, out_dir, ccs_bounds):
    # Plots mean float density across all years for one depth layer.
    #
    # To get a true mean (not just the mean of years-with-floats), we first
    # build a full year×cell grid by reindexing with all (year, lat_bin, lon_bin)
    # combinations, filling missing cells with 0. Then we average over years.
    # Without this reindex, cells with no floats in some years would show an
    # inflated mean (e.g., a cell with floats in only 5 years would average over
    # those 5 years only, not all 26).
    #
    # Inputs:
    #   census        - Full census DataFrame: [year, lat_bin, lon_bin, n_floats]
    #   layer_name    - Short key for filename (e.g., "source")
    #   display_label - Human-readable name for title
    #   out_dir       - Absolute path to output subfolder
    #   ccs_bounds    - Dict from get_ccs_bounds()
    #
    # Saves: float_census_{layer_name}_mean.png
    # Returns: absolute path to the saved PNG
    all_years = census["year"].unique()
    all_lat   = census["lat_bin"].unique()
    all_lon   = census["lon_bin"].unique()

    # Build a full grid so empty cells contribute zeros to the mean
    full_index = pd.MultiIndex.from_product(
        [all_years, all_lat, all_lon],
        names=["year", "lat_bin", "lon_bin"],
    )
    census_full = (
        census.set_index(["year", "lat_bin", "lon_bin"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    mean_density = (
        census_full
        .groupby(["lat_bin", "lon_bin"])["n_floats"]
        .mean()
        .reset_index()
        .rename(columns={"n_floats": "mean_n_floats"})
    )

    pivot = (
        mean_density
        .pivot(index="lat_bin", columns="lon_bin", values="mean_n_floats")
        .sort_index()
        .sort_index(axis=1)
        .fillna(0.0)
    )

    lat_centers = np.array(pivot.index)
    lon_centers = np.array(pivot.columns)
    lat_edges   = np.concatenate([[lat_centers[0] - 2.5], lat_centers + 2.5])
    lon_edges   = np.concatenate([[lon_centers[0] - 2.5], lon_centers + 2.5])

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon_edges, lat_edges, pivot.values,
        vmin=VMIN, vmax=VMAX,
        cmap="YlOrRd",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    ax.add_feature(cfeature.LAND, zorder=100, edgecolor="k", facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, zorder=101, linewidth=0.7)

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5, zorder=102)
    gl.top_labels   = False
    gl.right_labels = False

    draw_ccs_bounds(ax, ccs_bounds)

    cbar = plt.colorbar(mesh, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Mean Unique Floats per 5°×5° Cell (1999–2025)", fontsize=11)

    n_years = len(all_years)
    ax.set_title(
        f"Mean Argo Float Density ({display_label}) — 1999–2025 "
        f"(averaged over {n_years} years)",
        fontsize=13, pad=10,
    )

    out_path = os.path.join(out_dir, f"float_census_{layer_name}_mean.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[census/{layer_name}] Mean map saved → {out_path}")
    return out_path


def plot_all_years_for_layer(census, layer_name, display_label, out_dir, ccs_bounds):
    # Iterates over every year present in the census and calls plot_year().
    #
    # Prints one progress line per year so the user can track the loop.
    # plt.close() inside plot_year() ensures memory does not accumulate
    # across the ~26 iterations per layer call.
    #
    # Inputs:
    #   census        - Full census DataFrame: [year, lat_bin, lon_bin, n_floats]
    #   layer_name    - Short key (e.g., "source") for filenames and log messages
    #   display_label - Human-readable name (e.g., "Source (150-400m)") for plot titles
    #   out_dir       - Absolute path to output subfolder
    #   ccs_bounds    - Dict from get_ccs_bounds(), passed to each plot_year() call
    years = sorted(census["year"].unique())
    print(f"[census/{layer_name}] Generating {len(years)} per-year PNGs ...")
    for year in years:
        year_df = census[census["year"] == year][["lat_bin", "lon_bin", "n_floats"]].copy()
        path = plot_year(year_df, year, layer_name, display_label, out_dir, ccs_bounds)
        print(f"[census/{layer_name}]   {year} → {os.path.basename(path)}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Setup: output directory + CCS domain bounds for plot overlays.
    # ccs_bounds is read once here and passed to every plot call so that
    # the overlay stays consistent across all figures.
    out_dir    = build_output_dir()
    ccs_bounds = get_ccs_bounds()

    # Process each layer in order. "alldepths" goes first (matches script 09
    # baseline), then the three scientific layers. This ordering means that if
    # the run is interrupted, the all-depths and skin results are already saved
    # before the deeper, slower source/background fetches.
    for layer_name, (pres_min, pres_max, display_label) in LAYERS.items():
        print(f"\n[census] === Layer: {layer_name} ({display_label}) ===")

        # Fetch raw dive positions from ERDDAP for this layer
        raw = fetch_layer_data(layer_name, pres_min, pres_max)

        # Bin to 5°x5° unique-float census
        census = build_census(raw)

        # Archive census to CSV
        save_census_csv(census, layer_name, out_dir)

        # Per-year PNGs
        plot_all_years_for_layer(census, layer_name, display_label, out_dir, ccs_bounds)

        # Per-layer mean PNG (all years averaged)
        plot_layer_mean(census, layer_name, display_label, out_dir, ccs_bounds)

    print("\n[census] All layers complete. Done.")
    print(f"[census/{layer_name}] Per-year PNGs done.")
