"""
Diagnostic: float trajectory spaghetti map.

Fetches raw Argo dive positions via get_float_history() and plots each float as
a distinct colored line over a Cartopy basemap. Intended as a standalone sanity
check that the data-access layer works and as a visual coverage diagnostic.

Usage (defaults match the canonical 2015 California run):
    python 03_ae_plot_float_paths.py

All parameters mirror get_ae_config() so the save path is always consistent with
the run being analyzed.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Allow running from the ArgoEBUSCloud directory without installing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ebus_core.ae_utils import get_float_history, get_ae_config

# ---------------------------------------------------------------------------
# Parameters — edit here to target a different run
# ---------------------------------------------------------------------------
REGION      = "california"
START_DATE  = "2015-01-01"
END_DATE    = "2015-12-31"
DEPTH_RANGE = (0, 100)
LAT_STEP    = 0.5
LON_STEP    = 0.5
TIME_STEP   = 30.0
# ---------------------------------------------------------------------------


def main():
    # --- 1. Fetch dive-level float positions ---
    # get_float_history() queries ERDDAP with &distinct() to collapse per-pressure
    # rows to one row per dive.  Expect ~14,000 rows and ~70 floats for 2015 California.
    print(f"Fetching float histories for '{REGION}' ({START_DATE} → {END_DATE})...")
    df = get_float_history(REGION, START_DATE, END_DATE)

    print(f"  DataFrame shape : {df.shape}")
    n_floats = df["platform_number"].nunique()
    print(f"  Unique floats   : {n_floats}")

    if df.empty:
        print("ERROR: No data returned. Check region / date range.")
        sys.exit(1)

    # --- 2. Resolve the save path using get_ae_config() ---
    # Float trajectory plots live directly in aeplots/ (not inside a snapshot subfolder)
    # because they represent the full study period, not a single kriging snapshot.
    config  = get_ae_config(
        region      = REGION,
        start_date  = START_DATE,
        end_date    = END_DATE,
        depth_range = DEPTH_RANGE,
        lat_step    = LAT_STEP,
        lon_step    = LON_STEP,
        time_step   = TIME_STEP,
    )

    # Filename mirrors run_id so it's immediately clear what region, dates, resolution,
    # and depth this plot covers.  Sits directly in aeplots/, not inside a snapshot
    # subfolder, because it represents the full study period rather than one kriging
    # snapshot.
    run_id    = config["run_id"]
    plots_dir = config["paths"]["plots"]
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"float_path_traj_{run_id}.png")

    # --- 3. Build the spaghetti map ---
    # Style matches argoebus_gp_physics.py: PlateCarree projection, LAND + COASTLINE
    # features, dashed gridlines.
    reg = config  # has lat/lon keys
    lon_min, lon_max = reg["lon"]
    lat_min, lat_max = reg["lat"]

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      zorder=100, edgecolor="k",  facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, zorder=101)
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)

    # Assign a distinct color to each float by cycling through a colormap.
    # We use a tab20 palette (20 distinct colors) — it wraps around for >20 floats,
    # which is visually acceptable at ~70 floats since the goal is coverage, not identity.
    floats  = df["platform_number"].unique()
    cmap    = matplotlib.colormaps.get_cmap("tab20").resampled(len(floats))
    colors  = {fid: cmap(i) for i, fid in enumerate(floats)}

    # Plot each float's dive sequence as a line (sorted by time so the path is
    # topologically correct) with a small marker at each actual dive position.
    for fid, grp in df.groupby("platform_number"):
        grp_sorted = grp.sort_values("time")
        ax.plot(
            grp_sorted["lon"].values,
            grp_sorted["lat"].values,
            color     = colors[fid],
            linewidth = 0.8,
            alpha     = 0.7,
            transform = ccrs.PlateCarree(),
        )
        # Small dot at each dive — distinguishes multi-visit positions from straight lines
        ax.scatter(
            grp_sorted["lon"].values,
            grp_sorted["lat"].values,
            color     = colors[fid],
            s         = 4,
            alpha     = 0.5,
            transform = ccrs.PlateCarree(),
            zorder    = 102,
        )

    ax.set_title(
        f"Argo Float Trajectories — {REGION.capitalize()} "
        f"({START_DATE} to {END_DATE})\n"
        f"n={n_floats} floats, {len(df):,} dives",
        fontsize=11,
    )

    plt.tight_layout()

    # --- 4. Save ---
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


if __name__ == "__main__":
    main()
