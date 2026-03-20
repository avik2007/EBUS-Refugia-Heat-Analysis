"""
Float trajectory spaghetti map — modular diagnostic tool.

Fetches raw Argo dive positions via get_float_history() and plots each float as
a distinct colored line over a Cartopy basemap.  Intended as a coverage diagnostic
that can be called serially alongside run_diagnostic_inspection() in any analysis
script — both functions share the same parameter signature so they compose cleanly.

Usage as a module (e.g., from a multi-region analysis script):
    from 03_ae_plot_float_paths import plot_float_paths
    plot_float_paths(region="california", depth_range=(0, 100))
    plot_float_paths(region="humboldt",   depth_range=(0, 100))

Usage as a standalone script:
    python 03_ae_plot_float_paths.py
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


def plot_float_paths(region="california", lat_step=0.5, lon_step=0.5,
                     time_step=30.0, depth_range=(0, 100)):
    # ---------------------------------------------------------------------------
    # Produce a float-trajectory spaghetti map for a given analysis run.
    #
    # Signature mirrors run_diagnostic_inspection() exactly so the two functions
    # can be called back-to-back in any parent analysis script without duplicating
    # parameter handling:
    #
    #   run_diagnostic_inspection(region="california", depth_range=(0, 100))
    #   plot_float_paths(         region="california", depth_range=(0, 100))
    #
    # Inputs (all match get_ae_config / run_diagnostic_inspection):
    #   region      - Key into get_ebus_registry() ("california", "humboldt", etc.)
    #   lat_step    - Spatial bin size in degrees latitude (default 0.5)
    #   lon_step    - Spatial bin size in degrees longitude (default 0.5)
    #   time_step   - Rolling-window step in days (default 30.0)
    #   depth_range - Tuple (min_m, max_m) defining the depth layer to label
    #                 (used only for consistent output filename; no depth filtering
    #                 is applied to the float-trajectory data itself, since
    #                 get_float_history uses &distinct() and drops depth entirely)
    #
    # Output:
    #   PNG saved to AEResults/aeplots/float_path_traj_{run_id}.png
    #   Returns the output path as a string so callers can log it.
    # ---------------------------------------------------------------------------

    # --- 1. Resolve config (dates, spatial bounds, output paths) ---
    # get_ae_config() reads the EBUS registry to resolve default start/end dates
    # for the chosen region and constructs the canonical run_id used for naming.
    config     = get_ae_config(
        region      = region,
        lat_step    = lat_step,
        lon_step    = lon_step,
        time_step   = time_step,
        depth_range = depth_range,
    )
    start_date = config["start_date"]
    end_date   = config["end_date"]
    run_id     = config["run_id"]

    # --- 2. Fetch dive-level float positions via ERDDAP ---
    # get_float_history() collapses the many per-pressure-level ERDDAP rows down
    # to one row per dive using &distinct(). Expect ~14,000 rows / ~70 floats for
    # California 2015.  The depth_range is NOT passed here — depth is irrelevant
    # for spatial coverage; we want all floats that visited the region.
    print(f"Fetching float histories for '{region}' ({start_date} -> {end_date})...")
    df = get_float_history(region, start_date, end_date)

    print(f"  DataFrame shape : {df.shape}")
    n_floats = df["platform_number"].nunique()
    print(f"  Unique floats   : {n_floats}")

    if df.empty:
        print("ERROR: No data returned. Check region / date range.")
        return None

    # --- 3. Resolve save path ---
    # Float trajectory plots go directly into aeplots/ (not inside a snapshot subfolder)
    # because they represent the full study period, not a single kriging snapshot.
    plots_dir = config["paths"]["plots"]
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"float_path_traj_{run_id}.png")

    # --- 4. Build the spaghetti map ---
    # Projection and feature style intentionally match argoebus_gp_physics.py
    # (PlateCarree, LAND + COASTLINE, dashed gridlines) so trajectory and kriging
    # plots look consistent when viewed side by side.
    lon_min, lon_max = config["lon"]
    lat_min, lat_max = config["lat"]

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      zorder=100, edgecolor="k",  facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, zorder=101)
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)

    # Assign a distinct color per float by cycling through tab20.
    # tab20 has 20 colors; it wraps for >20 floats, which is acceptable because
    # the goal here is spatial coverage, not individual float identity.
    floats = df["platform_number"].unique()
    cmap   = matplotlib.colormaps.get_cmap("tab20").resampled(len(floats))
    colors = {fid: cmap(i) for i, fid in enumerate(floats)}

    # Draw each float as a time-sorted line with small scatter dots at each dive
    # to distinguish multi-visit positions from straight-line interpolation.
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
        f"Argo Float Trajectories — {region.capitalize()} "
        f"({start_date} to {end_date})\n"
        f"n={n_floats} floats, {len(df):,} dives",
        fontsize=11,
    )

    plt.tight_layout()

    # --- 5. Save and clean up ---
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    # Default invocation mirrors the canonical 2015 California Skin Layer run.
    # To run multiple regions in sequence, import plot_float_paths() in a parent script:
    #
    #   plot_float_paths(region="california", depth_range=(0, 100))
    #   plot_float_paths(region="humboldt",   depth_range=(0, 100))
    plot_float_paths(
        region      = "california",
        lat_step    = 0.5,
        lon_step    = 0.5,
        time_step   = 30.0,
        depth_range = (0, 100),
    )
