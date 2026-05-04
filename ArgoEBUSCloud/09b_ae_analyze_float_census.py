"""
09b_ae_analyze_float_census.py

PURPOSE:
    Read and analyze the float census CSV produced by 09_ae_longterm_float_census.py.
    Surfaces the domain-recommendation data Gemini needs to define californiav3:
    specifically, which 5°x5° cells have had consistent float coverage over the
    26-year Argo record, and at what density.

    Run this AFTER 09_ae_longterm_float_census.py has completed.

WHAT IT PRODUCES (printed to stdout):
    1. Annual total float count table (how many unique floats per year)
    2. Top-10 most persistent cells (cells present in the most years)
    3. Domain recommendation table: cells with floats in ≥ 20 of 26 years,
       sorted by mean n_floats descending — the empirical basis for californiav3

WHAT IT SAVES (same subfolder as census data):
    float_census_annual_totals.png   — bar chart of annual total floats
    float_census_mean_density.png    — Cartopy map of mean float density over all years

USAGE:
    conda run -n ebus-cloud-env python ArgoEBUSCloud/09b_ae_analyze_float_census.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Headless
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ebus_core.ae_utils import get_project_paths

# ---------------------------------------------------------------------------
# Config — paths are fixed because this script exists solely to analyze the
# census output. No parameterization needed.
# ---------------------------------------------------------------------------

def get_census_dir():
    """
    Returns the absolute path to the float census subfolder.
    Must match the output dir used by 09_ae_longterm_float_census.py.
    """
    paths = get_project_paths()
    return os.path.join(paths["plots"], "float_census_california")


CSV_NAME = "float_census_california_1999_2025.csv"

# Domain extent for Cartopy plots — matches the broad california registry entry
LON_MIN, LON_MAX = -140.0, -110.0
LAT_MIN, LAT_MAX =   25.0,   50.0

# Minimum year-presence threshold for the californiav3 domain recommendation.
# A cell must have floats in at least this many years (out of 26) to be considered
# a reliable anchor point for GPR. 20/26 ≈ 77% temporal coverage.
MIN_YEARS_PRESENT = 20


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_census(census_dir):
    """
    Loads the float census CSV.

    Expected columns: year (int), lat_bin (float), lon_bin (float), n_floats (int)

    Raises FileNotFoundError with a helpful message if the census hasn't been
    generated yet (i.e., Script 09 hasn't been run).
    """
    csv_path = os.path.join(census_dir, CSV_NAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Census CSV not found at {csv_path}\n"
            "Run 09_ae_longterm_float_census.py first."
        )
    df = pd.read_csv(csv_path)
    print(f"[analyze] Loaded census: {len(df):,} records, "
          f"years {df['year'].min()}–{df['year'].max()}")
    return df


# ---------------------------------------------------------------------------
# Analysis 1 — Annual total floats
# ---------------------------------------------------------------------------

def analyze_annual_totals(census, census_dir):
    """
    Computes and prints the total number of unique float-cell observations
    per year (sum of n_floats across all cells for each year).

    This shows the growth trajectory of the Argo network — sparse pre-2005,
    growing through 2010, near-mature from ~2015 onward.

    Saves: float_census_annual_totals.png — bar chart, one bar per year
    """
    annual = census.groupby("year")["n_floats"].sum().reset_index()
    annual.columns = ["year", "total_float_obs"]

    # Print table
    print("\n[analyze] Annual float observations (sum of n_floats across all cells):")
    print(annual.to_string(index=False))

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(annual["year"], annual["total_float_obs"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Total Float Observations (Σ unique floats per cell)", fontsize=11)
    ax.set_title("Argo Float Data Volume — California Domain (25–50°N, 140–110°W)", fontsize=13)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.4, linestyle="--")

    out_path = os.path.join(census_dir, "float_census_annual_totals.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Bar chart saved → {out_path}")

    return annual


# ---------------------------------------------------------------------------
# Analysis 2 — Persistent hotspots
# ---------------------------------------------------------------------------

def analyze_persistent_hotspots(census, top_n=10):
    """
    For each (lat_bin, lon_bin) cell, counts how many distinct years it had
    at least one float. A cell that appears in 26/26 years is maximally
    persistent — it has continuous GP support across the full Argo record.

    Prints the top-N cells by year_count (ties broken by mean_n_floats).

    This is a qualitative check: high-persistence cells near the coast
    (roughly the California Undercurrent corridor) are exactly where we need
    reliable GPR coverage for the stealth warming signal.
    """
    persistence = (
        census[census["n_floats"] > 0]
        .groupby(["lat_bin", "lon_bin"])
        .agg(
            year_count=("year", "nunique"),
            mean_n_floats=("n_floats", "mean"),
        )
        .reset_index()
        .sort_values(["year_count", "mean_n_floats"], ascending=[False, False])
    )

    total_years = census["year"].nunique()
    print(f"\n[analyze] Top {top_n} most persistent cells (out of {total_years} years):")
    print(persistence.head(top_n).to_string(index=False))

    return persistence


# ---------------------------------------------------------------------------
# Analysis 3 — Mean density map
# ---------------------------------------------------------------------------

def plot_mean_density(census, census_dir):
    """
    Averages n_floats across all years per cell and plots a single Cartopy
    pcolormesh map. Cells that never had floats appear as zero (gray-ish
    at the low end of the YlOrRd scale).

    This map shows the 26-year average float density at a glance — the
    complement to the per-year maps from Script 09. Useful for Gemini to
    identify the stable coverage core that should define californiav3.

    Saves: float_census_mean_density.png
    """
    # Mean n_floats per cell across all years (including years with zero floats).
    # We need to include zero-float years to get a true mean, not just the
    # mean of years when floats were present. Build the full year×cell grid
    # first by reindexing with a MultiIndex that covers all combinations.
    all_years = census["year"].unique()
    all_lat   = census["lat_bin"].unique()
    all_lon   = census["lon_bin"].unique()

    full_index = pd.MultiIndex.from_product(
        [all_years, all_lat, all_lon],
        names=["year", "lat_bin", "lon_bin"],
    )
    # Reindex the census to the full grid, filling missing cells with 0
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

    # Pivot to 2D for pcolormesh
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
        vmin=0, vmax=10,          # Lower vmax than per-year maps: mean < peak
        cmap="YlOrRd",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )

    ax.add_feature(cfeature.LAND, zorder=100, edgecolor="k", facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, zorder=101, linewidth=0.7)

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5, zorder=102)
    gl.top_labels   = False
    gl.right_labels = False

    cbar = plt.colorbar(mesh, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Mean Unique Floats per 5°×5° Cell (1999–2024)", fontsize=11)

    n_years = len(all_years)
    ax.set_title(
        f"Mean Argo Float Density 1999–2024 (averaged over {n_years} years)",
        fontsize=13, pad=10,
    )

    out_path = os.path.join(census_dir, "float_census_mean_density.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Mean density map saved → {out_path}")


# ---------------------------------------------------------------------------
# Analysis 4 — Domain recommendation helper
# ---------------------------------------------------------------------------

def print_domain_recommendation(persistence, census):
    """
    Identifies cells that had floats in >= MIN_YEARS_PRESENT out of all years,
    sorted by mean n_floats descending.

    THIS IS THE KEY OUTPUT FOR GEMINI. The bounding box of these high-persistence
    cells is the empirical candidate for californiav3. Gemini should use this
    list to:
      - Identify the lat/lon range that captures the core coverage cluster
      - Exclude cells that are in the list only due to high-traffic transits
        (e.g., far offshore Pacific) vs. sustained coastal presence
      - Define the new domain bounds that will prevent Source Layer GPR regression

    Also prints the implied bounding box (min/max lat/lon of qualifying cells).
    """
    total_years = census["year"].nunique()
    qualifying = (
        persistence[persistence["year_count"] >= MIN_YEARS_PRESENT]
        .sort_values("mean_n_floats", ascending=False)
    )

    print(f"\n[analyze] === CALIFORNIAV3 DOMAIN RECOMMENDATION DATA ===")
    print(f"Cells present in >= {MIN_YEARS_PRESENT}/{total_years} years, "
          f"sorted by mean n_floats:")
    print(qualifying.to_string(index=False))

    if len(qualifying) > 0:
        lat_min = qualifying["lat_bin"].min() - 2.5
        lat_max = qualifying["lat_bin"].max() + 2.5
        lon_min = qualifying["lon_bin"].min() - 2.5
        lon_max = qualifying["lon_bin"].max() + 2.5
        print(f"\n[analyze] Implied bounding box of qualifying cells:")
        print(f"  lat: [{lat_min}, {lat_max}]")
        print(f"  lon: [{lon_min}, {lon_max}]")
        print(f"  (Gemini should refine these bounds based on oceanographic rationale)")
    else:
        print("[analyze] WARNING: No cells meet the persistence threshold.")

    print("[analyze] === END DOMAIN RECOMMENDATION DATA ===\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    census_dir = get_census_dir()
    census     = load_census(census_dir)

    analyze_annual_totals(census, census_dir)
    persistence = analyze_persistent_hotspots(census)
    plot_mean_density(census, census_dir)
    print_domain_recommendation(persistence, census)

    print("[analyze] Done.")
