# Depth-Aware Float Census Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a depth-layer-aware Argo float census that shows per-year and mean float coverage for the skin (0–100m), source (150–400m), and background (500–1000m) layers, with the californiav2 domain boundary overlaid on all figures to make domain-clipping effects visible.

**Architecture:** A new ERDDAP helper `get_float_history_by_layer()` in `ae_utils.py` adds pressure constraints to the existing query pattern. A new standalone script `09c_ae_depth_aware_float_census.py` runs four fetch passes (all-depths + 3 layers), builds 5°×5° unique-float censuses, and saves 107 PNGs plus 4 CSVs.

**Tech Stack:** Python 3, pandas, numpy, matplotlib (Agg), cartopy, requests — all in `ebus-cloud-env`. Always run scripts as `conda run -n ebus-cloud-env python <script>`.

---

## File Map

| Action  | Path | Responsibility |
|---------|------|----------------|
| Modify  | `ArgoEBUSCloud/ebus_core/ae_utils.py` | Add `get_float_history_by_layer()` below `get_float_history()` |
| Create  | `ArgoEBUSCloud/09c_ae_depth_aware_float_census.py` | Full depth-aware census script |

---

## Task 1: Add `get_float_history_by_layer()` to `ae_utils.py`

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/ae_utils.py` (insert after line 251, after `get_float_history()`)

- [ ] **Step 1: Read the existing `get_float_history()` function**

Open `ArgoEBUSCloud/ebus_core/ae_utils.py` and read lines 156–251. Understand the ERDDAP URL construction and column handling — we are copying it almost verbatim with two changes: (a) add `pres_min`/`pres_max` parameters, (b) insert a pressure constraint into `raw_query` before `&distinct()`.

- [ ] **Step 2: Insert `get_float_history_by_layer()` into `ae_utils.py`**

Add the following function immediately after the closing `return` of `get_float_history()` (after line 250, before `def calculate_bin`):

```python
def get_float_history_by_layer(region="california", pres_min=0, pres_max=100,
                                start_date=None, end_date=None):
    # Retrieve per-dive Argo float positions filtered to a specific pressure (depth) range.
    #
    # Identical to get_float_history() except that it adds pressure constraints to the
    # ERDDAP query, so only dives that had at least one measurement in [pres_min, pres_max]
    # dbar are returned. This is used by the depth-aware census (09c) to count floats
    # that actually profiled into each scientific layer.
    #
    # Critical detail: pres is a FILTER CONSTRAINT only — it is NOT included in the
    # returned columns. The column list is identical to get_float_history(). This means
    # &distinct() collapses to unique (float, time) dives, not unique (float, time, pres)
    # rows. A float with 10 measurements in the source layer still appears once per dive.
    #
    # Inputs:
    #   region     - Key into get_ebus_registry(). Determines lat/lon spatial window.
    #   pres_min   - Lower pressure bound in dbar (e.g., 150 for source layer).
    #   pres_max   - Upper pressure bound in dbar (e.g., 400 for source layer).
    #   start_date - ISO string "YYYY-MM-DD". Falls back to registry time[0] if None.
    #   end_date   - ISO string "YYYY-MM-DD". Falls back to registry time[1] if None.
    #
    # Output: DataFrame with columns:
    #   platform_number (str)  - Argo float WMO ID
    #   lat (float)            - Dive latitude, degrees N
    #   lon (float)            - Dive longitude, degrees E
    #   time (datetime, UTC)   - Dive timestamp
    #   time_days (float)      - Days since 1999-01-01 (matches OHC parquet baseline)
    import pandas as pd
    import io
    import requests

    registry = get_ebus_registry()
    if region not in registry:
        raise ValueError(f"Region '{region}' not found in registry. Known: {list(registry.keys())}")

    reg = registry[region]

    t_start = start_date if start_date else reg["time"][0]
    t_end   = end_date   if end_date   else reg["time"][1]

    lat_min, lat_max = reg["lat"]
    lon_min, lon_max = reg["lon"]

    base_url = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"

    # Build the query string. pres is a constraint, NOT a requested column.
    # Column list is identical to get_float_history() so downstream census code
    # can treat both functions' outputs identically.
    raw_query = (
        f"platform_number,time,latitude,longitude"
        f"&latitude>={lat_min}&latitude<={lat_max}"
        f"&longitude>={lon_min}&longitude<={lon_max}"
        f"&time>={t_start}T00:00:00Z"
        f"&time<={t_end}T23:59:59Z"
        f"&pres>={pres_min}&pres<={pres_max}"
        f"&distinct()"
    )

    # Tomcat 11 on erddap.ifremer.fr requires < and > to be percent-encoded
    encoded_query = raw_query.replace("<", "%3C").replace(">", "%3E")
    erddap_url = f"{base_url}?{encoded_query}"

    try:
        resp = requests.get(erddap_url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
    except Exception as e:
        raise RuntimeError(
            f"ERDDAP layer query failed for region '{region}' "
            f"pres [{pres_min}, {pres_max}] ({t_start} to {t_end}).\n"
            f"URL attempted: {erddap_url}\n"
            f"Original error: {e}"
        )

    df["time"] = pd.to_datetime(df["time"], utc=True)
    baseline = pd.Timestamp("1999-01-01", tz="UTC")
    df["time_days"] = (df["time"] - baseline).dt.total_seconds() / 86400.0
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    return df[["platform_number", "lat", "lon", "time", "time_days"]]
```

- [ ] **Step 3: Verify the query string is correct by inspection**

Mentally trace through a call with `pres_min=150, pres_max=400`. The `raw_query` should include `&pres>=150&pres<=400` between the time constraint and `&distinct()`. Verify `pres` does NOT appear in the column list (`platform_number,time,latitude,longitude`).

- [ ] **Step 4: Smoke-test the new function with a small date range**

```bash
conda run -n ebus-cloud-env python -c "
import sys; sys.path.insert(0, 'ArgoEBUSCloud')
from ebus_core.ae_utils import get_float_history_by_layer
df = get_float_history_by_layer('california', pres_min=150, pres_max=400,
                                 start_date='2015-01-01', end_date='2015-01-31')
print('Rows:', len(df))
print('Columns:', list(df.columns))
print(df.head(3))
assert list(df.columns) == ['platform_number', 'lat', 'lon', 'time', 'time_days'], 'Wrong columns'
assert len(df) > 0, 'No data returned — ERDDAP may be down or query wrong'
print('PASS')
"
```

Expected: 200–2000 rows, columns `['platform_number', 'lat', 'lon', 'time', 'time_days']`, no crash.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/ae_utils.py
git commit -m "feat: add get_float_history_by_layer() ERDDAP helper with pressure constraints"
```

---

## Task 2: Create script skeleton, constants, and output directory setup

**Files:**
- Create: `ArgoEBUSCloud/09c_ae_depth_aware_float_census.py`

- [ ] **Step 1: Create the script with module docstring, imports, and constants**

```python
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

# The three scientific depth layers.
# Layer name -> (pres_min_dbar, pres_max_dbar, display_label)
# "all" uses get_float_history() (no pressure filter) and is included for
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
```

- [ ] **Step 2: Add `build_output_dir()` and `get_ccs_bounds()`**

Append to the same file:

```python
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

    Why registry instead of hardcoding?  If californiav2 bounds change in the
    future, the overlay automatically stays in sync without touching this script.
    """
    reg = get_ebus_registry()["californiav2"]
    return {"lat": reg["lat"], "lon": reg["lon"]}
```

- [ ] **Step 3: Verify imports and constants load cleanly**

```bash
conda run -n ebus-cloud-env python -c "
import sys; sys.path.insert(0, 'ArgoEBUSCloud')
exec(open('ArgoEBUSCloud/09c_ae_depth_aware_float_census.py').read().split('if __name__')[0])
print('LAYERS:', list(LAYERS.keys()))
bounds = get_ccs_bounds()
print('CCS bounds:', bounds)
assert bounds['lat'] == [30.0, 45.0], 'Unexpected lat bounds'
assert bounds['lon'] == [-130.0, -115.0], 'Unexpected lon bounds'
print('PASS')
"
```

Expected output:
```
LAYERS: ['alldepths', 'skin', 'source', 'background']
CCS bounds: {'lat': [30.0, 45.0], 'lon': [-130.0, -115.0]}
PASS
```

- [ ] **Step 4: Commit**

```bash
git add ArgoEBUSCloud/09c_ae_depth_aware_float_census.py
git commit -m "feat: scaffold 09c depth-aware census — constants, output dir, CCS bounds helper"
```

---

## Task 3: Add data fetch and census-building functions

**Files:**
- Modify: `ArgoEBUSCloud/09c_ae_depth_aware_float_census.py`

- [ ] **Step 1: Add `fetch_layer_data()` and `build_census()`**

Append to the script:

```python
# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------

def fetch_layer_data(layer_name, pres_min, pres_max):
    """
    Fetches raw per-dive Argo float positions for one depth layer across all
    years 1999–2025, using the broad "california" domain.

    For the "alldepths" layer (pres_min is None), calls get_float_history()
    with no pressure filter. For all other layers, calls
    get_float_history_by_layer() which adds a pressure constraint so only
    dives that actually reached [pres_min, pres_max] dbar are included.

    Uses the same 5-year FETCH_CHUNKS as script 09 to stay within ERDDAP
    row limits. Chunk failures are caught individually — a single ERDDAP
    timeout does not abort the whole run.

    Returns a single DataFrame with columns:
        platform_number, lat, lon, time, time_days, year (int)
    """
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
    """
    Bins dive positions onto a 5°x5° grid and counts unique float WMO IDs
    (platform_number) per (year, lat_bin, lon_bin) cell.

    Bin centers:
        lat_bin = floor(lat / 5) * 5 + 2.5   e.g. lat=32.1 → bin 32.5
        lon_bin = floor(lon / 5) * 5 + 2.5   e.g. lon=-122.7 → bin -122.5

    We count unique floats, not dive counts. One float with 30 dives in a
    cell represents one independent GP observation location. Unique float
    count is the correct metric for assessing GPR spatial support.

    Returns DataFrame: [year, lat_bin, lon_bin, n_floats]
    """
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
    """
    Saves the census DataFrame for one layer to CSV.

    Filename pattern:
        float_census_depth_aware_{layer_name}_1999_2025.csv

    This is the primary archival output — the PNGs are derived from these CSVs
    and can be regenerated; the CSVs are the ground truth.
    """
    csv_name = f"float_census_depth_aware_{layer_name}_1999_2025.csv"
    csv_path = os.path.join(out_dir, csv_name)
    census.to_csv(csv_path, index=False)
    print(f"[census/{layer_name}] CSV saved → {csv_path}")
    return csv_path
```

- [ ] **Step 2: Smoke-test fetch + census with a tiny date range**

```bash
conda run -n ebus-cloud-env python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'ArgoEBUSCloud')
from ebus_core.ae_utils import get_float_history_by_layer

# Simulate a single small fetch to test census building logic
from ebus_core.ae_utils import get_float_history
df = get_float_history('california', start_date='2015-01-01', end_date='2015-03-31')
df['year'] = df['time'].dt.year

# Inline build_census logic
df['lat_bin'] = (np.floor(df['lat'] / 5.0) * 5.0) + 2.5
df['lon_bin'] = (np.floor(df['lon'] / 5.0) * 5.0) + 2.5
census = (
    df.groupby(['year', 'lat_bin', 'lon_bin'])['platform_number']
    .nunique().reset_index().rename(columns={'platform_number': 'n_floats'})
)
print('Census shape:', census.shape)
print('Max n_floats:', census['n_floats'].max())
assert census['n_floats'].max() > 0, 'Census is empty'
assert set(census.columns) == {'year', 'lat_bin', 'lon_bin', 'n_floats'}
print('PASS')
"
```

Expected: Census has rows, max n_floats > 0, correct columns.

- [ ] **Step 3: Commit**

```bash
git add ArgoEBUSCloud/09c_ae_depth_aware_float_census.py
git commit -m "feat: add fetch_layer_data(), build_census(), save_census_csv() to 09c"
```

---

## Task 4: Add plot functions with CCS bounds overlay

**Files:**
- Modify: `ArgoEBUSCloud/09c_ae_depth_aware_float_census.py`

- [ ] **Step 1: Add `draw_ccs_bounds()` helper and `plot_year()`**

Append to the script:

```python
# ---------------------------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------------------------

def draw_ccs_bounds(ax, ccs_bounds):
    """
    Draws the californiav2 domain boundary as a dashed red rectangle on a
    Cartopy axes. Labels it "CCS Analysis Bounds" in red at the top-left
    corner of the rectangle.

    Why? The californiav2 domain (lon [-130,-115], lat [30,45]) is tighter
    than the broad census domain (lon [-140,-110], lat [25,50]). Floats outside
    this rectangle are excluded from the GPR runs. Overlaying this boundary makes
    visible exactly which census cells are inside vs. outside the GPR domain —
    the key diagnostic for the 2015 source layer regression.

    Inputs:
        ax         - Cartopy GeoAxes to draw on (must be in PlateCarree projection)
        ccs_bounds - dict {"lat": [lat_min, lat_max], "lon": [lon_min, lon_max]}
                     from get_ccs_bounds()
    """
    lat_min, lat_max = ccs_bounds["lat"]
    lon_min, lon_max = ccs_bounds["lon"]

    # Draw the four sides of the rectangle as a polygon boundary.
    # We use ax.plot() with transform=PlateCarree so the line stays on the map.
    box_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    box_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(
        box_lons, box_lats,
        color="red", linewidth=1.5, linestyle="--",
        transform=ccrs.PlateCarree(),
        zorder=200,   # Above land and coastline (zorders 100–101)
    )

    # Label at the top-left corner of the rectangle, offset slightly inward
    # so the text sits inside the box and is legible against the ocean color.
    ax.text(
        lon_min + 0.5, lat_max - 1.5,
        "CCS Analysis Bounds",
        color="red", fontsize=8, fontweight="bold",
        transform=ccrs.PlateCarree(),
        zorder=201,
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
    )


def plot_year(year_census, year, layer_name, display_label, out_dir, ccs_bounds):
    """
    Produces a Cartopy pcolormesh heatmap for one year and one depth layer.

    Inputs:
        year_census   - DataFrame for one year: [lat_bin, lon_bin, n_floats]
        year          - Integer year for title and filename
        layer_name    - Short key (e.g., "source") for filename
        display_label - Human-readable label (e.g., "Source (150–400m)") for title
        out_dir       - Absolute path to output subfolder
        ccs_bounds    - Dict from get_ccs_bounds(), passed to draw_ccs_bounds()

    The pivot step converts tidy (lat_bin, lon_bin, n_floats) to a 2D array.
    Cells with no floats are filled with 0. pcolormesh expects bin EDGES, so
    we derive edges from sorted bin centers by ±2.5° (half the 5° bin width).
    """
    pivot = (
        year_census
        .pivot(index="lat_bin", columns="lon_bin", values="n_floats")
        .sort_index()
        .sort_index(axis=1)
        .fillna(0.0)
    )

    lat_centers = np.array(pivot.index)
    lon_centers = np.array(pivot.columns)
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

    # Overlay CCS Analysis Bounds — the core diagnostic overlay of this script
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
    plt.close(fig)   # Critical: prevents memory accumulation in the year loop
    return out_path


def plot_layer_mean(census, layer_name, display_label, out_dir, ccs_bounds):
    """
    Plots mean float density across all years for one depth layer.

    To get a true mean (not just mean over years-with-floats), we build a full
    year×cell grid by reindexing with all (year, lat_bin, lon_bin) combinations,
    filling missing cells with 0, then averaging over years.

    Saves: float_census_{layer_name}_mean.png
    """
    all_years = census["year"].unique()
    all_lat   = census["lat_bin"].unique()
    all_lon   = census["lon_bin"].unique()

    # Reindex to full grid so empty cells contribute zeros to the mean
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
    cbar.set_label(f"Mean Unique Floats per 5°×5° Cell (1999–2025)", fontsize=11)

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
```

- [ ] **Step 2: Add `plot_all_years_for_layer()`**

Append to the script:

```python
def plot_all_years_for_layer(census, layer_name, display_label, out_dir, ccs_bounds):
    """
    Iterates over every year present in the census and calls plot_year().

    Prints one progress line per year so the user can track the ~26-iteration loop.
    """
    years = sorted(census["year"].unique())
    print(f"[census/{layer_name}] Generating {len(years)} per-year PNGs ...")
    for year in years:
        year_df = census[census["year"] == year][["lat_bin", "lon_bin", "n_floats"]].copy()
        path = plot_year(year_df, year, layer_name, display_label, out_dir, ccs_bounds)
        print(f"[census/{layer_name}]   {year} → {os.path.basename(path)}")
    print(f"[census/{layer_name}] Per-year PNGs done.")
```

- [ ] **Step 3: Smoke-test the plot functions with synthetic data**

```bash
conda run -n ebus-cloud-env python -c "
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, 'ArgoEBUSCloud')
exec(open('ArgoEBUSCloud/09c_ae_depth_aware_float_census.py').read().split('if __name__')[0])

# Synthetic 1-year census with a few cells
ccs = get_ccs_bounds()
fake = pd.DataFrame({
    'year':     [2015, 2015, 2015],
    'lat_bin':  [32.5, 37.5, 42.5],
    'lon_bin':  [-122.5, -127.5, -122.5],
    'n_floats': [3, 5, 2],
})
import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    path = plot_year(fake, 2015, 'source', 'Source (150-400m)', tmp, ccs)
    assert os.path.exists(path), 'PNG not created'
    print('plot_year OK:', path)
    path2 = plot_layer_mean(fake, 'source', 'Source (150-400m)', tmp, ccs)
    assert os.path.exists(path2), 'Mean PNG not created'
    print('plot_layer_mean OK:', path2)
print('PASS')
"
```

Expected: Both PNGs created, no crash.

- [ ] **Step 4: Commit**

```bash
git add ArgoEBUSCloud/09c_ae_depth_aware_float_census.py
git commit -m "feat: add plot functions with CCS bounds overlay to 09c"
```

---

## Task 5: Wire up `__main__` and run end-to-end

**Files:**
- Modify: `ArgoEBUSCloud/09c_ae_depth_aware_float_census.py`

- [ ] **Step 1: Add `__main__` block**

Append to the script:

```python
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Setup: output directory + CCS domain bounds for plot overlays
    out_dir    = build_output_dir()
    ccs_bounds = get_ccs_bounds()

    # Process each layer in order. "alldepths" first (matches script 09 baseline),
    # then the three scientific layers. This order also means if the run is
    # interrupted, the all-depths and skin results are already saved before
    # the slower source/background fetches.
    for layer_name, (pres_min, pres_max, display_label) in LAYERS.items():
        print(f"\n[census] === Layer: {layer_name} ({display_label}) ===")

        # Fetch raw dives from ERDDAP for this layer
        raw = fetch_layer_data(layer_name, pres_min, pres_max)

        # Bin to 5°x5° unique-float census
        census = build_census(raw)

        # Save CSV archive
        save_census_csv(census, layer_name, out_dir)

        # Per-year PNGs
        plot_all_years_for_layer(census, layer_name, display_label, out_dir, ccs_bounds)

        # Per-layer mean PNG (all years averaged)
        plot_layer_mean(census, layer_name, display_label, out_dir, ccs_bounds)

    print("\n[census] All layers complete. Done.")
```

- [ ] **Step 2: Run the full script**

```bash
conda run -n ebus-cloud-env python ArgoEBUSCloud/09c_ae_depth_aware_float_census.py
```

Expected terminal output (abbreviated):
```
[census] Output directory: .../AEResults/aeplots/float_census_depth_aware
[census] === Layer: alldepths (All Depths) ===
[census/alldepths] Fetching 1999-01-01 → 2003-12-31 ...
[census/alldepths]   Got X,XXX dives.
...
[census/alldepths] Per-year PNGs done.
[census/alldepths] Mean map saved → ...
[census] === Layer: skin (Skin (0–100m)) ===
...
[census] All layers complete. Done.
```

Full run takes roughly 3–8 minutes (18 ERDDAP fetches at ~10s each + plotting).

- [ ] **Step 3: Verify output**

```bash
# Check CSV count (expect 4)
ls AEResults/aeplots/float_census_depth_aware/*.csv | wc -l

# Check PNG count (expect ~107)
ls AEResults/aeplots/float_census_depth_aware/*.png | wc -l

# Spot check: 2015 source layer PNG exists
ls AEResults/aeplots/float_census_depth_aware/float_census_source_2015.png

# Spot check: all 3 mean PNGs exist
ls AEResults/aeplots/float_census_depth_aware/float_census_*_mean.png
```

Expected: 4 CSVs, ~107 PNGs, all spot-checks pass.

- [ ] **Step 4: Visual check on the key diagnostic figure**

Open `AEResults/aeplots/float_census_depth_aware/float_census_source_2015.png` and verify:
- Dashed red rectangle is visible, covering roughly lon [-130, -115], lat [30, 45]
- The label "CCS Analysis Bounds" appears near the top-left of the rectangle
- The heatmap shows noticeably fewer floats inside the rectangle than the `float_census_alldepths_2015.png` — or at least clustered differently
- Color scale runs 0–8

- [ ] **Step 5: Final commit**

```bash
git add ArgoEBUSCloud/09c_ae_depth_aware_float_census.py
git commit -m "feat: complete 09c_ae_depth_aware_float_census.py with main loop and all layers"
```

---

## Verification Checklist

End-to-end test:
```bash
conda run -n ebus-cloud-env python ArgoEBUSCloud/09c_ae_depth_aware_float_census.py
```

Pass criteria:
- [ ] Script exits with `[census] All layers complete. Done.` and no unhandled exceptions
- [ ] `AEResults/aeplots/float_census_depth_aware/` contains exactly 4 CSVs
- [ ] PNG count ≥ 100 (26 years × 4 layers + 3 mean maps = 107 expected; ERDDAP chunk failures may reduce this slightly)
- [ ] `float_census_source_2015.png` shows visibly sparser coverage than `float_census_alldepths_2015.png`
- [ ] Dashed red "CCS Analysis Bounds" rectangle appears on all figures at lon [-130, -115] / lat [30, 45]
- [ ] `float_census_source_mean.png` shows structurally sparser coverage than `float_census_skin_mean.png`
