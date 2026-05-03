import numpy as np
import os

def get_ebus_registry():
    return {
        # THE OLD DEFINITION (For small-scale testing/debugging)
        "california_testbox": {
            "lat": [30.0, 35.0],
            "lon": [-125.0, -120.0],
            "time": ["2015-01-01", "2015-12-31"], # Restoring the time key
            "s3_bucket": "argo-ebus-project-data-abm"
        },
        # THE FULL CCS (For actual scientific analysis)
        # Preserved as-is: this is the 140W dataset used in all 2015 runs.
        # Do NOT alter these bounds; existing S3 artifacts reference this exact window.
        "california": {
            "lat": [25.0, 50.0],
            "lon": [-140.0, -110.0],
            "time": ["2015-01-01", "2015-12-31"],
            "s3_bucket": "argo-ebus-project-data-abm"
        },
        # CCS window aligned with Frontiers 2024 paper spatial domain.
        # Tighter longitude (130W–115W) cuts offshore Pacific and focuses on
        # the coastal upwelling zone and California Undercurrent corridor.
        "californiav2": {
            "lat": [30.0, 45.0],
            "lon": [-130.0, -115.0],
            "time": ["2015-01-01", "2015-12-31"],
            "s3_bucket": "argo-ebus-project-data-abm"
        },
        # californiav3: optimized spatial bounds based on the 26-year float census.
        # Lat [30, 48] captures the high-density core of the CCS.
        # Lon [-135, -115] provides a wider offshore buffer to stabilize the GPR
        # in the Source Layer (150-400m) where data is sparser than at the surface.
        "californiav3": {
            "lat": [30.0, 48.0],
            "lon": [-135.0, -115.0],
            "time": ["2015-01-01", "2015-12-31"],
            "s3_bucket": "argo-ebus-project-data-abm"
        },
        # Humboldt Current System — bounds corrected to Frontiers 2024 domain.
        # Previous lat [-45,0] was too broad; [-35,-5] isolates the active CUS.
        "humboldt": {
            "lat": [-35.0, -5.0],
            "lon": [-85.0, -70.0],
            "time": ["2015-01-01", "2015-12-31"],
            "s3_bucket": "argo-ebus-project-data-abm"
        },
        # Canary Current System — bounds corrected to Frontiers 2024 domain.
        # Previous lat [10,45] too broad; [15,35] tightens to core upwelling band.
        # Previous lon [-30,-5] too wide; [-25,-10] removes open-ocean bias.
        "canary": {
            "lat": [15.0, 35.0],
            "lon": [-25.0, -10.0],
            "time": ["2015-01-01", "2015-12-31"],
            "s3_bucket": "argo-ebus-project-data-abm"
        },
        # Benguela Current System — bounds corrected to Frontiers 2024 domain.
        # Previous lat [-35,-10] slightly off; [-35,-15] removes equatorial noise.
        # Longitude [5,20] unchanged.
        "benguela": {
            "lat": [-35.0, -15.0],
            "lon": [5.0, 20.0],
            "time": ["2015-01-01", "2015-12-31"],
            "s3_bucket": "argo-ebus-project-data-abm"
        }
    }

def get_vertical_layers():
    # Returns the canonical "Vertical Sandwich" depth layer definitions used
    # throughout the stealth warming analysis.
    #
    # Response  (0–100m):   fast atmospheric response layer — SST proxy; dominated
    #                       by air-sea heat exchange and mixed-layer dynamics.
    # Source    (150–400m): Ekman upwelling source water — where stealth heat hides.
    #                       This is the layer expected to warm fastest if the
    #                       California Undercurrent is transporting anomalous heat.
    # Background (500–1000m): deep ocean baseline; slow thermocline variability
    #                         used as a reference to isolate the Source signal.
    #
    # Key diagnostic: Source warming rate > Background warming rate → refugia signal.
    return {
        "Response":   [0, 100],
        "Source":     [150, 400],
        "Background": [500, 1000],
    }


def get_ae_config(region="california", lat_step=0.5, lon_step=0.5, time_step=30.0, 
                  depth_range=(0, 100), start_date=None, end_date=None):
    """
    Fetch configuration for a target study site with dynamic resolutions and depth.
    Standardizes naming for S3 files and Coiled clusters.
    """
    registry = get_ebus_registry()
    if region not in registry:
        raise ValueError(f"Region '{region}' not found.")
    
    # Use .copy() to avoid modifying the global registry dictionary
    config = registry[region].copy()
    
    # Dates
    config["start_date"] = start_date if start_date else config["time"][0]
    config["end_date"] = end_date if end_date else config["time"][1]
    
    # Resolutions and Depth
    config["resolutions"] = {
        "lat_step": lat_step,
        "lon_step": lon_step,
        "time_step": time_step
    }
    config["depth_range"] = depth_range
    
    # --- FILENAME GENERATION (run_id) ---
    start_clean = config["start_date"].replace("-", "")
    end_clean = config["end_date"].replace("-", "")
    date_str = f"{start_clean}_{end_clean}"
    
    # Safe naming: periods to underscores for S3/Coiled compatibility
    lat_safe = str(lat_step).replace(".", "_")
    lon_safe = str(lon_step).replace(".", "_")
    time_safe = str(time_step).replace(".", "_")
    depth_str = f"d{depth_range[0]}_{depth_range[1]}"

    # Example: california_20150101_20151231_res0_5x0_5_t30_0_d0_700
    config["run_id"] = f"{region}_{date_str}_res{lat_safe}x{lon_safe}_t{time_safe}_{depth_str}"
    
    config["paths"] = get_project_paths()
    return config


def get_project_paths():
    """
    Calculates paths to AEResults relative to the ArgoEBUSCloud directory.
    Assumes structure: /ArgoEBUSAnalysis/[ArgoEBUSCloud, AEResults]
    """
    # Get the directory where this utils file lives (ebus_core)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to reach /ArgoEBUSAnalysis/
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    
    results_dir = os.path.join(project_root, "AEResults")
    
    return {
        "root": project_root,
        "results": results_dir,
        "plots": os.path.join(results_dir, "aeplots"),
        "models": os.path.join(results_dir, "aemodels")
    }

def ensure_ae_dirs():
    """
    Guarantees project directory tree exists.
    AEResults lives at ArgoEBUSAnalysis/AEResults/, one level above ArgoEBUSCloud/.
    Derives the absolute path from this file's location so it works regardless of cwd.
    """
    import os
    # This file is at ArgoEBUSAnalysis/ArgoEBUSCloud/ebus_core/ae_utils.py.
    # Two levels up (..) lands at ArgoEBUSAnalysis/.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(project_root, "AEResults")
    for sub in ["aeplots", "aedata", "aelogs"]:
        path = os.path.join(base_dir, sub)
        os.makedirs(path, exist_ok=True)

def get_float_history(region="california", start_date=None, end_date=None):
    # Retrieve per-dive Argo float positions for a region and date range.
    #
    # The binned OHC parquet (Script 02) only retains one platform_number per bin —
    # individual dive positions are lost after aggregation. This function queries
    # ERDDAP directly to recover those raw positions, collapsing per-pressure-level
    # rows to one row per dive using ERDDAP's &distinct() modifier.
    #
    # Inputs:
    #   region     - Key into get_ebus_registry(). Determines lat/lon spatial window
    #                and default date range.
    #   start_date - ISO string "YYYY-MM-DD". Falls back to registry time[0] if None.
    #   end_date   - ISO string "YYYY-MM-DD". Falls back to registry time[1] if None.
    #
    # Output: DataFrame with columns:
    #   platform_number (str)  - Argo float WMO ID
    #   lat (float)            - Dive latitude, degrees N
    #   lon (float)            - Dive longitude, degrees E
    #   time (datetime, UTC)   - Dive timestamp
    #   time_days (float)      - Days since 1999-01-01 (matches OHC parquet baseline)
    #
    # Expect ~14,000 rows for California 2015 (2–10 sec ERDDAP latency).
    import pandas as pd

    registry = get_ebus_registry()
    if region not in registry:
        raise ValueError(f"Region '{region}' not found in registry. Known regions: {list(registry.keys())}")

    reg = registry[region]

    # Resolve dates from arguments or registry defaults
    t_start = start_date if start_date else reg["time"][0]
    t_end   = end_date   if end_date   else reg["time"][1]

    lat_min, lat_max = reg["lat"]
    lon_min, lon_max = reg["lon"]

    # Build ERDDAP URL.
    # We request only the four columns we need (platform_number, time, latitude, longitude).
    # &distinct() tells ERDDAP to collapse the many per-pressure-level rows down to a
    # single row per unique (float, time) dive — this is the key reduction that makes
    # the download tractable (~14k rows vs ~500k raw).
    #
    # The domain moved from www.ifremer.fr/erddap to erddap.ifremer.fr/erddap.
    # We use requests instead of pd.read_csv(url) directly because Tomcat 11 on the
    # new host is strict about RFC 3986 — the < and > in comparison filters must be
    # percent-encoded (%3C, %3E).  requests.get() with a params dict handles this;
    # urllib (used internally by pandas) does not encode them.
    import io
    import requests

    base_url = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"

    # Build query string manually so we control encoding.
    # requests.get() percent-encodes the entire params value, but ERDDAP constraint
    # syntax uses operators inside the value (e.g. latitude>=25.0), so we construct
    # the full query string ourselves and pass it pre-built.
    raw_query = (
        f"platform_number,time,latitude,longitude"
        f"&latitude>={lat_min}&latitude<={lat_max}"
        f"&longitude>={lon_min}&longitude<={lon_max}"
        f"&time>={t_start}T00:00:00Z"
        f"&time<={t_end}T23:59:59Z"
        f"&distinct()"
    )

    # Percent-encode only the characters Tomcat 11 rejects: < and >
    # We keep & = ( ) as-is since ERDDAP needs them unencoded for query parsing.
    encoded_query = raw_query.replace("<", "%3C").replace(">", "%3E")
    erddap_url = f"{base_url}?{encoded_query}"

    try:
        resp = requests.get(erddap_url, timeout=60)
        resp.raise_for_status()
        # skiprows=[1] drops the ERDDAP units row (second CSV line) — same pattern as Script 02
        df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
    except Exception as e:
        raise RuntimeError(
            f"ERDDAP query failed for region '{region}' ({t_start} to {t_end}).\n"
            f"URL attempted: {erddap_url}\n"
            f"Original error: {e}"
        )

    # Parse the time column to UTC-aware datetimes so downstream code can use .dt accessors
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Compute time_days to match the baseline used in the OHC parquet (Script 02)
    baseline = pd.Timestamp("1999-01-01", tz="UTC")
    df["time_days"] = (df["time"] - baseline).dt.total_seconds() / 86400.0

    # Rename to match the rest of the codebase schema
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    # Enforce column order for readability
    return df[["platform_number", "lat", "lon", "time", "time_days"]]


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
        # timeout=120: deep-layer queries scan the full raw dataset before filtering;
        # 60s is too short when the server is under load.
        resp = requests.get(erddap_url, timeout=120)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
    except Exception as e:
        raise RuntimeError(
            f"ERDDAP layer query failed for region '{region}' "
            f"pres [{pres_min}, {pres_max}] ({t_start} to {t_end}).\n"
            f"URL attempted: {erddap_url}\n"
            f"Original error: {e}"
        )

    # Guard: ERDDAP returned a valid response but no qualifying rows.
    # Without this check, pd.to_datetime(df["time"]) raises KeyError when the
    # "time" column is absent in a zero-row (or header-only) response.
    # Return an empty typed DataFrame so callers can detect this via len(df)==0.
    if df.empty:
        return pd.DataFrame(columns=["platform_number", "lat", "lon", "time", "time_days"])

    df["time"] = pd.to_datetime(df["time"], utc=True)
    baseline = pd.Timestamp("1999-01-01", tz="UTC")
    df["time_days"] = (df["time"] - baseline).dt.total_seconds() / 86400.0
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    # Defensive dedup: &distinct() should already guarantee one row per
    # (platform_number, time) pair, but guard explicitly against any future
    # ERDDAP server behavior change.
    df = df.drop_duplicates(subset=["platform_number", "time"])

    return df[["platform_number", "lat", "lon", "time", "time_days"]]


def fmt_dec(x: float) -> str:
    # Convert a float to a filesystem-safe string by replacing '.' with '_'.
    # Used to build canonical run_id strings and S3 parquet paths.
    # Examples: 0.5 -> '0_5', 10.0 -> '10_0', 0.25 -> '0_25'
    # WHY: dots in directory/filename components confuse shell globs and some
    # path-parsing utilities. All AEResults/ paths embed underscored floats.
    s = f"{x:g}"
    if "." not in s:
        s = s + ".0"
    return s.replace(".", "_")


def calculate_bin(value, step):
    """Generic binning helper."""
    return np.floor(value / step) * step

def get_coastline_points(resolution='50m'):
    """
    Extracts lon/lat points from Cartopy's coastline feature at a given resolution.
    Returns a (N, 2) numpy array of [longitude, latitude].
    """
    import cartopy.feature as cfeature
    from shapely.geometry import MultiLineString, LineString

    # Fetch the coastline feature
    coastline = cfeature.COASTLINE.with_scale(resolution)
    geoms = list(coastline.geometries())

    points = []
    for geom in geoms:
        if isinstance(geom, MultiLineString):
            for line in geom.geoms:
                points.extend(list(line.coords))
        elif isinstance(geom, LineString):
            points.extend(list(geom.coords))

    return np.array(points)

def calculate_dist_to_coast(lats, lons, resolution='10m'):
    """
    Calculates the minimum great-circle distance (km) from each (lat, lon) point
    to the nearest coastline vertex.

    Uses a KDTree built in 3D unit-vector (ECEF) space to avoid the angular-distance
    bias that arises from naive KDTree on raw (lon, lat) degree coordinates.
    At CCS latitudes (30–48°N), 1° lon ≈ 85 km but 1° lat ≈ 111 km, so a degree-space
    tree systematically returns the wrong nearest vertex. Unit-vector space is isotropic
    on the sphere, so the tree result is unbiased. One exact Haversine computation is
    then run on the single nearest candidate to obtain km distance.

    resolution: Natural Earth coastline detail — '10m' (~1-2 km vertex spacing),
                '50m' (~10-20 km), '110m' (~100 km). Use '10m' for ML features
                in coastal-gradient-sensitive applications like upwelling analysis.
    """
    from scipy.spatial import KDTree

    # 1. Load coastline vertices as (N, 2) array of [lon, lat] in degrees.
    coast_pts = get_coastline_points(resolution)

    # 2. Convert coastline vertices to 3D unit vectors on the unit sphere.
    # ECEF-style: x = cos(lat)*cos(lon), y = cos(lat)*sin(lon), z = sin(lat).
    # Nearest neighbor in this space = nearest neighbor on the sphere surface,
    # with no distortion from the irregular degree-to-km mapping.
    coast_lons_r = np.radians(coast_pts[:, 0])
    coast_lats_r = np.radians(coast_pts[:, 1])
    coast_xyz = np.column_stack([
        np.cos(coast_lats_r) * np.cos(coast_lons_r),
        np.cos(coast_lats_r) * np.sin(coast_lons_r),
        np.sin(coast_lats_r)
    ])
    tree = KDTree(coast_xyz)

    # 3. Convert query points to the same 3D unit-vector space and query the tree.
    lons_r = np.radians(lons)
    lats_r = np.radians(lats)
    query_xyz = np.column_stack([
        np.cos(lats_r) * np.cos(lons_r),
        np.cos(lats_r) * np.sin(lons_r),
        np.sin(lats_r)
    ])
    _, dist_indices = tree.query(query_xyz)

    # 4. Retrieve the nearest coastline vertex (lon, lat) for each query point.
    nearest_coast_pts = coast_pts[dist_indices]

    # 5. Compute exact Haversine distance (km) between each query point and its
    # nearest coastline vertex. This is fast: one vectorized pass over N pairs.
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c

    dists_km = haversine(lons, lats, nearest_coast_pts[:, 0], nearest_coast_pts[:, 1])

    return dists_km