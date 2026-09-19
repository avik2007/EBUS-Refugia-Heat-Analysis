"""
=============================================================================
VERSION 2.2: High-Resolution Granularity & Depth-Aware Labeling
=============================================================================
1. 0.5 Degree Resolution: Halves the bin size to better resolve coastal gradients.
2. Depth Awareness: Explicitly passes depth_range to the physics engine and 
   includes the depth in the S3 filename (run_id).
3. Coiled Infrastructure: Dynamic provisioning for Dask distributed workloads.
=============================================================================
"""
import coiled
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import warnings
import gsw
import os

# Import your custom ocean physics and regional utilities
from ebus_core.argoebus_thermodynamics import estimate_ohc_from_raw_bins
from ebus_core.ae_utils import get_ae_config

# --- GLOBAL CLOUD SETTINGS ---
cloud_provider = "aws"
compute_region = "us-east-1" 

def run_cloud_pipeline(region="california", lat_step=0.5, lon_step=0.5, time_step=30.0, 
                       depth_range=(0, 100), n_workers=3,
                       start_date=None, end_date=None):
    """
    API-based ingestion pipeline using Ifremer ERDDAP.
    Dynamically requests temporal and spatial bounds and applies OHC physics.
    """
    
    # --- 1. CONFIGURATION (The Depth-Aware Step) ---
    config = get_ae_config(
        region, 
        lat_step=lat_step, 
        lon_step=lon_step, 
        time_step=time_step,
        depth_range=depth_range,
        # ISO "YYYY-MM-DD" strings select the ingest year; None keeps the
        # registry's default window (legacy behaviour for direct script runs).
        start_date=start_date,
        end_date=end_date
    )

    # Destination path automatically includes the region, dates, resolution, and depth
    output_s3 = f"s3://{config['s3_bucket']}/{config['run_id']}.parquet"

    print(f"☁️ Step 1: Provisioning {cloud_provider.upper()} Infrastructure for {config['run_id']}...")
    
    # --- 2. CLUSTER SETUP ---
    cluster = coiled.Cluster(
        name=f"ae-{config['run_id'].replace('_', '-')[:30]}", # Coiled name length limit
        n_workers=n_workers,
        region=compute_region,
        worker_vm_types=["m5.large", "m4.large", "t3.large"], 
        spot_policy="spot_with_fallback",
    )
    
    client = Client(cluster)
    print(f"✅ Cloud Cluster Ready! Dashboard: {client.dashboard_link}")

    # Wrap everything after cluster creation in try/finally so the cluster is
    # always shut down even if ERDDAP, Dask, or S3 raises before the write block.
    try:
        # --- 3. DYNAMIC API QUERY ---
        print(f"\n🗺️ Step 2: Requesting {region.upper()} Data ({config['start_date']} to {config['end_date']})...")

        # Use %3E/%3C for > and < — fsspec treats bare > and < as glob characters
        # and also fails to follow the www.ifremer.fr → erddap.ifremer.fr redirect
        # when the URL contains unencoded comparison operators.
        erddap_url = (
            f"https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv?"
            f"platform_number,time,latitude,longitude,pres,temp,psal"
            f"&latitude%3E={config['lat'][0]}&latitude%3C={config['lat'][1]}"
            f"&longitude%3E={config['lon'][0]}&longitude%3C={config['lon'][1]}"
            f"&time%3E={config['start_date']}T00:00:00Z"
            f"&time%3C={config['end_date']}T23:59:59Z"
        )

        # Read CSV stream from ERDDAP
        ddf = dd.read_csv(erddap_url, skiprows=[1], blocksize=None)
        ddf = ddf.repartition(npartitions=n_workers * 4)

        # --- 4. DATA CLEANING & DEPTH CONVERSION ---
        print("🌉 Step 3: Formatting and Converting Pressure to Depth...")

        ddf = ddf.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

        # Calculate exact Depth from Pressure using TEOS-10
        ddf['depth'] = gsw.z_from_p(ddf['pres'], ddf['lat']) * -1

        # Format Datetime and baseline for 'time_days'
        ddf['time'] = dd.to_datetime(ddf['time'], utc=True)
        baseline = pd.Timestamp('1999-01-01', tz='UTC')
        ddf['time_days'] = (ddf['time'] - baseline).dt.total_seconds() / 86400

        # --- 5. DISTRIBUTED PHYSICS ---
        res = config["resolutions"]
        d_min, d_max = config["depth_range"]

        print(f"🚀 Step 4: Distributing Physics (Depth: {d_min}-{d_max}m, Res: {res['lat_step']}x{res['lon_step']})...")

        # Pre-warm the Cartopy Natural Earth coastline shapefile on every worker
        # before map_partitions fires. Workers start with an empty cache; if the
        # download happens concurrently across many workers the shapefile can be
        # written simultaneously and corrupted (struct.error on unpack). Running
        # client.run() serialises the download — one call per worker, fully
        # resolved before any compute task reads the shapefile.
        def _warm_cartopy_cache():
            from ebus_core.ae_utils import get_coastline_points
            get_coastline_points('10m')  # populates Cartopy's local cache

        print("    Pre-warming Cartopy coastline cache on workers...")
        client.run(_warm_cartopy_cache)
        print("    Cache ready.")

        # Define meta for Dask output schema
        meta = pd.DataFrame({
            'time_bin': pd.Series(dtype='float64'),
            'lat_bin': pd.Series(dtype='float64'),
            'lon_bin': pd.Series(dtype='float64'),
            'ohc': pd.Series(dtype='float64'),
            'ohc_per_m': pd.Series(dtype='float64'),
            'n_raw_points': pd.Series(dtype='int64'),
            'platform_number': pd.Series(dtype='str'),
            'dist_to_coast_km': pd.Series(dtype='float64')
        })

        # Apply physics function across cluster
        ddf_binned = ddf.map_partitions(
            estimate_ohc_from_raw_bins,
            resolution_lat=res['lat_step'],
            resolution_lon=res['lon_step'],
            resolution_time_days=res['time_step'],
            depth_min=d_min,  # CRITICAL: Respecting chosen depth
            depth_max=d_max,  # CRITICAL: Respecting chosen depth
            meta=meta
        )

        # --- 6. EXECUTION ---
        print(f"\n💾 Step 5: Computing and Saving to Data Lake...")
        ddf_binned.to_parquet(output_s3, write_index=False)
        print(f"\n🎉 SUCCESS! File saved as: {config['run_id']}.parquet")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise
    finally:
        client.close()
        cluster.shutdown()

# MLOps runner seam. runner.py calls run_ingestion_pipeline(**dispatch_kwargs) where
# dispatch_kwargs includes date_start, date_end, worker_region, s3_bucket — fields
# controlled by the registry inside run_cloud_pipeline, not accepted as params there.
# This wrapper absorbs the extras so run_cloud_pipeline receives only what it handles.
def run_ingestion_pipeline(**kwargs):
    run_cloud_pipeline(
        region=kwargs["region"],
        lat_step=kwargs.get("lat_step", 0.5),
        lon_step=kwargs.get("lon_step", 0.5),
        time_step=kwargs.get("time_step", 30.0),
        depth_range=kwargs.get("depth_range", (0, 100)),
        n_workers=kwargs.get("n_workers", 3),
        # Runner passes datetime.date objects; get_ae_config wants ISO strings.
        # Without this the config's year was silently replaced by the registry
        # default (2015) while run_id/manifest still claimed the requested year.
        start_date=kwargs["date_start"].isoformat() if kwargs.get("date_start") else None,
        end_date=kwargs["date_end"].isoformat() if kwargs.get("date_end") else None,
    )
    return {}

if __name__ == "__main__":
    # --- CANONICAL FX2 CONFIGURATION ---
    # Run all three Vertical Sandwich layers for the californiav2 domain with
    # 10-day temporal bins. This is the FX2 fix for the structural aliasing
    # problem identified in the T1/T2 experiments: setting time_step=10 makes
    # the bin width equal to the step_size_days used in the GPR analysis, so
    # there are no duplicate or partially-overlapping window observations.
    #
    # Layers run sequentially to avoid simultaneous Coiled cluster cost.
    # Each layer provisions its own cluster and shuts it down after writing.
    #
    # Expected S3 outputs:
    #   californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100.parquet
    #   californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400.parquet
    #   californiav2_20150101_20151231_res0_5x0_5_t10_0_d500_1000.parquet

    COMMON = dict(region="californiav2", lat_step=0.5, lon_step=0.5,
                  time_step=10.0, n_workers=3)

    print("=" * 60)
    print("SKIN LAYER (0-100m)")
    print("=" * 60)
    run_cloud_pipeline(**COMMON, depth_range=(0, 100))

    print("=" * 60)
    print("SOURCE LAYER (150-400m)")
    print("=" * 60)
    run_cloud_pipeline(**COMMON, depth_range=(150, 400))

    print("=" * 60)
    print("BACKGROUND LAYER (500-1000m)")
    print("=" * 60)
    run_cloud_pipeline(**COMMON, depth_range=(500, 1000))

