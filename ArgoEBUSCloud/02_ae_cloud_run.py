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
                       depth_range=(0, 100), n_workers=3):
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
        depth_range=depth_range
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

    # --- 3. DYNAMIC API QUERY ---
    print(f"\n🗺️ Step 2: Requesting {region.upper()} Data ({config['start_date']} to {config['end_date']})...")
    
    erddap_url = (
        f"https://www.ifremer.fr/erddap/tabledap/ArgoFloats.csv?"
        f"platform_number,time,latitude,longitude,pres,temp,psal"
        f"&latitude>={config['lat'][0]}&latitude<={config['lat'][1]}"
        f"&longitude>={config['lon'][0]}&longitude<={config['lon'][1]}"
        f"&time>={config['start_date']}T00:00:00Z"
        f"&time<={config['end_date']}T23:59:59Z"
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
    
    # Define meta for Dask output schema
    meta = pd.DataFrame({
        'time_bin': pd.Series(dtype='float64'),
        'lat_bin': pd.Series(dtype='float64'),
        'lon_bin': pd.Series(dtype='float64'),
        'ohc': pd.Series(dtype='float64'),
        'ohc_per_m': pd.Series(dtype='float64'),
        'n_raw_points': pd.Series(dtype='int64'),
        'platform_number': pd.Series(dtype='str')
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
    
    try:
        ddf_binned.to_parquet(output_s3, write_index=False)
        print(f"\n🎉 SUCCESS! File saved as: {config['run_id']}.parquet")
    except Exception as e:
        print(f"❌ ERROR writing to S3: {e}")
    finally:
        client.close()
        cluster.shutdown()

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

