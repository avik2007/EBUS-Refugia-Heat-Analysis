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
        'n_raw_points': pd.Series(dtype='int64')
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
    # --- PRODUCTION CONFIGURATION ---
    # We are moving to 0.5 degree granularity to solve the coastal error issues.
    run_cloud_pipeline(
        region="california", 
        lat_step=0.5,    
        lon_step=0.5,    
        time_step=30.0,  
        depth_range=(0, 100), # Standard research depth
        n_workers=3
    )
"""
import coiled
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import warnings
import gsw

# Import your custom ocean physics and regional utilities
from ebus_core.argoebus_thermodynamics import estimate_ohc_from_raw_bins
from ebus_core.ae_utils import get_ae_config

def run_cloud_pipeline(
    region="california",
    lat_step=1.0, 
    lon_step=1.0, 
    time_step=30.0,
    cloud_provider='aws',
    compute_region='eu-west-3', # Default to Paris for Ifremer proximity
    n_workers=10
):
    
    # API-based ingestion pipeline using Ifremer ERDDAP.
    # Ideal for smaller, targeted annual runs (e.g., 2015 analysis).
    
    
    # --- 1. CONFIGURATION ---
    # Fetch the regional bounds and bucket from our source of truth
    config = get_ae_config(region, lat_step=lat_step, lon_step=lon_step, time_step=time_step)
    res = config['resolutions']
    
    # Destination path is now dynamic based on the region's bucket and the run_id
    output_s3 = f"s3://{config['s3_bucket']}/{config['run_id']}.parquet"

    print(f"☁️ Step 1: Provisioning {cloud_provider.upper()} Infrastructure for {config['run_id']}...")
    
    # --- 2. CLUSTER SETUP ---
    cluster = coiled.Cluster(
        name=f"ae-erddap-{config['run_id']}",
        backend=cloud_provider,
        n_workers=n_workers,
        region=compute_region,
        worker_vm_types=["m5.large", "m4.large", "t3.large"], 
        spot_policy="spot_with_fallback",
    )
    
    client = Client(cluster)
    print(f"✅ Cloud Cluster Ready! Dashboard: {client.dashboard_link}")

    # --- 3. DYNAMIC API QUERY ---
    print(f"\n🗺️ Step 2: Ingesting Ocean Data for {region.upper()}...")
    
    # We inject the registry bounds directly into the ERDDAP URL
    # This ensures the API only sends us the data we actually need.
    erddap_url = (
        f"https://www.ifremer.fr/erddap/tabledap/ArgoFloats.csv?"
        f"time,latitude,longitude,pres,temp,psal"
        f"&latitude>={config['lat'][0]}&latitude<={config['lat'][1]}"
        f"&longitude>={config['lon'][0]}&longitude<={config['lon'][1]}"
        f"&time>=2015-01-01T00:00:00Z&time<=2015-12-31T23:59:59Z"
    )
    
    # Read the CSV stream into a Dask DataFrame
    ddf = dd.read_csv(erddap_url, skiprows=[1]) # skip the units row

    # --- 4. DATA CLEANING ---
    print("🌉 Step 3: Formatting and Cleaning Data...")
    ddf['time'] = dd.to_datetime(ddf['time'])
    # Calculate days since your 1999 baseline
    ddf['days_since_1999'] = (ddf['time'] - pd.Timestamp('1999-01-01')).dt.total_seconds() / 86400

    # --- 5. DISTRIBUTED PHYSICS ---
    print(f"🚀 Step 4: Distributing Physics Engine (Res: {lat_step}x{lon_step})...")
    
    meta = pd.DataFrame({
        'time_bin': pd.Series(dtype='float64'),
        'lat_bin': pd.Series(dtype='float64'),
        'lon_bin': pd.Series(dtype='float64'),
        'ohc': pd.Series(dtype='float64'),
        'ohc_per_m': pd.Series(dtype='float64'),
        'n_raw_points': pd.Series(dtype='int64')
    })

    ddf_binned = ddf.map_partitions(
        estimate_ohc_from_raw_bins, 
        resolution_lat=res['lat_step'], 
        resolution_lon=res['lon_step'], 
        resolution_time_days=res['time_step'],
        meta=meta
    )

    # --- 6. EXECUTION ---
    print(f"\n💾 Step 5 & 6: Computing and Saving to {output_s3}...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ddf_binned.to_parquet(output_s3, write_index=False)
    
    print(f"\n🎉 SUCCESS! Data saved to: {output_s3}")
    
    client.close()
    cluster.shutdown()

if __name__ == "__main__":
    # Example: Run a higher-res 0.5 degree analysis for the California system
    run_cloud_pipeline(region="california", lat_step=1.0, lon_step=1.0)
"""

