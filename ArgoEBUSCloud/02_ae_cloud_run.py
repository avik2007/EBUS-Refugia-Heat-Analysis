"""
=============================================================================
VERSION 2.1 UPDATE: Global Scalability & Exact Temporal Slicing
=============================================================================
Previous Version: 
- Hard-coded to the California region for the year 2015.
- Used static S3 bucket paths and static ERDDAP/Zarr data queries.
- Resulted in overwriting data if run multiple times.

Current Version: 
1. Dynamic Registry: Uses `ebus_core.ae_utils` to automatically fetch 
   regional bounds (Lat/Lon) and specific event windows (exact start/end dates).
2. Temporal Slicing: Directly slices the time dimension at the source 
   (Xarray/ERDDAP) to drastically reduce data transfer and processing time.
3. Automated Precise Naming: Output files now dynamically include the region, 
   exact dates, and resolution in the filename 
   (e.g., `california_20150101_20151231_res1.0x1.0.parquet`) 
   to completely prevent data collisions in the S3 Data Lake.
=============================================================================
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
    compute_region='eu-west-3', # Paris (closest to Ifremer)
    n_workers=10
):
    """
    API-based ingestion pipeline using Ifremer ERDDAP.
    Dynamically requests temporal and spatial bounds based on the ae_utils registry.
    """
    
    # --- 1. CONFIGURATION (The Time-Aware Step) ---
    config = get_ae_config(region="california", lat_step=lat_step, lon_step=lon_step, time_step=time_step,)
    res = config['resolutions']
    
    # Destination path automatically includes the region, dates, and resolutions
    output_s3 = f"s3://{config['s3_bucket']}/{config['run_id']}.parquet"

    print(f"☁️ Step 1: Provisioning {cloud_provider.upper()} Infrastructure for {config['run_id']}...")
    
    # --- 2. CLUSTER SETUP ---
    cluster = coiled.Cluster(
        name=f"ae-erddap-{config['run_id']}",
        n_workers=n_workers,
        region=compute_region,
        worker_vm_types=["m5.large", "m4.large", "t3.large"], 
        spot_policy="spot_with_fallback",
    )
    
    client = Client(cluster)
    print(f"✅ Cloud Cluster Ready! Dashboard: {client.dashboard_link}")

    # --- 3. DYNAMIC API QUERY ---
    print(f"\n🗺️ Step 2: Requesting {region.upper()} Data ({config['start_date']} to {config['end_date']})...")
    
    # Reverted URL to only ask for what the server actually has
    erddap_url = (
        f"https://www.ifremer.fr/erddap/tabledap/ArgoFloats.csv?"
        f"time,latitude,longitude,pres,temp,psal"
        f"&latitude>={config['lat'][0]}&latitude<={config['lat'][1]}"
        f"&longitude>={config['lon'][0]}&longitude<={config['lon'][1]}"
        f"&time>={config['start_date']}T00:00:00Z"
        f"&time<={config['end_date']}T23:59:59Z"
    )
    
    # blocksize=None tells Dask to read the live stream without pre-measuring it
    ddf = dd.read_csv(erddap_url, skiprows=[1], blocksize=None)
    
    # Split the data into n_workers*3 chunks so that they can share the physics load
    ddf = ddf.repartition(npartitions=n_workers * 4)

    # --- 4. DATA CLEANING ---
    print("🌉 Step 3: Formatting and Cleaning Data...")
    
    # Rename ERDDAP columns to match the Pangeo standard expected by the physics engine
    ddf = ddf.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    # Calculate exact Depth from Pressure using TEOS-10 (gsw)
    # gsw.z_from_p returns negative values (depth below surface), so we multiply by -1
    ddf['depth'] = gsw.z_from_p(ddf['pres'], ddf['lat']) * -1

    # Force the incoming ERDDAP strings into strict UTC datetime objects
    ddf['time'] = dd.to_datetime(ddf['time'], utc=True)
    
    # Give the 1999 baseline a matching UTC timezone so Pandas can do the math
    baseline = pd.Timestamp('1999-01-01', tz='UTC')
    
    # CRITICAL FIX: Name this column 'time_days' exactly as the physics engine expects
    ddf['time_days'] = (ddf['time'] - baseline).dt.total_seconds() / 86400

    # --- 5. DISTRIBUTED PHYSICS ---
    # Pull resolutions from our new config object
    res = config["resolutions"]
    
    print(f"🚀 Step 4: Distributing Physics Engine (Res: {res['lat_step']}x{res['lon_step']}, Time: {res['time_step']} days)...")
    
    # We define 'meta' so Dask knows the structure of the returned binned data
    meta = pd.DataFrame({
        'time_bin': pd.Series(dtype='float64'),
        'lat_bin': pd.Series(dtype='float64'),
        'lon_bin': pd.Series(dtype='float64'),
        'ohc': pd.Series(dtype='float64'),
        'ohc_per_m': pd.Series(dtype='float64'),
        'n_raw_points': pd.Series(dtype='int64')
    })

    # We map the physics function across all Dask partitions
    ddf_binned = ddf.map_partitions(
        estimate_ohc_from_raw_bins, 
        resolution_lat=res['lat_step'], 
        resolution_lon=res['lon_step'], 
        resolution_time_days=res['time_step'], # Passing the dynamic time_step here
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
    run_cloud_pipeline(
        region="california", 
        lat_step=1.0, 
        lon_step=1.0, 
        time_step=30.0,  # Now explicitly supported!
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

