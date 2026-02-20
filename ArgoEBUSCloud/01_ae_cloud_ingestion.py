import coiled
from dask.distributed import Client
import dask.dataframe as dd
import xarray as xr
import pandas as pd
import numpy as np
import s3fs
import warnings
import gsw
# Import YOUR custom physics engine
from ebus_core.thermodynamics import estimate_ohc_from_raw_bins

def process_ebus_cloud_pipeline(
    lat_bounds, 
    lon_bounds, 
    time_bounds,
    s3_input_uri,
    s3_output_uri,
    ref_date='1999-01-01', # The dawn of the Argo era
    n_workers=20
):
    """
    Streams raw Argo Zarr data from AWS, calculates OHC thermodynamics, 
    and saves the binned results back to cloud storage.
    """
    print(f"☁️ Step 1: Requesting Cloud Cluster ({n_workers} workers)...")
    cluster = coiled.Cluster(
        name="argo-ebus-ingestion",
        n_workers=n_workers,
        worker_memory="16 GiB",
        region="us-west-2" 
    )
    client = Client(cluster)
    print(f"✅ Cluster Ready! Dashboard: {client.dashboard_link}")

    # Convert ref_date string to timestamp for math later
    ref_timestamp = pd.to_datetime(ref_date)

    print(f"\n🗺️ Step 2: Lazy-Loading AWS Argo Archive from {s3_input_uri}...")
    fs = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=s3_input_uri, s3=fs, check=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_zarr(store, consolidated=True, chunks='auto')

    print(f"\n✂️ Step 3: Slicing the Region (Lazy Evaluation)...")
    print(f"   Lat: {lat_bounds} | Lon: {lon_bounds} | Time: {time_bounds}")
    
    ds_ebus = ds.sel(
        lat=slice(lat_bounds[0], lat_bounds[1]),
        lon=slice(lon_bounds[0], lon_bounds[1]),
        time=slice(time_bounds[0], time_bounds[1])
    )

    print("\n🌉 Step 4: Converting to Distributed Dask DataFrame...")
    # Convert multidimensional array to tabular format
    ddf = ds_ebus[['TEMP', 'PSAL', 'PRES']].to_dask_dataframe(dim_order=['time', 'lat', 'lon'])
    
    ddf = ddf.rename(columns={'TEMP': 'temp', 'PSAL': 'psal', 'PRES': 'pres'})
    ddf = ddf.dropna(subset=['temp', 'psal', 'pres', 'lat', 'lon'])

    # --- EXACT DEPTH CALCULATION (GSW via Dask) ---
    def calculate_exact_depth(partition):
        """Helper function applied to each worker's chunk of data"""
        # gsw.z_from_p returns height (negative down). We multiply by -1 for positive depth.
        return -1 * gsw.z_from_p(partition['pres'].values, partition['lat'].values)
    
    ddf['depth'] = ddf.map_partitions(calculate_exact_depth, meta=('depth', 'float64'))
    
    # --- EXACT TIME CALCULATION ---
    ddf['time_days'] = (ddf['time'] - ref_timestamp).dt.total_seconds() / 86400.0

    print("\n🚀 Step 5: Applying TEOS-10 & Binning across Cloud Computers...")
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
        resolution_lat=1.0, 
        resolution_lon=1.0, 
        resolution_time_days=30,
        meta=meta
    )

    print(f"\n💾 Step 6: Computing and Saving to {s3_output_uri}...")
    # Trigger the cluster to do the math and save
    ddf_binned.to_parquet(s3_output_uri, write_index=False)
    
    print(f"🎉 SUCCESS! Cleaned OHC Data saved.")
    
    # Teardown
    client.close()
    cluster.close()

# --- EXECUTION BLOCK ---
if __name__ == '__main__':
    # You can now cleanly call this with different parameters for different EBUS systems
    process_ebus_cloud_pipeline(
        lat_bounds=(20.0, 50.0),
        lon_bounds=(-135.0, -105.0),
        time_bounds=('2010-01-01', '2020-12-31'),
        s3_input_uri='s3://argovis-public-release/argo_zarr_archive',
        s3_output_uri='s3://my-ebus-data-bucket/california_current_ohc.parquet', # Replace with your bucket/path
        ref_date='1999-01-01',
        n_workers=20
    )