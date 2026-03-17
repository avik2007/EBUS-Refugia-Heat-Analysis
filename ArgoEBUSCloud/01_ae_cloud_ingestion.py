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
from dask.distributed import Client
import dask.dataframe as dd
import xarray as xr
import pandas as pd
import fsspec       
import warnings
import gsw

# Import from your core library
from ebus_core.argoebus_thermodynamics import estimate_ohc_from_raw_bins
from ebus_core.ae_utils import get_ae_config

def process_ebus_cloud_pipeline(
    region="california",
    lat_step=1.0, 
    lon_step=1.0, 
    time_step=30.0,
    n_workers=20
):
    # 1. FETCH CONFIG (The "Time-Aware" Step)
    # This pulls start_date/end_date from ae_utils registry
    config = get_ae_config(region, lat_step=lat_step, lon_step=lon_step, time_step=time_step)
    
    # Generate the time-stamped URI for S3
    output_uri = f"s3://{config['s3_bucket']}/{config['run_id']}_processed.parquet"

    # 2. RENT CLOUD INFRASTRUCTURE
    print(f"☁️ Step 1: Starting Cluster for {config['run_id']}...")
    cluster = coiled.Cluster(
        name=f"ae-{config['run_id']}",
        n_workers=n_workers,
        region="eu-west-3", # Paris
        spot_policy="spot_with_fallback",
    )
    client = Client(cluster)

    # 3. CONNECT & SLICE (Temporal Slicing)
    print(f"🌊 Step 2: Slicing {region} from {config['start_date']} to {config['end_date']}...")
    input_zarr = "s3://pangeo-forge-argo-v2/argo.zarr"
    ds = xr.open_zarr(fsspec.get_mapper(input_zarr), consolidated=True)

    # CRITICAL CHANGE: We now slice by 'time' as well
    ds_ebus = ds.sel(
        lat=slice(config['lat'][0], config['lat'][1]),
        lon=slice(config['lon'][0], config['lon'][1]),
        time=slice(config['start_date'], config['end_date']) # Standardized Temporal Slice
    )

    # 4. DISTRIBUTED PHYSICS
    ddf = ds_ebus.to_dask_dataframe()
    
    # Metadata for the Dask Workers
    meta = pd.DataFrame({
        'time_bin': pd.Series(dtype='float64'),
        'lat_bin': pd.Series(dtype='float64'),
        'lon_bin': pd.Series(dtype='float64'),
        'ohc': pd.Series(dtype='float64'),
        'ohc_per_m': pd.Series(dtype='float64'),
        'n_raw_points': pd.Series(dtype='int64')
    })

    print(f"⚙️ Step 3: Running thermodynamics on {len(ddf.partitions)} partitions...")
    ddf_binned = ddf.map_partitions(
        estimate_ohc_from_raw_bins, 
        resolution_lat=config['resolutions']['lat_step'], 
        resolution_lon=config['resolutions']['lon_step'], 
        resolution_time_days=config['resolutions']['time_step'],
        meta=meta
    )

    # 5. SAVE WITH TEMPORAL ID
    print(f"💾 Step 4: Saving to {output_uri}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ddf_binned.to_parquet(output_uri, write_index=False)
    
    print(f"🎉 SUCCESS! {config['run_id']} complete.")
    client.close()
    cluster.shutdown()

if __name__ == "__main__":
    process_ebus_cloud_pipeline(region="california", n_workers=3)

# THE OLD VERSION, PULLING STRAIGHT FROM ERDDAP
"""
import coiled
from dask.distributed import Client
import dask.dataframe as dd
import xarray as xr
import pandas as pd
import fsspec       
import warnings
import gsw

# Import your custom ocean physics and utility modules
from ebus_core.argoebus_thermodynamics import estimate_ohc_from_raw_bins
from ebus_core.ae_utils import get_ae_config

def process_ebus_cloud_pipeline(
    region="california",
    lat_step=1.0, 
    lon_step=1.0, 
    time_step=30.0,
    input_uri="s3://pangeo-forge-argo-v2/argo.zarr", # Default source
    cloud_provider='aws',  
    compute_region='eu-west-3', 
    n_workers=20
):
    
    # Cloud-agnostic ingestion pipeline.- right now, only supports aws, but could in theory do azure or gcp
    # Uses ae_utils to define regional boundaries while allowing the user
    # to specify the cloud backend and storage locations.
    # 
    
    # 1. FETCH REGIONAL TRUTHS
    # Pulls the lat/lon bounds and the default S3 bucket for the chosen region
    config = get_ae_config(region, lat_step=lat_step, lon_step=lon_step, time_step=time_step)
    
    # Build the output path. If you change cloud_provider to 'gcp', 
    # you would simply pass a 'gs://' URI to this function.
    output_uri = f"s3://{config['s3_bucket']}/{config['run_id']}_processed.parquet"

    # 2. RENT THE CLOUD COMPUTERS (Agnostic via Coiled)
    print(f"☁️ Step 1: Requesting {cloud_provider.upper()} Cluster in {compute_region}...")
    
    cluster = coiled.Cluster(
        name=f"ae-{config['run_id']}",
        backend=cloud_provider,  # This makes it agnostic (aws, gcp, or azure)
        region=compute_region,   
        n_workers=n_workers,
        worker_vm_types=["m5.large", "m4.large", "t3.large"], # AWS types (Coiled maps these to GCP equivalents automatically)
        spot_policy="spot_with_fallback",
    )
    
    client = Client(cluster)
    print(f"✅ Cluster Ready! Dashboard: {client.dashboard_link}")

    # 3. MAP DATA (Agnostic via fsspec)
    print(f"🌊 Step 2: Mapping Input Data from {input_uri}...")
    store = fsspec.get_mapper(input_uri)
    ds = xr.open_zarr(store, consolidated=True)

    # 4. SPATIAL SLICE
    # Coordinates come from ae_utils, ensuring consistency across all regions
    print(f"📍 Step 3: Slicing {region.upper()} boundaries...")
    ds_ebus = ds.sel(
        lat=slice(config['lat'][0], config['lat'][1]),
        lon=slice(config['lon'][0], config['lon'][1])
    )

    # 5. CONVERT TO DASK & PREP METADATA
    ddf = ds_ebus.to_dask_dataframe()

    meta = pd.DataFrame({
        'time_bin': pd.Series(dtype='float64'),
        'lat_bin': pd.Series(dtype='float64'),
        'lon_bin': pd.Series(dtype='float64'),
        'ohc': pd.Series(dtype='float64'),
        'ohc_per_m': pd.Series(dtype='float64'),
        'n_raw_points': pd.Series(dtype='int64')
    })

    # 6. DISTRIBUTED PHYSICS
    print(f"⚙️ Step 4: Computing Physics (Res: {lat_step}x{lon_step})...")
    ddf_binned = ddf.map_partitions(
        estimate_ohc_from_raw_bins, 
        resolution_lat=config['resolutions']['lat_step'], 
        resolution_lon=config['resolutions']['lon_step'], 
        resolution_time_days=config['resolutions']['time_step'],
        meta=meta
    )

    # 7. EXECUTE AND SAVE
    print(f"💾 Step 5: Saving Results to {output_uri}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ddf_binned.to_parquet(output_uri, write_index=False)
    
    print(f"🎉 SUCCESS! {region.upper()} processing complete.")
    client.close()
    cluster.shutdown()

if __name__ == "__main__":
    # To run for a different region or provider, just change the arguments here:
    process_ebus_cloud_pipeline(
        region="california",
        cloud_provider="aws",
        lat_step=1.0,
        lon_step=1.0
    )
"""