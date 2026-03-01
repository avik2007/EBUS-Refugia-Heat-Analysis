import coiled
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import warnings
import gsw

# Import the custom ocean physics module. 
# Coiled automatically packages this local file and distributes it to the cloud workers.
from ebus_core.argoebus_thermodynamics import estimate_ohc_from_raw_bins

def run_cloud_pipeline():
    print("☁️ Step 1: Provisioning Cloud Infrastructure...")
    
    # Allocate a cluster of 10 workers in the AWS Paris region (eu-west-3).
    # Using 'spot_with_fallback' requests discounted spare AWS capacity to minimize costs.
    # The m5.large instance type provides 2 CPUs and 8GB of RAM per worker.
    cluster = coiled.Cluster(
        name="ebus-production-run",
        n_workers=10,
        region="eu-west-3",
        worker_vm_types=["m5.large", "m4.large", "t3.large"], 
        scheduler_vm_types=["m5.large", "m4.large","t3.large"],
        spot_policy="spot_with_fallback" ,
    )
    
    # Connect the local Python session to the remote Dask scheduler.
    client = Client(cluster)
    print(f"✅ Cloud Cluster Ready! Dashboard: {client.dashboard_link}")

    print("\n🗺️ Step 2: Ingesting Ocean Data...")
    
    # Define the API endpoint for the French Ifremer ERDDAP server.
    # The query parameters request a specific spatial bounding box and a 1-year time range.
    erddap_url = (
        "https://www.ifremer.fr/erddap/tabledap/ArgoFloats.csv?"
        "time,latitude,longitude,pres,temp,psal"
        "&latitude>=30.0&latitude<=35.0"
        "&longitude>=-125.0&longitude<=-120.0"
        "&time>=2015-01-01T00:00:00Z&time<=2015-12-31T23:59:59Z"
    )
    
    # Lazily stream the CSV data. 
    # blocksize=None is required because the server generates the stream dynamically,
    # meaning the total file size is unknown prior to reading.
    ddf = dd.read_csv(
        erddap_url, 
        skiprows=[1], # Skip the second row containing string unit descriptors
        blocksize=None,
        dtype={'pres': 'float64', 'temp': 'float64', 'psal': 'float64'}
    )

    # Distribute the single data stream into 20 equal partitions.
    # This allows the 10 cloud workers to process chunks of the dataset in parallel.
    ddf = ddf.repartition(npartitions=20)

    print("\n🌉 Step 3: Formatting and Cleaning Data...")
    
    # Standardize coordinate column names.
    ddf = ddf.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    
    # Remove any profiles containing missing sensor data to prevent math errors.
    ddf = ddf.dropna(subset=['temp', 'psal', 'pres', 'lat', 'lon'])
    
    # Enforce UTC timezone awareness on the incoming timestamps.
    # Calculate the elapsed time in days since the January 1, 1999 baseline.
    ddf['time'] = dd.to_datetime(ddf['time'], utc=True)
    ddf['time_days'] = (ddf['time'] - pd.to_datetime('1999-01-01', utc=True)).dt.total_seconds() / 86400.0

    # Calculate physical depth (meters) from pressure (dbar) and latitude.
    # map_partitions applies this calculation across all worker nodes simultaneously.
    def calculate_exact_depth(partition):
        return -1 * gsw.z_from_p(partition['pres'].values, partition['lat'].values)
    
    ddf['depth'] = ddf.map_partitions(calculate_exact_depth, meta=('depth', 'float64'))

    print("\n🚀 Step 4: Distributing Physics Engine (TEOS-10)...")
    
    # Define the expected structure of the final output DataFrame.
    # Dask requires this 'meta' schema to build the computation graph before executing.
    meta = pd.DataFrame({
        'time_bin': pd.Series(dtype='float64'),
        'lat_bin': pd.Series(dtype='float64'),
        'lon_bin': pd.Series(dtype='float64'),
        'ohc': pd.Series(dtype='float64'),
        'ohc_per_m': pd.Series(dtype='float64'),
        'n_raw_points': pd.Series(dtype='int64')
    })

    # Apply the custom thermodynamics binning function to all 20 partitions in parallel.
    ddf_binned = ddf.map_partitions(
        estimate_ohc_from_raw_bins, 
        resolution_lat=1.0, 
        resolution_lon=1.0, 
        resolution_time_days=30,
        meta=meta
    )

    print("\n💾 Step 5 & 6: Computing and Saving to AWS S3...")
    
    # Define the destination path in the private AWS S3 bucket.
    s3_path = "s3://argo-ebus-project-data-abm/processed_ebus_2015.parquet"
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Trigger the computation. The cloud workers will download the data,
        # run the math, and write the output directly to the S3 bucket as a Parquet dataset.
        ddf_binned.to_parquet(s3_path, write_index=False)
    
    print(f"\n🎉 SUCCESS! Distributed computing complete. Data saved to: {s3_path}")
    
    # Fetch a 5-row preview from the completed dataset to display in the local terminal.
    print(ddf_binned.head())
    
    print("\n🛑 Shutting down cloud resources...")
    # Terminate the client and cluster to halt AWS billing immediately.
    client.close()
    cluster.close()

if __name__ == '__main__':
    run_cloud_pipeline()