import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
import warnings
import gsw

# Import your custom ocean physics math
from ebus_core.argoebus_thermodynamics import estimate_ohc_from_raw_bins

def run_local_smoke_test():
    print("💻 Step 1: Spinning up LOCAL Dask Cluster...")
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    print(f"✅ Local Cluster Ready! Dashboard: {client.dashboard_link}")

    print("\n🗺️ Step 2: Connecting to LIVE Euro-Argo Cloud Server...")
    # We query the official French Ifremer server for a 3-month, 5x5 degree slice
    erddap_url = (
        "https://www.ifremer.fr/erddap/tabledap/ArgoFloats.csv?"
        "time,latitude,longitude,pres,temp,psal"
        "&latitude>=30.0&latitude<=35.0"
        "&longitude>=-125.0&longitude<=-120.0"
        "&time>=2015-01-01T00:00:00Z&time<=2015-03-31T23:59:59Z"
    )
    
    print("   Streaming actual data directly into Dask...")
    # 1. We set blocksize=None because ERDDAP is a live stream and it doesn't know what the file size is as it is streaming data
    # ERDDAP's second row contains string units (e.g., "degrees_north"), so we skip it
    ddf = dd.read_csv(
        erddap_url, 
        skiprows=[1], 
        blocksize=None, 
        dtype={'pres': 'float64', 'temp': 'float64', 'psal': 'float64'}
    )
    # 2. We split that single stream into 4 chunks so your workers can process in parallel
    ddf = ddf.repartition(npartitions=4)

    print("\n🌉 Step 3: Formatting Data for the Physics Engine...")
    # Rename ERDDAP columns to match what your math expects
    ddf = ddf.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    ddf = ddf.dropna(subset=['temp', 'psal', 'pres', 'lat', 'lon'])
    
    # Convert string timestamps to datetime, then to days since 1999
    ddf['time'] = dd.to_datetime(ddf['time'], utc=True)
    ddf['time_days'] = (ddf['time'] - pd.to_datetime('1999-01-01', utc=True)).dt.total_seconds() / 86400.0

    def calculate_exact_depth(partition):
        return -1 * gsw.z_from_p(partition['pres'].values, partition['lat'].values)
    
    ddf['depth'] = ddf.map_partitions(calculate_exact_depth, meta=('depth', 'float64'))

    print("\n🚀 Step 4: Applying TEOS-10 & Binning locally...")
    meta = pd.DataFrame({
        'time_bin': pd.Series(dtype='float64'),
        'lat_bin': pd.Series(dtype='float64'),
        'lon_bin': pd.Series(dtype='float64'),
        'ohc': pd.Series(dtype='float64'),
        'ohc_per_m': pd.Series(dtype='float64'),
        'n_raw_points': pd.Series(dtype='int64')
    })

    # This is where your custom function is triggered
    ddf_binned = ddf.map_partitions(
        estimate_ohc_from_raw_bins, 
        resolution_lat=1.0, 
        resolution_lon=1.0, 
        resolution_time_days=30,
        meta=meta
    )

    print("\n💾 Step 5: Executing the math...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # .compute() pulls the trigger to start the download and the math
        result_df = ddf_binned.compute()
    
    print("\n🎉 SUCCESS! Here is your calculated Ocean Heat Content from LIVE Argo data:")
    print(result_df.head())
    
    client.close()
    cluster.close()

if __name__ == '__main__':
    run_local_smoke_test()