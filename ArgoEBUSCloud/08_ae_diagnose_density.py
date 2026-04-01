import pandas as pd
import numpy as np
import os
import sys
from ebus_core.ae_utils import get_ae_config

def diagnose_density(region="california", lat_step=0.5, lon_step=0.5, time_step=10.0, depth_range=(0, 100)):
    config = get_ae_config(
        region=region,
        lat_step=lat_step,
        lon_step=lon_step,
        time_step=time_step,
        depth_range=depth_range
    )
    s3_uri = f"s3://{config['s3_bucket']}/{config['run_id']}.parquet"
    print(f"\n📊 DIAGNOSING DENSITY: {region} | {depth_range}m | {time_step}d bins")
    print(f"   Source: {s3_uri}")

    try:
        df = pd.read_parquet(s3_uri)
    except Exception as e:
        print(f"   ❌ Could not load parquet: {e}")
        return None

    n_rows = len(df)
    n_floats = df['platform_number'].nunique() if 'platform_number' in df.columns else "N/A"
    
    # Calculate density metrics
    # Unique 3D bins (lat/lon/time)
    n_bins = len(df.groupby(['lat_bin', 'lon_bin', 'time_bin']).size())
    
    # Average points per 10-day bin
    avg_per_bin = df['n_raw_points'].mean() if 'n_raw_points' in df.columns else "N/A"
    
    print(f"   ✅ Rows (3D Bins): {n_rows:,}")
    print(f"   ✅ Unique Floats : {n_floats}")
    print(f"   ✅ Avg Obs/Bin   : {avg_per_bin}")
    
    # Check temporal distribution
    if 'time_bin' in df.columns:
        time_counts = df.groupby('time_bin').size()
        print(f"   ✅ Windows (10d) : {len(time_counts)}")
        print(f"   ✅ Min Bins/10d  : {time_counts.min()}")
        print(f"   ✅ Max Bins/10d  : {time_counts.max()}")
        print(f"   ✅ Median Bins/10d: {time_counts.median()}")

    return {
        "region": region,
        "depth": depth_range,
        "n_rows": n_rows,
        "n_floats": n_floats,
        "median_bins_per_10d": time_counts.median() if 'time_bin' in df.columns else 0
    }

if __name__ == "__main__":
    results = []
    
    # Compare california vs californiav2 across layers
    for reg in ["california", "californiav2"]:
        for d_range in [(0, 100), (150, 400), (500, 1000)]:
            # Note: california baseline used t=30.0, californiav2 used t=10.0
            t_step = 10.0 if reg == "californiav2" else 30.0
            res = diagnose_density(region=reg, depth_range=d_range, time_step=t_step)
            if res:
                results.append(res)
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
