"""
=============================================================================
VERSION 4.0: The Complete Diagnostic Pipeline
=============================================================================
This script downloads binned thermodynamic data from AWS and passes it 
through the full Rolling Correlation analysis to extract the physical 
length scales of the marine heatwave over time. It then visualizes the 
model's reliability and maps a specific snapshot in time.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
# Import the core registry to dynamically find your data
from ebus_core.ae_utils import get_ae_config, ensure_ae_dirs
# Import your newly completed GP physics pipeline
from ebus_core.argoebus_gp_physics import (
    analyze_rolling_correlations,
    plot_physics_history,
    plot_kriging_snapshot
)

def run_diagnostic_inspection(region="california", lat_step=1.0, lon_step=1.0, time_step=30.0):
    # --- 1. FETCH CONFIGURATION & DATA ---
    config = get_ae_config(region=region, lat_step=lat_step, lon_step=lon_step, time_step=time_step)
    s3_uri = f"s3://{config['s3_bucket']}/{config['run_id']}.parquet"
    
    print(f"\n📥 Step 1: Connecting to AWS Data Lake...")
    print(f"   Target: {s3_uri}")
    try:
        df = pd.read_parquet(s3_uri)
    except FileNotFoundError:
        print(f"❌ Error: Could not find dataset. Run ingestion first.")
        return
    print(f"✅ Data loaded! Shape: {df.shape}")

    # --- 2. THE ROLLING OPTIMIZATION ENGINE ---
    print("\n🧮 Step 2: Engaging the Spatio-Temporal Physics Engine...")
    print("   (This will slide a 60-day window across 2015, tuning the math at every step)")
    
    results_df, cv_details = analyze_rolling_correlations(
        df=df,
        feature_cols=['lat_bin', 'lon_bin'],
        target_col='ohc_per_m',
        time_col='time_bin',
        window_size_days=30,      # Analyze a 2-month chunk at a time
        step_size_days=15,        # Step forward 1 month at a time
        k_fold_data_percent=15,   # Hold out 15% of the data to prove the model works
        auto_tune=True,           # Find the true physical length scales
        auto_calibrate=True       # Guarantee the Z-Scores hit the Goldilocks zone
    )

    # --- 3. THE DIAGNOSTIC DASHBOARD ---
    print("\n📊 Step 3: Generating Physics History Dashboard...")
    print("   Close the popup window to proceed to the map.")
    plot_physics_history(results_df, cv_details=cv_details, time_unit='days since 1999')

    # --- 4. SYSTEMATIC MAPPING ---
    print("\n🗺️ Step 4: Generating Storyboard of all Analysis Windows...")
    ensure_ae_dirs()
    plot_dir = config['paths']['plots']

    for target_t in results_df['window_center']:
        print(f"   Mapping window centered at Day {target_t:.1f}...")
        
        # We call your snapshot function
        plot_kriging_snapshot(
            df_raw=df,
            results_df=results_df,
            target_date=target_t,
            feature_cols=['lat_bin', 'lon_bin'],
            target_col='ohc_per_m',
            time_col='time_bin',
            window_size_days=60,
            grid_res=0.5
        )
        
        # Save the figure with a descriptive name
        out_file = os.path.join(plot_dir, f"snapshot_{config['run_id']}_day{int(target_t)}.png")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close() # Crucial to prevent memory leaks

    print(f"\n🎉 STORYBOARD COMPLETE! Check your plots folder: {plot_dir}")

if __name__ == "__main__":
    # Ensure this matches exactly what you ran in 02_ae_cloud_run.py
    run_diagnostic_inspection(
        region="california", 
        lat_step=1.0, 
        lon_step=1.0, 
        time_step=30.0
    )



"""
import pandas as pd
import matplotlib.pyplot as plt
import s3fs
import os

# Import regional configuration and directory management
from ebus_core.ae_utils import get_ae_config, ensure_ae_dirs

def inspect_processed_data(
    region="california",
    lat_step=1.0, 
    lon_step=1.0, 
    time_step=30.0
):
    
    #Downloads processed EBUS data from S3 and generates a monthly 
    #Ocean Heat Content (OHC) trend plot.
    
    
    # 1. FETCH CONFIGURATION
    # This ensures we look in the right bucket and use the right filename
    config = get_ae_config(region, lat_step=lat_step, lon_step=lon_step, time_step=time_step)
    
    # Standardized S3 path (matches the output of scripts 01 and 02)
    s3_path = f"s3://{config['s3_bucket']}/{config['run_id']}.parquet"
    
    print(f"📡 Downloading {region.upper()} results from: {s3_path}")
    
    # 2. LOAD DATA
    try:
        df = pd.read_parquet(s3_path)
    except Exception as e:
        print(f"❌ Error: Could not find data at {s3_path}.")
        print("Check if you ran 01_ae or 02_ae with the same resolution settings.")
        return

    # 3. DATA AUDIT
    print(f"\n📊 {region.upper()} Dataset Summary ({lat_step}x{lon_step} resolution):")
    print(df.info())
    print("\n🔍 Sample Data (Top 5 Rows):")
    print(df.head())

    # 4. ANALYSIS: Monthly Heat Trend
    # We aggregate the spatial bins to see the temporal evolution
    monthly_trend = df.groupby('time_bin')['ohc_per_m'].mean().reset_index()
    
    # 5. VISUALIZATION
    plt.figure(figsize=(12, 6))
    plt.plot(
        monthly_trend['time_bin'], 
        monthly_trend['ohc_per_m'], 
        marker='o', 
        linestyle='-', 
        color='teal',
        linewidth=2,
        label='Mean OHC per Meter'
    )
    
    # Aesthetics
    plt.title(f'{region.upper()} Ocean Heat Content Trend (Run: {config["run_id"]})', fontsize=14)
    plt.xlabel('Days since 1999-01-01', fontsize=12)
    plt.ylabel('Average OHC per Meter (J/m)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # 6. ORGANIZED SAVING
    # Ensure the /ArgoEBUSAnalysis/AEResults/aeplots/ directory exists
    ensure_ae_dirs()
    
    # Use the run_id in the filename so different resolutions don't overwrite each other
    plot_filename = f"trend_{config['run_id']}.png"
    save_path = os.path.join(config['paths']['plots'], plot_filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Success! Analysis plot saved to:\n   {save_path}")
    
    # Show in VS Code Interactive window if available
    plt.show()

if __name__ == "__main__":
    # To inspect a different run, change the steps or region here:
    inspect_processed_data(region="california", lat_step=1.0, lon_step=1.0)
"""