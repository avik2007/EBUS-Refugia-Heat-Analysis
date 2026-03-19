"""
=============================================================================
VERSION 4.1: The Response Layer Diagnostic (0-100m)
=============================================================================
1. Depth Awareness: Now specifically targets the 0-100m 'Response Layer' files.
2. Storyboard Export: Automatically saves Kriging snapshots to AEResults/aeplots.
3. High-Res Compatibility: Matches the 0.5° granularity of the cloud ingestion.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from ebus_core.ae_utils import get_ae_config, ensure_ae_dirs
from ebus_core.argoebus_gp_physics import (
    analyze_rolling_correlations, 
    plot_physics_history, 
    plot_kriging_snapshot
)

def run_diagnostic_inspection(region="california", lat_step=0.5, lon_step=0.5, 
                              time_step=30.0, depth_range=(0, 100)):
    # --- 1. SETUP & HOUSEKEEPING ---
    # Fetch config that matches the 0-100m depth naming
    config = get_ae_config(
        region=region, 
        lat_step=lat_step, 
        lon_step=lon_step, 
        time_step=time_step,
        depth_range=depth_range
    )
    
    # Ensure AEResults/aeplots exists
    ensure_ae_dirs()
    plot_dir = os.path.join("AEResults", "aeplots")
    
    print(f"📥 Loading Response Layer Dataset: {config['run_id']}...")
    s3_uri = f"s3://{config['s3_bucket']}/{config['run_id']}.parquet"
    
    try:
        df = pd.read_parquet(s3_uri)
        print(f"✅ Data loaded! Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Could not find file in S3. Did Script 02 finish? Error: {e}")
        return

    # --- 2. ROLLING PHYSICS ANALYSIS ---
    # We use a 30d window to capture the 'fast' response layer physics
    print(f"🧮 Engaging Physics Engine (Window: 30d, Step: 15d)...")
    results_df, cv_details = analyze_rolling_correlations(
        df=df,
        feature_cols=['lat_bin', 'lon_bin'],
        target_col='ohc_per_m',
        time_col='time_bin',
        window_size_days=30,  
        step_size_days=15,
        auto_tune=True
    )

    # --- 3. SAVE STORYBOARD TO AEResults/aeplots ---
    # --- 3. SAVE STORYBOARD (NUCLEAR OPTION) ---


    # 1. Force Absolute Pathing
    # This ensures it goes to the EXACT folder regardless of where you run the script
    base_path = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(base_path, "AEResults", "aeplots")
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
    
    print(f"📌 HARD-CODED DESTINATION: {plot_dir}")
    
    for target_t in results_df['window_center']:
        # Generate the figure
        fig = plot_kriging_snapshot(
            df_raw=df, 
            results_df=results_df, 
            target_date=target_t,
            feature_cols=['lat_bin', 'lon_bin'], 
            time_col='time_bin', 
            grid_res=0.25
        )
        
        
        if fig is None:
            print(f"   ⚠️  SKIPPED: Not enough data in window {int(target_t)}")
            continue

        # 2. CONSTRUCT DESCRIPTIVE FILENAME
        # Use config['run_id'] which contains: california_20150101_20151231_res0_5x0_5_t30_0_d0_100
        snapshot_name = f"snapshot_{config['run_id']}_day{int(target_t)}.png"
        holding_folder_name = f"snapshot_{config['run_id']}"
        holding_folder_path = os.path.join(plot_dir, holding_folder_name)

        if not os.path.exists(holding_folder_path):
            os.makedirs(holding_folder_path, exist_ok=True)
        save_path = os.path.join(holding_folder_path, snapshot_name)
        
        print(f"   💾 ATTEMPTING WRITE: {snapshot_name}")
        
        # 3. Force the Save
        fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=False)
        
        # 4. Clean up memory
        plt.close(fig)
        gc.collect()
    # --- 4. SAVE NUMERICAL DATA (THE STEALTH WARMING REGISTRY) ---
    # Create a matching 'results' folder within the standardized directory structure
    data_out_dir = os.path.join(base_path, "AEResults", "aelogs", config['run_id'])
    os.makedirs(data_out_dir, exist_ok=True)

    # Save the Rolling Audit (Accuracy, Reliability, and Anisotropy Scales)
    audit_filename = f"audit_{config['run_id']}.csv"
    audit_path = os.path.join(data_out_dir, audit_filename)
    results_df.to_csv(audit_path, index=False)
    
    # Save the Cross-Validation Details (Raw error points for subtle bias detection)
    cv_filename = f"cv_details_{config['run_id']}.pkl"
    cv_path = os.path.join(data_out_dir, cv_filename)
    pd.to_pickle(cv_details, cv_path)

    print(f"\n📊 DATA REGISTRY UPDATED")
    print(f"   Audit CSV: {audit_path}")
    print(f"   CV Pickle: {cv_path}")
    print(f"🚀 PIPELINE COMPLETE for {config['run_id']}")
    print(f"\n🚀 DONE. Open your terminal and run: 'ls -lh {plot_dir}'")

if __name__ == "__main__":
    # --- MATCHING THE 0.5 DEGREE 100M RUN ---
    run_diagnostic_inspection(
        region="california",
        lat_step=0.5,
        lon_step=0.5,
        time_step=30.0,
        depth_range=(0, 100)
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