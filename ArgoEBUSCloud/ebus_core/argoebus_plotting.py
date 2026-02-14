# cloud plotting file for argo ebus analysis
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import train_test_split
import warnings
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


"""
PIPELINE: TAKES OUTPUT FROM ArgoGPR.analyze_rolling_correlations()


    Diagnostic Tool: Reconstructs a GP model for a specific date using TUNED parameters
    and plots a spatial map with error bars.

    Unlike 'produce_kriging_map' (which uses a moving neighborhood for mass production),
    this function performs 'Windowed Global Kriging'. It fits a single GP to ALL data 
    in the time window. This is ideal for inspecting the physics and quality of your 
    tuned parameters, but does not scale to thousands of points.

    Parameters:
    -----------
    df_raw : pd.DataFrame
        The full source dataset (must contain lat, lon, time, target).
    results_df : pd.DataFrame
        The output of 'analyze_rolling_correlations'. Used to look up the
        optimal Length Scales and Noise for the specific date.
    target_date : float
        The specific time (in days) you want to visualize. The function finds
        the closest analysis window in results_df.
    grid_res : float
        Resolution of the output map in degrees (e.g., 0.5 deg).
    """
def plot_kriging_snapshot(df_raw, 
                          results_df, 
                          target_date, 
                          # --- CONFIG ---
                          feature_cols=['lat_bin', 'lon_bin'], # Must match what you ran analysis with
                          target_col='ohc_per_m', 
                          time_col='time_bin',        # The column name in df_raw
                          window_size_days=90,
                          grid_res=0.5, 
                          cmap='magma_r'):
    
    # ---------------------------------------------------------
    # 1. PARAMETER LOOKUP (The Bridge)
    # ---------------------------------------------------------
    # The results_df does not have 'time_days'. It has 'window_center'.
    # We find the row in results_df closest to the user's requested target_date.
    
    if 'window_center' not in results_df.columns:
        raise ValueError("results_df must contain 'window_center'. Did you run analyze_rolling_correlations?")

    # Calculate time distance to every analyzed window
    time_diffs = (results_df['window_center'] - target_date).abs()
    closest_idx = time_diffs.idxmin()
    best_params = results_df.loc[closest_idx]
    
    center_val = best_params['window_center']
    print(f"🎨 PLOTTING SNAPSHOT")
    print(f"   Target Date: {target_date}")
    print(f"   Using Tuned Window: {center_val:.1f} (Diff: {time_diffs.min():.1f} days)")
    
    # ---------------------------------------------------------
    # 2. SLICE RAW DATA (The "Time Capsule")
    # ---------------------------------------------------------
    # We slice df_raw using the 'time_col' (e.g. 'time_days') 
    # based on the window size centered at the parameter time.
    
    t_min = center_val - (window_size_days/2)
    t_max = center_val + (window_size_days/2)
    
    mask = (df_raw[time_col] >= t_min) & (df_raw[time_col] < t_max)
    df_window = df_raw[mask].copy()
    
    print(f"   Sliced {len(df_window)} points from {time_col} [{t_min:.0f} to {t_max:.0f}]")
    
    if len(df_window) < 10:
        print("   ❌ Not enough data in this window to plot.")
        return

    # ---------------------------------------------------------
    # 3. PREPARE GP & SCALING
    # ---------------------------------------------------------
    # We must replicate the exact scaling used during analysis
    X = df_window[feature_cols].values
    y = df_window[target_col].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # ---------------------------------------------------------
    # 4. RECONSTRUCT KERNEL (Physical -> Math Units)
    # ---------------------------------------------------------
    # The GP needs "Scaled Lengths". The results_df has "Physical Lengths".
    # Logic: results_df has columns named f"scale_{col}" (e.g., scale_lat)
    
    learned_scales = []
    for i, col in enumerate(feature_cols):
        scale_col_name = f'scale_{col}'
        if scale_col_name not in best_params:
            raiseKeyError(f"results_df is missing '{scale_col_name}'. Check your feature_cols.")
            
        phys_val = best_params[scale_col_name]
        data_std = scaler_X.scale_[i]
        
        # Convert back to optimizer units
        scaled_val = phys_val / data_std
        learned_scales.append(scaled_val)
        print(f"   Dim '{col}': Phys={phys_val:.2f} -> Scaled={scaled_val:.2f}")

    noise_val = best_params['noise_val']

    # Build Fixed Kernel (We do NOT re-fit/optimize parameters, we just use them)
    k = ConstantKernel(1.0, "fixed") * \
        RBF(length_scale=learned_scales, length_scale_bounds="fixed") + \
        WhiteKernel(noise_level=noise_val, noise_level_bounds="fixed")
    
    gp = GaussianProcessRegressor(kernel=k, optimizer=None)
    gp.fit(X_scaled, y_scaled)
    
    # ---------------------------------------------------------
    # 5. GENERATE PREDICTION GRID
    # ---------------------------------------------------------
    lat_min, lat_max = df_raw[feature_cols[0]].min(), df_raw[feature_cols[0]].max()
    lon_min, lon_max = df_raw[feature_cols[1]].min(), df_raw[feature_cols[1]].max()
    
    lat_grid = np.arange(lat_min, lat_max, grid_res)
    lon_grid = np.arange(lon_min, lon_max, grid_res)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    
    # Flatten for prediction
    X_grid = np.column_stack([LAT.ravel(), LON.ravel()])
    X_grid_scaled = scaler_X.transform(X_grid)
    
    print("   🔮 Kriging (Predicting on Grid)...")
    y_pred_scaled, y_std_scaled = gp.predict(X_grid_scaled, return_std=True)
    
    # Inverse Transform
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).reshape(LAT.shape)
    y_std = (y_std_scaled * scaler_y.scale_[0]).reshape(LAT.shape)
    
    # ---------------------------------------------------------
    # 6. PLOT WITH CARTOPY
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(15, 6))
    
    # --- PLOT A: MEAN ---
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, zorder=100, edgecolor='k', facecolor='lightgray')
    ax1.add_feature(cfeature.COASTLINE, zorder=101)
    ax1.gridlines(draw_labels=True, linestyle='--')
    
    mesh1 = ax1.pcolormesh(LON, LAT, y_pred, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto')
    plt.colorbar(mesh1, ax=ax1, label=target_col)
    ax1.scatter(df_window[feature_cols[1]], df_window[feature_cols[0]], c='green', s=15, marker='x', alpha=0.5, label='Argo Profiles')
    ax1.set_title(f"Predicted Map (Window Center: {center_val:.0f})")

    # --- PLOT B: UNCERTAINTY ---
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, zorder=100, edgecolor='k', facecolor='lightgray')
    ax2.add_feature(cfeature.COASTLINE, zorder=101)
    ax2.gridlines(draw_labels=True, linestyle='--')
    
    mesh2 = ax2.pcolormesh(LON, LAT, y_std, transform=ccrs.PlateCarree(), cmap='Reds', shading='auto', vmin=0)
    plt.colorbar(mesh2, ax=ax2, label=f"Uncertainty (1$\sigma$)")
    ax2.scatter(df_window[feature_cols[1]], df_window[feature_cols[0]], c='green', s=15, marker='o', alpha=0.3)
    ax2.set_title(f"Uncertainty Map")
    
    plt.show()


"""
    Visualizes how the learned Ocean Physics (Correlation Lengths & Noise) 
    evolve over the study period.

    TAKES THE OUTPUT FROM ARGOPPR.analyze_rolling_correlations()
"""
def plot_physics_history(results_df, cv_details=None, time_unit='days'):
    """
    Visualizes the evolution of Ocean Physics, Model Reliability, and Error Statistics.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Output 1 from analyze_rolling_correlations (The metadata).
    cv_details : dict, optional
        Output 2 from analyze_rolling_correlations (The raw validation data).
        Required to plot the Z-score distribution.
    """
    
    # Determine layout based on whether we have cv_details
    rows = 4 if cv_details is not None else 3
    height = 16 if cv_details is not None else 12
    
    fig, axes = plt.subplots(rows, 1, figsize=(12, height))
    
    t = results_df['window_center']
    
    # --- PLOT 1: SPATIAL SCALES (The Physics) ---
    scale_cols = [c for c in results_df.columns if 'scale_' in c]
    for col in scale_cols:
        label = col.replace('scale_', '').title()
        axes[0].plot(t, results_df[col], marker='o', linestyle='-', linewidth=2, label=f"{label} Scale")
        
    axes[0].set_ylabel("Correlation Length (Degrees)")
    axes[0].set_title("Evolution of Spatial Correlation Scales")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # --- PLOT 2: MODEL UNCERTAINTY (The Network) ---
    axes[1].plot(t, results_df['noise_val'], color='purple', marker='s', linestyle='-')
    axes[1].set_ylabel("Noise Level (Scaled Variance)")
    axes[1].set_title("Evolution of Model Noise (Observation Uncertainty + Chaos)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # --- PLOT 3: RELIABILITY OVER TIME (The Stability Check) ---
    axes[2].plot(t, results_df['std_z'], color='green', marker='d', label='Z-Score Std Dev')
    
    # Add "Goldilocks Zone"
    axes[2].axhspan(0.9, 1.1, color='green', alpha=0.1, label='Target Zone (0.9-1.1)')
    axes[2].axhline(1.0, color='green', linestyle='--', alpha=0.5)
    
    axes[2].set_ylabel("Z-Score Std Dev")
    axes[2].set_title("Reliability Check (Is the model confident?)")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()
    
    # --- PLOT 4: ERROR DISTRIBUTION (The Gaussian Check) ---
    if cv_details is not None:
        ax4 = axes[3]
        
        # 1. Aggregate ALL Z-scores from history
        all_z_scores = []
        for window_date, df_cv in cv_details.items():
            if 'z_score' in df_cv.columns:
                all_z_scores.extend(df_cv['z_score'].dropna().values)
        
        all_z_scores = np.array(all_z_scores)
        
        # 2. Plot Histogram
        # Using density=True to compare with PDF
        bins = np.linspace(-4, 4, 40)
        ax4.hist(all_z_scores, bins=bins, density=True, alpha=0.6, color='gray', label='Observed Errors')
        
        # 3. Plot Ideal Normal Distribution
        x_range = np.linspace(-4, 4, 100)
        ax4.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'k--', linewidth=2, label='Ideal Gaussian')
        
        # 4. Styling
        

        ax4.set_title("🔔 STATISTICAL CHECK: Are errors Gaussian?", fontweight='bold')
        ax4.set_xlabel("Z-Score (Standardized Error)")
        ax4.set_ylabel("Probability Density")
        ax4.set_xlim(-4, 4)
        ax4.grid(True, linestyle='--', alpha=0.3)
        ax4.legend()
        
        # Add stats text box
        mean_z = np.mean(all_z_scores)
        std_z = np.std(all_z_scores)
        stats_text = f"Mean: {mean_z:.2f}\nStd Dev: {std_z:.2f}\nN Points: {len(all_z_scores)}"
        ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Shared X-axis formatting
    axes[-1].set_xlabel(f"Time ({time_unit})")
    
    plt.tight_layout()
    plt.show()