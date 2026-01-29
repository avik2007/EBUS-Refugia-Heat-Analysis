import numpy as np
import os
import argopy
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gsw
from datetime import datetime   
from argopy import DataFetcher as ArgoDataFetcher

def calculate_thermodynamics(sp, t, p, lon, lat):
    """
    Wrapper for GSW functions to be used with apply_ufunc.
    Inputs are expected to be numpy arrays (handled by xarray wrapper).
    """
    # 1. Absolute Salinity (SA)
    # GSW requires (SP, P, lon, lat)
    # We ensure P is broadcasted correctly by xarray before it gets here
    sa = gsw.SA_from_SP(sp, p, lon, lat)
    
    # 2. Conservative Temperature (CT)
    ct = gsw.CT_from_t(sa, t, p)
    
    # 3. Density (rho)
    rho = gsw.rho(sa, ct, p)
    
    # 4. Specific Heat Capacity (Cp)
    cp = gsw.cp_t_exact(sa, t, p)
    
    # 5. Energy Density (J/m^3) = rho * cp * T (in-situ)
    # We return the energy density directly
    return rho * cp * t

# once we have the ability to calculate thermodynamics for one ocean column, we can incorporate it into 
# a function that makes xarray inputs/outputs
def compute_ohc_layer(ds_input, layer_label):
    # Ensure inputs are present
    # Salinity sensors can be fouled up in Argo sensors, so we have to account for this possibility
    if 'PSAL' not in ds_input or 'TEMP' not in ds_input:
        print(f"⚠️ Missing variables in {layer_label}")
        return None

    # GSW needs Lat/Lon as arguments. 
    # ds_input['LONGITUDE'] and ['LATITUDE'] are likely 1D (by profile).
    # We don't need to manually broadcast them; apply_ufunc handles alignment 
    # if we pass them as xarray objects.
    
    # However, PRESSURE is a vertical coordinate (PRES_INTERPOLATED).
    # We must ensure it's passed correctly.
    
    # THE WRAPPER:
    # We use apply_ufunc to push the xarray objects into the numpy-based GSW function
    energy_density = xr.apply_ufunc(
        calculate_thermodynamics,
        ds_input['PSAL'],
        ds_input['TEMP'],
        ds_input['PRES_INTERPOLATED'],
        ds_input['LONGITUDE'],
        ds_input['LATITUDE'],
        input_core_dims=[[], [], [], [], []], # All inputs map point-to-point (broadcasting happens auto)
        output_core_dims=[[]], # Returns one array of same shape
        vectorize=True,        # Loops if core dims don't match (safe fallback)
        dask='parallelized',   # If you use Dask chunks later, this is ready
        output_dtypes=[float]
    )
    
    # 2. Integrate over depth (J/m^3 -> J/m^2)
    ohc = energy_density.integrate(coord='PRES_INTERPOLATED')
    
    # 3. Metadata
    ohc.name = layer_label
    ohc.attrs['units'] = 'J/m^2'
    ohc.attrs['description'] = 'Integrated Heat Content (TEOS-10: Rho*Cp*T)'
    
    return ohc


# --- TIME SERIES RESAMPLING ---
# we convert the xarray ocean heat content structure into a pandas structure. We then resample to look
# monthly mean trends. This function takes all the floats in the region we picked (no matter what the
# size, and lumps their data together, creating a terribly spatially averaged heat content, subject
# to biases in a region that may have heterogeneous heat content (north south bias, for example))
def make_timeseries(da_ohc):
    if da_ohc is None: return None
    df = da_ohc.to_dataframe().reset_index()
    df = df.set_index('TIME')
    # Resample to Monthly Mean
    return df[da_ohc.name].resample('MS').mean()

# this 
def make_spatially_weighted_timeseries(da_ohc):
    # 1. Convert to DataFrame
    df = da_ohc.to_dataframe().reset_index()
    
    # 2. Create Spatial Bins (e.g., 1-degree bins)
    # We round the Lat/Lon to the nearest integer to create "bins"
    """
     UPDATE THIS TO ALLOW FOR BINS OF ANY SIZE. 1 DEGREE COULD BE TOO SMALL 
    """
    df['lat_bin'] = df['LATITUDE'].round(0)
    df['lon_bin'] = df['LONGITUDE'].round(0)
    
    # 3. Create Time Bins (Month)
    # We use a string format for grouping: "2023-01"
    df['month_year'] = df['TIME'].dt.to_period('M')
    
    # --- THE TRICK ---
    # Step A: Average inside each 1x1 degree bin for each month
    # This collapses the 50 offshore floats into a single "grid value"
    grid_means = df.groupby(['month_year', 'lat_bin', 'lon_bin'])[da_ohc.name].mean().reset_index()
    
    # Step B: Average the grid cells for each month
    # Now every grid cell gets an equal vote, regardless of how many floats were in it
    regional_means = grid_means.groupby('month_year')[da_ohc.name].mean()
    
    # Convert index back to Timestamp for plotting
    regional_means.index = regional_means.index.to_timestamp()
    
    return regional_means
"""
    Fetches Argo data. Checks local cache first; if missing, downloads from Erddap and saves.
    
    Args:
        nc_dir (str): Directory where the processed DataFrame will be saved/loaded.
        start_date, end_date (str): 'YYYY-MM-DD'.
        lat_bounds, lon_bounds (list): [min, max].
        depth_bounds (list): [min, max] in meters.
    Fetches Argo data specifically for Ocean Heat Content analysis.
    
    Improvements:
    - Fetches Salinity (PSAL) alongside Temperature (TEMP).
    - Preserves FULL profiles (all valid depth levels).
    - SAVES 'DEPTH' instead of 'PRES' (Renames pressure variable).
    - Handles 'ADJUSTED' (Quality Controlled) variables automatically.

    Returns:
    - pd.DataFrame with columns: [float_id, time_days, lat, lon, pres, temp, psal]
    """
def load_argo_data_advanced(nc_dir, start_date, end_date, lat_bounds, lon_bounds, depth_bounds=[0, 2000]):
    
    # 1. SETUP & FILENAME GENERATION
    ref_date = pd.to_datetime(start_date)
    os.makedirs(nc_dir, exist_ok=True)
    
    # Generate Descriptive Filename
    # Added '_OHC' tag to filename
    fname = (f"argo_{start_date}_to_{end_date}_"
             f"lat{lat_bounds[0]}_{lat_bounds[1]}_"
             f"lon{lon_bounds[0]}_{lon_bounds[1]}_"
             f"z{depth_bounds[0]}_{depth_bounds[1]}.pkl")
             
    save_path = os.path.join(nc_dir, fname)

    # 2. CHECK LOCAL CACHE
    if os.path.exists(save_path):
        print(f"\n📂 FOUND LOCAL DATASET: {save_path}")
        print("   Loading processed DataFrame...")
        try:
            df = pd.read_pickle(save_path)
            print(f"   ✅ Loaded {len(df)} profile measurements from disk.")
            return df
        except Exception as e:
            print(f"   ⚠️ Error loading cached file (will re-fetch): {e}")

    # 3. FETCH DATA (If not found locally)
    print(f"\n🌊 INITIATING ARGO FETCH (Source: Erddap)")
    print(f"   Target: {start_date} to {end_date}")
    print(f"   Region: Lat {lat_bounds} | Lon {lon_bounds}")
    print(f"   Depth:  {depth_bounds[0]}m to {depth_bounds[1]}m")
    
    try:
        # Fetch standard variables (T, S, P)
        fetcher = ArgoDataFetcher(src='erddap').region(
            [lon_bounds[0], lon_bounds[1], 
             lat_bounds[0], lat_bounds[1], 
             depth_bounds[0], depth_bounds[1], 
             start_date, end_date]
        )
        ds = fetcher.to_xarray()
        print(f"   ✅ Data received from server.")
    except Exception as e:
        print(f"   ❌ FETCH FAILED: {e}")
        return pd.DataFrame()

    # 4. PROCESS DATA
    print(f"   🔄 Processing profiles...")
    
    # --- A. VARIABLE DETECTION (Standard vs Adjusted) ---
    t_var = 'TEMP_ADJUSTED' if 'TEMP_ADJUSTED' in ds else 'TEMP'
    s_var = 'PSAL_ADJUSTED' if 'PSAL_ADJUSTED' in ds else 'PSAL'
    p_var = 'PRES_ADJUSTED' if 'PRES_ADJUSTED' in ds else 'PRES'
    
    if t_var not in ds or s_var not in ds:
        print(f"   ❌ Missing Critical Vars: Found T={t_var in ds}, S={s_var in ds}")
        return pd.DataFrame()

    # --- B. FLATTEN TO DATAFRAME ---
    # Convert xarray directly to dataframe
    df_raw = ds[[t_var, s_var, p_var, 'LATITUDE', 'LONGITUDE', 'TIME', 'PLATFORM_NUMBER']].to_dataframe().reset_index()
    
    # Clean up column names (Renaming PRES -> DEPTH)
    df_raw = df_raw.rename(columns={
        t_var: 'temp',
        s_var: 'psal',
        p_var: 'pres',    # <--- CHANGED: Now keeping 'pres' as Pressure (dbar)
        'LATITUDE': 'lat',
        'LONGITUDE': 'lon',
        'TIME': 'date',
        'PLATFORM_NUMBER': 'float_id'
    })
    
    # --- C. FILTERING & CLEANING ---
    print(f"   🧹 Cleaning {len(df_raw)} raw points...")
    
    # 1. Drop NaNs in critical columns (Using 'depth' now)
    df = df_raw.dropna(subset=['temp', 'psal', 'pres', 'lat', 'lon', 'date'])
    
    # 2. Decode Byte Strings (Float IDs)
    if df['float_id'].dtype == object and isinstance(df['float_id'].iloc[0], bytes):
        df['float_id'] = df['float_id'].str.decode('utf-8').str.strip()
    df['float_id'] = df['float_id'].astype(str)

    # 3. Standardize Longitude (-180 to 180)
    df.loc[df['lon'] > 180, 'lon'] -= 360
    
    # [NEW] Calculate Depth in Meters from Pressure
    # z_from_p returns height (negative down), so we flip to positive Depth
    df['depth'] = -1 * gsw.z_from_p(df['pres'].values, df['lat'].values)
    
    # 4. Spatial Filter
    df = df[
        (df['lat'] >= lat_bounds[0]) & (df['lat'] <= lat_bounds[1]) &
        (df['lon'] >= lon_bounds[0]) & (df['lon'] <= lon_bounds[1])
    ]
    
    # 5. Calculate Numeric Time (Days since start)
    df['time_days'] = (df['date'] - ref_date).dt.total_seconds() / 86400.0
    
    # 6. Sort for clean integration later (Sorting by depth)
    df = df.sort_values(by=['float_id', 'date', 'pres'])
    
    print(f"✅ COMPLETE: Loaded {len(df)} valid profile levels.")
    
    # 5. SAVE & RETURN
    if not df.empty:
        try:
            df.to_pickle(save_path)
            print(f"   💾 DataFrame saved to: {save_path}")
        except Exception as e:
            print(f"   ⚠️ Save failed: {e}")
            
    return df



"""
    Calculates OHC by pooling ALL raw data in a Lat/Lon/Time box into one 
    'Synthetic Profile' and integrating it. DESIGNED TO TAKE INPUTS FROM
    load_argo_data_advanced().
    
    STRATEGY:
    1. Binning: Assign every raw measurement (from any float) to a 3D Bin.
    2. Thermodynamics: Calculate Energy Density (J/m^3) for every point.
    3. Vertical Interpolation: Average the energy density into standard vertical steps (e.g. every 10m).
    4. Integration: Integrate the vertical profile to get OHC (J/m^2).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must contain 'lat', 'lon', 'time_days', 'pres' (dbar), 'depth' (meters), 'temp', 'psal'.
    resolution_lat, resolution_lon : float
        Spatial size of the bin to aggregate floats into.
    resolution_time_days : int
        Temporal size of the bin.
    depth_min, depth_max : float
        Integration limits (in meters).
    vertical_step : int
        The vertical resolution (in meters) to smooth the pooled data before integrating.
        Default is 10m (standard for OHC).
    min_coverage_pct : float
        Robustness threshold. The profile must cover this fraction of the water column
        (AND have surface/deep data) to be considered valid.
        
    Returns:
    --------
    df_binned : pd.DataFrame
        One row per Lat/Lon/Time bin with 'ohc', 'ohc_per_m', 'n_points', etc.
    """

def estimate_ohc_from_raw_bins(df, 
                               resolution_lat=1.0, 
                               resolution_lon=1.0, 
                               resolution_time_days=30,
                               depth_min=0, 
                               depth_max=2000,
                               vertical_step=10, 
                               min_coverage_pct=0.1): # Lowered default to 10%
    
    
    print(f"📦 BINNING RAW DATA: {resolution_lat}° x {resolution_lon}° x {resolution_time_days} days...")
    
    # Work on a copy to avoid SettingWithCopy warnings
    work_df = df.copy()
    
    # --- STEP 1: CALCULATE THERMODYNAMICS (PER POINT) ---
    # We use 'pres' (dbar) for GSW calculations because that's what the physics requires.
    p = work_df['pres'].values
    lat = work_df['lat'].values
    lon = work_df['lon'].values
    
    # GSW Calculations (TEOS-10 Standard)
    # 1. Absolute Salinity
    work_df['SA'] = gsw.SA_from_SP(work_df['psal'].values, p, lon, lat)
    
    # 2. Conservative Temperature
    work_df['CT'] = gsw.CT_from_t(work_df['SA'].values, work_df['temp'].values, p)
    
    # 3. Density (rho)
    #    Calculated exactly at every point
    rho = gsw.rho(work_df['SA'].values, work_df['CT'].values, p)
    
    # 4. Specific Heat Capacity (cp)
    #    Calculated EXACTLY at every point using gsw.cp_t_exact
    cp = gsw.cp_t_exact(work_df['SA'].values, work_df['temp'].values, p)
    
    # 5. Energy Density (Joules / m^3)
    #    Heat Content = rho * cp * CT
    work_df['energy_density'] = rho * cp * work_df['CT']
    
    # --- STEP 2: ASSIGN SPATIAL BINS ---
    work_df['lat_bin'] = (work_df['lat'] // resolution_lat) * resolution_lat + (resolution_lat/2)
    work_df['lon_bin'] = (work_df['lon'] // resolution_lon) * resolution_lon + (resolution_lon/2)
    work_df['time_bin'] = (work_df['time_days'] // resolution_time_days) * resolution_time_days
    
    # --- STEP 3: VERTICAL BINNING (VECTORIZED) ---
    # We define the target integration grid in METERS
    z_grid_edges = np.arange(depth_min, depth_max + vertical_step, vertical_step)
    z_grid_centers = z_grid_edges[:-1] + (vertical_step / 2)
    
    # Assign every point to a vertical layer using the 'depth' column (Meters)
    # This is crucial: We integrate over real depth, not pressure.
    work_df['z_idx'] = pd.cut(work_df['depth'], bins=z_grid_edges, labels=z_grid_centers, include_lowest=True)
    
    # Drop points outside our depth target
    work_df = work_df.dropna(subset=['z_idx'])
    
    # --- STEP 4: AGGREGATE "SYNTHETIC PROFILES" ---
    print(f"   ⚡ Collapsing {len(work_df)} points into 4D Grid (Time/Lat/Lon/Depth)...")
    
    # Create mean energy density for every (Time, Lat, Lon, Depth) box
    # This replaces the slow apply loop with a vectorized GroupBy
    grid_4d = work_df.groupby(['time_bin', 'lat_bin', 'lon_bin', 'z_idx'], observed=True)['energy_density'].mean()
    
    # --- STEP 5: INTEGRATE (PIVOT STRATEGY) ---
    # Unstack the Depth index to columns
    # Result: Rows = (Time, Lat, Lon), Columns = Depth Levels
    grid_wide = grid_4d.unstack(level='z_idx')
    
    # --- [NEW] SMART ROBUSTNESS CHECK ---
    # Instead of just counting filled bins, we check if the profile "spans" the water column.
    
    # 1. Calculate Fill Rate (Density)
    n_cols = grid_wide.shape[1]
    fill_rate = grid_wide.count(axis=1) / n_cols
    
    # 2. Check Surface & Deep Coverage
    # We require at least ONE data point in the top 10% and bottom 20% of the grid
    # to ensure we aren't extrapolating blindly.
    has_data_mask = grid_wide.notna()
    
    top_limit_idx = int(n_cols * 0.10) # Top 10%
    bot_limit_idx = int(n_cols * 0.80) # Bottom 20%
    
    # .any(axis=1) checks if ANY column in that range has data
    has_surface = has_data_mask.iloc[:, :top_limit_idx].any(axis=1)
    has_deep    = has_data_mask.iloc[:, bot_limit_idx:].any(axis=1)
    
    # 3. Combine Logic
    # We relax the strict percentage (using min_coverage_pct as a density floor)
    # but enforce that the profile must have endpoints (Surface + Deep).
    robust_mask = (fill_rate >= min_coverage_pct) & has_surface & has_deep
    
    grid_wide_filtered = grid_wide[robust_mask].copy()
    
    if grid_wide_filtered.empty:
        print(f"   ❌ No bins met coverage criteria (Surface+Deep+Density>{min_coverage_pct}).")
        return pd.DataFrame()

    print(f"   🌊 Integrating {len(grid_wide_filtered)} valid profiles...")

    # 3. Interpolate small vertical gaps
    grid_filled = grid_wide_filtered.interpolate(axis=1, limit_direction='both')
    
    # 4. TRAPEZOIDAL INTEGRATION
    #    dx is the vertical_step (constant)
    ohc_values = np.trapezoid(grid_filled.values, dx=vertical_step, axis=1)
    
    # --- STEP 6: FORMAT OUTPUT ---
    results = pd.DataFrame({
        'ohc': ohc_values
    }, index=grid_wide_filtered.index).reset_index()
    
    # --- [NEW] NORMALIZE BY DEPTH ---
    # Calculates Average Energy Density (J/m^3)
    total_depth_range = depth_max - depth_min
    results['ohc_per_m'] = results['ohc'] / total_depth_range
    
    # Add n_raw_points count for metadata
    raw_counts = work_df.groupby(['time_bin', 'lat_bin', 'lon_bin'], observed=True).size()
    raw_counts.name = 'n_raw_points'
    
    results = results.merge(raw_counts, on=['time_bin', 'lat_bin', 'lon_bin'], how='left')
    
    print(f"✅ DONE. Generated {len(results)} OHC estimates.")
    return results