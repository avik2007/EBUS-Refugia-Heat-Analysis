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
        p_var: 'depth',   # <--- Renamed as requested
        'LATITUDE': 'lat',
        'LONGITUDE': 'lon',
        'TIME': 'date',
        'PLATFORM_NUMBER': 'float_id'
    })
    
    # --- C. FILTERING & CLEANING ---
    print(f"   🧹 Cleaning {len(df_raw)} raw points...")
    
    # 1. Drop NaNs in critical columns (Using 'depth' now)
    df = df_raw.dropna(subset=['temp', 'psal', 'depth', 'lat', 'lon', 'date'])
    
    # 2. Decode Byte Strings (Float IDs)
    if df['float_id'].dtype == object and isinstance(df['float_id'].iloc[0], bytes):
        df['float_id'] = df['float_id'].str.decode('utf-8').str.strip()
    df['float_id'] = df['float_id'].astype(str)

    # 3. Standardize Longitude (-180 to 180)
    df.loc[df['lon'] > 180, 'lon'] -= 360
    
    # 4. Spatial Filter
    df = df[
        (df['lat'] >= lat_bounds[0]) & (df['lat'] <= lat_bounds[1]) &
        (df['lon'] >= lon_bounds[0]) & (df['lon'] <= lon_bounds[1])
    ]
    
    # 5. Calculate Numeric Time (Days since start)
    df['time_days'] = (df['date'] - ref_date).dt.total_seconds() / 86400.0
    
    # 6. Sort for clean integration later (Sorting by depth)
    df = df.sort_values(by=['float_id', 'date', 'depth'])
    
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
        Must contain 'lat', 'lon', 'time_days', 'depth', 'temp', 'psal'.
    vertical_step : int
        The vertical resolution (in meters) to smooth the pooled data before integrating.
        Default is 10m (standard for OHC).
    min_coverage_pct : float
        Percentage of vertical bins that must have data. If a bin has a huge gap 
        (e.g., data at 0m and 1000m only), returns NaN.
        
    Returns:
    --------
    df_binned : pd.DataFrame
        One row per Lat/Lon/Time bin with 'ohc', 'n_points', etc.
    """

def estimate_ohc_from_raw_bins(df, 
                               resolution_lat=1.0, 
                               resolution_lon=1.0, 
                               resolution_time_days=30,
                               depth_min=0, 
                               depth_max=2000,
                               vertical_step=10, # Vertical resolution for integration (m)
                               min_coverage_pct=0.7):
    
    
    print(f"📦 BINNING RAW DATA: {resolution_lat}° x {resolution_lon}° x {resolution_time_days} days...")
    
    # --- STEP 1: CALCULATE THERMODYNAMICS (PER POINT) ---
    # We do this globally first because it's vectorized (fast)
    work_df = df.copy()
    
    # GSW Calculations (TEOS-10 Standard)
    # 1. Absolute Salinity
    work_df['SA'] = gsw.SA_from_SP(work_df['psal'].values, work_df['depth'].values, 
                                   work_df['lon'].values, work_df['lat'].values)
    # 2. Conservative Temperature
    work_df['CT'] = gsw.CT_from_t(work_df['SA'].values, work_df['temp'].values, work_df['depth'].values)
    
    # 3. Density & Specific Heat
    rho = gsw.rho(work_df['SA'].values, work_df['CT'].values, work_df['depth'].values)
    cp = gsw.cp_t_exact(work_df['SA'].values, work_df['temp'].values, work_df['depth'].values)
    
    # 4. Energy Density (Joules / m^3)
    # This represents how much heat is in a 1m cube of water at this point
    # We assume 0 deg C as the reference for 'Heat Content' (standard in oceanography)
    work_df['energy_density'] = rho * cp * work_df['CT']
    
    # --- STEP 2: ASSIGN SPATIAL BINS ---
    # We use simple rounding to create grid IDs
    work_df['lat_bin'] = (work_df['lat'] // resolution_lat) * resolution_lat + (resolution_lat/2)
    work_df['lon_bin'] = (work_df['lon'] // resolution_lon) * resolution_lon + (resolution_lon/2)
    work_df['time_bin'] = (work_df['time_days'] // resolution_time_days) * resolution_time_days
    
    # --- STEP 3: THE AGGREGATION FUNCTION ---
    # This runs once for every 3D box
    def integrate_bin(group):
        # 1. Define Standard Vertical Grid for this Layer
        z_grid = np.arange(depth_min, depth_max + vertical_step, vertical_step)
        
        # 2. Assign each raw point to a vertical level
        # This handles the "cloud" of points from different floats
        group['z_idx'] = np.digitize(group['depth'], z_grid)
        
        # 3. Calculate Mean Energy Density per vertical level
        # This collapses the multiple floats into one "Synthetic Profile"
        profile_means = group.groupby('z_idx')['energy_density'].mean()
        
        # 4. Reindex to the full grid (fills gaps with NaN)
        # We need indices 1 to len(z_grid)
        full_profile = profile_means.reindex(range(1, len(z_grid)), fill_value=np.nan)
        
        # 5. Robustness: Check for Gaps
        # If we are missing too much of the water column, don't guess.
        valid_fraction = full_profile.notna().mean()
        if valid_fraction < min_coverage_pct:
            return np.nan
        
        # 6. Interpolate small gaps (e.g., if missing 10-20m but have 0m and 30m)
        full_profile_interp = full_profile.interpolate(method='linear', limit_direction='both')
        
        # 7. Integrate (Trapezoidal Rule)
        # OHC = Integral( Energy_Density * dz )
        ohc = np.trapezoid(full_profile_interp.values, dx=vertical_step)
        
        return ohc

    # --- STEP 4: EXECUTE ---
    print(f"   ... Grouping and Integrating (Layer: {depth_min}-{depth_max}m)...")
    
    # Group by the bins and apply the custom integration
    # Note: This step can take a moment if you have thousands of bins
    results = work_df.groupby(['time_bin', 'lat_bin', 'lon_bin']).apply(
        integrate_bin, include_groups=False
    ).reset_index(name='ohc')
    
    # Add counts for metadata
    counts = work_df.groupby(['time_bin', 'lat_bin', 'lon_bin']).size().reset_index(name='n_raw_points')
    results = pd.merge(results, counts, on=['time_bin', 'lat_bin', 'lon_bin'])
    
    # Remove bins that failed the robustness check
    results = results.dropna(subset=['ohc'])
    
    print(f"✅ DONE. Generated {len(results)} valid bin estimates.")
    return results