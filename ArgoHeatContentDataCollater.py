import numpy as np
import os
import argopy
from argopy import DataFetcher
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gsw
from datetime import datetime   


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

def load_argo_data_advanced(nc_dir, start_date, end_date, lat_bounds, lon_bounds):
    """
    Loads Argo data preserving Time and Float ID for LOFO & Spatio-Temporal analysis.
    
    Args:
        nc_dir (str): Path to folder containing Argo NetCDF files.
        start_date, end_date (str): 'YYYY-MM-DD' range.
        lat_bounds, lon_bounds (list): [min, max] for spatial cropping.
        
    Returns:
        pd.DataFrame: Columns [lat, lon, temp, time_days, float_id, date]
    """
    all_rows = []
    
    # Reference date for continuous time (e.g., Days since Start)
    ref_date = pd.to_datetime(start_date)

    print(f"📂 Scanning {nc_dir} for floats ({start_date} to {end_date})...")
    
    for filename in os.listdir(nc_dir):
        if not filename.endswith(".nc"): continue
        filepath = os.path.join(nc_dir, filename)
        
        try:
            ds = xr.open_dataset(filepath)
            
            # --- 1. SPATIAL SUBSET (Fast Filter) ---
            mask_space = (
                (ds.LATITUDE >= lat_bounds[0]) & (ds.LATITUDE <= lat_bounds[1]) &
                (ds.LONGITUDE >= lon_bounds[0]) & (ds.LONGITUDE <= lon_bounds[1])
            )
            if not mask_space.any():
                ds.close(); continue
            
            ds_subset = ds.where(mask_space, drop=True)
            
            # --- 2. TEMPORAL SUBSET ---
            ds_subset = ds_subset.sel(TIME=slice(start_date, end_date))
            if len(ds_subset.TIME) == 0:
                ds.close(); continue

            # --- 3. EXTRACT METADATA (ID) ---
            # Try to get ID from attribute or filename
            try:
                # Often stored as byte string in PLATFORM_NUMBER
                raw_id = ds_subset.PLATFORM_NUMBER.values[0]
                if isinstance(raw_id, bytes):
                    float_id = raw_id.decode('utf-8').strip()
                else:
                    float_id = str(raw_id).strip()
            except:
                float_id = filename.split("_")[0] # Fallback to filename

            # --- 4. EXTRACT DATA ---
            # Prefer Adjusted Temp, fall back to Raw Temp
            t_var = 'TEMP_ADJUSTED' if 'TEMP_ADJUSTED' in ds_subset else 'TEMP'
            
            lats = ds_subset.LATITUDE.values
            lons = ds_subset.LONGITUDE.values
            times = ds_subset.TIME.values
            temps = ds_subset[t_var].values
            
            # Handle Depth: If data has depth dim, take surface (index 0)
            if temps.ndim > 1:
                temps = temps[:, 0]

            for i in range(len(temps)):
                t_val = temps[i]
                if np.isnan(t_val): continue
                
                # Convert Timestamp to "Days since Start"
                dt = pd.to_datetime(times[i])
                days_delta = (dt - ref_date).total_seconds() / 86400.0
                
                all_rows.append({
                    'lat': float(lats[i]),
                    'lon': float(lons[i]),
                    'temp': float(t_val),
                    'time_days': float(days_delta), # Continuous variable for GP
                    'float_id': float_id,           # Grouping variable for LOFO
                    'date': dt                      # Human readable for debugging
                })
            
            ds.close()

        except Exception as e:
            continue

    df = pd.DataFrame(all_rows)
    print(f"✅ LOAD COMPLETE:")
    print(f"   - Observations: {len(df)}")
    print(f"   - Unique Floats: {df['float_id'].nunique()}")
    print(f"   - Time Span: {df['time_days'].min():.1f} to {df['time_days'].max():.1f} days")
    return df