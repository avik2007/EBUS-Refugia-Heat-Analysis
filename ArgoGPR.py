import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics.pairwise import haversine_distances

def generalized_cross_validation(df, feature_cols, target_col='temp', method='LOOO'):
    """
    Runs Cross-Validation on Argo Data.
    
    Parameters:
    - df: DataFrame from the loader.
    - feature_cols: List of columns to use as X (e.g., ['lat','lon'] or ['lat','lon','time_days'])
    - method: 'LOOO' (Leave-One-Observation-Out) or 'LOFO' (Leave-One-Float-Out)
    
    Returns:
    - z_scores: Array of Z-scores for every prediction.
    """
    print(f"\n🚀 STARTING VALIDATION: {method} | Features: {feature_cols}")
    
    # 1. Prepare Data
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['float_id'].values # Only used if method='LOFO'
    
    # Scale (Crucial for GP)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 2. Select Splitter
    if method == 'LOFO':
        cv = LeaveOneGroupOut()
        print(f"   🛡️  Strategy: Hold out entire Float IDs (Groups).")
    elif method == 'LOOO':
        # KFold with n_splits=len(df) is mathematically identical to LOOO
        # but for speed on large datasets, you can set n_splits=10 or 20 (K-Fold)
        cv = KFold(n_splits=min(len(df), 100)) 
        print(f"   ⚠️  Strategy: K-Fold/LOOO (Holding out observations).")
    else:
        raise ValueError("Method must be 'LOOO' or 'LOFO'")

    # 3. Validation Loop
    y_preds = np.zeros_like(y)
    y_sigmas = np.zeros_like(y)
    
    # cv.split yields indices
    for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_scaled, groups=groups)):
        
        # Split
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
        
        # Fit GP
        # Use a generic anisotropic kernel. 
        # The length_scale array size matches number of features automatically.
        k = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.1, 10.0)) * RBF(length_scale=[1.0]*X.shape[1]) + WhiteKernel(noise_level=0.1)
        
        # optimizer=None speeds this up significantly for validation loops
        gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=0) 
        gp.fit(X_train, y_train)
        
        # Predict
        pred_scaled, std_scaled = gp.predict(X_test, return_std=True)
        
        # Store 
        y_preds[test_idx] = pred_scaled
        y_sigmas[test_idx] = std_scaled
        
        if i % 10 == 0:
            print(f"   ...processed fold {i}", end='\r')

    # 4. Calculate Metrics (in Scaled Space for Z-score consistency)
    z_scores = (y_scaled - y_preds) / y_sigmas
    
    # Convert predictions back to Celsius for RMSE
    y_pred_celsius = scaler_y.inverse_transform(y_preds.reshape(-1,1)).flatten()
    rmse = np.sqrt(np.mean((y - y_pred_celsius)**2))
    
    print("\n" + "-" * 40)
    print(f"✅ RESULTS ({method}):")
    print(f"   RMSE:   {rmse:.3f} °C")
    print(f"   Mean Z: {np.mean(z_scores):.3f} (Ideal: 0.0)")
    print(f"   Std Z:  {np.std(z_scores):.3f}  (Ideal: 1.0)")
    print("-" * 40)
    
    return z_scores

def validate_moving_window(df, feature_cols=['lat', 'lon'], target_col='temp', 
                           method='LOOO', radius_km=300, min_neighbors=5):
    """
    Validates the 'Moving Window' approach.
    
    Parameters:
    - min_neighbors (int): Minimum # of training neighbors required to make a prediction.
                           If neighbors < min, the point is counted as 'Ignored/Void'.
    """
    print(f"\n🚀 STARTING MOVING WINDOW VALIDATION: {method}")
    print(f"   Config: Radius={radius_km}km | Min Neighbors={min_neighbors}")
    
    # 1. Prepare Data
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['float_id'].values 
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Pre-calculate Radians (Assuming X[:,0]=Lat, X[:,1]=Lon)
    X_rad = np.radians(X[:, :2]) 
    EARTH_RADIUS_KM = 6371.0
    radius_rad = radius_km / EARTH_RADIUS_KM

    # 2. Select Splitter
    if method == 'LOFO':
        cv = LeaveOneGroupOut()
    else:
        # Batching LOOO (KFold with many splits) is faster than n_splits=len(df)
        # but statistically similar. 
        cv = KFold(n_splits=min(len(df), 50)) 

    y_preds = np.full_like(y, np.nan)
    y_sigmas = np.full_like(y, np.nan)
    
    # Counter for voids
    ignored_count = 0
    total_points_tested = 0
    
    # 3. The Validation Loop
    for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_scaled, groups=groups)):
        
        # Loop through EVERY point in the current test batch
        for t_idx in test_idx:
            total_points_tested += 1
            
            # A. Identify Target
            target_pt_rad = X_rad[t_idx].reshape(1, -1)
            target_feature = X_scaled[t_idx].reshape(1, -1)
            
            # B. Find Neighbors in Training Set
            # Distance from this test point to ALL training points
            train_subset_rad = X_rad[train_idx]
            dists = haversine_distances(train_subset_rad, target_pt_rad).flatten()
            
            # Boolean Mask
            neighbor_mask = dists < radius_rad
            valid_train_indices = train_idx[neighbor_mask]
            
            # C. Count Check
            if len(valid_train_indices) < min_neighbors:
                # VOID DETECTED
                ignored_count += 1
                y_preds[t_idx] = np.nan
                y_sigmas[t_idx] = np.nan
                continue
                
            # D. Fit Local GP (Only if enough neighbors)
            X_local = X_scaled[valid_train_indices]
            y_local = y_scaled[valid_train_indices]
            
            # Fast Local GP (Optimizer off for speed in validation)
            k = ConstantKernel(1.0) * RBF(length_scale=[1.0]*X.shape[1]) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=k, optimizer=None, alpha=0.0)
            gp.fit(X_local, y_local)
            
            # E. Predict
            pred, std = gp.predict(target_feature, return_std=True)
            y_preds[t_idx] = pred[0]
            y_sigmas[t_idx] = std[0]

        # Progress bar
        if i % 10 == 0:
            print(f"   Processed batch {i}...", end='\r')

    # 4. Results Calculation
    valid_mask = ~np.isnan(y_preds)
    
    # Fraction Ignored
    void_fraction = ignored_count / total_points_tested if total_points_tested > 0 else 0
    
    # Metrics (Only calculated on NON-IGNORED points)
    if np.sum(valid_mask) > 0:
        z_scores = (y_scaled[valid_mask] - y_preds[valid_mask]) / y_sigmas[valid_mask]
        
        y_true_c = scaler_y.inverse_transform(y_scaled[valid_mask].reshape(-1,1)).flatten()
        y_pred_c = scaler_y.inverse_transform(y_preds[valid_mask].reshape(-1,1)).flatten()
        rmse = np.sqrt(np.mean((y_true_c - y_pred_c)**2))
        mean_z = np.mean(z_scores)
        std_z = np.std(z_scores)
    else:
        rmse, mean_z, std_z = np.nan, np.nan, np.nan
        z_scores = np.array([])

    print(f"\n✅ MOVING WINDOW RESULTS ({method}):")
    print("-" * 40)
    print(f"   Total Profiles Tested:   {total_points_tested}")
    print(f"   Profiles in Voids:       {ignored_count}")
    print(f"   Void Fraction:           {void_fraction*100:.1f}% (Effective Data Loss)")
    print("-" * 40)
    print(f"   RMSE (Valid Areas):      {rmse:.3f} °C")
    print(f"   Mean Z (Valid Areas):    {mean_z:.3f} (Ideal: 0.0)")
    print(f"   Std Z (Valid Areas):     {std_z:.3f}  (Ideal: 1.0)")
    print("-" * 40)
    
    return z_scores