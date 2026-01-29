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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings


"""
The following function provides the option for the global validation of a Gaussian Process Regression.
Use this if you are happy with coming up with one set of correlation lengths (one for lat, one for lon, optionally one for time)
for your system. 
"""
def generalized_cross_validation(df, feature_cols=['lat', 'lon'], target_col='temp', 
                                 method='KFold', k_fold_data_percent=10,
                                 auto_tune=True, tune_subsample_frac=0.05, tune_iterations=5,
                                 length_scale_val=1.0, noise_val=0.1):
    """
    Global GP Validation (Smart Hybrid).
    Splits the data into Train/Test sets and fits ONE Gaussian Process to the entire training set.
    
    ---------------------------------------------------------------------------
    HOW TO INTERPRET & TUNE (BASED ON K-FOLD):
    ---------------------------------------------------------------------------
    The critical metric is 'Std Z' (Standard Deviation of Z-scores). 
    Ideal value is 1.0.

    1. If Std Z > 1.1 (The "Arrogant" Model):
       - Diagnosis: Model is Overconfident. It predicts small error bars, but real errors are large.
       - Cause: It thinks distant points are more related than they actually are.
       - ACTION: DECREASE 'length_scale_val' (make it rougher) OR INCREASE 'noise_val'.

    2. If Std Z < 0.9 (The "Paranoid" Model):
       - Diagnosis: Model is Underconfident. It predicts huge error bars, but predictions are actually good.
       - Cause: It ignores useful neighbors nearby, assuming they aren't relevant.
       - ACTION: INCREASE 'length_scale_val' (make it smoother) OR DECREASE 'noise_val'.

    * Note regarding LOFO: LOFO Z-scores naturally fluctuate due to spatial non-stationarity. 
      Do not obsess over tuning LOFO to 1.0; use K-Fold for tuning physics.
    ---------------------------------------------------------------------------
    
    PARAMETERS:
    ---------------------------------------------------------------------------
    1. df (pd.DataFrame): 
       The master table. Must contain feature_cols, target_col, and 'float_id'.

    2. feature_cols (list of str): 
       Dimensions for similarity (e.g., ['lat', 'lon'] for 2D, or ['lat', 'lon', 'time_days'] for 3D).
       
    3. target_col (str): 
       The variable to predict (e.g., 'temp', 'psal').

    4. method (str):
       - 'KFold': Tests interpolation accuracy by holding out random points. Best for tuning.
       - 'LOFO':  Tests scientific rigor by holding out entire instruments.

    5. k_fold_data_percent (float): 
       Percentage of data to hold out for TESTING in each fold (e.g., 10%).

    6. auto_tune (bool): 
       - True:  Runs a pre-step on random subsets to LEARN the best length/noise.
                Ignores the manual knobs below. 
       - False: Uses the manual length_scale_val and noise_val.

    7. tune_subsample_frac (float): 
       Fraction of data to use for EACH auto-tune iteration (0.0 to 1.0).

    8. tune_iterations (int):
       How many times to run the optimizer on random subsets (Default: 5).
       The final parameters are the average of these runs.

    9. length_scale_val (float or list): 
       MANUAL knob in PHYSICAL UNITS (Degrees for Lat/Lon, Days for Time).
       Used only if auto_tune=False.
       
    10. noise_val (float): 
       MANUAL knob in PHYSICAL UNITS (Variance in Target Units^2).
       Used only if auto_tune=False.
    ---------------------------------------------------------------------------
    """
    print(f"\n🚀 STARTING GLOBAL VALIDATION: {method}")

    # --- SAFETY CHECK: DIMENSION MISMATCH ---
    if not auto_tune:
        # Check if length_scale_val is a list/array (Anisotropic)
        if hasattr(length_scale_val, '__len__') and not isinstance(length_scale_val, (str, float, int)):
            if len(length_scale_val) != len(feature_cols):
                raise ValueError(
                    f"\n❌ CONFIG ERROR: You provided {len(feature_cols)} feature columns {feature_cols}, "
                    f"but a length_scale_val list of size {len(length_scale_val)}: {length_scale_val}.\n"
                    f"👉 Please provide exactly one length scale per feature, or a single float for isotropic mode."
                )
    # 1. SETUP & SCALING
    X = df[feature_cols].values
    y = df[target_col].values
    # Robust Grouping: Force to string to prevent splitter errors
    groups = df['float_id'].astype(str).values 
    mean_lat = df['lat'].mean()
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    phys_scale_X = scaler_X.scale_  
    phys_scale_y = scaler_y.scale_[0]

    # ---------------------------------------------------------
    # 2. ROBUST AUTO-TUNE STEP
    # ---------------------------------------------------------
    
    # [CHANGED]: Automatically scale Manual Inputs from Physical -> Scaled Space
    if not auto_tune:
        # User provided Physical Units (e.g., 2.0 degrees), we need Scaled Units for the Kernel
        if hasattr(length_scale_val, '__len__'):
            final_length_scale = np.array(length_scale_val) / phys_scale_X
        else:
            final_length_scale = length_scale_val / phys_scale_X
            
        # User provided Physical Variance (e.g., 0.1 C^2), we need Scaled Variance
        # Var_scaled = Var_phys / Var_total
        final_noise = noise_val / scaler_y.var_
    else:
        # Placeholders that will be overwritten by auto-tuner
        final_length_scale = length_scale_val 
        final_noise = noise_val
    
    if auto_tune:
        N_points = len(X_scaled)

        target_n = int(N_points * tune_subsample_frac)

        if N_points < 100:
            n_sub = N_points
        else:
            n_sub = max(100, min(target_n, 2000))
        
        # Calculate Guardrails (Bounds)
        data_span = np.max(X_scaled, axis=0) - np.min(X_scaled, axis=0)
        max_dist = np.max(data_span) 
        upper_bound = max_dist * 1.5 
        lower_bound = 0.05 
        
        print(f"   🤖 AutoTuning: Running {tune_iterations} iterations on {n_sub} points ({tune_subsample_frac*100:.1f}%) to estimate correlation lengths/times...")
        print(f"      (Constraint: Length Scale capped at {upper_bound:.2f} standard deviations)")
        
        learned_ls = []
        learned_noise = []
        
        for run in range(tune_iterations):
            idx_tune = np.random.choice(N_points, n_sub, replace=False)
            X_tune, y_tune = X_scaled[idx_tune], y_scaled[idx_tune]
            
            k_tune = ConstantKernel(1.0) * \
                     RBF(length_scale=[1.0]*X.shape[1], length_scale_bounds=(lower_bound, upper_bound)) + \
                     WhiteKernel(noise_level=0.1)
            
            gp_tune = GaussianProcessRegressor(kernel=k_tune, n_restarts_optimizer=0)
            gp_tune.fit(X_tune, y_tune)
            
            learned_ls.append(gp_tune.kernel_.k1.k2.length_scale)
            learned_noise.append(gp_tune.kernel_.k2.noise_level)
            
        final_length_scale = np.mean(learned_ls, axis=0) 
        final_noise = np.mean(learned_noise)
        
        # Convert back to Physical for Printing
        phys_ls = final_length_scale * phys_scale_X
        phys_noise_sigma = np.sqrt(final_noise) * phys_scale_y
        
        print(f"      ✅ LEARNED HYPERPARAMETERS (Avg of {tune_iterations} runs):")
        print(f"         Noise (Uncertainty): ±{phys_noise_sigma:.3f} (physical units)")
        print(f"         Correlation Lengths:")
        
        for i, col in enumerate(feature_cols):
            val = phys_ls[i]
            if 'lat' in col.lower():
                km_val = val * 111.0
                print(f"           - {col}: {val:.3f}°  (~{km_val:.0f} km)")
            elif 'lon' in col.lower():
                km_val = val * 111.0 * np.cos(np.radians(mean_lat))
                print(f"           - {col}: {val:.3f}°  (~{km_val:.0f} km at {mean_lat:.1f}N)")
            elif 'time' in col.lower() or 'day' in col.lower():
                print(f"           - {col}: {val:.1f} days")
            else:
                print(f"           - {col}: {val:.3f} (unknown units)")
    else:
        # [CHANGED]: Updated print statement to show the User's Physical Inputs, not the Scaled Internals
        print(f"   🔧 Manual Mode: Using Fixed Physical Parameters:")
        print(f"      - Length Scales: {length_scale_val}")
        print(f"      - Noise Variance: {noise_val}")

    # 3. CHOOSE SPLITTER
    if method == 'LOFO':
        cv = LeaveOneGroupOut()
    #("LOOO" is not the technically accurate name for the method, but I mix it up sometimes with KFolding.)
    elif (method == 'KFold' or method=='LOOO'): 
        n_splits = int(100 / k_fold_data_percent)
        if n_splits < 2: n_splits = 2
        if n_splits >= len(df): n_splits = len(df)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"   ⚡ Strategy: {n_splits}-Fold CV (Testing {k_fold_data_percent}% per fold)")
    else:
        raise ValueError(f"Unknown method '{method}'. Please use 'LOFO' or 'KFold'.")
    
    # 4. RUN LOOP
    y_preds = []
    y_true = []
    y_sigmas = []
    MAX_TRAIN = 2000 
    
    # Explicitly pass groups to avoid LOFO error
    for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_scaled, groups=groups)):
        
        if len(train_idx) > MAX_TRAIN:
            train_subset = np.random.choice(train_idx, size=MAX_TRAIN, replace=False)
        else:
            train_subset = train_idx
            
        X_train = X_scaled[train_subset]
        y_train = y_scaled[train_subset]
        X_test = X_scaled[test_idx]
        
        k = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            RBF(length_scale=final_length_scale, length_scale_bounds="fixed") + \
            WhiteKernel(noise_level=final_noise, noise_level_bounds="fixed")
        
        gp = GaussianProcessRegressor(kernel=k, optimizer=None, alpha=0.0)
        gp.fit(X_train, y_train)
        
        pred, std = gp.predict(X_test, return_std=True)
        
        y_preds.extend(pred)
        y_sigmas.extend(std)
        y_true.extend(y_scaled[test_idx])
        
        if method == 'KFold':
             if cv.get_n_splits() > 10 and i % (cv.get_n_splits() // 10) == 0:
                 print(f"   Processed fold {i+1}...", end='\r')
        else:
             if i % 5 == 0:
                 print(f"   Processed float {i+1}...", end='\r')

    # 5. SCORING
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    y_sigmas = np.array(y_sigmas)
    
    if len(y_preds) == 0: return np.array([])

    z_scores = (y_true - y_preds) / y_sigmas
    y_true_c = scaler_y.inverse_transform(y_true.reshape(-1,1)).flatten()
    y_pred_c = scaler_y.inverse_transform(y_preds.reshape(-1,1)).flatten()
    
    rmse = np.sqrt(np.mean((y_true_c - y_pred_c)**2))
    
    epsilon = 1e-9
    rel_error_vector = (y_pred_c - y_true_c) / (y_true_c + epsilon)
    rms_rel_error = np.sqrt(np.mean(rel_error_vector**2))
    
    print(f"\n✅ RESULTS ({method}):")
    print(f"   RMSE:                {rmse:.3f} (units depend on what field you inputted)")
    print(f"   Rel. Error (RMSRE):  {rms_rel_error:.4f} (dimensionless)")
    print(f"   Mean Z:              {np.mean(z_scores):.3f}")
    print(f"   Std Z:               {np.std(z_scores):.3f} (Ideal: 1.0)")
    
    return {
        "z_scores": z_scores,
        "rmse": rmse,
        # [CHANGED]: Return Physical Units (interpretable) instead of Scaled Units
        "best_length_scale": final_length_scale * phys_scale_X, 
        "best_noise": final_noise * scaler_y.var_  
    }



"""
NOT COMPLETED YET - WILL USE VARIABLE LAT, LON, AND TIME

This is almost certaintly the best thing to use, but it is computationally incredibly costly because you study space and
time varying correlation distances. Save this for cloud computing.
"""

def validate_moving_window(df, feature_cols=['lat', 'lon'], target_col='temp', 
                           method='LOFO', k_fold_data_percent=10,
                           radius_km=300, min_neighbors=10, max_samples=1000,
                           auto_tune=True, tune_subsample_frac=0.05, tune_iterations=5,
                           length_scale_val=1.0, noise_val=0.1,
                           optimization_mode='group'): 
    """
    Moving Window (Local GP) Validation with Adaptive Optimization.
    
    PURPOSE:
    Validates the mapping strategy by simulating the final map generation process.
    It iterates through test points, identifies local neighbors within 'radius_km',
    and fits a unique Gaussian Process for that specific window.
    
    OPTIMIZATION STRATEGY ("The Goldilocks Approach"):
    1. Global Init: First, we estimate a baseline length scale from the full dataset.
    2. Local Adaptation: Then, based on 'optimization_mode', we adapt that baseline 
       to the local conditions (e.g., eddies vs gyres).
    
    ---------------------------------------------------------------------------
    PARAMETERS:
    ---------------------------------------------------------------------------
    1. df (pd.DataFrame): 
       The master data table. Must contain:
       - Feature columns (e.g., lat, lon)
       - Target column (e.g., temp)
       - 'float_id': Used for grouping in LOFO validation.

    2. feature_cols (list of str): 
       Dimensions used for similarity.
       - ['lat', 'lon']: Standard 2D mapping.
       - ['lat', 'lon', 'time_days']: 3D Spatio-Temporal mapping.

    3. target_col (str): 
       The variable to predict (e.g., 'temp', 'psal').

    4. method (str):
       - 'LOFO' (Recommended): "Leave-One-Float-Out". Tests scientific robustness 
         by holding out entire instruments. Use with optimization_mode='group'.
       - 'KFold': Random sampling. Tests interpolation accuracy. Use with 
         optimization_mode='point' or 'global'.

    5. k_fold_data_percent (float): 
       Percentage of data to test per fold (if method='KFold').

    6. radius_km (float): 
       The Horizon. 
       - Filters neighbors: Only points within this radius are used for training.
       - Bounds optimizer: Local length scales are forbidden from exceeding 
         2x this radius (prevents "Infinite Length Scale" artifacts).

    7. min_neighbors (int): 
       The Void Threshold. If a test point has fewer than this many neighbors,
       we skip prediction. Prevents unstable models in sparse regions.

    8. max_samples (int): 
       Speed Limit. Stops validation after testing this many total points.
       Essential for 'point' mode which is computationally expensive.

    9. auto_tune (bool): 
       - True: Runs a pre-loop Global Estimation step to find baseline parameters.
       - False: Uses manual 'length_scale_val' and 'noise_val' as the baseline.

    10. tune_subsample_frac (float): 
        Fraction of data to use for Global Estimation (e.g., 0.05 = 5%).

    11. tune_iterations (int):
        Number of random subsets to test during Global Estimation. 
        Averaging these runs provides a stable starting point for local models.

    12. length_scale_val / noise_val (float): 
        Manual knobs used only if auto_tune=False.

    13. optimization_mode (str) - THE CRITICAL PERFORMANCE KNOB:
       - 'group' (Recommended for LOFO): "Per-Float Optimization".
         Calculates local physics ONCE per float (averaging 3 points on the track),
         then locks those parameters to predict the rest of that float.
         * Speed: Fast (~50x faster than point).
         * Accuracy: High (Captures local physics of the water mass).
         
       - 'point' (Scientific Rigor): "Locally Stationary".
         Re-runs the optimizer for every single test point. 
         * Speed: Very Slow.
         * Accuracy: Maximum.
       
       - 'global' (Speed Check): "Globally Stationary".
         Uses the fixed Global Baseline parameters for every window.
         * Speed: Fastest.
         * Accuracy: Lower (Ignores that physics change spatially).
    ---------------------------------------------------------------------------
    """
    print(f"\n🚀 STARTING MOVING WINDOW VALIDATION: {method}")
    print(f"   Config: Radius={radius_km}km | Mode: {optimization_mode.upper()}")
    
    # 1. DATA PREP
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['float_id'].astype(str).values 
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    phys_scale_X = scaler_X.scale_
    phys_scale_y = scaler_y.scale_[0]
    X_rad = np.radians(X[:, :2]) 
    radius_rad = radius_km / 6371.0

    # 2. GLOBAL BASELINE (Initialization)
    start_length_scale = length_scale_val
    start_noise = noise_val
    
    # Calculate Max Distance for Bounds
    data_span = np.max(X_scaled, axis=0) - np.min(X_scaled, axis=0)
    max_dist = np.max(data_span)
    
    if auto_tune:
        print(f"   🤖 Global Estimator: Running {tune_iterations} iterations to find baseline...")
        N_points = len(X_scaled)
        n_sub = int(N_points * tune_subsample_frac)
        n_sub = max(100, min(n_sub, 2000))
        
        learned_ls = []
        learned_noise = []
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for run in range(tune_iterations):
                idx_tune = np.random.choice(N_points, n_sub, replace=False)
                X_tune, y_tune = X_scaled[idx_tune], y_scaled[idx_tune]
                
                k_tune = ConstantKernel(1.0) * \
                         RBF(length_scale=[1.0]*X.shape[1], length_scale_bounds=(0.05, max_dist*1.5)) + \
                         WhiteKernel(noise_level=0.1)
                
                gp_tune = GaussianProcessRegressor(kernel=k_tune, n_restarts_optimizer=0)
                gp_tune.fit(X_tune, y_tune)
                
                learned_ls.append(gp_tune.kernel_.k1.k2.length_scale)
                learned_noise.append(gp_tune.kernel_.k2.noise_level)
            
        start_length_scale = np.mean(learned_ls, axis=0) 
        start_noise = np.mean(learned_noise)
        print(f"      ✅ Baseline Found: Length={start_length_scale}, Noise={start_noise:.4f}")
    
    # 3. SPLITTER LOOP
    if method == 'LOFO':
        cv = LeaveOneGroupOut()
    elif method == 'KFold':
        # 'group' mode requires discrete groups. If KFold (random), force global or point.
        if optimization_mode == 'group':
            print("   ⚠️  NOTE: 'group' mode requires LOFO. Switching to 'global' for KFold.")
            optimization_mode = 'global'
        n_splits = int(100 / k_fold_data_percent)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_preds = []
    y_true = []
    y_sigmas = []
    learned_local_ls = [] 
    
    samples_processed = 0
    ignored_count = 0
    
    # Loop over Folds/Floats
    for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_scaled, groups=groups)):
        if max_samples and samples_processed >= max_samples: break
        
        # Subsample test set
        points_needed = max_samples - samples_processed if max_samples else len(test_idx)
        n_take = min(len(test_idx), points_needed)
        if n_take < len(test_idx):
             current_test_idx = np.random.choice(test_idx, size=n_take, replace=False)
        else:
             current_test_idx = test_idx

        # -----------------------------------------------------------
        # STRATEGY: GROUP OPTIMIZATION (Hybrid Mode)
        # -----------------------------------------------------------
        current_ls = start_length_scale
        current_noise = start_noise
        
        if optimization_mode == 'group':
            # 1. Pick up to 3 random representative points from this float/group
            sample_size = min(3, len(current_test_idx))
            sample_indices = np.random.choice(current_test_idx, size=sample_size, replace=False)
            
            group_ls_list = []
            group_noise_list = []
            
            # 2. Optimize on just these 3 points to "Learn the Float's Physics"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                
                for s_idx in sample_indices:
                    # Geometry
                    target_pt_rad = X_rad[s_idx].reshape(1, -1)
                    train_subset_rad = X_rad[train_idx]
                    dists = haversine_distances(train_subset_rad, target_pt_rad).flatten()
                    neighbor_mask = dists < radius_rad
                    valid_train_indices = train_idx[neighbor_mask]
                    
                    if len(valid_train_indices) < min_neighbors: continue
                    
                    # Local Bounds logic
                    avg_scale_km = np.mean(phys_scale_X) * 111.0
                    radius_scaled = radius_km / avg_scale_km
                    upper_bound = max(1.0, radius_scaled * 2.0)

                    # Optimize
                    X_loc = X_scaled[valid_train_indices]
                    y_loc = y_scaled[valid_train_indices]
                    k = ConstantKernel(1.0) * \
                        RBF(length_scale=start_length_scale, length_scale_bounds=(0.05, upper_bound)) + \
                        WhiteKernel(noise_level=start_noise)
                    
                    gp_opt = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=0)
                    gp_opt.fit(X_loc, y_loc)
                    
                    group_ls_list.append(gp_opt.kernel_.k1.k2.length_scale)
                    group_noise_list.append(gp_opt.kernel_.k2.noise_level)
            
            # 3. Average them to get the parameters for this entire float
            if len(group_ls_list) > 0:
                current_ls = np.mean(group_ls_list, axis=0)
                current_noise = np.mean(group_noise_list)

        # -----------------------------------------------------------
        # PREDICTION LOOP (Window by Window)
        # -----------------------------------------------------------
        for t_idx in current_test_idx:
            samples_processed += 1
            
            # A. Neighbors
            target_pt_rad = X_rad[t_idx].reshape(1, -1)
            target_feature = X_scaled[t_idx].reshape(1, -1)
            train_subset_rad = X_rad[train_idx]
            dists = haversine_distances(train_subset_rad, target_pt_rad).flatten()
            
            neighbor_mask = dists < radius_rad
            valid_train_indices = train_idx[neighbor_mask]
            
            if len(valid_train_indices) < min_neighbors:
                ignored_count += 1
                continue 
            
            X_local = X_scaled[valid_train_indices]
            y_local = y_scaled[valid_train_indices]
            
            # B. Kernel Setup
            if optimization_mode == 'point':
                # Mode A: Re-Optimize every point (Scientific)
                optimizer_setting = 0
                bounds_setting = (0.05, max(1.0, (radius_km/(np.mean(phys_scale_X)*111.0))*2.0))
                # Use Global as start, but allow optimization
                ls_use = start_length_scale
                noise_use = start_noise
                
            else:
                # Mode B ('group') or C ('global'): Use FIXED parameters
                # We use the 'current_ls' we calculated above (either from group avg or global)
                optimizer_setting = None
                bounds_setting = "fixed"
                ls_use = current_ls
                noise_use = current_noise
            
            # C. Fit & Predict
            k = ConstantKernel(1.0, constant_value_bounds="fixed") * \
                RBF(length_scale=ls_use, length_scale_bounds=bounds_setting) + \
                WhiteKernel(noise_level=noise_use, noise_level_bounds=bounds_setting)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gp = GaussianProcessRegressor(kernel=k, optimizer=optimizer_setting, alpha=0.0)
                gp.fit(X_local, y_local)
            
            pred, std = gp.predict(target_feature, return_std=True)
            y_preds.append(pred[0])
            y_sigmas.append(std[0])
            y_true.append(y_scaled[t_idx])
            
            # Track what we used
            if optimization_mode == 'point':
                learned_local_ls.append(np.mean(gp.kernel_.k1.k2.length_scale))
            else:
                learned_local_ls.append(np.mean(current_ls))

        print(f"   Samples processed: {samples_processed}...", end='\r')

    # 4. SCORING
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    y_sigmas = np.array(y_sigmas)
    
    if len(y_preds) == 0: return np.array([])

    z_scores = (y_true - y_preds) / y_sigmas
    y_true_c = scaler_y.inverse_transform(y_true.reshape(-1,1)).flatten()
    y_pred_c = scaler_y.inverse_transform(y_preds.reshape(-1,1)).flatten()
    rmse = np.sqrt(np.mean((y_true_c - y_pred_c)**2))
    rms_rel_error = np.sqrt(np.mean(((y_pred_c - y_true_c) / (y_true_c + 1e-9))**2))
    
    print(f"\n✅ RESULTS ({method}):")
    print(f"   Avg Local LS:        {np.mean(learned_local_ls):.2f} (scaled)")
    print(f"   RMSE (Valid):        {rmse:.3f} °C")
    print(f"   Rel. Error (RMSRE):  {rms_rel_error:.4f}")
    print(f"   Mean Z:              {np.mean(z_scores):.3f}")
    print(f"   Std Z:               {np.std(z_scores):.3f}")
    
    return z_scores




    """
    PHASE 3: PRODUCTION MAPPER (The Generator).
    
    TO DO: IF YOU WANT TO KEEP USING THIS CODE (WHICH WORKS WITH LOARD_ARGO_DATA_ADVANCED,
      OR POSSIBLY TAKES OUTPUT FROM ArgoHeatContentDataCollider.estimate_ohc_from_raw_bins()), BEST
      LOOK INTO ADDING COASTLINES WITH THE HELP OF ARGOPY. ALTERNATIVELY, WE COULD MAKE ANOTHER
      FUNCTION THAT FORMS A NETCDF VIEWER.


    Generates a continuous Gridded Map (NetCDF/Xarray) from sparse Argo data
    using the "Fixed Kernel" parameters tuned in the Validation phase. DOES NOT 
    INCLUDE COASTLINES YET. 

    TAKES OUTPUT FROM load_argo_data_advanced()

    
    
    -------------------------------------------------------------------------
    STRATEGY: "Integrate First, Map Second"
    -------------------------------------------------------------------------
    Instead of 4D Kriging (Lat, Lon, Depth, Time), we rely on the user passing
    pre-integrated layers (e.g., 'ohc_source'). This allows us to map each 
    physical layer with its own unique correlation length (e.g., Surface = Chaotic, 
    Deep = Smooth).
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Input data. MUST contain:
        - 'lat', 'lon' : Spatial coordinates (degrees).
        - 'time_days'  : Numeric time (e.g., days since start).
        - target_col   : The variable to interpolate (e.g., 'ohc_source').
        
    grid_lat, grid_lon : 1D arrays
        The spatial mesh you want to produce (e.g., np.arange(30, 40, 0.5)).
        
    grid_time : 1D array
        The time steps you want to produce (must match units of 'time_days').
        
    target_col : str
        The specific column in 'df' to map (e.g. 'ohc_response').
        
    radius_km : float
        The "Horizon" of the model. Points further than this are ignored 
        to save compute time. (Standard: ~300km).
        
    final_length_scale : float or array-like
        The physical correlation length (in SIGMAS) you found during validation.
        Can be a scalar (isotropic) or an array matching dimensions (anisotropic).
        
    final_noise : float
        The noise floor (uncertainty) you found during validation.
        (e.g., 0.1).
        
    is_3d : bool
        If True, includes 'time_days' in the distance calculation.
        
    time_buffer: int
        Number of days we are including in each time data point in our trend. Basically a time_buffer number of days
        will be combined to make a monthly map for us to study trends. This buffer smooths out our kriged fields. If
        you set this number too small, any time a float enters the window, there will be a sharp spike in the field
        and in the uncertainty. Too large and you're wasting computational power. We are expecting long correlation times
        for climate scale behavior, so the default is 60 days
    RETURNS:
    --------
    xr.Dataset
        A 3D Xarray (Time, Lat, Lon) containing:
        - 'temp' (Prediction)
        - 'uncertainty' (Standard Deviation / Error Bars)
    """

def produce_kriging_map(df, 
                        grid_lat, grid_lon, grid_time,
                        target_col='temp',
                        radius_km=300, min_neighbors=5,
                        final_length_scale=1.0, final_noise=0.1,
                        is_3d=True, time_buffer = 60):
   
    
    # ---------------------------------------------------------
    # 1. SETUP & SCALING
    # ---------------------------------------------------------
    # We must scale inputs so Lat (deg), Lon (deg), and Time (days) 
    # are treated equally by the isotropic kernel.
    feature_cols = ['lat', 'lon', 'time_days'] if is_3d else ['lat', 'lon']
    
    print(f"\n🗺️ STARTING PRODUCTION MAPPING for '{target_col}'...")
    if np.ndim(final_length_scale) == 0:
        print(f"   Using Kernel: Length={final_length_scale:.3f} (Sigmas), Noise={final_noise:.3f}")
    else:
        print(f"   Using Kernel: Length={final_length_scale} (Sigmas), Noise={final_noise:.3f}")
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit the scaler on ALL data to establish the global coordinate system
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Pre-calculate radians for Lat/Lon to use fast Haversine distance later
    # (Only needed for the spatial dimensions 0 and 1)
    X_rad_space = np.radians(X[:, :2]) 
    radius_rad = radius_km / 6371.0  # Convert km to Earth Radians
    
    # ---------------------------------------------------------
    # 2. DEFINE THE PHYSICS (THE KERNEL)
    # ---------------------------------------------------------
    # We use a FIXED kernel. We do NOT optimize (fit) inside the loop.
    # The parameters (length_scale, noise) are hard-coded from your Validation results.
    # optimizer=None ensures the model doesn't try to "re-learn" physics at every pixel.
    k = ConstantKernel(1.0, "fixed") * \
        RBF(length_scale=final_length_scale, length_scale_bounds="fixed") + \
        WhiteKernel(noise_level=final_noise, noise_level_bounds="fixed")
    
    # ---------------------------------------------------------
    # 3. PREPARE THE OUTPUT GRID
    # ---------------------------------------------------------
    shape = (len(grid_time), len(grid_lat), len(grid_lon))
    map_mean = np.full(shape, np.nan) # The Prediction Map
    map_std = np.full(shape, np.nan)  # The Uncertainty Map
    
    total_slices = len(grid_time)
    
    # ---------------------------------------------------------
    # 4. THE MAIN LOOP (Time -> Space)
    # ---------------------------------------------------------
    for t_i, t_val in enumerate(grid_time):
        
        # --- OPTIMIZATION A: TEMPORAL FILTER ---
        # Instead of searching the entire dataset for every pixel, 
        # we first grab only floats that exist near this time step.
        # This reduces the matrix size from N=10,000 to N=200, making it fast.
        
        if is_3d:
            # We filter for data within +/- 60 days (default) (2 months) of the target date.
            # Why 60? It's a safe buffer. The kernel (length_scale) will naturally 
            # downweight points far away in time, but we hard-cut them here for speed.
            time_col_idx = 2
            # Note: We must look at RAW time (X[:,2]) not scaled X_scaled yet
            time_mask = np.abs(X[:, time_col_idx] - t_val) < time_buffer
            
            # If no data exists in this window, skip the whole month (leave as NaNs)
            if np.sum(time_mask) < min_neighbors:
                print(f"   Skipping Slice {t_i+1}/{total_slices} (Not enough data)...", end='\r')
                continue
                
            X_subset = X_scaled[time_mask]
            y_subset = y_scaled[time_mask]
            X_rad_subset = X_rad_space[time_mask]
        else:
            # 2D Mode: Use all data (Climatology)
            X_subset = X_scaled
            y_subset = y_scaled
            X_rad_subset = X_rad_space

        # Loop through Space (Lat/Lon)
        for lat_i, lat_val in enumerate(grid_lat):
            for lon_i, lon_val in enumerate(grid_lon):
                
                # --- OPTIMIZATION B: SPATIAL FILTER ---
                # Use Haversine distance to find neighbors within radius_km
                target_rad = np.radians([[lat_val, lon_val]])
                dists = haversine_distances(X_rad_subset, target_rad).flatten()
                
                # Create the local neighborhood mask
                mask = dists < radius_rad
                
                # If this pixel is in a "Void" (no nearby floats), leave it as NaN.
                # This explicitly identifies the "Observationally Opaque" regions.
                if np.sum(mask) < min_neighbors: continue
                
                # --- THE KRIGING STEP ---
                # 1. Prepare Target Point (Lat, Lon, Time) scaled to global norms
                coords = [lat_val, lon_val, t_val] if is_3d else [lat_val, lon_val]
                target_scaled = scaler_X.transform([coords])
                
                # 2. Instantiate GP with FIXED Physics
                gp = GaussianProcessRegressor(kernel=k, optimizer=None, alpha=0.0)
                
                # 3. Fit ONLY to the local neighborhood (Subset of Subset)
                gp.fit(X_subset[mask], y_subset[mask])
                
                # 4. Predict
                pred, std = gp.predict(target_scaled, return_std=True)
                
                # 5. Inverse Transform (Back to real units: Joules or Deg C)
                # Note: We must handle scaler shapes carefully
                map_mean[t_i, lat_i, lon_i] = scaler_y.inverse_transform([[pred[0]]])[0][0]
                
                # Uncertainty scales linearly, so we multiply by the scaler's scale factor
                map_std[t_i, lat_i, lon_i] = std[0] * scaler_y.scale_[0]
        
        # Progress Bar
        print(f"   Mapped Slice {t_i+1}/{total_slices} (Time={t_val:.0f})...", end='\r')

    print("\n✅ Mapping Complete.")
    
    # ---------------------------------------------------------
    # 5. EXPORT TO XARRAY
    # ---------------------------------------------------------
    # We package the 3D numpy arrays into a labelled Data Cube
    return xr.Dataset(
        data_vars={
            target_col: (["time", "lat", "lon"], map_mean),
            f"{target_col}_uncert": (["time", "lat", "lon"], map_std)
        },
        coords={
            "time": grid_time,
            "lat": grid_lat,
            "lon": grid_lon
        },
        attrs={
            "description": f"Kriged Map of {target_col}",
            "kernel_length_scale": final_length_scale,
            "kernel_noise": final_noise,
            "units": "Joules/m^2" if "ohc" in target_col else "degC"
        }
    )




"""
FOLLOWING CODE IS FOR PERFORMING A ROLLING WINDOW ANALYSIS IN TIME. THIS GETS US 
CORRELATIONS IN LATITUDE AND LONGITUDE THAT DEPEND ON TIME
"""


"""

PIPELINE: TAKES OUTPUT FROM ArgoHeatContentDataCollider.estimate_ohc_from_raw_bins(). 

Feeds into ag.plot_kriging_snapshot()
    Performs a "Rolling Window" Gaussian Process analysis to assess how ocean physics 
    (spatial correlation) and model reliability change over time.
    
    This function is compatible with both RAW profile data (lat/lon) and BINNED 
    OHC data (lat_bin/lon_bin). It slides a time window across the dataset, learning
    local physics (Length Scales) and calibrating uncertainty (Noise).

    Parameters
    ----------
    df : pd.DataFrame
        The master dataset. Must contain columns corresponding to feature_cols, target_col, 
        and time_col. 
    
    feature_cols : list of str, default=['lat_bin', 'lon_bin']
        The dimensions used to calculate similarity (distance).
        - If input is raw data: Use ['lat', 'lon']
        - If input is binned OHC: Use ['lat_bin', 'lon_bin']
        *NOTE*: The function attempts to auto-detect. If 'lat_bin' is requested but missing,
        it will look for 'lat' automatically.
        
    target_col : str, default='ohc_per_m'
        The variable to be mapped/interpolated.
        - For OHC analysis: 'ohc' or 'ohc_per_m'
        - For Temperature: 'temp'
        
    time_col : str, default='time_bin'
        The column used to SLICE the data into rolling windows.
        *NOTE*: If 'time_bin' is requested but missing, it checks for 'time_days'.
        
    k_fold_data_percent : float, default=10
        The percentage of data in each window to hide from the model for validation.
        (e.g., 10 means 10% of points are Test set, 90% are Training set).
        
    auto_tune : bool, default=True
        - True: The GP optimizer is allowed to find the best length_scale and noise_level 
          (Maximum Likelihood Estimation).
        - False: The model is forced to use the manual length_scale_val and noise_val. 
          (Use this to test if a fixed physics model holds up over time).
          
    tune_subsample_frac : float, default=0.1
        (Not currently used in this version but kept for consistency) Fraction of data used for tuning.
        
    tune_iterations : int, default=5
        Number of times to restart the optimizer to avoid local minima. 
        Maps to 'n_restarts_optimizer'.
        
    length_scale_val : float or list, default=1.0
        - If auto_tune=True: This is the starting guess for the optimizer.
        - If auto_tune=False: This is the fixed, hard-coded length scale used for all windows.
          If providing a list, must match len(feature_cols).
          
    noise_val : float, default=0.1
        - If auto_tune=True: This is the starting guess for the noise level (variance).
        - If auto_tune=False: This is the fixed noise level.
        
    window_size_days : int, default=90
        The duration of data included in one analysis block.
        (e.g., 90 days captures a seasonal snapshot).
        
    step_size_days : int, default=30
        How far to slide the window forward for the next step.
        (e.g., 30 days means we produce monthly updates).
        
    auto_calibrate : bool, default=True
        If True, enables the "Feedback Loop". If the model is over/under-confident 
        (Z-score outside bounds), it adjusts the noise parameter and re-fits 
        to ensure reliable error bars.
        
    target_z_bounds : tuple, default=(0.9, 1.1)
        The "Goldilocks Zone" for the Standard Deviation of Z-scores.
        - Ideal is 1.0. 
        - < 0.9 means the model is paranoid (errors too big).
        - > 1.1 means the model is arrogant (errors too small).
        
    target_rmsre : float, default=0.05
        The target Root Mean Squared Relative Error (e.g., 0.05 = 5% error).
        Used to flag "Low Quality" windows in the output.
        
    max_adjust_steps : int, default=3
        Maximum number of calibration loops to run per window to prevent infinite cycles.

    Returns
    -------
    results_df : pd.DataFrame
        Summary statistics for each time window. Columns include:
        - 'rmse': Root Mean Square Error (Accuracy)
        - 'rmsre': Relative Error (Quality)
        - 'std_z': Reliability (Z-score)
        - 'scale_lat', 'scale_lon': The learned physical correlation lengths (in degrees).
        - 'noise_val': The calibrated noise level.
        
    cv_details : dict
        A dictionary keyed by the window center date (float).
        Values are DataFrames containing the raw True vs Predicted values for every test point.
        Useful for deep-dive statistical debugging (e.g., checking for bias).
    """
def analyze_rolling_correlations(df, 
                                 # --- DATA INPUTS ---
                                 feature_cols=['lat_bin', 'lon_bin'], # Adjusted defaults for binned data
                                 target_col='ohc_per_m',              # Adjusted default target
                                 time_col='time_bin',                 # Adjusted default time column
                                 
                                 # --- VALIDATION STRATEGY ---
                                 k_fold_data_percent=10,      
                                 
                                 # --- OPTIMIZATION (HYPERPARAMETERS) ---
                                 auto_tune=True,
                                 tune_subsample_frac=0.1,     
                                 tune_iterations=5,           
                                 length_scale_val=1.0,        
                                 noise_val=0.1,               
                                 
                                 # --- ROLLING WINDOW CONFIG ---
                                 window_size_days=90, 
                                 step_size_days=30,
                                 
                                 # --- AUTO-CALIBRATION (RELIABILITY CONTROL) ---
                                 auto_calibrate=True,
                                 target_z_bounds=(0.9, 1.1),
                                 target_rmsre=0.05,          
                                 max_adjust_steps=3
                                 ):
    """
    Performs a "Rolling Window" Gaussian Process analysis to assess how ocean physics 
    (spatial correlation) and model reliability change over time.
    
    Compatible with output from 'estimate_ohc_from_raw_bins'.
    """
    
    print(f"🕵️ STARTING ROLLING ANALYSIS")
    print(f"   Window: {window_size_days}d | Step: {step_size_days}d | Validation: {k_fold_data_percent}%")
    print(f"   Targets: RMSRE < {target_rmsre} | Std Z in {target_z_bounds}")
    
    # --- 1. SETUP TIMELINE ---
    # Ensure time column exists
    if time_col not in df.columns:
        # Fallback for raw data if 'time_bin' is missing but 'time_days' exists
        if 'time_days' in df.columns:
            print(f"   ⚠️ '{time_col}' not found. Using 'time_days' instead.")
            time_col = 'time_days'
        else:
            raise ValueError(f"Time column '{time_col}' not found in dataframe.")

    # Check for feature columns (handle bin vs raw names)
    final_features = []
    for col in feature_cols:
        if col in df.columns:
            final_features.append(col)
        elif col.replace('_bin', '') in df.columns:
            # If 'lat_bin' missing but 'lat' exists, use 'lat'
            alt_col = col.replace('_bin', '')
            print(f"   ⚠️ '{col}' not found. Using '{alt_col}' instead.")
            final_features.append(alt_col)
        else:
             raise ValueError(f"Feature column '{col}' not found in dataframe.")
    feature_cols = final_features

    t_min = df[time_col].min()
    t_max = df[time_col].max()
    history = []     # To store high-level metrics
    cv_details = {}  # To store raw validation data
    
    current_t = t_min
    
    # Calculate test_size fraction once (e.g., 10% -> 0.1)
    test_split_frac = k_fold_data_percent / 100.0
    
    # --- 2. MAIN ROLLING LOOP ---
    # We iterate until the window hits the end of the dataset
    while current_t < t_max - (window_size_days/2):
        
        # A. Slice the Data (The "Moving Horizon")
        t_end = current_t + window_size_days
        mask = (df[time_col] >= current_t) & (df[time_col] < t_end)
        df_slice = df[mask].copy()
        
        # Guardrail: Skip windows that are too empty to model safely
        # We lower this threshold slightly since binned data is already aggregated
        if len(df_slice) < 10: 
            current_t += step_size_days
            continue
            
        # B. Prepare Feature Matrices
        X = df_slice[feature_cols].values
        y = df_slice[target_col].values
        
        # Scale Data (Crucial for GP convergence)
        # Note: We fit a FRESH scaler for every window. This is correct because
        # we care about variance *within this season*, not global variance.
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # C. Split Train/Test (In-Window Validation)
        # We assume interpolation within the window, so random split is valid.
        if len(df_slice) < 20:
             # If very few points, leave-one-out style or just skip test split to avoid errors
             # For robustness here, we force a minimal split or skip
             if len(df_slice) < 5:
                 current_t += step_size_days
                 continue
             X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.5, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, 
                test_size=test_split_frac, 
                random_state=42
            )
        
        # --- 3. CONFIGURE KERNEL & PHYSICS ---
        if auto_tune:
            # "Scientific Discovery Mode"
            l_bounds = (1e-2, 5) 
            n_bounds = (1e-5, 1e1)
            optimizer_restarts = tune_iterations
        else:
            # "Verification Mode"
            l_bounds = "fixed"
            n_bounds = "fixed"
            optimizer_restarts = 0
            
        # Handle isotropic (scalar) vs anisotropic (list) input for length_scale_val
        if isinstance(length_scale_val, list):
            ls_init = length_scale_val 
        else:
            ls_init = [length_scale_val] * len(feature_cols)

        # Define the Kernel (Physics Engine)
        k = ConstantKernel(1.0, constant_value_bounds="fixed") * \
            RBF(length_scale=ls_init, length_scale_bounds=l_bounds) + \
            WhiteKernel(noise_level=noise_val, noise_level_bounds=n_bounds)
            
        gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=optimizer_restarts, alpha=0.0)
        
        try:
            # --- 4. INITIAL FIT ---
            # Maximize Log-Likelihood to find best physics parameters
            gp.fit(X_train, y_train)
            
            # --- 5. AUTO-CALIBRATION LOOP ---
            # Even if the physics are right, the noise (error bars) might be wrong.
            # This loop adjusts the noise until Z-scores are healthy (~1.0).
            
            # Logic: Only calibrate if requested AND if we are allowed to tune (auto_tune=True)
            should_calibrate = auto_calibrate and auto_tune
            
            best_rmsre = 999.0
            best_std_z = 999.0
            
            # Capture the learned physics (Length Scales) from the initial fit.
            # We will LOCK these during calibration. We only want to widen/shrink
            # the error bars (Noise), not change the shape of the map.
            current_length_scale = gp.kernel_.k1.k2.length_scale
            current_noise = gp.kernel_.k2.noise_level
            
            # Determine how many passes to make (1 pass if no calibration, else max_steps)
            steps_to_run = max_adjust_steps + 1 if should_calibrate else 1
            
            for step in range(steps_to_run):
                
                # a. Predict on Hold-Out Set
                y_pred, y_std = gp.predict(X_test, return_std=True)
                
                # b. Convert to Physical Units (Degrees C, Joules, etc.)
                y_pred_phys = scaler_y.inverse_transform(y_pred.reshape(-1,1)).flatten()
                y_test_phys = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()
                y_std_phys = y_std * scaler_y.scale_[0] # Scale sigma appropriately
                
                # c. Calculate Accuracy Metric: RMSRE (Relative Error)
                epsilon = 1e-9 # Avoid division by zero
                rel_error_vector = (y_pred_phys - y_test_phys) / (y_test_phys + epsilon)
                rmsre = np.sqrt(np.mean(rel_error_vector**2))
                
                # d. Calculate Reliability Metric: Z-Score
                # Z = (Actual Error) / (Predicted Uncertainty)
                z_scores = (y_test_phys - y_pred_phys) / (y_std_phys + epsilon)
                std_z = np.std(z_scores)
                
                # e. Check Acceptance Criteria
                z_ok = (std_z >= target_z_bounds[0]) and (std_z <= target_z_bounds[1])
                
                # Stop if targets met, or if it's the last allowed step
                if (z_ok) or (step == steps_to_run - 1):
                    best_rmsre = rmsre
                    best_std_z = std_z
                    break
                
                # f. Feedback Control (Adjust Noise)
                # If Std Z > 1.0 (Overconfident), we need MORE noise.
                # Correction Factor ~ Z^2 (since Variance ~ Sigma^2)
                correction_factor = std_z**2
                correction_factor = np.clip(correction_factor, 0.5, 2.0) # Dampen to prevent explosions
                
                new_noise = current_noise * correction_factor
                current_noise = new_noise
                
                # g. Re-Fit with NEW Noise (but FIXED Length Scales)
                # optimizer=None ensures we don't waste time re-solving the physics
                k_calibrated = ConstantKernel(1.0, constant_value_bounds="fixed") * \
                               RBF(length_scale=current_length_scale, length_scale_bounds="fixed") + \
                               WhiteKernel(noise_level=new_noise, noise_level_bounds="fixed")
                
                gp = GaussianProcessRegressor(kernel=k_calibrated, optimizer=None, alpha=0.0)
                gp.fit(X_train, y_train)
                
            # --- END CALIBRATION LOOP ---

            # --- 6. RECORD RESULTS ---
            window_center = current_t + (window_size_days/2)
            
            # Convert Learned Length Scales back to Physical Units (Degrees/Days)
            learned_ls_phys = current_length_scale * scaler_X.scale_
            
            record = {
                'window_start': current_t,
                'window_center': window_center,
                'rmsre': best_rmsre,
                'std_z': best_std_z,
                'noise_val': gp.kernel_.k2.noise_level, # Final calibrated noise
                'n_points': len(df_slice)
            }
            # Save specific dimension scales (Lat vs Lon vs Time)
            for i, col in enumerate(feature_cols):
                record[f'scale_{col}'] = learned_ls_phys[i]
            
            history.append(record)
            
            # Store Raw Data for deep statistical checking
            cv_details[window_center] = pd.DataFrame({
                'y_true': y_test_phys, 
                'y_pred': y_pred_phys, 
                'rel_err': rel_error_vector,
                'z_score': z_scores
            })
            
            # --- 7. PRINT STATUS ---
            # Z-Score Status
            icon_z = "✅" if (best_std_z >= target_z_bounds[0] and best_std_z <= target_z_bounds[1]) else "⚠️"
            # Quality Status (RMSRE)
            icon_qual = "✅" if best_rmsre <= target_rmsre else "❌"
            
            print(f"   Window {int(current_t)}-{int(t_end)}: {icon_qual} RMSRE={best_rmsre:.3%} | {icon_z} Z={best_std_z:.2f}")
            
        except Exception as e:
            print(f"   ❌ Fit Failed for window {int(current_t)}: {e}")
            
        # Move sliding window forward
        current_t += step_size_days

    # --- 8. FINALIZE ---
    results_df = pd.DataFrame(history)
    return results_df, cv_details



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