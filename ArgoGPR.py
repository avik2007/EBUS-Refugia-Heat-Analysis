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
import numpy as np

from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import numpy as np

"""
The following function provides the option for the global validation of a Gaussian Process Regression.
Use this if you are happy with coming up with one set of correlation lengths (one for lat, one for lon)
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

    9. length_scale_val (float): 
       MANUAL knob. Used only if auto_tune=False.
       
    10. noise_val (float): 
       MANUAL knob. Used only if auto_tune=False.
    ---------------------------------------------------------------------------
    """
    print(f"\n🚀 STARTING GLOBAL VALIDATION: {method}")
    
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
    final_length_scale = length_scale_val
    final_noise = noise_val
    
    if auto_tune:
        N_points = len(X_scaled)
        n_sub = int(N_points * tune_subsample_frac)
        n_sub = max(100, min(n_sub, 2000))
        
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
        
        phys_ls = final_length_scale * phys_scale_X
        phys_noise_sigma = np.sqrt(final_noise) * phys_scale_y
        
        print(f"      ✅ LEARNED HYPERPARAMETERS (Avg of {tune_iterations} runs):")
        print(f"         Noise (Uncertainty): ±{phys_noise_sigma:.3f} °C")
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
        print(f"   🔧 Using Manual Parameters: Length={final_length_scale}, Noise={final_noise}")

    # 3. CHOOSE SPLITTER
    if method == 'LOFO':
        cv = LeaveOneGroupOut()
    elif method == 'KFold':
        n_splits = int(100 / k_fold_data_percent)
        if n_splits < 2: n_splits = 2
        if n_splits >= len(df): n_splits = len(df)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"   ⚡ Strategy: {n_splits}-Fold CV (Testing {k_fold_data_percent}% per fold)")

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
    print(f"   RMSE:                {rmse:.3f} °C")
    print(f"   Rel. Error (RMSRE):  {rms_rel_error:.4f} (dimensionless)")
    print(f"   Mean Z:              {np.mean(z_scores):.3f}")
    print(f"   Std Z:               {np.std(z_scores):.3f} (Ideal: 1.0)")
    
    return z_scores




def validate_moving_window(df, feature_cols=['lat', 'lon'], target_col='temp', 
                           method='KFold', k_fold_data_percent=10,
                           radius_km=300, min_neighbors=10, max_samples=1000,
                           auto_tune=True, tune_subsample_frac=0.05, tune_iterations=5,
                           length_scale_val=1.0, noise_val=0.1):
    """
    Locally Stationary Moving Window Validation.
    
    PURPOSE:
    Test the mapping strategy by mimicking the final map generation process.
    Unlike global methods, this fits a unique Gaussian Process for every single 
    test point ("Locally Stationary" assumption).
    
    STRATEGY (Global Init -> Local Opt):
    1. Runs a Global Auto-Tune on random subsets to estimate baseline parameters 
       (the "Global Ruler").
    2. For every test point, initializes a local GP with those baseline parameters.
    3. OPTIMIZES the local GP to fit the specific neighbors within 'radius_km'.
    
    ---------------------------------------------------------------------------
    PARAMETERS:
    ---------------------------------------------------------------------------
    1. df (pd.DataFrame): 
       The master data table. Must contain columns for:
       - Features (e.g., 'lat', 'lon')
       - Target (e.g., 'temp')
       - Grouping ID ('float_id') used for LOFO validation.

    2. feature_cols (list of str): 
       The dimensions used to calculate distance/similarity.
       - ['lat', 'lon']: Standard 2D spatial mapping.
       - ['lat', 'lon', 'time_days']: 3D Spatio-Temporal mapping.

    3. target_col (str): 
       The variable you are trying to map (e.g., 'temp', 'psal', 'doxy').

    4. method (str):
       - 'KFold': Randomly holds out 10% of points. Best for testing "Interpolation" 
         (filling small gaps). Expected Z-Score Std is ~1.0.
       - 'LOFO': "Leave-One-Float-Out". Holds out entire instruments. Best for testing 
         "Reconstruction" (scientific robustness). Z-Scores will fluctuate due to 
         spatial non-stationarity.

    5. k_fold_data_percent (float): 
       Only used if method='KFold'. Determines the size of the test set per fold.
       (e.g., 10 means 10% of data is used for testing).

    6. radius_km (float): 
       The "Horizon of Influence".
       - Data Filtering: Only neighbors within this distance are used for training.
       - Physics Bound: The local optimizer is forbidden from choosing a Length Scale 
         larger than 2x this radius (prevents the "Infinite Length Scale" error).

    7. min_neighbors (int): 
       The "Void Threshold". If a test point has fewer than this many neighbors 
       within radius_km, we skip prediction. (Standard GP requires ~10 points to be stable).

    8. max_samples (int): 
       Speed Limit. Since this runs a full optimization for every point, it is slow.
       This stops the validation after testing 'max_samples' points (e.g., 1000).

    9. auto_tune (bool): 
       - True: Runs the Global Initialization step to find good starting parameters.
       - False: Initializes local models with manual 'length_scale_val' and 'noise_val'.

    10. tune_subsample_frac (float): 
        (Auto-Tune only) The fraction of total data to use for global estimation.
        0.05 (5%) is usually sufficient to find the average physics.

    11. tune_iterations (int):
        (Auto-Tune only) How many random subsets to test during Global Initialization.
        We average the results to get a stable starting point. Default: 5.

    12. length_scale_val / noise_val (float): 
        Manual knobs used only if auto_tune=False.
        Useful if you already know the physics and want to skip the pre-calc step.
    ---------------------------------------------------------------------------
    """
    print(f"\n🚀 STARTING LOCALLY STATIONARY VALIDATION: {method}")
    print(f"   Config: Radius={radius_km}km | Optimizing every point...")
    
    # 1. DATA PREP
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['float_id'].astype(str).values 
    
    # Standard Scaling (Crucial for Optimizer convergence)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Capture physical scale for un-scaling later
    phys_scale_X = scaler_X.scale_
    phys_scale_y = scaler_y.scale_[0]
    
    # Pre-calc Radians for Haversine (Assumes lat/lon are first 2 cols)
    X_rad = np.radians(X[:, :2]) 
    EARTH_RADIUS_KM = 6371.0
    radius_rad = radius_km / EARTH_RADIUS_KM

    # ---------------------------------------------------------
    # 2. GLOBAL INITIALIZATION (The "Educated Guess")
    # ---------------------------------------------------------
    # We find the "Global Average" parameters to use as the STARTING POINT 
    # for every local optimizer. This prevents local minima.
    start_length_scale = length_scale_val
    start_noise = noise_val
    
    if auto_tune:
        N_points = len(X_scaled)
        n_sub = int(N_points * tune_subsample_frac)
        n_sub = max(100, min(n_sub, 2000))
        
        # Global Guardrails (Box Size)
        data_span = np.max(X_scaled, axis=0) - np.min(X_scaled, axis=0)
        max_dist = np.max(data_span)
        
        print(f"   🤖 Global Estimator: Running {tune_iterations} iterations to find baseline...")
        
        learned_ls = []
        learned_noise = []
        
        for run in range(tune_iterations):
            idx_tune = np.random.choice(N_points, n_sub, replace=False)
            X_tune, y_tune = X_scaled[idx_tune], y_scaled[idx_tune]
            
            # Allow global optimizer to search up to 1.5x the full box size
            k_tune = ConstantKernel(1.0) * \
                     RBF(length_scale=[1.0]*X.shape[1], length_scale_bounds=(0.05, max_dist*1.5)) + \
                     WhiteKernel(noise_level=0.1)
            
            gp_tune = GaussianProcessRegressor(kernel=k_tune, n_restarts_optimizer=0)
            gp_tune.fit(X_tune, y_tune)
            
            learned_ls.append(gp_tune.kernel_.k1.k2.length_scale)
            learned_noise.append(gp_tune.kernel_.k2.noise_level)
            
        start_length_scale = np.mean(learned_ls, axis=0) 
        start_noise = np.mean(learned_noise)
        
        print(f"      ✅ Baseline Found (will initialize local models):")
        print(f"         Length: {start_length_scale}")
        print(f"         Noise:  {start_noise:.4f}")
    else:
        print(f"   🔧 Manual Baseline: Length={start_length_scale}, Noise={start_noise}")

    # ---------------------------------------------------------
    # 3. CROSS VALIDATION LOOP (The "Swarm")
    # ---------------------------------------------------------
    if method == 'LOFO':
        cv = LeaveOneGroupOut()
    elif method == 'KFold':
        n_splits = int(100 / k_fold_data_percent)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_preds = []
    y_true = []
    y_sigmas = []
    learned_local_ls = [] 
    
    samples_processed = 0
    ignored_count = 0
    
    # We rely on cv.split yielding indices. For LOFO, 'groups' is required.
    for i, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_scaled, groups=groups)):
        if max_samples and samples_processed >= max_samples: break
        
        # Subsample the test set to stay under max_samples limit
        points_needed = max_samples - samples_processed if max_samples else len(test_idx)
        n_take = min(len(test_idx), points_needed)
        if n_take < len(test_idx):
             current_test_idx = np.random.choice(test_idx, size=n_take, replace=False)
        else:
             current_test_idx = test_idx

        # --- LOCAL OPTIMIZATION LOOP ---
        for t_idx in current_test_idx:
            samples_processed += 1
            
            # A. Find Neighbors
            target_pt_rad = X_rad[t_idx].reshape(1, -1)
            target_feature = X_scaled[t_idx].reshape(1, -1)
            train_subset_rad = X_rad[train_idx]
            dists = haversine_distances(train_subset_rad, target_pt_rad).flatten()
            
            neighbor_mask = dists < radius_rad
            valid_train_indices = train_idx[neighbor_mask]
            
            # B. Check Density
            if len(valid_train_indices) < min_neighbors:
                ignored_count += 1
                continue 
            
            X_local = X_scaled[valid_train_indices]
            y_local = y_scaled[valid_train_indices]
            
            # C. Define Kernel with LOCAL GUARDRAILS
            # The length scale cannot exceed 2x the Radius. 
            # This prevents the linear trend trap (Infinite length scale).
            avg_scale_km = np.mean(phys_scale_X) * 111.0
            radius_scaled = radius_km / avg_scale_km
            upper_bound = max(1.0, radius_scaled * 2.0)
            
            # INITIALIZATION: 
            # Start at 'start_length_scale' (Global Baseline), but allow optimization.
            k = ConstantKernel(1.0, (1e-1, 1e2)) * \
                RBF(length_scale=start_length_scale, length_scale_bounds=(0.05, upper_bound)) + \
                WhiteKernel(noise_level=start_noise, noise_level_bounds=(1e-5, 1.0))
            
            # D. Fit (Optimizer ON)
            # We use 0 restarts for speed because we have a high-quality starting point.
            gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=0)
            gp.fit(X_local, y_local)
            
            pred, std = gp.predict(target_feature, return_std=True)
            
            y_preds.append(pred[0])
            y_sigmas.append(std[0])
            y_true.append(y_scaled[t_idx])
            
            # Record what the local model learned (avg of dims)
            learned_local_ls.append(np.mean(gp.kernel_.k1.k2.length_scale))

        print(f"   Samples processed: {samples_processed}...", end='\r')

    # 4. SCORING & DIAGNOSTICS
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
    print(f"   Avg Local LS:        {np.mean(learned_local_ls):.2f} (scaled units)")
    print(f"   RMSE (Valid):        {rmse:.3f} °C")
    print(f"   Rel. Error (RMSRE):  {rms_rel_error:.4f}")
    print(f"   Mean Z:              {np.mean(z_scores):.3f}")
    print(f"   Std Z:               {np.std(z_scores):.3f} (Ideal: 1.0)")
    
    return z_scores