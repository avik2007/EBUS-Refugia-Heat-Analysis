import os
import numpy as np
import pandas as pd
import scipy.stats as stats

def dm_test(matern_rmsre, gibbs_rmsre, window_size_days=45.0, step_size_days=10.0, lag=None):
    """
    Diebold-Mariano test with Newey-West (HAC) variance correction and 
    Harvey-Leybourne-Newbold (HLN 1997) small-sample modification.

    Physical & Statistical Significance:
    -----------------------------------
    In rolling-window cross-validation for oceanographic GPR fields, successive 
    evaluation windows overlap in time whenever step_size_days < window_size_days.
    This introduces positive autocorrelation in the loss differential series d_t.
    The standard Diebold-Mariano (1995) test uses an asymptotic normal distribution,
    which is severely oversized (overstates statistical significance) when sample 
    sizes are small (N ~ 30-40).

    Harvey, Leybourne, and Newbold (1997) introduced a finite-sample correction factor
    and proved that comparing the modified statistic against Student's t-distribution
    with (N - 1) degrees of freedom restores correct nominal test size without 
    sacrificing power.

    Parameters:
    -----------
    matern_rmsre : np.ndarray
        Time series of RMSRE loss values from stationary Matérn GPR.
    gibbs_rmsre : np.ndarray
        Time series of RMSRE loss values from Gibbs non-stationary GPR.
    window_size_days : float, default=45.0
        Temporal window width of the rolling GPR evaluation in physical days.
    step_size_days : float, default=10.0
        Stride between successive evaluation window centers in physical days.
    lag : int, optional
        Maximum autocorrelation lag (h). If None, derived dynamically as
        h = int(np.floor((window_size_days - 1.0) / step_size_days)).

    Returns:
    --------
    dm_stat : float
        Original asymptotic Diebold-Mariano statistic.
    dm_stat_hln : float
        Harvey-Leybourne-Newbold small-sample modified statistic DM*.
    p_two_sided : float
        Two-sided p-value using Student's t(N-1) distribution.
    p_one_sided : float
        One-sided p-value (H1: Gibbs is superior / lower loss) using t(N-1).
    lag : int
        The autocorrelation lag h used for HAC variance and HLN adjustment.
    """
    d = np.asarray(matern_rmsre) - np.asarray(gibbs_rmsre)
    N = len(d)
    if N < 2:
        raise ValueError(f"Sample size N={N} is too small for Diebold-Mariano test.")

    mean_d = np.mean(d)

    # Dynamically derive maximum overlap lag h if not explicitly specified
    if lag is None:
        lag = int(np.floor((window_size_days - 1.0) / step_size_days))
        # Ensure lag does not exceed N - 2
        lag = max(1, min(lag, N - 2))
    
    # Calculate sample autocovariances gamma_k
    gamma = np.zeros(lag + 1)
    for k in range(lag + 1):
        if k == 0:
            gamma[0] = np.mean((d - mean_d) ** 2)
        else:
            gamma[k] = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
            
    # Calculate HAC variance (Newey-West spectral estimator with Bartlett kernel weights)
    var_d = gamma[0]
    for k in range(1, lag + 1):
        weight = 1.0 - (k / (lag + 1))
        var_d += 2.0 * weight * gamma[k]
        
    # Standard error of the mean differential
    se_d = np.sqrt(max(var_d, 1e-12) / N)
    
    # Asymptotic DM statistic
    dm_stat = mean_d / se_d
    
    # Harvey-Leybourne-Newbold (1997) small-sample correction factor
    # HLN_factor = sqrt((N + 1 - 2*h + h*(h-1)/N) / N)
    h = lag
    hln_inner = (N + 1.0 - 2.0 * h + (h * (h - 1.0)) / N) / N
    hln_factor = np.sqrt(max(hln_inner, 1e-12))
    dm_stat_hln = dm_stat * hln_factor

    # Degrees of freedom for Student's t-distribution
    df = N - 1

    # p-values using Student's t(N-1) distribution
    p_two_sided = 2.0 * stats.t.sf(np.abs(dm_stat_hln), df=df)
    p_one_sided = stats.t.sf(dm_stat_hln, df=df)  # H1: Gibbs is superior (d > 0)
    
    return dm_stat, dm_stat_hln, p_two_sided, p_one_sided, lag

def block_bootstrap_ci(matern, gibbs, block_size=5, n_boot=10000, confidence=0.95):
    """
    Overlapping Block Bootstrap for autocorrelated time series.
    Returns the confidence interval for the mean loss differential.
    """
    d = matern - gibbs
    N = len(d)
    boot_means = []
    
    # Possible block start indices
    start_indices = np.arange(N - block_size + 1)
    num_blocks = int(np.ceil(N / block_size))
    
    np.random.seed(42) # For reproducibility
    for _ in range(n_boot):
        # Sample block start indices with replacement
        selected_starts = np.random.choice(start_indices, size=num_blocks, replace=True)
        # Reconstruct bootstrap series
        boot_sample = []
        for start in selected_starts:
            boot_sample.extend(d[start:start+block_size])
        # Trim to length N
        boot_sample = np.array(boot_sample[:N])
        boot_means.append(np.mean(boot_sample))
        
    alpha = 1.0 - confidence
    ci_lower = np.percentile(boot_means, alpha / 2 * 100)
    ci_upper = np.percentile(boot_means, (1.0 - alpha / 2) * 100)
    
    return ci_lower, ci_upper

def compare_layers():
    aelogs_dir = "AEResults/aelogs"
    layers = [
        ("Skin Layer (0-100m)", "d0_100"),
        ("Source Layer (150-400m)", "d150_400"),
        ("Background Layer (500-1000m)", "d500_1000")
    ]
    
    print("==========================================================================")
    print("  STATISTICAL AUDIT: GIBBS (TIMELSFIX) vs STATIONARY MATERN (HLN-CORRECTED)")
    print("==========================================================================")
    
    for layer_title, layer_suffix in layers:
        matern_folder = f"californiav3_20150101_20151231_res0_5x0_5_t10_0_{layer_suffix}_3dmatern_w45_3dmatern_w45"
        gibbs_folder = f"californiav3_20150101_20151231_res0_5x0_5_t10_0_{layer_suffix}_3dgibbs_w45_timelsfix"
        
        matern_path = os.path.join(aelogs_dir, matern_folder, f"audit_{matern_folder}.csv")
        gibbs_path = os.path.join(aelogs_dir, gibbs_folder, f"audit_{gibbs_folder}.csv")
        
        if not os.path.exists(matern_path) or not os.path.exists(gibbs_path):
            print(f"\nMissing data for {layer_title}!")
            continue
            
        matern_df = pd.read_csv(matern_path)
        gibbs_df = pd.read_csv(gibbs_path)
        
        # Merge on window_center to ensure matched pairs
        merged = pd.merge(
            matern_df[['window_center', 'rmsre', 'std_z']], 
            gibbs_df[['window_center', 'rmsre', 'std_z']], 
            on='window_center', 
            suffixes=('_matern', '_gibbs')
        )
        
        n_matched = len(merged)
        
        # Calculate RMSRE stats
        med_matern = merged['rmsre_matern'].median()
        med_gibbs = merged['rmsre_gibbs'].median()
        mean_matern = merged['rmsre_matern'].mean()
        mean_gibbs = merged['rmsre_gibbs'].mean()
        
        # Calculate relative median improvement
        rel_improvement = (med_matern - med_gibbs) / med_matern * 100
        
        # Wilcoxon test (naive/optimistic, doesn't account for autocorrelation)
        wilcox_stat, wilcox_p = stats.wilcoxon(merged['rmsre_matern'], merged['rmsre_gibbs'])
        
        # Diebold-Mariano test with HLN small-sample correction and dynamic lag
        window_size_days = 45.0
        step_size_days = 10.0
        dm_stat, dm_stat_hln, dm_p_two, dm_p_one, lag_used = dm_test(
            merged['rmsre_matern'].values, 
            merged['rmsre_gibbs'].values, 
            window_size_days=window_size_days,
            step_size_days=step_size_days
        )
        
        # Block Bootstrap CI (block_size=5 ≈ 45/10)
        ci_low, ci_high = block_bootstrap_ci(merged['rmsre_matern'].values, merged['rmsre_gibbs'].values, block_size=5)
        
        # Calculate Std Z-Score stats
        z_matern_mean = merged['std_z_matern'].mean()
        z_gibbs_mean = merged['std_z_gibbs'].mean()
        z_matern_std = merged['std_z_matern'].std()
        z_gibbs_std = merged['std_z_gibbs'].std()
        
        print(f"\n--- {layer_title} (N = {n_matched} matched windows) ---")
        print(f"  RMSRE:")
        print(f"    Matérn:  median = {med_matern:.4f} ({med_matern*100:.2f}%), mean = {mean_matern:.4f} ({mean_matern*100:.2f}%)")
        print(f"    Gibbs:   median = {med_gibbs:.4f} ({med_gibbs*100:.2f}%), mean = {mean_gibbs:.4f} ({mean_gibbs*100:.2f}%)")
        print(f"    Absolute Delta: {mean_matern - mean_gibbs:.4f}")
        print(f"    Relative Median Improvement: {rel_improvement:.2f}%")
        print(f"  Uncertainty Calibration (Std Z-Score):")
        print(f"    Matérn:  mean = {z_matern_mean:.4f}, std = {z_matern_std:.4f}")
        print(f"    Gibbs:   mean = {z_gibbs_mean:.4f}, std = {z_gibbs_std:.4f} (Ideal: mean ≈ 1.0, lower variance is better)")
        print(f"  Statistical Significance (RMSRE Differential):")
        print(f"    Wilcoxon Signed-Rank p-val (naive):      {wilcox_p:.2e}")
        print(f"    Asymptotic DM stat (lag={lag_used}):              {dm_stat:.3f}")
        print(f"    HLN-Corrected DM* stat:                 {dm_stat_hln:.3f}")
        print(f"    HLN-Corrected p-val (two-sided, t-dist): {dm_p_two:.4e}")
        print(f"    HLN-Corrected p-val (one-sided, Gibbs):  {dm_p_one:.4e}")
        print(f"    95% Block-Bootstrap CI of Delta:         [{ci_low:.5f}, {ci_high:.5f}]")
        
        # Scientific validation
        if dm_p_one < 0.05:
            print("  ==> VERDICT: Gibbs kernel is STATISTICALLY SUPERIOR under HLN small-sample t-test (p < 0.05).")
        else:
            print("  ==> VERDICT: No statistically significant difference under HLN correction (p >= 0.05).")

if __name__ == '__main__':
    compare_layers()
