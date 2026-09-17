import os
import numpy as np
import pandas as pd
import scipy.stats as stats

def dm_test(matern_rmsre, gibbs_rmsre, lag=4):
    """
    Diebold-Mariano test with Newey-West (HAC) variance correction.
    Null Hypothesis H0: Matern and Gibbs have equal forecast accuracy.
    Alternative Hypothesis H1: Gibbs is more accurate (one-sided) or different (two-sided).
    Loss differential d_t = Matern_t - Gibbs_t (positive means Gibbs is better).
    """
    d = matern_rmsre - gibbs_rmsre
    N = len(d)
    mean_d = np.mean(d)
    
    # Calculate autocovariances gamma_k
    gamma = np.zeros(lag + 1)
    for k in range(lag + 1):
        if k == 0:
            gamma[0] = np.mean((d - mean_d) ** 2)
        else:
            gamma[k] = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
            
    # Calculate HAC variance (Newey-West estimator with Bartlett weights)
    var_d = gamma[0]
    for k in range(1, lag + 1):
        weight = 1.0 - (k / (lag + 1))
        var_d += 2.0 * weight * gamma[k]
        
    # Standard error of the mean
    se_d = np.sqrt(var_d / N)
    
    # DM statistic
    dm_stat = mean_d / se_d
    
    # p-values (using standard normal distribution)
    p_two_sided = 2 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))
    p_one_sided = 1.0 - stats.norm.cdf(dm_stat) # H1: Gibbs is superior (dm_stat > 0)
    
    return dm_stat, p_two_sided, p_one_sided

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
    print("      STATISTICAL COMPARISON: GIBBS (TIMELSFIX) vs STATIONARY MATERN      ")
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
        
        # Diebold-Mariano test (Newey-West lag=4)
        dm_stat, dm_p_two, dm_p_one = dm_test(merged['rmsre_matern'].values, merged['rmsre_gibbs'].values, lag=4)
        
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
        print(f"    Wilcoxon Signed-Rank p-val (naive): {wilcox_p:.2e}")
        print(f"    Diebold-Mariano stat: {dm_stat:.3f}")
        print(f"    Diebold-Mariano p-val (two-sided):  {dm_p_two:.2e}")
        print(f"    Diebold-Mariano p-val (one-sided):  {dm_p_one:.2e} (Gibbs superior)")
        print(f"    95% Block-Bootstrap CI of Delta:    [{ci_low:.5f}, {ci_high:.5f}]")
        
        # Scientific validation
        if dm_p_one < 0.05:
            print("  ==> VERDICT: Gibbs kernel is STATISTICALLY SUPERIOR to stationary Matérn (p < 0.05).")
        else:
            print("  ==> VERDICT: No statistically significant difference in accuracy.")

if __name__ == '__main__':
    compare_layers()
