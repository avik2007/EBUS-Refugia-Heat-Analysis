import os
import pandas as pd
import glob

def summarize():
    aelogs_dir = "AEResults/aelogs"
    pattern = os.path.join(aelogs_dir, "*", "audit_*.csv")
    csv_files = glob.glob(pattern)
    print(f"Found {len(csv_files)} audit CSV files.")
    
    results = []
    for f in sorted(csv_files):
        df = pd.read_csv(f)
        run_name = os.path.basename(os.path.dirname(f))
        
        # Standard columns
        cols = df.columns
        row_dict = {
            'run_name': run_name,
            'n_rows': len(df),
            'med_rmsre': df['rmsre'].median() if 'rmsre' in cols else None,
            'med_std_z': df['std_z'].median() if 'std_z' in cols else None,
            'mean_rmsre': df['rmsre'].mean() if 'rmsre' in cols else None,
            'mean_std_z': df['std_z'].mean() if 'std_z' in cols else None,
        }
        
        # Gibbs specific columns
        for param in ['d_transition_km', 'k_steepness', 'time_ls_days', 'anisotropy_ratio', 'scale_time_bin']:
            if param in cols:
                row_dict[f'med_{param}'] = df[param].median()
                row_dict[f'mean_{param}'] = df[param].mean()
            else:
                row_dict[f'med_{param}'] = None
                row_dict[f'mean_{param}'] = None
                
        results.append(row_dict)
        
    res_df = pd.DataFrame(results)
    
    # Print summary
    print("\nSummary of all runs:")
    print("==========================================================================")
    for idx, row in res_df.iterrows():
        print(f"\nRun: {row['run_name']}")
        print(f"  Rows: {row['n_rows']}")
        print(f"  RMSRE: med={row['med_rmsre']:.4f}, mean={row['mean_rmsre']:.4f}" if row['med_rmsre'] is not None else "  RMSRE: N/A")
        print(f"  Std Z-Score: med={row['med_std_z']:.4f}, mean={row['mean_std_z']:.4f}" if row['med_std_z'] is not None else "  Std Z-Score: N/A")
        if row['med_d_transition_km'] is not None:
            print(f"  d_transition_km: med={row['med_d_transition_km']:.1f}, mean={row['mean_d_transition_km']:.1f}")
        if row['med_k_steepness'] is not None:
            print(f"  k_steepness: med={row['med_k_steepness']:.4f}, mean={row['mean_k_steepness']:.4f}")
        if row['med_time_ls_days'] is not None:
            print(f"  time_ls_days: med={row['med_time_ls_days']:.1f}, mean={row['mean_time_ls_days']:.1f}")
        if row['med_anisotropy_ratio'] is not None:
            print(f"  anisotropy_ratio: med={row['med_anisotropy_ratio']:.2f}, mean={row['mean_anisotropy_ratio']:.2f}")
        if row['med_scale_time_bin'] is not None:
            print(f"  scale_time_bin: med={row['med_scale_time_bin']:.2f}, mean={row['mean_scale_time_bin']:.2f}")

if __name__ == '__main__':
    summarize()
