"""
04_ae_testmatern_and_3dwindow.py

Compares two GP configurations on the same Argo OHC data used in 03_ae_inspect_data.py:

  Run A — 2D RBF (baseline):
      mode='2D', kernel_type='rbf'
      Identical to the analyze_rolling_correlations call in script 03. Used as
      the direct RMSRE baseline so any improvement from the 3D Exponential model
      is attributable to the kernel/dimension change, not data differences.

  Run B — 3D Matern(nu=0.5):
      mode='3D', kernel_type='matern0.5'
      Adds the time coordinate as a third GP dimension, normalized to the rolling
      window (window center = 0, edges = ±1). Uses the Exponential covariance,
      which allows sharper spatial fronts than RBF. See AE_plan_3d_gpr_matern.md
      for the full mathematical design.

Both runs use the same rolling window parameters (window=30d, step=15d) and load
from the same S3 parquet as script 03. Results are written to separate subfolders
so that 04b_ae_plot_matern_physics.py can regenerate or compare physics PNGs
without re-running the expensive analysis.

Output folders (under AEResults/aelogs/):
    {run_id}_2d_rbf/       — baseline audit CSV, CV pickle, physics PNGs
    {run_id}_3d_matern05/  — 3D Exponential audit CSV, CV pickle, physics PNGs

Note on kriging snapshots:
    Snapshot maps (plot_kriging_snapshot) are only generated for the 2D baseline.
    The 3D GP fits a 3-dimensional kernel; predicting onto a 2D spatial grid
    requires fixing the time coordinate, which plot_kriging_snapshot does not yet
    support. Snapshots for the 3D run can be added once that function is extended.

Run from ArgoEBUSCloud/:
    conda run -n ebus-cloud-env python 04_ae_testmatern_and_3dwindow.py
"""

import os
import gc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless: no display needed
import matplotlib.pyplot as plt

from ebus_core.ae_utils import get_ae_config, ensure_ae_dirs
from ebus_core.argoebus_gp_physics import (
    analyze_rolling_correlations,
    plot_physics_history,
    plot_kriging_snapshot,
)


def run_matern_comparison(region="california", lat_step=0.5, lon_step=0.5,
                          time_step=30.0, depth_range=(0, 100)):
    """
    Loads the processed OHC parquet from S3, runs both GP variants, and saves
    audit CSVs + physics PNGs to separate named folders for comparison.

    Parameters match get_ae_config exactly so the S3 path and run_id are derived
    consistently with scripts 01–03.
    """

    # --- 1. CONFIG & PATHS ---
    config = get_ae_config(
        region=region,
        lat_step=lat_step,
        lon_step=lon_step,
        time_step=time_step,
        depth_range=depth_range,
    )
    ensure_ae_dirs()

    run_id   = config['run_id']
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ae_root  = os.path.join(base_dir, "..", "AEResults")

    # Variant IDs and output folders.
    # The suffix encodes the mode and kernel so folder names are self-documenting.
    variant_2d  = f"{run_id}_2d_rbf"
    variant_3d  = f"{run_id}_3d_matern05"
    dir_2d      = os.path.join(ae_root, "aelogs", variant_2d)
    dir_3d      = os.path.join(ae_root, "aelogs", variant_3d)
    plot_dir    = os.path.join(ae_root, "aeplots")

    os.makedirs(dir_2d,   exist_ok=True)
    os.makedirs(dir_3d,   exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # --- 2. LOAD DATA ---
    s3_uri = f"s3://{config['s3_bucket']}/{run_id}.parquet"
    print(f"\nLoading data: {s3_uri}")
    try:
        df = pd.read_parquet(s3_uri)
        print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"  Could not load from S3. Did script 02 finish? Error: {e}")
        return

    # --- 3A. RUN A: 2D RBF BASELINE ---
    # Identical parameters to script 03 so results are directly comparable.
    print(f"\n{'='*60}")
    print(f"RUN A: 2D RBF (baseline, matches 03_ae_inspect_data.py)")
    print(f"{'='*60}")
    results_2d, cv_2d = analyze_rolling_correlations(
        df=df,
        feature_cols=['lat_bin', 'lon_bin'],
        target_col='ohc_per_m',
        time_col='time_bin',
        window_size_days=30,
        step_size_days=15,
        auto_tune=True,
        mode='2D',
        kernel_type='rbf',
    )

    # Save audit CSV and CV pickle for Run A.
    audit_path_2d = os.path.join(dir_2d, f"audit_{variant_2d}.csv")
    cv_path_2d    = os.path.join(dir_2d, f"cv_details_{variant_2d}.pkl")
    results_2d.to_csv(audit_path_2d, index=False)
    pd.to_pickle(cv_2d, cv_path_2d)
    print(f"\n  Audit saved: {audit_path_2d}")

    # Physics history PNGs for Run A (Plots 1-4; no Plot 5 because no time scale).
    plot_physics_history(results_2d, cv_details=None, time_unit='days',
                         save_dir=dir_2d, run_id=variant_2d)

    # Kriging snapshots for Run A only (2D GP maps onto 2D grid cleanly).
    snap_folder = os.path.join(plot_dir, f"snapshot_{variant_2d}")
    os.makedirs(snap_folder, exist_ok=True)
    print(f"\n  Generating kriging snapshots -> {snap_folder}")
    for target_t in results_2d['window_center']:
        fig = plot_kriging_snapshot(
            df_raw=df,
            results_df=results_2d,
            target_date=target_t,
            feature_cols=['lat_bin', 'lon_bin'],
            time_col='time_bin',
            grid_res=0.25,
        )
        if fig is None:
            print(f"    Skipped: not enough data at window {int(target_t)}")
            continue
        snap_name = f"snapshot_{variant_2d}_day{int(target_t)}.png"
        fig.savefig(os.path.join(snap_folder, snap_name), dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    # --- 3B. RUN B: 3D MATERN(nu=0.5) ---
    print(f"\n{'='*60}")
    print(f"RUN B: 3D Matern(nu=0.5) — Exponential kernel with time dimension")
    print(f"{'='*60}")
    results_3d, cv_3d = analyze_rolling_correlations(
        df=df,
        feature_cols=['lat_bin', 'lon_bin'],
        target_col='ohc_per_m',
        time_col='time_bin',
        window_size_days=30,
        step_size_days=15,
        auto_tune=True,
        mode='3D',
        kernel_type='matern0.5',
        time_ls_bounds_days=(2.0, 30.0),
        # Note: with a 30-day window (half_window=15 days), the normalized bounds are
        # [2/15, 30/15] = [0.13, 2.0]. The upper bound of 2.0 allows the optimizer
        # to find correlations up to the full window width, while the lower bound
        # prevents the model from treating adjacent days as uncorrelated.
    )

    # Save audit CSV and CV pickle for Run B.
    audit_path_3d = os.path.join(dir_3d, f"audit_{variant_3d}.csv")
    cv_path_3d    = os.path.join(dir_3d, f"cv_details_{variant_3d}.pkl")
    results_3d.to_csv(audit_path_3d, index=False)
    pd.to_pickle(cv_3d, cv_path_3d)
    print(f"\n  Audit saved: {audit_path_3d}")

    # Physics history PNGs for Run B (includes Plot 5: temporal persistence).
    plot_physics_history(results_3d, cv_details=None, time_unit='days',
                         save_dir=dir_3d, run_id=variant_3d)

    # --- 4. COMPARISON SUMMARY ---
    # Print side-by-side median RMSRE and Z-score so you can judge at a glance
    # whether the 3D Exponential model improved accuracy and/or calibration.
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY ({run_id})")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'2D RBF':>12} {'3D Matern0.5':>14}")
    print(f"{'-'*60}")

    for col, label, fmt in [
        ('rmsre',           'Median RMSRE',          '.3%'),
        ('rmsre',           'Mean RMSRE',             '.3%'),
        ('std_z',           'Median Std Z-Score',     '.3f'),
        ('anisotropy_ratio','Median Anisotropy Ratio','.3f'),
    ]:
        # Use median for the first RMSRE row, mean for the second.
        agg = 'median' if 'Median' in label else 'mean'
        v2 = getattr(results_2d[col].dropna(), agg)()
        v3 = getattr(results_3d[col].dropna(), agg)()
        # Pre-format values as strings to allow independent right-justification.
        s2 = format(v2, fmt)
        s3 = format(v3, fmt)
        print(f"  {label:<28} {s2:>12} {s3:>14}")

    if 'scale_time_days' in results_3d.columns:
        t_med = results_3d['scale_time_days'].dropna().median()
        print(f"\n  3D temporal persistence (median): {t_med:.1f} days")
        print(f"  (Interpretation: the GP weighted observations within ~{t_med:.0f} days"
              f" of the window center most heavily.)")

    target = 0.05
    won_2d = (results_2d['rmsre'].dropna() <= target).mean() * 100
    won_3d = (results_3d['rmsre'].dropna() <= target).mean() * 100
    print(f"\n  Windows meeting RMSRE < 5% target:")
    print(f"    2D RBF:       {won_2d:.0f}%")
    print(f"    3D Matern0.5: {won_3d:.0f}%")

    print(f"\nPIPELINE COMPLETE")
    print(f"  2D baseline : {dir_2d}")
    print(f"  3D matern   : {dir_3d}")
    print(f"  To regenerate physics PNGs without re-running:"
          f" python 04b_ae_plot_matern_physics.py")


if __name__ == "__main__":
    run_matern_comparison(
        region="california",
        lat_step=0.5,
        lon_step=0.5,
        time_step=30.0,
        depth_range=(0, 100),
    )
