"""
=============================================================================
VERSION 2.0: Canonical Skin Layer (0-100m) — 3D Matern(nu=0.5), Window=45d
             Permanent canonical config: californiav2 / 10d step / T3+S1 bounds
=============================================================================
This script supersedes 03_ae_inspect_data.py as the canonical Skin Layer
diagnostic. It adopts the C2 configuration determined optimal by the RMSRE
optimization experiment in 05_ae_rmsre_optimization.py, updated by the
completed T1–T3 temporal experiments and S1–S2 spatial-bound experiments:

  - mode='3D'                        : lat, lon, and time as GP features
  - kernel_type='matern0.5'          : Exponential (Ornstein-Uhlenbeck) kernel,
                                        better suited to non-differentiable
                                        ocean processes than RBF
  - window_size_days=45              : wider window captures more Jan floats,
                                        fixing the Cluster 1 sparse-data failure
                                        (RMSRE 6.42% -> 3.43% in that window)
  - time_ls_bounds_days=(15.0, 45.0) : T3 result — lower bound raised from 2d
                                        to 15d to prevent unphysical sub-ocean-
                                        timescale collapse in the GP optimizer
  - step_size_days=10                : matches the 10d Argo resurface / bin
                                        width, eliminating structural aliasing
                                        (beat frequency oscillation in scale_time)
  - spatial_ls_upper_bound=10        : S1 result — raised from 5 to allow the
                                        optimizer to reach deep-layer spatial
                                        coherence without bound saturation
  - region='californiav2'            : tighter CCS domain (lat [30,45],
                                        lon [-130,-115]), replacing 'california'

Output is saved under a distinct run_id suffix (_3dmatern_w45) so the
deprecated 2D-RBF results in the base run_id folder are not overwritten.

Script 03 (2D-RBF, window=30) is preserved as the historical baseline.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from ebus_core.ae_utils import get_ae_config, ensure_ae_dirs
from ebus_core.argoebus_gp_physics import (
    analyze_rolling_correlations,
    plot_physics_history,
    plot_kriging_snapshot
)


def run_diagnostic_inspection(region="california", lat_step=0.5, lon_step=0.5,
                              time_step=30.0, depth_range=(0, 100),
                              run_suffix="",
                              spatial_ls_upper_bound=10,
                              time_ls_bounds_days=(15.0, 45.0),
                              step_size_days=10):
    # --- 1. SETUP & HOUSEKEEPING ---
    # get_ae_config builds the run_id used to locate the S3 parquet. The S3
    # key is unchanged from script 03 — same data, different analysis config.
    #
    # run_suffix: appended to the output_run_id folder name to distinguish
    #   any future experiment variants without overwriting the canonical
    #   _3dmatern_w45 baseline. The canonical run leaves this as "" because
    #   the new defaults already encode the permanent config — no suffix needed.
    # spatial_ls_upper_bound: GP optimizer spatial length-scale upper bound in
    #   StandardScaler units. Permanent canonical value = 10 (S1 experiment result).
    #   The previous default of 5 caused bound saturation in deep layers (500–1000m);
    #   10 allows the optimizer to reach the true spatial coherence scale at depth.
    # time_ls_bounds_days: (lower, upper) physical-day bounds on the time length
    #   scale in 3D mode. Permanent canonical lower bound = 15d (T3 experiment result).
    #   The previous lower bound of 2d allowed the optimizer to collapse to sub-ocean-
    #   timescale solutions (< Argo resurface cycle) that are physically meaningless.
    #   15d is the minimum credible ocean-process timescale for this analysis.
    # step_size_days: stride between successive rolling windows.
    #   Permanent canonical value = 10d (T3/T1 combined result). The previous default
    #   of 15d introduced structural aliasing: a 15d stride on 30d-binned parquet data
    #   produces overlapping window sets that create a beat-frequency oscillation in
    #   scale_time_days. A 10d stride matches the Argo 10d resurface cycle and the
    #   10d bin width used by Script 02, eliminating the aliasing artifact.
    config = get_ae_config(
        region=region,
        lat_step=lat_step,
        lon_step=lon_step,
        time_step=time_step,
        depth_range=depth_range
    )

    # output_run_id is the canonical identifier for THIS run's artifacts.
    # It differs from config['run_id'] so outputs land in a separate folder
    # and do not overwrite the deprecated 2D-RBF results.
    # run_suffix further differentiates experiment variants from the baseline.
    output_run_id = config['run_id'] + "_3dmatern_w45" + run_suffix

    # Ensure AEResults directories exist.
    # AEResults lives one level above ArgoEBUSCloud/, at ArgoEBUSAnalysis/AEResults/.
    ensure_ae_dirs()
    base_path = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(base_path, "..", "AEResults", "aeplots")

    print(f"Loading Skin Layer Dataset: {config['run_id']} ...")
    s3_uri = f"s3://{config['s3_bucket']}/{config['run_id']}.parquet"

    try:
        df = pd.read_parquet(s3_uri)
        print(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Could not find file in S3. Did Script 02 finish? Error: {e}")
        return

    # --- 2. ROLLING PHYSICS ANALYSIS (3D Matern, Window=45d) ---
    # C2 configuration: wider window + Matern(nu=0.5) + time as 3rd GP feature.
    # time_ls_bounds_days upper limit matches the window width so the optimizer
    # is not artificially constrained below the half-window (22.5 days).
    print(f"Running 3D Matern(nu=0.5) physics engine (Window: 45d, Step: {step_size_days}d) ...")
    print(f"  time_ls_bounds_days={time_ls_bounds_days}  spatial_ls_upper_bound={spatial_ls_upper_bound}")
    results_df, cv_details = analyze_rolling_correlations(
        df=df,
        feature_cols=['lat_bin', 'lon_bin'],
        target_col='ohc_per_m',
        time_col='time_bin',
        window_size_days=45,              # C2: wider than baseline 30d
        step_size_days=step_size_days,    # default 15d; pass 10d for Experiment T1
        auto_tune=True,
        mode='3D',                        # add time as third GP input dimension
        kernel_type='matern0.5',          # Exponential / OU kernel
        time_ls_bounds_days=time_ls_bounds_days,        # physical-day bounds on temporal scale
        spatial_ls_upper_bound=spatial_ls_upper_bound,  # upper bound on spatial scale (scaled units)
    )

    # --- 3. SAVE KRIGING SNAPSHOTS TO AEResults/aeplots ---
    # One PNG per window center, collected in a subfolder named by output_run_id.
    plot_dir = os.path.join(base_path, "..", "AEResults", "aeplots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    print(f"Saving kriging snapshots to: {plot_dir}")

    for target_t in results_df['window_center']:
        # plot_kriging_snapshot requires mode='2D' feature_cols (spatial only).
        # The time dimension is used in training but not in the 2D map output.
        fig = plot_kriging_snapshot(
            df_raw=df,
            results_df=results_df,
            target_date=target_t,
            feature_cols=['lat_bin', 'lon_bin'],
            time_col='time_bin',
            grid_res=0.25
        )

        if fig is None:
            print(f"   SKIPPED: Not enough data in window {int(target_t)}")
            continue

        # Each snapshot is written to a named subfolder keyed by output_run_id,
        # not config['run_id'], so it is co-located with the audit CSV below.
        snapshot_name = f"snapshot_{output_run_id}_day{int(target_t)}.png"
        holding_folder_path = os.path.join(plot_dir, f"snapshot_{output_run_id}")
        os.makedirs(holding_folder_path, exist_ok=True)
        save_path = os.path.join(holding_folder_path, snapshot_name)

        fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=False)
        plt.close(fig)
        gc.collect()

    # --- 4. SAVE NUMERICAL DATA (AUDIT CSV, CV PICKLE, PHYSICS PNGs) ---
    # All output artifacts use output_run_id so they land in a separate folder
    # from the deprecated 2D-RBF results.
    data_out_dir = os.path.join(base_path, "..", "AEResults", "aelogs", output_run_id)
    os.makedirs(data_out_dir, exist_ok=True)

    # Rolling audit: per-window RMSRE, Z-score, length scales, anisotropy.
    audit_path = os.path.join(data_out_dir, f"audit_{output_run_id}.csv")
    results_df.to_csv(audit_path, index=False)

    # Cross-validation details: raw error points for bias detection.
    cv_path = os.path.join(data_out_dir, f"cv_details_{output_run_id}.pkl")
    pd.to_pickle(cv_details, cv_path)

    # Physics history PNGs: RMSRE, Z-score, length scales, anisotropy, time persistence.
    plot_physics_history(results_df, cv_details=None, time_unit='days',
                         save_dir=data_out_dir, run_id=output_run_id)

    # --- 5. PRINT SUMMARY ---
    # rmsre is stored as a decimal fraction (e.g. 0.0386 = 3.86%).
    # Multiply by 100 for display and compare against 0.05 for the 5% threshold.
    rmsre_vals = results_df['rmsre'].dropna()
    z_vals = results_df['std_z'].dropna()
    n_pass = (rmsre_vals <= 0.05).sum()
    print(f"\nRESULTS SUMMARY — {output_run_id}")
    print(f"  Windows:      {len(rmsre_vals)}")
    print(f"  Pass (<5%):   {n_pass} / {len(rmsre_vals)} "
          f"({100*n_pass/len(rmsre_vals):.0f}%)")
    print(f"  Median RMSRE: {rmsre_vals.median()*100:.2f}%")
    print(f"  Max RMSRE:    {rmsre_vals.max()*100:.2f}%")
    print(f"  Min RMSRE:    {rmsre_vals.min()*100:.2f}%")
    print(f"  Std Z range:  {z_vals.min():.2f} – {z_vals.max():.2f}")
    print(f"\n  Audit CSV: {audit_path}")
    print(f"  CV Pickle: {cv_path}")
    print(f"PIPELINE COMPLETE.")


if __name__ == "__main__":
    # Canonical run: californiav2 domain, 10d temporal resolution.
    # All defaults have been updated to the permanent canonical config:
    #   - step_size_days=10: matches 10d bin width, eliminates structural aliasing
    #   - time_ls_bounds_days=(15.0, 45.0): T3 floor — prevents sub-ocean-timescale collapse
    #   - spatial_ls_upper_bound=10: S1 expansion — allows deep-layer spatial coherence
    # No run_suffix is needed: the new run_id already encodes this config via
    # the californiav2 region tag and the _3dmatern_w45 output suffix.
    run_diagnostic_inspection(
        region="californiav2",
        lat_step=0.5,
        lon_step=0.5,
        time_step=30.0,
        depth_range=(0, 100),
    )
