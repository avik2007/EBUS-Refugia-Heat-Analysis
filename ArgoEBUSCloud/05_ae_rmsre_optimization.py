"""
05_ae_rmsre_optimization.py

Systematically tests parameter variants of the 3D Matern(nu=0.5) GPR pipeline
to reduce RMSRE in two identified failure clusters:

  Cluster 1 — Early January (window center ~day 5835):
      Only ~50 spatial bins vs. 136-155 typical. The GP is underdetermined.
      Two candidate fixes: skip sparse windows (min_bins=80) or widen the
      window to pull in more floats (window_size_days=45).

  Cluster 2 — Summer eddy season (window centers ~days 6030-6045, July 2015):
      Eddy-dominated dynamics (anisotropy ratio ~0.26-0.36) plus mild
      overconfidence (Std Z ~1.08-1.11). Two candidate fixes: shorten
      the window to avoid bridging eddy lifecycles (window_size_days=20)
      or raise the initial noise floor to reduce overconfidence (noise_val=0.5).

Variants
--------
  C0 (reference) — window=30, min_bins=10, noise=0.1, tb=30  [loaded from disk]
  C1             — window=30, min_bins=80, noise=0.1, tb=30  [skip sparse Jan window]
  C2             — window=45, min_bins=10, noise=0.1, tb=45  [wider window, bounds=45]
  C3             — window=20, min_bins=10, noise=0.1, tb=20  [shorter window]
  C4             — window=30, min_bins=10, noise=0.5, tb=30  [higher noise floor]
  C5             — window=45, min_bins=10, noise=0.1, tb=30  [wider window, bounds=30]

C1–C4 are loaded from disk if their audit CSVs exist; only C5 is re-run by default.
All variants use mode='3D', kernel_type='matern0.5', step_size_days=15.

Each run saves an audit CSV and physics PNGs to AEResults/aelogs/{variant}/.
The script ends with a printed comparison table and two diagnostic plots:
  - Multi-run RMSRE overlay (one line per variant)
  - Float coverage plot for C0 baseline

Run from ArgoEBUSCloud/:
    conda run -n ebus-cloud-env python 05_ae_rmsre_optimization.py
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
    plot_float_coverage,
)


def run_rmsre_optimization(region="california", lat_step=0.5, lon_step=0.5,
                           time_step=30.0, depth_range=(0, 100)):
    """
    Loads the processed OHC parquet from S3, runs four parameter variants of
    the 3D Matern GP pipeline, and produces a comparison table + diagnostic plots.

    Parameters match get_ae_config exactly so the S3 path and run_id are derived
    consistently with scripts 01-04.

    The C0 reference run is loaded from the existing audit CSV written by script
    04 (variant {run_id}_3d_matern05). It is NOT re-run here. If that file is
    missing, the comparison table will omit C0 but all four new runs still proceed.
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
    log_root = os.path.join(ae_root, "aelogs")
    plot_dir = os.path.join(ae_root, "aeplots")

    # Variant IDs. Suffix encodes the parameter being tested so folder names
    # are self-documenting without needing to open the audit CSV.
    variant_c0 = f"{run_id}_3d_matern05"        # reference — written by script 04
    variant_c1 = f"{run_id}_3d_m05_minbins80"   # C1: skip sparse windows
    variant_c2 = f"{run_id}_3d_m05_w45"         # C2: wider window, time bounds=45
    variant_c3 = f"{run_id}_3d_m05_w20"         # C3: shorter window
    variant_c4 = f"{run_id}_3d_m05_noise0p5"    # C4: higher noise floor
    variant_c5 = f"{run_id}_3d_m05_w45_tb30"    # C5: wider window, time bounds=30

    os.makedirs(os.path.join(log_root, variant_c1), exist_ok=True)
    os.makedirs(os.path.join(log_root, variant_c2), exist_ok=True)
    os.makedirs(os.path.join(log_root, variant_c3), exist_ok=True)
    os.makedirs(os.path.join(log_root, variant_c4), exist_ok=True)
    os.makedirs(os.path.join(log_root, variant_c5), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # --- 2. LOAD DATA FROM S3 ---
    # All four variants fit on the same input parquet written by script 02.
    # Fail immediately if the file is missing rather than silently producing empty outputs.
    s3_uri = f"s3://{config['s3_bucket']}/{run_id}.parquet"
    print(f"\nLoading data: {s3_uri}")
    try:
        df = pd.read_parquet(s3_uri)
        print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"  ERROR: Could not load from S3. Did script 02 finish?\n  {e}")
        return

    # --- 3. LOAD C0 REFERENCE (from script 04 output, not re-run) ---
    # The baseline result (30-day window, min_bins=10, noise=0.1, 3D Matern) was
    # already established by script 04. Loading it here avoids duplicating ~1 min
    # of compute and keeps the comparison apples-to-apples (same optimizer seed).
    c0_csv = os.path.join(log_root, variant_c0, f"audit_{variant_c0}.csv")
    results_c0 = None
    if os.path.exists(c0_csv):
        results_c0 = pd.read_csv(c0_csv)
        print(f"\nC0 reference loaded from disk: {c0_csv}")
        print(f"  {len(results_c0)} windows, median RMSRE={results_c0['rmsre'].median():.3%}")
    else:
        print(f"\nWARNING: C0 reference not found at {c0_csv}")
        print(f"  Run 04_ae_testmatern_and_3dwindow.py first, or comparison table will lack C0.")

    # --- 4. LOAD OR RUN C1: min_bins=80 (skip sparse Jan window) ---
    # Loads from disk if the audit CSV already exists, avoiding re-running expensive
    # GPR fits when only a new variant (e.g. C5) needs to be tested.
    c1_csv = os.path.join(log_root, variant_c1, f"audit_{variant_c1}.csv")
    if os.path.exists(c1_csv):
        results_c1 = pd.read_csv(c1_csv)
        print(f"\nC1 loaded from disk: {c1_csv}")
    else:
        print(f"\n{'='*60}")
        print(f"RUN C1: min_bins=80 — skip underdetermined sparse windows")
        print(f"{'='*60}")
        results_c1, _ = analyze_rolling_correlations(
            df=df,
            feature_cols=['lat_bin', 'lon_bin'],
            target_col='ohc_per_m',
            time_col='time_bin',
            window_size_days=30,
            step_size_days=15,
            min_bins=80,
            auto_tune=True,
            mode='3D',
            kernel_type='matern0.5',
            time_ls_bounds_days=(2.0, 30.0),
        )
        _save_run(results_c1, variant_c1, log_root, plot_dir)
        gc.collect()

    # --- 5. LOAD OR RUN C2: window_size_days=45, time bounds=45 ---
    c2_csv = os.path.join(log_root, variant_c2, f"audit_{variant_c2}.csv")
    if os.path.exists(c2_csv):
        results_c2 = pd.read_csv(c2_csv)
        print(f"\nC2 loaded from disk: {c2_csv}")
    else:
        print(f"\n{'='*60}")
        print(f"RUN C2: window_size_days=45 — wider window to boost Jan obs count")
        print(f"{'='*60}")
        results_c2, _ = analyze_rolling_correlations(
            df=df,
            feature_cols=['lat_bin', 'lon_bin'],
            target_col='ohc_per_m',
            time_col='time_bin',
            window_size_days=45,
            step_size_days=15,
            min_bins=10,
            auto_tune=True,
            mode='3D',
            kernel_type='matern0.5',
            time_ls_bounds_days=(2.0, 45.0),
        )
        _save_run(results_c2, variant_c2, log_root, plot_dir)
        gc.collect()

    # --- 6. LOAD OR RUN C3: window_size_days=20 ---
    c3_csv = os.path.join(log_root, variant_c3, f"audit_{variant_c3}.csv")
    if os.path.exists(c3_csv):
        results_c3 = pd.read_csv(c3_csv)
        print(f"\nC3 loaded from disk: {c3_csv}")
    else:
        print(f"\n{'='*60}")
        print(f"RUN C3: window_size_days=20 — shorter window to avoid eddy bridging")
        print(f"{'='*60}")
        results_c3, _ = analyze_rolling_correlations(
            df=df,
            feature_cols=['lat_bin', 'lon_bin'],
            target_col='ohc_per_m',
            time_col='time_bin',
            window_size_days=20,
            step_size_days=15,
            min_bins=10,
            auto_tune=True,
            mode='3D',
            kernel_type='matern0.5',
            time_ls_bounds_days=(2.0, 20.0),
        )
        _save_run(results_c3, variant_c3, log_root, plot_dir)
        gc.collect()

    # --- 7. LOAD OR RUN C4: noise_val=0.5 ---
    c4_csv = os.path.join(log_root, variant_c4, f"audit_{variant_c4}.csv")
    if os.path.exists(c4_csv):
        results_c4 = pd.read_csv(c4_csv)
        print(f"\nC4 loaded from disk: {c4_csv}")
    else:
        print(f"\n{'='*60}")
        print(f"RUN C4: noise_val=0.5 — higher initial noise floor for Cluster 2")
        print(f"{'='*60}")
        results_c4, _ = analyze_rolling_correlations(
            df=df,
            feature_cols=['lat_bin', 'lon_bin'],
            target_col='ohc_per_m',
            time_col='time_bin',
            window_size_days=30,
            step_size_days=15,
            min_bins=10,
            noise_val=0.5,
            auto_tune=True,
            mode='3D',
            kernel_type='matern0.5',
            time_ls_bounds_days=(2.0, 30.0),
        )
        _save_run(results_c4, variant_c4, log_root, plot_dir)
        gc.collect()

    # --- 8. RUN C5: window=45, time_ls_bounds_days=(2.0, 30.0) ---
    # Motivation: C2 (window=45, bounds=(2.0, 45.0)) showed scale_time_days repeatedly
    # pinning to the upper bound (45 days = the full window width). When the optimizer
    # hits the ceiling it is effectively saying "the entire window is one correlated
    # blob," which prevents the GP from resolving any temporal structure within the
    # window. The pre-C2 baseline (window=30, bounds=(2.0, 30.0)) was stable because
    # the upper bound was 30 days, matching the empirical eddy lifecycle.
    # C5 tests whether keeping the window at 45 (data density benefit) while
    # constraining the time length scale to ≤30 days (physical realism) gives
    # stable scale_time_days without sacrificing RMSRE relative to C2.
    print(f"\n{'='*60}")
    print(f"RUN C5: window=45, time_ls_bounds=(2,30) — tighter time bounds")
    print(f"{'='*60}")
    results_c5, _ = analyze_rolling_correlations(
        df=df,
        feature_cols=['lat_bin', 'lon_bin'],
        target_col='ohc_per_m',
        time_col='time_bin',
        window_size_days=45,              # same as C2: wider window for Jan coverage
        step_size_days=15,
        min_bins=10,
        auto_tune=True,
        mode='3D',
        kernel_type='matern0.5',
        time_ls_bounds_days=(2.0, 30.0),  # upper bound back to 30d: matches eddy lifecycle
    )
    _save_run(results_c5, variant_c5, log_root, plot_dir)
    gc.collect()

    # --- 9. COMPARISON SUMMARY TABLE ---
    # Summarize all variants side-by-side. Key metrics:
    #   n_windows   — how many windows were included (C1 may be lower due to min_bins skip)
    #   n_pass      — windows meeting the 5% RMSRE target
    #   median RMSRE, max RMSRE
    #   Cluster 1 RMSRE — window center closest to day 5835 (early Jan)
    #   Cluster 2 RMSRE — worst window in the day 6025-6050 range (July eddy season)
    #   median Std Z
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY — {run_id}")
    print(f"{'='*70}")

    TARGET_RMSRE = 0.05
    # Cluster 1 window: center closest to day 5835 (the early-Jan sparse window).
    C1_CENTER = 5835
    # Cluster 2 window: worst RMSRE in the summer band (days 6025-6050).
    C2_LO, C2_HI = 6025, 6050

    all_variants = [
        ("C0 (ref, w30 mb10 n0.1)", results_c0),
        ("C1 (w30 mb80 n0.1)",      results_c1),
        ("C2 (w45 tb45 n0.1)",      results_c2),
        ("C3 (w20 mb10 n0.1)",      results_c3),
        ("C4 (w30 mb10 n0.5)",      results_c4),
        ("C5 (w45 tb30 n0.1)",      results_c5),
    ]

    header = (f"  {'Variant':<28} {'N':>4} {'Pass':>5} "
              f"{'MedRMSRE':>9} {'MaxRMSRE':>9} "
              f"{'Clust1':>8} {'Clust2':>8} {'MedZ':>6}")
    print(header)
    print(f"  {'-'*75}")

    for label, rdf in all_variants:
        if rdf is None:
            print(f"  {label:<28}  (not available)")
            continue

        n_win  = len(rdf)
        n_pass = (rdf['rmsre'] <= TARGET_RMSRE).sum()
        med_r  = rdf['rmsre'].median()
        max_r  = rdf['rmsre'].max()
        med_z  = rdf['std_z'].median()

        # Cluster 1: RMSRE of the window whose center is closest to day 5835.
        # If min_bins skipped that window (C1), this returns the nearest surviving window.
        idx1   = (rdf['window_center'] - C1_CENTER).abs().idxmin()
        c1_r   = rdf.loc[idx1, 'rmsre']
        # Annotate if the window was skipped (center differs from C1_CENTER by more than
        # one step interval, meaning the sparse window was dropped).
        c1_flag = "*" if abs(rdf.loc[idx1, 'window_center'] - C1_CENTER) > 20 else " "

        # Cluster 2: worst (max) RMSRE in the summer eddy band.
        summer = rdf[(rdf['window_center'] >= C2_LO) & (rdf['window_center'] <= C2_HI)]
        c2_r   = summer['rmsre'].max() if len(summer) > 0 else float('nan')

        print(f"  {label:<28} {n_win:>4} {n_pass:>5} "
              f"{med_r:>9.3%} {max_r:>9.3%} "
              f"{c1_r:>7.3%}{c1_flag} {c2_r:>8.3%} {med_z:>6.2f}")

    print(f"\n  * = Cluster 1 window (~day {C1_CENTER}) was skipped (min_bins raised);")
    print(f"      value shown is nearest surviving window.")
    print(f"\n  Decision guide:")
    print(f"    Cluster 1: prefer C2 if Jan RMSRE improves without dropping the window.")
    print(f"               Fall back to C1 (skip) if C2 worsens Cluster 2.")
    print(f"    Cluster 2: prefer C4 if summer RMSRE drops (lowest-risk change).")
    print(f"               Accept C3 if C4 fails. Mark irreducible if neither helps.")

    # --- 9. MULTI-RUN RMSRE OVERLAY PLOT ---
    # Plots all five variants on the same axes so per-window differences are
    # visible at a glance. The two cluster bands are shaded to orient the viewer.
    print(f"\nGenerating RMSRE overlay plot...")
    fig, ax = plt.subplots(figsize=(15, 5))

    # Light shading for the two problem clusters so they stand out immediately.
    # Cluster 1: narrow band around day 5835 (half a step-width either side).
    ax.axvspan(C1_CENTER - 20, C1_CENTER + 20, color='salmon', alpha=0.15,
               label='Cluster 1 (sparse Jan)')
    # Cluster 2: the full summer eddy band identified from the C0 audit.
    ax.axvspan(C2_LO, C2_HI, color='lightyellow', alpha=0.5,
               label='Cluster 2 (summer eddies)')

    # One line per variant. Use distinct markers so the plot is readable in greyscale.
    styles = [
        ('C0 ref',       'black',       'o',  '-',       2.0),
        ('C1 mb80',      'tab:blue',    's',  '--',      1.5),
        ('C2 w45 tb45',  'tab:green',   'D',  '-.',      1.5),
        ('C3 w20',       'tab:red',     '^',  ':',       1.5),
        ('C4 n0.5',      'tab:purple',  'v',  (0,(5,2)), 1.5),
        ('C5 w45 tb30',  'tab:orange',  'P',  '-',       1.5),
    ]
    for (label_short, rdf), (style_label, color, marker, ls, lw) in zip(all_variants, styles):
        if rdf is None:
            continue
        ax.plot(rdf['window_center'], rdf['rmsre'] * 100,
                color=color, marker=marker, linestyle=ls, linewidth=lw,
                markersize=5, label=style_label, alpha=0.85)

    ax.axhline(5.0, color='grey', linestyle=':', linewidth=1.0, label='5% RMSRE target')
    ax.set_ylabel("RMSRE (%)")
    ax.set_xlabel("Window Center (days since 1999-01-01)")
    ax.set_title(f"RMSRE per Rolling Window — All Variants\n{run_id}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    overlay_path = os.path.join(plot_dir, f"rmsre_overlay_{run_id}_c0_c5.png")
    fig.savefig(overlay_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {overlay_path}")
    plt.close(fig)
    gc.collect()

    # --- 10. FLOAT COVERAGE PLOT (C0 BASELINE) ---
    # Shows n_bins vs. time with RMSRE overlaid so the sparsity/failure correlation
    # is immediately visible. The min_bins reference line is drawn at 10 (C0 default)
    # because this plot uses the C0 baseline (no windows were skipped there).
    if results_c0 is not None:
        print(f"\nGenerating float coverage plot (C0 baseline)...")
        cov_dir = os.path.join(log_root, variant_c0)
        plot_float_coverage(results_c0, min_bins_threshold=10,
                            save_dir=cov_dir, run_id=variant_c0)
        plt.close('all')
        gc.collect()
    else:
        print(f"\nFloat coverage plot skipped (C0 not available).")

    print(f"\nPIPELINE COMPLETE — 05_ae_rmsre_optimization")
    print(f"  Audit CSVs and physics PNGs in: {log_root}")
    print(f"  Overlay plot: {overlay_path}")


def _save_run(results_df, variant, log_root, plot_dir):
    """
    Saves the audit CSV and physics PNGs for a single optimization variant.

    Called after each analyze_rolling_correlations call in the main function.
    Factored out to avoid repeating the same 4-line save/plot block five times.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from analyze_rolling_correlations for this variant.
    variant : str
        Self-documenting variant ID (e.g. 'california_..._3d_m05_w45').
        Used as both the subfolder name and the run_id argument to plot_physics_history.
    log_root : str
        Absolute path to AEResults/aelogs/.
    plot_dir : str
        Absolute path to AEResults/aeplots/ (unused here but passed for future use).
    """
    save_dir  = os.path.join(log_root, variant)
    audit_csv = os.path.join(save_dir, f"audit_{variant}.csv")

    results_df.to_csv(audit_csv, index=False)
    print(f"\n  Audit saved: {audit_csv}")
    print(f"  {len(results_df)} windows | "
          f"median RMSRE={results_df['rmsre'].median():.3%} | "
          f"pass rate={(results_df['rmsre'] <= 0.05).mean():.0%}")

    # Physics history plots (Plots 1-5 depending on mode).
    # These use the same function as scripts 03 and 04 so visual style is consistent.
    plot_physics_history(results_df, cv_details=None, time_unit='days',
                         save_dir=save_dir, run_id=variant)
    plt.close('all')


if __name__ == "__main__":
    run_rmsre_optimization(
        region="california",
        lat_step=0.5,
        lon_step=0.5,
        time_step=30.0,
        depth_range=(0, 100),
    )
