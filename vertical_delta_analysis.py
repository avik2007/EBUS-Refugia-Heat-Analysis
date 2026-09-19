"""
=============================================================================
vertical_delta_analysis.py — ArgoEBUSAnalysis
=============================================================================
Vertical Sandwich Delta Analysis: Testing the "Stealth Warming" Hypothesis.

Physical Motivation & Scientific Objective:
-------------------------------------------
The "Ocean Refugia / Stealth Warming" hypothesis posits that subsurface heat
transported poleward by the California Undercurrent (CUC) in the Source Layer
(150–400m) warms faster than the deep ocean Background Layer (500–1000m),
accumulating thermal energy that has not yet surfaced into the atmospheric
Skin Layer (0–100m).

This script performs the cross-layer vertical delta audit:
1. Multi-Layer Dynamical Fingerprinting:
   - Evaluates the vertical profile of the learned anisotropy ratio
     (Lat_Scale / Lon_Scale) to verify poleward meridional channeling
     in the Source layer (CUC corridor) vs. deep baseline dynamics.
   - Evaluates the coastal transition midpoint d_0 (dist_to_coast_km)
     to map the width of coastal upwelling / undercurrent regimes.
2. Vertical Delta & Stability Audit:
   - Analyzes cross-layer uncertainty calibration (Std Z-scores) and
     reconstruction errors (RMSRE) to ensure statistical validity across
     the vertical sandwich.
3. Outputs publication-quality synthesis diagnostics to:
   - AEResults/aeplots/vertical_delta/
   - AEResults/aelogs/vertical_delta_californiav3_2015.csv
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Epoch used by the ArgoEBUS pipeline (days since 1999-01-01)
TIME_EPOCH = date(1999, 1, 1)


def days_to_datetime(days_series):
    """
    Converts pipeline numeric days (days since 1999-01-01) to pandas DatetimeIndex.
    
    Parameters:
    -----------
    days_series : pd.Series or np.ndarray
        Floating-point days since TIME_EPOCH (1999-01-01).

    Returns:
    --------
    pd.DatetimeIndex
        Calendar dates corresponding to each day offset.
    """
    epoch_ts = pd.Timestamp("1999-01-01")
    return epoch_ts + pd.to_timedelta(days_series, unit='D')


def load_layer_audits(base_results_dir="AEResults/aelogs", region="californiav3", year=2015):
    """
    Loads and aligns the canonical Gibbs non-stationary GPR audit logs
    across all three vertical layers for a given region and year.

    Parameters:
    -----------
    base_results_dir : str
        Path to the directory containing aelogs folders.
    region : str
        EBUS region key (default: 'californiav3').
    year : int
        Calendar year of the run (default: 2015).

    Returns:
    --------
    tuple (dict[str, pd.DataFrame], dict)
        Dictionary mapping layer names ('skin', 'source', 'background')
        to their respective audit dataframes, and layer configuration dict.
    """
    layer_configs = {
        'skin': {
            'depth_range': (0, 100),
            'title': 'Skin Layer (0–100m)',
            'folder': f"{region}_{year}0101_{year}1231_res0_5x0_5_t10_0_d0_100_3dgibbs_w45_timelsfix",
            'thickness_m': 100.0,
            'color': '#1f77b4',
            'd_transition_bounds_km': (50.0, 700.0)
        },
        'source': {
            'depth_range': (150, 400),
            'title': 'Source Layer (150–400m)',
            'folder': f"{region}_{year}0101_{year}1231_res0_5x0_5_t10_0_d150_400_3dgibbs_w45_timelsfix",
            'thickness_m': 250.0,
            'color': '#ff7f0e',
            'd_transition_bounds_km': (50.0, 700.0)
        },
        'background': {
            'depth_range': (500, 1000),
            'title': 'Background Layer (500–1000m)',
            # d_transition_bounds_km widened 700->1500km 2026-09-16: prior run's
            # median d0 was pegged at the 700km bound (optimizer artifact).
            'folder': f"{region}_{year}0101_{year}1231_res0_5x0_5_t10_0_d500_1000_3dgibbs_w45_timelsfix_d1500",
            'thickness_m': 500.0,
            'color': '#2ca02c',
            'd_transition_bounds_km': (50.0, 1500.0)
        }
    }

    layer_data = {}
    for key, cfg in layer_configs.items():
        folder_path = os.path.join(base_results_dir, cfg['folder'])
        csv_path = os.path.join(folder_path, f"audit_{cfg['folder']}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Audit log missing for {cfg['title']}: {csv_path}")
        
        df = pd.read_csv(csv_path)
        df['date'] = days_to_datetime(df['window_center'])
        df['layer_key'] = key
        df['layer_title'] = cfg['title']
        df['thickness_m'] = cfg['thickness_m']
        df['color'] = cfg['color']
        layer_data[key] = df

    return layer_data, layer_configs


def compute_vertical_delta_metrics(layer_data):
    """
    Computes cross-layer differentials and vertical statistics between
    the Source (150–400m) and Background (500–1000m) layers.

    Physical Meaning:
    -----------------
    - Delta Anisotropy = Anisotropy_Source - Anisotropy_Background.
      Positive values indicate enhanced alongshore meridional transport in the
      California Undercurrent relative to deep ocean baseline.
    - Delta Transition Distance = d_0_Source vs d_0_Background.
      Demonstrates confinement of the dynamic regime within the coastal boundary.

    Parameters:
    -----------
    layer_data : dict[str, pd.DataFrame]
        Loaded audit dataframes for skin, source, and background layers.

    Returns:
    --------
    pd.DataFrame
        Merged dataframe containing aligned time series and delta metrics.
    """
    skin_df = layer_data['skin'][['window_center', 'date', 'rmsre', 'std_z', 'anisotropy_ratio', 'd_transition_km', 'n_floats', 'n_bins']].copy()
    source_df = layer_data['source'][['window_center', 'date', 'rmsre', 'std_z', 'anisotropy_ratio', 'd_transition_km', 'n_floats', 'n_bins']].copy()
    bg_df = layer_data['background'][['window_center', 'date', 'rmsre', 'std_z', 'anisotropy_ratio', 'd_transition_km', 'n_floats', 'n_bins']].copy()

    merged = pd.merge(skin_df, source_df, on=['window_center', 'date'], suffixes=('_skin', '_source'))
    merged = pd.merge(merged, bg_df, on=['window_center', 'date'])
    merged = merged.rename(columns={
        'rmsre': 'rmsre_bg',
        'std_z': 'std_z_bg',
        'anisotropy_ratio': 'anisotropy_ratio_bg',
        'd_transition_km': 'd_transition_km_bg',
        'n_floats': 'n_floats_bg',
        'n_bins': 'n_bins_bg'
    })

    # Cross-layer differences
    merged['delta_anisotropy_source_bg'] = merged['anisotropy_ratio_source'] - merged['anisotropy_ratio_bg']
    merged['delta_anisotropy_source_skin'] = merged['anisotropy_ratio_source'] - merged['anisotropy_ratio_skin']
    merged['ratio_anisotropy_source_bg'] = merged['anisotropy_ratio_source'] / np.maximum(merged['anisotropy_ratio_bg'], 1e-4)

    return merged


def generate_vertical_delta_plots(layer_data, merged_df, layer_configs, output_dir="AEResults/aeplots/vertical_delta"):
    """
    Generates publication-quality diagnostic figures for the Vertical Delta analysis.

    Figures generated:
    1. vertical_delta_anisotropy_fingerprint.png: Meridional flow signature across layers.
    2. vertical_delta_coastal_transition.png: Coastal regime boundary d_0 evolution.
    3. vertical_delta_multipanel_synthesis.png: Comprehensive 4-panel scientific synthesis.

    Parameters:
    -----------
    layer_data : dict[str, pd.DataFrame]
        Audit data per layer.
    merged_df : pd.DataFrame
        Merged time-aligned dataframe with delta calculations.
    layer_configs : dict
        Layer metadata (titles, colors, depths).
    output_dir : str
        Destination directory for output PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Anisotropy Ratio Fingerprint (Meridional CUC Signature)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    for key in ['skin', 'source', 'background']:
        df = layer_data[key]
        cfg = layer_configs[key]
        ax.plot(df['date'], df['anisotropy_ratio'], marker='o', markersize=4, 
                label=cfg['title'], color=cfg['color'], linewidth=1.8)
        
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.7, label='Isotropic (1.0)')
    ax.set_ylabel("Anisotropy Ratio (Lat_Scale / Lon_Scale)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Date (2015)", fontsize=11, fontweight='bold')
    ax.set_title("Vertical Sandwich Dynamical Fingerprint: Anisotropy Across Depth Layers\n(californiav3 3D Gibbs Non-Stationary GPR)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "vertical_delta_anisotropy_fingerprint.png")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------------------
    # FIGURE 2: Coastal Transition Midpoint (d_0) Across Layers
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    for key in ['skin', 'source', 'background']:
        df = layer_data[key]
        cfg = layer_configs[key]
        ax.plot(df['date'], df['d_transition_km'], marker='s', markersize=4, 
                label=cfg['title'], color=cfg['color'], linewidth=1.8)
        
    ax.set_ylabel("Regime Transition Midpoint $d_0$ (km from coast)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Date (2015)", fontsize=11, fontweight='bold')
    ax.set_title("Coastal vs. Offshore Regime Boundary ($d_0$) Across Scientific Layers\n(Sigmoid Lengthscale Midpoint)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "vertical_delta_coastal_transition.png")
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------------------
    # FIGURE 3: 4-Panel Comprehensive Scientific Synthesis Figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, sharex=True)
    
    # Subplot A: Anisotropy Ratio
    ax_a = axes[0, 0]
    for key in ['skin', 'source', 'background']:
        df = layer_data[key]
        cfg = layer_configs[key]
        ax_a.plot(df['date'], df['anisotropy_ratio'], marker='o', markersize=3.5, 
                  label=cfg['title'], color=cfg['color'], linewidth=1.6)
    ax_a.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax_a.set_ylabel("Anisotropy Ratio ($\ell_{lat} / \ell_{lon}$)", fontsize=10, fontweight='bold')
    ax_a.set_title("(A) Spatial Anisotropy (Undercurrent vs Baseline)", fontsize=11, fontweight='bold')
    ax_a.grid(True, linestyle=':', alpha=0.5)
    ax_a.legend(loc='upper right', fontsize=9)

    # Subplot B: Coastal Transition Midpoint d_0
    ax_b = axes[0, 1]
    for key in ['skin', 'source', 'background']:
        df = layer_data[key]
        cfg = layer_configs[key]
        ax_b.plot(df['date'], df['d_transition_km'], marker='s', markersize=3.5, 
                  label=cfg['title'], color=cfg['color'], linewidth=1.6)
    ax_b.set_ylabel("Transition Midpoint $d_0$ (km)", fontsize=10, fontweight='bold')
    ax_b.set_title("(B) Coastal Regime Width ($d_0$)", fontsize=11, fontweight='bold')
    ax_b.grid(True, linestyle=':', alpha=0.5)
    ax_b.legend(loc='upper right', fontsize=9)

    # Subplot C: RMSRE Reconstruction Error
    ax_c = axes[1, 0]
    for key in ['skin', 'source', 'background']:
        df = layer_data[key]
        cfg = layer_configs[key]
        ax_c.plot(df['date'], df['rmsre'] * 100.0, marker='^', markersize=3.5, 
                  label=cfg['title'], color=cfg['color'], linewidth=1.6)
    ax_c.axhline(5.0, color='red', linestyle=':', linewidth=1.2, label='Target RMSRE (5%)')
    ax_c.set_ylabel("RMSRE Error (%)", fontsize=10, fontweight='bold')
    ax_c.set_xlabel("Date (2015)", fontsize=10, fontweight='bold')
    ax_c.set_title("(C) GPR Generalization Error (Target < 5%)", fontsize=11, fontweight='bold')
    ax_c.grid(True, linestyle=':', alpha=0.5)
    ax_c.legend(loc='upper right', fontsize=9)

    # Subplot D: Std Z-Score Calibration
    ax_d = axes[1, 1]
    for key in ['skin', 'source', 'background']:
        df = layer_data[key]
        cfg = layer_configs[key]
        ax_d.plot(df['date'], df['std_z'], marker='d', markersize=3.5, 
                  label=cfg['title'], color=cfg['color'], linewidth=1.6)
    ax_d.axhline(1.0, color='black', linestyle='--', linewidth=1.2, label='Ideal $\sigma_z = 1.0$')
    ax_d.axhspan(0.9, 1.1, color='gray', alpha=0.15, label='Target Bounds [0.9, 1.1]')
    ax_d.set_ylabel("Std Z-Score ($\sigma_z$)", fontsize=10, fontweight='bold')
    ax_d.set_xlabel("Date (2015)", fontsize=10, fontweight='bold')
    ax_d.set_title("(D) Uncertainty Calibration (Gibbs Stabilization)", fontsize=11, fontweight='bold')
    ax_d.grid(True, linestyle=':', alpha=0.5)
    ax_d.legend(loc='upper right', fontsize=9)

    plt.suptitle("Vertical Sandwich Synthesis: 2015 California Current System (californiav3)", 
                 fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "vertical_delta_multipanel_synthesis.png")
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"Saved: {fig3_path}")


def analyze_d0_distribution(layer_data, layer_configs, output_dir="AEResults/aeplots/vertical_delta",
                             bound_tol_km=10.0, cv_flag_threshold=0.5):
    """
    Checks whether the fitted coastal-transition midpoint d_0 is a stably
    identified physical parameter per layer, or just noise the optimizer is
    scattering across the window's rolling-CV fits.

    Physical Meaning:
    -----------------
    d_0 is the sigmoid inflection point in the GibbsKernel's length-scale
    function l(d) = l_min + (l_max - l_min) / (1 + exp(-k*(d - d_0))) --
    i.e. the distance from the coast where the fitted covariance structure
    switches from "coastal regime" to "offshore regime". If d_0 is a real,
    identifiable physical scale (e.g. shelf-break / coastal transition zone
    width), it should be roughly consistent window-to-window within a layer.
    If it varies wildly (large coefficient of variation) or a large fraction
    of windows sit pinned at the config's lower/upper bound, that means the
    per-window CV fit isn't actually resolving an inflection point -- the
    likelihood surface is flat/multimodal in d_0 and the optimizer is just
    landing wherever it started or hit a wall, not "identifying" a coastal
    transition scale.

    Parameters:
    -----------
    layer_data : dict[str, pd.DataFrame]
        Loaded audit dataframes for skin, source, and background layers.
    layer_configs : dict
        Layer metadata, including each layer's configured
        `d_transition_bounds_km` (lower, upper) tuple.
    output_dir : str
        Destination directory for the distribution figure.
    bound_tol_km : float
        A window's d_0 counts as "pinned" at a bound if it is within this
        many km of that bound.
    cv_flag_threshold : float
        Coefficient of variation (std/mean) above which a layer's d_0 is
        flagged as not meaningfully identified.

    Returns:
    --------
    pd.DataFrame
        One row per layer: n, mean, std, cv, median, IQR, min, max, and the
        fraction of windows pinned at the lower/upper bound.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    box_data = []
    box_labels = []
    box_colors = []

    for key in ['skin', 'source', 'background']:
        df = layer_data[key]
        cfg = layer_configs[key]
        d0 = df['d_transition_km'].dropna()
        lower, upper = cfg['d_transition_bounds_km']
        n = len(d0)
        mean = d0.mean()
        std = d0.std()
        cv = std / mean if mean else np.nan
        frac_lower = (d0 <= lower + bound_tol_km).mean()
        frac_upper = (d0 >= upper - bound_tol_km).mean()
        rows.append({
            'layer': key,
            'title': cfg['title'],
            'n_windows': n,
            'mean_km': mean,
            'std_km': std,
            'coefficient_of_variation': cv,
            'median_km': d0.median(),
            'iqr_low_km': d0.quantile(0.25),
            'iqr_high_km': d0.quantile(0.75),
            'min_km': d0.min(),
            'max_km': d0.max(),
            'bound_lower_km': lower,
            'bound_upper_km': upper,
            'frac_windows_pinned_lower': frac_lower,
            'frac_windows_pinned_upper': frac_upper,
            'identifiable': cv < cv_flag_threshold and (frac_lower + frac_upper) < 0.5
        })
        box_data.append(d0.values)
        box_labels.append(cfg['title'])
        box_colors.append(cfg['color'])

    stats_df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # FIGURE: d_0 distribution per layer (boxplot + individual window points)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True,
                     meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': 'black'})
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # Jittered individual-window scatter on top of each box, so bimodal/bound-pinned
    # distributions are visible rather than hidden inside the box summary.
    rng = np.random.default_rng(0)
    for i, (key, values) in enumerate(zip(['skin', 'source', 'background'], box_data), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, color=layer_configs[key]['color'],
                   edgecolor='black', linewidth=0.3, s=18, alpha=0.7, zorder=3)

    # Draw each layer's configured bounds as dashed reference lines so pinning is visible.
    for i, key in enumerate(['skin', 'source', 'background'], start=1):
        lower, upper = layer_configs[key]['d_transition_bounds_km']
        ax.hlines([lower, upper], i - 0.3, i + 0.3, colors='red', linestyles=':', linewidth=1.2)

    ax.set_ylabel("Coastal Transition Midpoint $d_0$ (km)", fontsize=11, fontweight='bold')
    ax.set_title("Is $d_0$ a Stable Physical Scale, or Optimizer Noise?\n"
                 "Per-window $d_0$ distribution by layer (red dashed = configured bounds)",
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "vertical_delta_d0_distribution.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Save the stats table alongside the figure for later reference.
    stats_csv_path = os.path.join(output_dir, "..", "..", "aelogs", "vertical_delta_d0_distribution_stats.csv")
    stats_csv_path = os.path.normpath(stats_csv_path)
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved: {stats_csv_path}")

    print("\n--- d_0 Identifiability Check (per layer) ---")
    for row in rows:
        flag = "OK" if row['identifiable'] else "RED FLAG -- not reliably identified"
        print(f"  {row['title']}: mean={row['mean_km']:.1f}km, std={row['std_km']:.1f}km, "
              f"CV={row['coefficient_of_variation']:.2f}, "
              f"pinned_lower={row['frac_windows_pinned_lower']*100:.0f}%, "
              f"pinned_upper={row['frac_windows_pinned_upper']*100:.0f}%  ==> {flag}")

    return stats_df


def run_vertical_delta_analysis(region="californiav3", year=2015):
    """
    Main driver for the Vertical Delta Analysis across the 3 scientific layers.

    Parameters:
    -----------
    region : str
        EBUS region key (default: 'californiav3').
    year : int
        Analysis calendar year (default: 2015).

    Returns:
    --------
    pd.DataFrame
        Compiled summary metrics table.
    """
    print("=" * 75)
    print(f"  RUNNING VERTICAL DELTA ANALYSIS: {region.upper()} ({year})")
    print("=" * 75)

    base_results_dir = "AEResults/aelogs"
    output_plots_dir = "AEResults/aeplots/vertical_delta"
    output_csv_path = os.path.join(base_results_dir, f"vertical_delta_{region}_{year}.csv")

    layer_data, layer_configs = load_layer_audits(base_results_dir, region=region, year=year)
    merged_df = compute_vertical_delta_metrics(layer_data)
    
    # Save merged summary table
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Saved summary metrics CSV: {output_csv_path}")

    # Generate plots
    generate_vertical_delta_plots(layer_data, merged_df, layer_configs, output_dir=output_plots_dir)

    # d_0 identifiability check -- is the coastal-transition midpoint a stable
    # physical scale per layer, or is the optimizer scattering it window to window?
    analyze_d0_distribution(layer_data, layer_configs, output_dir=output_plots_dir)

    # Print summary statistics
    print("\n" + "=" * 75)
    print("  VERTICAL DELTA SYNTHESIS SUMMARY")
    print("=" * 75)
    for key, cfg in layer_configs.items():
        df = layer_data[key]
        print(f"\n--- {cfg['title']} ---")
        print(f"  Median RMSRE:         {df['rmsre'].median()*100:.2f}% (Mean: {df['rmsre'].mean()*100:.2f}%)")
        print(f"  Uncertainty Std Z:    {df['std_z'].mean():.4f} ± {df['std_z'].std():.4f}")
        print(f"  Median Anisotropy:    {df['anisotropy_ratio'].median():.2f} (Mean: {df['anisotropy_ratio'].mean():.2f})")
        print(f"  Median Transition d0: {df['d_transition_km'].median():.1f} km (Mean: {df['d_transition_km'].mean():.1f} km)")
        print(f"  Mean Argo Floats / w: {df['n_floats'].mean():.1f} floats")

    print("\n--- Cross-Layer Comparisons (Source vs. Background) ---")
    print(f"  Mean Anisotropy Differential (Source - BG): {merged_df['delta_anisotropy_source_bg'].mean():+.2f}")
    print(f"  Mean Anisotropy Ratio (Source / BG):        {merged_df['ratio_anisotropy_source_bg'].mean():.2f}x")
    print(f"  Coastal Transition Difference (Source vs BG): {layer_data['source']['d_transition_km'].median():.1f} km vs {layer_data['background']['d_transition_km'].median():.1f} km")
    print("=" * 75)

    return merged_df


if __name__ == '__main__':
    run_vertical_delta_analysis()
