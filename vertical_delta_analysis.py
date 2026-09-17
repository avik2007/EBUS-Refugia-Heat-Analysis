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
            'color': '#1f77b4'
        },
        'source': {
            'depth_range': (150, 400),
            'title': 'Source Layer (150–400m)',
            'folder': f"{region}_{year}0101_{year}1231_res0_5x0_5_t10_0_d150_400_3dgibbs_w45_timelsfix",
            'thickness_m': 250.0,
            'color': '#ff7f0e'
        },
        'background': {
            'depth_range': (500, 1000),
            'title': 'Background Layer (500–1000m)',
            'folder': f"{region}_{year}0101_{year}1231_res0_5x0_5_t10_0_d500_1000_3dgibbs_w45_timelsfix",
            'thickness_m': 500.0,
            'color': '#2ca02c'
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
