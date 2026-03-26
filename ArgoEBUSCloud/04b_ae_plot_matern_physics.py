"""
04b_ae_plot_matern_physics.py

Regenerates physics history PNGs from the two audit CSVs written by
04_ae_testmatern_and_3dwindow.py, without re-running the expensive GP analysis.

Analogous to 03b_ae_plot_physics.py but operates on both the 2D-RBF baseline
and the 3D-Matern(nu=0.5) audit CSVs so you can compare physics plots side by
side after the fact.

Also prints a concise comparison table (RMSRE, Z-score, anisotropy ratio) and
saves a combined comparison PNG showing both RMSRE time series on one axes.

Run from ArgoEBUSCloud/:
    conda run -n ebus-cloud-env python 04b_ae_plot_matern_physics.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ebus_core.ae_utils import get_ae_config
from ebus_core.argoebus_gp_physics import plot_physics_history

# --- CONFIG ---
# Must match the run used in 04_ae_testmatern_and_3dwindow.py exactly.
config = get_ae_config(
    region="california",
    lat_step=0.5, lon_step=0.5,
    time_step=30.0,
    depth_range=(0, 100),
)

run_id   = config['run_id']
base_dir = os.path.dirname(os.path.abspath(__file__))
ae_root  = os.path.join(base_dir, "..", "AEResults", "aelogs")

# Variant IDs must match what 04_ae_testmatern_and_3dwindow.py wrote.
variant_2d = f"{run_id}_2d_rbf"
variant_3d = f"{run_id}_3d_matern05"
dir_2d     = os.path.join(ae_root, variant_2d)
dir_3d     = os.path.join(ae_root, variant_3d)

print(f"run_id   : {run_id}")
print(f"2D folder: {dir_2d}")
print(f"3D folder: {dir_3d}")

# --- LOAD AUDIT CSVs ---
csv_2d = os.path.join(dir_2d, f"audit_{variant_2d}.csv")
csv_3d = os.path.join(dir_3d, f"audit_{variant_3d}.csv")

if not os.path.exists(csv_2d):
    raise FileNotFoundError(
        f"2D baseline audit CSV not found: {csv_2d}\n"
        f"Run 04_ae_testmatern_and_3dwindow.py first."
    )
if not os.path.exists(csv_3d):
    raise FileNotFoundError(
        f"3D Matern audit CSV not found: {csv_3d}\n"
        f"Run 04_ae_testmatern_and_3dwindow.py first."
    )

results_2d = pd.read_csv(csv_2d)
results_3d = pd.read_csv(csv_3d)
print(f"\nLoaded 2D audit: {results_2d.shape[0]} windows")
print(f"Loaded 3D audit: {results_3d.shape[0]} windows")

# --- REGENERATE INDIVIDUAL PHYSICS PNGs ---
# Each call saves Plots 1-4 (and Plot 5 for the 3D run if scale_time_days present).
print("\nGenerating 2D RBF physics PNGs...")
plot_physics_history(results_2d, cv_details=None, time_unit='days',
                     save_dir=dir_2d, run_id=variant_2d)

print("\nGenerating 3D Matern physics PNGs...")
plot_physics_history(results_3d, cv_details=None, time_unit='days',
                     save_dir=dir_3d, run_id=variant_3d)

# --- COMBINED RMSRE COMPARISON PLOT ---
# Overlay both RMSRE time series on one axes so the improvement (or lack thereof)
# is immediately visible. Saved to both the 2D and 3D folders.
t2 = results_2d['window_center']
t3 = results_3d['window_center']

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(t2, results_2d['rmsre'] * 100, color='royalblue', marker='o',
        linestyle='-', linewidth=2, label='2D RBF baseline')
ax.plot(t3, results_3d['rmsre'] * 100, color='tomato', marker='s',
        linestyle='-', linewidth=2, label='3D Matern(nu=0.5)')
ax.axhline(5.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
           label='Target: 5% RMSRE')
ax.set_ylabel("RMSRE (%)")
ax.set_xlabel("Window Center (days since 1999-01-01)")
ax.set_title(f"RMSRE Comparison: 2D RBF vs. 3D Matern(nu=0.5)\n{run_id}")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

# Save a copy to both variant folders so neither audit folder is orphaned.
for save_dir, variant in [(dir_2d, variant_2d), (dir_3d, variant_3d)]:
    path = os.path.join(save_dir, f"rmsre_comparison_{run_id}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
plt.close(fig)

# --- COMPARISON SUMMARY TABLE ---
print(f"\n{'='*60}")
print(f"COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"{'Metric':<32} {'2D RBF':>10} {'3D Matern':>12}")
print(f"{'-'*60}")

for col, label, fmt in [
    ('rmsre',           'Median RMSRE',          '.3%'),
    ('rmsre',           'Mean RMSRE',             '.3%'),
    ('std_z',           'Median Std Z-Score',     '.3f'),
    ('anisotropy_ratio','Median Anisotropy Ratio','.3f'),
]:
    agg = 'median' if 'Median' in label else 'mean'
    v2 = getattr(results_2d[col].dropna(), agg)()
    v3 = getattr(results_3d[col].dropna(), agg)()
    s2 = format(v2, fmt)
    s3 = format(v3, fmt)
    print(f"  {label:<30} {s2:>10} {s3:>12}")

if 'scale_time_days' in results_3d.columns:
    t_med = results_3d['scale_time_days'].dropna().median()
    print(f"\n  3D temporal persistence (median): {t_med:.1f} days")

target = 0.05
won_2d = (results_2d['rmsre'].dropna() <= target).mean() * 100
won_3d = (results_3d['rmsre'].dropna() <= target).mean() * 100
print(f"\n  Windows meeting RMSRE < 5% target:")
print(f"    2D RBF:       {won_2d:.0f}%")
print(f"    3D Matern0.5: {won_3d:.0f}%")
