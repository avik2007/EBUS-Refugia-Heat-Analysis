"""
03b_ae_plot_physics.py

Generates 4 physics history PNGs from an already-saved audit CSV.
No AWS or pipeline re-run needed. Reads the audit CSV from the aelogs directory
that was written by 03_ae_inspect_data.py and calls plot_physics_history() directly.

Run from ArgoEBUSCloud/:
    python 03b_ae_plot_physics.py
"""
import os
import pandas as pd
import matplotlib
# Use non-interactive backend so this runs headless on a server with no display.
matplotlib.use('Agg')

from ebus_core.ae_utils import get_ae_config
from ebus_core.argoebus_gp_physics import plot_physics_history

# Config must match the completed Skin Layer run exactly.
# These parameters determine the run_id, which is used to locate the audit CSV.
config = get_ae_config(
    region="california",
    lat_step=0.5, lon_step=0.5,
    time_step=30.0,
    depth_range=(0, 100)
)

run_id   = config['run_id']
# Resolve paths relative to this script's location so it works from any cwd.
# This script lives at ArgoEBUSCloud/03b_ae_plot_physics.py.
# AEResults is one level up at ArgoEBUSAnalysis/AEResults/.
base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir  = os.path.join(base_dir, "..", "AEResults", "aelogs", run_id)
csv_path = os.path.join(log_dir, f"audit_{run_id}.csv")

print(f"run_id  : {run_id}")
print(f"log_dir : {log_dir}")
print(f"csv     : {csv_path}")

results_df = pd.read_csv(csv_path)

# Produce and save all 4 physics history figures.
# save_dir and run_id tell plot_physics_history() where to write the PNGs.
plot_physics_history(results_df, cv_details=None, time_unit='days',
                     save_dir=log_dir, run_id=run_id)
