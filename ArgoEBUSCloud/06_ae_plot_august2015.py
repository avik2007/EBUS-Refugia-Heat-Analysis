"""
=============================================================================
06_ae_plot_august2015.py
=============================================================================
Generates an improved kriged OHC-per-m map for August 2015 using the
existing cloud-run data (california_20150101_20151231_res0_5x0_5_t30_0_d0_100).

Does NOT re-run the GPR pipeline — it loads:
  1. The raw Argo parquet from S3 (needed to reconstruct the GP and plot
     the Argo float positions on the map).
  2. The pre-computed audit CSV from AEResults/aelogs/ (contains the
     tuned kernel parameters for every window).

Then calls plot_kriging_snapshot() for the window closest to mid-August 2015.

Label improvements applied in plot_kriging_snapshot() (see argoebus_gp_physics.py):
  - Colorbar: "OHC per m (J/m²)" instead of raw column name
  - Title: "Predicted Map: August 2015" instead of opaque numeric offset
  - Uncertainty colorbar: "1σ Uncertainty (J/m²)" — same units as predicted field
=============================================================================
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend; safe for script execution
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Ensure we can import ebus_core from within ArgoEBUSCloud/
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ebus_core.ae_utils import get_ae_config, ensure_ae_dirs
from ebus_core.argoebus_gp_physics import plot_kriging_snapshot

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# These parameters must match the existing cloud run exactly so we load the
# right parquet and the right audit CSV.
REGION     = "california"
LAT_STEP   = 0.5
LON_STEP   = 0.5
TIME_STEP  = 30.0
DEPTH_RANGE = (0, 100)

# Pipeline epoch: all time_bin / window_center values are days since this date.
TIME_EPOCH = date(1999, 1, 1)

# Target: mid-August 2015 in days since TIME_EPOCH.
# 1999-01-01 to 2015-08-15:
#   1999-01-01 to 2015-01-01 = 16 years * 365 + 4 leap days = 5844 days
#   2015-01-01 to 2015-08-15 = 31+28+31+30+31+30+31+15 = 227 days
# Total = 6071 days.  The function finds the nearest window_center anyway.
TARGET_DATE_DAYS = (date(2015, 8, 15) - TIME_EPOCH).days

print(f"Target date: 2015-08-15 = day {TARGET_DATE_DAYS} since {TIME_EPOCH}")

# ---------------------------------------------------------------------------
# 1. LOAD CONFIG & PATHS
# ---------------------------------------------------------------------------
config = get_ae_config(
    region=REGION,
    lat_step=LAT_STEP,
    lon_step=LON_STEP,
    time_step=TIME_STEP,
    depth_range=DEPTH_RANGE
)
run_id = config['run_id']
print(f"Run ID: {run_id}")

ensure_ae_dirs()

# Locate audit CSV — pre-computed kernel parameters for every rolling window.
# AEResults/ lives one level above ArgoEBUSCloud/.
base_dir    = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "..", "AEResults", "aelogs", run_id)
audit_path  = os.path.join(results_dir, f"audit_{run_id}.csv")

if not os.path.exists(audit_path):
    raise FileNotFoundError(
        f"Audit CSV not found at {audit_path}. "
        "Run 03_ae_inspect_data.py first to generate it."
    )

print(f"Loading audit CSV from: {audit_path}")
results_df = pd.read_csv(audit_path)
print(f"  Audit shape: {results_df.shape}")
print(f"  Window center range: {results_df['window_center'].min():.0f} — "
      f"{results_df['window_center'].max():.0f} days")

# Verify August 2015 is covered by the audit data
wc_min = results_df['window_center'].min()
wc_max = results_df['window_center'].max()
if not (wc_min <= TARGET_DATE_DAYS <= wc_max):
    raise ValueError(
        f"TARGET_DATE_DAYS={TARGET_DATE_DAYS} is outside the audit range "
        f"[{wc_min:.0f}, {wc_max:.0f}]. "
        "Check that the run covers 2015-08."
    )

# ---------------------------------------------------------------------------
# 2. LOAD RAW ARGO DATA FROM S3
# ---------------------------------------------------------------------------
# The raw parquet is needed to:
#   a) Slice the data window for GP fitting
#   b) Overlay Argo float positions on the map
s3_uri = f"s3://{config['s3_bucket']}/{run_id}.parquet"
print(f"Loading raw Argo data from: {s3_uri}")
df_raw = pd.read_parquet(s3_uri)
print(f"  Raw data shape: {df_raw.shape}")

# ---------------------------------------------------------------------------
# 3. GENERATE THE KRIGED MAP FOR AUGUST 2015
# ---------------------------------------------------------------------------
print(f"\nGenerating kriged snapshot for day {TARGET_DATE_DAYS} (mid-August 2015)...")

fig = plot_kriging_snapshot(
    df_raw=df_raw,
    results_df=results_df,
    target_date=TARGET_DATE_DAYS,
    feature_cols=['lat_bin', 'lon_bin'],
    target_col='ohc_per_m',
    time_col='time_bin',
    # Use a 30-day window consistent with how the audit was generated
    window_size_days=30,
    grid_res=0.25,
    cmap='magma_r',
    # units_label and time_epoch are passed explicitly so plot_kriging_snapshot
    # produces "J/m²" colorbars and "August 2015" titles without guessing.
    units_label="J/m²",
    time_epoch=TIME_EPOCH
)

if fig is None:
    print("ERROR: Not enough data in window — cannot generate plot.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. SAVE OUTPUT
# ---------------------------------------------------------------------------
plot_dir  = os.path.join(base_dir, "..", "AEResults", "aeplots")
save_path = os.path.join(plot_dir, f"august2015_ohc_kriged_{run_id}.png")

fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=False)
plt.close(fig)

print(f"\nSaved: {save_path}")
