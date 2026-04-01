"""
=============================================================================
07_ae_deeper_layers.py
=============================================================================
Runs the canonical GPR analysis on the two deeper Vertical Sandwich layers:
  - Source Layer:     depth_range=(150, 400)  — Ekman upwelling source water
  - Background Layer: depth_range=(500, 1000) — deep ocean baseline

Canonical configuration (now baked into run_diagnostic_inspection() defaults):
  region              = californiav2       (updated boundary polygon)
  lat_step/lon_step   = 0.5° x 0.5°
  time_step           = 30.0 d            (window stride)
  step_size_days      = 10               (Argo sampling cadence)
  spatial_ls_upper_bound = 10            (allows large coherence lengths in deep water)
  time_ls_bounds_days = (15.0, 45.0)     (temporal correlation window)

No per-layer overrides are needed — all parameters match the canonical defaults.

Outputs per layer:
  AEResults/aelogs/{run_id}/
    audit_{...}.csv         — per-window RMSRE, Z-score, length scales
    cv_details_{...}.pkl    — raw CV error points
    lat_lon_evolution, noise_evolution, zscore_std, anisotropy,
    temporal_persistence    — physics PNGs
  AEResults/aeplots/snapshot_{run_id}/
    snapshot_..._day{N}.png — kriged OHC map per window
=============================================================================
"""

import importlib.util
import sys
import os

# Load run_diagnostic_inspection from 05_ae_update_tomatern0.5.py.
# importlib is required because the filename starts with a digit.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "skin_layer_pipeline",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_ae_update_tomatern0.5.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_diagnostic_inspection = mod.run_diagnostic_inspection

# Common config — must match the cloud run parameters used in Script 02
COMMON = dict(region="californiav2", lat_step=0.5, lon_step=0.5, time_step=10.0)  # FX2: matches 10d bin width from Script 02

# Accept an optional command-line argument to run only one layer:
#   python 07_ae_deeper_layers.py source      -> Source Layer only
#   python 07_ae_deeper_layers.py background  -> Background Layer only
#   python 07_ae_deeper_layers.py             -> both (sequential)
arg = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

if arg in ("source", "both"):
    print("=" * 60)
    print("SOURCE LAYER (150-400m)")
    print("=" * 60)
    run_diagnostic_inspection(**COMMON, depth_range=(150, 400))

if arg in ("background", "both"):
    print("=" * 60)
    print("BACKGROUND LAYER (500-1000m)")
    print("=" * 60)
    # All layers use canonical defaults — no per-layer overrides required.
    # spatial_ls_upper_bound=10 is now the permanent default in run_diagnostic_inspection(),
    # having been validated as necessary for deep water masses with large coherence lengths.
    run_diagnostic_inspection(**COMMON, depth_range=(500, 1000))
