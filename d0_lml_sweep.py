"""
d0_lml_sweep.py -- is the GibbsKernel's d_0 (sigmoid midpoint distance, km) identified by the data?

For every rolling 45-day window of one layer-year, refit the Gibbs GP with d_0 clamped at each
value of a log-spaced grid and record the log-marginal-likelihood (LML) of each fit. If LML is
nearly flat across the grid the data cannot tell one d_0 from another (the per-window free-fit
d_0 scatter, CV 0.6-0.7 in 2015, is then an optimiser wandering on a plateau). If LML has a
clear peak, d_0 is identified and the scatter has another cause (init sensitivity, multimodality).

Same method as the session-19 time_ls sensitivity sweep, but with d_0 as the swept parameter.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ArgoEBUSCloud"))
from ebus_core.ae_utils import get_ae_config
from ebus_core.argoebus_gp_physics import analyze_rolling_correlations

# Default d_0 grid in km: log-spaced, spanning the narrow Skin/Source bounds (50-700) and the
# widened Background bound (50-1500; domain max dist_to_coast is 1404.8 km).
D0_GRID_KM = (50.0, 100.0, 200.0, 400.0, 800.0, 1500.0)


def run_d0_lml_sweep(region, lat_step, lon_step, time_step, depth_range, year,
                     d0_grid_km=D0_GRID_KM,
                     time_ls_bounds_days=(15.0, 200.0),
                     spatial_ls_upper_bound=10.0,
                     step_size_days=10,
                     clamp_rel_width=5e-4):
    """
    Sweep d_0 over `d0_grid_km` for one layer-year and return a long-format DataFrame.

    region, lat_step, lon_step, time_step, depth_range: same meaning as run_diagnostic_inspection;
        they pick the S3 parquet (one layer, one year).
    year: calendar year (int); the parquet covers Jan 1 - Dec 31 of that year.
    d0_grid_km: d_0 values to clamp at (km, distance-to-coast at the sigmoid midpoint).
    time_ls_bounds_days, spatial_ls_upper_bound, step_size_days: passed through unchanged so the
        fits match the free-fit runs (configs *_gibbs_lml.yaml: time_ls 15-200 d, spatial 10, 10 d stride).
    clamp_rel_width: d_0 is held by bounds d0*(1 -/+ clamp_rel_width). Bounds must not be exactly
        equal (sklearn treats that as a fixed hyperparameter and breaks the custom theta layout).

    Returns columns: d0_grid_km, window_start, window_center, lml, d_transition_km (the value the
        optimiser actually returned; must sit within the clamp band), rmsre, std_z.
    """
    cfg = get_ae_config(region=region, lat_step=lat_step, lon_step=lon_step, time_step=time_step,
                        depth_range=depth_range,
                        start_date=f"{year}-01-01", end_date=f"{year}-12-31")
    df = pd.read_parquet(f"s3://{cfg['s3_bucket']}/{cfg['run_id']}.parquet")

    frames = []
    for d0 in d0_grid_km:
        gibbs_params = {
            'l_min_km': 100.0, 'l_max_km': 400.0,
            'd_transition_init_km': float(d0),
            'd_transition_bounds_km': (d0 * (1 - clamp_rel_width), d0 * (1 + clamp_rel_width)),
            'k_steepness_init': 0.01, 'k_steepness_bounds': (1.0e-4, 1.0),
            'anisotropy_lat_lon_ratio': 2.0, 'anisotropy_lat_lon_ratio_bounds': (1.0, 4.0),
        }
        res, _ = analyze_rolling_correlations(
            df=df, feature_cols=['lat_bin', 'lon_bin'], target_col='ohc_per_m', time_col='time_bin',
            window_size_days=45, step_size_days=step_size_days, auto_tune=True, mode='3D',
            kernel_type='gibbs', gibbs_params=gibbs_params,
            time_ls_bounds_days=time_ls_bounds_days, spatial_ls_upper_bound=spatial_ls_upper_bound,
        )
        res = res[['window_start', 'window_center', 'lml', 'd_transition_km', 'rmsre', 'std_z']].copy()
        res.insert(0, 'd0_grid_km', float(d0))
        frames.append(res)
        print(f"  {year} d{depth_range[0]}_{depth_range[1]} d0={d0:7.1f} km -> {len(res)} windows, "
              f"median LML {res['lml'].median():.2f}", flush=True)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--depth", type=int, nargs=2, required=True, metavar=("TOP", "BOTTOM"))
    ap.add_argument("--region", default="californiav3")
    a = ap.parse_args()
    out = run_d0_lml_sweep(a.region, 0.5, 0.5, 10.0, tuple(a.depth), a.year)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AEResults", "aelogs", "d0_lml_sweep")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"d0_lml_sweep_d{a.depth[0]}_{a.depth[1]}_{a.year}.csv")
    out.to_csv(path, index=False)
    print(f"saved {path}")
