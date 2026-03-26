# Design Plan: 3D Spatio-Temporal GP with Exponential Kernel

**Branch:** `gaussian-kriging-rework`
**Target file:** `ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py`
**Scope:** `analyze_rolling_correlations` and `plot_physics_history` only.
All other functions (`generalized_cross_validation`, `produce_kriging_map`) are unchanged.

---

## 1. Motivation

The current 2D model treats every observation within a rolling 90-day window as
existing in a purely spatial domain. Two floats 5 km apart but 60 days apart are
treated identically to two floats 5 km apart and 1 day apart. This is physically
wrong: ocean temperature fields have finite temporal memory.

Adding time as a third GP dimension forces the model to downweight temporally distant
observations when interpolating to a target date, potentially reducing RMSRE by
eliminating spurious spatial correlations from observations made under different
seasonal conditions.

---

## 2. Kernel Choice: Exponential vs. Squared Exponential

The **Squared Exponential (RBF)** kernel, currently in use, is:

    k_RBF(x, x') = exp( -r² / 2 )

    where r² = Σ_i  ( (x_i - x_i') / l_i )²

This covariance function is infinitely differentiable in space, implying
arbitrarily smooth predicted fields. For large-scale ocean heat content this is
reasonable, but in eddy-rich regimes (e.g. California Current summer) the
temperature field has sharp fronts at mesoscale (~10–50 km) that RBF
over-smoothes.

The **Exponential kernel** (Matérn ν = 1/2, also called the
Ornstein-Uhlenbeck covariance) is:

    k_Exp(x, x') = exp( -r )

    where r = sqrt( Σ_i  ( (x_i - x_i') / l_i )² )

It is only once-differentiable, generating sample paths with sharp transitions.
In one dimension, it is the covariance of the OU process — the simplest model of
mean-reverting noise — which is the standard first-principles model for
ocean anomaly decay.

The key difference: k_RBF decays as a Gaussian in r (fast decay near origin,
then sub-exponential tails), while k_Exp decays linearly in r on a log scale
(constant exponential decay at all distances). Observations just outside the
correlation length receive more weight under k_Exp than under k_RBF.

Both kernels are available as a runtime parameter (`kernel_type`). The function
signature, scaling logic, and output format are identical regardless of which
kernel is chosen, so a caller can run both back-to-back on the same data and
compare results directly without changing any other code. This modularity is the
primary design goal of this branch.

---

## 3. 3D Feature Space

In **2D mode** (unchanged default), the feature vector for each observation is:

    x = ( lat, lon )

In **3D mode** (new), the feature vector is extended to:

    x = ( lat, lon, t̃ )

where t̃ is the time coordinate normalized to the rolling window (see §4).

The kernel then computes distances in this three-dimensional space, so the GP
naturally downweights observations that are far in time from the prediction
target, even if they are spatially nearby.

---

## 4. Scaling and Normalization

### 4.1 Spatial dimensions (unchanged)

Latitude and longitude are each centered and scaled by their empirical standard
deviation within the rolling window using a `StandardScaler`:

    x̃_lat = ( lat - μ_lat ) / σ_lat
    x̃_lon = ( lon - μ_lon ) / σ_lon

This is identical to the current 2D implementation.

### 4.2 Time dimension (new in 3D mode)

A StandardScaler cannot be used for time. Within a 90-day window, the standard
deviation of time is roughly 26 days, making the "1 sigma" unit inconsistent
between windows with different data distributions.

Instead, time is normalized **relative to the window**:

    t̃ = ( t - t_center ) / ( W / 2 )

where:
- t is the raw `time_days` value of the observation
- t_center = t_window_start + W/2 is the center date of the rolling window
- W = window_size_days (default 90)

This maps the window uniformly: observations at the window center map to t̃ = 0,
observations at either edge map to t̃ = ±1, regardless of how many floats are
present and when they arrived.

The physical time length scale can be recovered after optimization:

    l_t_phys  =  l̃_t  ×  ( W / 2 )   [days]

For example, l̃_t = 0.5 in a 90-day window means l_t_phys = 22.5 days.

### 4.3 Why keep the scalers separate

If time were passed to the same StandardScaler as lat/lon, the scaler's `scale_`
array would mix degrees and days, and the physical-unit recovery formula
(`learned_ls_phys = scaled_ls × scaler.scale_`) would produce wrong units for
the time dimension. By building a separate `phys_scale_X` array that contains
[σ_lat, σ_lon, W/2], the recovery formula works uniformly across all dimensions.

---

## 5. Length-Scale Bounds for the Time Dimension

During hyperparameter optimization, the time length scale l̃_t is constrained to:

    l̃_t  ∈  [ l_min / (W/2),  l_max / (W/2) ]

where l_min = 2 days and l_max = 30 days are the physical bounds.

For a 90-day window (W/2 = 45 days):

    l̃_t  ∈  [ 2/45,  30/45 ]  ≈  [ 0.044,  0.667 ]

Physical rationale:
- **Lower bound (2 days):** Below this, the GP treats observations 2 days apart as
  nearly uncorrelated. The ocean does not change coherently faster than mesoscale
  adjustment timescales (~2–5 days for surface eddies), so values below this
  indicate numerical instability, not real physics.
- **Upper bound (30 days):** Above this, the GP ignores time entirely and behaves
  like a 2D spatial model. Keeping the upper bound at 30 days ensures the 3D
  model always uses temporal information and does not degenerate to the 2D case.

Spatial dimensions retain their existing bounds (0.01 to 5 in scaled units).

---

## 6. Centric-Snapshot Prediction

When the trained 3D GP generates predictions onto the output spatial grid, the
time coordinate of every prediction point is fixed to **t̃ = 0** (the window
center date). This is equivalent to asking: "what is the best estimate of the
ocean state at the center of this time window, using all observations from the
window as evidence?"

In normalized units, t̃ = 0 maps to t = t_center, so the transform
`t̃_predict = (t_center - t_center) / (W/2) = 0` is exact. No further
modification to the kriging output loop is needed.

---

## 7. Changes to `analyze_rolling_correlations`

**New parameters added to the function signature:**

| Parameter | Default | Purpose |
|---|---|---|
| `mode` | `'2D'` | `'2D'` = original behaviour. `'3D'` = add time dimension. |
| `kernel_type` | `'rbf'` | `'rbf'` = Squared Exponential. `'matern0.5'` = Exponential / OU. |
| `time_ls_bounds_days` | `(2.0, 30.0)` | Physical day bounds for the time length scale. Only used in 3D mode. |

All defaults preserve existing 2D-RBF behaviour. Existing callers require no changes.

**Modularity principle:** `kernel_type` is the only switch. The scaling, output
columns, calibration loop, and diagnostic plots are the same regardless of which
kernel is chosen. This means a comparison script can call `analyze_rolling_correlations`
twice — once with `kernel_type='rbf'` and once with `kernel_type='matern0.5'` — and
directly compare the two `results_df` outputs (e.g. median RMSRE, anisotropy ratio,
temporal persistence) without touching any other part of the pipeline. The same
`plot_physics_history` call works for both outputs.

**Internal changes in 3D mode:**
1. Time column is detected and appended to the feature list.
2. A split scaler is applied (StandardScaler for spatial, relative norm for time).
3. A `phys_scale_X` array is constructed: [σ_lat, σ_lon, W/2].
4. The kernel is built via a local factory function that switches between RBF and
   Matern(ν=0.5) based on `kernel_type`, with per-dimension bounds in 3D mode.
5. The auto-calibration loop (noise adjustment) re-uses the same factory.
6. The results record loop iterates over `all_feature_cols` (spatial + time), so
   `scale_time_days` is written automatically.
7. The anisotropy ratio calculation is changed from hardcoded `scale_lat_bin /
   scale_lon_bin` to a dynamic lookup that finds any column matching `scale_lat*`
   and `scale_lon*`.

---

## 8. Changes to `plot_physics_history`

| Plot | Change |
|---|---|
| Plot 1 (Spatial Scales) | Filter to spatial-only columns; exclude `scale_time*` and `scale_day*`. |
| Plot 4 (Anisotropy) | Resolve lat/lon column names dynamically; skip gracefully if absent. |
| Plot 5 (Temporal Persistence) | **New.** Conditional on `scale_time_days` present in results_df. Shows optimized time length scale in days over the 12-month cycle. |

Plot 5 physical interpretation: high values (> 20 days) indicate the model finds
long-memory water masses (typical of the Source and Background layers). Low values
(< 10 days) indicate rapidly evolving conditions (typical of Skin Layer summer).
This plot is a direct observable of the stealth warming hypothesis: the temporal
persistence of heat anomalies should increase with depth.

---

## 9. What Is Not Changed

- `generalized_cross_validation`: remains 2D-only, RBF kernel. Its purpose is
  global validation of a single set of hyperparameters, not rolling physics
  tracking. Adding time here would conflate data from all seasons.
- `produce_kriging_map`: remains as-is. It already implements is_3d logic and its
  centric-snapshot behaviour (predicting at t_val = the current time slice) is
  already correct.
- `03b_ae_plot_physics.py`: no changes. It calls `plot_physics_history` generically
  and will receive Plot 5 automatically when the audit CSV contains `scale_time_days`.

---

## 10. Validation Criterion

Two runs on the 2015 California Skin Layer audit data:

1. **Baseline:** mode='2D', kernel_type='rbf' (current production behaviour)
2. **New:** mode='3D', kernel_type='matern0.5'

Success criterion: median RMSRE across all rolling windows drops below 5% in the
new run, OR the new run demonstrates a statistically consistent improvement over
the baseline that justifies using the 3D model for the Source and Background layer
cloud runs.
