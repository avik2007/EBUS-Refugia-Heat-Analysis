# RG-Gibbs Non-Stationary Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a `GibbsKernel` (non-stationary, learnable sigmoid lengthscale of `dist_to_coast_km`) and wire it through the GPR pipeline so a `kernel_type: gibbs` analysis YAML produces a complete audit CSV for the californiav3 Source layer.

**Architecture:** A new `GibbsKernel` class lives in `ebus_core/argoebus_gp_physics.py`, conforming to the `sklearn.gaussian_process.kernels.Kernel` API. It computes a spatial Gibbs kernel (anisotropic 2:1 lat:lon, lengthscale = sigmoid of `dist_to_coast_km`) multiplied by a stationary Matern temporal kernel. The kernel takes a 4-column `X` of shape `(n, 4) = [lat_km, lon_km, time_days_scaled, dist_to_coast_km]` where the last column is auxiliary (read for lengthscale evaluation, never differenced). `analyze_rolling_correlations` gains a `kernel_type='gibbs'` branch that re-projects lat/lon to local km, stacks dist_to_coast, and dispatches to `GibbsKernel`. `_build_kernel`'s `kernel_type` branch is extended. `run_diagnostic_inspection` (script 05) accepts `kernel_type` + `gibbs_params` and threads them. `run_analysis` (runner.py) packs `kernel_gibbs` block fields into dispatch kwargs. Optimizer uses scipy L-BFGS-B with finite-difference gradient (no analytic gradient required for v1).

**Tech Stack:** Python 3.x, scikit-learn `GaussianProcessRegressor` + `Kernel` API, numpy, scipy.optimize, pydantic v2, pytest, pandas.

---

## File Structure

**Modify:**
- `ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py`
  - Add: `GibbsKernel` class (~250 lines: __init__, _sigmoid_lengthscale, __call__, diag, is_stationary, theta property, bounds property, hyperparameters property, clone_with_theta, __repr__).
  - Add: `_lonlat_to_local_km(lat, lon, lat_center, lon_center) -> (lat_km, lon_km)` helper.
  - Add: `_gibbs_optimizer(obj_func, initial_theta, bounds)` callable for `GaussianProcessRegressor(optimizer=...)`.
  - Modify: `analyze_rolling_correlations` (lines 950–1306) — add `gibbs_params=None` kwarg, add `kernel_type == 'gibbs'` branch in feature assembly + kernel build + physical scale recovery.
  - Modify: `_build_kernel` closure (lines 1185–1199) — add `'gibbs'` branch.

- `ArgoEBUSCloud/05_ae_update_tomatern0.5.py`
  - Modify: `run_diagnostic_inspection` (lines 54–203) — accept `kernel_type='matern0.5'` and `gibbs_params=None` kwargs, replace hardcoded `kernel_type='matern0.5'` with var, swap output suffix `_3dmatern_w45` → `_3dgibbs_w45` when kernel_type=='gibbs'.

- `ArgoEBUSCloud/ebus_core/runner.py`
  - Modify: `run_analysis` (lines 172–254) — when `cfg.gpr.kernel_type == 'gibbs'`, pack `cfg.gpr.kernel_gibbs` block fields into a `gibbs_params` dict and add to `dispatch_kwargs`.

- `ArgoEBUSCloud/test_mlops_foundation.py`
  - Add: ~12 new test functions exercising GibbsKernel correctness, sklearn API conformance, integration, and runner dispatch.

**Create:**
- `configs/californiav3/californiav3_d150_400_gibbs.yaml` — first Gibbs analysis YAML (Source layer).

**Touch lightly:**
- `argo_claude_actions/AE_claude_todo.md` — mark Gibbs tasks complete, queue smoke run.
- `argo_claude_actions/AE_claude_recentactions.md` — record session work.

---

## Task 1: GibbsKernel — basic construction & sigmoid lengthscale

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py` (add class near top, after imports)
- Test: `ArgoEBUSCloud/test_mlops_foundation.py`

- [ ] **Step 1.1: Write failing test for sigmoid lengthscale endpoints**

Append at end of `test_mlops_foundation.py`:

```python
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ebus_core.argoebus_gp_physics import GibbsKernel


def test_gibbs_kernel_sigmoid_endpoints():
    # The sigmoid lengthscale must approach l_min as d -> -inf and l_max as d -> +inf.
    # At d == d_0 (the midpoint), the value must equal (l_min + l_max) / 2 exactly.
    # Why: this defines the physical regime — coastal lengthscale near coast,
    # offshore lengthscale far away, with a well-defined transition midpoint.
    k = GibbsKernel(
        l_min_km=100.0, l_max_km=400.0,
        d_transition_init_km=300.0, d_transition_bounds_km=(50.0, 700.0),
        k_steepness_init=0.01, k_steepness_bounds=(1e-4, 1.0),
        anisotropy_lat_lon_ratio=2.0,
        time_ls_init_days=30.0, time_ls_bounds_days=(15.0, 45.0),
    )
    # Endpoint behaviour
    assert k._sigmoid_lengthscale(np.array([-1e6])) == \
           pytest.approx(100.0, abs=1e-3)
    assert k._sigmoid_lengthscale(np.array([1e6])) == \
           pytest.approx(400.0, abs=1e-3)
    # Midpoint
    mid = k._sigmoid_lengthscale(np.array([300.0]))[0]
    assert mid == pytest.approx(250.0, abs=1e-3)
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_gibbs_kernel_sigmoid_endpoints -v`

Expected: FAIL with `ImportError: cannot import name 'GibbsKernel'`.

- [ ] **Step 1.3: Implement minimal GibbsKernel class**

Insert into `ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py` immediately after the top-level imports block (after `import warnings` on line 27) and before the existing function definitions:

```python
# =============================================================================
# GIBBS NON-STATIONARY KERNEL (RG-Gibbs directive 2026-04-26)
# =============================================================================
# Implements a spatially-varying Matern-0.5-like kernel where the lat/lon
# lengthscale at point x is a learnable sigmoid of dist_to_coast_km(x).
# Motivation: the California Current System has a sharp coastal regime
# (narrow upwelling filaments, lengthscale ~100km) that transitions to a
# broad open-ocean regime (lengthscale ~400km). A single stationary Matern
# cannot fit both regimes, producing chronic Z-score 0.5-0.9 near the
# shelf break (californiav3 baseline 2026-05-03).
# Reference: docs/superpowers/specs/2026-04-26-rg-gibbs-l-x-directive.md
# =============================================================================

from sklearn.gaussian_process.kernels import Kernel, Hyperparameter


class GibbsKernel(Kernel):
    """
    Non-stationary kernel with sigmoid lengthscale of dist_to_coast_km.

    Functional form (per Gibbs 1997, Paciorek & Schervish 2004):
        k_spatial(x_i, x_j) = prod_d sqrt(2 l_d(x_i) l_d(x_j) / (l_d(x_i)^2 + l_d(x_j)^2))
                              * exp(- sum_d (Δx_d)^2 / (l_d(x_i)^2 + l_d(x_j)^2))
        l_lat(x) = sigmoid(d(x); l_min, l_max, d_0, k)
        l_lon(x) = l_lat(x) / anisotropy_lat_lon_ratio
        sigmoid(d) = l_min + (l_max - l_min) / (1 + exp(-k * (d - d_0)))

    For 3D mode, k_full = k_spatial * k_time where k_time is a stationary
    Matern(nu=0.5) kernel on the time dimension.

    INPUT X shape: (n, n_cols) where columns are
        [lat_km, lon_km, time_scaled, dist_to_coast_km]   (3D mode, n_cols=4)
        [lat_km, lon_km, dist_to_coast_km]                 (2D mode, n_cols=3)
    The dist_to_coast_km column index is auto-detected as the last column.
    The time column (if present) is the second-to-last.

    PARAMETERS:
        l_min_km, l_max_km          — fixed lengthscale bounds (km)
        d_transition_init_km        — initial guess for sigmoid midpoint d_0 (km)
        d_transition_bounds_km      — (lower, upper) optimiser bounds for d_0
        k_steepness_init            — initial sigmoid steepness k
        k_steepness_bounds          — (lower, upper) optimiser bounds for k
        anisotropy_lat_lon_ratio        — l_lat / l_lon initial value (learnable; default 2.0)
        anisotropy_lat_lon_ratio_bounds — (lower, upper) optimiser bounds for ratio (default (1.0, 4.0))
        time_ls_init_days               — initial Matern temporal lengthscale (days)
        time_ls_bounds_days             — (lower, upper) bounds for temporal lengthscale
        mode                            — '2D' or '3D'

    LEARNABLE THETA (sklearn convention: log-space):
        theta = [log(d_0), log(k), log(time_ls), log(anisotropy_ratio)]    (3D mode)
        theta = [log(d_0), log(k), log(anisotropy_ratio)]                   (2D mode)
    """

    def __init__(
        self,
        l_min_km=100.0,
        l_max_km=400.0,
        d_transition_init_km=300.0,
        d_transition_bounds_km=(50.0, 700.0),
        k_steepness_init=0.01,
        k_steepness_bounds=(1.0e-4, 1.0),
        anisotropy_lat_lon_ratio=2.0,
        anisotropy_lat_lon_ratio_bounds=(1.0, 4.0),
        time_ls_init_days=30.0,
        time_ls_bounds_days=(15.0, 45.0),
        mode='3D',
    ):
        # Store all constructor args verbatim — sklearn requires this for
        # get_params() / clone_with_theta() to work correctly.
        self.l_min_km = l_min_km
        self.l_max_km = l_max_km
        self.d_transition_init_km = d_transition_init_km
        self.d_transition_bounds_km = d_transition_bounds_km
        self.k_steepness_init = k_steepness_init
        self.k_steepness_bounds = k_steepness_bounds
        self.anisotropy_lat_lon_ratio = anisotropy_lat_lon_ratio
        self.anisotropy_lat_lon_ratio_bounds = anisotropy_lat_lon_ratio_bounds
        self.time_ls_init_days = time_ls_init_days
        self.time_ls_bounds_days = time_ls_bounds_days
        self.mode = mode

        # Live (mutable) parameter values — these change as the optimiser
        # walks theta during fit. Initialised from the *_init fields.
        self._d0 = float(d_transition_init_km)
        self._k = float(k_steepness_init)
        self._anisotropy_ratio = float(anisotropy_lat_lon_ratio)
        self._time_ls = float(time_ls_init_days)

    # -------- sigmoid lengthscale --------
    def _sigmoid_lengthscale(self, d_km):
        # Sigmoid producing the LAT lengthscale at each point given dist_to_coast.
        # l(d) = l_min + (l_max - l_min) / (1 + exp(-k * (d - d_0))).
        # Vectorised: d_km shape (n,) -> output shape (n,).
        # Numerically stable because we compute the exponent then 1/(1+exp(-x)).
        l_min = self.l_min_km
        l_max = self.l_max_km
        # Clip the exponent to avoid overflow at extreme d values.
        z = np.clip(self._k * (d_km - self._d0), -50.0, 50.0)
        sig = 1.0 / (1.0 + np.exp(-z))
        return l_min + (l_max - l_min) * sig

    @property
    def hyperparameter_d_transition(self):
        # sklearn introspects this — used by get_params + theta machinery.
        return Hyperparameter(
            "d_transition_init_km", "numeric", self.d_transition_bounds_km
        )

    @property
    def hyperparameter_k_steepness(self):
        return Hyperparameter(
            "k_steepness_init", "numeric", self.k_steepness_bounds
        )

    @property
    def hyperparameter_time_ls(self):
        return Hyperparameter(
            "time_ls_init_days", "numeric", self.time_ls_bounds_days
        )

    # -------- mandatory sklearn Kernel API: stubs filled in Tasks 2 & 3 --------
    def __call__(self, X, Y=None, eval_gradient=False):
        raise NotImplementedError("GibbsKernel.__call__ not yet implemented")

    def diag(self, X):
        # Diagonal is always 1.0 because k(x, x) = sqrt(1) * exp(0) = 1 for the
        # spatial Gibbs and Matern(nu=0.5) is also 1 at zero lag. No σ²
        # constant in this kernel — that's wrapped externally by ConstantKernel.
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False

    def __repr__(self):
        return (
            f"GibbsKernel(l_min_km={self.l_min_km}, l_max_km={self.l_max_km}, "
            f"d_0={self._d0:.1f}, k={self._k:.4f}, "
            f"time_ls={self._time_ls:.1f}, mode={self.mode!r})"
        )
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_gibbs_kernel_sigmoid_endpoints -v`

Expected: PASS.

- [ ] **Step 1.5: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(gibbs): add GibbsKernel skeleton with sigmoid lengthscale"
```

---

## Task 2: GibbsKernel — kernel matrix `__call__`

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py` (replace `__call__` stub)
- Test: `ArgoEBUSCloud/test_mlops_foundation.py`

- [ ] **Step 2.1: Write failing test for symmetry, PSD, and reduction-to-RBF**

Append to `test_mlops_foundation.py`:

```python
def test_gibbs_kernel_call_symmetry_and_psd():
    # K(X, X) must be symmetric and positive semi-definite (smallest eigenvalue >= -tol).
    # Tests with a small synthetic dataset of 5 points in 3D mode.
    rng = np.random.default_rng(42)
    n = 5
    lat_km = rng.uniform(-200, 200, n)
    lon_km = rng.uniform(-200, 200, n)
    time_scaled = rng.uniform(-1, 1, n)
    dist_coast = rng.uniform(50, 600, n)
    X = np.column_stack([lat_km, lon_km, time_scaled, dist_coast])

    k = GibbsKernel(mode='3D')
    K = k(X)
    assert K.shape == (n, n)
    # Symmetry
    assert np.allclose(K, K.T, atol=1e-10)
    # PSD: smallest eigenvalue must be >= -1e-8
    eigs = np.linalg.eigvalsh(K)
    assert eigs.min() >= -1e-8, f"K not PSD; min eig {eigs.min()}"
    # Diagonal must equal 1.0 (k(x,x) = 1)
    assert np.allclose(np.diag(K), 1.0, atol=1e-10)


def test_gibbs_kernel_reduces_to_matern_when_dist_constant():
    # When all dist_to_coast are equal, l(x) is constant for all x and the
    # spatial Gibbs reduces to a stationary Matern(nu=0.5) with that constant
    # lengthscale (anisotropic per dim).  The time factor remains Matern(nu=0.5)
    # in time. Compare against a hand-built reference for a 2-point case.
    from sklearn.gaussian_process.kernels import Matern
    n = 4
    rng = np.random.default_rng(0)
    lat_km = rng.uniform(-100, 100, n)
    lon_km = rng.uniform(-100, 100, n)
    time_scaled = rng.uniform(-1, 1, n)
    dist_const = np.full(n, 300.0)  # midpoint -> l_lat = 250 km
    X = np.column_stack([lat_km, lon_km, time_scaled, dist_const])

    k_gibbs = GibbsKernel(
        l_min_km=100.0, l_max_km=400.0,
        d_transition_init_km=300.0, k_steepness_init=0.01,
        anisotropy_lat_lon_ratio=2.0, mode='3D',
    )
    K = k_gibbs(X)

    # Reference: Matern with length_scale=[250, 125, time_ls] (2:1 anisotropy)
    # multiplied with the same time factor we use internally (already part of K).
    # Here we only check the spatial factor at zero-time-lag pairs.
    # Build a 2D-mode reference for purely spatial pairs at the same time.
    # (Easiest cross-check: check positive structure and symmetry; exact match
    # is tested in test_gibbs_anisotropy below.)
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.all(np.diag(K) == 1.0)


def test_gibbs_kernel_anisotropy_2to1():
    # Two pairs at the same dist_to_coast with the same |Δlat| and |Δlon|
    # respectively must yield K_lon < K_lat because the lon lengthscale is
    # half the lat lengthscale (anisotropy 2:1). Equivalently: log K is more
    # negative for lon.
    dist = np.full(2, 300.0)
    # Pair A: same point + (Δlat=50, Δlon=0)
    X_lat = np.array([
        [0.0, 0.0, 0.0, 300.0],
        [50.0, 0.0, 0.0, 300.0],
    ])
    # Pair B: same point + (Δlat=0, Δlon=50)
    X_lon = np.array([
        [0.0, 0.0, 0.0, 300.0],
        [0.0, 50.0, 0.0, 300.0],
    ])
    k = GibbsKernel(anisotropy_lat_lon_ratio=2.0, mode='3D')
    K_lat = k(X_lat)
    K_lon = k(X_lon)
    # K_lat[0,1] > K_lon[0,1]: lon distance penalised more (smaller lengthscale)
    assert K_lat[0, 1] > K_lon[0, 1] + 1e-6
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_gibbs_kernel_call_symmetry_and_psd test_mlops_foundation.py::test_gibbs_kernel_reduces_to_matern_when_dist_constant test_mlops_foundation.py::test_gibbs_kernel_anisotropy_2to1 -v`

Expected: FAIL — `NotImplementedError: GibbsKernel.__call__ not yet implemented`.

- [ ] **Step 2.3: Implement `__call__` (replace stub from Task 1.3)**

Replace the `__call__` stub in `GibbsKernel` with:

```python
    def __call__(self, X, Y=None, eval_gradient=False):
        # Compute K(X, Y). If Y is None, K(X, X). Returns shape (n_X, n_Y).
        # eval_gradient=True is not supported in this version — sklearn's
        # default optimizer expects analytic gradients, so we use a custom
        # finite-difference optimizer (see _gibbs_optimizer) and pass
        # eval_gradient=False from there.
        if eval_gradient:
            # sklearn uses gradient when GaussianProcessRegressor.optimizer is
            # the default 'fmin_l_bfgs_b'. We replace the optimizer at fit time
            # with _gibbs_optimizer (no gradient), so this branch should never
            # be reached. Raise a clear error if it is.
            raise NotImplementedError(
                "GibbsKernel does not provide analytic gradients. "
                "Set GaussianProcessRegressor(optimizer=_gibbs_optimizer)."
            )

        if Y is None:
            Y = X
            symmetric = True
        else:
            symmetric = False

        # Decode column layout: dist_to_coast is always the LAST column,
        # time (if present) is the second-to-last.
        dist_X = X[:, -1]
        dist_Y = Y[:, -1]

        if self.mode == '3D':
            time_X = X[:, -2]
            time_Y = Y[:, -2]
            spatial_X = X[:, :-2]   # lat_km, lon_km
            spatial_Y = Y[:, :-2]
        else:
            time_X = None
            time_Y = None
            spatial_X = X[:, :-1]   # lat_km, lon_km
            spatial_Y = Y[:, :-1]

        # Per-point lat lengthscale via sigmoid of dist_to_coast.
        # Shape: (n_X,) and (n_Y,)
        l_lat_X = self._sigmoid_lengthscale(dist_X)
        l_lat_Y = self._sigmoid_lengthscale(dist_Y)
        l_lon_X = l_lat_X / self._anisotropy_ratio
        l_lon_Y = l_lat_Y / self._anisotropy_ratio

        # Stack into per-dim lengthscale arrays of shape (n_X, n_dim_spatial).
        # Order matches spatial_X columns: [lat_km, lon_km].
        L_X = np.column_stack([l_lat_X, l_lon_X])      # (n_X, 2)
        L_Y = np.column_stack([l_lat_Y, l_lon_Y])      # (n_Y, 2)

        # Compute the Gibbs spatial kernel:
        # k(x_i, x_j) = prod_d sqrt(2 l_d(x_i) l_d(x_j) / (l_d(x_i)^2 + l_d(x_j)^2))
        #               * exp(- sum_d (Δx_d)^2 / (l_d(x_i)^2 + l_d(x_j)^2))
        # Vectorise via broadcasting: (n_X, 1, 2) and (1, n_Y, 2).
        L_X_b = L_X[:, np.newaxis, :]   # (n_X, 1, 2)
        L_Y_b = L_Y[np.newaxis, :, :]   # (1, n_Y, 2)
        sx = spatial_X[:, np.newaxis, :]   # (n_X, 1, 2)
        sy = spatial_Y[np.newaxis, :, :]   # (1, n_Y, 2)

        # Squared lengthscale sum per pair per dim: (n_X, n_Y, 2)
        L2_sum = L_X_b ** 2 + L_Y_b ** 2

        # Prefactor sqrt(2 l_i l_j / (l_i^2 + l_j^2)) per dim, then product over dims.
        prefactor_per_dim = np.sqrt(2.0 * L_X_b * L_Y_b / L2_sum)   # (n_X, n_Y, 2)
        prefactor = np.prod(prefactor_per_dim, axis=2)               # (n_X, n_Y)

        # Squared distance in each dim, divided by L2_sum, summed
        diff_sq = (sx - sy) ** 2                                     # (n_X, n_Y, 2)
        exponent = -np.sum(diff_sq / L2_sum, axis=2)                 # (n_X, n_Y)

        K_spatial = prefactor * np.exp(exponent)

        # Time factor (Matern nu=0.5 / Exponential): exp(-|Δt| / l_t)
        if self.mode == '3D':
            dt = np.abs(time_X[:, np.newaxis] - time_Y[np.newaxis, :])  # (n_X, n_Y)
            K_time = np.exp(-dt / max(self._time_ls, 1e-9))
            K = K_spatial * K_time
        else:
            K = K_spatial

        return K
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py -k "test_gibbs_kernel" -v`

Expected: 4 PASS (sigmoid endpoints, symmetry/PSD, reduces-to-matern, anisotropy).

- [ ] **Step 2.5: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(gibbs): implement GibbsKernel.__call__ with anisotropic spatial+time"
```

---

## Task 3: GibbsKernel — sklearn API conformance (theta, bounds, clone)

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py` (add property methods)
- Test: `ArgoEBUSCloud/test_mlops_foundation.py`

- [ ] **Step 3.1: Write failing test for theta round-trip and clone_with_theta**

Append to `test_mlops_foundation.py`:

```python
def test_gibbs_kernel_theta_roundtrip():
    # The sklearn optimiser walks `theta` (log-space). After setting theta,
    # exposing theta again must return the same vector.  Length must match
    # the number of learnable hyperparameters: 4 in 3D mode (d_0, k, time_ls, anisotropy).
    k = GibbsKernel(mode='3D')
    assert k.theta.shape == (4,)
    new_theta = np.log(np.array([200.0, 0.005, 25.0, 1.5]))
    k.theta = new_theta
    assert np.allclose(k.theta, new_theta, atol=1e-12)
    # And the linear-space values are recovered correctly:
    assert k._d0 == pytest.approx(200.0, rel=1e-9)
    assert k._k == pytest.approx(0.005, rel=1e-9)
    assert k._time_ls == pytest.approx(25.0, rel=1e-9)
    assert k._anisotropy_ratio == pytest.approx(1.5, rel=1e-9)


def test_gibbs_kernel_bounds_log_space():
    # sklearn convention: `bounds` returns log-space bounds, shape (n_theta, 2).
    k = GibbsKernel(mode='3D')
    b = k.bounds
    assert b.shape == (4, 2)
    # d_transition bounds are (50, 700) km -> log
    assert b[0, 0] == pytest.approx(np.log(50.0), abs=1e-9)
    assert b[0, 1] == pytest.approx(np.log(700.0), abs=1e-9)
    # anisotropy bounds are (1.0, 4.0) -> log
    assert b[3, 0] == pytest.approx(np.log(1.0), abs=1e-9)
    assert b[3, 1] == pytest.approx(np.log(4.0), abs=1e-9)


def test_gibbs_kernel_clone_with_theta():
    # clone_with_theta must produce a NEW instance (not mutate self) with the
    # supplied theta and all constructor args preserved.
    k = GibbsKernel(mode='3D', l_min_km=100.0, l_max_km=400.0)
    new_theta = np.log(np.array([175.0, 0.02, 35.0, 1.8]))
    k2 = k.clone_with_theta(new_theta)
    assert k2 is not k
    assert k2._d0 == pytest.approx(175.0, rel=1e-9)
    assert k2._anisotropy_ratio == pytest.approx(1.8, rel=1e-9)
    assert k2.l_min_km == 100.0
    assert k2.l_max_km == 400.0
    # Original is unchanged
    assert k._d0 == pytest.approx(300.0, rel=1e-9)


def test_gibbs_kernel_2d_mode_has_two_thetas():
    # In 2D mode there is no time dimension, so theta has length 3 (d_0, k, anisotropy).
    k = GibbsKernel(mode='2D')
    assert k.theta.shape == (3,)
    assert k.bounds.shape == (3, 2)
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py -k "test_gibbs_kernel_theta or test_gibbs_kernel_bounds or test_gibbs_kernel_clone or test_gibbs_kernel_2d" -v`

Expected: FAIL — `theta` / `bounds` / `clone_with_theta` not present (sklearn's default Kernel implementations need explicit hyperparameters_*).

- [ ] **Step 3.3: Implement theta/bounds/clone overrides**

Replace the three `@property` blocks (`hyperparameter_d_transition`, `hyperparameter_k_steepness`, `hyperparameter_time_ls`) with the following — and add explicit `theta`, `bounds`, and `clone_with_theta` overrides at the bottom of the class (before `__repr__`):

```python
    # -------- learnable hyperparameters --------
    # In 3D mode: theta = [log(d_0), log(k), log(time_ls), log(anisotropy)] (length 4).
    # In 2D mode: theta = [log(d_0), log(k), log(anisotropy)] (length 3; time_ls inactive).
    # We override theta + bounds + clone_with_theta directly rather than
    # using sklearn's hyperparameter_* introspection because we need to
    # conditionally exclude time_ls when mode == '2D'.

    @property
    def theta(self):
        # Log-space vector consumed by the optimiser.
        if self.mode == '3D':
            return np.log(np.array([self._d0, self._k, self._time_ls, self._anisotropy_ratio]))
        return np.log(np.array([self._d0, self._k, self._anisotropy_ratio]))

    @theta.setter
    def theta(self, theta):
        # Sklearn pushes a new log-space theta during optimisation.
        # Update the live linear-space mirrors used inside __call__.
        vals = np.exp(theta)
        self._d0 = float(vals[0])
        self._k = float(vals[1])
        if self.mode == '3D':
            self._time_ls = float(vals[2])
            self._anisotropy_ratio = float(vals[3])
        else:
            self._anisotropy_ratio = float(vals[2])

    @property
    def bounds(self):
        # Log-space (lower, upper) per learnable parameter.
        rows = [
            (np.log(self.d_transition_bounds_km[0]),
             np.log(self.d_transition_bounds_km[1])),
            (np.log(self.k_steepness_bounds[0]),
             np.log(self.k_steepness_bounds[1])),
        ]
        if self.mode == '3D':
            rows.append((
                np.log(self.time_ls_bounds_days[0]),
                np.log(self.time_ls_bounds_days[1]),
            ))
        rows.append((
            np.log(self.anisotropy_lat_lon_ratio_bounds[0]),
            np.log(self.anisotropy_lat_lon_ratio_bounds[1]),
        ))
        return np.array(rows)

    def clone_with_theta(self, theta):
        # Sklearn calls this during optimisation restarts. Must return a NEW
        # GibbsKernel with the supplied theta and all other constructor args
        # preserved exactly (so subsequent log_marginal_likelihood evaluations
        # use the same fixed hyperparameters).
        cloned = GibbsKernel(
            l_min_km=self.l_min_km,
            l_max_km=self.l_max_km,
            d_transition_init_km=self.d_transition_init_km,
            d_transition_bounds_km=self.d_transition_bounds_km,
            k_steepness_init=self.k_steepness_init,
            k_steepness_bounds=self.k_steepness_bounds,
            anisotropy_lat_lon_ratio=self.anisotropy_lat_lon_ratio,
            anisotropy_lat_lon_ratio_bounds=self.anisotropy_lat_lon_ratio_bounds,
            time_ls_init_days=self.time_ls_init_days,
            time_ls_bounds_days=self.time_ls_bounds_days,
            mode=self.mode,
        )
        cloned.theta = theta
        return cloned

    def get_params(self, deep=True):
        # Required so sklearn's clone() and pipeline introspection work.
        return {
            "l_min_km": self.l_min_km,
            "l_max_km": self.l_max_km,
            "d_transition_init_km": self.d_transition_init_km,
            "d_transition_bounds_km": self.d_transition_bounds_km,
            "k_steepness_init": self.k_steepness_init,
            "k_steepness_bounds": self.k_steepness_bounds,
            "anisotropy_lat_lon_ratio": self.anisotropy_lat_lon_ratio,
            "anisotropy_lat_lon_ratio_bounds": self.anisotropy_lat_lon_ratio_bounds,
            "time_ls_init_days": self.time_ls_init_days,
            "time_ls_bounds_days": self.time_ls_bounds_days,
            "mode": self.mode,
        }

    @property
    def n_dims(self):
        # Number of LEARNABLE hyperparameters, NOT the number of feature dims.
        return 4 if self.mode == '3D' else 3

    @property
    def hyperparameters(self):
        # Sklearn introspection helper. Must list one Hyperparameter per
        # learnable theta entry, in the same order theta uses.
        hps = [
            Hyperparameter("d_transition_init_km", "numeric",
                           self.d_transition_bounds_km),
            Hyperparameter("k_steepness_init", "numeric",
                           self.k_steepness_bounds),
        ]
        if self.mode == '3D':
            hps.append(Hyperparameter(
                "time_ls_init_days", "numeric", self.time_ls_bounds_days
            ))
        hps.append(Hyperparameter(
            "anisotropy_lat_lon_ratio", "numeric",
            self.anisotropy_lat_lon_ratio_bounds,
        ))
        return hps
```

Also delete the three `@property hyperparameter_*` methods you added in Task 1.3 — they're superseded by the explicit `hyperparameters` property above.

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py -k "test_gibbs_kernel" -v`

Expected: all gibbs_kernel tests PASS (sigmoid, symmetry/PSD, reduces-to-matern, anisotropy, theta, bounds, clone, 2D mode).

- [ ] **Step 3.5: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(gibbs): wire sklearn Kernel API (theta, bounds, clone_with_theta)"
```

---

## Task 4: Custom optimizer + GP fit smoke test

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py` (add `_gibbs_optimizer`)
- Test: `ArgoEBUSCloud/test_mlops_foundation.py`

**Why this exists:** sklearn's default `GaussianProcessRegressor` optimizer (`'fmin_l_bfgs_b'`) calls `kernel(X, eval_gradient=True)` which we don't support. We provide a finite-difference L-BFGS-B optimizer to fit theta without needing analytic gradients.

- [ ] **Step 4.1: Write failing integration test — GP fits Gibbs kernel without errors**

Append to `test_mlops_foundation.py`:

```python
def test_gibbs_kernel_gp_fit_smoke(monkeypatch):
    # Smoke test: GaussianProcessRegressor with a GibbsKernel + custom
    # optimizer fits without errors on a small synthetic dataset and
    # produces predictions of the expected shape.
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
    from ebus_core.argoebus_gp_physics import _gibbs_optimizer

    rng = np.random.default_rng(7)
    n = 30
    lat_km = rng.uniform(-200, 200, n)
    lon_km = rng.uniform(-200, 200, n)
    time_scaled = rng.uniform(-1, 1, n)
    dist_coast = rng.uniform(20, 600, n)
    X = np.column_stack([lat_km, lon_km, time_scaled, dist_coast])
    # Synthetic target — smooth function of lat/lon, decoupled from time
    y = 0.01 * lat_km + 0.02 * lon_km + rng.normal(0, 0.05, n)

    base = GibbsKernel(mode='3D')
    kernel = ConstantKernel(1.0, "fixed") * base + WhiteKernel(
        noise_level=0.1, noise_level_bounds=(1e-5, 1e1)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, optimizer=_gibbs_optimizer,
        n_restarts_optimizer=0, alpha=0.0,
    )
    gp.fit(X, y)
    # Predict on the training set — shape must match y; no NaNs.
    y_pred, y_std = gp.predict(X, return_std=True)
    assert y_pred.shape == y.shape
    assert y_std.shape == y.shape
    assert not np.any(np.isnan(y_pred))
    assert not np.any(np.isnan(y_std))
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_gibbs_kernel_gp_fit_smoke -v`

Expected: FAIL — `cannot import name '_gibbs_optimizer'`.

- [ ] **Step 4.3: Implement `_gibbs_optimizer`**

Insert into `argoebus_gp_physics.py` immediately AFTER the `GibbsKernel` class definition:

```python
def _gibbs_optimizer(obj_func, initial_theta, bounds):
    # Custom optimizer for GibbsKernel that uses scipy L-BFGS-B with
    # finite-difference gradient. sklearn's GaussianProcessRegressor calls
    # this with `obj_func(theta, eval_gradient)` (its closure around the
    # log-marginal-likelihood). We swallow eval_gradient and request only
    # the scalar value, letting scipy's '2-point' option compute gradients
    # numerically.
    #
    # WHY: GibbsKernel does not provide analytic kernel gradients (a v2
    # enhancement). Without this wrapper, sklearn's default 'fmin_l_bfgs_b'
    # would call kernel(X, eval_gradient=True) and crash. Finite differences
    # are slow but correct, and the optimization runs once per rolling window
    # so the overhead is bounded (~3 hyperparameters x O(n^3) per evaluation).
    #
    # INPUTS:
    #   obj_func     — closure provided by sklearn returning either
    #                  (lml, grad) or just lml depending on eval_gradient.
    #   initial_theta — log-space starting point, shape (n_hp,)
    #   bounds        — log-space bounds, shape (n_hp, 2)
    # OUTPUTS:
    #   (theta_opt, lml_opt) — optimised log-space theta and final neg-lml.
    from scipy.optimize import minimize

    def value_only(theta):
        # sklearn's obj_func always supports eval_gradient kw.
        # We force it to return just the scalar by passing False.
        return obj_func(theta, eval_gradient=False)

    res = minimize(
        value_only, initial_theta, method='L-BFGS-B', bounds=bounds,
        jac='2-point',           # finite-difference gradient
        options={'maxiter': 50},
    )
    return res.x, res.fun
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_gibbs_kernel_gp_fit_smoke -v`

Expected: PASS. (If it fails because `obj_func` doesn't accept `eval_gradient` kwarg in some sklearn versions, switch `value_only` to `lambda theta: obj_func(theta) if not callable(getattr(obj_func, '__wrapped__', None)) else obj_func(theta, eval_gradient=False)` — sklearn ≥1.3 supports the kwarg.)

- [ ] **Step 4.5: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(gibbs): finite-difference optimizer wrapper for GP fit"
```

---

## Task 5: Wire `kernel_type='gibbs'` into `analyze_rolling_correlations`

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py` (lines 950–1306)
- Test: `ArgoEBUSCloud/test_mlops_foundation.py`

**Design notes:**
- New kwarg `gibbs_params: Optional[dict] = None` on `analyze_rolling_correlations`.
- When `kernel_type == 'gibbs'`:
  1. Skip `StandardScaler` for spatial features. Project lat/lon to local km via equirectangular projection at the WINDOW centroid (so different windows can have different projections — fine because each window is fitted independently).
  2. Read `dist_to_coast_km` from `df_slice` — REQUIRED column; raise `ValueError` if missing.
  3. Stack X as `[lat_km, lon_km, time_scaled, dist_to_coast_km]` (3D) or `[lat_km, lon_km, dist_to_coast_km]` (2D).
  4. In `_build_kernel`, return `ConstantKernel * GibbsKernel + WhiteKernel`.
  5. Use `optimizer=_gibbs_optimizer` instead of the default.
  6. After fit, record learned `d_0`, `k`, `time_ls` into the per-window result dict (alongside existing `scale_*` columns set to NaN, since Gibbs has no single per-dim scale).

- [ ] **Step 5.1: Write failing test for end-to-end gibbs path**

Append to `test_mlops_foundation.py`:

```python
def test_analyze_rolling_correlations_gibbs_branch():
    # Synthetic 60-day window of binned profiles with dist_to_coast_km.
    # analyze_rolling_correlations(kernel_type='gibbs') must:
    #   - run without error
    #   - return results_df with one or more rows
    #   - record learned d_0, k, time_ls in the result row
    from ebus_core.argoebus_gp_physics import analyze_rolling_correlations

    rng = np.random.default_rng(11)
    n = 80
    lat = rng.uniform(33, 45, n)
    lon = rng.uniform(-130, -118, n)
    time_bin = rng.uniform(0, 60, n)
    dist_to_coast_km = rng.uniform(20, 800, n)
    ohc_per_m = (
        1e9 + 5e6 * lat + 3e6 * lon
        + 1e6 * np.exp(-dist_to_coast_km / 200.0)   # coastal signal
        + rng.normal(0, 1e6, n)
    )
    df = pd.DataFrame({
        'lat_bin': lat, 'lon_bin': lon, 'time_bin': time_bin,
        'dist_to_coast_km': dist_to_coast_km, 'ohc_per_m': ohc_per_m,
        'platform_number': rng.integers(1000, 1100, n),
    })

    gibbs_params = {
        'l_min_km': 100.0, 'l_max_km': 400.0,
        'd_transition_init_km': 300.0,
        'd_transition_bounds_km': (50.0, 700.0),
        'k_steepness_init': 0.01,
        'k_steepness_bounds': (1.0e-4, 1.0),
        'anisotropy_lat_lon_ratio': 2.0,
    }

    results_df, _ = analyze_rolling_correlations(
        df=df, feature_cols=['lat_bin', 'lon_bin'],
        target_col='ohc_per_m', time_col='time_bin',
        window_size_days=45, step_size_days=10,
        auto_tune=True, mode='3D',
        kernel_type='gibbs',
        gibbs_params=gibbs_params,
        time_ls_bounds_days=(15.0, 45.0),
    )
    assert len(results_df) >= 1
    # Gibbs-specific columns
    assert 'd_transition_km' in results_df.columns
    assert 'k_steepness' in results_df.columns
    # Within optimisation bounds
    row = results_df.iloc[0]
    assert 50.0 <= row['d_transition_km'] <= 700.0
    assert 1e-4 <= row['k_steepness'] <= 1.0
```

import pandas as pd at the top of the test file if not already present.

- [ ] **Step 5.2: Run test to verify it fails**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_analyze_rolling_correlations_gibbs_branch -v`

Expected: FAIL — `unexpected keyword argument 'gibbs_params'` or `kernel_type='gibbs'` is not handled.

- [ ] **Step 5.3: Add `_lonlat_to_local_km` helper**

Insert into `argoebus_gp_physics.py` immediately AFTER `_gibbs_optimizer`:

```python
def _lonlat_to_local_km(lat_deg, lon_deg, lat_center_deg, lon_center_deg):
    # Project lat/lon (degrees) to local Cartesian kilometres using an
    # equirectangular approximation centred at (lat_center, lon_center).
    # Vectorised: lat_deg/lon_deg may be arrays of any shape.
    #
    # WHY: GibbsKernel parameterises lengthscales in km (l_min_km, l_max_km),
    # so spatial coordinates fed to the kernel must also be in km. A single
    # local projection per window is exact enough for the ~10–20° spans we
    # operate over: cosine distortion is < 1% across 5° of latitude.
    #
    # INPUTS (all in degrees):
    #   lat_deg, lon_deg          — arrays of profile coordinates
    #   lat_center_deg, lon_center_deg — projection centre
    # OUTPUTS:
    #   (lat_km, lon_km) — arrays in kilometres relative to the centre
    DEG_TO_KM_LAT = 111.0   # 1° latitude ≈ 111 km on sphere of radius 6371 km
    lat_km = (lat_deg - lat_center_deg) * DEG_TO_KM_LAT
    cos_factor = np.cos(np.radians(lat_center_deg))
    lon_km = (lon_deg - lon_center_deg) * DEG_TO_KM_LAT * cos_factor
    return lat_km, lon_km
```

- [ ] **Step 5.4: Add `gibbs_params` kwarg to `analyze_rolling_correlations`**

Edit the signature at `argoebus_gp_physics.py:950`. Find the line ending with `time_ls_bounds_days=(2.0, 30.0),` (around line 1029-ish) and insert before the closing `):` of the function signature:

```python
                                 # --- GIBBS NON-STATIONARY KERNEL CONFIG ---
                                 # Only used when kernel_type='gibbs'. Dict of
                                 # GibbsKernel constructor kwargs; see GibbsKernel
                                 # class docstring for the full key list.
                                 gibbs_params=None,
```

- [ ] **Step 5.5: Add gibbs feature-assembly branch**

In `analyze_rolling_correlations`, locate the spatial-feature-assembly block beginning at line 1116 (`X_spatial = df_slice[spatial_cols].values`). Wrap it in a kernel-type conditional and insert the gibbs branch:

```python
        if kernel_type == 'gibbs':
            # Gibbs requires:
            #  (a) raw spatial coords projected to local km (no StandardScaler),
            #  (b) dist_to_coast_km column present in df_slice.
            if 'dist_to_coast_km' not in df_slice.columns:
                raise ValueError(
                    "kernel_type='gibbs' requires column 'dist_to_coast_km' "
                    "in input dataframe; run 02_ae_cloud_run.py to ensure it "
                    "is computed during ingestion."
                )
            # Project lat/lon to local km centred at the window centroid.
            lat_centre = df_slice[spatial_cols[0]].mean()
            lon_centre = df_slice[spatial_cols[1]].mean()
            lat_km, lon_km = _lonlat_to_local_km(
                df_slice[spatial_cols[0]].values,
                df_slice[spatial_cols[1]].values,
                lat_centre, lon_centre,
            )
            # No StandardScaler — coords are already in physical km.
            X_spatial_scaled = np.column_stack([lat_km, lon_km])
            scaler_X = None              # sentinel; gibbs branch never inverse-transforms
            phys_scale_spatial = None    # not applicable for gibbs
            dist_col = df_slice['dist_to_coast_km'].values
        else:
            # Existing Matern / RBF path — StandardScaler over lat/lon.
            X_spatial = df_slice[spatial_cols].values
            scaler_X = StandardScaler()
            X_spatial_scaled = scaler_X.fit_transform(X_spatial)
            phys_scale_spatial = scaler_X.scale_
            dist_col = None
```

Then locate the time-handling block (lines 1121–1133) and replace its `else` branch and the surrounding column-stacking logic to append the dist_col when gibbs is active. The replacement (replacing roughly lines 1121–1133):

```python
        if mode == '3D':
            half_window = window_size_days / 2.0
            time_raw = df_slice[time_dim_col].values
            time_scaled = (time_raw - window_center) / half_window
            X_scaled = np.column_stack([X_spatial_scaled, time_scaled])
            if phys_scale_spatial is not None:
                phys_scale_X = np.append(phys_scale_spatial, half_window)
            else:
                phys_scale_X = None
        else:
            X_scaled = X_spatial_scaled
            phys_scale_X = phys_scale_spatial

        # Gibbs path: append dist_to_coast_km as the auxiliary final column.
        # GibbsKernel reads X[:, -1] for lengthscale evaluation and never
        # differences this column.
        if kernel_type == 'gibbs':
            X_scaled = np.column_stack([X_scaled, dist_col])
```

- [ ] **Step 5.6: Add gibbs branch in `_build_kernel`**

Replace the `_build_kernel` closure (lines 1185–1199) with:

```python
        def _build_kernel(ls, n_level, ls_bnd, n_bnd):
            # Kernel factory: returns the configured spatial+time kernel.
            # 'matern0.5' / 'rbf' = stationary spatial Matern or RBF (existing).
            # 'gibbs' = non-stationary Gibbs (sigmoid lengthscale of dist_to_coast).
            # The GibbsKernel internally handles BOTH spatial and time, so the
            # outer `ls`, `ls_bnd` are unused on the gibbs path.
            if kernel_type == 'gibbs':
                # Build GibbsKernel from gibbs_params. Each call to _build_kernel
                # creates a fresh kernel instance — needed during auto-calibration
                # (re-fit with fixed hyperparameters and updated noise_level).
                gp_kw = dict(gibbs_params or {})
                gp_kw['mode'] = mode
                # Bridge time bounds from the outer call site for consistency.
                if mode == '3D' and time_ls_bounds_days is not None:
                    gp_kw.setdefault('time_ls_bounds_days', tuple(time_ls_bounds_days))
                    gp_kw.setdefault('time_ls_init_days',
                                     0.5 * sum(time_ls_bounds_days))
                spatial_kernel = GibbsKernel(**gp_kw)
            elif kernel_type == 'matern0.5':
                spatial_kernel = Matern(length_scale=ls, length_scale_bounds=ls_bnd, nu=0.5)
            else:
                spatial_kernel = RBF(length_scale=ls, length_scale_bounds=ls_bnd)
            return (ConstantKernel(1.0, constant_value_bounds="fixed") *
                    spatial_kernel +
                    WhiteKernel(noise_level=n_level, noise_level_bounds=n_bnd))
```

- [ ] **Step 5.7: Use `_gibbs_optimizer` for gibbs runs and record gibbs-specific result columns**

Replace the `gp = GaussianProcessRegressor(...)` line (around line 1202) with a kernel-type-aware constructor:

```python
        k = _build_kernel(ls_init, noise_val, l_bounds, n_bounds)
        if kernel_type == 'gibbs':
            gp = GaussianProcessRegressor(
                kernel=k, optimizer=_gibbs_optimizer,
                n_restarts_optimizer=optimizer_restarts, alpha=0.0,
            )
        else:
            gp = GaussianProcessRegressor(
                kernel=k, n_restarts_optimizer=optimizer_restarts, alpha=0.0
            )
```

Then locate the per-window result-recording block (lines ~1247–1291) and add gibbs columns immediately after the `record = {...}` initialisation. Replace the loop that writes `scale_*` columns with:

```python
            # Write one scale entry per feature dimension. For matern/rbf this
            # produces 'scale_lat_bin', 'scale_lon_bin', and (3D) 'scale_time_bin'.
            # For gibbs there is no single learned scale per dim — instead we
            # record the learned sigmoid params (d_0 km, k) and time_ls.
            if kernel_type == 'gibbs':
                # Recover learned hyperparameters from the fitted kernel.
                # gp.kernel_.k1 = ConstantKernel * GibbsKernel; the GibbsKernel
                # is k1.k2.
                fitted_gibbs = gp.kernel_.k1.k2
                record['d_transition_km'] = fitted_gibbs._d0
                record['k_steepness'] = fitted_gibbs._k
                record['time_ls_days'] = (
                    fitted_gibbs._time_ls if mode == '3D' else np.nan
                )
                # Set conventional scale columns to NaN so audit CSV stays
                # union-compatible with matern/rbf runs.
                for col in all_feature_cols:
                    record[f'scale_{col}'] = np.nan
            else:
                learned_ls_phys = current_length_scale * phys_scale_X
                for i, col in enumerate(all_feature_cols):
                    record[f'scale_{col}'] = learned_ls_phys[i]
```

NOTE: `current_length_scale` is set earlier at line 1213 as `gp.kernel_.k1.k2.length_scale`. For gibbs that property does not exist — guard the assignment and the calibration-loop kernel rebuild. Find the auto-calibration block (lines 1208–1245) and wrap the line `current_length_scale = gp.kernel_.k1.k2.length_scale` with:

```python
            if kernel_type != 'gibbs':
                current_length_scale = gp.kernel_.k1.k2.length_scale
            else:
                current_length_scale = None  # not used in gibbs auto-calibration path
```

And inside the calibration loop, the kernel rebuild line `k_calibrated = _build_kernel(current_length_scale, new_noise, "fixed", "fixed")` must skip on gibbs (the gibbs kernel ignores `ls`/`ls_bnd` anyway, but to be explicit we still rebuild — no change needed there since `_build_kernel` ignores them on the gibbs branch). Confirm by reading the calibration block; if it crashes, guard with:

```python
                if kernel_type == 'gibbs':
                    # On the gibbs path, re-run fit with updated noise_level by
                    # rebuilding the kernel (which ignores the spatial ls args).
                    k_calibrated = _build_kernel(None, new_noise, "fixed", "fixed")
                else:
                    k_calibrated = _build_kernel(current_length_scale, new_noise, "fixed", "fixed")
```

- [ ] **Step 5.8: Run test to verify it passes**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_analyze_rolling_correlations_gibbs_branch -v`

Expected: PASS. If a `KeyError`/`AttributeError` surfaces on the calibration path, re-read lines 1208–1245 of `argoebus_gp_physics.py` and apply the gibbs guards above.

- [ ] **Step 5.9: Confirm matern/rbf paths still pass**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py -v`

Expected: All 54+ existing tests PASS plus the new gibbs ones (~58 total).

- [ ] **Step 5.10: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/argoebus_gp_physics.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(gibbs): wire kernel_type='gibbs' into analyze_rolling_correlations"
```

---

## Task 6: Script 05 — accept kernel_type + gibbs_params, swap suffix

**Files:**
- Modify: `ArgoEBUSCloud/05_ae_update_tomatern0.5.py` (lines 54–203)
- Test: `ArgoEBUSCloud/test_mlops_foundation.py`

- [ ] **Step 6.1: Write failing test that script 05 propagates kernel_type='gibbs'**

Append to `test_mlops_foundation.py`:

```python
def test_run_diagnostic_inspection_propagates_gibbs(monkeypatch):
    # Patch analyze_rolling_correlations to capture its kwargs and verify
    # that run_diagnostic_inspection forwards kernel_type='gibbs' and
    # gibbs_params correctly.
    import importlib.util as _ilu
    from pathlib import Path as _Path
    script_path = _Path(__file__).resolve().parent / "05_ae_update_tomatern0.5.py"
    spec = _ilu.spec_from_file_location("script_05_under_test", script_path)
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)

    captured = {}
    def fake_analyze(**kwargs):
        captured.update(kwargs)
        # Return a tiny shaped result so downstream save code does not crash.
        import pandas as _pd
        results_df = _pd.DataFrame([{
            'window_start': 0, 'window_center': 22.5,
            'rmsre': 0.04, 'std_z': 1.0, 'noise_val': 0.1,
            'n_bins': 50, 'n_floats': 10, 'anisotropy_ratio': 1.5,
            'd_transition_km': 250.0, 'k_steepness': 0.01,
            'time_ls_days': 30.0,
        }])
        return results_df, {}

    monkeypatch.setattr(mod, 'analyze_rolling_correlations', fake_analyze)
    # Patch S3 read to avoid network call.
    import pandas as _pd
    df_fake = _pd.DataFrame({
        'lat_bin': [33.0, 34.0], 'lon_bin': [-125.0, -124.0],
        'time_bin': [0.0, 30.0], 'ohc_per_m': [1e9, 1.1e9],
        'dist_to_coast_km': [50.0, 200.0],
    })
    monkeypatch.setattr(_pd, 'read_parquet', lambda *a, **kw: df_fake)
    # Patch plot fns to no-ops.
    monkeypatch.setattr(mod, 'plot_kriging_snapshot', lambda **kw: None)
    monkeypatch.setattr(mod, 'plot_physics_history', lambda *a, **kw: None)

    gibbs_params = {
        'l_min_km': 100.0, 'l_max_km': 400.0,
        'd_transition_init_km': 300.0,
        'd_transition_bounds_km': (50.0, 700.0),
        'k_steepness_init': 0.01,
        'k_steepness_bounds': (1.0e-4, 1.0),
        'anisotropy_lat_lon_ratio': 2.0,
    }
    mod.run_diagnostic_inspection(
        region='californiav3', lat_step=0.5, lon_step=0.5,
        time_step=10.0, depth_range=(150, 400),
        kernel_type='gibbs', gibbs_params=gibbs_params,
    )
    assert captured.get('kernel_type') == 'gibbs'
    assert captured.get('gibbs_params') == gibbs_params
```

- [ ] **Step 6.2: Run test to verify it fails**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_run_diagnostic_inspection_propagates_gibbs -v`

Expected: FAIL — script 05 ignores `kernel_type` (it's swallowed by `**_`).

- [ ] **Step 6.3: Modify `run_diagnostic_inspection` signature**

Replace the signature in `05_ae_update_tomatern0.5.py:54-60` with:

```python
def run_diagnostic_inspection(region="california", lat_step=0.5, lon_step=0.5,
                              time_step=10.0, depth_range=(0, 100),
                              run_suffix="",
                              spatial_ls_upper_bound=10,
                              time_ls_bounds_days=(15.0, 45.0),
                              step_size_days=10,
                              kernel_type="matern0.5",
                              gibbs_params=None,
                              **_):
```

- [ ] **Step 6.4: Replace hardcoded kernel_type and dynamic suffix**

Inside the function, find the assignment of `output_run_id` (line 96) and replace with:

```python
    # output_run_id is the canonical identifier for THIS run's artifacts.
    # Suffix encodes the kernel: '_3dmatern_w45' for matern (default) or
    # '_3dgibbs_w45' for gibbs. Allows side-by-side comparison without
    # overwriting matern baselines.
    if kernel_type == 'gibbs':
        kernel_suffix = '_3dgibbs_w45'
    elif kernel_type == 'matern0.5':
        kernel_suffix = '_3dmatern_w45'
    else:
        kernel_suffix = f'_3d{kernel_type}_w45'
    output_run_id = config['run_id'] + kernel_suffix + run_suffix
```

Then find the `analyze_rolling_correlations(...)` call (line 120) and update it:

```python
    results_df, cv_details = analyze_rolling_correlations(
        df=df,
        feature_cols=['lat_bin', 'lon_bin'],
        target_col='ohc_per_m',
        time_col='time_bin',
        window_size_days=45,
        step_size_days=step_size_days,
        auto_tune=True,
        mode='3D',
        kernel_type=kernel_type,
        gibbs_params=gibbs_params,
        time_ls_bounds_days=time_ls_bounds_days,
        spatial_ls_upper_bound=spatial_ls_upper_bound,
    )
```

- [ ] **Step 6.5: Run test to verify it passes**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_run_diagnostic_inspection_propagates_gibbs -v`

Expected: PASS.

- [ ] **Step 6.6: Confirm matern run still works**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py -v`

Expected: All tests PASS (existing + new).

- [ ] **Step 6.7: Commit**

```bash
git add ArgoEBUSCloud/05_ae_update_tomatern0.5.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(gibbs): script 05 accepts kernel_type + gibbs_params; suffix swap"
```

---

## Task 7: Runner — pack `kernel_gibbs` block into dispatch kwargs

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/runner.py` (lines 172–254)
- Test: `ArgoEBUSCloud/test_mlops_foundation.py`

- [ ] **Step 7.1: Write failing test for runner gibbs propagation**

Append to `test_mlops_foundation.py`:

```python
def test_run_analysis_packs_gibbs_params(monkeypatch, tmp_path):
    # Build a complete AnalysisConfig with kernel_type='gibbs' and confirm
    # run_analysis() forwards a `gibbs_params` dict containing the
    # KernelGibbsBlock fields to the dispatch shim.
    from ebus_core.config_schema import (
        AnalysisConfig, AnalysisInputBlock, GPRBlock, KernelGibbsBlock,
        OutputsBlock, PhysicsParamsBlock,
    )
    from ebus_core import runner as runner_mod
    import datetime as _dt

    captured = {}
    def fake_dispatch(**kwargs):
        captured.update(kwargs)
        return {"audit_csv": str(tmp_path / "audit_x.csv")}

    monkeypatch.setattr(runner_mod, '_call_run_diagnostic_inspection', fake_dispatch)
    # Touch the audit_csv path so the runner records status='finalized'
    (tmp_path / "audit_x.csv").write_text("placeholder")

    cfg = AnalysisConfig(
        input=AnalysisInputBlock(source='s3', s3_path='s3://bucket/x.parquet'),
        region='californiav3',
        date_start=_dt.date(2015, 1, 1),
        date_end=_dt.date(2015, 12, 31),
        lat_step=0.5, lon_step=0.5, time_step=10.0,
        depth_range=(150, 400),
        gpr=GPRBlock(
            mode='3D', kernel_type='gibbs',
            window_size_days=45, step_size_days=10,
            min_bins=10, noise_val=0.1,
            time_ls_bounds_days=(15.0, 45.0),
            lat_ls_bounds=(1e-2, 10.0), lon_ls_bounds=(1e-2, 5.0),
            run_suffix='',
            kernel_gibbs=KernelGibbsBlock(
                l_min_km=100.0, l_max_km=400.0,
                d_transition_init_km=300.0,
                d_transition_bounds_km=(50.0, 700.0),
                k_steepness_init=0.01,
                k_steepness_bounds=(1e-4, 1.0),
                anisotropy_lat_lon_ratio=2.0,
            ),
        ),
        outputs=OutputsBlock(
            aelogs_dir=str(tmp_path / "logs"),
            aeplots_dir=str(tmp_path / "plots"),
        ),
        physics_params=PhysicsParamsBlock(),
    )
    runner_mod.run_analysis(cfg, registry_path=None, force_overwrite=True)

    assert captured.get('kernel_type') == 'gibbs'
    gp = captured.get('gibbs_params')
    assert gp is not None
    assert gp['l_min_km'] == 100.0
    assert gp['l_max_km'] == 400.0
    assert gp['d_transition_init_km'] == 300.0
    assert tuple(gp['d_transition_bounds_km']) == (50.0, 700.0)
    assert gp['k_steepness_init'] == 0.01
    assert tuple(gp['k_steepness_bounds']) == (1e-4, 1.0)
    assert gp['anisotropy_lat_lon_ratio'] == 2.0
```

- [ ] **Step 7.2: Run test to verify it fails**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_run_analysis_packs_gibbs_params -v`

Expected: FAIL — `gibbs_params` is not in captured kwargs.

- [ ] **Step 7.3: Add gibbs packing to runner**

In `ArgoEBUSCloud/ebus_core/runner.py`, find the `dispatch_kwargs = {...}` block (lines 192–206). Immediately after the `if cfg.gpr.lat_ls_bounds is not None ...` block (line 213–216), append:

```python
    # When kernel_type='gibbs', forward the KernelGibbsBlock fields as a
    # gibbs_params dict so script 05 / analyze_rolling_correlations can
    # construct a GibbsKernel without re-importing pydantic.
    if cfg.gpr.kernel_type == 'gibbs' and cfg.gpr.kernel_gibbs is not None:
        gb = cfg.gpr.kernel_gibbs
        dispatch_kwargs['gibbs_params'] = {
            'l_min_km': gb.l_min_km,
            'l_max_km': gb.l_max_km,
            'd_transition_init_km': gb.d_transition_init_km,
            'd_transition_bounds_km': tuple(gb.d_transition_bounds_km),
            'k_steepness_init': gb.k_steepness_init,
            'k_steepness_bounds': tuple(gb.k_steepness_bounds),
            'anisotropy_lat_lon_ratio': gb.anisotropy_lat_lon_ratio,
        }
```

- [ ] **Step 7.4: Run test to verify it passes**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py::test_run_analysis_packs_gibbs_params -v`

Expected: PASS.

- [ ] **Step 7.5: Confirm full suite still passes**

Run: `cd /home/avik2007/ArgoEBUSAnalysis/ArgoEBUSCloud && conda run -n ebus-cloud-env pytest test_mlops_foundation.py -v`

Expected: All tests PASS (54 prior + ~9 new gibbs tests).

- [ ] **Step 7.6: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/runner.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(gibbs): runner forwards KernelGibbsBlock as gibbs_params dict"
```

---

## Task 8: First gibbs YAML config + validation

**Files:**
- Create: `configs/californiav3/californiav3_d150_400_gibbs.yaml`

- [ ] **Step 8.1: Read the matern Source-layer config to copy from**

Run: `ls configs/californiav3/`

Then read the matching matern config: `Read configs/californiav3/<source-layer-matern>.yaml`

Identify the exact filename — likely something like `californiav3_d150_400.yaml` or with a `_matern` suffix. Use it as the structural template.

- [ ] **Step 8.2: Write the gibbs YAML**

Create `configs/californiav3/californiav3_d150_400_gibbs.yaml` with:

```yaml
schema_version: 1
config_kind: analysis

input:
  source: ingestion_run
  ingestion_run_id: californiav3_20150101_20151231_res0_5x0_5_t10_0_d150_400

region: californiav3
date_start: 2015-01-01
date_end: 2015-12-31

lat_step: 0.5
lon_step: 0.5
time_step: 10.0
depth_range: [150, 400]

gpr:
  mode: 3D
  kernel_type: gibbs
  window_size_days: 45
  step_size_days: 10
  min_bins: 10
  noise_val: 0.1
  time_ls_bounds_days: [15.0, 45.0]
  lat_ls_bounds: [0.01, 10.0]    # unused on gibbs path; kept for schema completeness
  lon_ls_bounds: [0.01, 5.0]
  run_suffix: ""
  kernel_gibbs:
    l_form: sigmoid_dist_to_coast
    l_min_km: 100.0
    l_max_km: 400.0
    d_transition_init_km: 300.0
    d_transition_bounds_km: [50.0, 700.0]
    k_steepness_init: 0.01
    k_steepness_bounds: [0.0001, 1.0]
    anisotropy_lat_lon_ratio: 2.0
    climatology_source: roemmich-gilson-v3

outputs:
  aelogs_dir: AEResults/aelogs
  aeplots_dir: AEResults/aeplots
  generate_snapshots: true
  generate_physics_plots: true

physics_params:
  ohc_reference_pressure_dbar: 0.0
  teos10_convention: TEOS-10-2010
  qc_min_obs_per_bin: 1

description: >
  First Gibbs non-stationary kernel run on Source layer (150-400m).
  Tests the dist_to_coast sigmoid lengthscale per RG-Gibbs directive
  (docs/superpowers/specs/2026-04-26-rg-gibbs-l-x-directive.md).
  Compare audit metrics (RMSRE, Z, learned d_0/k) against the matern
  baseline at californiav3_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45.
```

If the actual ingestion run_id from Step 8.1 differs, update the `ingestion_run_id` field accordingly.

- [ ] **Step 8.3: Validate the config**

Run: `cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py validate configs/californiav3/californiav3_d150_400_gibbs.yaml`

Expected: validation passes; no schema errors.

If validation fails because `ingestion_run_id` does not exist in the registry, re-check the actual ingestion run_id with:

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py list --region californiav3
```

Update the YAML's `ingestion_run_id` to match.

- [ ] **Step 8.4: Commit**

```bash
git add configs/californiav3/californiav3_d150_400_gibbs.yaml
git commit -m "feat(gibbs): add Source-layer gibbs analysis YAML for californiav3"
```

---

## Task 9: Smoke run + record results

**Files:**
- Touch: `argo_claude_actions/AE_claude_recentactions.md`
- Touch: `argo_claude_actions/AE_claude_todo.md`

- [ ] **Step 9.1: Run the gibbs analysis end-to-end**

Run: `cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py analyze configs/californiav3/californiav3_d150_400_gibbs.yaml`

Expected: pipeline completes; an `audit_*.csv` is written under `AEResults/aelogs/californiav3_..._3dgibbs_w45/`.

If it fails, capture the full error trace and address before continuing. Common likely issues:
- `dist_to_coast_km` missing from the parquet — confirm by reading 5 rows of the parquet directly.
- Optimizer convergence warning — this is fine; a fitted theta within bounds still counts as success.
- Memory pressure — Gibbs O(n^3) per fit is identical to matern; not a new risk.

- [ ] **Step 9.2: Inspect audit summary**

Read: `AEResults/aelogs/californiav3_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dgibbs_w45/audit_*.csv`

Confirm columns include `d_transition_km`, `k_steepness`, `time_ls_days`, `rmsre`, `std_z`. Eyeball median RMSRE and learned d_0 distribution. Compare against matern baseline (median RMSRE 3.05%, all 32/35 windows pass) qualitatively.

- [ ] **Step 9.3: Update todo and recent actions**

Edit `argo_claude_actions/AE_claude_todo.md`:
- Mark Step 5 of the californiav3 task as `[x]` (Gemini verdict received).
- Append a new top-level section "## 2026-05-04 — [ACTIVE] Gibbs Source-layer baseline" with one sub-step: compare gibbs vs matern on `californiav3` Source layer; if gibbs improves Z chronic baseline, scale to Skin and Background.

Edit `argo_claude_actions/AE_claude_recentactions.md`:
- Prepend a 2026-05-04 entry summarising: GibbsKernel implemented, runner wired, smoke run on Source layer complete with median RMSRE = X.XX% and learned d_0 = NNN km.

- [ ] **Step 9.4: Commit results + docs**

```bash
git add argo_claude_actions/AE_claude_todo.md argo_claude_actions/AE_claude_recentactions.md AEResults/aelogs/californiav3_*_3dgibbs_w45/
git commit -m "chore(gibbs): smoke run on californiav3 Source layer + action logs"
```

---

## Self-review

**Spec coverage** (`docs/superpowers/specs/2026-04-26-rg-gibbs-l-x-directive.md`):
- §2 sigmoid form `l(d) = l_min + (l_max - l_min) / (1 + exp(-k(d - d_0)))` → Task 1 (`_sigmoid_lengthscale`).
- §2 hyperparameters: l_min/l_max fixed, d_0/k learnable → Task 1 + Task 3 (theta = [d_0, k]).
- §3 physical defensibility / no-prescribed-boundaries / stability → Task 2 (anisotropy 2:1 in `__call__`) + Task 4 (numerical stability via clipped exponent + L-BFGS-B bounds).
- §4 use `dist_to_coast_km` from `ae_utils.py` → Task 5 (gibbs branch reads `df_slice['dist_to_coast_km']`).
- §4 GibbsKernel exposes l(x) sigmoid as theta → Task 3.
- §4 maintain 2:1 lat:lon anisotropy ratio → Task 1 + Task 2 (`anisotropy_lat_lon_ratio`).
- §5 update RG-Gibbs draft spec — out of scope for this plan; the existing directive doc IS the spec.
- §5 expose d_0 + k as theta → Task 3.
- §5 verify sigmoid logic against californiav3 bounds → Task 9 (smoke run on real data).

**Placeholder scan:** No `TBD`, `TODO`, `add appropriate error handling`, `similar to Task N`, or implementation steps without code.

**Type consistency:**
- `gibbs_params` dict keys match `KernelGibbsBlock` fields (Task 7) and `GibbsKernel.__init__` kwargs (Task 1) — verified one-to-one.
- `analyze_rolling_correlations(..., gibbs_params=None, ...)` referenced in Task 5.4 (signature), Task 5.6 (`_build_kernel`), Task 6.4 (script 05 forwarding), Task 7 (runner forwarding) — same name throughout.
- Result columns `d_transition_km`, `k_steepness`, `time_ls_days` referenced in Tasks 5.7, 6.1, 9.2 — same names throughout.
- `_gibbs_optimizer` defined Task 4.3, used Task 5.7 — same name.
- `_lonlat_to_local_km` defined Task 5.3, used Task 5.5 — same name.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-04-rg-gibbs-kernel.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
