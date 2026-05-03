# Gemini Lessons Learned — ArgoEBUSAnalysis

## 2026-04-11: FX2 High-Res Temporal Run (t10_0) Diagnosis

### 1. Source Layer (150-400m) Regression
- **Finding:** The tight `californiav2` domain (130W–115W) is insufficient for the Source Layer.
- **Root Cause:** Domain clipping of float trajectories at depth. Narrow features like the California Undercurrent (CUC) require either higher float density or wider longitudinal bounds to stabilize the GPR kernel.
- **Action:** Define `californiav3` based on the 26-year float census to identify a high-density "Golden Age" sub-region.

### 2. Temporal Scale (scale_time_bin) Saturation
- **Finding:** `scale_time_bin` consistently hits the 45-day upper bound in Skin and Source layers.
- **Conclusion:** Ocean memory at these depths likely exceeds the 45-day window width.
- **Action:** Widen `time_ls_bounds_days` to (15.0, 60.0) or (15.0, 90.0) for deeper layers to allow the optimizer to find the true physical correlation time.

### 3. Background Layer (500-1000m) Stationarity Violation (Sep 2015)
- **Finding:** Z-score spike of 18.73 at window 6102.5.
- **Conclusion:** Confirmed as a stationarity violation caused by the **Pacific Blob** onset. The GP accurately fits the temperature anomaly (low RMSRE) but drastically underestimates the uncertainty during this extreme, non-stationary event.
- **Action:** Flag this window as a physical outlier/event. No change to kernel required as Z-score returned to ~1.0 in subsequent windows.

### 4. Coastal Distance Feature
- **Action:** Implemented `calculate_dist_to_coast` using Cartopy and KDTree. Integrated into the preprocessing pipeline (`01_ae_cloud_ingestion.py`, `02_ae_cloud_run.py`, and `ebus_core/argoebus_thermodynamics.py`).
- **Goal:** Provide a physical coordinate for distance from coast to improve future ML/XGBoost modeling of coastal upwelling.
