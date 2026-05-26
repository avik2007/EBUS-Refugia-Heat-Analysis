# MLOps Foundation — Gemini Audit & Gap Analysis

**Date:** 2026-04-30
**Status:** AUDIT COMPLETED
**Author:** Gemini (Interactive CLI Agent)

---

## 1. Overview
This report evaluates the Phase 1–5 implementation of the MLOps Foundation (branch `feat/mlops-phase2`). While the structural engineering is robust, five technical gaps have been identified that impact scientific precision and operational reliability.

---

## 2. Identified Gaps & Required Fixes

### 2.1 Kwarg Mismatch in Analysis Shim (`runner.py`)
*   **Issue:** `AnalysisConfig` uses split-anisotropy fields (`lat_ls_bounds`, `lon_ls_bounds`), but the underlying `run_diagnostic_inspection` in scripts `05` and `07` still expects a single `spatial_ls_upper_bound`.
*   **Impact:** YAML spatial bounds are currently ignored by the physics engine.
*   **Fix:** Update `runner.py` to pass `max(cfg.gpr.lat_ls_bounds[1], cfg.gpr.lon_ls_bounds[1])` to the physics script as a temporary bridge.

### 2.2 Potential "Ghost" Successes in Registry
*   **Issue:** The registry appends a run line upon execution, but does not verify if the science script completed successfully or if `manifest.json` was actually written (e.g., in case of a Dask worker crash).
*   **Impact:** `aebus list` may show successful runs that are actually "zombie" directories with missing data.
*   **Fix:** Implement a "Finalized" flag in the registry that is only set after `manifest.json` is confirmed on disk.

### 2.3 Incomplete Depth Validation
*   **Issue:** `IngestionConfig` validates `depth_range[0] < depth_range[1]`, but this check is not consistently applied to `AnalysisConfig` or the `PhysicsParamsBlock`.
*   **Impact:** Risk of non-physical OHC integration (e.g., negative depth layers).
*   **Fix:** Consolidate `_depth_range_ordered` into a shared validator in `config_schema.py`.

### 2.4 S3 Path Consistency
*   **Issue:** Path derivation logic is duplicated between `runner.py` and legacy science scripts.
*   **Impact:** Small changes in decimal formatting (`_fmt_dec`) could break the link between MLOps-managed analysis and legacy-managed parquets.
*   **Fix:** Centralize `derive_run_id` and `_fmt_dec` in `ebus_core/ae_utils.py`.

### 2.5 Backfill Forensic Accuracy
*   **Issue:** The backfill tool assumes script defaults for missing parameters.
*   **Impact:** Historical runs with non-standard noise or bounds are recorded incorrectly.
*   **Fix:** Add a `metadata` section to backfilled YAMLs explicitly marking which fields were "recovered" vs. "assumed."

---

## 3. Recommended Next Steps
Priority should be given to **Fix 2.1 (Shim Mismatch)** before any further science runs are executed on the `californiav3` domain.
