# MLOps Foundation — Gemini Review Results

**Date:** 2026-04-26
**Status:** REVIEW COMPLETED
**Reviewer:** Gemini (Large-Scale Documentation & Science Expert)

---

## 1. Executive Summary
The MLOps Foundation design is sound and addresses the primary scaling bottlenecks. However, three critical gaps regarding **science reproducibility**, **data lineage**, and **backfill integrity** must be addressed before Phase 1 implementation.

---

## 2. Review Findings & Gaps

### §3 Config Schema — Science Coverage
- **GAP: Quality Control (QC) Logic:** The schema lacks a `qc_policy` field. We need to explicitly record which Argo flags were accepted and whether any specific float models or project IDs were excluded.
- **GAP: Anisotropy Configuration:** For 3D runs, the YAML must include separate bounds or ratios for `lat_ls_bounds` vs `lon_ls_bounds`. The current `spatial_ls_upper_bound` is too reductive for EBUS jet dynamics.
- **RECOMMENDATION:** Add a `physics_params` block to `AnalysisConfig` to capture thermodynamic constants (e.g., reference pressure for OHC) and QC thresholds.

### §4 Manifest — Data Lineage
- **GAP: ERDDAP Identity:** `parquet_etag` is insufficient because ERDDAP datasets are frequently re-processed upstream. 
- **REQUIREMENT:** Include the **ERDDAP dataset ID** (e.g., `argo_global`) and the **`data_access_timestamp`** in the manifest. This allows us to verify if a re-run uses the same underlying data snapshot.
- **GAP: Library Precision:** Explicitly capture the **TEOS-10 version** (usually provided via the `gsw` library) and the **Scipy/Sklearn** engine versions, as these impact the numerical stability of the GP kernel.

### §6 Backfill — Historical Integrity
- **RISK:** `run_suffix` is a human-entered string and is not a reliable primary key for backfilling.
- **REQUIREMENT:** The backfill tool MUST prioritize extracting parameters from the `audit_csv` headers (if present). If a parameter cannot be found, it must be marked as `null` or `unknown` in the generated YAML rather than assuming script defaults.

### §9 RG-Gibbs Slotting
- **REQUIREMENT:** The `gpr` block must be polymorphic. A `gibbs` type will require sub-fields for the lengthscale function parameters (see Gemini's $l(x)$ discussion) and a `climatology_source` reference.

---

## 3. Approval Conditions
The spec is **Approved for Implementation** contingent on the inclusion of the ERDDAP lineage fields and the expansion of the `physics_params` block in the Pydantic models.
