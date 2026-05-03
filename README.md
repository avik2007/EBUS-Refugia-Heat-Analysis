EBUS-Refugia-Heat-Analysis
The "Vertical Audit" of the California Current System
Research Question: Is the California Current System (CCS) acting as a true thermal refugium, or is the "upwelling mask" hiding significant subsurface warming in the source waters?

While most climate resilience studies rely on Satellite SST (Skin Temperature), this project conducts a Vertical Audit using the Argo float array. By strictly separating the water column into physically distinct thermodynamic layers (Response, Source, and Background), we test for "Stealth Warming"—a decoupling event where the upwelling source waters warm independently of the surface signal.

Key Methodology:

Sparsity Quantification: Rigorous evaluation of the "Void Ratio" in observational arrays.

Probabilistic Modeling: Application of advanced spatial interpolation techniques to generate continuous subsurface heat fields with explicit uncertainty quantification.

Trend Decoupling: Statistical comparison of surface vs. subsurface warming rates to detect "Refugia Failure."

Climate Attribution: Isolation of ENSO-driven variability from secular warming trends.

This repository contains the data pipeline, statistical modeling framework, and validation suite for quantifying the resilience of Eastern Boundary Upwelling Systems.

---

## MLOps Tooling

This project includes a config-driven MLOps layer that wraps the existing pipeline for reproducible, collision-safe runs.

- **YAML configs** in `configs/<region>/` define every run parameter. Each config is schema-validated by Pydantic before anything executes.
- **`aebus` CLI** dispatches validate / analyze / ingest / list / show subcommands.
- **Manifests** capture config hash, git SHA, conda env, and S3 lineage alongside every run.
- **Collision detection** blocks duplicate runs (same config hash, different result) and skips identical re-runs gracefully.

### aebus quickstart

```bash
# Validate a config and print its canonical run_id
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py validate \
    configs/california/california_20150101_20151231_res0_5x0_5_t30_0_d0_100_3dmatern_w45.yaml

# List all registered runs for a region
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py list --region california

# Show the manifest for a specific run
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py show \
    california_20150101_20151231_res0_5x0_5_t30_0_d0_100_3dmatern_w45

# Run analysis end-to-end (writes audit CSV + manifest)
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py analyze \
    configs/california/california_20150101_20151231_res0_5x0_5_t30_0_d0_100_3dmatern_w45.yaml
```

The underlying scripts (`02_ae_cloud_run.py`, `05_ae_update_tomatern0.5.py`, etc.) remain usable directly as an escape hatch for one-off exploration.

**Backfill:** `10_ae_backfill_configs.py` auto-generates YAML configs for all 18 historical runs by parsing canonical `run_id` strings and reading per-window noise values from audit CSVs. Enables retroactive reproducibility without requiring the original operator notes.

**Test suite:** 44 integration tests cover schema validation, collision detection, manifest round-trips, registry queries, and CLI dispatch (`ArgoEBUSCloud/test_mlops_foundation.py`).

---

## Data Coverage

Before defining new experiment domains, we audit where Argo floats actually are.

`09_ae_longterm_float_census.py` — fetches per-dive positions from ERDDAP in 5-year chunks, bins on a 5°×5° grid, and counts **unique floats** per (year, cell) from 1999–2024. Outputs one heatmap PNG per year on a fixed color scale for valid cross-year comparison, plus a full census CSV.

`09b_ae_analyze_float_census.py` — reads the census CSV and surfaces domain-recommendation statistics: annual totals, top-10 most persistent cells, and cells with float presence in ≥20 of 26 years — the empirical basis for defining future experiment domains (e.g., `californiav3`).

```bash
conda run -n ebus-cloud-env python ArgoEBUSCloud/09_ae_longterm_float_census.py
conda run -n ebus-cloud-env python ArgoEBUSCloud/09b_ae_analyze_float_census.py
```

---

## Example Outputs

**Argo Float Tracks — California Current System (2015)**
![Argo float trajectories showing spatial coverage across the CCS](docs/images/float_tracks.png)

**Kriged Ocean Heat Content Map (Skin Layer, August 2015)**
![GPR-interpolated OHC field for the 0–100m layer](docs/images/ohc_kriged.png)

**Cross-Validation: RMSRE Overlay**
![Leave-one-out CV error map showing model skill vs. holdout observations](docs/images/rmsre_cv_overlay.png)

**Float Census — Annual Totals (1999–2025)**
![Annual count of Argo floats in the CCS bounding box](docs/images/float_census_annual.png)
