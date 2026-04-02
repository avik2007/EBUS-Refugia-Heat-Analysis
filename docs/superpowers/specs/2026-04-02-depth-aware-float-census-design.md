# Design: Depth-Aware Float Census

**Date:** 2026-04-02  
**Motivation:** The 2015 source layer (150–400m) GPR run showed severe regression (RMSRE up to 22%, pass rate 24%) under the `californiav2` config. The root cause is suspected to be the tighter lon [-130, -115] domain clipping float trajectories at depth. The existing float census (script 09) is depth-agnostic — it cannot distinguish whether a float actually profiled into the source layer. A depth-aware census will make the coverage gap visible and empirically diagnose the 2015 sparsity.

---

## Intended Outcome

- Per-year maps (1999–2025) for all-depths plus each of the 3 scientific layers
- Per-layer mean density maps (averaged across all years)
- All figures include the `californiav2` domain boundary ("CCS Analysis Bounds") so the clipping effect is visually apparent
- 4 CSVs capturing the raw census data by layer for downstream analysis

---

## Part 1: New ERDDAP helper in `ae_utils.py`

**Function:** `get_float_history_by_layer(region, pres_min, pres_max, start_date, end_date)`

- Reuses the existing ERDDAP URL construction from `get_float_history()`
- Inserts `pres%3E={pres_min}&pres%3C={pres_max}` constraints into the query before `&distinct()`
- Returns the same column schema as `get_float_history()`: `platform_number, lat, lon, time, time_days`
- "At least one measurement in range" semantics: request columns `platform_number, time, latitude, longitude` only (NOT `pres` as a return column) — pressure is used as a filter constraint only. `&distinct()` then collapses correctly to one row per unique (float, dive), not one row per pressure level.

**File:** `ArgoEBUSCloud/ebus_core/ae_utils.py`

---

## Part 2: New script `09c_ae_depth_aware_float_census.py`

**File:** `ArgoEBUSCloud/09c_ae_depth_aware_float_census.py`

### Depth layers

| Layer      | Pressure range (dbar) | Physical meaning                        |
|------------|----------------------|-----------------------------------------|
| Skin       | 0–100                | Atmosphere-forced surface layer         |
| Source     | 150–400              | Ekman upwelling source, "stealth heat"  |
| Background | 500–1000             | Deep ocean baseline                     |

### Fetching

Four fetch passes over the same 6 time chunks (1999–2025, 5-year splits):

1. **All-depths** — `get_float_history()` (existing function, no pressure filter)
2. **Skin** — `get_float_history_by_layer(..., pres_min=0, pres_max=100)`
3. **Source** — `get_float_history_by_layer(..., pres_min=150, pres_max=400)`
4. **Background** — `get_float_history_by_layer(..., pres_min=500, pres_max=1000)`

### Census building

Same 5°×5° grid binning logic as script 09: unique floats (by `platform_number`) per `(year, lat_bin, lon_bin)`.

### Output directory

`AEResults/aeplots/float_census_depth_aware/`

### CSVs saved (4 files)

- `float_census_depth_aware_alldepths_1999_2025.csv`
- `float_census_depth_aware_skin_1999_2025.csv`
- `float_census_depth_aware_source_1999_2025.csv`
- `float_census_depth_aware_background_1999_2025.csv`

### PNGs produced (~107 total)

| Type               | Count | Naming pattern                                    |
|--------------------|-------|---------------------------------------------------|
| Per-year all-depths | 26   | `float_census_alldepths_{year}.png`               |
| Per-year skin       | 26   | `float_census_skin_{year}.png`                    |
| Per-year source     | 26   | `float_census_source_{year}.png`                  |
| Per-year background | 26   | `float_census_background_{year}.png`              |
| Mean skin           | 1    | `float_census_skin_mean.png`                      |
| Mean source         | 1    | `float_census_source_mean.png`                    |
| Mean background     | 1    | `float_census_background_mean.png`                |

### Plot structure (all figures)

- Cartopy PlateCarree, extent: lat [25, 50], lon [-140, -110] (broad california domain)
- `pcolormesh` on 5°×5° grid, `YlOrRd` colormap, `vmin=0, vmax=8`
- Land + coastline overlaid (zorder 100/101)
- **CCS Analysis Bounds rectangle:** dashed red border, no fill, bounds read from `get_ebus_registry()["californiav2"]` → lat [30, 45], lon [-130, -115]; labeled "CCS Analysis Bounds" in red text at top-left corner of rectangle
- Title: `"Argo Float Density ({layer_name}) — {year}   (total unique floats: {N})"`
- Colorbar label: `"Unique Floats per 5°×5° Cell"`

---

## Reuse from existing code

- `get_float_history()` in `ae_utils.py` — call unchanged for all-depths pass
- `get_project_paths()`, `ensure_ae_dirs()`, `get_ebus_registry()` in `ae_utils.py`
- `build_census()` logic from script 09 — replicate directly (5°×5° bins, unique float count)
- `plot_year()` layout from script 09 — extend with domain overlay parameter

---

## Verification

```bash
conda run -n ebus-cloud-env python ArgoEBUSCloud/09c_ae_depth_aware_float_census.py
```

Check:
1. Script completes without ERDDAP errors
2. Output dir contains 4 CSVs and ~107 PNGs
3. `float_census_source_2015.png` shows visibly fewer floats inside the CCS Analysis Bounds rectangle than `float_census_skin_2015.png`
4. The dashed red rectangle appears on all figures at the correct location
5. Mean source map shows which cells have persistent source-layer coverage — should be sparser than skin mean map
