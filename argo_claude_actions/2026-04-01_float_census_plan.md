# Implementation Plan: Long-Term Argo Float Census (Script 09)

**Date:** 2026-04-01
**Prompted by:** Source Layer GPR regression in californiav2 t10_0 run. Gemini plan: `argo_gemini_actions/AE_plan_longterm_float_census.md`
**Status:** Approved — next implementation priority.

---

## Context

The `californiav2` domain (lat [30,45], lon [-130,-115]) was adopted to align with the
Frontiers 2024 paper, but the first GPR run on the FX2 parquets shows the Source Layer
(150–400m) has severely degraded: median RMSRE 8.13% (was 4.2%), anisotropy ratios up
to 35.75 (non-physical). The likely cause is that the tighter domain clips float
trajectories at depth, leaving geometrically degenerate observations in some windows.

Before defining `californiav3`, we need empirical evidence of where floats actually are
across the full 26-year Argo record. This census will reveal the stable data hotspots
that can anchor the new domain bounds.

---

## New File

**`ArgoEBUSCloud/09_ae_longterm_float_census.py`**

No changes to any existing files.

---

## Implementation Steps

### Step 1 — Imports and config
Use `get_ebus_registry()["california"]` for the broad census bounds: lat [25,50],
lon [-140,-110]. Intentionally wider than californiav2.

### Step 2 — Fetch in 5-year chunks via `get_float_history()`
Reuse `get_float_history(region, start_date, end_date)` from `ebus_core/ae_utils.py`.
This function already handles: ERDDAP URL, `&distinct()` deduplication, timeout,
column normalization. Do NOT re-implement the query.

Chunks:
- 1999-01-01 → 2003-12-31
- 2004-01-01 → 2008-12-31
- 2009-01-01 → 2013-12-31
- 2014-01-01 → 2018-12-31
- 2019-01-01 → 2023-12-31
- 2024-01-01 → 2025-12-31

Wrap each chunk in try/except so a single ERDDAP timeout doesn't abort the whole run.
Add `year` column: `df['year'] = df['time'].dt.year`

### Step 3 — Bin on 5°×5° grid, count unique floats
```python
df['lat_bin'] = (np.floor(df['lat'] / 5) * 5) + 2.5
df['lon_bin'] = (np.floor(df['lon'] / 5) * 5) + 2.5
census = (df.groupby(['year', 'lat_bin', 'lon_bin'])['platform_number']
            .nunique().reset_index()
            .rename(columns={'platform_number': 'n_floats'}))
```
Unique floats (not dive count) is the correct metric: one float with 30 dives in a
cell still represents one independent sensor for the GP.

### Step 4 — Small multiples figure (26 panels, 1999–2024)
- Layout: 6 rows × 5 cols (30 slots; last 4 blank)
- Per panel: Cartopy coastlines + `pcolormesh` heatmap, fixed scale `vmin=0, vmax=15`
- Shared colorbar labeled "Unique Floats"
- Figure size: 20×26 inches
- Output: `AEResults/aeplots/float_census_california_1999_2025.png` (dpi=150)
- Use `ensure_ae_dirs()` and `get_project_paths()["plots"]` for path construction

### Step 5 — Top-10 hotspot table + CSV
- Print top-10 (year, lat_bin, lon_bin) by n_floats to stdout
- Save full census to `AEResults/aelogs/float_census_california_1999_2025.csv`

---

## Functions to Reuse (from `ebus_core/ae_utils.py`)

| Function | Purpose |
|----------|---------|
| `get_float_history(region, start_date, end_date)` | ERDDAP query → per-dive DataFrame |
| `get_ebus_registry()` | Provides broad `california` bounds |
| `ensure_ae_dirs()` | Creates `AEResults/` subdirs |
| `get_project_paths()` | Returns `AEResults/aeplots/` path |

---

## Verification

```bash
conda run -n ebus-cloud-env python 09_ae_longterm_float_census.py
```

Expected: 6 progress lines (one per chunk), top-10 table, two output files.

Visual check: sparse pre-2005, growing 2005–2010, full network from ~2015. Southern
California Bight (30–35°N) should be a persistent hotspot every year. This gives
Gemini the empirical basis to define `californiav3`.
