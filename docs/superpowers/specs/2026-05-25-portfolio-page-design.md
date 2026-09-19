# Portfolio Page Design — ArgoEBUSAnalysis

**Date:** 2026-05-25  
**URL target:** `https://avik2007.github.io/ArgoEBUSAnalysis`  
**Reference template:** `https://avik2007.github.io/mhw-risk-profiler/`

---

## Objective

A single static HTML portfolio page that describes the Stealth Warming / Ocean Refugia hypothesis,
shows key visual results (float tracks + kriged OHC map), and signals the next research direction
(Gibbs kernel). Same visual design system as mhw-risk-profiler.

---

## Technical Stack

- **Bootstrap 5.3.2** (CDN) — layout and responsive grid
- **Inter** (system-ui fallback) — typography
- **No JavaScript / no Plotly** — static images only (Option A)
- **GitHub Pages** — served from `docs/` on `main` branch

---

## Color System (identical to mhw-risk-profiler)

```css
--bg:      #f0f6ff
--surface: #ffffff
--border:  #dde4f0
--text:    #1a2540
--muted:   #5a6e8c
--accent:  #0d84d6
--warn:    #d97706   (v1 badge)
--warnbg:  #fffbeb
--warnbdr: #fcd34d
```

---

## File Layout

```
docs/
  index.html          ← new (portfolio page)
  images/
    float_tracks.png       ← existing
    ohc_kriged.png         ← existing
    rmsre_cv_overlay.png   ← existing
    float_census_annual.png← existing
```

GitHub Pages must be enabled via:
```
gh api repos/avik2007/ArgoEBUSAnalysis/pages -X POST \
  -f source[branch]=main -f source[path]=/docs
```

---

## Page Sections

### 1. Navbar (sticky)
- Brand: `🌊 Argo-EBUS Analysis`
- Right: `GitHub ↗` link to `https://github.com/avik2007/ArgoEBUSAnalysis`

### 2. Hero
- Section label: `California Current System · Subsurface Warming Research`
- H1: `Stealth Warming in the California Current System`
- Intro paragraph: the upwelling mask problem — how satellite SST misses subsurface source-water
  warming because cool upwelled water masks the signal at the surface. Argo floats + 3D GPR
  (Kriging) allow a vertical audit across 3 thermodynamically distinct depth layers.
- Stat chips: `1999–2025 Float Record`, `3 Depth Layers`, `0.5°×0.5° Grid`, `California CCS`, `2015 Baseline`
- Badge: `BASELINE RESULTS` (warn/amber style)

### 3. How It Works (Pipeline)
5-node horizontal `pipeline-flow`, same markup as mhw-risk-profiler:

| Node | Icon | Name | Sub-text |
|------|------|------|----------|
| 1 | 🌊 | Argo ERDDAP | Float profiles fetched from ERDDAP in 5-year chunks |
| 2 | 🔪 | Layer Filter | Profiles binned into Skin (0–100m), Source (150–400m), Background (500–1000m) |
| 3 | 🧮 | 3D GPR | Matérn-½ × Matérn-½ kernel; lat/lon/time; rolling 10-day windows |
| 4 | 📊 | Kriged Heat Map | Continuous OHC field with explicit uncertainty (posterior std) |
| 5 | 📐 | Anisotropy Analysis | Lat_Scale / Lon_Scale ratio per window — current signature |

Below the pipeline, 3 definition cards:
- **What is the upwelling mask?** — Ekman upwelling brings cold deep water to surface, hiding
  subsurface warming from satellite SST sensors.
- **Why 3 layers?** — Skin tracks atmospheric forcing; Source layer is where California
  Undercurrent delivers subsurface heat; Background is the deep baseline.
- **What is Kriging?** — Gaussian Process Regression over space + time, giving a probabilistic
  heat map with uncertainty estimates from sparse Argo float profiles.

### 4. Data Coverage
- Section label: `Observation Network`
- H2: `Argo Float Coverage — California Current System`
- Sub: sparse by design; void ratio quantified before modeling
- Full-width `arch-img-wrap`: `images/float_tracks.png`
- Alt text: `Argo float trajectories across the California Current System, 2015`
- Below: 2-col arch-bullet explanations:
  - **Sparse by necessity** — Argo floats are globally distributed; the CCS bounding box
    captures ~25k dives over the study period, clustered near coasts and fronts.
  - **Void ratio** — Fraction of 0.5°×0.5° grid cells with no float observations in a
    rolling window; used to mask unreliable kriged estimates.
  - **Domain bounds** — lat 30–50°N, lon 215–245°E; defined by persistent float presence
    in ≥20 of 26 years from the long-term float census.
  - **Depth-aware filtering** — each profile counted separately for each layer; a single
    float dive may contribute to Skin but not Background if it doesn't reach 500m.

### 5. Results
- Section label: `GPR Output`
- H2: `Kriged Ocean Heat Content — Skin Layer (0–100m), 2015`
- Sub: rolling 10-day window snapshots across the California Current
- Full-width `arch-img-wrap`: `images/ohc_kriged.png`
- Alt text: `GPR-interpolated OHC field for the 0–100m skin layer, California CCS 2015`
- 3 stat cards below (same `.stat-card` style):
  - `RMSRE < 5%` — leave-one-out cross-validation target
  - `Std Z-Score ≈ 1.0` — posterior uncertainty calibration target
  - `Anisotropy Ratio > 1.0` — Source layer signal (meridional dominance)
- Below stat cards: full-width `arch-img-wrap`: `images/rmsre_cv_overlay.png`
- Caption: `Leave-one-out CV error map — model skill vs held-out Argo observations`

### 6. Key Finding
- Section label: `Scientific Result`
- H2: `Vertical Anisotropy Fingerprint — Upwelling Current Signature`
- Prose: Source layer (150–400m) shows anisotropy ratio > 1.0 (meridional / along-current
  structure) matching the California Undercurrent's flow axis. Skin layer (0–100m) remains
  zonal throughout, driven by atmospheric forcing. This decoupling is the expected fingerprint
  of stealth warming transported by the CUC.

### 7. What's Next (Roadmap)
- Section label: `What's Next`
- H2: `Current State → Gibbs Kernel`
- Same 2-col roadmap cards (v1 warn / v2 green):

**v1 — Baseline (current)**
- Matérn-½ × Matérn-½ kernel with fixed spatial lengthscale
- 2015 single-year results; Source layer RMSRE 3.05%; multi-year extension pending
- Anisotropy ratio computed from fitted kernel hyperparameters
- Predictive variance elevated near the continental shelf break

**v2 — In Progress**
- **Gibbs kernel** replaces fixed Matérn spatial component: lengthscale `l(d)` is a learnable
  sigmoid function of `dist_to_coast_km`, allowing the model to shrink correlation length
  near the shelf and expand it offshore
- Reduces predictive variance in the shelf-break region where float density drops sharply
- Preserves Matérn-½ in the time dimension; spatial component changes only
- Learnable parameters: `d_0` (shelf inflection point), `k` (sigmoid steepness);
  optimized via L-BFGS-B with finite-difference gradients

### 8. Footer
```
Argo-EBUS Analysis · California Current System
Data: Argo Global Array (GDAC / ERDDAP) · TEOS-10 (GSW) · Argo float trajectories
GitHub ↗
```

---

## Success Criteria

1. `docs/index.html` exists and matches the mhw-risk-profiler visual style
2. Page is served at `https://avik2007.github.io/ArgoEBUSAnalysis` via GitHub Pages
3. `float_tracks.png` and `ohc_kriged.png` render in the page
4. Gibbs kernel section is present at the end with accurate technical description
5. Page is mobile-responsive (Bootstrap grid handles this automatically)

---

## Out of Scope

- Interactive Plotly charts (v2)
- Multi-year animation / slider
- Live data fetch from ERDDAP
