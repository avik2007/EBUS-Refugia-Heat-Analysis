# 5-Minute Mini-Talk Plan
**Date:** 2026-05-19
**Context:** 5-minute slot in a 15-minute research presentation. Audience: ML-literate + mixed academic.
**Goal:** Introduce problem, sketch technical approach, state goals.

---

## Narrative Arc

Hook (stealth warming question) → Data + method → Key result → What's next

---

## Slide Plan (4 slides)

### Slide 1 — Data / Domain
**Asset:** `AEResults/aeplots/float_census_depth_aware/float_census_alldepths_mean.png`
**Caption:** ~58k Argo dives, 25 years, California Current System. Red dashed box = analysis domain (californiav3).
**Script:** "We're using Argo profiling floats — autonomous robots that drift at depth and resurface every 10 days. 25 years of data, California coast. The question: is the ocean warming in ways that haven't reached the surface yet?"

### Slide 2 — What GPR produces
**Asset:** A kriging snapshot from `AEResults/aeplots/snapshot_californiav3_.../`
**Recommended:** `..._d150_400.../..._day6032.png` (July 2015, Source layer)
**Note:** Current size is small/busy. Consider regenerating at larger figsize, or crop to prediction panel only.
**Script:** "We bin the float data onto a 0.5° grid and fit a 3D Gaussian Process (lat × lon × time). The GP learns spatial correlation structure — it gives us a continuous OHC field plus calibrated uncertainty."

### Slide 3 — THE result: anisotropy vertical contrast
**Asset:** Two anisotropy plots side-by-side (or stacked):
- `AEResults/aelogs/californiav3_..._d0_100_.../anisotropy_...png` (Skin, 0–100m) — stays mostly below 1.0 (zonal)
- `AEResults/aelogs/californiav3_..._d150_400_.../anisotropy_...png` (Source, 150–400m) — persistently above 1.0 in Jan–Apr and Jun–Aug, peaks ~2.4

**Note:** These are separate files. Either make a combined 2-panel figure, or use them sequentially.
**Caption:** Anisotropy ratio = lat_lengthscale / lon_lengthscale. > 1.0 = meridional (current) dominance.
**Script:** "The GP optimizes separate lat and lon length scales. Ratio > 1 means the ocean 'looks' further north-south than east-west — signature of a meridional current. At the surface: stays zonal all year. At 150–400m: flips to meridional from winter through summer. That's the California Undercurrent."

### Slide 4 — Model diagnostic + what's next
**Asset:** `AEResults/aelogs/californiav3_..._d150_400_.../zscore_std_...png` (Source layer Z-score)
**Caption:** Z-score std dev = 1.0 means model is well-calibrated. Two spikes at day 6065–6075 = Pacific Blob 2015 onset.
**Script:** "This is a calibration check. Mostly well-behaved. But the Blob events break our stationary assumption near the shelf break. Next step: Gibbs kernel where lengthscale varies with distance to coast — lets the model know to use shorter scales nearshore."

---

## Assets To-Make Before the Talk

1. **Combined 2-panel anisotropy figure** (Skin vs Source, same time axis) — makes the vertical contrast story immediate. Easy to generate from existing audit CSVs or just use matplotlib subplot on the two existing PNGs.
2. **Larger/cleaner kriging snapshot** — regenerate at `figsize=(12, 5)` or crop prediction panel only. Optional if slide 2 gets cut for time.
3. **Conceptual depth-layer schematic** — no file exists. Sketch showing Skin/Source/Background stack + California Undercurrent arrow. Even a hand-drawn figure works.

---

## Numbers Worth Quoting

| Layer | Median RMSRE | Pass Rate | Key pattern |
|---|---|---|---|
| Skin (0–100m) | 4.25% | 27/35 | Zonal throughout |
| Source (150–400m) | 3.05% | 32/35 | Meridional Jan–Apr, Jun–Aug |
| Background (500–1000m) | 2.50% | 35/35 | Zonal; Blob spikes |

Target RMSRE < 5%. Matérn baseline meets it for all three layers on californiav3.

---

## What's Still Missing From the Full Study

- Gibbs kernel (non-stationary, dist_to_coast l(x)) — implementation plan ready, not yet run
- Multi-year analysis (only 2015 so far)
- Vertical delta comparison (Source vs Background OHC trend over years)
