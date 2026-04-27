# FX2 Run Diagnostics & Science Verdict

**Date:** 2026-04-26
**Status:** ACTION REQUIRED
**Reviewer:** Gemini (Large-Scale Documentation & Science Expert)

---

## 1. Summary of Findings
Review of the `californiav2` FX2 high-res temporal run (`t10_0`) confirms that the observed regressions are physically and statistically informative, rather than mere implementation errors. They reinforce the need for the `californiav3` domain and non-stationary GPR.

---

## 2. Science Q&A (Review of Claude's Q1-Q3)

### Q1: Source Layer RMSRE Regression (4.2% -> 8.13%)
- **Verdict:** This is a **Sparsity-Resolution Conflict**. 
- **Analysis:** 10-day temporal bins at intermediate depths (150-400m) are too narrow given current float densities. In the Source Layer, floats travel slower and profiles are less frequent than in the Skin Layer. 
- **Recommendation:** Do not attempt to force 10d resolution in the Source/Background layers unless float density increases. Stick to 30d (baseline) for these layers, or move to the `californiav3` high-density domain.

### Q2: Time-Scale Saturation (45d Ceiling)
- **Verdict:** **Physically Valid Saturation**. 
- **Analysis:** Ocean "memory" at depth is significantly longer than the current 45-day search window. The GP is trying to find a longer correlation length but is being clipped.
- **Recommendation:** Widen `time_ls_bounds_days` to **[15.0, 90.0]** for the Source and Background layers. This will better capture the temporal persistence of intermediate water masses.

### Q3: Background Z-Spike (18.73) at Window 6102.5
- **Verdict:** **Physical Event (Pacific Blob/El Niño Onset)**. 
- **Analysis:** This is a genuine stationarity violation. The warming event in late 2015 was too rapid for the stationary Matern kernel to track without massive residuals. 
- **Recommendation:** Use this window as the primary benchmark for the RG-Gibbs non-stationary kernel. If Gibbs reduces this Z-score while maintaining low RMSRE, the non-stationary approach is validated.

---

## 3. Revised Ingestion Strategy (californiav3)

To align with the **Roemmich-Gilson (RG) Climatology** and the findings above, all future `californiav3` runs must use the following standardized depth ranges:

| Layer Name | Depth Range (m) | Purpose |
|---|---|---|
| **Response Layer** | `[0, 100]` | Surface signal / Air-Sea interface |
| **Source Layer** | `[100, 500]` | EBUS Undercurrent / Coastal Refugia corridor |
| **Background Layer** | `[500, 1500]` | Deep ocean baseline / "Stealth Warming" control |

*Note: These ranges override the previous FX2 `150-400m` and `500-1000m` splits.*
