# Gemini Recent Actions — ArgoEBUSAnalysis

---

## 2026-04-01 — Data-Driven Domain Strategy: The "Long-Term Census"

**Action:** Diagnosed the "Data Desert" at depth in `californiav2` and pivoted to an empirical boundary optimization strategy for `californiav3`.

### 1. Diagnosis of californiav2 Source Layer Failure
*   **Finding:** Verified that `californiav2` (130W–115W) is severely under-sampled at 150-400m depth, with only **39 unique floats** (down from 97 in the original domain) and a median of **23 bins per 10-day window**.
*   **Verdict:** The 3D GP model is underdetermined in this tight domain, causing the extreme anisotropy ratios (up to 35.75) and RMSRE regressions (8.13% median).

### 2. Implementation Plan: Long-Term Float Census (1999–2025)
*   **Action:** Drafted `argo_gemini_actions/AE_plan_longterm_float_census.md` for Claude to implement.
*   **Strategy:** Map float availability in 5°x5° bins over a 26-year period to identify stable data "Hotspots" (e.g., Southern California Bight).
*   **Goal:** Use the resulting "Small Multiples" heatmap to define `californiav3` based on where the sensors actually are, ensuring the "Stealth Warming" study has sufficient statistical power.

### 3. Repository Hygiene
*   **Updated:** `argo_gemini_actions/AE_gemini_todo.md` to reflect the priority shift toward the Census and `californiav3`.
*   **Created:** `ArgoEBUSCloud/08_ae_diagnose_density.py` (diagnostic script used to confirm the data desert).

---
