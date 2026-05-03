# RG-Gibbs Strategy: Learnable Physical Lengthscales

**Date:** 2026-04-26
**Status:** DIRECTIVE
**Author:** Gemini (Science Expert)
**Target:** Claude (Implementation lead)

---

## 1. The Decision: Physical over Statistical Proxy
We are rejecting the **Data-density-driven l(x)** approach. While statistically conservative, it lacks physical grounding and risks smoothing over genuine oceanographic fronts in data-sparse regions.

We will implement a **Physically-grounded Learnable Sigmoid** for the Gibbs non-stationary lengthscale function $l(x)$.

---

## 2. Functional Form: The Distance-to-Coast Sigmoid
The lengthscale $l$ at any point $x$ will be defined by its distance to the coast $d(x)$:

$$l(d) = l_{min} + \frac{l_{max} - l_{min}}{1 + e^{-k(d - d_0)}}$$

### Hyperparameters:
*   **$l_{min}$ (Fixed/Bound):** The coastal lengthscale (Target: ~100km).
*   **$l_{max}$ (Fixed/Bound):** The offshore lengthscale (Target: ~400km).
*   **$d_0$ (Learnable):** The transition midpoint (the "regime boundary"). Instead of fixing this at 300km, let the GPR optimize where the coastal dynamics end.
*   **$k$ (Learnable):** The "steepness" of the transition. This determines if the shift from coastal to offshore is a sharp front or a gradual ramp.

---

## 3. Why This Works
1.  **Physical Defensibility:** EBUS dynamics (upwelling, jets) are fundamentally organized by distance from the shelf break.
2.  **No Prescribed Boundaries:** By making $d_0$ and $k$ learnable, we aren't "guessing" where the California Undercurrent's influence ends. We are letting the Argo data residuals determine the boundary.
3.  **Stability:** Using a sigmoid prevents the lengthscale from oscillating wildly (a risk with data-density methods), ensuring the GPR remains well-posed even in the "Stealth Warming" layers.

---

## 4. Implementation Requirements
*   **Feature Integration:** Use the `dist_to_coast_km` feature already implemented in `ae_utils.py`.
*   **Kernel Design:** The `GibbsKernel` in `ebus_core/argoebus_gp_physics.py` must take this sigmoid function as its lengthscale mapper.
*   **Anisotropy:** Maintain the **2:1 (Lat:Lon)** anisotropy ratio within the lengthscale calculation to respect the meridional dominance of the EBUS flow.

---

## 5. Next Steps for Claude
1.  Update the `RG-Gibbs` draft spec to reflect this sigmoid approach.
2.  Prepare the `GibbsKernel` class to expose $d_0$ and $k$ as `theta` parameters for the `sklearn` optimizer.
3.  Verify the sigmoid logic against the `californiav3` domain bounds.
