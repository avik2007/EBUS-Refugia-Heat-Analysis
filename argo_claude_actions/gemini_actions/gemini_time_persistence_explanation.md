# Analysis: Oscillating Time Persistence in 3D Matern GP

**Date:** 2026-03-30
**Topic:** Alternating `scale_time_days` in the 2015 California Current System (CCS) Skin Layer.

### 1. Diagnosis: Sampling Aliasing

The "striking" alternating pattern (every other 15-day window pinning to the upper bound) appears to be a **sampling aliasing effect** between the Argo float surfacing frequency and the analysis step size.

*   **Argo Cycle:** Standard Argo floats surface approximately every **10 days**.
*   **Window Step:** The analysis steps forward by **15 days**.
*   **Aliasing Pattern:** The Least Common Multiple is $LCM(10, 15) = 30$ days (exactly **two windows**).

This means the distribution of sampling times relative to your window center (the prediction target $t̃ = 0$) is identical every other window.

### 2. Physical Interaction (Skin Layer Specifics)

This oscillation is a signature of the **Skin Layer (0–100m)** because it is dominated by high-frequency atmospheric forcing (wind stress, heat flux). 

*   **Low Temporal Coherence:** In the Skin Layer, the physical decorrelation time is likely **short** (e.g., 3–7 days).
*   **Information Gap:** When the 15-day step causes the nearest float surfacings to shift away from the window center (e.g. from $\pm 2.5$ days to $\pm 7.5$ days), the temporal gap exceeds the coherence time of the water mass.
*   **Model Response:** Lacking sufficient temporal information to resolve a gradient, the GP "fails gracefully" by pinning to the maximum length scale (treating the window as a 2D spatial model).

### 3. Hypothesis: "Stealth Warming" Validation

The disappearance of this "beat" in deeper layers will be strong evidence for the **Stealth Warming hypothesis**.

*   **Source Layer Expectation:** In the **Source Layer (150–400m)**, where upwelling and sub-thermocline persistence are much higher ($l_t > 20$ days), the model should be able to bridge the 7.5-day gap easily. 
*   **Prediction:** The time length scale will remain stable and "free" (not pinned) across all windows in deeper layers, regardless of the sampling aliasing.

### 4. Recommendation

*   **Do not adjust bounds further.** The ceiling pinning is an informative diagnostic of data/physics mismatch, not a numerical error.
*   **Proceed to Source Layer Analysis.** Use the stable length scales in deeper water as the comparative benchmark against this "shredded" surface signal.
