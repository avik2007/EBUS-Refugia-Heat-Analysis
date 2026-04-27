# Project Directive: The RG-Gibbs Non-Stationary Model

**Objective:** Implement a non-stationary Gaussian Process Regression (GPR) using a Gibbs Kernel and a Roemmich-Gilson (RG) Mean Function to resolve fine-scale "Refugia" features despite variable data density.

## 1. The Gibbs Kernel (The "Zoom" Engine)
**Mechanism:** Unlike a standard stationary kernel with a fixed lengthscale ($l$), the Gibbs kernel uses a lengthscale function $l(x)$.
**Coastal Logic:** We will define $l(x)$ to be small (~100km) near the California coastline (high data density, complex physics) and large (~400km) as we move toward the $-135^\circ$ western boundary (lower data density, smoother physics).
**Benefit:** This satisfies the requirement for statistical confidence by allowing the model to reach further for data in empty regions without blurring the high-stakes coastal undercurrent.

## 2. The Roemmich-Gilson (RG) Anchor
**Mean Function Prior:** Ingest the Copernicus RG Climatology (INSITU_GLO_PHY_TS_OA_MY_013_052) as the base layer.
**The Residual Workflow:** The Gibbs GPR will model the Residual Anomaly ($T_{observed} - T_{RG\_Mean}$).
**Statistical Integrity:** Where $l(x)$ is large and data is sparse, the Gibbs kernel naturally reverts the prediction to the RG Mean, ensuring the model never "hallucinates" warming where no data exists.

## 3. Anisotropy & The Vertical Sandwich
**Directional Weighting:** Even within the Gibbs framework, we will maintain an Anisotropic Ratio (Meridional > Zonal) to respect the along-shore flow of the California Current.
**Layered Analysis:** This hybrid model will be applied to the three Roemmich-defined layers: 
- **Response Layer (0–100m)**
- **Source Water (100–500m)**
- **Background Control (500–1500m)**

## 4. Technical Implementation Task (Claude Code / Cursor)
1. **Gibbs Implementation:** Initialize the GPR using a Gibbs Kernel where the lengthscale parameter is a function of longitude.
2. **Prior Integration:** Set the `mean_function` to the interpolated Copernicus RG grid.
3. **Interactive Demo:** Feature a 'Focus Slider' that shows how the Gibbs kernel tightens its resolution around the thermal refugia compared to the 'flat' global smoothing of standard products.
