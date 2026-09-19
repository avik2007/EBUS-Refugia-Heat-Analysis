# Ocean Turbulence, Tracer Spectra & Hydrography — Master Reading List

> **Standing Instruction for Antigravity:** This reading list is an active study queue for the USER. Antigravity must proactively remind the user to read, review, and synthesize these texts during milestone checkpoints, science reviews, and planning sessions.

---

## Priority 1: Tracer Spectra, Structure Functions & Power Laws

### 1. McCaffrey, Fox-Kemper, and Forget (2015)
* **Full Citation:** McCaffrey, K., B. Fox-Kemper, and G. Forget, 2015: Estimates of Ocean Macroturbulence: Structure Function and Spectral Slope from Argo Profiling Floats. *Journal of Physical Oceanography*, **45**(7), 1773–1793. [doi:10.1175/JPO-D-14-0023.1](https://doi.org/10.1175/JPO-D-14-0023.1).
* **Core Topic:** Structure functions from irregular Argo float pairs; empirical tracer spectra and macroturbulence scaling.
* **Key Mathematical Concepts:**
  * Second-order structure function:
    $$D_2(r) = \left\langle [T(\mathbf{x} + \mathbf{r}) - T(\mathbf{x})]^2 \right\rangle$$
  * Connection to empirical spatial variogram and covariance:
    $$\gamma(r) = \frac{1}{2} D_2(r) = \sigma^2 [1 - \rho(r)]$$
  * Power-law relation between structure function exponent $\zeta_p$ and 1D wavenumber spectral slope $E(k) \propto k^{-n}$:
    $$D_p(r) \propto r^{\zeta_p}, \quad \text{where } n = \zeta_2 + 1 \quad (\text{for } 1 < n < 3)$$
* **Why This Matters for ArgoEBUS:**
  * Bypasses the GPR optimizer's flat marginal-likelihood surface on sparse rolling windows (which caused the $d_0$ identifiability failure, CV $0.6\text{--}0.74$).
  * Provides a model-independent empirical covariance function directly from float pair distributions.
  * Allows cross-layer comparisons of integral scales and spectral slopes between Skin ($0\text{--}100\,\text{m}$), Source ($150\text{--}400\,\text{m}$), and Background ($500\text{--}1000\,\text{m}$).

---

### 2. Klein, Treguier, and Hua (1998)
* **Full Citation:** Klein, P., A. M. Treguier, and B. L. Hua, 1998: Three-dimensional stirring of thermohaline fronts. *Journal of Marine Research*, **56**(3), 589–612. [doi:10.1357/002224098321822349](https://doi.org/10.1357/002224098321822349).
* **Core Topic:** Frontogenesis, submesoscale stirring, and tracer variance power laws in active frontal zones.
* **Key Mathematical & Physical Concepts:**
  * Predicts that straining by mesoscale eddies on sharp upper-ocean buoyancy gradients generates an active/passive tracer variance spectrum scaling as:
    $$E_T(k) \propto k^{-2}$$
  * Corresponds to a first-order structure function exponent $\zeta_1 \approx 1$ and second-order exponent $\zeta_2 \approx 1\text{--}2$.
  * Emphasizes the role of ageostrophic vertical circulations associated with 3D frontogenesis that inject tracer variance across scales.
* **Why This Matters for ArgoEBUS:**
  * In the California Current System, wind-driven coastal upwelling creates intense thermohaline fronts and filaments.
  * The $k^{-2}$ scaling provides the physical baseline for the Skin layer ($0\text{--}100\,\text{m}$), explaining why shallow lengthscales are much shorter and steeper than deep interior dynamics.

---

### 3. Vallis (2017) — Textbook Reference
* **Full Citation:** Vallis, G. K., 2017: *Atmospheric and Oceanic Fluid Dynamics: Fundamentals and Large-Scale Circulation*. 2nd Edition, Cambridge University Press. [doi:10.1017/9781107588417](https://doi.org/10.1017/9781107588417).
* **Key Chapters to Study:**
  * **Chapters on Geostrophic Turbulence and Tracer Dynamics** (2D turbulence cascades, passive and active scalar power laws).
* **Core Power-Law Regimes Summary:**
  1. **$k^{-1}$ (Batchelor Regime):**
     * Tracer variance cascade in the viscous-convective subrange where the velocity field acts as a persistent large-scale strain field: $E_T(k) \propto k^{-1}$.
  2. **$k^{-5/3}$ (Kolmogorov–Obukhov–Corrsin Regime):**
     * Inertial-convective subrange of 3D isotropic turbulence or 2D inverse energy cascade range: $E_T(k) \propto \chi \epsilon^{-1/3} k^{-5/3}$.
  3. **$k^{-2}$ (Surface Quasi-Geostrophic / Frontogenesis Regime):**
     * Surface buoyancy-dominated stirring where buoyancy acts as active surface potential vorticity (Klein et al. 1998, Held et al. 1995): $E_b(k) \propto k^{-2}$.
  4. **$k^{-3}$ (Charney QG Enstrophy Inertial Range):**
     * Interior quasi-geostrophic turbulence where potential vorticity variance cascades downscale: kinetic energy $E(k) \propto k^{-3}$, passive tracer variance $E_T(k) \propto k^{-1}$ to $k^{-3}$ depending on injection scale and non-locality.
* **Why This Matters for ArgoEBUS:**
  * Establishes the theoretical criteria for selecting covariance kernel smoothness ($\nu$ in Matérn) across the vertical sandwich.
  * Tells us what physical dynamics (surface frontogenesis vs. interior geostrophic macroturbulence) we are actually sampling in each layer.

---

## Priority 2: Climatology, Large-Scale Hydrography & Coastal Dynamics

### 4. Roemmich and Gilson (2009)
* **Full Citation:** Roemmich, D., and J. Gilson, 2009: The 2004–2008 mean and annual cycle of temperature, salinity, and steric height in the global ocean from the Argo Program. *Progress in Oceanography*, **82**(2), 81–100. [doi:10.1016/j.pocean.2009.03.004](https://doi.org/10.1016/j.pocean.2009.03.004).
* **Role in Project:** Benchmark RG climatology used as our Prior Mean Anchor function. Dual-Gaussian covariance model (140 km & 1111 km).

### 5. Desbruyères et al. (2017)
* **Full Citation:** Desbruyères, D., E. L. McDonagh, B. A. King, and V. Thierry, 2017: Global and full-depth ocean temperature trends during the early twenty-first century from Argo and repeat hydrography. *Journal of Climate*, **30**(6), 1985–1997. [doi:10.1175/JCLI-D-16-0396.1](https://doi.org/10.1175/JCLI-D-16-0396.1).
* **Role in Project:** Isopycnal decomposition of temperature trends into *heave* (vertical displacement) vs. *spiciness* (water-mass changes along isopycnals).

### 6. Zaba and Rudnick (2016)
* **Full Citation:** Zaba, K. D., and D. L. Rudnick, 2016: The 2014–2015 warming anomaly in the Southern California Current System observed by underwater gliders. *Geophysical Research Letters*, **43**(3), 1241–1248. [doi:10.1002/2015GL067550](https://doi.org/10.1002/2015GL067550).
* **Role in Project:** Observational ground truth for the 2014–2016 "Blob" warming in the California Current System across shelf and offshore glider transects.
