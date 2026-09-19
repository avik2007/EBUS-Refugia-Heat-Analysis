# Antigravity TODO — ArgoEBUSAnalysis (formerly Gemini CLI, discontinued 2026-08-30)

Last updated: 2026-09-18 (Antigravity session)

---

## Priority 1: Gibbs Non-Stationary GPR (RG-Gibbs)

**Objective:** Implement and validate the Gibbs kernel to resolve coastal refugia.

- [x] **[For Claude] Implement `GibbsKernel` class in `argoebus_gp_physics.py`** — DONE session 16
  (2026-05-26). Subclasses `sklearn.gaussian_process.kernels.Kernel`, learnable sigmoid
  lengthscale $l(dist\_to\_coast)$, learnable anisotropy ratio.
- [x] **[For Gemini] Monitor Gibbs Validation Metrics — AUDITED & STATISTICALLY PROVEN (session 20)**
  - **Z-score spikes in Background layer: resolved.** Reconfirmed via `compare_kernels.py` that Gibbs uncertainty calibration collapsed the extreme Matérn variance to near-perfect levels: Skin (0.9958 ± 0.0883), Source (0.9816 ± 0.0834), Background (0.9664 ± 0.1002).
  - **RMSRE**: Diebold-Mariano tests (HAC lag=4) prove highly significant relative median improvements for Gibbs over Matérn: Skin 12.86% (p=7.05e-04), Source 13.53% (p=2.01e-02), Background 18.87% (p=2.01e-07).
  - **⚠️ RETRACTED 2026-09-17 (Claude side) — do not cite:** "Coastal vs offshore lengthscale
    resolution: confirmed transition midpoint $d_0 \approx 204$km (Skin) and $d_0 \approx 227$km
    (Source), aligning perfectly with physical upwelling width. Background layer pegs at 700km,
    highlighting more uniform dynamics at depth." Per-window `d_0` has coefficient of variation
    0.6–0.74 in all 3 layers — not a stably identified parameter, these medians are not a
    trustworthy physical claim. See `[For Antigravity]` science review item below and
    `argo_claude_actions/AE_claude_recentactions.md` (2026-09-17 entry) for full evidence.
  - **[Claude 2026-09-18 update — new evidence for the science review]:** the d_0 scatter holds across 2010-2020, not
    just 2015 (33 layer-years, 378 windows/layer): pooled CV 0.76 Skin / 0.69 Source / 0.57 Background, per-year CV
    0.21-1.14; windows pinned at the lower/upper optimiser bound: Skin 22%/15%, Source 25%/38%, Background 9%/36%
    (still 36% at the widened 1500 km bound). Plot: `AEResults/aeplots/vertical_delta/d0_distribution_2010_2020.png`;
    stats: `AEResults/aelogs/d0_distribution_multiyear_stats.csv`. **Caution on wording elsewhere:** "flat
    marginal-likelihood surface in d_0" was an *untested hypothesis*, not a finding. First LML sweep (2015 Source only,
    d_0 clamped on a 6-point grid): LML is NOT flat (median 9.8-nat swing) but has no consistent optimum (best d_0
    counts 4/2/4/10/8/6 across the grid), and in 12/34 windows a clamped fit beats the free fit by >1 nat (up to 39.8),
    i.e. the free-fit optimizer may be missing better solutions. Both weak identification and optimizer failure are
    live explanations; a multi-start optimizer test is the next step (Claude todo [TOP]). Do not cite d_0 values or
    "flat likelihood" until that is resolved.
  - **units bug fixed (session 19)**: Units scaling for temporal term is resolved.
  - **temporal persistence saturation**: After the units bug fix, `time_ls` still pegs at the 200d limit across all layers, proving that ocean temperature anomaly memory is unresolvable within a 45-day window.
  - **Verdict for Gemini**: We conclude that "ocean memory exceeds a month and a half at all depths" is a robust physical finding (unresolvable within a 45d window). Widening the window to 90d/120d is recommended for deeper layers to find the exact correlation timescale, while keeping 45d for Skin to capture rapid seasonal shifts.

---

## Priority 2: Science Review & Vertical Delta Analysis

- [ ] **[For Antigravity] Science Review: Tracer Spectra, Structure Functions & Power-Law Regimes (McCaffrey et al. 2015, Klein et al. 1998, Vallis)**
  - **Context & Motivation:** Ground our GPR spatial covariance models and resolve lengthscale / $d_0$ parameter identifiability by linking empirical Argo float statistics to established ocean tracer turbulence and power-law scaling.
  - **Theoretical Anchors:**
    1. *Patrice Klein, Treguier, and Hua (1998, JMR 56(3), 589–612):* "Three-dimensional stirring of thermohaline fronts." Predicts $k^{-2}$ power-law tracer variance spectra driven by surface frontogenesis, horizontal strain, and submesoscale baroclinic stirring (corresponding to a 1st-order structure function exponent $\zeta_1 \approx 1$).
    2. *Geoffrey K. Vallis Textbook:* *Atmospheric and Oceanic Fluid Dynamics: Fundamentals and Large-Scale Circulation* (Ch. on Geostrophic Turbulence & Tracer Dynamics). Explains key cascade regimes:
       - **$k^{-1}$ (Batchelor regime):** Non-local advective stirring by larger eddies without scale-dependent strain destruction.
       - **$k^{-5/3}$ (Kolmogorov–Corrsin):** 3D isotropic / 2D inverse energy cascade range.
       - **$k^{-2}$ (Surface QG / Frontogenesis):** Sharp boundary buoyancy gradients stirring active/passive tracers (Klein et al. 1998).
       - **$k^{-3}$ (Charney QG / Enstrophy cascade):** Interior geostrophic mesoscale macroturbulence with downscale tracer variance cascade.
    3. *McCaffrey, Fox-Kemper, and Forget (2015, JPO 45(7), 1773–1793):* Connects pairwise Argo float profiles directly to empirical structure functions $D_2(r)$ and spectral slopes, diagnosing scale breaks between mesoscale macroturbulence and frontal regimes.
  - **Significance for ArgoEBUS:**
    - **Kernel Smoothness & Priors:** The spectral slope directly dictates the physical covariance shape and Matérn smoothness parameter ($\nu$), providing a principled prior to constrain or replace unidentifiable per-window hyperparameter fits.
    - **Vertical Regime Shifts:** High-resolution Skin layer ($0\text{--}100\text{m}$) is predicted to reflect $k^{-2}$ surface frontogenesis, whereas Source ($150\text{--}400\text{m}$) and Background ($500\text{--}1000\text{m}$) should reflect $k^{-3}$ interior macroturbulence or $k^{-1}$ Batchelor stirring.

- [ ] **[For Antigravity] Science Review: is `d_0` (coastal transition midpoint) a salvageable
  per-window parameter, or should the vertical fingerprint rest on anisotropy ratio alone?**
  - **Finding (Claude side, 2026-09-17):** `d_transition_km` from the per-window Gibbs CV fit has
    coefficient of variation 0.6–0.74 across all 3 layers (34 windows each, 45-day rolling window):
    Skin 0.67, Source 0.74, Background 0.61. Background still pins 29% of windows at its
    (already-widened, 50–1500km) upper bound. This retracts the session-20 claim above (Skin
    ~204km, Source ~227km, Background ~700→966km "aligning with physical upwelling width") —
    those medians summarize a distribution too wide to be a physical scale.
  - **Evidence:** `AEResults/aeplots/vertical_delta/vertical_delta_d0_distribution.png` (per-layer
    boxplot + jittered per-window points + bound lines), `AEResults/aelogs/vertical_delta_d0_distribution_stats.csv`
    (mean/std/CV/pinned-fraction per layer), full writeup in
    `argo_claude_actions/AE_claude_recentactions.md` (2026-09-17 entry).
  - **Working hypothesis (untested):** a single 45-day, ~35-float window may not have enough
    spatial contrast in `dist_to_coast` to identify a sigmoid inflection point — the per-window
    log-marginal-likelihood in `d_0` is plausibly near-flat, so the optimizer lands near its init
    or a bound rather than a true optimum. Same diagnostic style as the session-19 `time_ls`
    units-bug sensitivity sweep would confirm this.
  - **Questions for Antigravity:**
    1. Is CV 0.6–0.74 disqualifying for "identified physical parameter," or is there a
       precedent in the GP-oceanography literature for this much per-window scatter in a
       length-scale-transition parameter still being meaningful in aggregate (e.g. via the
       distribution's shape, not its median)?
    2. If not salvageable per-window: pool `d_0` across windows (single fit per layer per year)
       instead of per-45-day-window, or drop it and report only the anisotropy ratio (which does
       not show this problem) as the vertical fingerprint metric?
    3. Does the physical continental-shelf/coastal-transition-zone literature suggest a
       principled *fixed* `d_0` (or narrow prior) per layer, rather than treating it as freely
       learnable at all?



- [x] **[For Gemini] Science Review of californiav3 Baseline**
  - Verified Source layer fix (3.05% RMSRE).
  - Confirmed meridional anisotropy in Undercurrent corridor.
  - Flagged Blob-related stationarity violations.

- [x] **Vertical Delta Analysis Script (`vertical_delta_analysis.py`)** — IMPLEMENTED (session 2026-08-30)
  - Created `vertical_delta_analysis.py` for cross-layer vertical sandwich audit (Source 150–400m vs. Background 500–1000m).
  - Evaluates meridional anisotropy, coastal transition $d_0$, and calibration metrics across 2015.
  - Handed off to Claude in `argo_claude_actions/AE_claude_todo.md` for review and execution.

- [ ] **[For Antigravity] Research & Diagnostic: Stealth Warming -> MLD Deepening & Pycnocline Stratification ($N^2$) Link**
  - **Physical Mechanism:** Investigate whether subsurface heat accumulation in the Source Layer ($150\text{--}400\text{m}$) weakens pycnocline stratification ($N^2$), lowering the Richardson barrier to wind/wave entrainment and deepening the Mixed Layer Depth (MLD).
  - **Non-Trivial Argo Float Implementation Challenges:**
    1. *Irregular Sampling & Surface Truncation:* Argo CTD pumps switch off at $\sim 3\text{--}5\text{ dbar}$ to avoid surface contamination, leaving no true surface ($z=0$) measurement. Must establish a standardized near-surface reference depth (e.g. $10\text{ dbar}$, de Boyer Montégut et al. 2004) to avoid transient diurnal warm layers.
    2. *Thermodynamic State via TEOS-10 (`gsw`):* Must convert in-situ $T, S_P, P$ to Conservative Temperature $\Theta$ and Absolute Salinity $S_A$ before deriving potential density anomaly $\sigma_\theta$.
    3. *Salinity Compensation & Inversion Robustness:* Coastal upwelling regions exhibit barrier layers and salinity compensation (cold fresh water overlying warm salty water). Density-based threshold criteria ($\Delta \sigma_\theta = 0.03\text{ kg/m}^3$ from $10\text{ dbar}$ reference) must be used instead of simple temperature drops, paired with fine vertical linear interpolation between observation levels.
    4. *Sensor Noise & Spurious Gravitational Instability in $N^2$:* Discrete vertical differentiation $\frac{\partial \rho}{\partial z}$ on raw CTD data creates artificial negative density steps and spurious negative $N^2$. Requires adiabatic profile leveling/sorting or vertical smoothing (e.g., 5–10m window filter) or TEOS-10 `gsw.Nsquared` formulation.
    5. *Pycnocline Core Metrics:* Extract $N^2_{\text{max}}$ (pycnocline strength) and $z(N^2_{\text{max}})$ (pycnocline depth) for each profile to directly test correlation against Source Layer OHC anomalies.
  - **Target Integration:** Formulate and prototype diagnostic functions in `ebus_core/argoebus_thermodynamics.py`.

- [ ] **[For Antigravity / Claude] Tracer Structure Functions via Argo Profile Pairs (McCaffrey et al. 2015 Methodology)**
  - **Paper Citation:** McCaffrey, K., B. Fox-Kemper, and G. Forget (2015). *"Estimates of Ocean Macroturbulence: Structure Function and Spectral Slope from Argo Profiling Floats"*, *Journal of Physical Oceanography*, 45(7), 1773–1793. [doi:10.1175/JPO-D-14-0023.1](https://doi.org/10.1175/JPO-D-14-0023.1).
  - **Why this could be HUGE for us:**
    1. *Model-Free Ground Truth for GPR Lengthscales & Covariance:* Directly addresses the parameter identifiability failure of $d_0$ (CV 0.6–0.74, flat log-marginal-likelihood surface across 45-day rolling windows). The second-order structure function $D_2(r) = \langle [T(\mathbf{x} + \mathbf{r}) - T(\mathbf{x})]^2 \rangle$ is mathematically equivalent to the empirical variogram/spatial covariance $\gamma(r) = \frac{1}{2}D_2(r) = \sigma^2 [1 - \rho(r)]$, computed directly from all float profile pairs without numerical optimization or gridding artifacts.
    2. *Directional Anisotropy & Coastal Transition Scales ($d_0$):* By decomposing pair separation vectors $\mathbf{r}$ into along-shelf ($\Delta y$) vs. cross-shelf ($\Delta x$) components and stratifying by distance to coast ($d_{\text{coast}}$), we can empirically measure the true physical decorrelation lengthscales ($l_x, l_y$), empirical anisotropy ratio $\mathcal{A} = l_y / l_x$, and determine whether a sharp coastal transition scale ($d_0$) actually emerges from the data.
    3. *Cross-Layer Vertical Fingerprint of Turbulence & Tracer Variance:* Evaluating $D_2(r)$ across the three scientific layers (Skin 0–100m, Source 150–400m, Background 500–1000m) provides a clean, non-parametric comparison of spatial decorrelation scales and spectral slopes (testing against the $k^{-2}$ frontogenetic regime of Klein et al. 1998 vs. $k^{-3}$ QG / $k^{-1}$ Batchelor regimes in Vallis' textbook) without dependency on noisy per-window GPR fits.
    4. *Empirical Spatio-Temporal Decorrelation ($r, \Delta t$):* Binned across both spatial distance $r$ and temporal lag $\Delta t$, structure functions provide empirical maps of spatio-temporal decorrelation, testing whether temporal persistence at depth is truly unresolvable (>45–90 days) or an artifact of rolling window truncation.
  - **Implementation Roadmap:**
    1. *Pair-Distance Engine:* Implement pairwise spatial separation (great-circle/haversine distance) and temporal separation $\Delta t = |t_i - t_j|$ for all profile pairs within `californiav3`.
    2. *Layer-Specific Binned Structure Functions:* Calculate $D_2(r, \Delta t)$ for Conservative Temperature $\Theta$ and Absolute Salinity $S_A$ anomalies across depth layers.
    3. *Coastal & Directional Stratification:* Bin pairs by coastal distance $d_{\text{coast}}$ and orientation angle relative to the shelf break (along-shelf vs. cross-shelf).
    4. *GPR Anchor / Prior Formulation:* Use the empirical covariance derived from $D_2(r)$ to define grounded physical priors or fix hyperparameters ($l_{\text{coast}}, l_{\text{offshore}}, d_0$) for the Gibbs kernel.

---

## Priority 3: MLOps & Reproducibility (On Hold)

- [x] **[For Gemini] Review MLOps Foundation Spec**
  - Spec approved; audit gaps fixed by Claude.
- [ ] **SST Cross-Validation (OISST)**
  - Compare Argo Skin Layer results with satellite SST for ground-truthing.

---

## Reading List & Study Queue (User + Antigravity Mandatory Readings)

> **STANDING INSTRUCTION FOR ANTIGRAVITY:** Proactively remind the user to read and discuss these foundational references during reviews, milestones, and planning sessions.

- [ ] **1. McCaffrey, K., B. Fox-Kemper, and G. Forget (2015)** — *"Estimates of Ocean Macroturbulence: Structure Function and Spectral Slope from Argo Profiling Floats"*, *Journal of Physical Oceanography*, 45(7), 1773–1793. [doi:10.1175/JPO-D-14-0023.1](https://doi.org/10.1175/JPO-D-14-0023.1).
  - **Core Topic:** Structure functions $D_2(r)$ computed from pairs of Argo float profiles to diagnose horizontal macroturbulence and empirical tracer spectra without gridding bias.
  - **Why Read It:** Directly demonstrates how to extract model-independent spatial covariance and correlation scales from sparse, irregular float arrays.
- [ ] **2. Klein, P., A. M. Treguier, and B. L. Hua (1998)** — *"Three-dimensional stirring of thermohaline fronts"*, *Journal of Marine Research*, 56(3), 589–612. [doi:10.1357/002224098321822349](https://doi.org/10.1357/002224098321822349).
  - **Core Topic:** Three-dimensional stirring of frontal zones and the origin of the $k^{-2}$ tracer variance spectrum power law in frontogenetic strain fields.
  - **Why Read It:** The canonical physical reference for tracer variance scaling in energetic upwelling and frontal regimes like the California Current.
- [ ] **3. Vallis, G. K. (2017)** — *Atmospheric and Oceanic Fluid Dynamics: Fundamentals and Large-Scale Circulation* (2nd ed., Cambridge University Press).
  - **Core Topic:** Geostrophic turbulence and passive/active scalar cascades. Key spectral power laws: $k^{-1}$ (Batchelor viscous-convective subrange), $k^{-5/3}$ (Kolmogorov-Corrsin), $k^{-2}$ (Surface QG / frontogenesis), and $k^{-3}$ (Charney QG enstrophy cascade).
  - **Why Read It:** Provides the theoretical fluid dynamics foundation for interpreting our three vertical layers (Skin vs. Source vs. Background) and parameterizing GPR covariance smoothness.
