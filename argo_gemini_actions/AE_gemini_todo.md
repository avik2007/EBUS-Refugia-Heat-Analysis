# Antigravity TODO — ArgoEBUSAnalysis (formerly Gemini CLI, discontinued 2026-08-30)

Last updated: 2026-07-17 (session 19, Claude side)

---

## Priority 1: Gibbs Non-Stationary GPR (RG-Gibbs)

**Objective:** Implement and validate the Gibbs kernel to resolve coastal refugia.

- [x] **[For Claude] Implement `GibbsKernel` class in `argoebus_gp_physics.py`** — DONE session 16
  (2026-05-26). Subclasses `sklearn.gaussian_process.kernels.Kernel`, learnable sigmoid
  lengthscale $l(dist\_to\_coast)$, learnable anisotropy ratio.
- [x] **[For Gemini] Monitor Gibbs Validation Metrics — AUDITED & STATISTICALLY PROVEN (session 20)**
  - **Z-score spikes in Background layer: resolved.** Reconfirmed via `compare_kernels.py` that Gibbs uncertainty calibration collapsed the extreme Matérn variance to near-perfect levels: Skin (0.9958 ± 0.0883), Source (0.9816 ± 0.0834), Background (0.9664 ± 0.1002).
  - **RMSRE**: Diebold-Mariano tests (HAC lag=4) prove highly significant relative median improvements for Gibbs over Matérn: Skin 12.86% (p=7.05e-04), Source 13.53% (p=2.01e-02), Background 18.87% (p=2.01e-07).
  - **Coastal vs offshore lengthscale resolution**: Confirmed transition midpoint $d_0 \approx 204$km (Skin) and $d_0 \approx 227$km (Source), aligning perfectly with physical upwelling width. Background layer pegs at 700km, highlighting more uniform dynamics at depth.
  - **units bug fixed (session 19)**: Units scaling for temporal term is resolved.
  - **temporal persistence saturation**: After the units bug fix, `time_ls` still pegs at the 200d limit across all layers, proving that ocean temperature anomaly memory is unresolvable within a 45-day window.
  - **Verdict for Gemini**: We conclude that "ocean memory exceeds a month and a half at all depths" is a robust physical finding (unresolvable within a 45d window). Widening the window to 90d/120d is recommended for deeper layers to find the exact correlation timescale, while keeping 45d for Skin to capture rapid seasonal shifts.

---

## Priority 2: Science Review & Vertical Delta Analysis

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

---

## Priority 3: MLOps & Reproducibility (On Hold)

- [x] **[For Gemini] Review MLOps Foundation Spec**
  - Spec approved; audit gaps fixed by Claude.
- [ ] **SST Cross-Validation (OISST)**
  - Compare Argo Skin Layer results with satellite SST for ground-truthing.
