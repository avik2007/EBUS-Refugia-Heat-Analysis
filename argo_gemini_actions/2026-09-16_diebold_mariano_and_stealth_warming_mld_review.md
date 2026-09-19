# Scientific Review: Diebold–Mariano Significance Audit & Stealth Warming / MLD Dynamics

**Date:** 2026-09-16  
**Author:** Antigravity (Science Partner & Reviewer)  
**Status:** Completed & Logged

---

## Executive Summary

This review addresses two foundational pillars of the ArgoEBUS project:
1. **Statistical Verification:** Reviewing the Diebold–Mariano (DM) tests with the Harvey–Leybourne–Newbold (HLN 1997) finite-sample modification across all three scientific layers, verifying that the Gibbs non-stationary kernel's superiority over the stationary Matérn-0.5 is statistically sound despite temporal autocorrelation.
2. **Physical Hypothesis Refinement:** Examining the interplay between **Ekman coastal upwelling** and **subsurface thermocline/pycnocline warming**, specifically addressing how subsurface heat accumulation modulates stratification ($N^2$) and deepens/alters the mixed layer depth (MLD).

---

## 1. Diebold–Mariano Audit with HLN Finite-Sample Correction

### 1.1 Methodological Context
In our rolling-window cross-validation framework ($W = 45\text{ days}$, $S = 10\text{ days}$), consecutive windows share $\approx 78\%$ of their float profiles. This induces strong positive autocorrelation in the loss differential series:
$$d_t = \text{RMSRE}_{\text{Matérn}, t} - \text{RMSRE}_{\text{Gibbs}, t}$$

* **Naive Tests Overstate Significance:** Standard independent-sample tests ($t$-test, Wilcoxon) underestimate standard errors, creating spurious statistical significance.
* **HAC / Newey–West Variance:** The Diebold–Mariano (1995) test uses a Bartlett kernel to account for autocovariance up to the overlap truncation lag:
  $$h = \left\lfloor \frac{W - 1}{S} \right\rfloor = \left\lfloor \frac{45 - 1}{10} \right\rfloor = 4$$
* **HLN (1997) Correction for Small Samples:** With $N = 34$ evaluation windows, the asymptotic normal distribution $\mathcal{N}(0, 1)$ is anti-conservative. The Harvey–Leybourne–Newbold correction applies the finite-sample scaling:
  $$DM^* = DM \times \left[ \frac{N + 1 - 2h + h(h-1)/N}{N} \right]^{1/2}$$
  and evaluates $DM^*$ against Student's $t$-distribution with $N - 1 = 33$ degrees of freedom.

### 1.2 Quantitative Audit Results

| Layer | Depth | Matérn Med. RMSRE | Gibbs Med. RMSRE | Rel. Median Imprv. | Asymp. $DM$ ($p_{\text{one}}$) | HLN $DM^*$ | **HLN $p$-val ($t_{33}$, one-sided)** | Gibbs Uncertainty Calibration $\text{Std}(Z)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Skin** | $0\text{--}100\text{m}$ | 4.25% | 3.71% | **+12.86%** | 3.193 ($7.05 \times 10^{-4}$) | **2.864** | **$3.55 \times 10^{-3}$ ($< 0.01$)** | $0.9958 \pm 0.0883$ |
| **Source** | $150\text{--}400\text{m}$ | 3.05% | 2.63% | **+13.53%** | 2.051 ($2.01 \times 10^{-2}$) | **1.840** | **$0.0374$ ($< 0.05$)** | $0.9816 \pm 0.0834$ |
| **Background** | $500\text{--}1000\text{m}$ | 2.50% | 2.03% | **+18.87%** | 5.068 ($2.01 \times 10^{-7}$) | **4.546** | **$3.65 \times 10^{-5}$ ($\ll 0.001$)** | $0.9664 \pm 0.1002$ |

### 1.3 Key Findings
1. **Source Layer Significance Holds:** Even under the conservative HLN finite-sample penalty, the Source Layer remains statistically superior at the $95\%$ confidence level ($p = 0.0374$).
2. **Uncertainty Calibration Collapse:** The stationary Matérn kernel suffered catastrophic variance inflation ($\text{Std}(Z) = 2.63$) during the 2015 Pacific Blob onset. The Gibbs kernel stabilized $\text{Std}(Z)$ to $0.97\text{--}1.00$ ($\sigma \sim 0.08\text{--}0.10$), proving its adaptive spatial lengthscale prevents severe underestimation of uncertainty near dynamic coastal transitions.
3. **Physical Regime Boundary ($d_0$):** The learned sigmoid midpoint converged to $d_0 \approx 204\text{ km}$ (Skin) and $d_0 \approx 227\text{ km}$ (Source), matching the physical width of California Current coastal upwelling filaments without manual tuning.

---

## 2. Hypothesis Review: Ekman Upwelling vs. Pycnocline Stratification & MLD

### 2.1 The Baseline "Stealth Warming" Hypothesis
* **Upwelling Physics:** Alongshore equatorward winds generate offshore surface Ekman transport:
  $$M_E = \frac{\tau_y}{\rho_0 f}$$
  To conserve mass, cold, nutrient-rich water is drawn upward from $100\text{--}300\text{m}$ along the coast.
* **Thermal Refugia:** Historically, this upwelled water provides a thermal sanctuary for marine ecosystems, shielding coastal biomes from atmospheric marine heatwaves (e.g., the 2014–2015 "Blob").
* **Stealth Warming Mechanism:** The California Undercurrent (CUC) transports subsurface heat poleward within the Source Layer ($150\text{--}400\text{m}$). Because this heat is capped beneath the surface, it remains undetected by satellite SST.
* **The Refugia Failure:** As Ekman suction continues, it draws from this pre-warmed subsurface reservoir. Instead of supplying cold water to the shelf, it pumps warm water directly into coastal ecosystems, causing refugia collapse from below.

---

### 2.2 The Pycnocline & Mixed Layer Depth (MLD) Feedback

The addition of thermocline/pycnocline thermodynamic modifications introduces a vital second physical pathway:

```
                            [ Atmospheric Wind Stress τ ]
                                          │
                                          ▼ (Mechanical Stirring u*³/κz)
    ┌───────────────────────────────────────────────────────────────────────────┐
    │ Surface Mixed Layer (Skin Layer, 0–100m)                                  │
    │ Temperature homogenized by surface turbulence; MLD typically ~20–50m     │
    └───────────────────────────────────────────────────────────────────────────┘
                                          ▲
                   Entrainment Flux       │   Stratification Barrier:
                 w_e = ∂h_mld/∂t + ...    │   N² = -g/ρ₀ · ∂ρ/∂z
                                          ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │ Pycnocline / Upper Transition Layer (100–150m)                            │
    │ * Subsurface heat accumulation reduces vertical density gradient ∂ρ/∂z    │
    │ * Stratification N² drops -> Richardson number Ri drops                   │
    │ * Lower barrier to turbulent vertical entrainment                         │
    └───────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │   Coastal Upwelling Suction:
                                          │   w_E = (1/ρ₀) curl(τ/f)
    ┌───────────────────────────────────────────────────────────────────────────┐
    │ Source Layer (150–400m): California Undercurrent (CUC) Heat Core          │
    │ Stealth heat advected poleward in narrow coastal corridor                 │
    └───────────────────────────────────────────────────────────────────────────┘
```

#### A. Stratification Weakening ($N^2$ Erosion)
The buoyancy frequency $N^2$ defines the energetic barrier separating the turbulent mixed layer from the quiescent interior:
$$N^2 = -\frac{g}{\rho_0} \frac{\partial \rho}{\partial z} \approx g \left( \alpha \frac{\partial T}{\partial z} - \beta \frac{\partial S}{\partial z} \right)$$
* When subsurface stealth warming heats the water column at $100\text{--}200\text{m}$, $\frac{\partial T}{\partial z}$ between the mixed layer base and the pycnocline decreases.
* As a direct result, **$N^2$ decreases (stratification weakens)**.
* The gradient Richardson number:
  $$Ri = \frac{N^2}{(\partial u / \partial z)^2}$$
  drops toward or below the critical stability threshold ($Ri_{\text{crit}} \approx 0.25$).

#### B. Mixed Layer Deepening ("Lengthening")
* With a weakened density gradient at the pycnocline, mechanical stirring generated by surface winds ($u_*^3$) and wave breaking can penetrate deeper before being arrested by buoyancy forces.
* **Turbulent entrainment deepens the mixed layer ($h_{\text{MLD}}$ increases)**.
* **Dual Thermal Consequences:**
  1. **Larger Thermal Inertia:** A deepened mixed layer has higher column heat capacity ($C = \rho_0 C_p h_{\text{MLD}}$). It dampens short-term daily SST spikes, but stores a much larger integrated volume of heat, extending thermal anomalies well into autumn and winter.
  2. **Dilution vs. Direct Upwelling:** At the immediate coast, Ekman suction tilts isopycnals upward, causing the pycnocline to shoal and surface. If the water being upwelled has already been entrained and warmed, the coastal boundary layer becomes a deepened, moderately warm layer rather than a shallow, cold upwelling wedge.

#### C. Surface Heatwave Coupling (Blob Interaction)
* During extreme surface marine heatwaves (e.g., summer 2014–2015), atmospheric heating can warm the upper 10–20m faster than subsurface warming occurs, creating transient hyper-stratification that caps the mixed layer shallowly.
* Once seasonal cooling or storm-driven wind events occur in autumn, this thin surface layer rapidly mixes downward into the pre-warmed pycnocline, triggering rapid, deep mixed-layer deepening and long-duration subsurface heat retention.

---

## 3. Scientific Conclusions & Architecture Validation

1. **The 3-Layer Sandwich is Mechanistically Sound:**
   * **Skin ($0\text{--}100\text{m}$):** Captures atmospheric forcing and the active mixed layer.
   * **Buffer ($100\text{--}150\text{m}$):** Houses the pycnocline transition where $N^2$ regulates entrainment.
   * **Source ($150\text{--}400\text{m}$):** Isolates the CUC advective corridor before entrainment occurs.
   * **Background ($500\text{--}1000\text{m}$):** Provides the unperturbed deep ocean control.
2. **Next Diagnostic Priority:**
   * Implementing profile-level $N^2(z)$ and MLD algorithms in `ebus_core/argoebus_thermodynamics.py` will allow us to quantify whether the 2015 stealth warming event was accompanied by pycnocline erosion and mixed layer deepening.
