# LinkedIn draft — Gibbs kernel results (2026-07-17, session 19/20)

Scope: science-only (RMSRE + calibration). Time-persistence angle dropped —
see `argo_claude_actions/AE_claude_lessons.md` lesson #6 (units bug, superseded claim).

---

Testing a hypothesis about "stealth warming" in the California Current System —
subsurface heat that hasn't reached the surface yet — means the kriging (Gaussian
Process) kernel choice matters as much as the oceanography.

Swapped a standard Matérn kernel for a **Gibbs kernel** (spatially-varying
lengthscale, tied to distance from coast) across three depth layers of Argo float
data: 0-100m, 150-400m, 500-1000m.

Results, cross-validated:

**Prediction error (RMSRE) dropped across all three layers:**
- Skin (0-100m): 4.25% → 3.71%
- Source (150-400m): 3.05% → 2.63%
- Background (500-1000m): 2.50% → 2.03%

**Uncertainty calibration improved far more dramatically.** Z-score standard
deviation — how well the model's stated confidence matches its actual error —
went from 1.05-2.63 (Matérn, badly overconfident in places) to 0.083-0.100
(Gibbs) across all layers. Roughly an order of magnitude tighter.

The takeaway: a non-stationary kernel that lets lengthscale vary with distance
from the coast — where the physical process (upwelling, the California
Undercurrent) actually changes character — doesn't just fit slightly better.
It fixes a real calibration problem the stationary kernel couldn't see past.

[chart 1: RMSRE bar comparison, 3 layers]
[chart 2: Z-std calibration comparison, 3 layers]

---

## Data provenance

All numbers pulled directly from audit CSVs (not summarized memory) and
verified 2026-07-17:

| Layer | Matérn RMSRE | Gibbs RMSRE | Matérn std(Z) | Gibbs std(Z) |
|---|---|---|---|---|
| Skin | 4.254% | 3.707% | 1.301 | 0.088 |
| Source | 3.047% | 2.634% | 1.054 | 0.083 |
| Background | 2.503% | 2.031% | 2.630 | 0.100 |

RMSRE = median across 34 rolling CV windows per layer. std(Z) = standard
deviation of per-window Z-score across the same 34 windows (calibration
tightness; ideal Z-score mean is 1.0, see project CLAUDE.md).

Source runs:
- Matérn: `AEResults/aelogs/californiav3_..._3dmatern_w45_3dmatern_w45/`
- Gibbs (post time_ls-fix): `AEResults/aelogs/californiav3_..._3dgibbs_w45_timelsfix/`

Charts: `docs/linkedin/gibbs_vs_matern_charts.html`
