# CLAUDE.md — ArgoEBUSAnalysis Project Guide

## Project Mission
Testing the "Ocean Refugia / Stealth Warming" hypothesis: that the California Current System
is experiencing long-term subsurface warming that is transported by the California Undercurrent
(Ekman pumping source water) and has not yet surfaced to dominate the Skin Layer signal.

The 20-year study uses Argo float data and Gaussian Process Regression (Kriging) to build
spatiotemporal OHC maps across three depth layers:
- **Skin Layer** (0–100m): High volatility, atmospheric forcing dominates
- **Source Layer** (150–400m): Ekman upwelling source water — where "stealth heat" hides
- **Background Layer** (500–1000m): Deep ocean control — baseline global warming rate

If the Source Layer warms faster than the Background, that is the "stealth" signal.

---

## Workflow Principles

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions).
- If a process goes sideways, STOP and re-plan immediately.
- Write detailed specs upfront to reduce ambiguity.

### 2. Subagent Strategy
- Use subagents to keep main context clean.
- Offload research, exploration, and parallel analysis to subagents.
- One task per subagent.

### 3. Self-Improvement Loop
- After ANY correction: update `claude_reports/claude_lessons.md`.
- Write rules that prevent the same mistake from recurring.
- Review lessons at the start of each session.

### 4. Verification Before Done
- Never mark a task complete without proving it works.
- Run tests, check logs, demonstrate correctness.

### 5. Demand Elegance
- For non-trivial changes: ask "is there a more elegant way?"
- Skip for simple, obvious fixes.

### 6. Autonomous Bug Fixing
- When given a bug: fix it without hand-holding.
- Identify logs, errors, and failing tests, then resolve them.

---

## Scientific Context

### Science AI Collaboration Model
- **Gemini** (Google): Science planning, hypothesis development, research direction
- **Claude** (Anthropic): Implementation, code writing, debugging, execution

Claude should expect prompts that contain science already planned by Gemini. The job is
to implement cleanly without second-guessing the science, but to flag any implementation
risk (e.g., edge cases in data, numerical issues).

### Key Physical Metrics
- **Anisotropy Ratio** = `Lat_Scale / Lon_Scale`
  - < 1.0: Atmospheric/zonal forcing dominates (eddy-shredded)
  - > 1.0: Meridional current flow dominates (stealth transport)
  - Prediction: Ratio should **increase** with depth, proving Ekman transport pathway

- **RMSRE** (Root Mean Square Relative Error): Target < 5%
- **Std Z-Score**: Ideal = 1.0 (model calibration check)

### Depth Layer Naming Convention
Files use naming like: `california_20150101_20151231_res0_5x0_5_t30_0_d0_100`
- `d0_100` = depth 0–100m (Skin)
- `d150_400` = depth 150–400m (Source)
- `d500_1000` = depth 500–1000m (Background)

---

## Repository Structure

```
ArgoEBUSAnalysis/
├── ArgoEBUSCloud/
│   ├── ebus_core/
│   │   ├── argoebus_gp_physics.py      # GPR validation, kriging, rolling analysis
│   │   ├── argoebus_thermodynamics.py  # OHC calculation from raw bins
│   │   ├── argoebus_plotting.py        # Plotting utilities
│   │   └── ae_utils.py                 # Config, S3 paths, directories
│   ├── 01_ae_cloud_ingestion.py        # ERDDAP data pull (legacy)
│   ├── 02_ae_cloud_run.py              # Cloud pipeline: ingest + OHC calc -> S3 parquet
│   ├── 03_ae_inspect_data.py           # Analysis: rolling GP + save kriging storyboard
│   └── AEResults/
│       ├── aeplots/                    # Kriging snapshot PNGs organized by run_id
│       └── aelogs/                     # audit_*.csv and cv_details_*.pkl
├── claude_reports/
│   ├── claude_todo.md                  # Future tasks and plans
│   ├── claude_recentactions.md         # Recent actions for user review
│   └── claude_lessons.md              # Lessons learned after mistakes
├── References/                         # Scientific papers (RIS format)
├── Notebooks (1-5.ipynb)               # Exploratory notebooks (legacy)
└── CLAUDE.md                           # This file
```

---

## Key Functions (ebus_core/argoebus_gp_physics.py)

| Function | Purpose |
|---|---|
| `generalized_cross_validation()` | Global GPR validation (KFold or LOFO), auto-tune hyperparams |
| `validate_moving_window()` | Moving window local GPR validation |
| `analyze_rolling_correlations()` | Rolling window analysis — learns physics + anisotropy over time |
| `produce_kriging_map()` | Production mapper: sparse Argo → continuous gridded NetCDF |
| `plot_kriging_snapshot()` | Diagnostic: reconstruct GP map for a specific date |
| `plot_physics_history()` | Visualize Anisotropy, Noise, Z-score evolution over time |

---

## Cloud Pipeline (Scripts 02 → 03)

1. **Script 02** (`02_ae_cloud_run.py`): Pull Argo data via ERDDAP → compute OHC bins → save parquet to S3
2. **Script 03** (`03_ae_inspect_data.py`): Load parquet from S3 → rolling GP analysis → save kriging PNGs + audit CSV + CV pickle

---

## Data Notes

- **Argo float data**: ~70 unique platforms in California region per 30-day window in 2015
- **Spatial resolution**: 0.5° × 0.5° bins
- **Time resolution**: 30-day bins, 15-day step
- **Baseline**: Days since 1999-01-01
- **Validated kernel**: RBF (Squared Exponential), length scale ~1.4 sigma (~450km), noise ~0.1–0.3

---

## Style Preferences

- Keep responses concise and direct.
- No trailing summaries of what was just done.
- No emojis unless the user uses them.
- When referencing code, use `file:line_number` format.
- Prefer editing existing files over creating new ones.
- Do NOT add docstrings, comments, or type annotations to code that wasn't changed.
