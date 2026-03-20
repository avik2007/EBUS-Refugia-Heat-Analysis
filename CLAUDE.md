# CLAUDE.md — ArgoEBUSAnalysis Project Guide

## Project Mission
Testing the "Ocean Refugia / Stealth Warming" hypothesis: subsurface warming in the California
Current System transported by the California Undercurrent that has not yet surfaced.

Three depth layers via Argo float GPR (Kriging):
- **Skin Layer** (0–100m): atmospheric forcing dominates
- **Source Layer** (150–400m): Ekman upwelling source water — where "stealth heat" hides
- **Background Layer** (500–1000m): deep ocean baseline

Signal: Source Layer warming faster than Background.

---

## Workflow Principles

1. **Plan mode** for any task with 3+ steps or architectural decisions. Re-plan if derailed.
2. **Subagents** for research, exploration, parallel analysis. One task per subagent.
3. **Self-improvement**: after any correction, update `AE_claude_lessons.md` AND this file.
4. **Verification**: never mark done without running the code and confirming output.
5. **Elegance check**: for non-trivial changes, ask "is there a more elegant way?"
6. **Autonomous bug fixing**: identify logs/errors, resolve without hand-holding.

---

## Scientific Context

- **Gemini**: science planning and hypothesis. **Claude**: implementation and debugging. Don't second-guess the science; flag implementation risks.
- **Anisotropy Ratio** = `Lat_Scale / Lon_Scale`. < 1.0 = zonal/atm forcing. > 1.0 = meridional current. Should increase with depth.
- **RMSRE** target < 5%. **Std Z-Score** ideal = 1.0.
- **run_id naming**: `california_20150101_20151231_res0_5x0_5_t30_0_d0_100` — `d0_100` = Skin, `d150_400` = Source, `d500_1000` = Background.

---

## Repository Structure

See `ae_file_structure.txt` for the full layout.

Key directories:
- `ArgoEBUSCloud/ebus_core/` — core library (config, thermodynamics, GPR, plotting)
- `AEResults/aeplots/`, `aelogs/` — output PNGs, audit CSVs, CV pickles
- `argo_claude_actions/` — todo, recent actions, lessons

---

## Style Preferences

- Keep responses concise and direct.
- No trailing summaries of what was just done.
- No emojis unless the user uses them.
- When referencing code, use `file:line_number` format.
- Prefer editing existing files over creating new ones.
- Do NOT add docstrings, comments, or type annotations to code that wasn't changed.
- **Always write verbose comments for any code you write or modify.** Every function must have a header comment explaining what it does, why it exists, what inputs mean physically, and what the output represents. Inline comments should explain non-obvious logic step by step. Code can be brief; comments should be thorough. The goal is that a human reading the code cold can follow every decision.

---

## Hard-Won Rules

**Always use `ebus-cloud-env`**: `conda run -n ebus-cloud-env python <script>`. Base env lacks all packages.

**Pipeline tools share a common signature**: every analysis/diagnostic function must be importable with `(region, lat_step, lon_step, time_step, depth_range)` — matching `run_diagnostic_inspection()`. No module-level constants. Enables serial calls across regions/layers in one script.

**Completed tasks leave `AE_claude_todo.md`**: record them in `AE_claude_recentactions.md`. Todo = forward-looking only.

**`AEResults/` is at `ArgoEBUSAnalysis/`**, not inside `ArgoEBUSCloud/`. Paths must traverse up: `os.path.join(base_dir, "..", "AEResults", ...)`.
