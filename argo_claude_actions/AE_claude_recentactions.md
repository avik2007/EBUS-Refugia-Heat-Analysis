# Claude Recent Actions — ArgoEBUSAnalysis

---

## 2026-03-20 — Session 3 (physics history plots + AEResults path fix + repo hygiene)

**What was done:**

1. **Upgraded `plot_physics_history()` in `argoebus_gp_physics.py`**
   - Added `save_dir` and `run_id` parameters so figures are saved to disk headlessly
   - Added 4th subplot: Anisotropy Ratio (Lat_Scale / Lon_Scale) over time
   - Split the single 4-panel figure into 4 individual PNGs (one per metric) for easier viewing
   - Each figure saved as `{metric}_{run_id}.png` under `{save_dir}/`

2. **Created `03b_ae_plot_physics.py`** — standalone headless script
   - Loads existing audit CSV from `AEResults/aelogs/`
   - Calls `plot_physics_history()` to regenerate all 4 physics PNGs without re-running kriging
   - Useful for re-styling figures after the expensive analysis run is complete

3. **Moved `AEResults/` to correct location**
   - Was erroneously inside `ArgoEBUSCloud/AEResults/`; moved to `ArgoEBUSAnalysis/AEResults/` per `ae_file_structure.txt`
   - Fixed all path references in `03_ae_inspect_data.py`, `03b_ae_plot_physics.py`, and `ae_utils.py` (`ensure_ae_dirs`) to use `../AEResults/` from inside `ArgoEBUSCloud/`

4. **Renamed `claude_reports/` → `argo_claude_actions/`** with `AE_` file prefixes
   - `claude_lessons.md` → `AE_claude_lessons.md`
   - `claude_recentactions.md` → `AE_claude_recentactions.md`
   - `claude_todo.md` → `AE_claude_todo.md`

5. **Updated `CLAUDE.md`**
   - Added `## Hard-Won Rules` section at the bottom
   - Added `### 7. Python Environment` rule (always use `ebus-cloud-env`)
   - Updated Self-Improvement Loop (§3) to require dual updates: both `AE_claude_lessons.md` and `CLAUDE.md`
   - Added rule: `AEResults/` lives at `ArgoEBUSAnalysis/`, not inside `ArgoEBUSCloud/`

6. **Updated `ae_file_structure.txt`** — added `aelogs/` entry under `AEResults/`

**Verification result:** Physics PNGs (anisotropy, noise, length_scales, z_score) saved successfully for 2015 Skin Layer run.

---

## 2026-03-20 — Session 2 (get_float_history + float trajectory plot)

**What was done:**

1. **Added `get_float_history()` to `ae_utils.py`** (after `get_ae_config`)
   - Queries ERDDAP for raw per-dive float positions using `&distinct()` to collapse per-pressure rows
   - Returns 5-column DataFrame: `platform_number, lat, lon, time, time_days`
   - `time_days` uses same 1999-01-01 baseline as the OHC parquet

2. **Created `03_ae_plot_float_paths.py`** — standalone diagnostic script
   - Calls `get_float_history()`, prints shape and unique float count
   - Draws spaghetti map: each float as a distinct colored line over Cartopy basemap (PlateCarree, LAND + COASTLINE style matching `argoebus_gp_physics.py`)
   - Saves directly to `AEResults/aeplots/` (not inside a snapshot subfolder)
   - Filename mirrors `run_id`: `float_path_traj_california_20150101_20151231_res0_5x0_5_t30_0_d0_100.png`

3. **Filename correction** — initial attempt used a `day5844_6208` suffix; reverted to `run_id` suffix on user instruction (run_id already encodes dates, resolution, and depth cleanly)

4. **Updated `AE_claude_todo.md`** — marked both old Priority 1 tasks complete; promoted old Priority 4 (RMSRE Optimization) to Priority 1; Priority 2 and 3 unchanged

**Verification result:** 4,348 rows, 99 unique floats, figure saved to correct path.

**Bugs fixed during verification:**
- ERDDAP domain migration `www.ifremer.fr` → `erddap.ifremer.fr` (HTTP 302 → 400 chain); switched to `requests` with manual `%3C`/`%3E` percent-encoding to satisfy Tomcat 11 RFC 3986 strictness
- Matplotlib `cm.get_cmap` deprecation (→ `matplotlib.colormaps.get_cmap(...).resampled(...)`)

---

## 2026-03-19 — Session 1 (Onboarding)

**What was done:**

1. **Read the full codebase** — explored all Python files in `ArgoEBUSCloud/`, notebooks 1–5, helper modules
2. **Read the Gemini discussion log** (`Gemini-Testing Ocean Refugia Hypothesis March 19, 2026.txt`)
   - ~44,760 lines covering the full arc of scientific development with Gemini
   - Key science outcome: "Stealth Warming" 3-layer vertical fingerprinting study design
   - Identified the complete agenda of work still to be done
3. **Read the proposed CLAUDE.md** (`proposal_for_claudemd.txt`) and adapted it to this project
4. **Created `CLAUDE.md`** in the project root with:
   - Project mission statement (Ocean Refugia / Stealth Warming)
   - Workflow principles from the proposal
   - Scientific context (metrics, depth layer naming, key functions)
   - Repository structure map
5. **Created `claude_reports/` folder** with three files:
   - `claude_todo.md` — full agenda derived from Gemini session, prioritized
   - `claude_recentactions.md` — this file
   - `claude_lessons.md` — initially empty (no mistakes yet to document)
6. **Saved persistent memories** to `~/.claude/projects/.../memory/`

**Key scientific state understood:**
- 2015 Skin Layer (0–100m) is DONE: 23 snapshots, audit CSV, CV pickle all saved to S3/local
- Anisotropy Ratio in Skin Layer: ~0.36 (winter) → ~0.49 (summer) — atmospheric/zonal dominance
- Next critical run: Source Layer (150–400m) cloud job via Script 02
- RMSRE currently ~8%, target is <5%

**Nothing was modified in the codebase** — this was a read/onboard session only.

---

_Add new entries at the top after each session._
