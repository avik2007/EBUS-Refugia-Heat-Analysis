# Claude Lessons Learned — ArgoEBUSAnalysis

This file records mistakes made during this project and the rules derived from them.

---

## Format

Each entry follows this structure:
- **Mistake**: What went wrong
- **Rule**: The corrective behavior going forward
- **Why**: The reason this matters for this project

---

## Lessons

### 1. Always activate ebus-cloud-env before running code

- **Mistake**: Attempted to run project scripts with the base conda environment active, which lacks required packages (pandas, sklearn, etc.), causing `ModuleNotFoundError`.
- **Rule**: Before running ANY project code, use `conda run -n ebus-cloud-env python <script>` or confirm the environment is already activated. Never assume the base environment has the required packages.
- **Why**: The base conda environment is bare. All scientific dependencies (pandas, scikit-learn, xarray, gsw, etc.) are pinned inside `ebus-cloud-env`.

### 2. AEResults lives at ArgoEBUSAnalysis/AEResults/, not inside ArgoEBUSCloud/

- **Mistake**: Scripts and the new `03b_ae_plot_physics.py` initially pointed at `ArgoEBUSCloud/AEResults/` because `base_dir` was the script's own directory with no parent traversal.
- **Rule**: `AEResults/` is a sibling of `ArgoEBUSCloud/`, one level up at `ArgoEBUSAnalysis/AEResults/`. Any path construction must include `".."` to escape `ArgoEBUSCloud/`. The canonical pattern is `os.path.join(base_dir, "..", "AEResults", ...)`.
- **Why**: The file structure (`ae_file_structure.txt`) defines `AEResults/` at the `ArgoEBUSAnalysis/` level. Putting outputs inside `ArgoEBUSCloud/` mixes code and data, breaks the intended layout, and can cause silent writes to the wrong location.

### 3. Pipeline diagnostic tools must share a common function signature

- **Mistake**: `03_ae_plot_float_paths.py` was built as a standalone script with module-level constants (`REGION`, `START_DATE`, etc.) and a `main()` function. To run it for a different region or depth, the user had to edit constants inside the file — making it impossible to call alongside `run_diagnostic_inspection()` in a parent script.
- **Rule**: Every pipeline tool (diagnostic plots, float trajectories, physics history, etc.) must be a proper importable function whose signature matches the other pipeline tools it will be used alongside. At minimum that means `(region, lat_step, lon_step, time_step, depth_range)` for any analysis function. Module-level constants are a red flag. The `if __name__ == "__main__"` block is fine for standalone use, but the real interface is the function.
- **Why**: The goal is to run California and Humboldt (or Skin and Source layers) in the same script by calling the functions serially with different arguments — not by editing internal parameters and running the same script multiple times.

### 4. GPRBlock does not have spatial_ls_upper_bound — kwargs drift between config schema and script signature

- **Mistake**: Plan's `run_analysis` dispatch_kwargs included `spatial_ls_upper_bound: cfg.gpr.spatial_ls_upper_bound`. That field was removed from GPRBlock in §A.2 (replaced by split `lat_ls_bounds` / `lon_ls_bounds`), so accessing it raises `AttributeError` at runtime.
- **Rule**: Before writing dispatch_kwargs that reference config fields, grep the actual model definition (`config_schema.py`) to confirm the field exists. Do not trust plan code verbatim — the schema evolves independently.
- **Why**: The plan was written before §A.2 split-anisotropy amendment. Schema and plan drifted. The runner shim passes ALL config fields to the shim (which filters for production), but the config fields must exist first.

### 5. REMINDER: Use statistical tests to gauge progress

- **Mistake**: Risk of comparing kernel/pipeline variants (e.g. Gibbs vs Matern 5/2 RMSRE) by eyeballing summary-table deltas (median/max RMSRE) without a significance test — rolling windows overlap (`step_size_days` < `window_size_days`), so per-window RMSRE values are autocorrelated, and a naive paired t-test/Wilcoxon would overstate confidence anyway.
- **Rule**: Before claiming a variant "improved" or "regressed" results, run a proper paired significance test on matched `window_center` rows — Diebold-Mariano test with HAC (Newey-West) variance correction on the RMSRE differential, not a raw comparison of summary statistics.
- **Why**: User reminder (2026-07-07), prompted by prep for a statistical-significance question during an interview. Applies to any future kernel/parameter comparison, not just Gibbs vs Matern.

### 6. GibbsKernel's time_ls was defined in days but consumed against a normalized time column

- **Mistake**: `GibbsKernel.__call__` (`argoebus_gp_physics.py`) divided `dt` — the pairwise time difference from `X`'s time column — directly by `self._time_ls`. `time_ls_init_days`/`time_ls_bounds_days` are documented, recorded (audit CSV `time_ls_days`), and reported as physical days, but `X`'s time column is window-normalized to `[-1, +1]` (`time_scaled = (t - window_center) / half_window` in `analyze_rolling_correlations`). No conversion between the two ever happened. The Matérn path does this conversion correctly (`ls_time_scaled = scale_time_bin_days / half_window`) — comparing the two paths (Phase 2 pattern analysis) is what surfaced the missing line. Net effect: the Gibbs temporal kernel barely decayed regardless of the fitted `time_ls` value (verified empirically: LML sensitivity to `time_ls` was ~12x weaker than the correctly-scaled Matérn equivalent over the same sweep). This directly undermined the "temporal persistence increases with depth" claim from session 17 (44d→54d→58d) — those numbers were real optimizer outputs, but the kernel's actual decay behavior barely reflected them.
- **Rule**: When a kernel/model hyperparameter has physical units (days, km) but operates on a normalized/scaled feature column, grep every reference to that hyperparameter across the file and verify the conversion between the two unit systems actually exists in the code that consumes it — don't assume it does because the units are named correctly in the docstring and constructor. Cross-check against a working sibling implementation (here, the Matérn time-dimension scaling) as a concrete pattern to diff against.
- **Why**: User caught this during a request for a LinkedIn illustration plot, after independently tracing `GibbsKernel.__call__` line by line while reconstructing the real fitted kernel (not the RBF-proxy `plot_kriging_snapshot` uses). Fixed in `GibbsKernel` (added `window_size_days` constructor param, converts `dt` to days before dividing by `time_ls`; see `test_gibbs_kernel_time_ls_converts_normalized_dt_to_days`). Re-running the fix surfaced a second, separate finding: `time_ls` is not identifiable within a 45-day window regardless of bound (tested up to 200d, all layers still mostly peg at the ceiling) — a window-design limitation, not a units bug. Report `time_ls_days` as a floor ("≥200d, unresolvable within this window") for all three californiav3 Gibbs layers until window design is revisited; the depth-trend claim from session 17 no longer holds and should not be reused.

---

_Update this file immediately after any user correction._
