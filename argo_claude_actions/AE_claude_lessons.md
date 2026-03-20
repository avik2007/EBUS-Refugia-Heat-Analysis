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

---

_Update this file immediately after any user correction._
