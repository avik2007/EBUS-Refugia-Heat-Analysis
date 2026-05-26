# MLOps Audit Gaps 2–5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the four remaining Gemini MLOps audit gaps: depth cross-validation (Gap 3), backfill metadata transparency (Gap 5), centralize fmt_dec (Gap 4), and registry status field (Gap 2).

**Architecture:** All changes are isolated to `ebus_core/` — no changes to pipeline scripts or configs. Tasks 1–3 are independent and can be done in any order. Task 4 (registry status) touches runner.py last so it doesn't conflict with Task 3's runner.py refactor.

**Tech Stack:** Python 3.x, Pydantic v2, PyYAML, pytest

---

## Task 1: Gap 3 — Physics depth cross-validation

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/config_schema.py` (add validator after `_non_legacy_complete`, around line 566)
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py` (add tests after `test_physics_params_depth_ordering`, around line 213)

- [ ] **Step 1: Write the failing tests**

Add these two tests immediately after the existing `test_physics_params_depth_ordering` block (around line 230):

```python
def test_analysis_config_ohc_bot_exceeds_depth_range():
    # ohc_depth_bot_m > depth_range[1] must raise ValidationError
    from pydantic import ValidationError
    kwargs = _valid_analysis_kwargs()
    # depth_range is (150, 400); set ohc_depth_bot_m to 500 — outside range
    kwargs["physics_params"] = {"ohc_depth_bot_m": 500}
    with pytest.raises(ValidationError, match="ohc_depth_bot_m"):
        AnalysisConfig(**kwargs)


def test_analysis_config_ohc_top_below_depth_range():
    # ohc_depth_top_m < depth_range[0] must raise ValidationError
    from pydantic import ValidationError
    kwargs = _valid_analysis_kwargs()
    # depth_range is (150, 400); set ohc_depth_top_m to 100 — above the layer
    kwargs["physics_params"] = {"ohc_depth_top_m": 100}
    with pytest.raises(ValidationError, match="ohc_depth_top_m"):
        AnalysisConfig(**kwargs)


def test_analysis_config_ohc_bounds_within_depth_range_valid():
    # ohc bounds within depth_range must parse cleanly
    kwargs = _valid_analysis_kwargs()
    kwargs["physics_params"] = {"ohc_depth_top_m": 150, "ohc_depth_bot_m": 400}
    cfg = AnalysisConfig(**kwargs)
    assert cfg.physics_params.ohc_depth_top_m == 150
    assert cfg.physics_params.ohc_depth_bot_m == 400
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py \
    -k "ohc_bot_exceeds or ohc_top_below or ohc_bounds_within" -v
```

Expected: FAIL — `ValidationError` not raised (validator doesn't exist yet).

- [ ] **Step 3: Add the validator to AnalysisConfig**

In `config_schema.py`, insert this model_validator directly after `_non_legacy_complete` (after line 565, before `_dates_ordered`):

```python
    @model_validator(mode="after")
    def _physics_depth_within_range(self) -> "AnalysisConfig":
        # Cross-validate PhysicsParamsBlock OHC integration bounds against depth_range.
        # ohc_depth_top_m must be >= depth_range[0]: the OHC integration cannot start
        # above the analysis layer (would include water we didn't model).
        # ohc_depth_bot_m must be <= depth_range[1]: similarly cannot integrate below
        # the analysis layer. When either field is None the runtime inherits the value
        # from depth_range, so there is nothing to cross-validate in that case.
        # Input: AnalysisConfig after all field validators and prior model validators.
        # Output: self unchanged if valid.
        # Raises: pydantic.ValidationError (wraps ValueError) if bounds violate depth_range.
        p = self.physics_params
        d0, d1 = self.depth_range
        if p.ohc_depth_top_m is not None and p.ohc_depth_top_m < d0:
            raise ValueError(
                f"ohc_depth_top_m ({p.ohc_depth_top_m}) must be >= depth_range[0] "
                f"({d0}); OHC integration cannot start above the analysis layer."
            )
        if p.ohc_depth_bot_m is not None and p.ohc_depth_bot_m > d1:
            raise ValueError(
                f"ohc_depth_bot_m ({p.ohc_depth_bot_m}) must be <= depth_range[1] "
                f"({d1}); OHC integration cannot extend below the analysis layer."
            )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py \
    -k "ohc_bot_exceeds or ohc_top_below or ohc_bounds_within" -v
```

Expected: 3 PASS.

- [ ] **Step 5: Run full suite to verify no regressions**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py -q
```

Expected: all tests pass (count increases by 3).

- [ ] **Step 6: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/config_schema.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "fix(schema): cross-validate ohc_depth bounds against depth_range (Gap 3)"
```

---

## Task 2: Gap 5 — Backfill metadata transparency

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/config_schema.py` (add `BackfillMetadataBlock`, add field to `AnalysisConfig`)
- Modify: `ArgoEBUSCloud/ebus_core/backfill.py` (populate `backfill_metadata` in `backfill_configs`)
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py` (add tests after the backfill round-trip test)

- [ ] **Step 1: Write the failing tests**

Find `test_backfilled_configs_round_trip` in the test file and add these new tests directly after it:

```python
def test_backfill_metadata_recovered_fields_present(tmp_path):
    # backfill_configs must write a backfill_metadata block with recovered_fields
    # listing fields extracted from the run_id and suffix.
    import csv as csv_mod
    from ebus_core.backfill import backfill_configs
    from ebus_core.config_schema import load_config

    aelogs = tmp_path / "aelogs"
    run_id = "california_20150101_20151231_res0_5x0_5_t30_0_d0_100_3dmatern_w45"
    run_dir = aelogs / run_id
    run_dir.mkdir(parents=True)
    # Write a minimal audit CSV with noise_val so noise_vals_audit is recovered
    audit = run_dir / f"audit_{run_id}.csv"
    with audit.open("w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["noise_val"])
        w.writerow([0.001])

    configs_root = tmp_path / "configs"
    backfill_configs(aelogs, configs_root)

    cfg = load_config(configs_root / "california" / f"{run_id}.yaml")
    assert cfg.backfill_metadata is not None
    # Fields parsed from the run_id must appear in recovered_fields
    for field in ["region", "date_start", "date_end", "lat_step", "lon_step",
                  "time_step", "depth_range"]:
        assert field in cfg.backfill_metadata.recovered_fields, \
            f"{field} missing from recovered_fields"


def test_backfill_metadata_assumed_fields_present(tmp_path):
    # When mode/kernel_type/window_size_days are not in the suffix they are
    # assumed defaults; they must appear in assumed_fields, not recovered_fields.
    from ebus_core.backfill import backfill_configs
    from ebus_core.config_schema import load_config

    aelogs = tmp_path / "aelogs"
    # Suffix has no 2d/rbf/w{N} tokens — all GPR fields default
    run_id = "california_20150101_20151231_res0_5x0_5_t30_0_d0_100"
    run_dir = aelogs / run_id
    run_dir.mkdir(parents=True)

    configs_root = tmp_path / "configs"
    backfill_configs(aelogs, configs_root)

    cfg = load_config(configs_root / "california" / f"{run_id}.yaml")
    assert cfg.backfill_metadata is not None
    # mode/kernel_type/window_size_days fell back to defaults — must be assumed
    for field in ["gpr.mode", "gpr.kernel_type", "gpr.window_size_days"]:
        assert field in cfg.backfill_metadata.assumed_fields, \
            f"{field} missing from assumed_fields"
    # run_id-derived fields must NOT appear in assumed
    assert "region" not in cfg.backfill_metadata.assumed_fields
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py \
    -k "backfill_metadata" -v
```

Expected: FAIL — `cfg.backfill_metadata` is None (field doesn't exist yet).

- [ ] **Step 3: Add BackfillMetadataBlock to config_schema.py**

In `config_schema.py`, add this class immediately before `AnalysisInputBlock` (around line 381):

```python
class BackfillMetadataBlock(BaseModel):
    """
    Provenance record written by the backfill script into every legacy config.

    recovered_fields: fields whose values were parsed directly from the run_id
        string or audit CSV — these are known to be accurate.
    assumed_fields: fields that fell back to pipeline defaults because the
        run_id / suffix contained no explicit value — these could differ from
        what was actually used in the original run.

    The distinction prevents audit readers from treating all backfilled values
    equally: recovered fields are trustworthy, assumed fields carry uncertainty.
    Unrecoverable fields (noise_val, *_ls_bounds) are written as null in the
    config and do not appear in either list.
    """

    model_config = ConfigDict(extra="forbid")

    recovered_fields: List[str] = Field(default_factory=list)
    assumed_fields: List[str] = Field(default_factory=list)
```

Then add the field to `AnalysisConfig` (after the `legacy_backfill: bool = False` line, around line 479):

```python
    # backfill_metadata: populated by 10_ae_backfill_configs.py to record which
    # fields were recovered from the run_id vs assumed from pipeline defaults.
    # None for configs written by hand or by the MLOps runner (not backfilled).
    backfill_metadata: Optional[BackfillMetadataBlock] = None
```

- [ ] **Step 4: Populate backfill_metadata in backfill.py**

In `backfill_configs()` (around line 225 in backfill.py), after `gpr_recoverable = _parse_suffix(...)`, add this block to compute the metadata:

```python
        # Determine which GPR fields were explicitly recovered from the suffix
        # vs which fell back to pipeline defaults. This feeds backfill_metadata.
        sfx_lower = run_suffix.lower()
        gpr_recovered = []
        gpr_assumed = []

        # mode: explicit only if "2d" is present (otherwise 3D is assumed)
        if "2d" in sfx_lower:
            gpr_recovered.append("gpr.mode")
        else:
            gpr_assumed.append("gpr.mode")

        # kernel_type: explicit only if "rbf" is present
        if "rbf" in sfx_lower:
            gpr_recovered.append("gpr.kernel_type")
        else:
            gpr_assumed.append("gpr.kernel_type")

        # window_size_days: explicit only if w{N} pattern present
        if re.search(r"w\d+", sfx_lower):
            gpr_recovered.append("gpr.window_size_days")
        else:
            gpr_assumed.append("gpr.window_size_days")

        # min_bins: explicit only if minbins{N} pattern present
        if re.search(r"minbins\d+", sfx_lower):
            gpr_recovered.append("gpr.min_bins")
        else:
            gpr_assumed.append("gpr.min_bins")

        # step_size_days: explicit only if t{X}s{N} pattern present
        if re.search(r"t\d+s\d+", sfx_lower):
            gpr_recovered.append("gpr.step_size_days")
        else:
            gpr_assumed.append("gpr.step_size_days")

        # noise_vals_audit: recovered if audit CSV existed and had noise_val column
        if noise_vals is not None:
            gpr_recovered.append("gpr.noise_vals_audit")

        # All run_id-derived fields are always recovered
        run_id_recovered = [
            "region", "date_start", "date_end",
            "lat_step", "lon_step", "time_step", "depth_range",
        ]
        all_recovered = run_id_recovered + gpr_recovered
```

Then add `backfill_metadata` to the `yaml_dict` (after the `"description"` key, before the closing brace):

```python
            "backfill_metadata": {
                "recovered_fields": all_recovered,
                "assumed_fields": gpr_assumed,
            },
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py \
    -k "backfill_metadata" -v
```

Expected: 2 PASS.

- [ ] **Step 6: Run full suite**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py -q
```

Expected: all tests pass (count increases by 2).

- [ ] **Step 7: Re-run backfill to update the 18 existing YAML configs**

```bash
conda run -n ebus-cloud-env python ArgoEBUSCloud/10_ae_backfill_configs.py
```

Expected: 18 YAML files overwritten with `backfill_metadata` block.

Spot-check one:
```bash
grep -A 10 "backfill_metadata" configs/california/california_20150101_20151231_res0_5x0_5_t30_0_d0_100_3dmatern_w45.yaml
```

Expected: `recovered_fields` and `assumed_fields` lists present.

- [ ] **Step 8: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/config_schema.py ArgoEBUSCloud/ebus_core/backfill.py \
    ArgoEBUSCloud/test_mlops_foundation.py configs/
git commit -m "feat(schema): backfill_metadata block records recovered vs assumed fields (Gap 5)"
```

---

## Task 3: Gap 4 — Centralize fmt_dec in ae_utils

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/ae_utils.py` (add `fmt_dec` at bottom of file, before `calculate_bin`)
- Modify: `ArgoEBUSCloud/ebus_core/runner.py` (import `fmt_dec` from ae_utils, delete local `_fmt_dec`)
- Modify: `ArgoEBUSCloud/ebus_core/backfill.py` (replace inline `fmt` in `_infer_s3_path` with `fmt_dec`)
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py` (add import test)

Note: `derive_run_id` stays in `runner.py` — moving it to `ae_utils` would create a circular import because `config_schema.py` (which runner imports) already imports from `ae_utils`.

- [ ] **Step 1: Write the failing test**

Add this test to `test_mlops_foundation.py` near the top with other import tests:

```python
def test_fmt_dec_importable_from_ae_utils():
    # fmt_dec must be importable directly from ae_utils (Gap 4 — centralize formatting)
    from ebus_core.ae_utils import fmt_dec
    assert fmt_dec(0.5) == "0_5"
    assert fmt_dec(10.0) == "10_0"
    assert fmt_dec(0.25) == "0_25"
    assert fmt_dec(30.0) == "30_0"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py \
    -k "test_fmt_dec_importable_from_ae_utils" -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add fmt_dec to ae_utils.py**

In `ae_utils.py`, add this function immediately before `calculate_bin` (around line 359):

```python
def fmt_dec(x: float) -> str:
    # Convert a float to a filesystem-safe string by replacing '.' with '_'.
    # Used to build canonical run_id strings and S3 parquet paths.
    # Examples: 0.5 -> '0_5', 10.0 -> '10_0', 0.25 -> '0_25'
    # WHY: dots in directory/filename components confuse shell globs and some
    # path-parsing utilities. All AEResults/ paths embed underscored floats.
    s = f"{x:g}"
    if "." not in s:
        s = s + ".0"
    return s.replace(".", "_")
```

- [ ] **Step 4: Update runner.py to import fmt_dec and drop local _fmt_dec**

At the top of `runner.py`, change:
```python
from ebus_core.config_schema import AnalysisConfig, IngestionConfig
```
to:
```python
from ebus_core.ae_utils import fmt_dec
from ebus_core.config_schema import AnalysisConfig, IngestionConfig
```

In `derive_run_id`, change every call from `_fmt_dec(...)` to `fmt_dec(...)`:
```python
    lat = fmt_dec(cfg.lat_step)
    lon = fmt_dec(cfg.lon_step)
    t = fmt_dec(cfg.time_step)
```

Delete the entire `_fmt_dec` function (lines 85–101):
```python
# DELETE THIS ENTIRE FUNCTION:
def _fmt_dec(x: float) -> str:
    ...
```

- [ ] **Step 5: Update backfill.py _infer_s3_path to use fmt_dec**

In `backfill.py`, add import at top of file:
```python
from ebus_core.ae_utils import fmt_dec
```

In `_infer_s3_path`, delete the inline `def fmt(v)` closure and replace the call sites. Current code:
```python
    def fmt(v: float) -> str:
        return str(v).replace(".", "_")
    ...
    f"_res{fmt(lat_step)}x{fmt(lon_step)}"
    f"_t{fmt(time_step)}_d{d0}_{d1}"
```

Replace with:
```python
    # (remove the local def fmt line entirely)
    ...
    f"_res{fmt_dec(lat_step)}x{fmt_dec(lon_step)}"
    f"_t{fmt_dec(time_step)}_d{d0}_{d1}"
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py \
    -k "test_fmt_dec_importable_from_ae_utils" -v
```

Expected: PASS.

- [ ] **Step 7: Run full suite**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py -q
```

Expected: all tests pass (count increases by 1).

- [ ] **Step 8: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/ae_utils.py ArgoEBUSCloud/ebus_core/runner.py \
    ArgoEBUSCloud/ebus_core/backfill.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "refactor: centralize fmt_dec in ae_utils; runner + backfill import from there (Gap 4)"
```

---

## Task 4: Gap 2 — Registry status field

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/manifest.py` (add `"status"` to `_REGISTRY_FIELDS`, update `append_registry` signature)
- Modify: `ArgoEBUSCloud/ebus_core/runner.py` (determine status after dispatch, pass to `append_registry`)
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py` (add two status tests)

- [ ] **Step 1: Write the failing tests**

Add these tests after `test_run_analysis_omits_spatial_ls_upper_bound_for_legacy` (the Gap 1 tests):

```python
def test_run_analysis_registry_status_finalized_when_audit_exists(tmp_path, monkeypatch):
    # When dispatch returns a path that exists on disk, registry status = "finalized"
    import json as _json
    kwargs = _valid_analysis_kwargs()
    kwargs["outputs"] = {
        "aelogs_dir": str(tmp_path / "aelogs"),
        "aeplots_dir": str(tmp_path / "aeplots"),
        "generate_snapshots": False,
        "generate_physics_plots": False,
    }
    cfg = AnalysisConfig(**kwargs)

    def fake_dispatch(**kw):
        run_id = derive_run_id(cfg)
        out_dir = tmp_path / "aelogs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        audit = out_dir / f"audit_{run_id}.csv"
        audit.write_text("dummy,csv\n")
        return {"audit_csv": str(audit)}

    monkeypatch.setattr("ebus_core.runner._call_run_diagnostic_inspection", fake_dispatch)
    run_analysis(cfg, registry_path=tmp_path / "registry.jsonl")

    line = _json.loads((tmp_path / "registry.jsonl").read_text().strip())
    assert line["status"] == "finalized"


def test_run_analysis_registry_status_incomplete_when_audit_missing(tmp_path, monkeypatch):
    # When dispatch returns a path that does NOT exist, registry status = "incomplete"
    import json as _json
    kwargs = _valid_analysis_kwargs()
    kwargs["outputs"] = {
        "aelogs_dir": str(tmp_path / "aelogs"),
        "aeplots_dir": str(tmp_path / "aeplots"),
        "generate_snapshots": False,
        "generate_physics_plots": False,
    }
    cfg = AnalysisConfig(**kwargs)

    def fake_dispatch(**kw):
        run_id = derive_run_id(cfg)
        out_dir = tmp_path / "aelogs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        # Return a path that does NOT exist — simulates a partial/crashed run
        return {"audit_csv": str(out_dir / "ghost_audit.csv")}

    monkeypatch.setattr("ebus_core.runner._call_run_diagnostic_inspection", fake_dispatch)
    run_analysis(cfg, registry_path=tmp_path / "registry.jsonl")

    line = _json.loads((tmp_path / "registry.jsonl").read_text().strip())
    assert line["status"] == "incomplete"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py \
    -k "registry_status" -v
```

Expected: FAIL — `line["status"]` raises `KeyError` (status field doesn't exist yet).

- [ ] **Step 3: Update manifest.py**

In `manifest.py`, update `_REGISTRY_FIELDS` (line 240):

```python
_REGISTRY_FIELDS = (
    "run_id", "kind", "config_hash", "created_at",
    "region", "depth_range", "manifest_path", "status",
)
```

Update `append_registry` signature to accept `status`:

```python
def append_registry(
    manifest: Dict[str, Any],
    registry_path: Path,
    manifest_path: Path,
    status: str = "finalized",
) -> None:
```

Add `"status"` to the `line` dict (after `"manifest_path"`):

```python
    line = {
        "run_id": manifest["run_id"],
        "kind": manifest["kind"],
        "config_hash": manifest["config_hash"],
        "created_at": manifest["created_at"],
        "region": manifest["config"].get("region"),
        "depth_range": manifest["config"].get("depth_range"),
        "manifest_path": str(manifest_path),
        "status": status,
    }
```

- [ ] **Step 4: Update runner.py to determine and pass status**

In `run_analysis()` in `runner.py`, after the dispatch block (after the `audit_csv = ...` assignment, around line 229), add:

```python
    # Determine run completeness: "finalized" if the audit CSV landed on disk,
    # "incomplete" if the dispatch returned but outputs are missing. The latter
    # can happen when a Dask job crashes after returning a partial result dict.
    # Writing "incomplete" to the registry prevents silent ghost-success entries.
    status = (
        "finalized"
        if audit_csv and Path(audit_csv).exists()
        else "incomplete"
    )
```

Then update the `append_registry` call (near the end of `run_analysis`) to pass `status`:

```python
    if registry_path is not None:
        append_registry(manifest, registry_path, manifest_path, status=status)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py \
    -k "registry_status" -v
```

Expected: 2 PASS.

- [ ] **Step 6: Run full suite**

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py -q
```

Expected: all tests pass (count increases by 2).

- [ ] **Step 7: Commit**

```bash
git add ArgoEBUSCloud/ebus_core/manifest.py ArgoEBUSCloud/ebus_core/runner.py \
    ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(registry): add status field (finalized/incomplete) to registry entries (Gap 2)"
```

---

## Final: Push and open PR

- [ ] **Push branch**

```bash
git push origin main
```

Or, if working on a feature branch:

```bash
git push origin <branch-name>
```

- [ ] **Verify test count**

Final expected test count: 46 (current) + 3 (Gap 3) + 2 (Gap 5) + 1 (Gap 4) + 2 (Gap 2) = **54 tests**.

```bash
conda run -n ebus-cloud-env python -m pytest ArgoEBUSCloud/test_mlops_foundation.py -q
```
