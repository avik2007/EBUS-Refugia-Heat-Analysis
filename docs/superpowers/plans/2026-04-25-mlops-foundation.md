# MLOps Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-driven CLI + reproducibility manifest layer atop the existing ArgoEBUSAnalysis pipeline. No existing script touched. Every run produces a `manifest.json` with config hash, git SHA, conda env, S3 lineage; every run validated against a strict Pydantic schema.

**Architecture:** Three new library modules in `ebus_core/` (`config_schema.py`, `manifest.py`, `runner.py`) wrapped by an `aebus` CLI in `ArgoEBUSCloud/aebus_cli.py`. YAML configs live under `configs/<region>/`. Manifests written next to existing `AEResults/aelogs/{run_id}/` outputs. Append-only registry at `AEResults/run_registry.jsonl`.

**Tech Stack:** Python 3.11+, Pydantic v2, PyYAML, argparse, pytest. Conda env `ebus-cloud-env`. All commands run via `conda run -n ebus-cloud-env <cmd>` per project rule.

**Spec:** `docs/superpowers/specs/2026-04-25-mlops-foundation-design.md`

---

## Conventions for the Engineer

- **Always run Python via the conda env:** `conda run -n ebus-cloud-env python <args>`. The base env lacks all packages.
- **`AEResults/` is at `ArgoEBUSAnalysis/AEResults/`**, NOT inside `ArgoEBUSCloud/`. Paths from `ArgoEBUSCloud/` traverse up: `os.path.join("..", "AEResults", ...)`.
- **Existing canonical signature** for analysis functions: `(region, lat_step, lon_step, time_step, depth_range, ...)`. Wrappers in this plan do not change this signature; they unpack from configs and call.
- **Imports** use the `ebus_core` package: `from ebus_core.config_schema import AnalysisConfig`. The package is the existing `ArgoEBUSCloud/ebus_core/` directory.
- **Run tests from the repo root:** `cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v`.
- **Commit small, commit often.** Each task ends with a commit. Use the message exactly as given.

---

## Phase 0: Environment Setup

### Task 0.1: Install missing dependencies

**Files:**
- Modify: `ArgoEBUSCloud/ocean_cloud.yml` (add pydantic + pytest)
- Modify: `ocean_env_clean.yml` (same)

The conda env `ebus-cloud-env` currently lacks `pydantic` and `pytest`. Both are required for this plan.

- [ ] **Step 1: Verify the gap**

```bash
conda run -n ebus-cloud-env python -c "import pydantic" 2>&1 | tail -1
conda run -n ebus-cloud-env python -c "import pytest" 2>&1 | tail -1
```
Expected: `ModuleNotFoundError` for both.

- [ ] **Step 2: Install pydantic and pytest into the env**

```bash
conda run -n ebus-cloud-env pip install "pydantic>=2.0" "pytest>=7.0"
```

- [ ] **Step 3: Verify install**

```bash
conda run -n ebus-cloud-env python -c "import pydantic; print(pydantic.VERSION)"
conda run -n ebus-cloud-env python -c "import pytest; print(pytest.__version__)"
```
Expected: prints versions, no errors. Pydantic ≥ 2.0 (the v2 API is what this plan uses).

- [ ] **Step 4: Add to env spec files for reproducibility**

Open `ArgoEBUSCloud/ocean_cloud.yml`. Find the `dependencies:` list. Add the following entries under the existing `pip:` sub-list (creating a `pip:` block if absent):
```yaml
  - pip:
      - pydantic>=2.0
      - pytest>=7.0
```
Repeat for `ocean_env_clean.yml`.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/ocean_cloud.yml ocean_env_clean.yml
git commit -m "chore: add pydantic and pytest to ebus-cloud-env spec"
```

---

## Phase 1: Schema + Manifest + Hash

### Task 1.1: Create empty test file + IngestionConfig stub

**Files:**
- Create: `ArgoEBUSCloud/test_mlops_foundation.py`
- Create: `ArgoEBUSCloud/ebus_core/config_schema.py`

- [ ] **Step 1: Write the failing test**

Create `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
"""Tests for the MLOps foundation: config schema, manifest, runner, CLI."""
import datetime as dt

import pytest
from pydantic import ValidationError

from ebus_core.config_schema import IngestionConfig


def test_ingestion_config_valid_minimal():
    cfg = IngestionConfig(
        schema_version=1,
        config_kind="ingestion",
        region="californiav2",
        date_start=dt.date(2015, 1, 1),
        date_end=dt.date(2015, 12, 31),
        lat_step=0.5,
        lon_step=0.5,
        time_step=10.0,
        depth_range=(0, 100),
    )
    assert cfg.region == "californiav2"
    assert cfg.depth_range == (0, 100)
    assert cfg.cloud.backend == "coiled"  # default
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py::test_ingestion_config_valid_minimal -v
```
Expected: ImportError / ModuleNotFoundError on `ebus_core.config_schema`.

- [ ] **Step 3: Create `ebus_core/config_schema.py` with IngestionConfig**

Create `ArgoEBUSCloud/ebus_core/config_schema.py`:
```python
"""
Pydantic models for ArgoEBUS pipeline configs.

Two top-level configs exist: IngestionConfig (drives script-02 logic) and
AnalysisConfig (drives script-05/07 logic). Both validate strictly: unknown
fields raise. Cross-field rules (e.g., bin-aliasing prevention) live as
model_validators.

Schema is versioned via the `schema_version: int` field. Bumping requires
a documented migration in configs/README.md.
"""
import datetime as dt
from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ebus_core.ae_utils import get_ebus_registry


class CloudBlock(BaseModel):
    """Cloud-compute settings for ingestion runs."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["coiled", "local"] = "coiled"
    n_workers: int = Field(3, ge=1, le=64)
    worker_region: str = "us-east-1"


class S3Block(BaseModel):
    """S3 storage settings for ingestion outputs."""

    model_config = ConfigDict(extra="forbid")

    bucket: str = "argo-ebus-project-data-abm"


class IngestionConfig(BaseModel):
    """
    Drives the cloud-ingestion stage (analogous to script
    `02_ae_cloud_run.py` __main__). One config = one parquet write to S3.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    config_kind: Literal["ingestion"] = "ingestion"

    region: str
    date_start: dt.date
    date_end: dt.date
    lat_step: float = Field(gt=0)
    lon_step: float = Field(gt=0)
    time_step: float = Field(gt=0, description="days per bin")
    depth_range: Tuple[int, int]

    cloud: CloudBlock = Field(default_factory=CloudBlock)
    s3: S3Block = Field(default_factory=S3Block)

    description: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py::test_ingestion_config_valid_minimal -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/config_schema.py
git commit -m "feat(mlops): add IngestionConfig pydantic model with first valid-parse test"
```

---

### Task 1.2: IngestionConfig validation rules

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py` (add tests)
- Modify: `ArgoEBUSCloud/ebus_core/config_schema.py` (add validators)

- [ ] **Step 1: Write the failing tests**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
def test_ingestion_config_unknown_key_rejected():
    with pytest.raises(ValidationError) as excinfo:
        IngestionConfig(
            schema_version=1,
            config_kind="ingestion",
            region="californiav2",
            date_start=dt.date(2015, 1, 1),
            date_end=dt.date(2015, 12, 31),
            lat_step=0.5,
            lon_step=0.5,
            time_step=10.0,
            depth_range=(0, 100),
            mystery_key="nope",
        )
    assert "mystery_key" in str(excinfo.value)


def test_ingestion_config_unknown_region_rejected():
    with pytest.raises(ValidationError) as excinfo:
        IngestionConfig(
            schema_version=1,
            config_kind="ingestion",
            region="atlantis",
            date_start=dt.date(2015, 1, 1),
            date_end=dt.date(2015, 12, 31),
            lat_step=0.5,
            lon_step=0.5,
            time_step=10.0,
            depth_range=(0, 100),
        )
    assert "atlantis" in str(excinfo.value)


def test_ingestion_config_bad_dates_rejected():
    with pytest.raises(ValidationError) as excinfo:
        IngestionConfig(
            schema_version=1,
            config_kind="ingestion",
            region="californiav2",
            date_start=dt.date(2016, 1, 1),
            date_end=dt.date(2015, 12, 31),
            lat_step=0.5,
            lon_step=0.5,
            time_step=10.0,
            depth_range=(0, 100),
        )
    assert "date_start" in str(excinfo.value).lower() or "date_end" in str(excinfo.value).lower()


def test_ingestion_config_bad_depth_range_rejected():
    with pytest.raises(ValidationError):
        IngestionConfig(
            schema_version=1,
            config_kind="ingestion",
            region="californiav2",
            date_start=dt.date(2015, 1, 1),
            date_end=dt.date(2015, 12, 31),
            lat_step=0.5,
            lon_step=0.5,
            time_step=10.0,
            depth_range=(100, 100),  # top == bottom
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k "unknown or bad_dates or bad_depth"
```
Expected: at least the date and depth tests FAIL (no validators yet); unknown-key test should already PASS due to `extra="forbid"` and unknown-region FAILs (no validator yet).

- [ ] **Step 3: Add validators to IngestionConfig**

Open `ArgoEBUSCloud/ebus_core/config_schema.py`. Add this import at the top:
```python
from pydantic import field_validator, model_validator
```

Append the following methods inside the `IngestionConfig` class (after the field declarations, before the closing of the class):
```python
    @field_validator("region")
    @classmethod
    def _region_in_registry(cls, v: str) -> str:
        registry = get_ebus_registry()
        if v not in registry:
            raise ValueError(
                f"region '{v}' not in get_ebus_registry(); "
                f"valid: {sorted(registry.keys())}"
            )
        return v

    @field_validator("depth_range")
    @classmethod
    def _depth_range_ordered(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        top, bottom = v
        if top < 0 or bottom <= top:
            raise ValueError(
                f"depth_range must be (top, bottom) with 0 <= top < bottom; got {v}"
            )
        return v

    @model_validator(mode="after")
    def _dates_ordered(self) -> "IngestionConfig":
        if self.date_start >= self.date_end:
            raise ValueError(
                f"date_start ({self.date_start}) must be before date_end ({self.date_end})"
            )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/config_schema.py
git commit -m "feat(mlops): add IngestionConfig validators for region, depth, dates"
```

---

### Task 1.3: AnalysisConfig with cross-field rules

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`
- Modify: `ArgoEBUSCloud/ebus_core/config_schema.py`

- [ ] **Step 1: Write the failing tests**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
from ebus_core.config_schema import AnalysisConfig


def _valid_analysis_kwargs():
    return dict(
        schema_version=1,
        config_kind="analysis",
        input={
            "source": "s3",
            "s3_path": (
                "s3://argo-ebus-project-data-abm/"
                "californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400.parquet"
            ),
        },
        region="californiav2",
        date_start=dt.date(2015, 1, 1),
        date_end=dt.date(2015, 12, 31),
        lat_step=0.5,
        lon_step=0.5,
        time_step=10.0,
        depth_range=(150, 400),
        gpr={
            "mode": "3D",
            "kernel_type": "matern0.5",
            "window_size_days": 45,
            "step_size_days": 10,
            "min_bins": 10,
            "noise_val": 0.1,
            "time_ls_bounds_days": (15.0, 45.0),
            "spatial_ls_upper_bound": 10,
            "run_suffix": "_3dmatern_w45",
        },
        description="canonical Source FX2",
    )


def test_analysis_config_valid():
    cfg = AnalysisConfig(**_valid_analysis_kwargs())
    assert cfg.gpr.kernel_type == "matern0.5"
    assert cfg.outputs.aelogs_dir == "AEResults/aelogs"  # default
    assert cfg.outputs.generate_snapshots is True


def test_analysis_config_bin_aliasing_rejected():
    """time_step (10) > step_size_days (5) creates bin-aliasing — must reject."""
    bad = _valid_analysis_kwargs()
    bad["gpr"]["step_size_days"] = 5
    with pytest.raises(ValidationError) as excinfo:
        AnalysisConfig(**bad)
    assert "step_size_days" in str(excinfo.value) or "time_step" in str(excinfo.value)


def test_analysis_config_time_lower_bound_below_bin_rejected():
    """time_ls_bounds_days[0] (5.0) < time_step (10.0) — must reject."""
    bad = _valid_analysis_kwargs()
    bad["gpr"]["time_ls_bounds_days"] = (5.0, 45.0)
    with pytest.raises(ValidationError) as excinfo:
        AnalysisConfig(**bad)
    assert "time_ls_bounds_days" in str(excinfo.value).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k analysis
```
Expected: ImportError on `AnalysisConfig`.

- [ ] **Step 3: Add AnalysisConfig + sub-models to schema**

Append to `ArgoEBUSCloud/ebus_core/config_schema.py`:
```python
class AnalysisInputBlock(BaseModel):
    """Pointer to the parquet source for an analysis run."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["s3", "ingestion_run"]
    s3_path: str | None = None
    ingestion_run_id: str | None = None

    @model_validator(mode="after")
    def _exactly_one_pointer(self) -> "AnalysisInputBlock":
        if self.source == "s3" and not self.s3_path:
            raise ValueError("source=s3 requires s3_path")
        if self.source == "ingestion_run" and not self.ingestion_run_id:
            raise ValueError("source=ingestion_run requires ingestion_run_id")
        return self


class GPRBlock(BaseModel):
    """GPR (Gaussian-process regression) hyperparameters for a single
    analysis run. Mirrors the kwargs of run_diagnostic_inspection()."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["2D", "3D"] = "3D"
    kernel_type: Literal["rbf", "matern0.5", "gibbs"] = "matern0.5"
    window_size_days: int = Field(45, gt=0)
    step_size_days: int = Field(10, gt=0)
    min_bins: int = Field(10, ge=1)
    noise_val: float = Field(0.1, gt=0)
    time_ls_bounds_days: Tuple[float, float] = (15.0, 45.0)
    spatial_ls_upper_bound: float = Field(10.0, gt=0)
    run_suffix: str = ""


class OutputsBlock(BaseModel):
    """Where artifacts are written. Defaults match existing pipeline layout."""

    model_config = ConfigDict(extra="forbid")

    aelogs_dir: str = "AEResults/aelogs"
    aeplots_dir: str = "AEResults/aeplots"
    generate_snapshots: bool = True
    generate_physics_plots: bool = True


class AnalysisConfig(BaseModel):
    """
    Drives a GPR analysis run (analogous to script
    05_ae_update_tomatern0.5.py / 07_ae_deeper_layers.py __main__).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    config_kind: Literal["analysis"] = "analysis"

    input: AnalysisInputBlock

    region: str
    date_start: dt.date
    date_end: dt.date
    lat_step: float = Field(gt=0)
    lon_step: float = Field(gt=0)
    time_step: float = Field(gt=0)
    depth_range: Tuple[int, int]

    gpr: GPRBlock = Field(default_factory=GPRBlock)
    outputs: OutputsBlock = Field(default_factory=OutputsBlock)

    description: str = ""

    @field_validator("region")
    @classmethod
    def _region_in_registry(cls, v: str) -> str:
        registry = get_ebus_registry()
        if v not in registry:
            raise ValueError(
                f"region '{v}' not in get_ebus_registry(); "
                f"valid: {sorted(registry.keys())}"
            )
        return v

    @field_validator("depth_range")
    @classmethod
    def _depth_range_ordered(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        top, bottom = v
        if top < 0 or bottom <= top:
            raise ValueError(
                f"depth_range must be (top, bottom) with 0 <= top < bottom; got {v}"
            )
        return v

    @model_validator(mode="after")
    def _no_bin_aliasing(self) -> "AnalysisConfig":
        # Hard-won rule: step_size_days >= time_step prevents windows
        # from sharing identical 30d-bin contents (FX2 lesson).
        if self.gpr.step_size_days < self.time_step:
            raise ValueError(
                f"step_size_days ({self.gpr.step_size_days}) must be "
                f">= time_step ({self.time_step}) to prevent bin aliasing."
            )
        if self.gpr.time_ls_bounds_days[0] < self.time_step:
            raise ValueError(
                f"time_ls_bounds_days lower ({self.gpr.time_ls_bounds_days[0]}) "
                f"must be >= time_step ({self.time_step})."
            )
        if self.date_start >= self.date_end:
            raise ValueError(
                f"date_start ({self.date_start}) must be before date_end ({self.date_end})"
            )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/config_schema.py
git commit -m "feat(mlops): add AnalysisConfig with GPR/input/output sub-blocks and bin-aliasing rule"
```

---

### Task 1.4: YAML loader for configs

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`
- Modify: `ArgoEBUSCloud/ebus_core/config_schema.py`

- [ ] **Step 1: Write the failing test**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
import textwrap

from ebus_core.config_schema import load_config


def test_load_config_dispatches_by_kind(tmp_path):
    p = tmp_path / "ingest.yaml"
    p.write_text(textwrap.dedent("""\
        schema_version: 1
        config_kind: ingestion
        region: californiav2
        date_start: "2015-01-01"
        date_end:   "2015-12-31"
        lat_step: 0.5
        lon_step: 0.5
        time_step: 10.0
        depth_range: [0, 100]
    """))
    cfg = load_config(p)
    assert isinstance(cfg, IngestionConfig)
    assert cfg.region == "californiav2"


def test_load_config_rejects_missing_kind(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("schema_version: 1\nregion: californiav2\n")
    with pytest.raises(ValueError) as excinfo:
        load_config(p)
    assert "config_kind" in str(excinfo.value)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k load_config
```
Expected: ImportError on `load_config`.

- [ ] **Step 3: Add `load_config()` to schema module**

Append to `ArgoEBUSCloud/ebus_core/config_schema.py`:
```python
import os
from pathlib import Path
from typing import Union

import yaml


def load_config(path: Union[str, os.PathLike]) -> Union[IngestionConfig, AnalysisConfig]:
    """
    Parse a YAML config file and return the matching pydantic model.

    Dispatches on the top-level `config_kind` field. Strict validation
    (extra='forbid') applies — typos in field names raise ValidationError
    rather than being silently ignored.
    """
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(data)}")
    kind = data.get("config_kind")
    if kind == "ingestion":
        return IngestionConfig(**data)
    if kind == "analysis":
        return AnalysisConfig(**data)
    raise ValueError(
        f"{path}: missing or unknown config_kind (got {kind!r}); "
        f"expected 'ingestion' or 'analysis'"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/config_schema.py
git commit -m "feat(mlops): add load_config() YAML loader dispatching on config_kind"
```

---

### Task 1.5: Config canonicalization + sha256 hash

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`
- Create: `ArgoEBUSCloud/ebus_core/manifest.py`

- [ ] **Step 1: Write the failing tests**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
from ebus_core.manifest import canonical_config_dict, config_hash


def test_canonical_config_dict_excludes_description():
    cfg = IngestionConfig(
        schema_version=1, config_kind="ingestion",
        region="californiav2",
        date_start=dt.date(2015, 1, 1), date_end=dt.date(2015, 12, 31),
        lat_step=0.5, lon_step=0.5, time_step=10.0, depth_range=(0, 100),
        description="version A",
    )
    canon = canonical_config_dict(cfg)
    assert "description" not in canon


def test_config_hash_stable_across_description_changes():
    base = dict(
        schema_version=1, config_kind="ingestion",
        region="californiav2",
        date_start=dt.date(2015, 1, 1), date_end=dt.date(2015, 12, 31),
        lat_step=0.5, lon_step=0.5, time_step=10.0, depth_range=(0, 100),
    )
    cfg_a = IngestionConfig(**base, description="version A")
    cfg_b = IngestionConfig(**base, description="version B")
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_config_hash_changes_with_real_field():
    base = dict(
        schema_version=1, config_kind="ingestion",
        region="californiav2",
        date_start=dt.date(2015, 1, 1), date_end=dt.date(2015, 12, 31),
        lat_step=0.5, lon_step=0.5, time_step=10.0, depth_range=(0, 100),
    )
    cfg_a = IngestionConfig(**base)
    cfg_b = IngestionConfig(**{**base, "time_step": 30.0})
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_config_hash_is_64_char_hex():
    cfg = IngestionConfig(
        schema_version=1, config_kind="ingestion",
        region="californiav2",
        date_start=dt.date(2015, 1, 1), date_end=dt.date(2015, 12, 31),
        lat_step=0.5, lon_step=0.5, time_step=10.0, depth_range=(0, 100),
    )
    h = config_hash(cfg)
    assert len(h) == 64
    int(h, 16)  # valid hex
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k hash
```
Expected: ImportError on `ebus_core.manifest`.

- [ ] **Step 3: Create `ebus_core/manifest.py` with canonicalization + hash**

Create `ArgoEBUSCloud/ebus_core/manifest.py`:
```python
"""
Manifest module for the ArgoEBUS MLOps foundation.

Responsibilities:
- Canonicalize a pydantic config to a hashable dict (sorted keys, drop
  free-form fields like `description`).
- Compute a sha256 hash of the canonical dict — the run identity used by
  the collision detector.
- Capture run-time provenance: git SHA, conda env, host, timing.
- Read/write manifest.json files and append to the JSONL run registry.

The manifest is the bridge between a config (what was asked for) and
its outputs (what was produced). Once written, a manifest is immutable.
"""
import hashlib
import json
from typing import Any, Dict, Union

from pydantic import BaseModel


# Fields excluded from the hash because they don't affect run behavior.
# `description` is free-form annotation. `schema_version` is included
# because schema bumps DO affect interpretation.
_HASH_EXCLUDE = {"description"}


def canonical_config_dict(cfg: BaseModel) -> Dict[str, Any]:
    """
    Convert a pydantic config to a deterministic dict suitable for hashing.

    - Uses model_dump(mode='json') so dates become ISO strings and tuples
      become lists — both round-trippable and stable across Python versions.
    - Recursively sorts keys so dict ordering doesn't affect the hash.
    - Drops keys in _HASH_EXCLUDE (free-form annotation only).
    """
    raw = cfg.model_dump(mode="json")
    return _sorted_dict(_strip(raw))


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _HASH_EXCLUDE}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _sorted_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sorted_dict(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sorted_dict(x) for x in obj]
    return obj


def config_hash(cfg: BaseModel) -> str:
    """
    Return the sha256 (hex, 64 chars) of the canonicalized config.

    Same config (modulo `description`) → same hash. Different config →
    different hash with overwhelming probability.
    """
    canon = canonical_config_dict(cfg)
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/manifest.py
git commit -m "feat(mlops): canonicalize configs and compute sha256 hash"
```

---

### Task 1.6: Git + env + host capture

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`
- Modify: `ArgoEBUSCloud/ebus_core/manifest.py`

- [ ] **Step 1: Write the failing tests**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
from ebus_core.manifest import capture_code, capture_env, capture_host


def test_capture_code_returns_required_fields():
    code = capture_code()
    assert set(code.keys()) >= {"git_sha", "git_dirty", "git_branch", "repo_root"}
    assert isinstance(code["git_dirty"], bool)
    # On a real repo, git_sha is a 40-char hex string. Allow short SHA fallback.
    assert isinstance(code["git_sha"], str) and len(code["git_sha"]) >= 7


def test_capture_env_includes_key_packages_and_python():
    env = capture_env(conda_env_name="ebus-cloud-env", conda_list_dest=None)
    assert "python_version" in env
    assert "key_packages" in env
    assert isinstance(env["key_packages"], dict)
    expected = {"scikit-learn", "xarray", "numpy", "pandas",
                "gsw", "coiled", "dask", "matplotlib", "cartopy"}
    assert set(env["key_packages"].keys()) == expected


def test_capture_host_returns_hostname_and_platform():
    host = capture_host()
    assert "hostname" in host and host["hostname"]
    assert "platform" in host and host["platform"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k capture
```
Expected: ImportError on capture_code/env/host.

- [ ] **Step 3: Implement capture functions**

Append to `ArgoEBUSCloud/ebus_core/manifest.py`:
```python
import os
import platform as _platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Whitelisted package names that get inlined into the manifest. Bumping
# this list is a schema_version bump; full conda_list.txt is the
# authoritative source.
KEY_PACKAGES = (
    "scikit-learn", "xarray", "numpy", "pandas", "gsw",
    "coiled", "dask", "matplotlib", "cartopy",
)


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` until a .git/ dir is found. Raise if not under git."""
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError(f"no .git directory found in any parent of {start}")


def capture_code(cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Snapshot of the source-code state at run time."""
    cwd = Path(cwd) if cwd else Path.cwd()
    root = _find_repo_root(cwd)
    sha = _git(["rev-parse", "HEAD"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    dirty = bool(_git(["status", "--porcelain"], root))
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "git_branch": branch,
        "repo_root": str(root),
    }


def capture_env(
    conda_env_name: str,
    conda_list_dest: Optional[Path],
) -> Dict[str, Any]:
    """
    Snapshot of the Python execution environment.

    Always inlines the KEY_PACKAGES dict (versions or "MISSING" sentinel).
    If `conda_list_dest` is given, writes the full `conda list --json`
    output to that path so the manifest references it.
    """
    full = subprocess.run(
        ["conda", "list", "--json", "-n", conda_env_name],
        capture_output=True, text=True, check=True,
    ).stdout
    parsed = json.loads(full)
    versions = {pkg["name"]: pkg["version"] for pkg in parsed}
    key_pkgs = {name: versions.get(name, "MISSING") for name in KEY_PACKAGES}

    out: Dict[str, Any] = {
        "conda_env_name": conda_env_name,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "key_packages": key_pkgs,
    }
    if conda_list_dest is not None:
        conda_list_dest = Path(conda_list_dest)
        conda_list_dest.parent.mkdir(parents=True, exist_ok=True)
        conda_list_dest.write_text(full)
        out["conda_list_full"] = str(conda_list_dest)
    return out


def capture_host() -> Dict[str, Any]:
    """Snapshot of the host machine."""
    return {
        "hostname": socket.gethostname(),
        "platform": _platform.platform(),
    }
```

You also need to update the imports at the top of `manifest.py` — add to the existing import block:
```python
from typing import Any, Dict, Optional, Union
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/manifest.py
git commit -m "feat(mlops): capture git, conda env, and host metadata for manifests"
```

---

### Task 1.7: Manifest read/write + collision detector

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`
- Modify: `ArgoEBUSCloud/ebus_core/manifest.py`

- [ ] **Step 1: Write the failing tests**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
from ebus_core.manifest import (
    write_manifest, read_manifest, check_collision, ManifestCollisionError,
)


def _make_manifest_dict(hash_value="aaaaaaaa"):
    return {
        "schema_version": 1,
        "kind": "analysis",
        "run_id": "test_run",
        "config_hash": hash_value,
        "created_at": "2026-04-25T00:00:00Z",
        "duration_sec": 1.23,
        "config": {"region": "californiav2"},
        "code": {"git_sha": "abcd1234", "git_dirty": False,
                 "git_branch": "main", "repo_root": "/repo"},
        "env": {"conda_env_name": "ebus-cloud-env",
                "python_version": "3.11.7",
                "key_packages": {}},
        "inputs": {"source": "s3", "s3_path": "s3://b/k"},
        "outputs": {"aelogs_dir": "/tmp/x", "audit_csv": "/tmp/x/a.csv",
                    "snapshots_dir": "/tmp/y"},
        "host": {"hostname": "h", "platform": "p"},
    }


def test_manifest_roundtrip(tmp_path):
    p = tmp_path / "manifest.json"
    src = _make_manifest_dict()
    write_manifest(src, p)
    out = read_manifest(p)
    assert out == src


def test_check_collision_no_existing(tmp_path):
    # No prior manifest at this path → collision check is a no-op.
    p = tmp_path / "manifest.json"
    check_collision(p, new_hash="anything")  # no raise


def test_check_collision_identical_hash_warns(tmp_path):
    p = tmp_path / "manifest.json"
    write_manifest(_make_manifest_dict(hash_value="aaaa"), p)
    # Same hash → returns "rerun" verdict, no raise.
    verdict = check_collision(p, new_hash="aaaa")
    assert verdict == "rerun"


def test_check_collision_different_hash_raises(tmp_path):
    p = tmp_path / "manifest.json"
    write_manifest(_make_manifest_dict(hash_value="aaaa"), p)
    with pytest.raises(ManifestCollisionError) as excinfo:
        check_collision(p, new_hash="bbbb")
    msg = str(excinfo.value)
    assert "aaaa" in msg and "bbbb" in msg
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k "manifest_round or collision"
```
Expected: ImportErrors.

- [ ] **Step 3: Add manifest IO and collision logic**

Append to `ArgoEBUSCloud/ebus_core/manifest.py`:
```python
class ManifestCollisionError(Exception):
    """Raised when a run_id collides with a prior run whose config differs."""


def write_manifest(manifest: Dict[str, Any], path: Path) -> None:
    """Write a manifest dict to JSON. Parent dir created if missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def read_manifest(path: Path) -> Dict[str, Any]:
    """Read a manifest dict from JSON."""
    with Path(path).open("r") as f:
        return json.load(f)


def check_collision(manifest_path: Path, new_hash: str) -> str:
    """
    Compare a prospective run's config_hash against any existing manifest at
    the target path.

    Returns:
      'fresh'  → no prior manifest, safe to write
      'rerun'  → prior manifest with identical config_hash (warn + overwrite)

    Raises:
      ManifestCollisionError → prior manifest exists with a different
        config_hash. The caller must either bump run_suffix in the config
        or pass --force-overwrite to delete the prior outputs.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return "fresh"
    prior = read_manifest(manifest_path)
    if prior.get("config_hash") == new_hash:
        return "rerun"
    raise ManifestCollisionError(
        f"run_id collision at {manifest_path.parent}/\n"
        f"  existing manifest config_hash: {prior.get('config_hash')}\n"
        f"  new run config_hash:           {new_hash}\n"
        f"Configs differ. Either:\n"
        f"  - bump run_suffix in your config (e.g., add _v2)\n"
        f"  - OR re-run with --force-overwrite to delete prior outputs"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/manifest.py
git commit -m "feat(mlops): manifest read/write and run-id collision detector"
```

---

### Task 1.8: Run registry append

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`
- Modify: `ArgoEBUSCloud/ebus_core/manifest.py`

- [ ] **Step 1: Write the failing test**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
from ebus_core.manifest import append_registry


def test_append_registry_appends_one_line(tmp_path):
    reg = tmp_path / "registry.jsonl"
    m = _make_manifest_dict()
    m["config"]["depth_range"] = [0, 100]
    m["config"]["region"] = "californiav2"
    append_registry(m, reg, manifest_path=tmp_path / "manifest.json")
    append_registry(m, reg, manifest_path=tmp_path / "manifest.json")
    lines = reg.read_text().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert set(parsed.keys()) == {
        "run_id", "kind", "config_hash", "created_at",
        "region", "depth_range", "manifest_path",
    }
    assert parsed["region"] == "californiav2"
    assert parsed["depth_range"] == [0, 100]
```

Note: the test references `json` — add `import json` at the top of the test file if not already present (it is from earlier tasks via the `_make_manifest_dict` helper, but verify).

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k append_registry
```
Expected: ImportError.

- [ ] **Step 3: Add `append_registry()` to manifest module**

Append to `ArgoEBUSCloud/ebus_core/manifest.py`:
```python
# Fixed schema for the registry index line. Adding fields requires a
# schema_version bump on the manifest schema.
_REGISTRY_FIELDS = (
    "run_id", "kind", "config_hash", "created_at",
    "region", "depth_range", "manifest_path",
)


def append_registry(
    manifest: Dict[str, Any],
    registry_path: Path,
    manifest_path: Path,
) -> None:
    """
    Append a one-line denormalized index entry to the JSONL registry.

    The registry is the cross-run query surface. Per-run detail still
    lives in the per-run manifest pointed to by `manifest_path`.
    """
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "run_id": manifest["run_id"],
        "kind": manifest["kind"],
        "config_hash": manifest["config_hash"],
        "created_at": manifest["created_at"],
        "region": manifest["config"].get("region"),
        "depth_range": manifest["config"].get("depth_range"),
        "manifest_path": str(manifest_path),
    }
    assert set(line.keys()) == set(_REGISTRY_FIELDS)
    with registry_path.open("a") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/manifest.py
git commit -m "feat(mlops): append run summaries to JSONL registry"
```

---

## Phase 2: Runner

### Task 2.1: derive_run_id helper

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`
- Create: `ArgoEBUSCloud/ebus_core/runner.py`

- [ ] **Step 1: Write the failing test**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
from ebus_core.runner import derive_run_id


def test_derive_run_id_matches_existing_canonical_pattern():
    cfg = AnalysisConfig(**_valid_analysis_kwargs())
    rid = derive_run_id(cfg)
    # Must match the existing canonical run_id pattern produced by
    # script 05/07 — backward compatibility is the contract.
    assert rid == (
        "californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45"
    )


def test_derive_run_id_for_ingestion():
    cfg = IngestionConfig(
        schema_version=1, config_kind="ingestion",
        region="californiav2",
        date_start=dt.date(2015, 1, 1), date_end=dt.date(2015, 12, 31),
        lat_step=0.5, lon_step=0.5, time_step=10.0, depth_range=(150, 400),
    )
    rid = derive_run_id(cfg)
    assert rid == (
        "californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k derive_run_id
```
Expected: ImportError.

- [ ] **Step 3: Create `ebus_core/runner.py` with `derive_run_id`**

Create `ArgoEBUSCloud/ebus_core/runner.py`:
```python
"""
Runner module for the ArgoEBUS MLOps foundation.

Wraps the existing pipeline functions (script 02 ingestion, scripts 05/07
analysis) with a uniform interface that:
- accepts a validated pydantic config
- derives the canonical run_id (matching existing AEResults/aelogs/ naming)
- runs the collision detector
- captures wall-clock duration
- writes a manifest.json + appends to the JSONL registry
"""
from typing import Union

from ebus_core.config_schema import AnalysisConfig, IngestionConfig


def derive_run_id(cfg: Union[IngestionConfig, AnalysisConfig]) -> str:
    """
    Reproduce the canonical run_id naming used by existing scripts:

        {region}_{date_start_compact}_{date_end_compact}
        _res{lat}x{lon}_t{time_step}_d{depth0}_{depth1}{run_suffix}

    Where:
      - dates are YYYYMMDD (no separators)
      - lat/lon/time floats use underscore for the decimal point
        (matches existing convention: 0.5 -> "0_5", 10.0 -> "10_0")
      - run_suffix only applies to AnalysisConfig and is appended verbatim
    """
    region = cfg.region
    ds = cfg.date_start.strftime("%Y%m%d")
    de = cfg.date_end.strftime("%Y%m%d")
    lat = _fmt_dec(cfg.lat_step)
    lon = _fmt_dec(cfg.lon_step)
    t = _fmt_dec(cfg.time_step)
    d0, d1 = cfg.depth_range
    base = f"{region}_{ds}_{de}_res{lat}x{lon}_t{t}_d{d0}_{d1}"
    if isinstance(cfg, AnalysisConfig):
        return base + cfg.gpr.run_suffix
    return base


def _fmt_dec(x: float) -> str:
    """0.5 -> '0_5', 10.0 -> '10_0', 0.25 -> '0_25'."""
    s = f"{x:g}"  # canonical short form: '0.5', '10', '0.25'
    if "." not in s:
        s = s + ".0"
    return s.replace(".", "_")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k derive_run_id
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/runner.py
git commit -m "feat(mlops): derive_run_id matches existing canonical naming"
```

---

### Task 2.2: build_manifest pure function

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`
- Modify: `ArgoEBUSCloud/ebus_core/runner.py`

- [ ] **Step 1: Write the failing test**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
from ebus_core.runner import build_manifest


def test_build_manifest_has_all_required_top_keys():
    cfg = AnalysisConfig(**_valid_analysis_kwargs())
    m = build_manifest(
        cfg,
        outputs={
            "aelogs_dir": "/tmp/aelogs/run",
            "audit_csv": "/tmp/aelogs/run/audit.csv",
            "snapshots_dir": "/tmp/aeplots/snap",
        },
        inputs_extra={"parquet_etag": "etag1", "parquet_size_bytes": 12345},
        duration_sec=3.21,
        conda_list_dest=None,
    )
    required = {
        "schema_version", "kind", "run_id", "config_hash", "created_at",
        "duration_sec", "config", "code", "env", "inputs", "outputs", "host",
    }
    assert required <= set(m.keys())
    assert m["kind"] == "analysis"
    assert m["duration_sec"] == 3.21
    assert m["inputs"]["s3_path"].startswith("s3://")
    assert m["inputs"]["parquet_etag"] == "etag1"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k build_manifest
```
Expected: ImportError.

- [ ] **Step 3: Implement `build_manifest`**

Append to `ArgoEBUSCloud/ebus_core/runner.py`:
```python
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Optional

from ebus_core.manifest import (
    canonical_config_dict, capture_code, capture_env, capture_host, config_hash,
)


def build_manifest(
    cfg: Union[IngestionConfig, AnalysisConfig],
    outputs: Dict[str, Any],
    inputs_extra: Dict[str, Any],
    duration_sec: float,
    conda_list_dest: Optional[Path],
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Assemble a manifest dict from a validated config + execution metadata.

    Pure function — does no IO besides spawning git/conda subprocesses
    via the capture_* helpers (which themselves are read-only).
    """
    kind = "ingestion" if isinstance(cfg, IngestionConfig) else "analysis"

    # Inputs block depends on kind. Ingestion has no inputs (ERDDAP fetch
    # is implicit). Analysis carries the parquet pointer + etag.
    if isinstance(cfg, AnalysisConfig):
        inputs: Dict[str, Any] = {
            "source": cfg.input.source,
            "s3_path": cfg.input.s3_path,
            "ingestion_run_id": cfg.input.ingestion_run_id,
            **inputs_extra,
        }
    else:
        inputs = {"source": "erddap", **inputs_extra}

    return {
        "schema_version": 1,
        "kind": kind,
        "run_id": derive_run_id(cfg),
        "config_hash": config_hash(cfg),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "duration_sec": duration_sec,
        "config": canonical_config_dict(cfg),
        "code": capture_code(cwd=cwd),
        "env": capture_env(
            conda_env_name="ebus-cloud-env",
            conda_list_dest=conda_list_dest,
        ),
        "inputs": inputs,
        "outputs": outputs,
        "host": capture_host(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 25 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/runner.py
git commit -m "feat(mlops): build_manifest assembles full manifest dict from config + metadata"
```

---

### Task 2.3: run_analysis wrapper around existing GPR function

**Files:**
- Modify: `ArgoEBUSCloud/ebus_core/runner.py`
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`

This task wires the runner to the existing `run_diagnostic_inspection()` function. First, **read the existing signature** to know what kwargs to pass.

- [ ] **Step 1: Read the existing analysis function signature**

Open `ArgoEBUSCloud/05_ae_update_tomatern0.5.py` and find the `def run_diagnostic_inspection(` line. Note the full parameter list and defaults. This wrapper must pass those kwargs that AnalysisConfig encodes (region, lat_step, lon_step, time_step, depth_range, plus all `gpr.*` fields).

- [ ] **Step 2: Write the failing test**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
import importlib
from unittest import mock

from ebus_core.runner import run_analysis


def test_run_analysis_dispatches_to_existing_function(tmp_path, monkeypatch):
    """run_analysis must call run_diagnostic_inspection with config-derived kwargs."""
    cfg = AnalysisConfig(**_valid_analysis_kwargs())

    captured_kwargs = {}

    def fake_dispatch(**kwargs):
        captured_kwargs.update(kwargs)
        # Simulate run producing a manifest + audit CSV under aelogs_dir
        run_id = derive_run_id(cfg)
        out_dir = tmp_path / "aelogs" / run_id
        out_dir.mkdir(parents=True)
        (out_dir / f"audit_{run_id}.csv").write_text("dummy,csv\n")
        return {"audit_csv": str(out_dir / f"audit_{run_id}.csv")}

    # Override outputs dir to tmp_path so we don't pollute AEResults/.
    cfg = AnalysisConfig(**{
        **_valid_analysis_kwargs(),
        "outputs": {
            "aelogs_dir": str(tmp_path / "aelogs"),
            "aeplots_dir": str(tmp_path / "aeplots"),
            "generate_snapshots": False,
            "generate_physics_plots": False,
        },
    })

    monkeypatch.setattr(
        "ebus_core.runner._call_run_diagnostic_inspection", fake_dispatch
    )

    result = run_analysis(cfg, registry_path=tmp_path / "registry.jsonl")

    assert captured_kwargs["region"] == "californiav2"
    assert captured_kwargs["depth_range"] == (150, 400)
    assert captured_kwargs["kernel_type"] == "matern0.5"
    assert captured_kwargs["window_size_days"] == 45
    assert (tmp_path / "aelogs" / result["run_id"] / "manifest.json").exists()
    assert (tmp_path / "registry.jsonl").exists()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k run_analysis_dispatches
```
Expected: ImportError on `run_analysis`.

- [ ] **Step 4: Implement `run_analysis` and the dispatch shim**

Append to `ArgoEBUSCloud/ebus_core/runner.py`:
```python
import shutil
import time as _time

from ebus_core.manifest import (
    ManifestCollisionError, append_registry, check_collision, write_manifest,
)


def _call_run_diagnostic_inspection(**kwargs):
    """
    Thin shim that imports and calls the existing GPR analysis function.

    Isolated so tests can monkeypatch it without standing up the full
    sklearn/xarray/coiled dependency stack. The real call is deferred to
    invocation time so import of `runner` stays cheap.
    """
    # Import inside the function so tests can swap this whole shim out
    # via monkeypatch without triggering heavy imports.
    import importlib.util
    import sys
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "05_ae_update_tomatern0.5.py"
    spec = importlib.util.spec_from_file_location("script_05", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_05"] = mod
    spec.loader.exec_module(mod)
    return mod.run_diagnostic_inspection(**kwargs)


def run_analysis(
    cfg: AnalysisConfig,
    registry_path: Optional[Path] = None,
    force_overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Execute one GPR analysis run end-to-end.

    Steps:
      1. Derive run_id, target manifest path
      2. Compute config hash
      3. Collision check (abort, rerun-warn, or fresh)
      4. Optionally clear prior outputs dir if --force-overwrite
      5. Time-stamp start
      6. Dispatch to existing run_diagnostic_inspection() via the shim
      7. Time-stamp end, compute duration
      8. Build manifest
      9. Write manifest.json
     10. Append to registry
     11. Return summary dict
    """
    run_id = derive_run_id(cfg)
    aelogs_dir = Path(cfg.outputs.aelogs_dir) / run_id
    manifest_path = aelogs_dir / "manifest.json"
    cfg_hash = config_hash(cfg)

    verdict = check_collision(manifest_path, new_hash=cfg_hash)
    if verdict == "rerun":
        # Identical re-run: keep going, will overwrite outputs in place.
        pass
    if force_overwrite and aelogs_dir.exists():
        shutil.rmtree(aelogs_dir)

    aelogs_dir.mkdir(parents=True, exist_ok=True)

    # Map AnalysisConfig → run_diagnostic_inspection kwargs.
    dispatch_kwargs = {
        "region": cfg.region,
        "lat_step": cfg.lat_step,
        "lon_step": cfg.lon_step,
        "time_step": cfg.time_step,
        "depth_range": cfg.depth_range,
        "mode": cfg.gpr.mode,
        "kernel_type": cfg.gpr.kernel_type,
        "window_size_days": cfg.gpr.window_size_days,
        "step_size_days": cfg.gpr.step_size_days,
        "min_bins": cfg.gpr.min_bins,
        "noise_val": cfg.gpr.noise_val,
        "time_ls_bounds_days": cfg.gpr.time_ls_bounds_days,
        "spatial_ls_upper_bound": cfg.gpr.spatial_ls_upper_bound,
        "run_suffix": cfg.gpr.run_suffix,
    }

    t0 = _time.time()
    dispatch_result = _call_run_diagnostic_inspection(**dispatch_kwargs)
    duration = _time.time() - t0

    audit_csv = (
        dispatch_result.get("audit_csv")
        if isinstance(dispatch_result, dict)
        else str(aelogs_dir / f"audit_{run_id}.csv")
    )
    snapshots_dir = str(Path(cfg.outputs.aeplots_dir) / f"snapshot_{run_id}")

    manifest = build_manifest(
        cfg,
        outputs={
            "aelogs_dir": str(aelogs_dir),
            "audit_csv": audit_csv,
            "snapshots_dir": snapshots_dir,
        },
        inputs_extra={},
        duration_sec=duration,
        conda_list_dest=aelogs_dir / "conda_list.txt",
    )
    write_manifest(manifest, manifest_path)
    if registry_path is not None:
        append_registry(manifest, registry_path, manifest_path)

    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "duration_sec": duration,
        "verdict": verdict,
    }
```

You must also update the imports at the top of `runner.py` to include `Optional` if the existing import block doesn't already (`from typing import Any, Dict, Optional, Union`).

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k run_analysis_dispatches
```
Expected: 1 passed (the test monkeypatches the dispatch shim, so the real script-05 is not invoked).

- [ ] **Step 6: Run the full suite to check nothing regressed**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 26 passed.

- [ ] **Step 7: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/runner.py
git commit -m "feat(mlops): run_analysis wraps existing GPR fn with manifest + collision detection"
```

---

### Task 2.4: run_ingestion wrapper

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`
- Modify: `ArgoEBUSCloud/ebus_core/runner.py`

- [ ] **Step 1: Read the existing ingestion function signature**

Open `ArgoEBUSCloud/02_ae_cloud_run.py` and find the function called from `__main__`. Note its kwargs. The wrapper must map IngestionConfig → those kwargs.

- [ ] **Step 2: Write the failing test**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
from ebus_core.runner import run_ingestion


def test_run_ingestion_dispatches(tmp_path, monkeypatch):
    cfg = IngestionConfig(
        schema_version=1, config_kind="ingestion",
        region="californiav2",
        date_start=dt.date(2015, 1, 1), date_end=dt.date(2015, 12, 31),
        lat_step=0.5, lon_step=0.5, time_step=10.0, depth_range=(0, 100),
    )

    captured = {}

    def fake_ingest(**kwargs):
        captured.update(kwargs)
        return {"s3_path": "s3://bucket/key.parquet", "etag": "abc", "size_bytes": 100}

    monkeypatch.setattr("ebus_core.runner._call_run_ingestion", fake_ingest)

    aelogs_root = tmp_path / "aelogs"
    monkeypatch.setattr(
        "ebus_core.runner.INGESTION_AELOGS_DIR", aelogs_root
    )

    result = run_ingestion(cfg, registry_path=tmp_path / "registry.jsonl")

    assert captured["region"] == "californiav2"
    assert captured["depth_range"] == (0, 100)
    assert (aelogs_root / result["run_id"] / "manifest.json").exists()
    assert (tmp_path / "registry.jsonl").exists()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k run_ingestion_dispatches
```
Expected: ImportError on `run_ingestion`.

- [ ] **Step 4: Implement `run_ingestion`**

Append to `ArgoEBUSCloud/ebus_core/runner.py`:
```python
# Default location for ingestion manifests. Module-level so tests can
# monkeypatch via the symbol name.
INGESTION_AELOGS_DIR = Path("AEResults/aelogs/ingestion")


def _call_run_ingestion(**kwargs):
    """Thin shim around the existing ingestion entry point in script 02."""
    import importlib.util
    import sys
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "02_ae_cloud_run.py"
    spec = importlib.util.spec_from_file_location("script_02", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["script_02"] = mod
    spec.loader.exec_module(mod)
    # Existing script exposes a top-level callable that accepts the
    # canonical signature. The exact name (e.g., run_ingestion_pipeline)
    # is read from script_02 at execution time. Tests monkeypatch this
    # whole shim, so production wiring lives only here.
    if hasattr(mod, "run_ingestion_pipeline"):
        return mod.run_ingestion_pipeline(**kwargs)
    raise RuntimeError(
        "02_ae_cloud_run.py must expose run_ingestion_pipeline(**kwargs); "
        "this entry point is the additive seam for the MLOps runner."
    )


def run_ingestion(
    cfg: IngestionConfig,
    registry_path: Optional[Path] = None,
    force_overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Execute one cloud-ingestion run end-to-end.

    Mirrors run_analysis() but writes manifests under
    INGESTION_AELOGS_DIR / {run_id} / manifest.json.
    """
    run_id = derive_run_id(cfg)
    aelogs_dir = INGESTION_AELOGS_DIR / run_id
    manifest_path = aelogs_dir / "manifest.json"
    cfg_hash = config_hash(cfg)

    verdict = check_collision(manifest_path, new_hash=cfg_hash)
    if force_overwrite and aelogs_dir.exists():
        shutil.rmtree(aelogs_dir)
    aelogs_dir.mkdir(parents=True, exist_ok=True)

    dispatch_kwargs = {
        "region": cfg.region,
        "lat_step": cfg.lat_step,
        "lon_step": cfg.lon_step,
        "time_step": cfg.time_step,
        "depth_range": cfg.depth_range,
        "date_start": cfg.date_start,
        "date_end": cfg.date_end,
        "n_workers": cfg.cloud.n_workers,
        "worker_region": cfg.cloud.worker_region,
        "s3_bucket": cfg.s3.bucket,
    }

    t0 = _time.time()
    result = _call_run_ingestion(**dispatch_kwargs)
    duration = _time.time() - t0

    inputs_extra = {
        "parquet_etag": result.get("etag"),
        "parquet_size_bytes": result.get("size_bytes"),
    }
    outputs = {
        "aelogs_dir": str(aelogs_dir),
        "s3_path": result.get("s3_path"),
    }

    manifest = build_manifest(
        cfg, outputs=outputs, inputs_extra=inputs_extra,
        duration_sec=duration,
        conda_list_dest=aelogs_dir / "conda_list.txt",
    )
    write_manifest(manifest, manifest_path)
    if registry_path is not None:
        append_registry(manifest, registry_path, manifest_path)

    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "s3_path": result.get("s3_path"),
        "duration_sec": duration,
        "verdict": verdict,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 27 passed.

- [ ] **Step 6: Commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py ArgoEBUSCloud/ebus_core/runner.py
git commit -m "feat(mlops): run_ingestion wraps existing cloud-ingest pipeline"
```

---

### Task 2.5: Audit existing scripts expose the callable seams

**Files:**
- Read-only inspection: `ArgoEBUSCloud/02_ae_cloud_run.py`, `05_ae_update_tomatern0.5.py`, `07_ae_deeper_layers.py`

The runner shims assume `02_ae_cloud_run.py` exposes `run_ingestion_pipeline(**kwargs)` and `05_ae_update_tomatern0.5.py` exposes `run_diagnostic_inspection(**kwargs)`. Verify both exist with the kwargs the runner passes.

- [ ] **Step 1: Confirm `run_diagnostic_inspection` accepts the runner's kwargs**

```bash
grep -n "def run_diagnostic_inspection" ArgoEBUSCloud/05_ae_update_tomatern0.5.py
```
Expected: function definition found. Verify its parameter list includes (or accepts via `**kwargs`): `region, lat_step, lon_step, time_step, depth_range, mode, kernel_type, window_size_days, step_size_days, min_bins, noise_val, time_ls_bounds_days, spatial_ls_upper_bound, run_suffix`.

If any are missing, those kwargs are dropped silently — the run will use the function's defaults. **Document any drift in `argo_claude_actions/AE_claude_lessons.md`** so the next maintainer knows. **Do not modify the existing function** — surgical changes only, per project rules.

- [ ] **Step 2: Confirm `run_ingestion_pipeline` (or analog) exists in script 02**

```bash
grep -n "^def " ArgoEBUSCloud/02_ae_cloud_run.py
```
Expected: a function callable from `__main__` exists. If its name is NOT `run_ingestion_pipeline`, update the shim in `runner.py` (`_call_run_ingestion`) to call the actual function name. If no such function exists (logic is inlined in `__main__`), the script needs a non-invasive refactor: extract the `__main__` body into a `run_ingestion_pipeline(**kwargs)` function, leaving the `__main__` block as a one-liner that calls it. Surgical change only.

- [ ] **Step 3: If a refactor was needed, commit it separately**

```bash
git add ArgoEBUSCloud/02_ae_cloud_run.py
git commit -m "refactor(02): extract __main__ body into run_ingestion_pipeline()"
```
(Skip this step if no refactor was needed.)

---

## Phase 3: CLI

### Task 3.1: argparse skeleton + validate command

**Files:**
- Create: `ArgoEBUSCloud/aebus_cli.py`
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`

- [ ] **Step 1: Write the failing test**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
import subprocess as _subprocess


def test_cli_validate_prints_run_id(tmp_path):
    cfg_path = tmp_path / "ingest.yaml"
    cfg_path.write_text(textwrap.dedent("""\
        schema_version: 1
        config_kind: ingestion
        region: californiav2
        date_start: "2015-01-01"
        date_end:   "2015-12-31"
        lat_step: 0.5
        lon_step: 0.5
        time_step: 10.0
        depth_range: [0, 100]
    """))
    proc = _subprocess.run(
        ["python", "ArgoEBUSCloud/aebus_cli.py", "validate", str(cfg_path)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "californiav2_20150101_20151231_res0_5x0_5_t10_0_d0_100" in proc.stdout


def test_cli_validate_bad_yaml_exits_2(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("schema_version: 1\nconfig_kind: ingestion\nregion: atlantis\n")
    proc = _subprocess.run(
        ["python", "ArgoEBUSCloud/aebus_cli.py", "validate", str(cfg_path)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 2
    assert "atlantis" in proc.stderr or "atlantis" in proc.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k cli_validate
```
Expected: process returncode != 0, file-not-found.

- [ ] **Step 3: Create `aebus_cli.py` with argparse skeleton + validate**

Create `ArgoEBUSCloud/aebus_cli.py`:
```python
"""
aebus — config-driven CLI for the ArgoEBUS pipeline.

Subcommands:
  validate  Parse + schema-check a config and print its derived run_id.
  ingest    Run an ingestion config end-to-end (writes parquet + manifest).
  analyze   Run an analysis config end-to-end (writes audit + manifest).
  list      Tabulate runs from AEResults/run_registry.jsonl.
  show      Print a run's manifest.json.

All commands invoke library code under ebus_core.{config_schema, manifest, runner}.
"""
import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

# Allow `python ArgoEBUSCloud/aebus_cli.py ...` from the repo root by
# pointing imports at the ArgoEBUSCloud directory.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from ebus_core.config_schema import load_config  # noqa: E402
from ebus_core.runner import derive_run_id  # noqa: E402


DEFAULT_REGISTRY = Path("AEResults/run_registry.jsonl")


def cmd_validate(args: argparse.Namespace) -> int:
    """Parse + schema-check config; print run_id and target manifest path."""
    try:
        cfg = load_config(args.config)
    except (ValidationError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    rid = derive_run_id(cfg)
    print(f"config_kind: {cfg.config_kind}")
    print(f"run_id:      {rid}")
    if cfg.config_kind == "analysis":
        print(f"manifest:    AEResults/aelogs/{rid}/manifest.json")
    else:
        print(f"manifest:    AEResults/aelogs/ingestion/{rid}/manifest.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aebus", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="parse and schema-check a config")
    p_val.add_argument("config", type=Path)
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k cli_validate
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/aebus_cli.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(mlops): aebus CLI with validate subcommand"
```

---

### Task 3.2: aebus analyze + ingest subcommands

**Files:**
- Modify: `ArgoEBUSCloud/aebus_cli.py`
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`

- [ ] **Step 1: Write the failing test**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
def test_cli_analyze_collision_aborts(tmp_path):
    """If two analyses with different configs target the same run_id, the
    second must abort with exit code 3."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    aelogs_dir = tmp_path / "aelogs"
    aelogs_dir.mkdir()
    run_id = "californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45"
    (aelogs_dir / run_id).mkdir()
    # Plant a prior manifest with a different config_hash
    (aelogs_dir / run_id / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "kind": "analysis", "run_id": run_id,
        "config_hash": "deadbeef",
        "created_at": "2026-04-01T00:00:00Z", "duration_sec": 0,
        "config": {}, "code": {}, "env": {}, "inputs": {},
        "outputs": {}, "host": {},
    }))
    cfg_path = cfg_dir / "a.yaml"
    cfg_path.write_text(textwrap.dedent(f"""\
        schema_version: 1
        config_kind: analysis
        input:
          source: s3
          s3_path: "s3://b/k.parquet"
        region: californiav2
        date_start: "2015-01-01"
        date_end: "2015-12-31"
        lat_step: 0.5
        lon_step: 0.5
        time_step: 10.0
        depth_range: [150, 400]
        gpr:
          mode: "3D"
          kernel_type: matern0.5
          window_size_days: 45
          step_size_days: 10
          time_ls_bounds_days: [15.0, 45.0]
          spatial_ls_upper_bound: 10
          run_suffix: "_3dmatern_w45"
        outputs:
          aelogs_dir: "{aelogs_dir}"
          aeplots_dir: "{tmp_path / 'aeplots'}"
          generate_snapshots: false
          generate_physics_plots: false
    """))
    proc = _subprocess.run(
        ["python", "ArgoEBUSCloud/aebus_cli.py", "analyze", str(cfg_path)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    assert "collision" in (proc.stdout + proc.stderr).lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k cli_analyze_collision
```
Expected: returncode != 3 (subcommand not yet wired).

- [ ] **Step 3: Add `analyze` and `ingest` subcommands to CLI**

Edit `ArgoEBUSCloud/aebus_cli.py`. Add these imports near the top:
```python
from ebus_core.manifest import ManifestCollisionError  # noqa: E402
from ebus_core.runner import run_analysis, run_ingestion  # noqa: E402
```

Add these command handlers (above the `def main` line):
```python
def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (ValidationError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if cfg.config_kind != "analysis":
        print(f"ERROR: expected analysis config, got {cfg.config_kind}", file=sys.stderr)
        return 2
    try:
        result = run_analysis(
            cfg,
            registry_path=args.registry,
            force_overwrite=args.force_overwrite,
        )
    except ManifestCollisionError as e:
        print(f"COLLISION: {e}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (ValidationError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if cfg.config_kind != "ingestion":
        print(f"ERROR: expected ingestion config, got {cfg.config_kind}", file=sys.stderr)
        return 2
    try:
        result = run_ingestion(
            cfg,
            registry_path=args.registry,
            force_overwrite=args.force_overwrite,
        )
    except ManifestCollisionError as e:
        print(f"COLLISION: {e}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2))
    return 0
```

Inside `def main(...)`, after the existing `p_val` block but before the final `args.func(args)` call, add:
```python
    p_ana = sub.add_parser("analyze", help="run an analysis config end-to-end")
    p_ana.add_argument("config", type=Path)
    p_ana.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p_ana.add_argument("--force-overwrite", action="store_true")
    p_ana.set_defaults(func=cmd_analyze)

    p_ing = sub.add_parser("ingest", help="run an ingestion config end-to-end")
    p_ing.add_argument("config", type=Path)
    p_ing.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p_ing.add_argument("--force-overwrite", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k cli_analyze_collision
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/aebus_cli.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(mlops): aebus analyze + ingest subcommands with collision exit code 3"
```

---

### Task 3.3: aebus list + show subcommands

**Files:**
- Modify: `ArgoEBUSCloud/aebus_cli.py`
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`

- [ ] **Step 1: Write the failing tests**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
def test_cli_list_filters_by_region(tmp_path):
    reg = tmp_path / "registry.jsonl"
    reg.write_text(
        json.dumps({"run_id": "a1", "kind": "analysis", "config_hash": "h1",
                    "created_at": "t1", "region": "californiav2",
                    "depth_range": [0, 100], "manifest_path": "/x/m.json"}) + "\n" +
        json.dumps({"run_id": "b2", "kind": "analysis", "config_hash": "h2",
                    "created_at": "t2", "region": "humboldt",
                    "depth_range": [0, 100], "manifest_path": "/y/m.json"}) + "\n"
    )
    proc = _subprocess.run(
        ["python", "ArgoEBUSCloud/aebus_cli.py", "list",
         "--registry", str(reg), "--region", "californiav2"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "a1" in proc.stdout
    assert "b2" not in proc.stdout


def test_cli_show_prints_manifest_json(tmp_path):
    m = {"schema_version": 1, "kind": "analysis", "run_id": "z9",
         "config_hash": "h", "created_at": "t",
         "duration_sec": 1.0, "config": {}, "code": {}, "env": {},
         "inputs": {}, "outputs": {}, "host": {}}
    aelogs = tmp_path / "aelogs" / "z9"
    aelogs.mkdir(parents=True)
    (aelogs / "manifest.json").write_text(json.dumps(m))
    reg = tmp_path / "registry.jsonl"
    reg.write_text(json.dumps({
        "run_id": "z9", "kind": "analysis", "config_hash": "h",
        "created_at": "t", "region": None, "depth_range": None,
        "manifest_path": str(aelogs / "manifest.json"),
    }) + "\n")
    proc = _subprocess.run(
        ["python", "ArgoEBUSCloud/aebus_cli.py", "show", "z9",
         "--registry", str(reg)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert parsed["run_id"] == "z9"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k "cli_list or cli_show"
```
Expected: command-not-found / argparse exit.

- [ ] **Step 3: Implement list + show**

Edit `ArgoEBUSCloud/aebus_cli.py`. Add these handlers (above `def main`):
```python
def cmd_list(args: argparse.Namespace) -> int:
    if not args.registry.exists():
        print(f"(no registry at {args.registry})")
        return 0
    rows = []
    with args.registry.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.region and row.get("region") != args.region:
                continue
            if args.kind and row.get("kind") != args.kind:
                continue
            rows.append(row)
    if not rows:
        print("(no matching runs)")
        return 0
    # Simple fixed-width table.
    cols = ("created_at", "kind", "region", "depth_range", "run_id")
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    if not args.registry.exists():
        print(f"ERROR: no registry at {args.registry}", file=sys.stderr)
        return 1
    target = None
    with args.registry.open("r") as f:
        for line in f:
            row = json.loads(line)
            if row.get("run_id") == args.run_id:
                target = row
                break
    if target is None:
        print(f"ERROR: run_id {args.run_id!r} not in registry", file=sys.stderr)
        return 1
    manifest_path = Path(target["manifest_path"])
    if not manifest_path.exists():
        print(f"ERROR: registry points to missing manifest {manifest_path}", file=sys.stderr)
        return 1
    print(manifest_path.read_text())
    return 0
```

Inside `def main(...)`, register the subparsers:
```python
    p_list = sub.add_parser("list", help="list runs from the registry")
    p_list.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p_list.add_argument("--region", default=None)
    p_list.add_argument("--kind", default=None, choices=["analysis", "ingestion"])
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="print the manifest.json for a run_id")
    p_show.add_argument("run_id")
    p_show.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p_show.set_defaults(func=cmd_show)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: 31 passed.

- [ ] **Step 5: Commit**

```bash
git add ArgoEBUSCloud/aebus_cli.py ArgoEBUSCloud/test_mlops_foundation.py
git commit -m "feat(mlops): aebus list + show subcommands query the registry"
```

---

## Phase 4: Backfill Existing Runs

### Task 4.1: Write a backfill script (one-off, not committed to the library)

**Files:**
- Create: `ArgoEBUSCloud/scripts/backfill_configs.py`
- Create: `configs/` (empty dir, populated by the script)

This script enumerates `AEResults/aelogs/*/` directories, parses each `run_id` to recover its parameters, and writes a YAML config under `configs/<region>/`. It is a one-off utility — keep it under `scripts/` (a new dir) so library code in `ebus_core/` stays clean.

- [ ] **Step 1: Confirm the existing aelogs layout**

```bash
ls AEResults/aelogs/
```
Expected: a list of run_id-named directories, each containing `audit_*.csv` and (after this plan lands) optionally `manifest.json`.

- [ ] **Step 2: Create the backfill script**

```bash
mkdir -p ArgoEBUSCloud/scripts
```

Create `ArgoEBUSCloud/scripts/backfill_configs.py`:
```python
"""
One-off: enumerate existing AEResults/aelogs/*/ directories and write a
configs/<region>/<derived_filename>.yaml for each.

Recovery rules per run_id:
  region                — first underscore-separated segment
  date_start, date_end  — second + third segments (YYYYMMDD)
  lat_step, lon_step    — extracted from "resA_BxC_D" segment
  time_step             — extracted from "tA_B" segment
  depth_range           — extracted from "dA_B" segment
  run_suffix            — everything left after the depth segment

GPR parameters and ingestion cloud settings are filled with the canonical
defaults documented in the spec; runs that used non-default values must be
hand-edited after backfill (the script logs a WARN if it cannot infer).

Run from repo root:
  conda run -n ebus-cloud-env python ArgoEBUSCloud/scripts/backfill_configs.py
"""
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AELOGS = REPO_ROOT / "AEResults" / "aelogs"
CONFIGS = REPO_ROOT / "configs"

# Pattern matches the canonical run_id grammar:
#   <region>_<YYYYMMDD>_<YYYYMMDD>_res<lat>x<lon>_t<time>_d<d0>_<d1><suffix>
_RID = re.compile(
    r"^(?P<region>[^_]+(?:_v\d+)?)_"
    r"(?P<ds>\d{8})_(?P<de>\d{8})_"
    r"res(?P<lat>\d+_\d+)x(?P<lon>\d+_\d+)_"
    r"t(?P<t>\d+_\d+)_"
    r"d(?P<d0>\d+)_(?P<d1>\d+)"
    r"(?P<suffix>.*)$"
)


def _undec(s: str) -> float:
    return float(s.replace("_", "."))


def parse_run_id(rid: str) -> dict | None:
    m = _RID.match(rid)
    if not m:
        return None
    return {
        "region": m["region"],
        "date_start": f"{m['ds'][:4]}-{m['ds'][4:6]}-{m['ds'][6:]}",
        "date_end": f"{m['de'][:4]}-{m['de'][4:6]}-{m['de'][6:]}",
        "lat_step": _undec(m["lat"]),
        "lon_step": _undec(m["lon"]),
        "time_step": _undec(m["t"]),
        "depth_range": [int(m["d0"]), int(m["d1"])],
        "run_suffix": m["suffix"],
    }


def write_analysis_config(parts: dict, source_dir: Path) -> Path:
    region_dir = CONFIGS / parts["region"]
    region_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"analyze_d{parts['depth_range'][0]}_{parts['depth_range'][1]}"
        f"{parts['run_suffix']}.yaml"
    )
    out = region_dir / fname
    body = {
        "schema_version": 1,
        "config_kind": "analysis",
        "input": {
            "source": "s3",
            "s3_path": (
                f"s3://argo-ebus-project-data-abm/"
                f"{parts['region']}_{parts['date_start'].replace('-', '')}_"
                f"{parts['date_end'].replace('-', '')}_"
                f"res{str(parts['lat_step']).replace('.', '_')}"
                f"x{str(parts['lon_step']).replace('.', '_')}_"
                f"t{str(parts['time_step']).replace('.', '_')}_"
                f"d{parts['depth_range'][0]}_{parts['depth_range'][1]}.parquet"
            ),
        },
        "region": parts["region"],
        "date_start": parts["date_start"],
        "date_end": parts["date_end"],
        "lat_step": parts["lat_step"],
        "lon_step": parts["lon_step"],
        "time_step": parts["time_step"],
        "depth_range": parts["depth_range"],
        "gpr": {
            "mode": "3D",
            "kernel_type": "matern0.5",
            "window_size_days": 45,
            "step_size_days": 10,
            "min_bins": 10,
            "noise_val": 0.1,
            "time_ls_bounds_days": [15.0, 45.0],
            "spatial_ls_upper_bound": 10,
            "run_suffix": parts["run_suffix"],
        },
        "outputs": {
            "aelogs_dir": "AEResults/aelogs",
            "aeplots_dir": "AEResults/aeplots",
            "generate_snapshots": True,
            "generate_physics_plots": True,
        },
        "description": (
            f"backfilled from {source_dir.relative_to(REPO_ROOT)} "
            f"on 2026-04-25; verify GPR defaults match the original run"
        ),
    }
    with out.open("w") as f:
        yaml.safe_dump(body, f, sort_keys=False)
    return out


def main() -> int:
    if not AELOGS.exists():
        print(f"no aelogs at {AELOGS}", file=sys.stderr)
        return 1
    written = 0
    skipped = 0
    for d in sorted(AELOGS.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "ingestion":
            continue  # subdir for ingestion manifests, not a run
        parts = parse_run_id(d.name)
        if parts is None:
            print(f"WARN: cannot parse run_id, skipping: {d.name}", file=sys.stderr)
            skipped += 1
            continue
        out = write_analysis_config(parts, d)
        print(f"wrote {out.relative_to(REPO_ROOT)}")
        written += 1
    print(f"\n{written} configs written, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the backfill script**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env python ArgoEBUSCloud/scripts/backfill_configs.py
```
Expected: stdout lists each `configs/<region>/<file>.yaml` written, ends with a count summary. Any `WARN: cannot parse run_id` lines should be inspected manually — those are runs that don't follow the canonical naming (likely experimental dirs).

- [ ] **Step 4: Commit the backfill script + generated configs**

```bash
git add ArgoEBUSCloud/scripts/backfill_configs.py configs/
git commit -m "feat(mlops): backfill configs/ from existing AEResults/aelogs/ run_ids"
```

---

### Task 4.2: Round-trip test backfilled configs through the schema

**Files:**
- Modify: `ArgoEBUSCloud/test_mlops_foundation.py`

- [ ] **Step 1: Write the failing test**

Append to `ArgoEBUSCloud/test_mlops_foundation.py`:
```python
def test_all_backfilled_configs_parse_and_round_trip():
    """
    Every configs/**/analyze_*.yaml must (a) parse to a valid AnalysisConfig
    and (b) produce a derive_run_id() that matches the source aelogs dir.
    """
    repo = Path(__file__).resolve().parents[1]
    configs_dir = repo / "configs"
    if not configs_dir.exists():
        pytest.skip("no configs/ dir yet (backfill not run)")
    yaml_files = list(configs_dir.rglob("analyze_*.yaml"))
    assert yaml_files, "expected backfilled analysis configs"
    for path in yaml_files:
        cfg = load_config(path)
        assert isinstance(cfg, AnalysisConfig), path
        # Backfilled configs must produce a run_id that exists on disk.
        rid = derive_run_id(cfg)
        aelogs = repo / "AEResults" / "aelogs" / rid
        assert aelogs.is_dir(), f"backfilled config {path} → run_id {rid} not on disk"
```

- [ ] **Step 2: Run test**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v -k backfilled_configs
```
Expected: 1 passed (or skip if backfill produced no files).

If the test FAILS for specific configs, the cause is one of:
  (a) the backfill script's regex missed a real run_id pattern → fix the regex in `backfill_configs.py`, re-run, then re-test
  (b) the backfilled config's `gpr.run_suffix` doesn't match the source dir → adjust the suffix in the YAML (or the regex)
  (c) the AnalysisConfig validators reject defaults that were used historically → flag and discuss with the user before relaxing

- [ ] **Step 3: If tests pass, commit; if they revealed a regex bug, fix and commit**

```bash
git add ArgoEBUSCloud/test_mlops_foundation.py
# if regex was fixed:
# git add ArgoEBUSCloud/scripts/backfill_configs.py configs/
git commit -m "test(mlops): every backfilled config parses and round-trips to its aelogs dir"
```

---

## Phase 5: Documentation

### Task 5.1: configs/README.md

**Files:**
- Create: `configs/README.md`

- [ ] **Step 1: Write the README**

Create `configs/README.md`:
```markdown
# configs/

YAML configs for the ArgoEBUS pipeline. One YAML = one run.

## Layout

```
configs/
├── <region>/
│   ├── ingest_d<d0>_<d1>.yaml
│   └── analyze_d<d0>_<d1><run_suffix>.yaml
└── README.md
```

Region matches `get_ebus_registry()` keys (`california`, `californiav2`,
`californiav3`, `humboldt`, `canary`, `benguela`).

## Schema versioning

Every config begins with `schema_version: 1`. A schema bump is documented
here and accompanied by a migration plan.

## Strict parsing

Configs are validated by Pydantic models (`ebus_core.config_schema`):
unknown keys raise. Typos in field names will not be silently ignored.

## Hash stability

Config hash (sha256) is computed on the canonicalized config body
**excluding** the `description` field. Editing `description` does not
change the hash; editing any other field does.

## Backfill conventions

Configs in this directory that were generated by
`ArgoEBUSCloud/scripts/backfill_configs.py` carry a `description: backfilled
from ...` line. Their GPR sub-blocks use the canonical defaults documented
in the spec — runs that used non-default GPR settings must be hand-edited.

## Usage

```bash
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py validate <cfg.yaml>
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py analyze  <cfg.yaml>
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py ingest   <cfg.yaml>
```
```

- [ ] **Step 2: Commit**

```bash
git add configs/README.md
git commit -m "docs(mlops): configs/README explains schema, hash, backfill"
```

---

### Task 5.2: README + CLAUDE.md + ae_file_structure.txt updates

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `ae_file_structure.txt`

- [ ] **Step 1: Add MLOps section to README.md**

Open `README.md`. Append (or insert before any existing trailing license/footer block):
```markdown
## MLOps Foundation (config-driven runs)

The pipeline can be run via configs + a thin CLI:

```bash
# validate a config (schema + show derived run_id, no execution)
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py validate \
    configs/californiav2/analyze_d150_400_3dmatern_w45.yaml

# run an analysis end-to-end (writes audit + manifest + appends to registry)
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py analyze \
    configs/californiav2/analyze_d150_400_3dmatern_w45.yaml

# list past runs
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py list --region californiav2
```

Each successful run writes a `manifest.json` next to its outputs (config hash,
git SHA, conda env, S3 lineage) and appends one line to
`AEResults/run_registry.jsonl`. See `configs/README.md` and the design spec
at `docs/superpowers/specs/2026-04-25-mlops-foundation-design.md`.

The legacy `__main__` blocks in `02_ae_cloud_run.py`, `05_ae_update_tomatern0.5.py`,
and `07_ae_deeper_layers.py` continue to work as escape hatches.
```

- [ ] **Step 2: Add MLOps note to CLAUDE.md**

Open `CLAUDE.md`. Find the `## Hard-Won Rules` section. Append a new bullet:
```markdown
**Config-driven runs preferred over `__main__` edits**: new pipeline runs should
land in `configs/<region>/*.yaml` and execute via `aebus analyze` / `aebus ingest`.
The legacy `__main__` blocks still work but do not produce manifests or registry
entries. See `docs/superpowers/specs/2026-04-25-mlops-foundation-design.md`.
```

- [ ] **Step 3: Update ae_file_structure.txt**

Open `ae_file_structure.txt`. Inside the `ArgoEBUSCloud/ebus_core/` block, add entries for:
- `config_schema.py` — Pydantic models for IngestionConfig + AnalysisConfig
- `manifest.py` — config hash, git/env capture, manifest IO, registry
- `runner.py` — run_analysis + run_ingestion wrappers around existing pipeline fns

Inside the `ArgoEBUSCloud/` block, add:
- `aebus_cli.py` — config-driven CLI (`validate`, `analyze`, `ingest`, `list`, `show`)
- `scripts/backfill_configs.py` — one-off backfill of `configs/` from existing `aelogs/`
- `test_mlops_foundation.py` — pytest suite for the foundation modules

At the repo root, add:
- `configs/` — versioned YAML configs (one per region × layer × experiment)

- [ ] **Step 4: Commit all docs together**

```bash
git add README.md CLAUDE.md ae_file_structure.txt
git commit -m "docs(mlops): document config-driven workflow and new files"
```

---

### Task 5.3: Update AE_claude_recentactions.md

**Files:**
- Modify: `argo_claude_actions/AE_claude_recentactions.md`

- [ ] **Step 1: Prepend a session entry**

Open `argo_claude_actions/AE_claude_recentactions.md`. Insert a new entry at the
TOP of the file (just below the title), following the existing dated-entry format:
```markdown
## 2026-04-25 — MLOps Foundation Implemented (config-driven runs + manifests)

Implemented the MLOps foundation per spec
`docs/superpowers/specs/2026-04-25-mlops-foundation-design.md` and plan
`docs/superpowers/plans/2026-04-25-mlops-foundation.md`.

### Components delivered
- `ebus_core/config_schema.py` — IngestionConfig, AnalysisConfig (Pydantic v2,
  strict parsing, registry + bin-aliasing validators)
- `ebus_core/manifest.py` — canonical hash, git/env/host capture, manifest IO,
  collision detector, JSONL registry append
- `ebus_core/runner.py` — `run_analysis()` and `run_ingestion()` wrappers
- `aebus_cli.py` — `validate | analyze | ingest | list | show`
- `configs/` — backfilled from existing AEResults/aelogs/ via
  `scripts/backfill_configs.py`
- `test_mlops_foundation.py` — full pytest suite (target: all green)

### Convention
Every successful run now writes `manifest.json` (config hash, git SHA, conda env,
S3 lineage, duration) and appends one line to `AEResults/run_registry.jsonl`.

### Out of scope (explicit deferral)
- MLflow / W&B integration (next-spec, "B" tier)
- pip-installable package, Docker, dashboard ("C" tier)
- RG-Gibbs slots cleanly into AnalysisConfig.gpr.kernel_type once the
  separate spec lands
```

- [ ] **Step 2: Commit**

```bash
git add argo_claude_actions/AE_claude_recentactions.md
git commit -m "docs(actions): record MLOps foundation implementation session"
```

---

## Verification

After all phases land, run the full test suite and verify:

- [ ] **All tests green**

```bash
cd /home/avik2007/ArgoEBUSAnalysis && conda run -n ebus-cloud-env pytest ArgoEBUSCloud/test_mlops_foundation.py -v
```
Expected: ~32+ passed, 0 failed.

- [ ] **End-to-end smoke** (skip if no parquet locally available; only run if you can spare ~10 min)

Pick one backfilled analysis config that points to an S3 parquet you have access to:
```bash
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py validate configs/californiav2/analyze_d0_100_3dmatern_w45.yaml
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py analyze  configs/californiav2/analyze_d0_100_3dmatern_w45.yaml --force-overwrite
```
Expected: stdout shows the resolved run_id, the run completes, `AEResults/aelogs/<run_id>/manifest.json` exists, and `AEResults/run_registry.jsonl` has a new line.

- [ ] **Existing scripts still work**

```bash
conda run -n ebus-cloud-env python -c "import importlib.util, pathlib; spec = importlib.util.spec_from_file_location('s5', pathlib.Path('ArgoEBUSCloud/05_ae_update_tomatern0.5.py')); spec is not None"
```
Expected: prints `True` (script still importable; no syntax errors introduced).

- [ ] **Registry queryable**

```bash
conda run -n ebus-cloud-env python ArgoEBUSCloud/aebus_cli.py list
```
Expected: at least one row from the smoke test.

---

## End of plan
