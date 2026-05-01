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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class QCPolicyBlock(BaseModel):
    """
    Records which Argo QC flags were accepted and what was excluded so an
    ingestion run is reproducible. Argo flag domain: 1=good, 2=probably_good,
    3=probably_bad, 4=bad, 5=changed, 8=interpolated, 9=missing.
    """

    model_config = ConfigDict(extra="forbid")

    # Flags accepted during ingestion filtering; default = good + probably_good
    argo_qc_flags_accepted: list[int] = Field(default_factory=lambda: [1, 2])
    # Platform WMO numbers explicitly excluded from this run
    excluded_platform_numbers: list[int] = Field(default_factory=list)
    # Argo project name strings to drop (e.g. bad deployment programs)
    excluded_project_names: list[str] = Field(default_factory=list)
    # Free-text rationale for any non-default QC choices
    notes: str = ""

    @field_validator("argo_qc_flags_accepted")
    @classmethod
    def _flags_in_argo_domain(cls, v):
        # Argo quality control flag spec (ADMT): valid values are 1–5, 8, 9
        allowed = {1, 2, 3, 4, 5, 8, 9}
        bad = [f for f in v if f not in allowed]
        if bad:
            raise ValueError(f"Argo QC flags must be in {allowed}; got {bad}")
        return v


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
    qc_policy: QCPolicyBlock = Field(default_factory=QCPolicyBlock)

    description: str = ""

    @field_validator("region")
    @classmethod
    def _region_in_registry(cls, v: str) -> str:
        # Validate that region exists in the EBUS registry. Valid regions are
        # defined in get_ebus_registry() and include coastal upwelling systems
        # like california, californiav2, californiav3, humboldt, canary, benguela.
        # Input: region name string (e.g., "california" or "atlantis")
        # Output: validated region name, or raises ValueError if not found.
        # Raises: pydantic.ValidationError (wraps the ValueError) if region not in registry.
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
        # Validate that depth_range is a valid (top, bottom) tuple with
        # 0 <= top < bottom. This ensures the depth layer is positive and
        # well-defined for kriging/analysis.
        # Input: tuple of (top_depth, bottom_depth) in meters
        # Output: validated depth tuple, or raises ValueError if invalid.
        # Raises: pydantic.ValidationError (wraps the ValueError) if top < 0 or top >= bottom.
        top, bottom = v
        if top < 0 or bottom <= top:
            raise ValueError(
                f"depth_range must be (top, bottom) with 0 <= top < bottom; got {v}"
            )
        return v

    @model_validator(mode="after")
    def _dates_ordered(self) -> "IngestionConfig":
        # Validate that date_start < date_end to ensure the time window is
        # positive. This is a model_validator (not field_validator) because it
        # depends on comparing two fields.
        # Input: IngestionConfig instance with date_start and date_end
        # Output: validated self, or raises ValueError if dates are invalid.
        # Raises: pydantic.ValidationError (wraps the ValueError) if date_start >= date_end.
        if self.date_start >= self.date_end:
            raise ValueError(
                f"date_start ({self.date_start}) must be before date_end ({self.date_end})"
            )
        return self
