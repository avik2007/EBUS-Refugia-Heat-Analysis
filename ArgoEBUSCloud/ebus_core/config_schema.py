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
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import yaml
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

    # legacy_backfill: marks configs reconstructed from pre-manifest runs.
    # When True, unrecoverable fields may be null. Not used at runtime.
    legacy_backfill: bool = False

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


# ---------------------------------------------------------------------------
# Task 1.3: AnalysisConfig and its sub-models
# ---------------------------------------------------------------------------


class KernelRBFBlock(BaseModel):
    """Placeholder for RBF kernel config. No extra fields yet."""

    model_config = ConfigDict(extra="forbid")


class KernelMatern05Block(BaseModel):
    """Placeholder for Matern-0.5 kernel config. No extra fields yet."""

    model_config = ConfigDict(extra="forbid")


class KernelGibbsBlock(BaseModel):
    """
    RG-Gibbs kernel config (per 2026-04-26 l(x) directive).

    Lengthscale varies spatially as a sigmoid of dist_to_coast_km:
        l(d) = l_min + (l_max - l_min) / (1 + exp(-k * (d - d_0)))
    where d_0 and k are learnable parameters, l_min/l_max are bounds.
    This captures the physical transition from narrow coastal upwelling
    filaments (short lengthscale near coast) to broad open-ocean smoothness.
    """

    model_config = ConfigDict(extra="forbid")

    # l_form: which functional form the spatially-varying lengthscale takes.
    # Only sigmoid_dist_to_coast is implemented; kept as Literal for future extension.
    l_form: Literal["sigmoid_dist_to_coast"] = "sigmoid_dist_to_coast"

    # l_min_km: minimum lengthscale (near coast) in kilometres
    l_min_km: float = 100.0
    # l_max_km: maximum lengthscale (far offshore) in kilometres
    l_max_km: float = 400.0

    # d_transition_init_km: initial guess for d_0 (distance at sigmoid midpoint)
    d_transition_init_km: float = 300.0
    # d_transition_bounds_km: (lower, upper) optimisation bounds for d_0
    d_transition_bounds_km: Tuple[float, float] = (50.0, 700.0)

    # k_steepness_init: initial guess for sigmoid steepness k
    k_steepness_init: float = 0.01
    # k_steepness_bounds: (lower, upper) optimisation bounds for k
    k_steepness_bounds: Tuple[float, float] = (1.0e-4, 1.0)

    # anisotropy_lat_lon_ratio: ratio of lat lengthscale to lon lengthscale.
    # > 1 indicates meridional (current-driven) structure; < 1 indicates zonal
    # (atmospheric-forcing) structure. Physically expected to increase with depth.
    anisotropy_lat_lon_ratio: float = 2.0

    # climatology_source: which climatology dataset was used to build priors
    climatology_source: str = "roemmich-gilson-v3"


class GPRBlock(BaseModel):
    """
    Gaussian Process Regression (kriging) settings for one analysis run.

    §A.2: lat and lon lengthscale bounds are split (not a single spatial_ls)
    to allow anisotropic kernels. Anisotropy Ratio = lat_ls / lon_ls.
    Expected to increase with depth (atm forcing → current forcing).

    §A.2 polymorphic kernel: exactly one of kernel_rbf / kernel_matern05 /
    kernel_gibbs may be non-None; which one is determined by kernel_type.
    The model_validator auto-instantiates the matching block with defaults
    if the user omits it.
    """

    model_config = ConfigDict(extra="forbid")

    # mode: whether to interpolate in 2D (lat-lon per time slice) or 3D (lat-lon-time)
    mode: Literal["2D", "3D"] = "3D"

    # kernel_type: selects which covariance function is used. Determines which
    # kernel_* sub-block is active. "matern0.5" = exponential decay (default);
    # "rbf" = squared-exponential; "gibbs" = spatially-varying (RG-Gibbs).
    kernel_type: Literal["rbf", "matern0.5", "gibbs"] = "matern0.5"

    # window_size_days: width of the sliding temporal window in days
    window_size_days: int = Field(45, gt=0)
    # step_size_days: stride of the sliding window in days. Must be >= time_step
    # to prevent bin aliasing (two windows sharing the exact same bin contents).
    step_size_days: int = Field(10, gt=0)

    # min_bins: minimum occupied spatial bins required to attempt a GPR fit
    min_bins: int = Field(10, ge=1)
    # noise_val: observation noise variance added to the diagonal (Tikhonov reg).
    # Optional so legacy backfill can record null when the input value is unrecoverable.
    noise_val: Optional[float] = Field(default=0.1)

    # time_ls_bounds_days: (lower, upper) optimisation bounds for the temporal
    # lengthscale in days. Lower must be >= time_step (bin-aliasing guard in AnalysisConfig).
    # Optional so legacy backfill can record null when unrecoverable.
    time_ls_bounds_days: Optional[Tuple[float, float]] = (15.0, 45.0)

    # §A.2: split anisotropy — lat and lon have separate lengthscale bounds.
    # lat_ls_bounds: (lower, upper) in degrees latitude for the lat lengthscale
    # Optional so legacy backfill can record null when unrecoverable.
    lat_ls_bounds: Optional[Tuple[float, float]] = (1.0e-2, 10.0)
    # lon_ls_bounds: (lower, upper) in degrees longitude for the lon lengthscale
    # Optional so legacy backfill can record null when unrecoverable.
    lon_ls_bounds: Optional[Tuple[float, float]] = (1.0e-2, 5.0)

    # noise_vals_audit: per-window optimized noise values read from the audit CSV.
    # Only populated by the legacy backfill script — not used at runtime.
    # Records the empirical noise distribution for forensic review without
    # implying that any one value was the regularization input.
    noise_vals_audit: Optional[List[float]] = None

    # run_suffix: appended to the output run_id for human-readable labelling
    run_suffix: str = ""

    # §A.2 polymorphic kernel sub-blocks — exactly one is active (see model_validator)
    kernel_rbf: Optional[KernelRBFBlock] = None
    kernel_matern05: Optional[KernelMatern05Block] = None
    kernel_gibbs: Optional[KernelGibbsBlock] = None

    @field_validator("noise_val")
    @classmethod
    def _noise_val_positive(cls, v: Optional[float]) -> Optional[float]:
        # Validate noise_val > 0 when provided. Null is permitted for legacy backfill
        # configs where the original regularization input cannot be recovered.
        # Input: float noise value or None
        # Output: validated value unchanged.
        # Raises: pydantic.ValidationError if v is not None and v <= 0.
        if v is not None and v <= 0:
            raise ValueError(f"noise_val must be > 0; got {v}")
        return v

    @field_validator("lat_ls_bounds", "lon_ls_bounds", "time_ls_bounds_days")
    @classmethod
    def _ls_bounds_ordered(cls, v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        # Validate that the (lower, upper) lengthscale bounds satisfy 0 < lower < upper.
        # Physical requirement: lengthscales must be strictly positive, and the
        # optimisation search interval must be non-degenerate.
        # Null is permitted for legacy backfill configs (unrecoverable fields).
        # Input: tuple (lower, upper) as floats in physical units (degrees or days), or None
        # Output: validated tuple unchanged, or None.
        # Raises: pydantic.ValidationError (wraps ValueError) if constraint violated.
        if v is None:
            return v
        lo, hi = v
        if lo <= 0 or lo >= hi:
            raise ValueError(f"ls_bounds must satisfy 0 < lower < upper; got {v}")
        return v

    @model_validator(mode="after")
    def _kernel_sub_block_exclusive(self) -> "GPRBlock":
        # Enforce §A.2 polymorphism: exactly the sub-block matching kernel_type
        # may be non-None; all others must remain None. If the matching block
        # was not supplied by the user, auto-instantiate it with defaults so
        # callers can always access e.g. cfg.gpr.kernel_matern05 unconditionally.
        # Input: GPRBlock after all field validators have run.
        # Output: self with exactly one kernel sub-block populated.
        # Raises: pydantic.ValidationError (wraps ValueError) if a non-matching
        #         sub-block was explicitly set.
        kernel_map = {
            "rbf": ("kernel_rbf", KernelRBFBlock),
            "matern0.5": ("kernel_matern05", KernelMatern05Block),
            "gibbs": ("kernel_gibbs", KernelGibbsBlock),
        }
        active_attr, active_cls = kernel_map[self.kernel_type]

        # Collect any non-matching sub-blocks that were explicitly set by the user
        extra_set = [
            attr
            for attr, _ in kernel_map.values()
            if attr != active_attr and getattr(self, attr) is not None
        ]
        if extra_set:
            raise ValueError(
                f"kernel_type='{self.kernel_type}' but non-matching sub-blocks "
                f"are set: {extra_set}"
            )

        # Auto-instantiate the active block if the user omitted it.
        # Use setattr (not object.__setattr__) on non-frozen models so Pydantic's
        # own __setattr__ runs and any future post-assignment validators are honoured.
        if getattr(self, active_attr) is None:
            setattr(self, active_attr, active_cls())
        return self


class PhysicsParamsBlock(BaseModel):
    """
    Physical constants and QC thresholds for Ocean Heat Content (OHC) computation.

    All defaults reflect TEOS-10 conventions used in the existing pipeline.
    ohc_depth_top_m and ohc_depth_bot_m default to None so AnalysisConfig can
    substitute depth_range[0] and depth_range[1] at runtime, avoiding duplication.
    """

    model_config = ConfigDict(extra="forbid")

    # ohc_reference_pressure_dbar: reference pressure for TEOS-10 potential temperature
    # 0.0 dbar = sea surface (standard choice for OHC anomalies)
    ohc_reference_pressure_dbar: float = 0.0

    # ohc_depth_top_m: integration upper bound in metres; None = use depth_range[0]
    ohc_depth_top_m: Optional[int] = None
    # ohc_depth_bot_m: integration lower bound in metres; None = use depth_range[1]
    ohc_depth_bot_m: Optional[int] = None

    # teos10_convention: which TEOS-10 standard was applied (audit trail)
    teos10_convention: str = "TEOS-10-2010"

    # qc_min_obs_per_bin: minimum Argo profiles per spatial bin to include in OHC
    qc_min_obs_per_bin: int = 1

    # qc_outlier_sigma: if set, profiles more than this many σ from the mean are
    # flagged. None disables outlier rejection entirely.
    qc_outlier_sigma: Optional[float] = None

    @model_validator(mode="after")
    def _depth_top_below_bot(self) -> "PhysicsParamsBlock":
        # When both ohc_depth_top_m and ohc_depth_bot_m are provided, validate
        # that the integration direction is physically meaningful (top < bottom).
        # When either is None the runtime inherits the value from depth_range,
        # so we cannot validate the pairing here.
        # Input: PhysicsParamsBlock after fields are set.
        # Output: self unchanged if valid.
        # Raises: pydantic.ValidationError (wraps ValueError) if top >= bot.
        if self.ohc_depth_top_m is not None and self.ohc_depth_bot_m is not None:
            if self.ohc_depth_top_m >= self.ohc_depth_bot_m:
                raise ValueError(
                    f"ohc_depth_top_m ({self.ohc_depth_top_m}) must be < "
                    f"ohc_depth_bot_m ({self.ohc_depth_bot_m})"
                )
        return self


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


class AnalysisInputBlock(BaseModel):
    """
    Pointer to the parquet source for an analysis run.

    source=s3 requires an explicit s3_path; source=ingestion_run requires
    an ingestion_run_id matching a manifest entry. Exactly one path must
    be set (the other must remain None) — enforced by model_validator.
    """

    model_config = ConfigDict(extra="forbid")

    # source: whether the parquet data comes directly from S3 or from a
    # prior IngestionConfig run recorded in the manifest
    source: Literal["s3", "ingestion_run"]

    # s3_path: full s3:// URI to the parquet file (used when source="s3")
    s3_path: Optional[str] = None
    # ingestion_run_id: run_id key in the manifest (used when source="ingestion_run")
    ingestion_run_id: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_pointer(self) -> "AnalysisInputBlock":
        # Ensure the chosen source has a corresponding pointer set.
        # Prevents silent misconfiguration where source disagrees with the
        # populated path field.
        # Input: AnalysisInputBlock after fields are set.
        # Output: self unchanged if valid.
        # Raises: pydantic.ValidationError (wraps ValueError) on pointer mismatch.
        if self.source == "s3" and not self.s3_path:
            raise ValueError("source=s3 requires s3_path")
        if self.source == "ingestion_run" and not self.ingestion_run_id:
            raise ValueError("source=ingestion_run requires ingestion_run_id")
        return self


class OutputsBlock(BaseModel):
    """
    Where artifacts (logs, plots) are written.

    Defaults match the existing pipeline layout under AEResults/.
    Both generate_* flags allow individual plot categories to be disabled
    during fast diagnostic runs.
    """

    model_config = ConfigDict(extra="forbid")

    # aelogs_dir: relative path for audit CSV logs from the project root
    aelogs_dir: str = "AEResults/aelogs"
    # aeplots_dir: relative path for output PNG plots from the project root
    aeplots_dir: str = "AEResults/aeplots"

    # generate_snapshots: whether to emit temporal snapshot plots
    generate_snapshots: bool = True
    # generate_physics_plots: whether to emit OHC / thermohaline diagnostic plots
    generate_physics_plots: bool = True


class AnalysisConfig(BaseModel):
    """
    Drives a GPR analysis run (analogous to 05_ae_update_tomatern0.5.py /
    07_ae_deeper_layers.py __main__). One config = one GPR kriging run over
    a single region, depth layer, and time period.

    Cross-field rules (each has its own model_validator so errors surface independently):
    - _no_bin_aliasing: step_size_days >= time_step and time_ls_bounds_days[0]
      >= time_step prevent two windows sharing identical bin contents.
    - _non_legacy_complete: non-legacy configs must supply all GPR provenance fields.
    - _physics_depth_within_range: OHC integration bounds must lie within depth_range.
    - _dates_ordered: date_start < date_end (same logic as IngestionConfig).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    config_kind: Literal["analysis"] = "analysis"

    # input: pointer to the parquet source (S3 URI or ingestion manifest entry)
    input: AnalysisInputBlock

    region: str
    date_start: dt.date
    date_end: dt.date

    lat_step: float = Field(gt=0)
    lon_step: float = Field(gt=0)
    time_step: float = Field(gt=0, description="days per temporal bin")
    depth_range: Tuple[int, int]

    # gpr: GPR hyperparameters and kernel selection
    gpr: GPRBlock = Field(default_factory=GPRBlock)
    # outputs: artifact destination directories and plot flags
    outputs: OutputsBlock = Field(default_factory=OutputsBlock)
    # physics_params: OHC integration constants and QC thresholds (§A.3)
    physics_params: PhysicsParamsBlock = Field(default_factory=PhysicsParamsBlock)

    # legacy_backfill: marks configs reconstructed from pre-manifest runs.
    # When True, gpr.noise_val, gpr.time_ls_bounds_days, gpr.lat_ls_bounds, and
    # gpr.lon_ls_bounds may be null. The bin-aliasing time_ls check is also skipped.
    legacy_backfill: bool = False

    # backfill_metadata: populated by 10_ae_backfill_configs.py to record which
    # fields were recovered from the run_id vs assumed from pipeline defaults.
    # None for configs written by hand or by the MLOps runner (not backfilled).
    backfill_metadata: Optional[BackfillMetadataBlock] = None

    description: str = ""

    @field_validator("region")
    @classmethod
    def _region_in_registry(cls, v: str) -> str:
        # Same logic as IngestionConfig._region_in_registry — kept as a copy
        # to avoid cross-model dependency. Both configs validate independently.
        # Input: region name string (e.g., "californiav2")
        # Output: validated region name unchanged.
        # Raises: pydantic.ValidationError (wraps ValueError) if not in registry.
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
        # Validate that depth_range is a valid (top, bottom) tuple with 0 <= top < bottom.
        # Physical requirement: the depth layer must be positive and well-defined for
        # kriging. top is the shallower bound (e.g. 150m), bottom the deeper (e.g. 400m).
        # Input: tuple (top_depth, bottom_depth) in metres
        # Output: validated tuple unchanged.
        # Raises: pydantic.ValidationError (wraps ValueError) if top < 0 or top >= bottom.
        top, bottom = v
        if top < 0 or bottom <= top:
            raise ValueError(
                f"depth_range must be (top, bottom) with 0 <= top < bottom; got {v}"
            )
        return v

    @model_validator(mode="after")
    def _no_bin_aliasing(self) -> "AnalysisConfig":
        # Hard-won rule: if step_size_days < time_step, successive sliding windows
        # overlap so much that two windows share exactly the same set of binned
        # profiles, yielding duplicate GPR fits and inflated temporal coverage.
        # Similarly, time_ls_bounds_days[0] < time_step means the optimiser can
        # produce a temporal lengthscale shorter than one bin — physically meaningless.
        # The time_ls check is skipped for legacy backfill configs where
        # time_ls_bounds_days is null (unrecoverable from pre-manifest run).
        # Input: AnalysisConfig after all fields and GPRBlock have been validated.
        # Output: self unchanged if both checks pass.
        # Raises: pydantic.ValidationError (wraps ValueError) on any violated rule.
        # Legacy backfill configs may record experimental step values that violate
        # the aliasing rule (e.g. t1s10 had step=10 with time_step=30 deliberately).
        # Both checks are skipped for legacy configs — the runs already happened.
        if not self.legacy_backfill:
            if self.gpr.step_size_days < self.time_step:
                raise ValueError(
                    f"step_size_days ({self.gpr.step_size_days}) must be >= "
                    f"time_step ({self.time_step}) to prevent bin aliasing."
                )
            if self.gpr.time_ls_bounds_days is not None:
                if self.gpr.time_ls_bounds_days[0] < self.time_step:
                    raise ValueError(
                        f"time_ls_bounds_days lower ({self.gpr.time_ls_bounds_days[0]}) "
                        f"must be >= time_step ({self.time_step})."
                    )
        return self

    @model_validator(mode="after")
    def _non_legacy_complete(self) -> "AnalysisConfig":
        # Enforce that non-legacy configs specify all GPR provenance fields.
        # Legacy backfill configs may have null for fields that cannot be
        # recovered from pre-manifest run_id or audit CSV.
        # Input: AnalysisConfig after all fields and GPRBlock have been validated.
        # Output: self unchanged if valid.
        # Raises: pydantic.ValidationError (wraps ValueError) if any required GPR
        #         bound field is null on a non-legacy config.
        if not self.legacy_backfill:
            missing = []
            if self.gpr.noise_val is None:
                missing.append("gpr.noise_val")
            if self.gpr.lat_ls_bounds is None:
                missing.append("gpr.lat_ls_bounds")
            if self.gpr.lon_ls_bounds is None:
                missing.append("gpr.lon_ls_bounds")
            if missing:
                raise ValueError(
                    f"Fields required when legacy_backfill=False: {missing}. "
                    f"Set legacy_backfill=True for backfilled configs."
                )
        return self

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

    @model_validator(mode="after")
    def _dates_ordered(self) -> "AnalysisConfig":
        # Validate date_start < date_end to ensure the time window is positive.
        # Separated from _no_bin_aliasing so Pydantic can surface both errors independently.
        # Input: AnalysisConfig with date_start and date_end set.
        # Output: self unchanged if dates are ordered correctly.
        # Raises: pydantic.ValidationError (wraps ValueError) if date_start >= date_end.
        if self.date_start >= self.date_end:
            raise ValueError(
                f"date_start ({self.date_start}) must be before date_end ({self.date_end})"
            )
        return self


def load_config(path: Union[str, Path]) -> Union[IngestionConfig, AnalysisConfig]:
    # Parse a YAML config file and return the matching validated Pydantic model.
    # Dispatches on the top-level config_kind field ('ingestion' or 'analysis').
    # Input: path — filesystem path to a .yaml config file
    # Output: IngestionConfig or AnalysisConfig with all fields validated.
    # Raises: ValueError if the file is not a YAML mapping, or config_kind is missing/unknown.
    # Raises: pydantic.ValidationError if any field fails schema validation.
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
