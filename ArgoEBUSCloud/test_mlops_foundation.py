"""Tests for the MLOps foundation: config schema, manifest, runner, CLI."""
import datetime as dt

import pytest
from pydantic import ValidationError

from ebus_core.config_schema import (
    AnalysisConfig,
    GPRBlock,
    IngestionConfig,
    PhysicsParamsBlock,
    QCPolicyBlock,
)


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


def test_qc_policy_defaults_and_validation():
    # Default qc_policy attaches with [1, 2] flags
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
    assert cfg.qc_policy.argo_qc_flags_accepted == [1, 2]

    # Flag 8 (interpolated) is in Argo domain — should validate
    qc = QCPolicyBlock(argo_qc_flags_accepted=[1, 2, 8])
    assert 8 in qc.argo_qc_flags_accepted

    # Flag 99 not in Argo domain — should raise
    with pytest.raises(ValidationError):
        QCPolicyBlock(argo_qc_flags_accepted=[99])


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
            depth_range=(100, 100),  # top == bottom — must fail
        )


# ---------------------------------------------------------------------------
# Task 1.3: AnalysisConfig + sub-block tests
# ---------------------------------------------------------------------------


def _valid_analysis_kwargs():
    # Returns a dict of kwargs that build a valid AnalysisConfig.
    # Uses canonical Source-layer FX2 config: californiav2, t=10d, d=150–400m,
    # Matern-0.5 kernel. All fields are representative of a real production run.
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
            "lat_ls_bounds": (1.0e-2, 10.0),
            "lon_ls_bounds": (1.0e-2, 5.0),
            "run_suffix": "_3dmatern_w45",
        },
        description="canonical Source FX2",
    )


def test_analysis_config_valid():
    # Happy-path: canonical FX2 config parses cleanly and exposes expected defaults.
    cfg = AnalysisConfig(**_valid_analysis_kwargs())
    assert cfg.gpr.kernel_type == "matern0.5"
    assert cfg.outputs.aelogs_dir == "AEResults/aelogs"
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


def test_gpr_polymorphic_kernel_blocks_exclusive():
    # kernel_type=rbf but kernel_matern05 is also set — must raise because
    # exactly one sub-block (matching kernel_type) may be non-None.
    with pytest.raises(ValidationError):
        GPRBlock(
            kernel_type="rbf",
            lat_ls_bounds=(1e-2, 10.0),
            lon_ls_bounds=(1e-2, 5.0),
            kernel_matern05={},  # non-matching block set — should reject
        )


def test_lat_lon_ls_bounds_split_validates():
    # Valid split bounds pass; swapped (lower > upper) must reject.
    g = GPRBlock(
        lat_ls_bounds=(1e-2, 10.0),
        lon_ls_bounds=(1e-2, 5.0),
    )
    assert g.lat_ls_bounds == (1e-2, 10.0)

    with pytest.raises(ValidationError):
        GPRBlock(lat_ls_bounds=(10.0, 5.0))  # lower > upper — must fail


def test_physics_params_depth_ordering():
    # When both ohc_depth_top_m and ohc_depth_bot_m are supplied and top >= bot,
    # the model_validator must raise ValidationError.
    with pytest.raises(ValidationError):
        PhysicsParamsBlock(ohc_depth_top_m=400, ohc_depth_bot_m=150)  # top > bot
