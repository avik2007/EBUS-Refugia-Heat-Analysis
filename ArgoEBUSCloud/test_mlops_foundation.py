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
    from ebus_core.config_schema import QCPolicyBlock
    qc = QCPolicyBlock(argo_qc_flags_accepted=[1, 2, 8])
    assert 8 in qc.argo_qc_flags_accepted

    # Flag 99 not in Argo domain — should raise
    with pytest.raises(ValidationError):
        QCPolicyBlock(argo_qc_flags_accepted=[99])
