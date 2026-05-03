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


def test_analysis_config_bad_dates_rejected():
    # AnalysisConfig: date_start >= date_end must reject (mirrors ingestion test).
    bad = _valid_analysis_kwargs()
    bad["date_start"] = dt.date(2016, 1, 1)
    bad["date_end"] = dt.date(2015, 12, 31)
    with pytest.raises(ValidationError) as excinfo:
        AnalysisConfig(**bad)
    assert "date_start" in str(excinfo.value).lower() or "date_end" in str(excinfo.value).lower()


def test_analysis_input_block_ingestion_run_requires_run_id():
    # source=ingestion_run without ingestion_run_id must raise (untested branch).
    from ebus_core.config_schema import AnalysisInputBlock
    with pytest.raises(ValidationError):
        AnalysisInputBlock(source="ingestion_run")  # ingestion_run_id=None — must reject


def test_gpr_gibbs_auto_instantiates_kernel_block():
    # kernel_type=gibbs with no kernel_gibbs supplied: validator must auto-fill it.
    g = GPRBlock(kernel_type="gibbs", lat_ls_bounds=(1e-2, 10.0), lon_ls_bounds=(1e-2, 5.0))
    assert g.kernel_gibbs is not None
    assert g.kernel_gibbs.l_form == "sigmoid_dist_to_coast"
    assert g.kernel_matern05 is None
    assert g.kernel_rbf is None


def test_fmt_dec_importable_from_ae_utils():
    # fmt_dec must be importable directly from ae_utils (Gap 4 — centralize formatting)
    from ebus_core.ae_utils import fmt_dec
    assert fmt_dec(0.5) == "0_5"
    assert fmt_dec(10.0) == "10_0"
    assert fmt_dec(0.25) == "0_25"
    assert fmt_dec(30.0) == "30_0"


import textwrap

from ebus_core.config_schema import load_config


def test_load_config_dispatches_by_kind(tmp_path):
    # load_config must parse a YAML file and return the correct pydantic model type.
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
    # load_config must raise ValueError when config_kind is absent.
    p = tmp_path / "bad.yaml"
    p.write_text("schema_version: 1\nregion: californiav2\n")
    with pytest.raises(ValueError) as excinfo:
        load_config(p)
    assert "config_kind" in str(excinfo.value)


from ebus_core.manifest import canonical_config_dict, config_hash


def test_canonical_config_dict_excludes_description():
    # canonical_config_dict must strip the free-form description field so it
    # does not affect the hash. description is annotation-only, not run behavior.
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
    # Two configs identical except for description must produce the same hash.
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
    # Changing a run-relevant field (time_step) must produce a different hash.
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
    # sha256 hex digest must be exactly 64 characters of valid hex.
    cfg = IngestionConfig(
        schema_version=1, config_kind="ingestion",
        region="californiav2",
        date_start=dt.date(2015, 1, 1), date_end=dt.date(2015, 12, 31),
        lat_step=0.5, lon_step=0.5, time_step=10.0, depth_range=(0, 100),
    )
    h = config_hash(cfg)
    assert len(h) == 64
    int(h, 16)  # raises ValueError if not valid hex


from ebus_core.manifest import capture_code, capture_env, capture_host


def test_capture_code_returns_required_fields():
    # capture_code must return at minimum git_sha, git_dirty, git_branch, repo_root.
    code = capture_code()
    assert set(code.keys()) >= {"git_sha", "git_dirty", "git_branch", "repo_root"}
    assert isinstance(code["git_dirty"], bool)
    assert isinstance(code["git_sha"], str) and len(code["git_sha"]) >= 7


def test_capture_env_includes_key_packages_and_python():
    # capture_env must include python_version and key_packages dict with the
    # canonical set of package names (all present in ebus-cloud-env).
    env = capture_env(conda_env_name="ebus-cloud-env", conda_list_dest=None)
    assert "python_version" in env
    assert "key_packages" in env
    assert isinstance(env["key_packages"], dict)
    expected = {"scikit-learn", "xarray", "numpy", "pandas",
                "gsw", "coiled", "dask", "matplotlib", "cartopy", "scipy"}
    assert set(env["key_packages"].keys()) == expected


def test_capture_host_returns_hostname_and_platform():
    # capture_host must return non-empty hostname and platform strings.
    host = capture_host()
    assert "hostname" in host and host["hostname"]
    assert "platform" in host and host["platform"]


from ebus_core.manifest import (
    write_manifest, read_manifest, check_collision, ManifestCollisionError,
)


def _make_manifest_dict(hash_value="aaaaaaaa"):
    # Helper: builds a minimal but valid manifest dict for IO + collision tests.
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
    # write_manifest + read_manifest must produce identical dicts.
    p = tmp_path / "manifest.json"
    src = _make_manifest_dict()
    write_manifest(src, p)
    out = read_manifest(p)
    assert out == src


def test_check_collision_no_existing(tmp_path):
    # No prior manifest → returns "fresh", no raise.
    p = tmp_path / "manifest.json"
    result = check_collision(p, new_hash="anything")
    assert result == "fresh"


def test_check_collision_identical_hash_warns(tmp_path):
    # Same hash → returns "rerun", no raise (safe to re-use the run_id).
    p = tmp_path / "manifest.json"
    write_manifest(_make_manifest_dict(hash_value="aaaa"), p)
    verdict = check_collision(p, new_hash="aaaa")
    assert verdict == "rerun"


def test_check_collision_different_hash_raises(tmp_path):
    # Different hash → ManifestCollisionError with both hashes in message.
    p = tmp_path / "manifest.json"
    write_manifest(_make_manifest_dict(hash_value="aaaa"), p)
    with pytest.raises(ManifestCollisionError) as excinfo:
        check_collision(p, new_hash="bbbb")
    msg = str(excinfo.value)
    assert "aaaa" in msg and "bbbb" in msg


import json as _json

from ebus_core.manifest import append_registry


def test_append_registry_appends_one_line(tmp_path):
    # append_registry must write one JSONL line per call with the canonical fields.
    reg = tmp_path / "registry.jsonl"
    m = _make_manifest_dict()
    m["config"]["depth_range"] = [0, 100]
    m["config"]["region"] = "californiav2"
    append_registry(m, reg, manifest_path=tmp_path / "manifest.json")
    append_registry(m, reg, manifest_path=tmp_path / "manifest.json")
    lines = reg.read_text().splitlines()
    assert len(lines) == 2
    parsed = _json.loads(lines[0])
    assert set(parsed.keys()) == {
        "run_id", "kind", "config_hash", "created_at",
        "region", "depth_range", "manifest_path", "status",
    }
    assert parsed["region"] == "californiav2"
    assert parsed["depth_range"] == [0, 100]


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


import datetime as _dt_mod  # avoid shadowing dt alias already in scope

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


def test_manifest_includes_erddap_lineage_and_teos10():
    """§A.4: manifest inputs must carry ERDDAP lineage fields."""
    cfg = AnalysisConfig(**_valid_analysis_kwargs())
    m = build_manifest(
        cfg,
        outputs={"aelogs_dir": "/tmp/x"},
        inputs_extra={
            "erddap_dataset_id": "ArgoFloats",
            "erddap_server_url": "https://coastwatch.pfeg.noaa.gov/erddap/",
            "data_access_timestamp": "2026-04-30T00:00:00Z",
        },
        duration_sec=1.0,
        conda_list_dest=None,
    )
    assert m["inputs"]["erddap_dataset_id"] == "ArgoFloats"
    assert m["inputs"]["erddap_server_url"].startswith("https://")
    assert m["inputs"]["data_access_timestamp"] == "2026-04-30T00:00:00Z"
    assert m["env"]["teos10_convention"] == "gsw-3.x"


def test_build_manifest_ingestion_config():
    """build_manifest IngestionConfig branch: kind='ingestion', source='erddap'."""
    cfg = IngestionConfig(
        schema_version=1, config_kind="ingestion",
        region="californiav2",
        date_start=dt.date(2015, 1, 1), date_end=dt.date(2015, 12, 31),
        lat_step=0.5, lon_step=0.5, time_step=10.0, depth_range=(0, 100),
    )
    m = build_manifest(
        cfg,
        outputs={"aelogs_dir": "/tmp/x"},
        inputs_extra={"erddap_dataset_id": "ArgoFloats"},
        duration_sec=2.0,
        conda_list_dest=None,
    )
    assert m["kind"] == "ingestion"
    assert m["inputs"]["source"] == "erddap"
    assert m["inputs"]["erddap_dataset_id"] == "ArgoFloats"


import importlib
from unittest import mock

from ebus_core.runner import run_analysis


def test_run_analysis_dispatches_to_existing_function(tmp_path, monkeypatch):
    """run_analysis must call _call_run_diagnostic_inspection with config-derived kwargs."""
    # Override outputs to tmp_path so we don't pollute AEResults/
    import copy
    kwargs = _valid_analysis_kwargs()
    kwargs["outputs"] = {
        "aelogs_dir": str(tmp_path / "aelogs"),
        "aeplots_dir": str(tmp_path / "aeplots"),
        "generate_snapshots": False,
        "generate_physics_plots": False,
    }
    cfg = AnalysisConfig(**kwargs)

    captured_kwargs = {}

    def fake_dispatch(**kwargs):
        captured_kwargs.update(kwargs)
        # Simulate run producing an audit CSV under aelogs_dir
        from ebus_core.runner import derive_run_id
        run_id = derive_run_id(cfg)
        out_dir = tmp_path / "aelogs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"audit_{run_id}.csv").write_text("dummy,csv\n")
        return {"audit_csv": str(out_dir / f"audit_{run_id}.csv")}

    monkeypatch.setattr("ebus_core.runner._call_run_diagnostic_inspection", fake_dispatch)

    result = run_analysis(cfg, registry_path=tmp_path / "registry.jsonl")

    assert captured_kwargs["region"] == "californiav2"
    assert captured_kwargs["depth_range"] == (150, 400)
    assert captured_kwargs["kernel_type"] == "matern0.5"
    assert captured_kwargs["window_size_days"] == 45
    assert (tmp_path / "aelogs" / result["run_id"] / "manifest.json").exists()
    assert (tmp_path / "registry.jsonl").exists()


def test_run_analysis_passes_spatial_ls_upper_bound(tmp_path, monkeypatch):
    # Gap 1 regression: lat_ls_bounds=(1.0, 8.0) and lon_ls_bounds=(1.0, 6.0)
    # must produce spatial_ls_upper_bound=8.0 in the dispatch kwargs.
    kwargs = _valid_analysis_kwargs()
    kwargs["outputs"] = {
        "aelogs_dir": str(tmp_path / "aelogs"),
        "aeplots_dir": str(tmp_path / "aeplots"),
        "generate_snapshots": False,
        "generate_physics_plots": False,
    }
    kwargs["gpr"]["lat_ls_bounds"] = (1.0, 8.0)
    kwargs["gpr"]["lon_ls_bounds"] = (1.0, 6.0)
    cfg = AnalysisConfig(**kwargs)

    captured_kwargs = {}

    def fake_dispatch(**kw):
        captured_kwargs.update(kw)
        run_id = derive_run_id(cfg)
        out_dir = tmp_path / "aelogs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"audit_{run_id}.csv").write_text("dummy,csv\n")
        return {"audit_csv": str(out_dir / f"audit_{run_id}.csv")}

    monkeypatch.setattr("ebus_core.runner._call_run_diagnostic_inspection", fake_dispatch)
    run_analysis(cfg)

    # max(8.0, 6.0) == 8.0; lat upper bound is the binding constraint
    assert captured_kwargs.get("spatial_ls_upper_bound") == 8.0


def test_run_analysis_omits_spatial_ls_upper_bound_for_legacy(tmp_path, monkeypatch):
    # Legacy configs with null bounds must NOT pass spatial_ls_upper_bound,
    # so the script default (10.0) applies unchanged.
    kwargs = _valid_analysis_kwargs()
    kwargs["outputs"] = {
        "aelogs_dir": str(tmp_path / "aelogs"),
        "aeplots_dir": str(tmp_path / "aeplots"),
        "generate_snapshots": False,
        "generate_physics_plots": False,
    }
    kwargs["gpr"]["lat_ls_bounds"] = None
    kwargs["gpr"]["lon_ls_bounds"] = None
    kwargs["legacy_backfill"] = True
    cfg = AnalysisConfig(**kwargs)

    captured_kwargs = {}

    def fake_dispatch(**kw):
        captured_kwargs.update(kw)
        run_id = derive_run_id(cfg)
        out_dir = tmp_path / "aelogs" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"audit_{run_id}.csv").write_text("dummy,csv\n")
        return {"audit_csv": str(out_dir / f"audit_{run_id}.csv")}

    monkeypatch.setattr("ebus_core.runner._call_run_diagnostic_inspection", fake_dispatch)
    run_analysis(cfg)

    assert "spatial_ls_upper_bound" not in captured_kwargs


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
    monkeypatch.setattr("ebus_core.runner.INGESTION_AELOGS_DIR", aelogs_root)

    result = run_ingestion(cfg, registry_path=tmp_path / "registry.jsonl")

    assert captured["region"] == "californiav2"
    assert captured["depth_range"] == (0, 100)
    assert (aelogs_root / result["run_id"] / "manifest.json").exists()
    assert (tmp_path / "registry.jsonl").exists()


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
    (aelogs_dir / run_id / "manifest.json").write_text(_json.dumps({
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


def test_cli_list_filters_by_region(tmp_path):
    reg = tmp_path / "registry.jsonl"
    reg.write_text(
        _json.dumps({"run_id": "a1", "kind": "analysis", "config_hash": "h1",
                     "created_at": "t1", "region": "californiav2",
                     "depth_range": [0, 100], "manifest_path": "/x/m.json"}) + "\n" +
        _json.dumps({"run_id": "b2", "kind": "analysis", "config_hash": "h2",
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
    (aelogs / "manifest.json").write_text(_json.dumps(m))
    reg = tmp_path / "registry.jsonl"
    reg.write_text(_json.dumps({
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
    parsed = _json.loads(proc.stdout)
    assert parsed["run_id"] == "z9"


# ---------------------------------------------------------------------------
# Phase 4: legacy_backfill schema + backfill script round-trip
# ---------------------------------------------------------------------------

import csv
import os

from ebus_core.config_schema import load_config
from ebus_core.runner import derive_run_id


def test_legacy_backfill_marker_permits_null_provenance():
    # When legacy_backfill=True, noise_val and all three gpr bound fields may be
    # null — they are forensic records of pre-manifest runs where these inputs
    # cannot be recovered. The schema must accept null and NOT fill in defaults.
    cfg = AnalysisConfig(
        schema_version=1,
        config_kind="analysis",
        legacy_backfill=True,
        input={"source": "s3", "s3_path": "s3://b/k.parquet"},
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
            "noise_val": None,
            "time_ls_bounds_days": None,
            "lat_ls_bounds": None,
            "lon_ls_bounds": None,
            "run_suffix": "_3dmatern_w45",
        },
        description="backfilled",
    )
    # All unrecoverable fields are null — not silently defaulted.
    assert cfg.legacy_backfill is True
    assert cfg.gpr.noise_val is None
    assert cfg.gpr.time_ls_bounds_days is None
    assert cfg.gpr.lat_ls_bounds is None
    assert cfg.gpr.lon_ls_bounds is None


def test_non_legacy_config_requires_gpr_bounds():
    # When legacy_backfill=False (the default), null gpr bounds must raise
    # ValidationError — real run configs must be fully specified.
    with pytest.raises(ValidationError):
        AnalysisConfig(
            schema_version=1,
            config_kind="analysis",
            input={"source": "s3", "s3_path": "s3://b/k.parquet"},
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
                "noise_val": None,
                "time_ls_bounds_days": None,
                "lat_ls_bounds": None,
                "lon_ls_bounds": None,
            },
        )


def test_backfilled_configs_round_trip(tmp_path):
    # backfill_configs must write a YAML per aelog dir such that load_config
    # parses it cleanly and derive_run_id(cfg) reproduces the original dir name.
    # Fixture: one canonical 3D-matern Source-layer run dir with a minimal audit CSV.
    from ebus_core.backfill import backfill_configs

    run_id = "californiav2_20150101_20151231_res0_5x0_5_t10_0_d150_400_3dmatern_w45"
    aelog_dir = tmp_path / "aelogs" / run_id
    aelog_dir.mkdir(parents=True)
    # Minimal audit CSV — two windows with different noise_val values.
    audit_csv = aelog_dir / f"audit_{run_id}.csv"
    audit_csv.write_text(
        "window_start,window_center,rmsre,std_z,noise_val,n_bins,n_floats,"
        "scale_lat_bin,scale_lon_bin,scale_time_bin,anisotropy_ratio\n"
        "5820.0,5842.5,0.034,0.73,0.000215,198,73,26.4,34.6,45.0,0.76\n"
        "5835.0,5857.5,0.020,0.88,0.000153,148,72,30.5,33.5,43.6,0.91\n"
    )

    configs_root = tmp_path / "configs"
    backfill_configs(
        aelogs_root=tmp_path / "aelogs",
        configs_root=configs_root,
        today="2026-04-30",
    )

    # Exactly one YAML written under configs/californiav2/
    written = list((configs_root / "californiav2").glob("*.yaml"))
    assert len(written) == 1

    cfg = load_config(written[0])
    # run_id derived from config must match the original aelog dir name exactly.
    assert derive_run_id(cfg) == run_id
    # legacy_backfill marker must be set.
    assert cfg.legacy_backfill is True
    # per-window noise values recorded, not collapsed to a single float.
    assert cfg.gpr.noise_vals_audit == [0.000215, 0.000153]
    # unrecoverable bounds are null.
    assert cfg.gpr.noise_val is None
    assert cfg.gpr.time_ls_bounds_days is None


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
