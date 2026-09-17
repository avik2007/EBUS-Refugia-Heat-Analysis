import numpy as np
import pandas as pd
import pytest
import gsw

from ebus_core.argoebus_thermodynamics import estimate_ohc_from_raw_bins


# ---------------------------------------------------------------------------
# Isolation: the function calls ae_utils.calculate_dist_to_coast, which loads
# Cartopy Natural Earth coastline shapefiles (network + slow + non-deterministic).
# It is imported *inside* the function body as `from .ae_utils import ...`, so it
# resolves ae_utils.calculate_dist_to_coast at call time -- patch it there. The
# stub returns a constant 123.0 km per point so tests can assert on the column.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _stub_coast(monkeypatch):
    import ebus_core.ae_utils as ae_utils
    monkeypatch.setattr(
        ae_utils, "calculate_dist_to_coast",
        lambda lats, lons, *a, **k: np.full(len(np.atleast_1d(lats)), 123.0),
    )


# ---------------------------------------------------------------------------
# make_synthetic_df -- build an Argo-like raw-measurement DataFrame, ONE ROW PER
# MEASUREMENT with scalar cells, shaped exactly like the loader output that
# estimate_ohc_from_raw_bins consumes.
#
#   coverage : which depth layers inside one lat/lon/time bin carry data
#       "full"      -> a point at every 10 m from 5 m to 1995 m (200 points)
#       "shallow"   -> only points shallower than `shallow_max_m`
#       "deep"      -> only points deeper than `deep_min_m`
#       "sparse"    -> `n_sparse` points spread evenly 5..1995 m (spans column)
#   temp_offset_C : adds a constant offset to the whole temperature profile
#   n_floats      : split the rows across this many platform_number values,
#                   interleaved by depth (tests float pooling)
#   n_bins        : stamp out this many bins, each shifted +1 deg lat, +1 deg lon,
#                   +30 days so it lands in its own box
#   lat0, lon0, time0 : location / time of the first bin
#   include_out_of_window : also add a point at depth -50 m and +2500 m
#
# temp profile: 4 + offset + 12*exp(-z/400)  => ~16 C at surface, ~4 C at depth
# pres from gsw.p_from_z so it is physically consistent with depth.
# ---------------------------------------------------------------------------
def make_synthetic_df(
    coverage="full",
    temp_offset_C=0.0,
    n_floats=1,
    n_bins=1,
    lat0=35.25,
    lon0=-124.75,
    time0=100.0,
    shallow_max_m=160.0,
    deep_min_m=1840.0,
    n_sparse=12,
    include_out_of_window=False,
):
    all_depths = np.arange(5.0, 2000.0, 10.0)  # 5, 15, ..., 1995
    if coverage == "full":
        depths = all_depths
    elif coverage == "shallow":
        depths = all_depths[all_depths < shallow_max_m]
    elif coverage == "deep":
        depths = all_depths[all_depths > deep_min_m]
    elif coverage == "sparse":
        idx = np.linspace(0, len(all_depths) - 1, n_sparse).round().astype(int)
        depths = all_depths[np.unique(idx)]
    else:
        raise ValueError(f"unknown coverage {coverage!r}")

    rows = []
    for b in range(n_bins):
        blat, blon, btime = lat0 + b, lon0 + b, time0 + b * 30.0
        zseq = depths
        if include_out_of_window:
            zseq = np.concatenate([[-50.0], depths, [2500.0]])
        for i, z in enumerate(zseq):
            rows.append({
                "lat": blat,
                "lon": blon,
                "time_days": btime,
                "pres": float(gsw.p_from_z(-z, blat)),
                "depth": float(z),
                "temp": 4.0 + temp_offset_C + 12.0 * np.exp(-z / 400.0),
                "psal": 34.5,
                "platform_number": f"f{b}_{i % n_floats}",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# reference_energy_density -- recompute the pipeline's per-point energy density
# with the SAME GSW chain, so tests can derive expected OHC instead of hardcoding.
# `use_ct=True`  -> rho*cp*CT (what the function does).
# `use_ct=False` -> rho*cp*t  (the dormant in-situ convention) for the lock test.
# Returns a 1-D numpy array aligned with the rows of `df`.
# ---------------------------------------------------------------------------
def reference_energy_density(df, use_ct=True):
    sa = gsw.SA_from_SP(df["psal"].values, df["pres"].values,
                        df["lon"].values, df["lat"].values)
    ct = gsw.CT_from_t(sa, df["temp"].values, df["pres"].values)
    rho = gsw.rho(sa, ct, df["pres"].values)
    cp = gsw.cp_t_exact(sa, df["temp"].values, df["pres"].values)
    temp_term = ct if use_ct else df["temp"].values
    return rho * cp * temp_term


# ---------------------------------------------------------------------------
# expected_full_profile_ohc -- the OHC the pipeline should produce for a single
# "full" bin: energy density at every 10 m layer, ordered shallow->deep, then
# trapezoid-integrated with dx = vertical_step = 10 m. Same math the function
# runs internally, so a "full" bin needs no interpolation and this is exact.
# ---------------------------------------------------------------------------
def expected_full_profile_ohc(df, use_ct=True):
    E = reference_energy_density(df, use_ct=use_ct)
    order = np.argsort(df["depth"].values)
    return np.trapezoid(E[order], dx=10.0)


# ---------------------------------------------------------------------------
# test_full_profile_ohc_matches_gsw_trapezoid
# A dense (every-10-m) profile in one lat/lon/time bin should integrate to the
# trapezoid of rho*cp*CT over depth. Confirms the end-to-end numeric path:
# per-point GSW thermodynamics -> vertical binning -> depth integration.
# ---------------------------------------------------------------------------
def test_full_profile_ohc_matches_gsw_trapezoid():
    df = make_synthetic_df("full")
    result = estimate_ohc_from_raw_bins(df)
    expected = expected_full_profile_ohc(df, use_ct=True)
    assert len(result) == 1
    assert np.isclose(result["ohc"].iloc[0], expected, rtol=1e-6), \
        f"Expected {expected}, got {result['ohc'].iloc[0]}"


# ---------------------------------------------------------------------------
# test_ohc_per_m_identity
# ohc_per_m is defined as ohc / (depth_max - depth_min). Check the identity holds
# for the default 0-2000 m window and also tracks a non-default 100-400 m window.
# ---------------------------------------------------------------------------
def test_ohc_per_m_identity():
    df = make_synthetic_df("full")
    result = estimate_ohc_from_raw_bins(df)
    assert np.allclose(result["ohc_per_m"], result["ohc"] / 2000.0)

    result_narrow = estimate_ohc_from_raw_bins(df, depth_min=100, depth_max=400)
    assert not result_narrow.empty
    assert np.allclose(result_narrow["ohc_per_m"], result_narrow["ohc"] / 300.0)


# ---------------------------------------------------------------------------
# test_ct_convention_is_used_not_in_situ_t
# The live path uses Conservative Temperature: energy_density = rho*cp*CT. The
# dormant helper calculate_thermodynamics uses in-situ t (rho*cp*t). Pin that the
# live result matches the CT integral and NOT the in-situ-t integral, and that
# the two references differ enough (~1.4%) for the check to be meaningful.
# This is intended current behavior, not a bug -- see argoebus_thermodynamics.py.
# ---------------------------------------------------------------------------
def test_ct_convention_is_used_not_in_situ_t():
    df = make_synthetic_df("full")
    result = estimate_ohc_from_raw_bins(df)
    exp_ct = expected_full_profile_ohc(df, use_ct=True)
    exp_t = expected_full_profile_ohc(df, use_ct=False)
    assert np.isclose(result["ohc"].iloc[0], exp_ct, rtol=1e-6)
    assert not np.isclose(result["ohc"].iloc[0], exp_t, rtol=1e-3)
    assert abs(exp_ct - exp_t) / exp_ct > 1e-3


# ---------------------------------------------------------------------------
# test_warmer_water_more_heat
# Shifting the whole temperature profile up by 2 C must raise the integrated heat
# content for the same bin. Basic physical-sign sanity check.
# ---------------------------------------------------------------------------
def test_warmer_water_more_heat():
    warm = estimate_ohc_from_raw_bins(make_synthetic_df("full", temp_offset_C=2.0))
    base = estimate_ohc_from_raw_bins(make_synthetic_df("full"))
    assert warm["ohc"].iloc[0] > base["ohc"].iloc[0]


# ---------------------------------------------------------------------------
# test_interpolation_fills_interior_nans
# A lone bin's missing layers vanish as columns and are never interpolated; you
# only see interpolation when a sparse bin is unstacked beside a denser bin that
# populates the shared depth columns. Pair a "full" bin with a 40-point "sparse"
# bin: the sparse row's ~160 interior NaNs are linearly filled before
# integration, so its OHC lands within 1% of the full row's OHC (smooth profile).
# ---------------------------------------------------------------------------
def test_interpolation_fills_interior_nans():
    full = make_synthetic_df("full")
    sparse = make_synthetic_df("sparse", n_sparse=40, lat0=36.25, lon0=-123.75)
    df = pd.concat([full, sparse], ignore_index=True)
    result = estimate_ohc_from_raw_bins(df)
    assert len(result) == 2
    full_ohc = result.loc[result["lat_bin"] == 35.5, "ohc"].iloc[0]
    sparse_ohc = result.loc[result["lat_bin"] == 36.5, "ohc"].iloc[0]
    assert result.loc[result["lat_bin"] == 36.5, "n_raw_points"].iloc[0] == 40
    assert np.isclose(sparse_ohc, full_ohc, rtol=1e-2)


# ---------------------------------------------------------------------------
# test_output_schema_and_dtypes
# The output frame must carry exactly the documented columns, with float64 OHC
# columns, string platform ids (object dtype), and the stubbed constant coast
# distance flowing through untouched.
# ---------------------------------------------------------------------------
def test_output_schema_and_dtypes():
    df = make_synthetic_df("full")
    result = estimate_ohc_from_raw_bins(df)
    assert set(result.columns) == {
        "time_bin", "lat_bin", "lon_bin", "ohc", "ohc_per_m",
        "n_raw_points", "platform_number", "dist_to_coast_km",
    }
    assert result["ohc"].dtype == np.float64
    assert result["ohc_per_m"].dtype == np.float64
    assert result["platform_number"].dtype == object
    assert isinstance(result["platform_number"].iloc[0], str)
    assert (result["dist_to_coast_km"] == 123.0).all()


# ---------------------------------------------------------------------------
# test_bin_arithmetic
# lat_bin/lon_bin are floor-division bin centres: (x // 1.0) * 1.0 + 0.5.
# time_bin is the left edge: (t // 30) * 30. Includes a negative-longitude case
# where -125.25 // 1.0 == -126.0, so lon_bin == -125.5.
# ---------------------------------------------------------------------------
def test_bin_arithmetic():
    result = estimate_ohc_from_raw_bins(
        make_synthetic_df("full", lat0=35.25, lon0=-124.75, time0=100)
    )
    assert result["lat_bin"].iloc[0] == 35.5
    assert result["lon_bin"].iloc[0] == -124.5
    assert result["time_bin"].iloc[0] == 90

    result_neg = estimate_ohc_from_raw_bins(
        make_synthetic_df("full", lat0=36.25, lon0=-125.25, time0=100)
    )
    assert result_neg["lon_bin"].iloc[0] == -125.5


# ---------------------------------------------------------------------------
# test_two_distinct_bins_two_rows
# Two bins offset by +1 deg lat, +1 deg lon, +30 days must produce two output
# rows with the expected (time_bin, lat_bin, lon_bin) keys.
# ---------------------------------------------------------------------------
def test_two_distinct_bins_two_rows():
    result = estimate_ohc_from_raw_bins(make_synthetic_df("full", n_bins=2))
    assert len(result) == 2
    keys = set(map(tuple, result[["time_bin", "lat_bin", "lon_bin"]].values.tolist()))
    assert keys == {(90.0, 35.5, -124.5), (120.0, 36.5, -123.5)}


# ---------------------------------------------------------------------------
# test_multi_float_pooling
# Three platform ids interleaved by depth within one bin collapse to a single
# pooled synthetic profile: one output row, all 200 raw points counted, a string
# platform id retained.
# ---------------------------------------------------------------------------
def test_multi_float_pooling():
    result = estimate_ohc_from_raw_bins(make_synthetic_df("full", n_floats=3))
    assert len(result) == 1
    assert result["n_raw_points"].iloc[0] == 200
    assert isinstance(result["platform_number"].iloc[0], str)


# ---------------------------------------------------------------------------
# test_out_of_window_points_dropped
# Points at depth -50 m and +2500 m fall outside [0, 2000] and are dropped by the
# vertical cut before the raw-point count, so n_raw_points stays 200 and the OHC
# equals the plain "full" result.
# ---------------------------------------------------------------------------
def test_out_of_window_points_dropped():
    with_oow = estimate_ohc_from_raw_bins(
        make_synthetic_df("full", include_out_of_window=True)
    )
    plain = estimate_ohc_from_raw_bins(make_synthetic_df("full"))
    assert len(with_oow) == 1
    assert with_oow["n_raw_points"].iloc[0] == 200
    assert np.isclose(with_oow["ohc"].iloc[0], plain["ohc"].iloc[0], rtol=1e-9)


# ---------------------------------------------------------------------------
# test_custom_depth_range_is_respected
# The window-dropping behavior above only exercises the DEFAULT depth_min=0,
# depth_max=2000. depth_min/depth_max are themselves the "CRITICAL: Respecting
# chosen depth" params threaded from Script 02's per-layer call (Skin/Source/
# Background use different windows) -- this pins that a non-default window is
# actually honored, not just the default one. "full" coverage has one point
# every 10m from 5m to 1995m; [150, 400) contains exactly 25 of them
# (155, 165, ..., 395).
# ---------------------------------------------------------------------------
def test_custom_depth_range_is_respected():
    df = make_synthetic_df("full")
    result = estimate_ohc_from_raw_bins(df, depth_min=150, depth_max=400)
    assert len(result) == 1
    assert result["n_raw_points"].iloc[0] == 25
    assert np.isclose(
        result["ohc_per_m"].iloc[0], result["ohc"].iloc[0] / 250.0, rtol=1e-9
    )


# ---------------------------------------------------------------------------
# test_gate_excludes_when_paired_shallow_and_deep
# Unstack a shallow-only bin next to a deep-only bin (different lat bins). The
# shared column space now spans the whole water column, so the shallow row has
# NaNs in its bottom 20% (fails has_deep) and the deep row has NaNs in its top
# 10% (fails has_surface). Both are dropped -> an empty DataFrame is returned.
# ---------------------------------------------------------------------------
def test_gate_excludes_when_paired_shallow_and_deep():
    df = pd.concat([
        make_synthetic_df("shallow"),
        make_synthetic_df("deep", lat0=36.25, lon0=-123.75),
    ], ignore_index=True)
    result = estimate_ohc_from_raw_bins(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# test_gate_keeps_full_bin_when_paired_with_shallow
# Same unstack mechanism, but now a "full" bin next to a shallow-only bin. The
# full bin spans surface and deep, so it survives the coverage gate; the shallow
# bin is dropped. Exactly one row, and it is the full bin with the full OHC.
# ---------------------------------------------------------------------------
def test_gate_keeps_full_bin_when_paired_with_shallow():
    df = pd.concat([
        make_synthetic_df("full"),
        make_synthetic_df("shallow", lat0=36.25, lon0=-123.75),
    ], ignore_index=True)
    result = estimate_ohc_from_raw_bins(df)
    reference = estimate_ohc_from_raw_bins(make_synthetic_df("full"))
    assert len(result) == 1
    assert result["lat_bin"].iloc[0] == 35.5
    assert np.isclose(result["ohc"].iloc[0], reference["ohc"].iloc[0], rtol=1e-6)


# ---------------------------------------------------------------------------
# test_single_isolated_bin_always_passes_gate
# The coverage gate's surface/deep test is relative to the OBSERVED columns, so a
# lone bin (dense-but-shallow, or sparse-but-spanning) has no NaNs and always
# passes. Documents that exclusion only happens when bins are unstacked together.
# ---------------------------------------------------------------------------
def test_single_isolated_bin_always_passes_gate():
    shallow = estimate_ohc_from_raw_bins(make_synthetic_df("shallow"))
    assert not shallow.empty

    sparse = estimate_ohc_from_raw_bins(make_synthetic_df("sparse", n_sparse=12))
    assert not sparse.empty
    assert sparse["n_raw_points"].iloc[0] == 12


# ---------------------------------------------------------------------------
# test_empty_and_all_out_of_window_return_empty_frame
# Degenerate inputs must return an empty DataFrame (never None, never raise):
# (a) an input frame with no rows, (b) a frame whose only rows sit outside the
# [0, 2000] m depth window.
# ---------------------------------------------------------------------------
def test_empty_and_all_out_of_window_return_empty_frame():
    empty_in = pd.DataFrame(
        columns=["lat", "lon", "time_days", "pres", "depth", "temp", "psal", "platform_number"]
    )
    result_empty = estimate_ohc_from_raw_bins(empty_in)
    assert isinstance(result_empty, pd.DataFrame)
    assert result_empty.empty

    oow_only = pd.DataFrame([
        {
            "lat": 35.25, "lon": -124.75, "time_days": 100.0,
            "pres": float(gsw.p_from_z(-z, 35.25)), "depth": float(z),
            "temp": 10.0, "psal": 34.5, "platform_number": "p",
        }
        for z in (-50.0, 2500.0, 3000.0)
    ])
    result_oow = estimate_ohc_from_raw_bins(oow_only)
    assert isinstance(result_oow, pd.DataFrame)
    assert result_oow.empty


# ---------------------------------------------------------------------------
# test_nan_measurement_pins_current_behavior
# Characterization, not a spec: one NaN temperature row does not blank the bin --
# the GSW chain yields NaN energy density for that point, groupby-mean skips it,
# and the OHC stays within ~0.1% of the clean "full" result.
# ---------------------------------------------------------------------------
def test_nan_measurement_pins_current_behavior():
    df = make_synthetic_df("full")
    df.loc[0, "temp"] = np.nan
    result = estimate_ohc_from_raw_bins(df)
    reference = estimate_ohc_from_raw_bins(make_synthetic_df("full"))
    assert len(result) == 1
    assert np.isfinite(result["ohc"].iloc[0])
    assert np.isclose(result["ohc"].iloc[0], reference["ohc"].iloc[0], rtol=1e-3)
