"""The archive results layer's product access, geometry, blocks and bootstrap."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.characterization import blocks
from pilot_proxy.products.reader import (
    COARSE_BIN_HZ, FINE_BIN_HZ, FINE_BINS, HEALTH_GATE_SCHEMA, NFFT, PSD_BIN_HZ,
    Product, fine_bin_of_hz, fine_hz_of_bin, fine_offset_to_rf_hz, fine_power_ratio, grid_residual_hz,
)

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("v5_fixture", ROOT / "tests" / "products" / "test_pilotproxy_v5.py")
v5_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v5_fixture)


@pytest.fixture
def product_path(tmp_path):
    path = v5_fixture._write_product(tmp_path / "506.npz", 36)
    # the frame-health gate reads the per-frame baseband power, which the
    # threshold fixture does not carry; add an in-range value per frame
    with np.load(path, allow_pickle=False) as product:
        frames = int(np.asarray(product["valid"]).shape[0])
    v5_fixture._replace(path, baseband_power_linear=np.full((frames, 1), 4.0))
    return path


def test_product_exposes_small_fields_without_copying_the_archive(product_path):
    with Product(product_path) as p:
        assert p.n_frames == p.valid.size == p.rejected.size
        assert p.valid.all()
        np.testing.assert_array_equal(p.rejected, p.view.rejected)
        assert p.mu0 == pytest.approx(2.0 * int(p.scalar("target_norm_sq")) / int(p.scalar("reference_norm_sum_sq")))
        finite = np.isfinite(p.level_db)
        assert finite.all()  # the fixture has positive Q on every frame
        assert np.isfinite(p.frame_time).all() and p.frame_time.shape == (p.n_frames,)
        assert (np.diff(p.frame_time[p.frame_unit_index == 0]) > 0).all()
        assert p.unit_time.shape == (p.n_frames,)
        assert p.health.schema in (HEALTH_GATE_SCHEMA, "valid_only")
        assert p.health.include.shape == p.valid.shape
        assert p.selected.sum() <= p.valid.sum()


def test_geometry_reproduces_the_scan_prediction_when_pilot_proxy_is_available(product_path):
    geometry_module = pytest.importorskip("pilot_proxy.detector_geometry")
    with Product(product_path) as p:
        g = p.geometry
        predicted = geometry_module.predicted_pilot_fine_bin(
            pilot_rf_hz=g.pilot_hz, coarse_center_hz=g.centre_hz, sample_rate_hz=390625.0,
            detector_window_samples=128, nfft=NFFT,
            spectral_sense="inverted" if g.sense < 0 else "normal", pad_factor=2)
        assert g.nominal_fine_bin == predicted


def test_geometry_axes_round_trip(product_path):
    with Product(product_path) as p:
        g = p.geometry
        assert 0 <= g.nominal_fine_bin < FINE_BINS
        assert fine_bin_of_hz(g.grid_residual_hz) == g.nominal_fine_bin
        # the nominal pilot's spectrum bin maps back to (near) zero RF offset
        rf = g.psd_rf_offset_hz([round(g.nominal_psd_bin)])
        assert abs(rf[0]) <= PSD_BIN_HZ / 2 + 1e-9
        for b in (0.0, 17.5, 8191.0, 16000.25):
            back = g.psd_bin_of_rf_offset(g.psd_rf_offset_hz([b]))
            assert back[0] == pytest.approx(b % NFFT, abs=1e-6)
        # the channel-centre line sits at (centre - pilot) from the nominal pilot
        assert g.psd_rf_offset_hz([0])[0] == pytest.approx(g.centre_line_rf_offset_hz)


def test_grid_residual_and_rf_conversion_follow_the_receiver_sense():
    # channel 36: pilot 602.309441 MHz, centre 602.343750 MHz, sense -1 -> +739.7 Hz, bin 62
    residual = grid_residual_hz(602_309_441.0, 602_343_750.0, -1)
    assert residual == pytest.approx(739.7, abs=0.1)
    assert fine_bin_of_hz(residual) == 62
    # an anchor one fine bin above the residual is one bin LOWER in RF under sense -1
    rf = fine_offset_to_rf_hz(residual + FINE_BIN_HZ, residual, -1)
    assert rf == pytest.approx(-FINE_BIN_HZ)
    # unwrapping: a fine offset half a coarse bin away wraps to the other edge, not beyond it
    rf_wrap = fine_offset_to_rf_hz(residual - COARSE_BIN_HZ + 5.0, residual, -1)
    assert rf_wrap == pytest.approx(-5.0)
    np.testing.assert_allclose(fine_hz_of_bin([0, 1, 255, 128]), [0.0, FINE_BIN_HZ, -FINE_BIN_HZ, -128 * FINE_BIN_HZ])


def test_fine_power_ratio_is_the_deployed_statistic():
    terms = np.zeros((2, 3, 4), dtype=np.uint64)
    terms[:, 0] = [[10, 20, 30, 0], [7, 7, 7, 7]]
    terms[:, 1] = [[5, 5, 5, 0], [3, 4, 0, 1]]
    terms[:, 2] = [[5, 15, 25, 0], [4, 3, 0, 6]]
    ratio = fine_power_ratio(terms)
    np.testing.assert_allclose(ratio[0], [2.0, 2.0, 2.0, 0.0])
    np.testing.assert_allclose(ratio[1], [2.0, 2.0, 0.0, 2.0])
    with pytest.raises(ValueError):
        fine_power_ratio(np.zeros((2, 2, 4)))


def test_fine_terms_rows_match_the_full_member(product_path):
    with Product(product_path) as p:
        rows = np.array([0, 5, 7])
        np.testing.assert_array_equal(p.fine_terms(rows), p.fine_terms_all()[rows])


# ------------------------------------------------------------------ blocks
def _synthetic_units(n_units=40, frames_per_unit=5, start=1.6e9, spacing=86400.0 * 3):
    unit_index = np.repeat(np.arange(n_units), frames_per_unit)
    unit_time0 = start + np.arange(n_units) * spacing
    unit_time = unit_time0[unit_index]
    return unit_index, unit_time


def test_month_support_applies_the_section_8_1_minimums():
    unit_index, unit_time = _synthetic_units(n_units=12, frames_per_unit=6, spacing=86400.0 * 2)  # ~24 days
    months = blocks.month_support(unit_time, unit_index, np.ones(unit_index.size, bool))
    assert months and all(m.frames >= 30 and m.units >= 5 and m.days >= 3 for m in months)
    # too few acquisitions in a month -> not populated
    few = blocks.month_support(unit_time, unit_index, np.ones(unit_index.size, bool), min_units=100)
    assert few == []
    label = blocks.month_label(2026 * 12 + 8)
    assert label == "2026-09"


def test_split_blocks_is_chronological_whole_acquisition_and_balanced():
    unit_index, unit_time = _synthetic_units(n_units=41, frames_per_unit=3)
    selected = np.ones(unit_index.size, bool)
    selected[unit_index == 3] = False                       # an excluded acquisition
    split = blocks.split_blocks(unit_index, unit_time, selected, minimum_months=1)
    assert not (split.calibration & split.evaluation).any()
    assert (split.calibration | split.evaluation).sum() == selected.sum()
    assert abs(split.calibration_frames - split.evaluation_frames) <= 3
    # every acquisition is whole on one side, and calibration precedes evaluation in time
    for u in np.unique(unit_index[selected]):
        rows = unit_index == u
        assert split.calibration[rows].all() or split.evaluation[rows].all()
    assert unit_time[split.calibration].max() < unit_time[split.evaluation].min() == split.boundary_time
    assert split.calibration_units + split.evaluation_units == 40
    assert split.status == "supported"


def test_split_blocks_reports_insufficient_support_and_empty_selection():
    unit_index, unit_time = _synthetic_units(n_units=6, frames_per_unit=2)
    split = blocks.split_blocks(unit_index, unit_time, np.ones(unit_index.size, bool), minimum_months=1)
    assert split.status == "insufficient_support" and "populated months" in split.detail
    empty = blocks.split_blocks(unit_index, unit_time, np.zeros(unit_index.size, bool))
    assert empty.status == "empty" and not empty.calibration.any()


def test_block_bootstrap_resamples_acquisitions_with_a_recorded_seed():
    unit_index, unit_time = _synthetic_units(n_units=30, frames_per_unit=4)
    flag = (unit_index % 3 == 0)                            # a per-acquisition property
    selected = np.ones(unit_index.size, bool)
    stat = lambda w: blocks.weighted_fraction(w, flag)
    a = blocks.block_bootstrap(unit_index, selected, stat, replicates=300, seed=7)
    b = blocks.block_bootstrap(unit_index, selected, stat, replicates=300, seed=7)
    assert a.estimate == pytest.approx(flag.mean())
    assert a.values == b.values and a.blocks == 30
    assert a.low <= a.estimate <= a.high
    # frames are never resampled individually: every replicate's weights are constant within an acquisition
    with pytest.raises(ValueError, match="at least"):
        blocks.block_bootstrap(unit_index, unit_index < 3, stat)      # three acquisitions only


def test_weighted_quantile_ignores_nan_and_honours_weights():
    values = np.array([1.0, 2.0, np.nan, 4.0])
    assert blocks.weighted_quantile(np.array([1, 1, 1, 1]), values, 0.5) == 2.0
    assert blocks.weighted_quantile(np.array([0, 0, 0, 3]), values, 0.5) == 4.0
    assert np.isnan(blocks.weighted_quantile(np.zeros(4), values, 0.5))
    assert np.isnan(blocks.weighted_fraction(np.zeros(3), np.ones(3, bool)))
