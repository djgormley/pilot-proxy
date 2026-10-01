"""Digital signal, alias-boundary, arithmetic and time-lattice checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import freqz

from pilot_proxy.integration.sdr_upgrade_adapter import (
    DigitalAdapterConfig, FRAME_SAMPLES, INPUT_RATE_HZ, OUTPUT_RATE_HZ,
    K, L, adapt_record, digital_resample, projector_weights,
    resampler_coefficients,
)


def scalar_projections(packed_rows, packed_weights):
    """Independent scalar nibble decode and integer complex dot product."""
    def decode(value):
        value = int(value) & 255
        real, imag = value >> 4, value & 15
        return (real - 16 if real >= 8 else real,
                imag - 16 if imag >= 8 else imag)

    output = np.zeros((3, len(packed_rows), 2), dtype=np.int32)
    for term in range(3):
        weights = [decode(v) for v in packed_weights[term]]
        for row_index, row in enumerate(packed_rows):
            real = imag = 0
            for sample, (wr, wi) in zip(row, weights):
                xr, xi = decode(sample)
                real += xr * wr + xi * wi
                imag += xi * wr - xr * wi
            output[term, row_index] = real, imag
    return output


@pytest.mark.parametrize("target", [-4, 4])
@pytest.mark.parametrize("term", [0, 1, 2])
def test_signed_target_and_reference_placement(target, term):
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, 100_000.0, target, 5.0)
    frequency = cfg.mixer_hz + cfg.term_frequencies_hz[term]
    x = np.exp(2j * np.pi * frequency * np.arange(100_000) / INPUT_RATE_HZ + .3j)
    result = adapt_record(x, cfg)
    power = result["term_power_sums"][0]
    assert np.argmax(power) == term
    assert power[term] > 1000 * max(power[i] for i in range(3) if i != term)
    assert result["metadata"]["sample_quantization"]["saturated_component_count"] == 0
    np.testing.assert_array_equal(
        result["projections_i32"][0],
        scalar_projections(result["packed_frames"][0], result["packed_weights"]),
    )


def test_resampler_matches_direct_convolution_and_time_origin():
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, -23_456.0, 0, 3.0)
    rng = np.random.default_rng(1109)
    x = rng.normal(size=1000) + 1j * rng.normal(size=1000)
    y, timing = digital_resample(x, cfg)
    h = resampler_coefficients()
    mixed = x * np.exp(-2j * np.pi * cfg.mixer_hz * np.arange(x.size) / INPUT_RATE_HZ)
    first = timing["first_untrimmed_output_index"]
    for index in [0, 1, 17, len(y) - 1]:
        j = first + index
        total = 0j
        for n in range(len(x)):
            tap = j * 128 - n * 25
            if 0 <= tap < len(h):
                total += 25 * h[tap] * mixed[n]
        assert y[index] == pytest.approx(total, abs=1e-13)
        assert j * 128 >= len(h) - 1
        assert j * 128 <= (x.size - 1) * 25
    assert timing["first_sample_center_seconds_from_record_start"] == pytest.approx(81.92e-6)


def test_passband_and_entire_alias_stopband():
    h = resampler_coefficients()
    frequencies, response = freqz(h, worN=262144, fs=INPUT_RATE_HZ * 25)
    assert np.max(abs(abs(response[frequencies <= 135_000]) - 1)) < 5e-5
    assert np.max(abs(response[frequencies >= OUTPUT_RATE_HZ / 2])) < 4e-5


@pytest.mark.parametrize("offset", [-225_000.0, 225_000.0])
def test_out_of_band_tone_does_not_alias_into_output(offset):
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, 100_000.0, 0, 5.0)
    x = np.exp(2j * np.pi * (cfg.mixer_hz + offset) * np.arange(100_000) / INPUT_RATE_HZ)
    y, _ = digital_resample(x, cfg)
    assert np.mean(abs(y) ** 2) < 1e-8


def test_frame_and_row_time_lattice_and_explicit_tail():
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, 0.0, 0, 1.0)
    result = adapt_record(np.ones(200_000, dtype=np.complex64), cfg)
    assert result["packed_frames"].shape == (2, L, K)
    assert FRAME_SAMPLES == K * L == 16_384
    assert result["metadata"]["frame_duration_seconds"] == pytest.approx(.04194304)
    np.testing.assert_allclose(np.diff(result["frame_start_seconds"]), .04194304)
    np.testing.assert_allclose(np.diff(result["row_start_seconds"], axis=1), .00032768)
    assert result["metadata"]["unused_full_support_tail_samples"] > 0
    assert result["metadata"]["timing"]["physical_rf_mapping"] is None
    assert result["metadata"]["ideal_iid_unquantized_equal_norm_null_dof"] == [256, 512]


def test_undefined_zero_record_is_not_silently_noise_or_detection():
    result = adapt_record(np.zeros(100_000, dtype=complex), DigitalAdapterConfig(INPUT_RATE_HZ, 0., 0, 10.))
    assert np.isnan(result["coarse_ratio"]).all()
    assert result["metadata"]["ratio_status"]["undefined_both_zero"] == 1


def test_sample_saturation_is_counted_and_rails_are_symmetric():
    result = adapt_record(np.full(100_000, 100 + 100j), DigitalAdapterConfig(INPUT_RATE_HZ, 0., 0, 10.))
    assert result["metadata"]["sample_quantization"]["saturated_component_count"] == 2 * FRAME_SAMPLES
    assert np.all(result["packed_frames"].view(np.uint8) == 0x77)


@pytest.mark.parametrize("kwargs", [
    {"input_rate_hz": 1_999_999}, {"sample_scale": 0.},
    {"mixer_hz": 900_000.}, {"mixer_hz": float("nan")},
    {"target_bin": 43}, {"target_bin": 4.5}, {"target_bin": True},
])
def test_invalid_or_unqualified_config_refused(kwargs):
    settings = dict(input_rate_hz=INPUT_RATE_HZ, mixer_hz=0., target_bin=0, sample_scale=1.)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        DigitalAdapterConfig(**settings)


@pytest.mark.parametrize("x", [np.ones(10, dtype=complex), np.ones(1000), np.full(1000, np.nan + 1j)])
def test_incomplete_or_invalid_input_refused(x):
    with pytest.raises(ValueError):
        adapt_record(x, DigitalAdapterConfig(INPUT_RATE_HZ, 0., 0, 1.))


def test_natural_projection_is_independently_signed_dft_bin():
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, 0., -4, 2.)
    weights, _ = projector_weights(cfg)
    rng = np.random.default_rng(4109)
    rows = rng.normal(size=(L, K)) + 1j * rng.normal(size=(L, K))
    projected = rows @ weights.conj().T
    dft = np.fft.fft(rows, axis=1)
    np.testing.assert_allclose(projected, dft[:, np.array([-4, -6, -2]) % K], atol=1e-12)


def test_tone_report_explicitly_labels_zero_other_power_lower_bound():
    path = Path(__file__).parents[2] / "tools/validate_sdr_upgrade_adapter.py"
    spec = importlib.util.spec_from_file_location("validate_sdr_upgrade_adapter", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    zero_other = tool.tone_power_comparison(np.array([17., 0., 0.]), 0)
    assert zero_other["largest_other_term_power"] == 0.
    assert zero_other["denominator_floor_code_squared"] == 1.
    assert zero_other["desired_to_other_power_lower_bound_with_unit_floor"] == 17.
    assert "not the exact ratio" in zero_other["power_comparison_definition"]
    positive_other = tool.tone_power_comparison(np.array([17., 2., 1.]), 0)
    assert positive_other["desired_to_other_power_lower_bound_with_unit_floor"] == 8.5
