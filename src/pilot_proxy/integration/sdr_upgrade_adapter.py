"""CPU-only, single-input digital SDR adapter for the upgrade frame geometry.

This adapter has no hardware API and no RF-frequency mapping. It accepts one
contiguous 2 MHz complex-IQ record with an explicit digital mixer and scale.
SciPy is required (the ``test`` extra supplies it).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import firwin, upfirdn

from pilot_proxy.detector_reference import (
    matched_filter_row_projections_cpu_reference_packed,
    quantize_complex_numpy,
    unpack_packed_complex,
)

INPUT_RATE_HZ = 2_000_000
OUTPUT_RATE_HZ = 390_625
UP = 25
DOWN = 128
FRAME_SAMPLES = 16_384
K = 128
L = 128
FIR_TAPS = 8_193
FIR_CUTOFF_HZ = 165_000.0
QUALIFIED_DIGITAL_PASSBAND_HZ = 135_000.0


def array_sha256(array: np.ndarray, dtype: str) -> str:
    """Hash canonical C-order, explicitly endian-typed array bytes."""
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class DigitalAdapterConfig:
    input_rate_hz: int
    mixer_hz: float
    target_bin: int
    sample_scale: float

    def __post_init__(self) -> None:
        if isinstance(self.input_rate_hz, bool) or self.input_rate_hz != INPUT_RATE_HZ:
            raise ValueError("This adapter requires an explicitly declared 2000000 Hz input.")
        if (not math.isfinite(self.mixer_hz)
                or abs(self.mixer_hz) >= INPUT_RATE_HZ / 2):
            raise ValueError("mixer_hz must be finite and inside the input Nyquist interval.")
        if isinstance(self.target_bin, bool) or not isinstance(self.target_bin, int):
            raise ValueError("target_bin must be a signed integer.")
        if not math.isfinite(self.sample_scale) or self.sample_scale <= 0:
            raise ValueError("sample_scale must be explicit, positive and finite.")
        for frequency in self.term_frequencies_hz:
            if abs(frequency) > QUALIFIED_DIGITAL_PASSBAND_HZ:
                raise ValueError("All three projector centers must lie inside the qualified passband.")
            if abs(self.mixer_hz + frequency) >= INPUT_RATE_HZ / 2:
                raise ValueError("A requested input projector frequency crosses input Nyquist.")
        # Keep the entire anti-alias band away from the raw input wrap boundary.
        if abs(self.mixer_hz) + OUTPUT_RATE_HZ / 2 >= INPUT_RATE_HZ / 2:
            raise ValueError("The translated output band crosses input Nyquist.")

    @property
    def term_frequencies_hz(self) -> tuple[float, float, float]:
        width = OUTPUT_RATE_HZ / K
        return tuple((self.target_bin + delta) * width for delta in (0, -2, 2))


def resampler_coefficients() -> np.ndarray:
    """Unity-DC low-pass at 50 MHz, before the upfirdn interpolation gain."""
    return firwin(
        FIR_TAPS, FIR_CUTOFF_HZ, window=("kaiser", 8.6),
        fs=INPUT_RATE_HZ * UP, scale=True,
    ).astype("<f8")


def digital_resample(
    iq: np.ndarray, config: DigitalAdapterConfig,
) -> tuple[np.ndarray, dict]:
    """Mix, anti-alias and rationally resample a contiguous finite record.

    Only output samples whose entire FIR support is inside the input are
    retained. No padding-based edge sample enters a detector frame. This is
    a batch adapter; calling it separately on chunks does not preserve state.
    """
    x = np.asarray(iq)
    if x.ndim != 1 or not np.iscomplexobj(x) or x.size == 0:
        raise ValueError("iq must be a nonempty one-dimensional complex record.")
    if not np.all(np.isfinite(x)):
        raise ValueError("iq must contain only finite samples.")
    h = resampler_coefficients()
    phase = -2j * np.pi * config.mixer_hz * np.arange(x.size) / INPUT_RATE_HZ
    mixed = np.asarray(x, dtype=np.complex128) * np.exp(phase)
    y = upfirdn(h * UP, mixed, up=UP, down=DOWN)
    first = (h.size - 1 + DOWN - 1) // DOWN
    stop = ((x.size - 1) * UP) // DOWN + 1
    if stop <= first:
        raise ValueError("Record is too short for a full-support resampler output.")
    valid = np.ascontiguousarray(y[first:stop])
    half = (h.size - 1) // 2
    return valid, {
        "input_samples": int(x.size),
        "output_samples_with_full_filter_support": int(valid.size),
        "upsample": UP, "downsample": DOWN,
        "first_untrimmed_output_index": first,
        "stop_untrimmed_output_index_exclusive": stop,
        "sample_center_time_formula_seconds": "(j * 128 - 4096) / 50000000, with j the untrimmed output index",
        "first_sample_center_seconds_from_record_start": (first * DOWN - half) / (INPUT_RATE_HZ * UP),
        "filter_group_delay_seconds": half / (INPUT_RATE_HZ * UP),
        "coefficients_sha256_float64_le": array_sha256(h, "<f8"),
        "filter_taps": int(h.size), "filter_cutoff_hz": FIR_CUTOFF_HZ,
        "filter_design": "firwin, Kaiser beta 8.6, unity DC; upfirdn coefficients multiplied by 25",
        "qualified_projector_center_passband_hz": [-QUALIFIED_DIGITAL_PASSBAND_HZ, QUALIFIED_DIGITAL_PASSBAND_HZ],
        "boundary_policy": "drop incomplete FIR support at both ends; never fill a detector frame with zeros",
        "digital_orientation": "exp(+2j*pi*f*n/input_rate) denotes positive input digital frequency",
        "mixing": "multiply by exp(-2j*pi*mixer_hz*n/input_rate); output digital frequency = input digital frequency - mixer_hz",
        "physical_rf_mapping": None,
    }


def projector_weights(config: DigitalAdapterConfig) -> tuple[np.ndarray, np.ndarray]:
    """Natural +phase weights; projection is x dot conjugate(w)."""
    n = np.arange(K)
    weights = np.exp(2j * np.pi * np.array(config.term_frequencies_hz)[:, None] * n / OUTPUT_RATE_HZ)
    return weights, quantize_complex_numpy(weights, bits=4, scale=7.0)


def adapt_record(iq: np.ndarray, config: DigitalAdapterConfig) -> dict:
    """Return exact packed CPU row projections and per-frame coarse powers.

    Frame sums pool 128 rows of one input. The ideal unfiltered, unquantized
    IID equal-power null has F(256,512), not the 2048-input array degrees of
    freedom. Filtered or measured SDR data need empirical validation.
    """
    y, timing = digital_resample(iq, config)
    count = y.size // FRAME_SAMPLES
    if count == 0:
        raise ValueError("No complete 16384-sample upgrade frame after boundary trimming.")
    used = y[:count * FRAME_SAMPLES].reshape(count, L, K)
    packed = quantize_complex_numpy(used.reshape(-1, K), 4, config.sample_scale).reshape(count, L, K)
    natural, weights = projector_weights(config)
    projections = np.stack([
        matched_filter_row_projections_cpu_reference_packed(frame, weights, 4)
        for frame in packed
    ])
    powers = (projections.astype(np.float64) ** 2).sum(axis=(2, 3))
    denominator = powers[:, 1] + powers[:, 2]
    ratio = np.full(count, np.nan)
    np.divide(2 * powers[:, 0], denominator, out=ratio, where=denominator > 0)
    ratio[(denominator == 0) & (powers[:, 0] > 0)] = np.inf
    w = unpack_packed_complex(weights, 4)
    norms = np.sum(abs(w) ** 2, axis=1)
    rounded_r = np.rint(used.real * config.sample_scale)
    rounded_i = np.rint(used.imag * config.sample_scale)
    first_time = timing["first_sample_center_seconds_from_record_start"]
    starts = first_time + np.arange(count) * FRAME_SAMPLES / OUTPUT_RATE_HZ
    row_starts = starts[:, None] + np.arange(L) * K / OUTPUT_RATE_HZ
    return {
        "config": asdict(config),
        "metadata": {
            "schema": "sdr-upgrade-digital-adapter-v1",
            "scope": "single-input CPU digital adapter; no hardware operation, physical RF mapping, array replication or calibrated null claim",
            "input_rate_hz": INPUT_RATE_HZ, "output_rate_hz": OUTPUT_RATE_HZ,
            "frame_size_samples": FRAME_SAMPLES, "K": K, "L": L,
            "num_input_streams": 1, "frame_count": count,
            "frame_duration_seconds": FRAME_SAMPLES / OUTPUT_RATE_HZ,
            "row_spacing_seconds": K / OUTPUT_RATE_HZ,
            "unused_full_support_tail_samples": int(y.size - count * FRAME_SAMPLES),
            "frame_interval_convention": "sample-center lattice; frame end is the next sample center, exclusive",
            "term_order": ["target", "lower_digital_reference", "upper_digital_reference"],
            "term_digital_frequencies_hz": list(config.term_frequencies_hz),
            "term_input_digital_frequencies_hz": [config.mixer_hz + f for f in config.term_frequencies_hz],
            "weight_quantization": "nearest, ties to even; scale 7; symmetric signed rails [-7,7]; packed two's-complement real-high/imag-low nibble",
            "weight_packed_sha256_int8": array_sha256(weights, "i1"),
            "weight_squared_norms": norms.tolist(),
            "sample_quantization": {
                "scale": config.sample_scale, "bits_per_component": 4,
                "rounding": "nearest, ties to even", "rails": [-7, 7],
                "saturated_component_count": int(np.count_nonzero(abs(rounded_r) > 7) + np.count_nonzero(abs(rounded_i) > 7)),
                "component_count": int(2 * used.size),
                "scale_estimation": "none; caller must supply a scale frozen independently of evaluation frames",
            },
            "ratio_definition": "2 * sum_rows(|z_target|^2) / sum_rows(|z_lower|^2 + |z_upper|^2), without weight-norm renormalization",
            "ratio_status": {
                "positive_denominator": int(np.count_nonzero(denominator > 0)),
                "infinite": int(np.count_nonzero(np.isinf(ratio))),
                "undefined_both_zero": int(np.count_nonzero(np.isnan(ratio))),
            },
            "ideal_iid_unquantized_equal_norm_null_dof": [2 * L, 4 * L],
            "timing": timing,
        },
        "resampled_frames": used,
        "packed_frames": packed,
        "natural_weights": natural,
        "packed_weights": weights,
        "projections_i32": projections,
        "term_power_sums": powers,
        "coarse_ratio": ratio,
        "frame_start_seconds": starts,
        "row_start_seconds": row_starts,
    }
