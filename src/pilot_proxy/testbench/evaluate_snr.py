#!/usr/bin/env python3
# coding=utf-8
"""Evaluate DTV data-shelf SNR estimates from the CUDA F-statistic kernel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from pilot_proxy.detector_geometry import (  # noqa: E402
    DetectorFrameLayout,
    apply_spectral_sense_to_detector_matrix,
    build_stream_map,
    flatten_feed_channel_streams,
    stream_time_block_to_detector_matrix,
)
from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz  # noqa: E402
from pilot_proxy.detector_contract import (
    normalized_positive_excess,
    weight_term_norms_sq,
)
from pilot_proxy.detector_reference import (  # noqa: E402
    REFERENCE_LOWER_TERM_INDEX,
    REFERENCE_TARGET_TERM_INDEX,
    REFERENCE_UPPER_TERM_INDEX,
    coarse_power_ratio_cpu_reference,
    coarse_power_ratio_cpu_reference_packed,
)
from pilot_proxy.detector_weights import DetectorWeightBank  # noqa: E402
from pilot_proxy.dtv_units import (  # noqa: E402
    DB_LINEAR_BASE,
    COARSE_POWER_RATIO_SCALE,
    DB_POWER_FACTOR,
    DEFAULT_THRESHOLD_MAX_DENOMINATOR,
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    UNIT_DATA_SHELF_POWER,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
    composite_to_data_shelf_snr_correction_db,
    pilot_capture_efficiency_db,
    power_terms_to_normalized_coarse_power_ratio,
    power_terms_to_normalized_coarse_power_ratio_db,
    power_terms_to_raw_pilot_excess,
    normalized_pilot_excess_to_db,
    normalize_coarse_power_ratio,
    normalized_coarse_power_ratio_to_pilot_excess,
    power_terms_to_normalized_pilot_excess,
    power_terms_to_coarse_power_ratio,
    pilot_to_data_power_ratio,
    pilot_excess_db_to_data_shelf_snr_db,
    pilot_excess_to_data_shelf_metadata,
    spreading_loss_db_from_bin_enbw_hz,
    data_shelf_snr_threshold_fields,
)
from pilot_proxy.json_utils import write_json_strict  # noqa: E402
from pilot_proxy.kernel import FStatKernel  # noqa: E402
from pilot_proxy.integration import QUANTIZATION_SCALE_MODE_GLOBAL  # noqa: E402
from pilot_proxy.integration.packing import (  # noqa: E402
    pack_channelized_streams_for_detector,
)
from pilot_proxy.paths import (  # noqa: E402
    DEFAULT_LIB_PATH,
    DEFAULT_WEIGHTS_PATH,
    GENERATED_DIR,
    SOURCE_CHECKOUT_ROOT,
    resolve_user_path,
)
from pilot_proxy.provenance import file_sha256, sidecar_manifest_path  # noqa: E402
from pilot_proxy.reference_channelizer import (  # noqa: E402
    REFERENCE_ADC_SAMPLE_RATE_HZ,
    REFERENCE_BAND_LOWER_HZ,
    REFERENCE_PFB_FFT_SIZE,
    REFERENCE_PFB_TAPS,
    ReferenceChannelizerSpec,
    apply_reference_archive_phase_convention,
    channelize_real_blocks_to_reference_channels,
    channelize_real_blocks_to_reference_channels_gpu,
    complex_envelope_to_real_adc_blocks,
    complex_envelope_to_real_adc_blocks_gpu,
    nearest_reference_channel_index,
    sinc_hamming_pfb_response,
)
from pilot_proxy.result_schema import (  # noqa: E402
    COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
    RESULT_SCHEMA_TOKEN,
    result_schema_object,
)
from pilot_proxy.testbench.quantize import (  # noqa: E402
    ATSC_CHANNEL_WIDTH_HZ,
    ATSC_PILOT_OFFSET_HZ,
    DEFAULT_DTV_PILOT_HZ,
    DEFAULT_FRAME_SIZE_SAMPLES,
    GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
    LOCKED_BITS_PER_COMPONENT,
    LOCKED_DETECTOR_WINDOW_SAMPLES,
)
from pilot_proxy.integration.packing import estimate_complex_scale  # noqa: E402
from pilot_proxy.secondary_python import (  # noqa: E402
    package_only_pythonpath,
    prepend_pythonpath,
)

HZ_PER_MHZ = 1.0e6
HALF_SCALE = 2.0
COMPLEX_COMPONENT_COUNT = 2.0
DEFAULT_EVALUATOR_SEED = 12345
DEFAULT_NOISE_TRIALS = 3
DEFAULT_NUM_INPUT_STREAMS = 4  # deployment-standard trial: 4 streams x 128 rows = 512
# (was 1; a silent 1-stream default cost a debugging night on 2026-07-19)
DEFAULT_GNURADIO_PYTHON = "/usr/bin/python3"
DEFAULT_CLIP_SIGMA = 3.0
DEFAULT_SNR_SWEEP_MIN_DB = -45.0
DEFAULT_SNR_SWEEP_MAX_DB = 21.0
DEFAULT_SNR_SWEEP_STEP_DB = 3.0
SUPPORTED_SNR_MIN_DB = -60.0
SUPPORTED_SNR_MAX_DB = 60.0
DEFAULT_CHANNEL_FREQUENCY_OFFSET_HZ = 0.0
STANDARD_FREQUENCY_OFFSET_SWEEP_HZ = (-1_000.0, 0.0, 1_000.0)
DEFAULT_CHANNEL_GAIN_DB = 0.0
DEFAULT_CHANNEL_PHASE_DEG = 0.0
DB_AMPLITUDE_FACTOR = 20.0
DEGREES_PER_HALF_TURN = 180.0
TWO_PI = 2.0 * math.pi
SNR_RANGE_EPSILON = 1e-12
CSV_SNR_LABEL_PRECISION = 3


def _artifact_identity(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    digest = file_sha256(resolved)
    if digest is None:
        raise FileNotFoundError(resolved)
    recorded_path = str(resolved)
    if SOURCE_CHECKOUT_ROOT is not None:
        try:
            recorded_path = resolved.relative_to(
                SOURCE_CHECKOUT_ROOT.resolve()
            ).as_posix()
        except ValueError:
            pass
    return {"path": recorded_path, "sha256": digest}


def _weight_coefficients_identity(weights: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(weights)
    return {
        "dtype": str(values.dtype),
        "shape": [int(value) for value in values.shape],
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _nanmean_or_nan(values: np.ndarray) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _nanstd_or_nan(values: np.ndarray) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


def _nanmax_or_nan(values: np.ndarray) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def required_iq_samples(
    *,
    iq_sample_rate_hz: float,
    adc_sample_rate_hz: float,
    num_output_samples: int,
    pfb_taps: int = REFERENCE_PFB_TAPS,
    pfb_fft_size: int = REFERENCE_PFB_FFT_SIZE,
) -> int:
    """Return input IQ samples needed for the requested channelizer output."""
    n_blocks = int(num_output_samples) + int(pfb_taps) - 1
    total_adc_samples = n_blocks * int(pfb_fft_size)
    last_source_position = (
        (total_adc_samples - 1) * float(iq_sample_rate_hz) / float(adc_sample_rate_hz)
    )
    return int(math.ceil(last_source_position)) + 1


def _signal_and_noise_power_for_snr(
    signal: np.ndarray,
    *,
    snr_db: float,
    sample_rate_hz: float | None = None,
    snr_bandwidth_hz: float | None = None,
) -> tuple[np.ndarray, float, float]:
    clean = np.asarray(signal, dtype=np.complex64)
    clean_signal_power = float(np.mean(np.abs(clean.astype(np.complex64)) ** 2))
    if not np.isfinite(clean_signal_power) or clean_signal_power <= 0.0:
        raise ValueError("signal power must be positive and finite.")
    noise_power = clean_signal_power / float(
        DB_LINEAR_BASE ** (float(snr_db) / DB_POWER_FACTOR)
    )
    if sample_rate_hz is not None or snr_bandwidth_hz is not None:
        if sample_rate_hz is None or snr_bandwidth_hz is None:
            raise ValueError(
                "sample_rate_hz and snr_bandwidth_hz must be provided together."
            )
        if sample_rate_hz <= 0.0 or snr_bandwidth_hz <= 0.0:
            raise ValueError("sample_rate_hz and snr_bandwidth_hz must be positive.")
        noise_power *= float(sample_rate_hz) / float(snr_bandwidth_hz)
    return clean, clean_signal_power, noise_power


def add_complex_awgn_for_snr(
    signal: np.ndarray,
    *,
    snr_db: float,
    rng: np.random.Generator,
    sample_rate_hz: float | None = None,
    snr_bandwidth_hz: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Add complex AWGN to match a requested signal/noise-band power ratio."""
    clean, clean_signal_power, noise_power = _signal_and_noise_power_for_snr(
        signal,
        snr_db=snr_db,
        sample_rate_hz=sample_rate_hz,
        snr_bandwidth_hz=snr_bandwidth_hz,
    )
    component_sigma = math.sqrt(noise_power / COMPLEX_COMPONENT_COUNT)
    noise = rng.normal(0.0, component_sigma, clean.shape) + 1j * rng.normal(
        0.0,
        component_sigma,
        clean.shape,
    )
    return (
        np.asarray(clean + noise.astype(np.complex64), dtype=np.complex64),
        clean_signal_power,
        noise_power,
    )


def add_gnuradio_awgn_for_snr(
    signal: np.ndarray,
    *,
    input_iq_path: Path,
    output_iq_path: Path,
    snr_db: float,
    seed: int,
    gnuradio_python: str,
    sample_rate_hz: float,
    snr_bandwidth_hz: float,
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    """Add AWGN with GNU Radio analog.noise_source_c in a helper process."""
    caller_cwd = Path.cwd()
    input_iq_path = resolve_user_path(input_iq_path, relative_to=caller_cwd)
    output_iq_path = resolve_user_path(output_iq_path, relative_to=caller_cwd)
    clean, clean_signal_power, noise_power = _signal_and_noise_power_for_snr(
        signal,
        snr_db=snr_db,
        sample_rate_hz=sample_rate_hz,
        snr_bandwidth_hz=snr_bandwidth_hz,
    )
    metadata_path = output_iq_path.with_suffix(output_iq_path.suffix + ".json")
    cmd = [
        str(gnuradio_python),
        "-m",
        "pilot_proxy.testbench.add_awgn",
        "--input-iq",
        str(input_iq_path),
        "--output-iq",
        str(output_iq_path),
        "--metadata-json",
        str(metadata_path),
        "--num-samples",
        str(clean.size),
        "--snr-db",
        str(float(snr_db)),
        "--sample-rate-hz",
        str(float(sample_rate_hz)),
        "--snr-bandwidth-hz",
        str(float(snr_bandwidth_hz)),
        "--seed",
        str(int(seed)),
    ]
    with package_only_pythonpath(gnuradio_python) as package_bridge:
        env = os.environ.copy()
        if package_bridge is not None:
            prepend_pythonpath(env, package_bridge)
        elif SOURCE_CHECKOUT_ROOT is not None:
            prepend_pythonpath(env, SOURCE_CHECKOUT_ROOT / "src")
        env["PYTHONNOUSERSITE"] = "1"
        result = subprocess.run(
            cmd,
            cwd=caller_cwd,
            env=env,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "GNU Radio AWGN helper failed. "
            f"Command: {' '.join(cmd)}\n{details}"
        )
    noisy = np.fromfile(output_iq_path, dtype=np.complex64)
    if noisy.size != clean.size:
        raise RuntimeError(
            "GNU Radio AWGN helper wrote "
            f"{noisy.size} samples; expected {clean.size}."
        )
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return np.ascontiguousarray(noisy), clean_signal_power, noise_power, metadata


def apply_channel_impairments(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    frequency_offset_hz: float = DEFAULT_CHANNEL_FREQUENCY_OFFSET_HZ,
    gain_db: float = DEFAULT_CHANNEL_GAIN_DB,
    phase_deg: float = DEFAULT_CHANNEL_PHASE_DEG,
) -> np.ndarray:
    """Apply deterministic lightweight channel effects before AWGN injection."""
    clean = np.asarray(signal, dtype=np.complex64)
    sample_rate = float(sample_rate_hz)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    gain = DB_LINEAR_BASE ** (float(gain_db) / DB_AMPLITUDE_FACTOR)
    phase_rad = float(phase_deg) * math.pi / DEGREES_PER_HALF_TURN
    if (
        float(frequency_offset_hz) == DEFAULT_CHANNEL_FREQUENCY_OFFSET_HZ
        and float(gain_db) == DEFAULT_CHANNEL_GAIN_DB
        and float(phase_deg) == DEFAULT_CHANNEL_PHASE_DEG
    ):
        return np.ascontiguousarray(clean)
    sample_index = np.arange(clean.size, dtype=np.float64)
    phase = (
        TWO_PI * float(frequency_offset_hz) * sample_index / sample_rate
        + phase_rad
    )
    rotation = np.exp(1j * phase).astype(np.complex64)
    return np.ascontiguousarray(clean * np.complex64(gain) * rotation)


def _resolve_rf_center_hz(args: argparse.Namespace) -> float:
    if args.rf_center_mhz is not None:
        return float(args.rf_center_mhz) * HZ_PER_MHZ
    pilot_hz = float(args.dtv_pilot_mhz) * HZ_PER_MHZ
    return pilot_hz + (
        ATSC_CHANNEL_WIDTH_HZ / HALF_SCALE - float(args.atsc_pilot_offset_hz)
    )


def _requested_snr_shelf_values(args: argparse.Namespace) -> list[float]:
    values = [float(v) for v in args.requested_data_shelf_snr_db or []]
    if args.snr_start_db is not None:
        if args.snr_stop_db is None or args.snr_step_db is None:
            raise SystemExit(
                "--snr-start-db requires --snr-stop-db and --snr-step-db."
            )
        if args.snr_step_db == 0.0:
            raise SystemExit("--snr-step-db must be non-zero.")
        current = float(args.snr_start_db)
        stop = float(args.snr_stop_db)
        step = float(args.snr_step_db)
        compare = (
            (lambda a, b: a <= b + SNR_RANGE_EPSILON)
            if step > 0
            else (
                lambda a, b: a >= b - SNR_RANGE_EPSILON
            )
        )
        while compare(current, stop):
            values.append(float(current))
            current += step
    if not values:
        current = DEFAULT_SNR_SWEEP_MIN_DB
        while current <= DEFAULT_SNR_SWEEP_MAX_DB + SNR_RANGE_EPSILON:
            values.append(float(current))
            current += DEFAULT_SNR_SWEEP_STEP_DB
    for value in values:
        if (
            value < SUPPORTED_SNR_MIN_DB - SNR_RANGE_EPSILON
            or value > SUPPORTED_SNR_MAX_DB + SNR_RANGE_EPSILON
        ):
            raise SystemExit(
                "Testbench SNR values must be in the supported range "
                f"[{SUPPORTED_SNR_MIN_DB:g}, {SUPPORTED_SNR_MAX_DB:g}] dB; "
                f"got {value:g} dB."
            )
    return values


def _frequency_offset_values(args: argparse.Namespace) -> list[float]:
    values = [float(v) for v in args.frequency_offset_hz or []]
    if args.standard_frequency_offset_sweep:
        values.extend(float(v) for v in STANDARD_FREQUENCY_OFFSET_SWEEP_HZ)
    if not values:
        values = [DEFAULT_CHANNEL_FREQUENCY_OFFSET_HZ]
    unique: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(float(value), 9)
        if key in seen:
            continue
        seen.add(key)
        unique.append(float(value))
    return unique


def _ideal_float_weights_from_layout(
    selected_weight_layout: dict[str, Any],
    *,
    detector_window_samples: int,
) -> np.ndarray:
    """Build unquantized complex DFT weights from manifest target frequencies."""
    keys = [
        "target_normalized_frequency",
        "lower_reference_normalized_frequency",
        "upper_reference_normalized_frequency",
    ]
    if not all(key in selected_weight_layout for key in keys):
        raise ValueError("selected weight layout lacks normalized frequencies.")
    sample_index = np.arange(int(detector_window_samples), dtype=np.float64)
    rows = [
        np.exp(-1j * TWO_PI * float(selected_weight_layout[key]) * sample_index)
        for key in keys
    ]
    return np.ascontiguousarray(np.stack(rows).astype(np.complex128))


def assert_clean_pilot_lands_on_target(
    clean_streams: np.ndarray,
    *,
    selected_weight_layout: dict[str, Any],
    cpu_float_weights: np.ndarray,
    detector_window_samples: int,
    frame_size_samples: int,
    spectral_sense: str,
    minimum_normalized_ratio: float = 8.0,
) -> dict[str, Any]:
    """Fail loudly when the detector statistic cannot see the noise-free pilot.

    Every quantity downstream is a ratio of the target term to its references,
    so a configuration whose clean line excites the wrong bin yields F ~= 1 at
    every SNR and produces a detection curve made entirely of noise --
    silently, and with the requested SNR still tracking perfectly. The check
    is made through the same CPU-float statistic the sweep reports rather than
    an independent FFT, because the two can disagree about spectral
    conventions while each looks internally consistent. One clean
    channelization at startup turns a curve made of noise into an error.
    """
    rows = _float_streams_for_reference(
        clean_streams,
        frame_size_samples=int(frame_size_samples),
        detector_window_samples=int(detector_window_samples),
        spectral_sense=str(spectral_sense),
    )
    fstat, _ = coarse_power_ratio_cpu_reference(rows, cpu_float_weights)
    target_norm_sq = float(
        np.sum(np.abs(cpu_float_weights[REFERENCE_TARGET_TERM_INDEX]) ** 2)
    )
    reference_norm_sum_sq = float(
        np.sum(np.abs(cpu_float_weights[REFERENCE_LOWER_TERM_INDEX]) ** 2)
        + np.sum(np.abs(cpu_float_weights[REFERENCE_UPPER_TERM_INDEX]) ** 2)
    )
    null_ratio = float(
        COARSE_POWER_RATIO_SCALE * target_norm_sq / reference_norm_sum_sq
    )
    normalized = float(normalize_coarse_power_ratio(fstat, null_ratio))
    matrix = np.asarray(rows).reshape(-1, int(detector_window_samples))
    statistic_spectrum = (np.abs(np.fft.fft(np.conj(matrix), axis=1)) ** 2).mean(
        axis=0
    )
    observed = int(np.argmax(statistic_spectrum))
    target_bin = int(
        round(
            float(selected_weight_layout["target_normalized_frequency"])
            * int(detector_window_samples)
        )
    ) % int(detector_window_samples)
    report = {
        "normalized_coarse_power_ratio": normalized,
        "target_bin": target_bin,
        "observed_statistic_peak_bin": observed,
        "spectral_sense": str(spectral_sense),
    }
    if observed != target_bin or not normalized >= float(minimum_normalized_ratio):
        raise SystemExit(
            "evaluate-snr: the clean pilot is not in the detector's target "
            f"bin. The noise-free waveform yields a normalized coarse power "
            f"ratio of {normalized:.2f} (need >= "
            f"{float(minimum_normalized_ratio):.1f}) and the statistic's "
            f"strongest bin is {observed} while the weight layout targets bin "
            f"{target_bin}; both must agree, or a straddling, attenuated line "
            "would be recorded as the on-target pilot. Every detection rate "
            "below would describe noise. Try the opposite --spectral-sense, "
            "toggle --reference-archive-phase, or check --physical-channel "
            "against the weight bank."
        )
    return report


def _float_streams_for_reference(
    streams: np.ndarray,
    *,
    frame_size_samples: int,
    detector_window_samples: int,
    spectral_sense: str,
) -> np.ndarray:
    """Convert unquantized streams to detector rows for the CPU float path."""
    matrix = stream_time_block_to_detector_matrix(
        np.asarray(streams)[:, : int(frame_size_samples)],
        detector_window_samples=int(detector_window_samples),
    )
    return apply_spectral_sense_to_detector_matrix(
        matrix,
        spectral_sense=spectral_sense,
    )


def _pack_streams_for_kernel(
    streams: np.ndarray,
    *,
    frame_size_samples: int,
    detector_window_samples: int,
    bits: int,
    scale: float,
    spectral_sense: str,
) -> np.ndarray:
    feed_channel_streams = np.asarray(streams)[:, np.newaxis, :]
    packed_input = pack_channelized_streams_for_detector(
        feed_channel_streams,
        frame_size_samples=int(frame_size_samples),
        detector_window_samples=int(detector_window_samples),
        spectral_sense=spectral_sense,
        quantization_scale_mode=QUANTIZATION_SCALE_MODE_GLOBAL,
        clip_sigma=DEFAULT_CLIP_SIGMA,
        bits_per_component=int(bits),
        scale=float(scale),
    )
    return np.ascontiguousarray(packed_input.packed[0])


def _kernel_measurements(
    *,
    cp: Any,
    kernel: FStatKernel,
    packed: np.ndarray,
    weights: np.ndarray,
    pilot_below_data_db: float,
    bin_enbw_hz: float,
    pilot_capture_efficiency: float,
    dtv_bandwidth_hz: float,
    threshold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run diagnostic float and exact uint64 power readbacks."""
    d_in = cp.asarray(packed)
    d_diag = cp.zeros(1, dtype=cp.float32)
    d_powers = cp.zeros(int(kernel.specs.N), dtype=cp.uint64)
    d_mask_num = cp.zeros(1, dtype=cp.uint64)
    d_mask_den = cp.zeros(1, dtype=cp.uint64)
    d_mask = cp.zeros(1, dtype=cp.uint8)
    d_overflow = cp.zeros(1, dtype=cp.uint32)
    handle = kernel.create_raw(d_in.shape[0], d_in.data.ptr, d_diag.data.ptr)
    try:
        kernel.compute_diagnostic_float(handle, weights.ctypes.data)
        cp.cuda.Device().synchronize()
        diagnostic_float = float(d_diag[0].get())

        kernel.compute_powers_u64(handle, weights.ctypes.data, d_powers.data.ptr)
        cp.cuda.Device().synchronize()
        powers = cp.asnumpy(d_powers).astype(np.uint64, copy=False)

        mask = 0
        overflow = 0
        if threshold is not None:
            if getattr(kernel, "_has_numden_mask_rational_half_checked", False):
                kernel.compute_numden_mask_rational_half_checked(
                    handle,
                    weights.ctypes.data,
                    threshold["threshold_half_num"],
                    threshold["threshold_half_den"],
                    d_mask_num.data.ptr,
                    d_mask_den.data.ptr,
                    d_mask.data.ptr,
                    d_overflow.data.ptr,
                )
                cp.cuda.Device().synchronize()
                overflow = int(cp.asnumpy(d_overflow)[0])
            else:
                kernel.compute_numden_mask_rational_half(
                    handle,
                    weights.ctypes.data,
                    threshold["threshold_half_num"],
                    threshold["threshold_half_den"],
                    d_mask_num.data.ptr,
                    d_mask_den.data.ptr,
                    d_mask.data.ptr,
                )
                cp.cuda.Device().synchronize()
                overflow = 0
            mask = int(cp.asnumpy(d_mask)[0])
    finally:
        kernel.destroy(handle)

    return _measurements_from_powers(
        diagnostic_float=diagnostic_float,
        p_target=int(powers[REFERENCE_TARGET_TERM_INDEX]),
        p_ref_lower=int(powers[REFERENCE_LOWER_TERM_INDEX]),
        p_ref_upper=int(powers[REFERENCE_UPPER_TERM_INDEX]),
        weights=weights,
        pilot_below_data_db=pilot_below_data_db,
        bin_enbw_hz=bin_enbw_hz,
        pilot_capture_efficiency=pilot_capture_efficiency,
        dtv_bandwidth_hz=dtv_bandwidth_hz,
        threshold=threshold,
        mask=mask,
        overflow=overflow,
    )


def _cpu_reference_measurements(
    *,
    packed: np.ndarray,
    weights: np.ndarray,
    bits: int,
    pilot_below_data_db: float,
    bin_enbw_hz: float,
    pilot_capture_efficiency: float,
    dtv_bandwidth_hz: float,
    threshold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CPU exact-integer sibling of ``_kernel_measurements``.

    Uses the validated CPU reference (``coarse_power_ratio_cpu_reference_packed``) for the
    power sums and Python-integer arithmetic for the rational-half threshold
    decision, so the fields are the kernel's semantics without a GPU. The
    kernel <-> reference equivalence itself is CI-gated by the kernel parity
    suite; a small same-seed GPU spot check ties a CPU-produced sweep to the
    deployed kernel (see docs/PUBLICATION_VALIDATION.md@30e50c2, item 2).
    """
    fstat, sums = coarse_power_ratio_cpu_reference_packed(packed, weights, int(bits))
    p_target = int(round(float(sums[0])))
    p_ref_lower = int(round(float(sums[1])))
    p_ref_upper = int(round(float(sums[2])))
    diagnostic_float = float(np.float32(fstat))
    mask = 0
    overflow = 0
    if threshold is not None:
        p_ref_sum = p_ref_lower + p_ref_upper
        mask = int(
            p_ref_sum != 0
            and p_target * int(threshold["threshold_half_den"])
            > int(threshold["threshold_half_num"]) * p_ref_sum
        )
    return _measurements_from_powers(
        diagnostic_float=diagnostic_float,
        p_target=p_target,
        p_ref_lower=p_ref_lower,
        p_ref_upper=p_ref_upper,
        weights=weights,
        pilot_below_data_db=pilot_below_data_db,
        bin_enbw_hz=bin_enbw_hz,
        pilot_capture_efficiency=pilot_capture_efficiency,
        dtv_bandwidth_hz=dtv_bandwidth_hz,
        threshold=threshold,
        mask=mask,
        overflow=overflow,
    )


def _measurements_from_powers(
    *,
    diagnostic_float: float,
    p_target: int,
    p_ref_lower: int,
    p_ref_upper: int,
    weights: np.ndarray,
    pilot_below_data_db: float,
    bin_enbw_hz: float,
    pilot_capture_efficiency: float,
    dtv_bandwidth_hz: float,
    threshold: dict[str, Any] | None,
    mask: int,
    overflow: int,
) -> dict[str, Any]:
    """Backend-agnostic measurement fields from the three integer powers."""
    p_ref_sum = int(p_ref_lower + p_ref_upper)
    coarse_power_ratio = float(
        power_terms_to_coarse_power_ratio(p_target, p_ref_sum)
    )
    raw_pilot_excess = float(
        power_terms_to_raw_pilot_excess(p_target, p_ref_sum)
    )
    _nt, _nl, _nu = weight_term_norms_sq(np.asarray(weights, dtype=np.int8))
    normalized_coarse_power_ratio_db = float(
        power_terms_to_normalized_coarse_power_ratio_db(
            p_target,
            p_ref_sum,
            target_norm_sq=_nt,
            reference_norm_sum_sq=_nl + _nu,
        )
    )
    normalized_coarse_power_ratio = float(
        power_terms_to_normalized_coarse_power_ratio(
            p_target,
            p_ref_sum,
            target_norm_sq=_nt,
            reference_norm_sum_sq=_nl + _nu,
        )
    )
    normalized_pilot_excess = float(
        power_terms_to_normalized_pilot_excess(
            p_target,
            p_ref_sum,
            target_norm_sq=_nt,
            reference_norm_sum_sq=_nl + _nu,
        )
    )
    pilot_excess_db = float(
        normalized_pilot_excess_to_db(normalized_pilot_excess)
    )
    estimated_data_shelf_snr_db = float(
        pilot_excess_db_to_data_shelf_snr_db(
            pilot_excess_db,
            pilot_below_data_db=float(
                pilot_below_data_db
            ),
            bin_enbw_hz=float(bin_enbw_hz),
            pilot_capture_efficiency=float(pilot_capture_efficiency),
            dtv_bandwidth_hz=float(dtv_bandwidth_hz),
        )
    )
    out = {
        "diagnostic_raw_float32": diagnostic_float,
        "diagnostic_level_db_float32": _positive_to_db(diagnostic_float),
        "p_target_u64": p_target,
        "p_ref_lower_u64": p_ref_lower,
        "p_ref_upper_u64": p_ref_upper,
        "p_ref_sum_u64": p_ref_sum,
        "coarse_power_ratio": coarse_power_ratio,
        "normalized_coarse_power_ratio": normalized_coarse_power_ratio,
        "normalized_coarse_power_ratio_db": normalized_coarse_power_ratio_db,
        "raw_pilot_excess": raw_pilot_excess,
        "normalized_pilot_excess": normalized_pilot_excess,
        "pilot_excess_db": pilot_excess_db,
        "estimated_data_shelf_snr_db": estimated_data_shelf_snr_db,
        "normalized_positive_excess_decision": normalized_positive_excess(
            p_target,
            p_ref_sum,
            target_norm_sq=_nt,
            reference_norm_sum_sq=_nl + _nu,
        ),
    }
    if threshold is not None:
        out["mask"] = int(mask)
        out["rational_overflow_count"] = int(overflow)
    return out


def _positive_to_db(value: float) -> float:
    """Convert a positive linear value to dB, preserving non-positive as -inf."""
    value = float(value)
    if value <= 0.0:
        return float("-inf")
    return float(DB_POWER_FACTOR * math.log10(value))


def _safe_snr_label(snr_db: float) -> str:
    return (
        f"{float(snr_db):+.{CSV_SNR_LABEL_PRECISION}f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )


def _load_waveform_audit(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _command_or_path_exists(command: str) -> bool:
    command_text = str(command)
    if not command_text:
        return False
    if Path(command_text).exists():
        return True
    has_path_separator = any(
        separator and separator in command_text for separator in (os.sep, os.altsep)
    )
    if has_path_separator:
        return False
    return any(
        (Path(directory) / command_text).exists()
        for directory in os.environ.get("PATH", "").split(os.pathsep)
        if directory
    )


def _evaluate_one_trial(
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
    cp: Any,
    kernel: FStatKernel,
    weights: np.ndarray,
    cpu_float_weights: np.ndarray,
    clean_iq: np.ndarray,
    gnuradio_input_iq_path: Path,
    output_dir: Path,
    noisy_iq_dir: Path,
    rf_center_hz: float,
    band_lower_hz: float,
    response: np.ndarray,
    spec: ReferenceChannelizerSpec,
    channel_index: int,
    n_blocks: int,
    threshold: dict[str, Any] | None,
    requested_data_shelf_snr_db: float,
    requested_composite_atsc_snr_db: float,
    pilot_data_ratio: float,
    frequency_offset_hz: float,
    trial: int,
) -> dict[str, Any]:
    """Evaluate one SNR/frequency-offset/noise trial."""
    feed_noisy_iq: list[np.ndarray] = []
    feed_signal_powers: list[float] = []
    feed_requested_noise_powers: list[float] = []
    feed_noise_seeds: list[int] = []
    feed_noise_iq_paths: list[str] = []
    feed_gnuradio_blocks: list[str] = []

    for feed_index in range(int(args.num_input_streams)):
        feed_noisy_sample: np.ndarray | None = None
        feed_signal_power: float | None = None
        feed_requested_noise_power: float | None = None
        feed_gnuradio_metadata: dict[str, Any] = {}
        if args.noise_source == "gnuradio":
            noise_seed = int(rng.integers(1, np.iinfo(np.int32).max))
            feed_noise_seeds.append(noise_seed)
            if args.save_noisy_iq:
                noise_iq_file = noisy_iq_dir / (
                    "snr_shelf_"
                    f"{_safe_snr_label(float(requested_data_shelf_snr_db))}_"
                    "freq_offset_"
                    f"{_safe_snr_label(float(frequency_offset_hz))}_hz_"
                    f"trial_{trial:04d}_feed_{feed_index:04d}_"
                    f"seed_{noise_seed}.cfile"
                )
                (
                    feed_noisy_sample,
                    feed_signal_power,
                    feed_requested_noise_power,
                    feed_gnuradio_metadata,
                ) = add_gnuradio_awgn_for_snr(
                    clean_iq,
                    input_iq_path=gnuradio_input_iq_path,
                    output_iq_path=noise_iq_file,
                    snr_db=requested_composite_atsc_snr_db,
                    seed=noise_seed,
                    gnuradio_python=str(args.gnuradio_python),
                    sample_rate_hz=float(args.iq_sample_rate_hz),
                    snr_bandwidth_hz=float(args.dtv_bandwidth_hz),
                )
                feed_noise_iq_paths.append(str(noise_iq_file))
            else:
                with tempfile.TemporaryDirectory(dir=str(output_dir)) as tmp_dir:
                    noise_iq_file = Path(tmp_dir) / "noisy_iq.cfile"
                    (
                        feed_noisy_sample,
                        feed_signal_power,
                        feed_requested_noise_power,
                        feed_gnuradio_metadata,
                    ) = add_gnuradio_awgn_for_snr(
                        clean_iq,
                        input_iq_path=gnuradio_input_iq_path,
                        output_iq_path=noise_iq_file,
                        snr_db=requested_composite_atsc_snr_db,
                        seed=noise_seed,
                        gnuradio_python=str(args.gnuradio_python),
                        sample_rate_hz=float(args.iq_sample_rate_hz),
                        snr_bandwidth_hz=float(args.dtv_bandwidth_hz),
                    )
            feed_gnuradio_blocks.append(
                str(feed_gnuradio_metadata.get("gnuradio_block", ""))
            )
        else:
            (
                feed_noisy_sample,
                feed_signal_power,
                feed_requested_noise_power,
            ) = add_complex_awgn_for_snr(
                clean_iq,
                snr_db=requested_composite_atsc_snr_db,
                rng=rng,
                sample_rate_hz=float(args.iq_sample_rate_hz),
                snr_bandwidth_hz=float(args.dtv_bandwidth_hz),
            )

        if (
            feed_noisy_sample is None
            or feed_signal_power is None
            or feed_requested_noise_power is None
        ):
            raise RuntimeError("noise generation did not produce feed samples")
        feed_noisy_iq.append(feed_noisy_sample)
        feed_signal_powers.append(float(feed_signal_power))
        feed_requested_noise_powers.append(float(feed_requested_noise_power))

    mean_signal_power = float(np.mean(feed_signal_powers))
    mean_requested_noise_power = float(np.mean(feed_requested_noise_powers))
    bandwidth_ratio = float(args.dtv_bandwidth_hz) / float(args.iq_sample_rate_hz)
    requested_in_band_noise_power = mean_requested_noise_power * bandwidth_ratio
    requested_combined_in_band_noise_power = (
        requested_in_band_noise_power * int(args.num_input_streams)
    )
    realized_noise_power_by_feed = [
        float(np.mean(np.abs(noisy.astype(np.complex64) - clean_iq) ** 2))
        for noisy in feed_noisy_iq
    ]
    realized_in_band_noise_power_by_feed = [
        float(value * bandwidth_ratio)
        for value in realized_noise_power_by_feed
    ]
    realized_noise_power = float(np.sum(realized_noise_power_by_feed))
    realized_in_band_noise_power = float(np.sum(realized_in_band_noise_power_by_feed))
    combined_signal_power = float(mean_signal_power * int(args.num_input_streams))
    measured_truth_composite_atsc_snr_db = _positive_to_db(
        combined_signal_power / realized_in_band_noise_power
    )
    measured_data_shelf_power = float(
        combined_signal_power / (UNIT_DATA_SHELF_POWER + pilot_data_ratio)
    )
    measured_truth_data_shelf_snr_db = _positive_to_db(
        measured_data_shelf_power / realized_in_band_noise_power
    )

    synthesis_cuda = str(getattr(args, "synthesis_backend", "cpu")) == "cuda"
    envelope_to_blocks = (
        complex_envelope_to_real_adc_blocks_gpu
        if synthesis_cuda
        else complex_envelope_to_real_adc_blocks
    )
    blocks_to_channels = (
        channelize_real_blocks_to_reference_channels_gpu
        if synthesis_cuda
        else channelize_real_blocks_to_reference_channels
    )
    feed_channel_streams = []
    for noisy_iq in feed_noisy_iq:
        raw_blocks = envelope_to_blocks(
            noisy_iq,
            iq_sample_rate_hz=float(args.iq_sample_rate_hz),
            rf_center_hz=rf_center_hz,
            adc_sample_rate_hz=float(args.adc_sample_rate_hz),
            band_lower_hz=band_lower_hz,
            n_blocks=n_blocks,
            block_size=REFERENCE_PFB_FFT_SIZE,
        )
        channel_streams = blocks_to_channels(
            raw_blocks,
            channel_indices=[channel_index],
            response=response,
            spec=spec,
        )
        if args.reference_archive_phase:
            channel_streams = apply_reference_archive_phase_convention(
                channel_streams
            )
        feed_channel_streams.append(channel_streams)
    streams = flatten_feed_channel_streams(np.stack(feed_channel_streams, axis=0))

    cpu_float_rows = _float_streams_for_reference(
        streams,
        frame_size_samples=int(args.frame_size_samples),
        detector_window_samples=int(args.detector_window_samples),
        spectral_sense=str(args.spectral_sense),
    )
    cpu_float_fstat, cpu_float_powers = coarse_power_ratio_cpu_reference(
        cpu_float_rows,
        cpu_float_weights,
    )
    cpu_float_target_norm_sq = float(
        np.sum(np.abs(cpu_float_weights[REFERENCE_TARGET_TERM_INDEX]) ** 2)
    )
    cpu_float_reference_norm_sum_sq = float(
        np.sum(np.abs(cpu_float_weights[REFERENCE_LOWER_TERM_INDEX]) ** 2)
        + np.sum(np.abs(cpu_float_weights[REFERENCE_UPPER_TERM_INDEX]) ** 2)
    )
    cpu_float_null_power_ratio = float(
        COARSE_POWER_RATIO_SCALE
        * cpu_float_target_norm_sq
        / cpu_float_reference_norm_sum_sq
    )
    cpu_float_normalized_ratio = normalize_coarse_power_ratio(
        cpu_float_fstat,
        cpu_float_null_power_ratio,
    )
    cpu_float_normalized_excess = float(
        normalized_coarse_power_ratio_to_pilot_excess(
            cpu_float_normalized_ratio
        )
    )
    cpu_float_pilot_excess_db = float(
        normalized_pilot_excess_to_db(
            cpu_float_normalized_excess
        )
    )
    cpu_float_estimated_data_shelf_snr_db = float(
        pilot_excess_db_to_data_shelf_snr_db(
            cpu_float_pilot_excess_db,
            pilot_below_data_db=float(args.pilot_below_data_db),
            bin_enbw_hz=float(args.bin_enbw_hz),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
        )
    )

    scale = (
        float(args.scale)
        if args.scale is not None
        else estimate_complex_scale(
            streams,
            bits_per_component=int(args.bits),
            clip_sigma=float(args.clip_sigma),
        )
    )
    packed = _pack_streams_for_kernel(
        streams,
        frame_size_samples=int(args.frame_size_samples),
        detector_window_samples=int(args.detector_window_samples),
        bits=int(args.bits),
        scale=scale,
        spectral_sense=str(args.spectral_sense),
    )
    cpu_packed_fstat, cpu_packed_powers = coarse_power_ratio_cpu_reference_packed(
        packed,
        weights,
        int(args.bits),
    )
    cpu_packed_p_target = int(
        round(float(cpu_packed_powers[REFERENCE_TARGET_TERM_INDEX]))
    )
    cpu_packed_p_ref_lower = int(
        round(float(cpu_packed_powers[REFERENCE_LOWER_TERM_INDEX]))
    )
    cpu_packed_p_ref_upper = int(
        round(float(cpu_packed_powers[REFERENCE_UPPER_TERM_INDEX]))
    )
    cpu_packed_p_ref_sum = cpu_packed_p_ref_lower + cpu_packed_p_ref_upper
    packed_target_norm_sq, packed_lower_norm_sq, packed_upper_norm_sq = (
        weight_term_norms_sq(np.asarray(weights, dtype=np.int8))
    )
    packed_reference_norm_sum_sq = packed_lower_norm_sq + packed_upper_norm_sq
    cpu_packed_normalized_ratio = float(
        power_terms_to_normalized_coarse_power_ratio(
            cpu_packed_p_target,
            cpu_packed_p_ref_sum,
            target_norm_sq=packed_target_norm_sq,
            reference_norm_sum_sq=packed_reference_norm_sum_sq,
        )
    )
    cpu_packed_normalized_excess = float(
        normalized_coarse_power_ratio_to_pilot_excess(
            cpu_packed_normalized_ratio
        )
    )
    cpu_packed_pilot_excess_db = float(
        normalized_pilot_excess_to_db(cpu_packed_normalized_excess)
    )
    cpu_packed_estimated_data_shelf_snr_db = float(
        pilot_excess_db_to_data_shelf_snr_db(
            cpu_packed_pilot_excess_db,
            pilot_below_data_db=float(args.pilot_below_data_db),
            bin_enbw_hz=float(args.bin_enbw_hz),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
        )
    )
    if str(getattr(args, "detector_backend", "cuda")) == "cpu-reference":
        gpu = _cpu_reference_measurements(
            packed=packed,
            weights=weights,
            bits=int(args.bits),
            pilot_below_data_db=float(args.pilot_below_data_db),
            bin_enbw_hz=float(args.bin_enbw_hz),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            threshold=threshold,
        )
    else:
        gpu = _kernel_measurements(
            cp=cp,
            kernel=kernel,
            packed=packed,
            weights=weights,
            pilot_below_data_db=float(args.pilot_below_data_db),
            bin_enbw_hz=float(args.bin_enbw_hz),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            threshold=threshold,
        )
    estimated_data_shelf_snr_db = float(gpu["estimated_data_shelf_snr_db"])
    row = {
        "detector_backend": str(getattr(args, "detector_backend", "cuda")),
        "requested_data_shelf_snr_db": float(requested_data_shelf_snr_db),
        "requested_composite_atsc_snr_db": float(requested_composite_atsc_snr_db),
        "frequency_offset_hz": float(frequency_offset_hz),
        "channel_gain_db": float(args.channel_gain_db),
        "channel_phase_deg": float(args.channel_phase_deg),
        "measured_truth_data_shelf_snr_db": measured_truth_data_shelf_snr_db,
        "measured_truth_composite_atsc_snr_db": measured_truth_composite_atsc_snr_db,
        "measured_data_shelf_power": measured_data_shelf_power,
        "measured_composite_atsc_power": float(combined_signal_power),
        "measured_noise_power": float(realized_noise_power),
        "measured_noise_power_per_feed_mean": float(
            np.mean(realized_noise_power_by_feed)
        ),
        "measured_in_band_noise_power": float(realized_in_band_noise_power),
        "measured_in_band_noise_power_per_feed_mean": float(
            np.mean(realized_in_band_noise_power_by_feed)
        ),
        "coarse_power_ratio": float(gpu["coarse_power_ratio"]),
        "normalized_coarse_power_ratio": float(
            gpu["normalized_coarse_power_ratio"]
        ),
        "normalized_coarse_power_ratio_db": float(gpu["normalized_coarse_power_ratio_db"]),
        "raw_pilot_excess": float(gpu["raw_pilot_excess"]),
        "normalized_pilot_excess": float(gpu["normalized_pilot_excess"]),
        "pilot_excess_db": float(gpu["pilot_excess_db"]),
        "estimated_data_shelf_snr_db": estimated_data_shelf_snr_db,
        "gpu_estimated_data_shelf_snr_db": estimated_data_shelf_snr_db,
        "snr_error_db": estimated_data_shelf_snr_db - measured_truth_data_shelf_snr_db,
        "p_target_u64": int(gpu["p_target_u64"]),
        "p_ref_lower_u64": int(gpu["p_ref_lower_u64"]),
        "p_ref_upper_u64": int(gpu["p_ref_upper_u64"]),
        "p_ref_sum_u64": int(gpu["p_ref_sum_u64"]),
        "normalized_positive_excess_decision": int(gpu["normalized_positive_excess_decision"]),
        "diagnostic_raw_float32": float(gpu["diagnostic_raw_float32"]),
        "diagnostic_level_db_float32": float(gpu["diagnostic_level_db_float32"]),
        "cpu_float_coarse_power_ratio": float(cpu_float_fstat),
        "cpu_float_p_target": float(cpu_float_powers[REFERENCE_TARGET_TERM_INDEX]),
        "cpu_float_p_ref_lower": float(cpu_float_powers[REFERENCE_LOWER_TERM_INDEX]),
        "cpu_float_p_ref_upper": float(cpu_float_powers[REFERENCE_UPPER_TERM_INDEX]),
        "cpu_float_p_ref_sum": float(
            cpu_float_powers[REFERENCE_LOWER_TERM_INDEX]
            + cpu_float_powers[REFERENCE_UPPER_TERM_INDEX]
        ),
        "cpu_float_normalized_coarse_power_ratio": float(
            cpu_float_normalized_ratio
        ),
        "cpu_float_normalized_pilot_excess": cpu_float_normalized_excess,
        "cpu_float_pilot_excess_db": cpu_float_pilot_excess_db,
        "cpu_float_estimated_data_shelf_snr_db": cpu_float_estimated_data_shelf_snr_db,
        "cpu_float_snr_error_db": (
            cpu_float_estimated_data_shelf_snr_db - measured_truth_data_shelf_snr_db
        ),
        "trial": int(trial),
        "num_input_streams": int(args.num_input_streams),
        "num_selected_channels": 1,
        "detector_rows_per_frame": int(packed.shape[0]),
        "noise_source": str(args.noise_source),
        "noise_seed": (
            "" if not feed_noise_seeds else ";".join(str(seed) for seed in feed_noise_seeds)
        ),
        "noise_seeds": [int(seed) for seed in feed_noise_seeds],
        "noise_iq_path": ";".join(feed_noise_iq_paths),
        "noise_iq_paths": [str(path) for path in feed_noise_iq_paths],
        "gnuradio_block": ";".join(
            sorted({block for block in feed_gnuradio_blocks if block})
        ),
        "requested_noise_power": float(mean_requested_noise_power),
        "requested_in_band_noise_power": float(requested_in_band_noise_power),
        "requested_combined_in_band_noise_power": float(
            requested_combined_in_band_noise_power
        ),
        "noise_amplitude": float(math.sqrt(mean_requested_noise_power)),
        "noise_component_sigma": float(
            math.sqrt(mean_requested_noise_power / COMPLEX_COMPONENT_COUNT)
        ),
        "quantization_scale": float(scale),
        "cpu_coarse_power_ratio": float(cpu_packed_fstat),
        "cpu_packed_coarse_power_ratio": float(cpu_packed_fstat),
        "cpu_packed_p_target": cpu_packed_p_target,
        "cpu_packed_p_ref_lower": cpu_packed_p_ref_lower,
        "cpu_packed_p_ref_upper": cpu_packed_p_ref_upper,
        "cpu_packed_p_ref_sum": cpu_packed_p_ref_sum,
        "cpu_packed_normalized_coarse_power_ratio": cpu_packed_normalized_ratio,
        "cpu_packed_normalized_pilot_excess": cpu_packed_normalized_excess,
        "cpu_packed_pilot_excess_db": cpu_packed_pilot_excess_db,
        "cpu_packed_estimated_data_shelf_snr_db": (
            cpu_packed_estimated_data_shelf_snr_db
        ),
        "pilot_below_data_db": float(args.pilot_below_data_db),
        "target_weight_norm_sq": int(packed_target_norm_sq),
        "reference_weight_norm_sum_sq": int(packed_reference_norm_sum_sq),
        "cpu_float_target_weight_norm_sq": cpu_float_target_norm_sq,
        "cpu_float_reference_weight_norm_sum_sq": (
            cpu_float_reference_norm_sum_sq
        ),
        "cpu_gpu_abs_diff": abs(float(gpu["coarse_power_ratio"]) - float(cpu_packed_fstat)),
        "cpu_float_gpu_snr_diff_db": (
            cpu_float_estimated_data_shelf_snr_db - estimated_data_shelf_snr_db
        ),
    }
    if threshold is not None:
        row.update(
            {
                "mask": int(gpu["mask"]),
                "threshold_data_shelf_snr_db": float(threshold["threshold_data_shelf_snr_db"]),
                "threshold_pilot_excess_db": float(threshold["threshold_pilot_excess_db"]),
                "threshold_coarse_power_ratio": float(threshold["threshold_coarse_power_ratio"]),
                "threshold_half_num": int(threshold["threshold_half_num"]),
                "threshold_half_den": int(threshold["threshold_half_den"]),
                "rational_overflow_count": int(gpu["rational_overflow_count"]),
            }
        )
    return row


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion.

    Detection-rate points on publication curves should carry these bounds;
    they stay meaningful at rates near 0 or 1 where the normal approximation
    fails. Returns (lo, hi); (nan, nan) when trials == 0.
    """
    n = int(trials)
    if n <= 0:
        return (float("nan"), float("nan"))
    p = float(successes) / n
    if successes <= 0:
        p = 0.0
    elif successes >= n:
        p = 1.0
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5) / denom
    lo = 0.0 if successes <= 0 else max(0.0, center - half)
    hi = 1.0 if successes >= n else min(1.0, center + half)
    return (lo, hi)


def _detection_rate_fields(group: list[dict]) -> dict:
    """Return estimator-censoring and requested-threshold fractions."""
    fields: dict = {}
    n = len(group)
    if n and all("normalized_positive_excess_decision" in row for row in group):
        detected = sum(int(row["normalized_positive_excess_decision"]) for row in group)
        lo, hi = wilson_interval(detected, n)
        fields["positive_excess_fraction"] = detected / n
        fields["positive_excess_fraction_wilson95_lo"] = lo
        fields["positive_excess_fraction_wilson95_hi"] = hi
    if n and all("mask" in row for row in group):
        if "positive_excess_fraction" in fields:
            fields["normalized_positive_excess_detection_rate"] = fields[
                "positive_excess_fraction"
            ]
            fields["normalized_positive_excess_detection_rate_wilson95_lo"] = (
                fields["positive_excess_fraction_wilson95_lo"]
            )
            fields["normalized_positive_excess_detection_rate_wilson95_hi"] = (
                fields["positive_excess_fraction_wilson95_hi"]
            )
        detected = sum(int(row["mask"]) for row in group)
        lo, hi = wilson_interval(detected, n)
        fields["threshold_detection_rate"] = detected / n
        fields["threshold_detection_rate_wilson95_lo"] = lo
        fields["threshold_detection_rate_wilson95_hi"] = hi
    return fields


def _pooled_measurement_fields(
    group: list[dict[str, Any]],
    *,
    prefix: str,
    p_target_key: str,
    p_ref_lower_key: str,
    p_ref_upper_key: str,
    target_norm_key: str,
    reference_norm_key: str,
    integer_powers: bool,
    pilot_below_data_db: float,
    bin_enbw_hz: float,
    pilot_capture_efficiency: float,
    dtv_bandwidth_hz: float,
) -> dict[str, Any]:
    """Pool powers before forming a ratio."""
    required = (
        p_target_key,
        p_ref_lower_key,
        p_ref_upper_key,
        target_norm_key,
        reference_norm_key,
    )
    if not group or not all(all(key in row for key in required) for row in group):
        return {}

    if integer_powers:
        p_target = sum(int(row[p_target_key]) for row in group)
        p_ref_lower = sum(int(row[p_ref_lower_key]) for row in group)
        p_ref_upper = sum(int(row[p_ref_upper_key]) for row in group)
    else:
        p_target = math.fsum(float(row[p_target_key]) for row in group)
        p_ref_lower = math.fsum(float(row[p_ref_lower_key]) for row in group)
        p_ref_upper = math.fsum(float(row[p_ref_upper_key]) for row in group)
    p_ref_sum = p_ref_lower + p_ref_upper
    target_norm_sq = float(group[0][target_norm_key])
    reference_norm_sum_sq = float(group[0][reference_norm_key])
    normalized_ratio = (
        float("nan")
        if p_ref_sum <= 0.0
        else float(
            float(p_target)
            * reference_norm_sum_sq
            / (float(p_ref_sum) * target_norm_sq)
        )
    )
    normalized_excess = float(
        normalized_coarse_power_ratio_to_pilot_excess(normalized_ratio)
    )
    pilot_excess_db = float(normalized_pilot_excess_to_db(normalized_excess))
    estimated_snr_db = float(
        pilot_excess_db_to_data_shelf_snr_db(
            pilot_excess_db,
            pilot_below_data_db=float(pilot_below_data_db),
            bin_enbw_hz=float(bin_enbw_hz),
            pilot_capture_efficiency=float(pilot_capture_efficiency),
            dtv_bandwidth_hz=float(dtv_bandwidth_hz),
        )
    )
    return {
        f"{prefix}_pooled_p_target": p_target,
        f"{prefix}_pooled_p_ref_lower": p_ref_lower,
        f"{prefix}_pooled_p_ref_upper": p_ref_upper,
        f"{prefix}_pooled_p_ref_sum": p_ref_sum,
        f"{prefix}_pooled_normalized_coarse_power_ratio": normalized_ratio,
        f"{prefix}_pooled_normalized_pilot_excess": normalized_excess,
        f"{prefix}_pooled_pilot_excess_db": pilot_excess_db,
        f"{prefix}_pooled_estimated_data_shelf_snr_db": estimated_snr_db,
    }


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    requested_values: list[float],
    frequency_offset_values: list[float],
    composite_to_shelf_db: float,
    num_input_streams: int,
    pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    bin_enbw_hz: float = EFFECTIVE_BIN_BW_HZ,
    pilot_capture_efficiency: float = PILOT_CAPTURE_EFFICIENCY,
    dtv_bandwidth_hz: float = DTV_BANDWIDTH_HZ,
) -> list[dict[str, Any]]:
    """Summarize validation rows by requested SNR and frequency offset."""
    summary_rows: list[dict[str, Any]] = []
    for frequency_offset_hz in frequency_offset_values:
        for requested_data_shelf_snr_db in requested_values:
            group = [
                row
                for row in rows
                if row["requested_data_shelf_snr_db"] == float(requested_data_shelf_snr_db)
                and row["frequency_offset_hz"] == float(frequency_offset_hz)
            ]
            if not group:
                continue
            estimates = np.asarray(
                [row["estimated_data_shelf_snr_db"] for row in group],
                dtype=np.float64,
            )
            errors = np.asarray(
                [row["snr_error_db"] for row in group],
                dtype=np.float64,
            )
            fstats = np.asarray([row["coarse_power_ratio"] for row in group], dtype=np.float64)
            fstat_levels = np.asarray(
                [row["normalized_coarse_power_ratio_db"] for row in group],
                dtype=np.float64,
            )
            pnr_bin = np.asarray([row["pilot_excess_db"] for row in group], dtype=np.float64)
            truth_shelf = np.asarray(
                [row["measured_truth_data_shelf_snr_db"] for row in group],
                dtype=np.float64,
            )
            truth_composite = np.asarray(
                [row["measured_truth_composite_atsc_snr_db"] for row in group],
                dtype=np.float64,
            )
            cpu_float_estimates = np.asarray(
                [row["cpu_float_estimated_data_shelf_snr_db"] for row in group],
                dtype=np.float64,
            )
            cpu_float_errors = np.asarray(
                [row["cpu_float_snr_error_db"] for row in group],
                dtype=np.float64,
            )
            cpu_float_fstats = np.asarray(
                [row["cpu_float_coarse_power_ratio"] for row in group],
                dtype=np.float64,
            )
            diffs = np.asarray(
                [row["cpu_gpu_abs_diff"] for row in group],
                dtype=np.float64,
            )
            cpu_gpu_snr_diff = np.asarray(
                [row["cpu_float_gpu_snr_diff_db"] for row in group],
                dtype=np.float64,
            )
            pooled_fields: dict[str, Any] = {}
            pooled_fields.update(
                _pooled_measurement_fields(
                    group,
                    prefix="gpu",
                    p_target_key="p_target_u64",
                    p_ref_lower_key="p_ref_lower_u64",
                    p_ref_upper_key="p_ref_upper_u64",
                    target_norm_key="target_weight_norm_sq",
                    reference_norm_key="reference_weight_norm_sum_sq",
                    integer_powers=True,
                    pilot_below_data_db=pilot_below_data_db,
                    bin_enbw_hz=bin_enbw_hz,
                    pilot_capture_efficiency=pilot_capture_efficiency,
                    dtv_bandwidth_hz=dtv_bandwidth_hz,
                )
            )
            pooled_fields.update(
                _pooled_measurement_fields(
                    group,
                    prefix="cpu_float",
                    p_target_key="cpu_float_p_target",
                    p_ref_lower_key="cpu_float_p_ref_lower",
                    p_ref_upper_key="cpu_float_p_ref_upper",
                    target_norm_key="cpu_float_target_weight_norm_sq",
                    reference_norm_key="cpu_float_reference_weight_norm_sum_sq",
                    integer_powers=False,
                    pilot_below_data_db=pilot_below_data_db,
                    bin_enbw_hz=bin_enbw_hz,
                    pilot_capture_efficiency=pilot_capture_efficiency,
                    dtv_bandwidth_hz=dtv_bandwidth_hz,
                )
            )
            pooled_fields.update(
                _pooled_measurement_fields(
                    group,
                    prefix="cpu_packed",
                    p_target_key="cpu_packed_p_target",
                    p_ref_lower_key="cpu_packed_p_ref_lower",
                    p_ref_upper_key="cpu_packed_p_ref_upper",
                    target_norm_key="target_weight_norm_sq",
                    reference_norm_key="reference_weight_norm_sum_sq",
                    integer_powers=True,
                    pilot_below_data_db=pilot_below_data_db,
                    bin_enbw_hz=bin_enbw_hz,
                    pilot_capture_efficiency=pilot_capture_efficiency,
                    dtv_bandwidth_hz=dtv_bandwidth_hz,
                )
            )
            truth_shelf_mean = _nanmean_or_nan(truth_shelf)
            for prefix in ("gpu", "cpu_float", "cpu_packed"):
                pooled_estimate = float(
                    pooled_fields.get(
                        f"{prefix}_pooled_estimated_data_shelf_snr_db",
                        math.nan,
                    )
                )
                pooled_fields[f"{prefix}_pooled_snr_error_db"] = (
                    pooled_estimate - truth_shelf_mean
                    if math.isfinite(pooled_estimate)
                    and math.isfinite(truth_shelf_mean)
                    else math.nan
                )
            summary_rows.append(
                {
                    "requested_data_shelf_snr_db": float(requested_data_shelf_snr_db),
                    "frequency_offset_hz": float(frequency_offset_hz),
                    "channel_gain_db": float(group[0]["channel_gain_db"]),
                    "channel_phase_deg": float(group[0]["channel_phase_deg"]),
                    "requested_composite_atsc_snr_db": float(
                        float(requested_data_shelf_snr_db) - composite_to_shelf_db
                    ),
                    "pilot_below_data_db": float(pilot_below_data_db),
                    "measured_truth_data_shelf_snr_db_mean": truth_shelf_mean,
                    "measured_truth_composite_atsc_snr_db_mean": _nanmean_or_nan(
                        truth_composite
                    ),
                    "coarse_power_ratio_mean": _nanmean_or_nan(fstats),
                    "normalized_coarse_power_ratio_db_mean": _nanmean_or_nan(fstat_levels),
                    "pilot_excess_db_mean": _nanmean_or_nan(pnr_bin),
                    "estimated_data_shelf_snr_db_mean": _nanmean_or_nan(estimates),
                    "estimated_data_shelf_snr_db_std": _nanstd_or_nan(estimates),
                    "snr_error_db_mean": _nanmean_or_nan(errors),
                    "snr_error_db_std": _nanstd_or_nan(errors),
                    "cpu_float_coarse_power_ratio_mean": _nanmean_or_nan(cpu_float_fstats),
                    "cpu_float_estimated_data_shelf_snr_db_mean": _nanmean_or_nan(
                        cpu_float_estimates
                    ),
                    "cpu_float_estimated_data_shelf_snr_db_std": _nanstd_or_nan(
                        cpu_float_estimates
                    ),
                    "cpu_float_snr_error_db_mean": _nanmean_or_nan(
                        cpu_float_errors
                    ),
                    "cpu_float_snr_error_db_std": _nanstd_or_nan(cpu_float_errors),
                    "cpu_float_gpu_snr_diff_db_mean": _nanmean_or_nan(
                        cpu_gpu_snr_diff
                    ),
                    **pooled_fields,
                    "trials": int(len(group)),
                    **_detection_rate_fields(group),
                    "num_input_streams": int(num_input_streams),
                    "cpu_gpu_abs_diff_max": _nanmax_or_nan(diffs),
                }
            )
    return summary_rows


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=add_help,
        description=(
            "Inject AWGN into a clean GNU Radio ATSC IQ waveform, run the "
            "reference-channelizer/4+4-bit pipeline and CUDA kernel, and "
            "compare the F-statistic-derived data-shelf SNR to measured truth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-iq", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GENERATED_DIR / "dtv_snr_eval",
    )
    parser.add_argument(
        "--requested-data-shelf-snr-db",
        type=float,
        action="append",
        default=None,
        help=(
            "Requested ATSC data-shelf SNR relative to non-DTV noise power "
            "integrated over dtv_bandwidth_hz."
        ),
    )
    parser.add_argument("--snr-start-db", type=float, default=None)
    parser.add_argument("--snr-stop-db", type=float, default=None)
    parser.add_argument("--snr-step-db", type=float, default=None)
    parser.add_argument(
        "--frequency-offset-hz",
        type=float,
        action="append",
        default=None,
        help=(
            "Apply a baseband frequency offset before AWGN. Repeat to sweep "
            "multiple offsets."
        ),
    )
    parser.add_argument(
        "--standard-frequency-offset-sweep",
        action="store_true",
        help="Evaluate the built-in -1 kHz, 0 Hz, +1 kHz offset sweep.",
    )
    parser.add_argument(
        "--channel-gain-db",
        type=float,
        default=DEFAULT_CHANNEL_GAIN_DB,
        help="Static gain applied before noise injection.",
    )
    parser.add_argument(
        "--channel-phase-deg",
        type=float,
        default=DEFAULT_CHANNEL_PHASE_DEG,
        help="Static phase rotation applied before noise injection.",
    )
    parser.add_argument(
        "--detector-backend",
        choices=("cuda", "cpu-reference"),
        default="cuda",
        help=(
            "Which detector computes the primary fields. 'cuda' runs the "
            "compiled kernel (requires a GPU). 'cpu-reference' uses the "
            "validated exact-integer CPU reference, so full publication "
            "sweeps run without a GPU; tie the result to the deployed "
            "kernel with a small same-seed GPU spot check afterwards."
        ),
    )
    parser.add_argument(
        "--synthesis-backend",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "Where the reference ADC interpolation and PFB channelization "
            "run. They dominate the trial cost, linearly in stream count; "
            "'cuda' computes them with cupy and agrees with 'cpu' to float "
            "rounding (pinned by a parity test). The clean-pilot guard "
            "always runs on the cpu path."
        ),
    )
    parser.add_argument(
        "--noise-trials",
        type=int,
        default=DEFAULT_NOISE_TRIALS,
        help=(
            "Independent noise realizations per (offset, SNR) point. The "
            "default is sized for quick sweeps; increase it for stable pooled "
            "estimates and bootstrap intervals."
        ),
    )
    parser.add_argument(
        "--noise-source",
        choices=["gnuradio", "python"],
        default="gnuradio",
        help=(
            "Source used to add AWGN. The GNU Radio mode uses "
            "analog.noise_source_c plus blocks.add_cc in a helper process."
        ),
    )
    parser.add_argument(
        "--gnuradio-python",
        default=DEFAULT_GNURADIO_PYTHON,
        help="Python executable that can import GNU Radio.",
    )
    parser.add_argument(
        "--save-noisy-iq",
        action="store_true",
        help="Keep per-trial noisy IQ files generated by the GNU Radio helper.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATOR_SEED)
    parser.add_argument(
        "--waveform-audit-json",
        type=Path,
        default=GENERATED_DIR / "atsc" / "atsc_waveform_audit.json",
        help="Optional waveform-audit JSON to embed in the validation report.",
    )
    parser.add_argument(
        "--iq-sample-rate-hz",
        type=float,
        default=GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
    )
    parser.add_argument(
        "--adc-sample-rate-hz",
        type=float,
        default=REFERENCE_ADC_SAMPLE_RATE_HZ,
    )
    parser.add_argument(
        "--band-lower-mhz",
        type=float,
        default=REFERENCE_BAND_LOWER_HZ / HZ_PER_MHZ,
    )
    parser.add_argument(
        "--dtv-pilot-mhz",
        type=float,
        default=DEFAULT_DTV_PILOT_HZ / HZ_PER_MHZ,
    )
    parser.add_argument("--physical-channel", type=int, default=None)
    parser.add_argument("--dtv-bandwidth-hz", type=float, default=DTV_BANDWIDTH_HZ)
    parser.add_argument(
        "--pilot-below-data-db",
        dest="pilot_below_data_db",
        type=float,
        default=PILOT_BELOW_DATA_DB,
        help="Positive dB offset: ATSC pilot power below average data-shelf power.",
    )
    parser.add_argument(
        "--pilot-below-data-db-from-audit",
        action="store_true",
        help="Use measured_pilot_below_data_db from the waveform audit.",
    )
    parser.add_argument("--bin-enbw-hz", type=float, default=EFFECTIVE_BIN_BW_HZ)
    parser.add_argument(
        "--pilot-capture-efficiency",
        type=float,
        default=PILOT_CAPTURE_EFFICIENCY,
    )
    parser.add_argument("--threshold-data-shelf-snr-db", type=float, default=None)
    parser.add_argument(
        "--max-denominator",
        type=int,
        default=DEFAULT_THRESHOLD_MAX_DENOMINATOR,
    )
    parser.add_argument("--rf-center-mhz", type=float, default=None)
    parser.add_argument(
        "--atsc-pilot-offset-hz",
        type=float,
        default=ATSC_PILOT_OFFSET_HZ,
    )
    parser.add_argument("--channel-index", type=int, default=None)
    parser.add_argument(
        "--frame-size-samples",
        dest="frame_size_samples",
        type=int,
        default=DEFAULT_FRAME_SIZE_SAMPLES,
        help="Frame size, in channelized samples, to evaluate per trial.",
    )
    parser.add_argument(
        "--num-input-streams",
        dest="num_input_streams",
        type=int,
        default=DEFAULT_NUM_INPUT_STREAMS,
        help=(
            "Number of independent input streams/feeds to combine into one "
            "detector decision. Each stream receives the same clean ATSC "
            "waveform and an independent AWGN realization."
        ),
    )
    parser.add_argument(
        "--experimental-detector-window-samples",
        dest="detector_window_samples",
        type=int,
        default=LOCKED_DETECTOR_WINDOW_SAMPLES,
        help="Advanced: select a detector window supported by the current kernel.",
    )
    parser.add_argument(
        "--experimental-bits",
        dest="bits",
        type=int,
        default=LOCKED_BITS_PER_COMPONENT,
        help="Advanced: the current kernel requires the locked 4+4-bit format.",
    )
    parser.add_argument("--clip-sigma", type=float, default=DEFAULT_CLIP_SIGMA)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument(
        "--spectral-sense",
        choices=["normal", "inverted"],
        default="normal",
        help=(
            "Sense of the channelized stream relative to the deployed weight "
            "bank. With the reference channelizer's native phase and the "
            "shipped bank, 'normal' places the clean pilot on the statistic's "
            "target term (measured on channel 14: clean normalized ratio 121 "
            "against 1.1 for every other sense/phase combination). The "
            "startup guard verifies the configured combination against the "
            "clean waveform and refuses to sweep when the pilot misses."
        ),
    )
    parser.add_argument(
        "--reference-archive-phase",
        dest="reference_archive_phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the conj/(-1)^n archive phase convention to channelized "
            "streams before detection. Its half-rate factor displaces the "
            "pilot by half a coarse channel relative to the shipped weight "
            "layouts, so the detector then measures noise at every SNR; "
            "leave this off except to diagnose coordinate conventions."
        ),
    )
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    return run(parser.parse_args(argv))


def run(args: argparse.Namespace) -> int:
    """Run the evaluator from a parsed namespace (shared with the CLI)."""
    caller_cwd = Path.cwd()
    args.input_iq = resolve_user_path(args.input_iq, relative_to=caller_cwd)
    args.output_dir = resolve_user_path(args.output_dir, relative_to=caller_cwd)
    args.waveform_audit_json = resolve_user_path(
        args.waveform_audit_json,
        relative_to=caller_cwd,
    )
    args.lib_path = resolve_user_path(args.lib_path, relative_to=caller_cwd)
    args.weights_path = resolve_user_path(args.weights_path, relative_to=caller_cwd)
    audit = _load_waveform_audit(args.waveform_audit_json)
    if args.pilot_below_data_db_from_audit:
        measured = _optional_float(
            None if audit is None else audit.get("measured_pilot_below_data_db")
        )
        if measured is None or not math.isfinite(measured) or measured <= 0.0:
            raise SystemExit(
                "The waveform audit lacks a positive measured_pilot_below_data_db."
            )
        args.pilot_below_data_db = float(measured)
    if args.bits != LOCKED_BITS_PER_COMPONENT:
        raise SystemExit("This evaluator is intended for locked 4+4 bit input.")
    if args.detector_window_samples != LOCKED_DETECTOR_WINDOW_SAMPLES:
        raise SystemExit(
            "This evaluator is intended for the locked 128-sample detector "
            "window used by the shipped kernel and weights."
        )
    if args.physical_channel is not None:
        args.dtv_pilot_mhz = physical_channel_to_pilot_hz(
            int(args.physical_channel)
        ) / HZ_PER_MHZ
    if args.noise_trials <= 0:
        raise SystemExit("--noise-trials must be positive.")
    if args.num_input_streams <= 0:
        raise SystemExit("--num-input-streams must be positive.")
    if args.frame_size_samples % args.detector_window_samples != 0:
        raise SystemExit(
            "--frame-size-samples must be an integer multiple of the locked "
            "128-sample detector window."
        )
    if args.noise_source == "gnuradio":
        gnuradio_python = str(args.gnuradio_python)
        if not _command_or_path_exists(gnuradio_python):
            raise SystemExit(
                f"Could not find GNU Radio Python executable: {gnuradio_python}"
            )

    if str(args.detector_backend) == "cpu-reference":
        cp = None
        kernel = None
    else:
        import cupy as cp
        kernel = FStatKernel(args.lib_path)
    if str(args.synthesis_backend) == "cuda" and cp is None:
        import cupy  # noqa: F401

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    noisy_iq_dir = output_dir / "noisy_iq"
    if args.save_noisy_iq:
        noisy_iq_dir.mkdir(parents=True, exist_ok=True)

    num_output_samples = int(args.frame_size_samples)
    required = required_iq_samples(
        iq_sample_rate_hz=float(args.iq_sample_rate_hz),
        adc_sample_rate_hz=float(args.adc_sample_rate_hz),
        num_output_samples=num_output_samples,
    )
    clean_iq = np.fromfile(args.input_iq, dtype=np.complex64)
    if clean_iq.size < required:
        raise SystemExit(
            f"Input IQ is too short: need {required} samples, got {clean_iq.size}."
        )
    clean_iq = np.ascontiguousarray(clean_iq[:required])

    band_lower_hz = float(args.band_lower_mhz) * HZ_PER_MHZ
    rf_center_hz = _resolve_rf_center_hz(args)
    spec = ReferenceChannelizerSpec(
        adc_sample_rate_hz=float(args.adc_sample_rate_hz),
        band_lower_hz=band_lower_hz,
    )
    channel_index = (
        int(args.channel_index)
        if args.channel_index is not None
        else nearest_reference_channel_index(float(args.dtv_pilot_mhz) * HZ_PER_MHZ, spec)
    )
    response = sinc_hamming_pfb_response(REFERENCE_PFB_TAPS, REFERENCE_PFB_FFT_SIZE)
    n_blocks = num_output_samples + REFERENCE_PFB_TAPS - 1

    weights_bank = DetectorWeightBank(
        explicit_path=args.weights_path,
        expected_kernel=(kernel.specs if kernel is not None else None),
    )
    weight_bank_identity = _artifact_identity(weights_bank.path)
    weight_manifest_path = sidecar_manifest_path(weights_bank.path)
    if weight_manifest_path is None:
        raise SystemExit("The detector weight manifest path is missing.")
    weight_manifest_identity = _artifact_identity(weight_manifest_path)
    selected_weight_layout = weights_bank.layout_for_pilot_frequency(
        float(args.dtv_pilot_mhz)
    )
    weights, valid = weights_bank.get_weights_for_pilot_frequency(
        float(args.dtv_pilot_mhz)
    )
    if weights is None or not valid:
        raise SystemExit(
            "No valid detector weights for DTV pilot "
            f"{float(args.dtv_pilot_mhz):.6f} MHz."
        )
    selected_weight_coefficients = _weight_coefficients_identity(weights)
    if kernel is None and int(weights.shape[1]) != int(args.detector_window_samples):
        raise SystemExit(
            "cpu-reference backend: weight bank K "
            f"({int(weights.shape[1])}) does not match "
            f"--detector-window-samples ({int(args.detector_window_samples)})."
        )

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    requested_values = _requested_snr_shelf_values(args)
    frequency_offset_values = _frequency_offset_values(args)
    composite_to_shelf_db = composite_to_data_shelf_snr_correction_db(
        pilot_below_data_db=float(args.pilot_below_data_db)
    )
    pilot_data_ratio = pilot_to_data_power_ratio(
        pilot_below_data_db=float(args.pilot_below_data_db)
    )
    threshold_target_norm_sq, threshold_ref_lower_norm_sq, threshold_ref_upper_norm_sq = (
        weight_term_norms_sq(np.asarray(weights, dtype=np.int8))
    )
    threshold = None
    if args.threshold_data_shelf_snr_db is not None:
        threshold = data_shelf_snr_threshold_fields(
            float(args.threshold_data_shelf_snr_db),
            max_denominator=int(args.max_denominator),
            pilot_below_data_db=float(args.pilot_below_data_db),
            bin_enbw_hz=float(args.bin_enbw_hz),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
            target_norm_sq=int(threshold_target_norm_sq),
            reference_norm_sum_sq=int(
                threshold_ref_lower_norm_sq + threshold_ref_upper_norm_sq
            ),
        )
    cpu_float_weights = _ideal_float_weights_from_layout(
        selected_weight_layout,
        detector_window_samples=int(args.detector_window_samples),
    )

    guard_blocks = complex_envelope_to_real_adc_blocks(
        clean_iq,
        iq_sample_rate_hz=float(args.iq_sample_rate_hz),
        rf_center_hz=rf_center_hz,
        adc_sample_rate_hz=float(args.adc_sample_rate_hz),
        band_lower_hz=band_lower_hz,
        n_blocks=n_blocks,
        block_size=REFERENCE_PFB_FFT_SIZE,
    )
    guard_channel_streams = channelize_real_blocks_to_reference_channels(
        guard_blocks,
        channel_indices=[channel_index],
        response=response,
        spec=spec,
    )
    if args.reference_archive_phase:
        guard_channel_streams = apply_reference_archive_phase_convention(
            guard_channel_streams
        )
    assert_clean_pilot_lands_on_target(
        flatten_feed_channel_streams(np.stack([guard_channel_streams], axis=0)),
        selected_weight_layout=selected_weight_layout,
        cpu_float_weights=cpu_float_weights,
        detector_window_samples=int(args.detector_window_samples),
        frame_size_samples=int(args.frame_size_samples),
        spectral_sense=str(args.spectral_sense),
    )

    for frequency_offset_hz in frequency_offset_values:
        channel_clean_iq = apply_channel_impairments(
            clean_iq,
            sample_rate_hz=float(args.iq_sample_rate_hz),
            frequency_offset_hz=float(frequency_offset_hz),
            gain_db=float(args.channel_gain_db),
            phase_deg=float(args.channel_phase_deg),
        )
        channel_effects_active = not (
            math.isclose(float(frequency_offset_hz), DEFAULT_CHANNEL_FREQUENCY_OFFSET_HZ)
            and math.isclose(float(args.channel_gain_db), DEFAULT_CHANNEL_GAIN_DB)
            and math.isclose(float(args.channel_phase_deg), DEFAULT_CHANNEL_PHASE_DEG)
        )
        channel_input_temp: tempfile.TemporaryDirectory[str] | None = None
        channel_input_iq_path = args.input_iq
        if channel_effects_active:
            channel_temp = tempfile.TemporaryDirectory(dir=str(output_dir))
            channel_input_temp = channel_temp
            channel_input_iq_path = Path(channel_temp.name) / "channel_iq.cfile"
            channel_clean_iq.tofile(channel_input_iq_path)
        try:
            for requested_data_shelf_snr_db in requested_values:
                requested_composite_atsc_snr_db = (
                    float(requested_data_shelf_snr_db) - composite_to_shelf_db
                )
                for trial in range(int(args.noise_trials)):
                    row = _evaluate_one_trial(
                        args=args,
                        rng=rng,
                        cp=cp,
                        kernel=kernel,
                        weights=weights,
                        cpu_float_weights=cpu_float_weights,
                        clean_iq=channel_clean_iq,
                        gnuradio_input_iq_path=channel_input_iq_path,
                        output_dir=output_dir,
                        noisy_iq_dir=noisy_iq_dir,
                        rf_center_hz=rf_center_hz,
                        band_lower_hz=band_lower_hz,
                        response=response,
                        spec=spec,
                        channel_index=channel_index,
                        n_blocks=n_blocks,
                        threshold=threshold,
                        requested_data_shelf_snr_db=float(requested_data_shelf_snr_db),
                        requested_composite_atsc_snr_db=(
                            requested_composite_atsc_snr_db
                        ),
                        pilot_data_ratio=pilot_data_ratio,
                        frequency_offset_hz=float(frequency_offset_hz),
                        trial=int(trial),
                    )
                    rows.append(row)
        finally:
            if channel_input_temp is not None:
                channel_input_temp.cleanup()

    if not rows:
        raise SystemExit("No validation rows were produced.")

    csv_path = output_dir / "dtv_snr_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = _summarize_rows(
        rows,
        requested_values=requested_values,
        frequency_offset_values=frequency_offset_values,
        composite_to_shelf_db=composite_to_shelf_db,
        num_input_streams=int(args.num_input_streams),
        pilot_below_data_db=float(args.pilot_below_data_db),
        bin_enbw_hz=float(args.bin_enbw_hz),
        pilot_capture_efficiency=float(args.pilot_capture_efficiency),
        dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
    )

    summary_csv_path = output_dir / "dtv_snr_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    conversion_metadata = pilot_excess_to_data_shelf_metadata()
    spreading_loss_db = spreading_loss_db_from_bin_enbw_hz(
        float(args.bin_enbw_hz),
        dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
    )
    capture_efficiency_db = pilot_capture_efficiency_db(
        float(args.pilot_capture_efficiency)
    )
    conversion_metadata.update(
        {
            "bin_enbw_hz": float(args.bin_enbw_hz),
            "dtv_bandwidth_hz": float(args.dtv_bandwidth_hz),
            "n_shelf_bins": float(args.dtv_bandwidth_hz / args.bin_enbw_hz),
            "spreading_loss_db": float(spreading_loss_db),
            "pilot_below_data_db": float(args.pilot_below_data_db),
            "pilot_to_data_power_db": float(-args.pilot_below_data_db),
            "pilot_capture_efficiency": float(args.pilot_capture_efficiency),
            "pilot_capture_efficiency_db": float(capture_efficiency_db),
            "pilot_to_data_power_ratio": float(pilot_data_ratio),
            "composite_to_data_shelf_snr_correction_db": float(
                composite_to_shelf_db
            ),
            "pilot_excess_to_data_shelf_snr_offset_db": float(
                args.pilot_below_data_db
                - spreading_loss_db
                - capture_efficiency_db
            ),
        }
    )
    measured_pilot_to_data_power_db = _optional_float(
        None if audit is None else audit.get("measured_pilot_to_data_power_db")
    )
    measured_pilot_below_data_db = _optional_float(
        None if audit is None else audit.get("measured_pilot_below_data_db")
    )
    if (
        measured_pilot_below_data_db is None
        and measured_pilot_to_data_power_db is not None
    ):
        measured_pilot_below_data_db = -float(measured_pilot_to_data_power_db)
    pooled_errors = np.asarray(
        [row.get("gpu_pooled_snr_error_db", math.nan) for row in summary_rows],
        dtype=np.float64,
    )
    input_layout = DetectorFrameLayout(
        frame_size_samples=int(args.frame_size_samples),
        detector_window_samples=int(args.detector_window_samples),
        num_input_streams=int(args.num_input_streams),
        num_selected_channels=1,
    ).to_dict()
    stream_map = build_stream_map(
        num_feeds=int(args.num_input_streams),
        selected_channel_indices=[int(channel_index)],
        physical_channel=(
            None if args.physical_channel is None else int(args.physical_channel)
        ),
    )
    detector_geometry = {
        "input_layout": input_layout,
        "stream_map": stream_map,
        "reference_offset_bins": int(weights_bank.reference_offset_bins),
        "nominal_reference_offset_bins": int(weights_bank.reference_offset_bins),
        "selected_lower_reference_offset_bins": selected_weight_layout.get(
            "lower_reference_offset_bins"
        ),
        "selected_upper_reference_offset_bins": selected_weight_layout.get(
            "upper_reference_offset_bins"
        ),
        "bin_enbw_hz": float(args.bin_enbw_hz),
        "dtv_bandwidth_hz": float(args.dtv_bandwidth_hz),
        "pilot_capture_efficiency": float(args.pilot_capture_efficiency),
        "stable_combine_mode": COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
        "power_sum_rule": (
            "sum target/reference powers over all detector rows before forming F"
        ),
        "selected_channel_index": int(channel_index),
        "dtv_pilot_hz": float(args.dtv_pilot_mhz) * HZ_PER_MHZ,
        "rf_center_hz": float(rf_center_hz),
        "reference_archive_phase": bool(args.reference_archive_phase),
        "spectral_sense": str(args.spectral_sense),
    }
    summary = {
        "schema_version": "pilotproxy_snr_validation_report_v1",
        "result_schema_version": RESULT_SCHEMA_TOKEN,
        "project_identity": (
            "Standalone CUDA F-statistic DTV pilot detector and GNU Radio "
            "ATSC 1.0 testbench."
        ),
        "input_iq": str(args.input_iq),
        "output_dir": str(output_dir),
        "atsc_waveform_audit": audit,
        "weight_bank": weight_bank_identity,
        "weight_manifest": weight_manifest_identity,
        "selected_weight_layout": selected_weight_layout,
        "selected_weight_coefficients": selected_weight_coefficients,
        "result_schema": result_schema_object(
            frame_size_samples=int(args.frame_size_samples),
            num_input_streams=int(args.num_input_streams),
            detector_window_samples=int(args.detector_window_samples),
            dtv_bandwidth_hz=float(args.dtv_bandwidth_hz),
            bin_enbw_hz=float(args.bin_enbw_hz),
            pilot_below_data_db=float(args.pilot_below_data_db),
            pilot_capture_efficiency=float(args.pilot_capture_efficiency),
            threshold=threshold,
            reference_offset_bins=int(weights_bank.reference_offset_bins),
            num_selected_channels=1,
        ),
        "detector_geometry": detector_geometry,
        "truth": {
            "snr_shelf_definition": (
                "ATSC data-shelf power relative to non-DTV noise power "
                "integrated over dtv_bandwidth_hz; pilot power is excluded "
                "from the shelf truth."
            ),
            "requested_data_shelf_snr_db_values": [float(v) for v in requested_values],
            "requested_data_shelf_snr_db_min": float(min(requested_values)),
            "requested_data_shelf_snr_db_max": float(max(requested_values)),
            "composite_to_shelf_snr_correction_db": float(composite_to_shelf_db),
            "pilot_to_data_power_ratio": float(pilot_data_ratio),
        },
        "calibration": {
            "mode": "standard"
            if (
                math.isclose(float(args.pilot_below_data_db), PILOT_BELOW_DATA_DB)
                and math.isclose(float(args.bin_enbw_hz), float(EFFECTIVE_BIN_BW_HZ))
                and math.isclose(
                    float(args.pilot_capture_efficiency),
                    PILOT_CAPTURE_EFFICIENCY,
                )
            )
            else "calibrated",
            "pilot_below_data_db_assumed": float(args.pilot_below_data_db),
            "pilot_below_data_db_used": float(args.pilot_below_data_db),
            "pilot_below_data_db_source": (
                "waveform_audit"
                if args.pilot_below_data_db_from_audit
                else "command_line_or_default"
            ),
            "measured_pilot_to_data_power_db": measured_pilot_to_data_power_db,
            "measured_pilot_below_data_db": measured_pilot_below_data_db,
            "bin_enbw_hz_assumed": float(args.bin_enbw_hz),
            "pilot_capture_efficiency_assumed": float(args.pilot_capture_efficiency),
            "snr_bias_basis": "pooled powers at each SNR point",
            "snr_bias_db_mean": _nanmean_or_nan(pooled_errors),
            "snr_bias_db_std": _nanstd_or_nan(pooled_errors),
        },
        "detector_output": {
            "fstat_definition": "F = 2*P_target/(P_ref_lower + P_ref_upper)",
            "normalized_coarse_power_ratio_definition": (
                "Q = P_target*reference_norm_sum_sq/"
                "((P_ref_lower + P_ref_upper)*target_norm_sq)"
            ),
            "normalized_coarse_power_ratio_db_definition": "10*log10(Q)",
            "raw_pilot_excess_definition": (
                "F - 1, diagnostic without weight-norm correction"
            ),
            "normalized_pilot_excess_definition": "rho = Q - 1",
            "pilot_excess_db_definition": "10*log10(rho), for rho > 0",
            "summary_pooling_rule": (
                "Sum target and reference powers over trials before forming Q, "
                "rho, or logarithmic estimates."
            ),
            "canonical_summary_prefixes": (
                "gpu_pooled_, cpu_float_pooled_, and cpu_packed_pooled_"
            ),
            "estimated_data_shelf_snr_db_definition": (
                "pilot_excess_db - 10*log10(dtv_bandwidth_hz / bin_enbw_hz) "
                "+ pilot_below_data_db - 10*log10(pilot_capture_efficiency)"
            ),
            "uses_exact_uint64_powers": True,
            "cpu_float_reference": (
                "Unquantized channelized detector rows with ideal complex DFT "
                "weights from the selected manifest layout."
            ),
            "cpu_packed_reference": (
                "NumPy reference using packed int4 samples and packed int4 "
                "weights, useful for CPU/GPU fixed-point agreement checks."
            ),
        },
        "threshold": threshold,
        "testbench": {
            "noise_source": str(args.noise_source),
            "gnuradio_python": str(args.gnuradio_python),
            "master_seed": int(args.seed),
            "noise_trials_per_point": int(args.noise_trials),
            "detector_backend": str(args.detector_backend),
            "synthesis_backend": str(args.synthesis_backend),
            "save_noisy_iq": bool(args.save_noisy_iq),
            "iq_samples_used": int(clean_iq.size),
            "quantization": {
                "scale_policy": (
                    "per-trial-estimated" if args.scale is None else "explicit"
                ),
                "explicit_scale": None if args.scale is None else float(args.scale),
                "clip_sigma": float(args.clip_sigma),
                "bits_per_component": int(args.bits),
            },
            "snr_sweep": {
                "default_min_db": float(DEFAULT_SNR_SWEEP_MIN_DB),
                "default_max_db": float(DEFAULT_SNR_SWEEP_MAX_DB),
                "default_step_db": float(DEFAULT_SNR_SWEEP_STEP_DB),
                "supported_min_db": float(SUPPORTED_SNR_MIN_DB),
                "supported_max_db": float(SUPPORTED_SNR_MAX_DB),
                "requested_values_db": [float(v) for v in requested_values],
            },
            "channel_effects": {
                "frequency_offset_hz_values": [
                    float(v) for v in frequency_offset_values
                ],
                "standard_frequency_offset_sweep_hz": [
                    float(v) for v in STANDARD_FREQUENCY_OFFSET_SWEEP_HZ
                ],
                "channel_gain_db": float(args.channel_gain_db),
                "channel_phase_deg": float(args.channel_phase_deg),
            },
            "kernel_version": (
                kernel.version.as_string() if kernel is not None
                else "cpu-reference"
            ),
            "kernel_specs": (
                kernel.specs.as_descriptive_dict() if kernel is not None
                else None
            ),
            "cuda_device": (
                cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
                if kernel is not None
                else None
            ),
            "conversion_metadata": conversion_metadata,
        },
        "csv_columns": {
            "requested_data_shelf_snr_db": (
                "Requested ATSC data-shelf SNR relative to non-DTV noise."
            ),
            "frequency_offset_hz": (
                "Baseband frequency offset applied before AWGN injection."
            ),
            "measured_truth_data_shelf_snr_db": (
                "Measured data-shelf truth from clean composite power, "
                "pilot/data correction, and realized in-band noise."
            ),
            "measured_truth_composite_atsc_snr_db": (
                "Measured clean composite ATSC IQ power over realized in-band noise."
            ),
            "coarse_power_ratio": "2*P_target/(P_ref_lower + P_ref_upper).",
            "normalized_coarse_power_ratio": (
                "Weight-norm-corrected local-reference power ratio Q."
            ),
            "normalized_coarse_power_ratio_db": "10*log10(Q).",
            "raw_pilot_excess": (
                "coarse_power_ratio - 1 without weight-norm correction."
            ),
            "normalized_pilot_excess": "Signed normalized excess rho = Q - 1.",
            "pilot_excess_db": "10*log10(rho), defined only for rho > 0.",
            "estimated_data_shelf_snr_db": "DTV shelf SNR inferred from pilot_excess_db.",
            "estimated_data_shelf_snr_db_mean": (
                "Legacy mean of finite per-trial log estimates; use "
                "gpu_pooled_estimated_data_shelf_snr_db for transfer results."
            ),
            "gpu_pooled_estimated_data_shelf_snr_db": (
                "Canonical GPU shelf-SNR estimate formed from pooled powers."
            ),
            "cpu_float_pooled_estimated_data_shelf_snr_db": (
                "Canonical unquantized CPU estimate formed from pooled powers."
            ),
            "cpu_packed_pooled_estimated_data_shelf_snr_db": (
                "Canonical packed CPU estimate formed from pooled powers."
            ),
            "positive_excess_fraction": (
                "Fraction of trials with a finite log-domain estimate; not a "
                "detection probability."
            ),
            "pilot_below_data_db": (
                "Run-specific pilot power below the ATSC data shelf."
            ),
            "snr_error_db": (
                "estimated_data_shelf_snr_db minus measured_truth_data_shelf_snr_db."
            ),
            "cpu_float_estimated_data_shelf_snr_db": (
                "DTV shelf SNR from the unquantized CPU float reference."
            ),
            "cpu_float_normalized_coarse_power_ratio": (
                "Weight-norm-corrected CPU float ratio Q."
            ),
            "cpu_float_normalized_pilot_excess": (
                "Signed CPU float excess rho = Q - 1."
            ),
            "cpu_float_snr_error_db": (
                "cpu_float_estimated_data_shelf_snr_db minus measured truth."
            ),
            "cpu_coarse_power_ratio": (
                "Packed NumPy CPU reference F-statistic for fixed-point "
                "CPU/GPU agreement diagnostics."
            ),
            "cpu_packed_normalized_coarse_power_ratio": (
                "Weight-norm-corrected packed CPU ratio Q."
            ),
            "cpu_packed_normalized_pilot_excess": (
                "Signed packed CPU excess rho = Q - 1."
            ),
        },
        "summary": summary_rows,
        "results": rows,
    }
    json_path = output_dir / "dtv_snr_eval.json"
    write_json_strict(json_path, summary, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {json_path}")
    print(
        "frequency_offset_hz, requested_data_shelf_snr_db, "
        "measured_truth_data_shelf_snr_db, cpu_float_pooled_data_shelf_snr_db, "
        "gpu_pooled_data_shelf_snr_db, gpu_pooled_snr_error_db"
    )
    for row in summary_rows:
        print(
            f"{row['frequency_offset_hz']:10.3f}, "
            f"{row['requested_data_shelf_snr_db']:8.3f}, "
            f"{row['measured_truth_data_shelf_snr_db_mean']:8.3f}, "
            f"{row['cpu_float_pooled_estimated_data_shelf_snr_db']:8.3f}, "
            f"{row['gpu_pooled_estimated_data_shelf_snr_db']:8.3f}, "
            f"{row['gpu_pooled_snr_error_db']:8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
