# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.detector_reference import (
    coarse_power_ratio_cpu_reference_packed,
    quantize_complex_numpy,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from gpu import cuda_available
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH

TARGET_FREQ_MHZ = 470.309441
REFERENCE_BANDWIDTH_MHZ = 400.0
REFERENCE_NUM_CHANNELS = 1024.0
HALF_CHANNEL_WIDTH = 2.0
KERNEL_SMOKE_RNG_SEED = 1234
KERNEL_SMOKE_ROWS = 8
KERNEL_SMOKE_NOISE_SIGMA = 1.0
QUANTIZER_CLIP_SIGMA = 3.0
COMPLEX_TONE_CYCLES = 2.0
GPU_MATCH_ABSOLUTE_TOLERANCE = 1e-4


def _import_cupy_or_skip():
    try:
        import cupy as cp
    except Exception as exc:
        pytest.skip(f"CuPy not available: {exc}")
    return cp


def _run_kernel(
    cp,
    kernel: FStatKernel,
    packed_samples: np.ndarray,
    packed_weights: np.ndarray,
) -> float:
    d_in = cp.asarray(packed_samples)
    d_out = cp.zeros(1, dtype=cp.float32)
    handle = kernel.create_raw(d_in.shape[0], d_in.data.ptr, d_out.data.ptr)
    try:
        kernel.compute_diagnostic_float(handle, packed_weights.ctypes.data)
        cp.cuda.Device().synchronize()
        return float(d_out[0].get())
    finally:
        kernel.destroy(handle)


def test_kernel_cpu_gpu_smoke_matches_prebuilt_dtv_weights() -> None:
    cp = _import_cupy_or_skip()
    ok, reason = cuda_available()
    if not ok:
        pytest.skip(f"CUDA not available: {reason}")
    if not DEFAULT_LIB_PATH.exists():
        pytest.skip(f"CUDA kernel library not built: {DEFAULT_LIB_PATH}")
    if not DEFAULT_WEIGHTS_PATH.exists():
        pytest.skip(f"Prebuilt weights not found: {DEFAULT_WEIGHTS_PATH}")

    kernel = FStatKernel(DEFAULT_LIB_PATH)
    wb = DetectorWeightBank(
        explicit_path=DEFAULT_WEIGHTS_PATH,
        expected_kernel=kernel.specs,
    )
    weights, valid = wb.get_weights_for_pilot_frequency(TARGET_FREQ_MHZ)
    assert valid and weights is not None

    channel_width_mhz = REFERENCE_BANDWIDTH_MHZ / REFERENCE_NUM_CHANNELS
    chan_idx = int(np.argmin(np.abs(wb.reference_freqs - TARGET_FREQ_MHZ)))
    chan_start = (
        float(wb.reference_freqs[chan_idx])
        - channel_width_mhz / HALF_CHANNEL_WIDTH
    )
    norm_target = (TARGET_FREQ_MHZ - chan_start) / channel_width_mhz

    rng = np.random.default_rng(KERNEL_SMOKE_RNG_SEED)
    rows = KERNEL_SMOKE_ROWS
    noise_sigma = KERNEL_SMOKE_NOISE_SIGMA
    noise = (
        rng.standard_normal((rows, kernel.specs.K))
        + 1j * rng.standard_normal((rows, kernel.specs.K))
    ) * noise_sigma
    k = np.arange(kernel.specs.K, dtype=np.float64)
    tone = np.exp(-1j * COMPLEX_TONE_CYCLES * np.pi * norm_target * k)[None, :]
    samples = noise + tone
    scale = float((1 << (kernel.specs.bits - 1)) - 1) / (
        QUANTIZER_CLIP_SIGMA * noise_sigma
    )
    packed = quantize_complex_numpy(samples, kernel.specs.bits, scale)

    cpu_value, _ = coarse_power_ratio_cpu_reference_packed(packed, weights, kernel.specs.bits)
    gpu_value = _run_kernel(cp, kernel, packed, weights)

    assert abs(gpu_value - cpu_value) <= GPU_MATCH_ABSOLUTE_TOLERANCE
