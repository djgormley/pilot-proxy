# coding=utf-8
"""GPU gates for the v2 row-sum front end (run on CUDA hosts, e.g. the A100).

Three gates, all pre-registered:
1. exact equality of kernel row sums against the numpy int reference;
2. bit-exact all-bin coarse marginal identity against Compute_Powers_U64;
3. ULP-tolerance parity of the cupy complex64 fine reduction against the
   numpy complex128 prototype.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.fine_reduction import (
    exact_coarse_power_by_term,
    fine_reduce,
)
from gpu import cuda_available
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH

TERMS = 3
STREAMS = 16
WINDOWS = 128
ROWS = STREAMS * WINDOWS
RNG_SEED = 20260729
# One float32 ULP at the fine-power scale, with margin for FFT ordering:
# relative gate, pre-registered before A100 runs.
FINE_FFT_RELATIVE_TOLERANCE = 5.0e-6


def _import_cupy_or_skip():
    try:
        import cupy as cp
    except Exception:  # pragma: no cover - GPU-less hosts
        pytest.skip("cupy is not available")
    ok, reason = cuda_available()
    if not ok:
        pytest.skip(f"CUDA device is not available: {reason}")
    return cp


def _kernel_or_skip() -> FStatKernel:
    try:
        kernel = FStatKernel(DEFAULT_LIB_PATH)
    except FileNotFoundError:  # pragma: no cover - library not built
        pytest.skip("libfstatistic.so is not built")
    if not kernel.supports_row_projections():
        pytest.fail(
            "libfstatistic.so predates kernel core 2.0.0: rebuild from the "
            "current CUDA sources before relaunching any scan."
        )
    return kernel


def _sign_extend_i4(v: np.ndarray) -> np.ndarray:
    v = v & 0xF
    return (v ^ 0x8) - 0x8


def _numpy_reference_matched_filter_row_projections(
    packed: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Exact int reference: unpack int4, conj-multiply, integer sums."""
    xr = _sign_extend_i4((packed.astype(np.int32) >> 4)).astype(np.int64)
    xi = _sign_extend_i4(packed.astype(np.int32)).astype(np.int64)
    wr = _sign_extend_i4((weights.astype(np.int32) >> 4)).astype(np.int64)
    wi = _sign_extend_i4(weights.astype(np.int32)).astype(np.int64)
    # z = sum_k x_k * conj(w_k)
    re = np.einsum("mk,nk->nm", xr, wr) + np.einsum("mk,nk->nm", xi, wi)
    im = np.einsum("mk,nk->nm", xi, wr) - np.einsum("mk,nk->nm", xr, wi)
    out = np.stack([re, im], axis=-1)  # [terms, rows, 2]
    assert np.abs(out).max() < 2**31
    return out.astype(np.int32)


def _random_inputs(rng: np.random.Generator):
    packed = rng.integers(0, 256, size=(ROWS, 128), dtype=np.uint8).astype(
        np.int8
    )
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(
        np.int8
    )
    return packed, weights


def test_gpu_matched_filter_row_projections_exactly_match_numpy_reference():
    cp = _import_cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED)
    packed, weights = _random_inputs(rng)

    d_in = cp.asarray(packed)
    d_out = cp.zeros(1, dtype=cp.float32)
    handle = kernel.create_raw(ROWS, d_in.data.ptr, d_out.data.ptr)
    d_matched_filter_row_projections = cp.zeros(TERMS * ROWS * 2, dtype=cp.int32)
    d_powers = cp.zeros(TERMS, dtype=cp.uint64)
    try:
        kernel.compute_row_projections_i32(
            handle, weights.ctypes.data, d_matched_filter_row_projections.data.ptr
        )
        kernel.compute_powers_u64(
            handle, weights.ctypes.data, d_powers.data.ptr
        )
        cp.cuda.Device().synchronize()
        got = cp.asnumpy(d_matched_filter_row_projections).reshape(TERMS, ROWS, 2)
        powers = cp.asnumpy(d_powers)
    finally:
        kernel.destroy(handle)

    expected = _numpy_reference_matched_filter_row_projections(packed, weights)
    np.testing.assert_array_equal(got, expected)

    marginal = exact_coarse_power_by_term(got, num_weight_terms=TERMS)
    np.testing.assert_array_equal(
        marginal.astype(np.uint64), powers.astype(np.uint64)
    )


def test_cupy_fine_reduction_matches_float64_prototype_within_ulp_gate():
    cp = _import_cupy_or_skip()
    rng = np.random.default_rng(RNG_SEED + 1)
    matched_filter_row_projections = rng.integers(
        -14336, 14337, size=(TERMS, ROWS, 2), dtype=np.int32
    )

    host = fine_reduce(
        matched_filter_row_projections, num_streams=STREAMS, windows_per_stream=WINDOWS, xp=np
    )
    dev = fine_reduce(
        cp.asarray(matched_filter_row_projections),
        num_streams=STREAMS,
        windows_per_stream=WINDOWS,
        xp=cp,
    )

    np.testing.assert_array_equal(dev.coarse_power_by_term, host.coarse_power_by_term)
    denom = np.maximum(np.abs(host.fine_power.astype(np.float64)), 1.0)
    rel = np.abs(
        dev.fine_power.astype(np.float64) - host.fine_power.astype(np.float64)
    ) / denom
    assert float(rel.max()) < FINE_FFT_RELATIVE_TOLERANCE, (
        f"fine-power relative error {rel.max():.3e} exceeds the "
        f"pre-registered ULP gate {FINE_FFT_RELATIVE_TOLERANCE:.1e}"
    )
