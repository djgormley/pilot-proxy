# coding=utf-8
"""GPU gates for the fused fine kernel (kernel core 2.2.0).

``FStat_Compute_FusedFine_U64`` is the fused form of the deployed fine
path: one launch from packed int4 samples to exact uint64 fine powers
and exact uint64 coarse marginals, with row sums held in shared memory
and materialized to global only through the optional debug tap.

Acceptance is bit-equality with the composed path --- no tolerances:

1. single block: fused fine powers == composed
   (``RowSums_I32 -> FinePowers_U64``), fused marginals ==
   ``Compute_Powers_U64``, tap == the composed row-sum buffer, and all
   three == the numpy/fxfft reference chain applied to the same frames;
2. batch case: same equalities per batch entry with a batched handle;
3. production form: tap pointer NULL, fine + marginal outputs unchanged
   (the tap is genuinely optional; row sums never touch global memory).

The timing report (printed rather than asserted) gives the deployment-footprint
numbers: fused without tap (production), fused with tap, and the
composed three-launch chain, against the ~41.9 ms CHIME frame cadence.

The CPU-side device-code verification (threaded emulation of the fused
kernel, real barrier, bit-compared against the same references) ran at
integration time; this file is the on-GPU gate.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from pilot_proxy.fine_reduction import exact_coarse_power_by_term
from pilot_proxy.fxfft import fine_power_fx
from gpu import cuda_available
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH

TERMS = 3
WINDOWS = 128
FINE_BINS = 256
RNG_SEED = 20260804


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
    if not kernel.supports_fused_fine():
        pytest.skip(
            "libfstatistic.so predates kernel core 2.2.0 (no fused fine "
            "kernel): rebuild from the current CUDA sources."
        )
    return kernel


def _random_frames(rng, batch: int, rows: int):
    packed = rng.integers(
        0, 256, size=(batch, rows, 128), dtype=np.uint8
    ).astype(np.int8)
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(
        np.int8
    )
    return packed, weights


def _reference_chain(matched_filter_row_projections: np.ndarray, streams: int):
    """fine powers + marginals from an int32 [terms, rows, 2] buffer."""
    fine = fine_power_fx(
        matched_filter_row_projections.astype(np.int64),
        num_streams=streams,
        windows_per_stream=WINDOWS,
    )
    marg = exact_coarse_power_by_term(matched_filter_row_projections, num_weight_terms=TERMS).astype(
        np.uint64
    )
    return fine, marg


@pytest.mark.cuda
def test_fused_matches_composed_path_bit_exact():
    cp = _import_cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED)
    streams = 16
    rows = streams * WINDOWS
    packed, weights = _random_frames(rng, 1, rows)

    d_in = cp.asarray(packed[0])
    d_diag = cp.zeros(1, dtype=cp.float32)
    handle = kernel.create_raw(rows, d_in.data.ptr, d_diag.data.ptr)
    d_matched_filter_row_projections = cp.zeros(TERMS * rows * 2, dtype=cp.int32)
    d_fine_composed = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow_composed = cp.zeros(TERMS, dtype=cp.uint64)
    d_fine_fused = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow_fused = cp.zeros(TERMS, dtype=cp.uint64)
    d_tap = cp.zeros(TERMS * rows * 2, dtype=cp.int32)
    try:
        # Composed path: the validated 2.0.0/2.1.0 chain.
        kernel.compute_row_projections_i32(
            handle, weights.ctypes.data, d_matched_filter_row_projections.data.ptr
        )
        kernel.compute_fine_powers_u64(
            int(d_matched_filter_row_projections.data.ptr), streams, WINDOWS, 1,
            int(d_fine_composed.data.ptr),
        )
        kernel.compute_powers_u64(
            handle, weights.ctypes.data, d_pow_composed.data.ptr
        )
        # Fused path: one launch, tap bound.
        kernel.compute_fused_fine_u64(
            handle,
            weights.ctypes.data,
            int(d_fine_fused.data.ptr),
            int(d_pow_fused.data.ptr),
            int(d_tap.data.ptr),
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        composed_rows = cp.asnumpy(d_matched_filter_row_projections).reshape(TERMS, rows, 2)
        fine_c = cp.asnumpy(d_fine_composed)
        pow_c = cp.asnumpy(d_pow_composed)
        fine_f = cp.asnumpy(d_fine_fused)
        pow_f = cp.asnumpy(d_pow_fused)
        tap = cp.asnumpy(d_tap).reshape(TERMS, rows, 2)
    finally:
        kernel.destroy(handle)

    # Fused == composed, bit for bit.
    np.testing.assert_array_equal(tap, composed_rows)
    np.testing.assert_array_equal(fine_f, fine_c)
    np.testing.assert_array_equal(pow_f, pow_c)
    # Both == the frozen reference chain.
    fine_ref, pow_ref = _reference_chain(composed_rows, streams)
    np.testing.assert_array_equal(fine_f, fine_ref)
    np.testing.assert_array_equal(pow_f, pow_ref)


@pytest.mark.cuda
def test_fused_batch_case():
    cp = _import_cupy_or_skip()
    kernel = _kernel_or_skip()
    if not kernel.supports_batch:
        pytest.skip("libfstatistic.so does not expose FStat_Create_Batch")
    rng = np.random.default_rng(RNG_SEED + 1)
    streams, batch = 8, 3
    rows = streams * WINDOWS
    packed, weights = _random_frames(rng, batch, rows)

    d_in = cp.asarray(packed)
    d_diag = cp.zeros(batch, dtype=cp.float32)
    handle = kernel.create_raw_batch(
        rows, batch, d_in.data.ptr, d_diag.data.ptr
    )
    d_fine = cp.zeros((batch, TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow = cp.zeros((batch, TERMS), dtype=cp.uint64)
    d_tap = cp.zeros(batch * TERMS * rows * 2, dtype=cp.int32)
    try:
        kernel.compute_fused_fine_u64(
            handle,
            weights.ctypes.data,
            int(d_fine.data.ptr),
            int(d_pow.data.ptr),
            int(d_tap.data.ptr),
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        fine = cp.asnumpy(d_fine)
        pow_ = cp.asnumpy(d_pow)
        tap = cp.asnumpy(d_tap).reshape(batch, TERMS, rows, 2)
    finally:
        kernel.destroy(handle)

    for b in range(batch):
        fine_ref, pow_ref = _reference_chain(tap[b], streams)
        np.testing.assert_array_equal(fine[b], fine_ref)
        np.testing.assert_array_equal(pow_[b], pow_ref)


@pytest.mark.cuda
def test_fused_production_form_without_tap():
    cp = _import_cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED + 2)
    streams = 8
    rows = streams * WINDOWS
    packed, weights = _random_frames(rng, 1, rows)

    d_in = cp.asarray(packed[0])
    d_diag = cp.zeros(1, dtype=cp.float32)
    handle = kernel.create_raw(rows, d_in.data.ptr, d_diag.data.ptr)
    d_fine_tap = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow_tap = cp.zeros(TERMS, dtype=cp.uint64)
    d_fine_notap = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow_notap = cp.zeros(TERMS, dtype=cp.uint64)
    d_tap = cp.zeros(TERMS * rows * 2, dtype=cp.int32)
    try:
        kernel.compute_fused_fine_u64(
            handle,
            weights.ctypes.data,
            int(d_fine_tap.data.ptr),
            int(d_pow_tap.data.ptr),
            int(d_tap.data.ptr),
        )
        kernel.compute_fused_fine_u64(
            handle,
            weights.ctypes.data,
            int(d_fine_notap.data.ptr),
            int(d_pow_notap.data.ptr),
            0,
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        np.testing.assert_array_equal(
            cp.asnumpy(d_fine_notap), cp.asnumpy(d_fine_tap)
        )
        np.testing.assert_array_equal(
            cp.asnumpy(d_pow_notap), cp.asnumpy(d_pow_tap)
        )
    finally:
        kernel.destroy(handle)


@pytest.mark.cuda
def test_fused_rate_margin_report():
    cp = _import_cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED + 3)
    streams = 2048
    rows = streams * WINDOWS
    packed, weights = _random_frames(rng, 1, rows)

    d_in = cp.asarray(packed[0])
    d_diag = cp.zeros(1, dtype=cp.float32)
    handle = kernel.create_raw(rows, d_in.data.ptr, d_diag.data.ptr)
    d_matched_filter_row_projections = cp.zeros(TERMS * rows * 2, dtype=cp.int32)
    d_fine = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    d_pow = cp.zeros(TERMS, dtype=cp.uint64)

    def fused_notap():
        kernel.compute_fused_fine_u64(
            handle, weights.ctypes.data,
            int(d_fine.data.ptr), int(d_pow.data.ptr), 0,
        )

    def fused_tap():
        kernel.compute_fused_fine_u64(
            handle, weights.ctypes.data,
            int(d_fine.data.ptr), int(d_pow.data.ptr),
            int(d_matched_filter_row_projections.data.ptr),
        )

    def composed():
        kernel.compute_row_projections_i32(
            handle, weights.ctypes.data, d_matched_filter_row_projections.data.ptr
        )
        kernel.compute_fine_powers_u64(
            int(d_matched_filter_row_projections.data.ptr), streams, WINDOWS, 1,
            int(d_fine.data.ptr),
        )
        kernel.compute_powers_u64(
            handle, weights.ctypes.data, d_pow.data.ptr
        )

    def measure(fn, n=50):
        fn()  # warmup + JIT/caches
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        cp.cuda.Device().synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n

    try:
        ms_fused = measure(fused_notap)
        ms_tap = measure(fused_tap)
        ms_composed = measure(composed)
    finally:
        kernel.destroy(handle)

    frame_ms = 16384.0 / 390625.0 * 1000.0
    print(
        f"\nfused fine kernel, 2048-stream frame "
        f"(cadence {frame_ms:.1f} ms):\n"
        f"  fused, no tap (production): {ms_fused:.3f} ms "
        f"(margin x{frame_ms / ms_fused:.0f})\n"
        f"  fused, tap bound (debug):   {ms_tap:.3f} ms\n"
        f"  composed three-launch:      {ms_composed:.3f} ms "
        f"(fused saves {ms_composed - ms_fused:.3f} ms)"
    )
    assert kernel.last_error() == ""
