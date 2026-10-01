# coding=utf-8
"""GPU gates for the on-device fine-power stage (kernel core 2.1.0).

The library's ``FStat_Compute_FinePowers_U64`` must reproduce the frozen
Python reference (``fine_power_fx``) bit-for-bit --- exact uint64
equality, no tolerances. Three gates plus a rate-margin report:

1. pure stage: seeded row-sum buffers uploaded directly, library output
   == reference, including a non-block-multiple stream count and batch;
2. chained: packed int4 frames -> device row sums
   (``FStat_Compute_RowSums_I32``) -> fine powers from the SAME device
   buffer, compared against the reference applied to the host readback ---
   the deployed composition with no host round-trip in between;
3. geometry probe: ``FStat_GetFineSpecs`` returns the frozen 128/2/256.

The timing report (printed rather than asserted) is the first rate-margin
datapoint for the fine stage: ms per 2048-stream frame against the
~41.9 ms CHIME frame cadence.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from pilot_proxy.fxfft import fine_power_fx
from gpu import cuda_available
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH

TERMS = 3
WINDOWS = 128
FINE_BINS = 256
RNG_SEED = 20260801


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
    if not kernel.supports_fine_powers():
        pytest.skip(
            "libfstatistic.so predates kernel core 2.1.0 (no fine-power "
            "stage): rebuild from the current CUDA sources."
        )
    return kernel


def test_fine_specs_are_frozen():
    kernel = _kernel_or_skip()
    specs = kernel.get_fine_specs()
    assert specs == {
        "windows_per_stream": 128,
        "pad_factor": 2,
        "fine_bins": 256,
    }


@pytest.mark.cuda
def test_device_fine_powers_match_reference_pure_stage():
    cp = _import_cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED)
    for streams, batch in ((100, 1), (256, 3), (2048, 1)):
        rows = streams * WINDOWS
        buf = rng.integers(
            -14336, 14337, size=(batch, TERMS, rows, 2), dtype=np.int32
        )
        d_rs = cp.asarray(buf)
        d_out = cp.zeros((batch, TERMS, FINE_BINS), dtype=cp.uint64)
        kernel.compute_fine_powers_u64(
            int(d_rs.data.ptr), streams, WINDOWS, batch, int(d_out.data.ptr)
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        got = cp.asnumpy(d_out)
        for b in range(batch):
            exp = fine_power_fx(
                buf[b].astype(np.int64),
                num_streams=streams,
                windows_per_stream=WINDOWS,
            )
            np.testing.assert_array_equal(got[b], exp)


@pytest.mark.cuda
def test_device_chain_matched_filter_row_projections_to_fine_powers():
    cp = _import_cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED + 1)
    streams = 16
    rows = streams * WINDOWS
    packed = rng.integers(0, 256, size=(rows, 128), dtype=np.uint8).astype(
        np.int8
    )
    weights = rng.integers(0, 256, size=(TERMS, 128), dtype=np.uint8).astype(
        np.int8
    )

    d_in = cp.asarray(packed)
    d_diag = cp.zeros(1, dtype=cp.float32)
    handle = kernel.create_raw(rows, d_in.data.ptr, d_diag.data.ptr)
    d_matched_filter_row_projections = cp.zeros(TERMS * rows * 2, dtype=cp.int32)
    d_fine = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)
    try:
        kernel.compute_row_projections_i32(
            handle, weights.ctypes.data, d_matched_filter_row_projections.data.ptr
        )
        # fine powers straight from the device row-sum buffer: the deployed
        # composition, no host round-trip
        kernel.compute_fine_powers_u64(
            int(d_matched_filter_row_projections.data.ptr), streams, WINDOWS, 1,
            int(d_fine.data.ptr)
        )
        cp.cuda.Device().synchronize()
        assert kernel.last_error() == ""
        host_rows = cp.asnumpy(d_matched_filter_row_projections).reshape(TERMS, rows, 2)
        got = cp.asnumpy(d_fine)
    finally:
        kernel.destroy(handle)

    exp = fine_power_fx(
        host_rows.astype(np.int64),
        num_streams=streams,
        windows_per_stream=WINDOWS,
    )
    np.testing.assert_array_equal(got, exp)


@pytest.mark.cuda
def test_fine_powers_rate_margin_report():
    cp = _import_cupy_or_skip()
    kernel = _kernel_or_skip()
    rng = np.random.default_rng(RNG_SEED + 2)
    streams = 2048
    rows = streams * WINDOWS
    buf = rng.integers(-14336, 14337, size=(TERMS, rows, 2), dtype=np.int32)
    d_rs = cp.asarray(buf)
    d_out = cp.zeros((TERMS, FINE_BINS), dtype=cp.uint64)

    def once():
        kernel.compute_fine_powers_u64(
            int(d_rs.data.ptr), streams, WINDOWS, 1, int(d_out.data.ptr)
        )

    once()  # warmup + JIT/caches
    cp.cuda.Device().synchronize()
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        once()
    cp.cuda.Device().synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0 / n
    frame_ms = 16384.0 / 390625.0 * 1000.0
    print(
        f"\nfine-power stage: {dt_ms:.3f} ms per 2048-stream frame "
        f"(frame cadence {frame_ms:.1f} ms; margin x{frame_ms / dt_ms:.0f})"
    )
    assert kernel.last_error() == ""
