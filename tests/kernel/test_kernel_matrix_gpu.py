"""Build and exercise both receiver kernels against independent integer oracles."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from pilot_proxy.fine_decision import fine_mask_decision, pack_bulk_mask
from pilot_proxy.fxfft import fine_power_fx
from gpu import cuda_available
from pilot_proxy.kernel import FStatKernel

pytestmark = pytest.mark.cuda
ROOT = Path(__file__).resolve().parents[2]
WINDOWS = 128

# Exercise each implementation choice at both K values, including grid stride.
CONFIGURATIONS = [
    pytest.param((k, mode, threads, grid), id=f"k{k}-{mode}-t{threads}-g{grid}")
    for k in (64, 128)
    for mode, threads, grid in (
        ("shared", 64, 4096),
        ("scalar", 32, 4),
        ("global", 128, 4),
        ("constant", 64, 4),
    )
]

COMPARE_SHIM = r"""
__global__ void test_wide_compare_kernel(const unsigned long long* v,
                                        int* out, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned long long* p = v + 6 * i;
    out[2 * i] = fstat_frac_less(p[0], p[1], p[2], p[3]);
    out[2 * i + 1] = fstat_triple_greater(p[0], p[1], p[2], p[3], p[4], p[5]);
}
extern "C" void test_wide_compare(const unsigned long long* v, int* out, int count) {
    test_wide_compare_kernel<<<(count + 127) / 128, 128>>>(v, out, count);
}
"""


@pytest.fixture(scope="module", params=CONFIGURATIONS)
def kernel_build(request, tmp_path_factory):
    cp = pytest.importorskip("cupy")
    available, reason = cuda_available()
    if not available:
        pytest.skip(reason)
    nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
    if nvcc is None:
        pytest.skip("nvcc is required for the kernel build matrix")
    k, mode, threads, grid = request.param
    build = tmp_path_factory.mktemp(f"kernel-{k}-{mode}")
    source = build / "f_statistic.cu"
    source.write_text((ROOT / "cuda/f_statistic.cu").read_text() + COMPARE_SHIM)
    library = build / "libfstatistic.so"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++14",
            "-O3",
            "-DNDEBUG",
            "-use_fast_math",
            "-Xcompiler=-fPIC",
            "--shared",
            f"-arch=sm_{cp.cuda.Device().compute_capability}",
            f"-DFSTAT_DETECTOR_WINDOW_SAMPLES={k}",
            f"-DFSTAT_USE_DP4A={int(mode != 'scalar')}",
            f"-DFSTAT_USE_SHARED_WEIGHT_LANES={int(mode == 'shared')}",
            f"-DFSTAT_USE_CONSTANT_WEIGHT_LANES={int(mode == 'constant')}",
            f"-DFSTAT_BLOCK_THREADS={threads}",
            f"-DFSTAT_GRID_MAX_BLOCKS={grid}",
            "-I",
            str(ROOT / "cuda"),
            str(source),
            "-o",
            str(library),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    kernel = FStatKernel(library)
    assert kernel.specs.detector_window_samples == k
    assert kernel.features.use_dp4a == (mode != "scalar")
    assert kernel.features.block_threads == threads
    assert kernel.features.grid_max_blocks == grid
    assert kernel.supports_fused_fine_mask()
    return cp, kernel, k


def integer_rows(packed, weights):
    """Direct signed-nibble dot products, independent of the CUDA unpacker."""
    x = packed.astype(np.int64) & 255
    w = weights.astype(np.int64) & 255
    xr, xi = ((x >> 4) ^ 8) - 8, ((x & 15) ^ 8) - 8
    wr, wi = ((w >> 4) ^ 8) - 8, ((w & 15) ^ 8) - 8
    real = xr @ wr.T + xi @ wi.T
    imag = xi @ wr.T - xr @ wi.T
    return np.stack((real.T, imag.T), axis=-1)


@pytest.mark.parametrize(
    "streams,batch,pattern",
    [
        (1, 1, "zero"),
        (3, 3, "random"),
        (5, 2, "extremes"),
        (128, 1, "sparse16"),
        (128, 1, "sparse32"),
    ],
)
def test_all_exact_paths_and_reused_outputs(kernel_build, streams, batch, pattern):
    cp, kernel, k = kernel_build
    rng = np.random.default_rng(20260909)
    rows = streams * WINDOWS
    packed = rng.integers(0, 256, (batch, rows, k), dtype=np.uint8).view(np.int8)
    weights = rng.integers(0, 256, (3, k), dtype=np.uint8).view(np.int8)
    if pattern == "zero":
        packed.fill(0)
    elif pattern == "extremes":
        packed[:] = np.resize(
            np.array([0x88, 0x77, 0x87, 0x78], np.uint8).view(np.int8), packed.shape
        )
        weights[:] = packed[0, :3]
    elif pattern.startswith("sparse"):
        packed[:, int(pattern[6:]) * WINDOWS :] = 0

    expected_rows = np.stack([integer_rows(frame, weights) for frame in packed])
    expected_power = (expected_rows**2).sum(axis=(2, 3)).astype(np.uint64)
    expected_fine = np.stack(
        [fine_power_fx(frame, num_streams=streams) for frame in expected_rows]
    )
    device_input = cp.asarray(packed)
    diag = cp.full(batch * 3, -1, dtype=cp.float32)
    handle = kernel.create_raw_batch(rows, batch, device_input.data.ptr, diag.data.ptr)
    tap = cp.full((batch, 3, rows, 2), -1, dtype=cp.int32)
    powers = cp.full((batch, 3), 123, dtype=cp.uint64)
    fine = cp.full((batch, 3, 256), 123, dtype=cp.uint64)
    mask = cp.full(batch, -1, dtype=cp.int32)
    coarse_mask = cp.full(batch, 255, dtype=cp.uint8)
    num = cp.full(batch, 123, dtype=cp.uint64)
    den = cp.full(batch, 123, dtype=cp.uint64)
    overflow = cp.full(1, 123, dtype=cp.uint64)
    bulk = np.zeros(256, dtype=bool)
    bulk[::2] = True
    try:
        kernel.compute_row_projections_i32(handle, weights.ctypes.data, tap.data.ptr)
        np.testing.assert_array_equal(cp.asnumpy(tap), expected_rows)
        kernel.compute_powers_u64(handle, weights.ctypes.data, powers.data.ptr)
        np.testing.assert_array_equal(cp.asnumpy(powers), expected_power)
        kernel.compute_powers(handle, weights.ctypes.data)
        np.testing.assert_array_equal(
            cp.asnumpy(diag).reshape(batch, 3), expected_power.astype(np.float32)
        )
        kernel.compute_diagnostic_float(handle, weights.ctypes.data)
        denominator = expected_power[:, 1].astype(np.float64) + expected_power[:, 2]
        expected_ratio = np.divide(
            2 * expected_power[:, 0],
            denominator,
            out=np.zeros(batch),
            where=denominator != 0,
        )
        np.testing.assert_allclose(
            cp.asnumpy(diag)[:batch], expected_ratio, rtol=2e-6, atol=0
        )
        kernel.compute_fine_powers_u64(
            tap.data.ptr, streams, WINDOWS, batch, fine.data.ptr
        )
        np.testing.assert_array_equal(cp.asnumpy(fine), expected_fine)
        kernel.compute_fused_fine_u64(
            handle, weights.ctypes.data, fine.data.ptr, powers.data.ptr, tap.data.ptr
        )
        np.testing.assert_array_equal(cp.asnumpy(fine), expected_fine)
        np.testing.assert_array_equal(cp.asnumpy(powers), expected_power)
        np.testing.assert_array_equal(cp.asnumpy(tap), expected_rows)

        for numerator, denominator in ((0, 1), (1, 1), (2, 3)):
            kernel.compute_numden_mask_rational_half_checked(
                handle,
                weights.ctypes.data,
                numerator,
                denominator,
                num.data.ptr,
                den.data.ptr,
                coarse_mask.data.ptr,
                overflow.data.ptr,
            )
            ref_den = expected_power[:, 1] + expected_power[:, 2]
            ref_mask = (ref_den != 0) & (
                expected_power[:, 0] * denominator > ref_den * numerator
            )
            np.testing.assert_array_equal(cp.asnumpy(num), expected_power[:, 0])
            np.testing.assert_array_equal(cp.asnumpy(den), ref_den)
            np.testing.assert_array_equal(cp.asnumpy(coarse_mask), ref_mask)
            assert int(overflow.get()[0]) == 0

        # Repeated calls must clear counters and powers, including zero frames.
        for anchor, width, rank, multiplier in (
            (0, 2, 0, 1),
            (255, 127, 127, (1 << 64) - 1),
            (63, 0, 255, 65536),
        ):
            expected_mask = [
                fine_mask_decision(
                    frame,
                    anchor_bin=anchor,
                    designated_half_width=width,
                    bulk_mask=bulk,
                    cfar_rank=rank,
                    multiplier_q16=multiplier,
                ).mask
                for frame in expected_fine
            ]
            for taps in (True, False, False):
                kernel.compute_fused_fine_mask_u64(
                    handle,
                    weights.ctypes.data,
                    anchor,
                    width,
                    pack_bulk_mask(bulk),
                    rank,
                    multiplier,
                    fine.data.ptr,
                    mask.data.ptr,
                    powers.data.ptr if taps else 0,
                    tap.data.ptr if taps else 0,
                )
                np.testing.assert_array_equal(cp.asnumpy(fine), expected_fine)
                np.testing.assert_array_equal(cp.asnumpy(mask), expected_mask)
                if taps:
                    np.testing.assert_array_equal(cp.asnumpy(powers), expected_power)
                    np.testing.assert_array_equal(cp.asnumpy(tap), expected_rows)
        assert kernel.last_error() == ""
    finally:
        kernel.destroy(handle)


def test_weight_changes_across_live_handles(kernel_build):
    cp, kernel, k = kernel_build
    rng = np.random.default_rng(812)
    packed = rng.integers(0, 256, (WINDOWS, k), dtype=np.uint8).view(np.int8)
    weights = [
        rng.integers(0, 256, (3, k), dtype=np.uint8).view(np.int8) for _ in range(2)
    ]
    device_input = cp.asarray(packed)
    diag = cp.zeros(1, dtype=cp.float32)
    powers = cp.empty(3, dtype=cp.uint64)
    handles = []
    try:
        for _ in range(2):
            handles.append(
                kernel.create_raw(WINDOWS, device_input.data.ptr, diag.data.ptr)
            )
        for index in (0, 1, 0, 1, 0):
            kernel.compute_powers_u64(
                handles[index], weights[index].ctypes.data, powers.data.ptr
            )
            expected = (integer_rows(packed, weights[index]) ** 2).sum(axis=(1, 2))
            np.testing.assert_array_equal(cp.asnumpy(powers), expected)
        # A host buffer can change without its address changing.
        weights[0][:] = weights[1]
        kernel.compute_powers_u64(handles[0], weights[0].ctypes.data, powers.data.ptr)
        expected = (integer_rows(packed, weights[0]) ** 2).sum(axis=(1, 2))
        np.testing.assert_array_equal(cp.asnumpy(powers), expected)
    finally:
        for handle in handles:
            kernel.destroy(handle)


def test_device_wide_integer_compares_against_python_bigints(kernel_build):
    cp, kernel, _ = kernel_build
    rng = np.random.default_rng(893)
    values = rng.integers(0, 1 << 64, (10000, 6), dtype=np.uint64)
    edges = [0, 1, (1 << 32) - 1, 1 << 32, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
    values[: len(edges)] = np.array(edges, dtype=np.uint64)[:, None]
    # Equal products exercise the strict comparison at all magnitudes.
    values[100:200, 3:] = values[100:200, :3]
    expected = []
    for row in values:
        a, b, c, d, e, f = map(int, row)
        expected.append((a * d < c * b, a * b * c > d * e * f))
    device_values = cp.asarray(values)
    output = cp.empty((len(values), 2), dtype=cp.int32)
    shim = kernel._lib.test_wide_compare
    shim.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    shim.restype = None
    shim(device_values.data.ptr, output.data.ptr, len(values))
    np.testing.assert_array_equal(cp.asnumpy(output), expected)


def test_native_error_does_not_poison_the_next_call(kernel_build):
    cp, kernel, k = kernel_build
    weights = np.ones((3, k), dtype=np.int8)
    powers = cp.empty(3, dtype=cp.uint64)
    with pytest.raises(RuntimeError, match="handle is null"):
        kernel.compute_powers_u64(None, weights.ctypes.data, powers.data.ptr)
    packed = cp.zeros((WINDOWS, k), dtype=cp.int8)
    diag = cp.zeros(1, dtype=cp.float32)
    handle = kernel.create_raw(WINDOWS, packed.data.ptr, diag.data.ptr)
    try:
        kernel.compute_powers_u64(handle, weights.ctypes.data, powers.data.ptr)
        np.testing.assert_array_equal(cp.asnumpy(powers), [0, 0, 0])
        assert kernel.last_error() == ""
    finally:
        kernel.destroy(handle)
