# coding=utf-8
"""GPU bit-match gate for fxfft256 v1 (run on CUDA hosts, e.g. the A100).

This is the acceptance test the deployment decision defined: a CUDA
implementation of the frozen fixed-point FFT must reproduce the golden
vectors and the Python reference BIT-FOR-BIT --- no tolerances anywhere.
The device kernel below is the port prototype (one thread per stream,
serial 256-point transform, the structure documented in
``cuda/fxfft256_ref.c``); its twiddle table is generated from the frozen
``TWIDDLE_Q15`` literals at test time, so the source of truth stays
single. Production integration moves this transform into the
``f_statistic`` library after the row-sum stage; this gate proves the
arithmetic ports exactly before that wiring exists.

Gates:
1. golden vectors: device output == ``tests/data/fxfft256_golden_v1.npz``;
2. randomized sweep: device == Python reference on 4096 seeded vectors
   spanning deployed (2^14) and contract (2^20) amplitudes;
3. exact-integer feed sums: uint64 fine powers accumulated from device
   spectra equal ``fine_power_fx`` exactly.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pilot_proxy.fxfft import (
    INPUT_ABS_MAX,
    TWIDDLE_Q15,
    fine_power_fx,
    fxfft256,
)
from gpu import cuda_available

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO / "tests" / "data" / "fxfft256_golden_v1.npz"
RNG_SEED = 20260731


def _import_cupy_or_skip():
    try:
        import cupy as cp
    except Exception:  # pragma: no cover - GPU-less hosts
        pytest.skip("cupy is not available")
    ok, reason = cuda_available()
    if not ok:
        pytest.skip(f"CUDA device is not available: {reason}")
    return cp


def _device_source() -> str:
    table = ",\n".join(
        "    {%d, %d}" % (c, s) for c, s in TWIDDLE_Q15
    )
    return r"""
extern "C" {

#define FX_N 256
#define FX_NIN 128
#define FX_SHIFT 15
#define FX_ROUND 16384

__device__ __constant__ int FX_TW[128][2] = {
""" + table + r"""
};

__device__ __forceinline__ int round15(long long v)
{
    return (int)((v + FX_ROUND) >> FX_SHIFT);
}

__device__ __forceinline__ unsigned bitrev8(unsigned i)
{
    unsigned r = 0;
    for (unsigned k = 0; k < 8; ++k) r |= ((i >> k) & 1u) << (7u - k);
    return r;
}

/* One thread transforms one stream: in[128][2] -> out[256][2]. */
__global__ void fxfft256_kernel(const int *in, int *out, int n_vec)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vec) return;
    const int *src = in + (size_t)v * FX_NIN * 2;
    int x[FX_N][2];
    for (unsigned i = 0; i < FX_N; ++i) {
        unsigned s = bitrev8(i);
        if (s < FX_NIN) { x[i][0] = src[2 * s]; x[i][1] = src[2 * s + 1]; }
        else { x[i][0] = 0; x[i][1] = 0; }
    }
    for (unsigned s = 1; s <= 8; ++s) {
        unsigned m = 1u << s, half = m >> 1;
        for (unsigned j0 = 0; j0 < FX_N; j0 += m) {
            for (unsigned t = 0; t < half; ++t) {
                const int *w = FX_TW[t << (8u - s)];
                long long c = w[0], sn = w[1];
                long long br = x[j0 + t + half][0], bi = x[j0 + t + half][1];
                int tr = round15(br * c - bi * sn);
                int ti = round15(bi * c + br * sn);
                int ar = x[j0 + t][0], ai = x[j0 + t][1];
                x[j0 + t][0] = ar + tr;
                x[j0 + t][1] = ai + ti;
                x[j0 + t + half][0] = ar - tr;
                x[j0 + t + half][1] = ai - ti;
            }
        }
    }
    int *dst = out + (size_t)v * FX_N * 2;
    for (unsigned i = 0; i < FX_N; ++i) {
        dst[2 * i] = x[i][0];
        dst[2 * i + 1] = x[i][1];
    }
}

}  /* extern "C" */
"""


def _run_device(cp, x_int32: np.ndarray) -> np.ndarray:
    """x: int32 [n, 128, 2] host -> int32 [n, 256, 2] host via the GPU."""
    n = int(x_int32.shape[0])
    mod = cp.RawModule(code=_device_source())
    kern = mod.get_function("fxfft256_kernel")
    d_in = cp.asarray(np.ascontiguousarray(x_int32, dtype=np.int32))
    d_out = cp.zeros((n, 256, 2), dtype=cp.int32)
    threads = 64
    blocks = (n + threads - 1) // threads
    kern((blocks,), (threads,), (d_in, d_out, np.int32(n)))
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(d_out)


@pytest.mark.cuda
def test_device_matches_golden_vectors():
    cp = _import_cupy_or_skip()
    z = np.load(GOLDEN_PATH, allow_pickle=False)
    got = _run_device(cp, z["inputs"])
    assert (got == z["outputs"]).all()


@pytest.mark.cuda
def test_device_matches_python_on_random_sweep():
    cp = _import_cupy_or_skip()
    rng = np.random.default_rng(RNG_SEED)
    x14 = rng.integers(-16384, 16385, size=(2048, 128, 2), dtype=np.int64)
    x20 = rng.integers(
        -INPUT_ABS_MAX, INPUT_ABS_MAX + 1, size=(2048, 128, 2), dtype=np.int64
    )
    x = np.concatenate([x14, x20]).astype(np.int32)
    got = _run_device(cp, x)
    exp = fxfft256(x)
    assert (got == exp).all()


@pytest.mark.cuda
def test_device_feed_sums_exact():
    cp = _import_cupy_or_skip()
    rng = np.random.default_rng(RNG_SEED + 1)
    streams, windows = 32, 128
    terms = rng.integers(
        -2000, 2001, size=(3, streams * windows, 2), dtype=np.int64
    )
    X = _run_device(
        cp, terms.reshape(3 * streams, windows, 2).astype(np.int32)
    ).astype(np.int64)
    mag = (X[..., 0] ** 2 + X[..., 1] ** 2).reshape(3, streams, 256)
    power_dev = mag.sum(axis=1).astype(np.uint64)
    power_ref = fine_power_fx(terms, num_streams=streams, windows_per_stream=windows)
    assert power_dev.dtype == power_ref.dtype
    assert (power_dev == power_ref).all()
