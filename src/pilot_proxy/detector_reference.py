#!/usr/bin/env python3
# coding=utf-8
"""
Pure-NumPy reference implementation for the local-reference power ratio kernel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

INT4_COMPONENT_BITS = 4
INT8_COMPONENT_BITS = 8
REFERENCE_WEIGHT_TERMS = 3
REFERENCE_TARGET_TERM_INDEX = 0
REFERENCE_LOWER_TERM_INDEX = 1
REFERENCE_UPPER_TERM_INDEX = 2
COARSE_POWER_RATIO_SCALE = 2.0


def packed_dtype_for_component_bits(bits: int) -> np.dtype:
    """Return a packed integer dtype for a per-component bit depth."""
    bits = int(bits)
    if bits == INT4_COMPONENT_BITS:
        return np.dtype(np.int8)
    if bits == INT8_COMPONENT_BITS:
        return np.dtype(np.int16)
    raise ValueError(f"Unsupported component bit depth: {bits}. Expected 4 or 8.")


def quantize_complex_numpy(
    data: np.ndarray,
    bits: int,
    scale: float,
) -> np.ndarray:
    """Quantize complex data to packed integer format (NumPy version)."""
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (M, K). Got shape {data.shape}.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("quantization scale must be positive and finite.")
    if not np.all(np.isfinite(data)):
        raise ValueError("quantization samples must be finite.")
    packed_dtype = packed_dtype_for_component_bits(bits)
    max_int = (1 << (bits - 1)) - 1
    mask = (1 << bits) - 1

    r = np.asarray(
        np.clip(np.round(data.real * scale), -max_int, max_int),
        dtype=np.int32,
    )
    i = np.asarray(
        np.clip(np.round(data.imag * scale), -max_int, max_int),
        dtype=np.int32,
    )
    packed = np.asarray((r << bits) | (i & mask), dtype=packed_dtype)
    return np.ascontiguousarray(packed)


def unpack_packed_complex(
    packed: np.ndarray,
    bits: int,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Unpack packed integer samples to a complex array."""
    packed_dtype = packed_dtype_for_component_bits(bits)
    p = np.asarray(packed, dtype=packed_dtype).astype(np.int32, copy=False)
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)

    real = p >> bits
    imag_raw = p & mask
    imag = np.where(imag_raw & sign_bit, imag_raw - (1 << bits), imag_raw)

    return real.astype(dtype) + 1j * imag.astype(dtype)  # type: ignore[arg-type]


def coarse_power_ratio_cpu_reference(
    samples: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute local-reference power ratio using natural weights and x * conj(w) dot products."""
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (M, K). Got shape {samples.shape}.")
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D (N, K). Got shape {weights.shape}.")
    if weights.shape[0] != REFERENCE_WEIGHT_TERMS:
        raise ValueError("weights must have N=3 (target, ref+, ref-).")
    if samples.shape[1] != weights.shape[1]:
        raise ValueError("samples K dimension must match weights K dimension.")

    dots = samples @ np.conjugate(weights).T
    power = np.abs(dots) ** 2
    sums = np.sum(power, axis=0)
    denom = float(sums[REFERENCE_LOWER_TERM_INDEX] + sums[REFERENCE_UPPER_TERM_INDEX])
    eps = np.finfo(np.float64).tiny
    if denom <= eps:
        return 0.0, sums
    power_ratio = float(
        COARSE_POWER_RATIO_SCALE * sums[REFERENCE_TARGET_TERM_INDEX] / denom
    )
    return power_ratio, sums


def coarse_power_ratio_cpu_reference_packed(
    packed_samples: np.ndarray,
    packed_weights: np.ndarray,
    bits: int,
) -> tuple[float, np.ndarray]:
    """Compute local-reference power ratio from packed integer samples and weights."""
    samples = unpack_packed_complex(packed_samples, bits)
    weights = unpack_packed_complex(packed_weights, bits)
    return coarse_power_ratio_cpu_reference(samples, weights)


def matched_filter_row_projections_cpu_reference_packed(
    packed_samples: np.ndarray,
    packed_weights: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Return the kernel's exact integer matched-filter row projections.

    The result has shape ``[terms, rows, 2]`` and dtype ``int32``.  Its
    arithmetic and layout are the CPU specification for
    ``FStat_Compute_RowSums_I32``: packed signed components are sign-extended,
    each row is dotted with the conjugated weight vector, and the real and
    imaginary integer sums are retained separately.  This public reference is
    used by the synthetic sensitivity ablations as the boundary between
    input/weight quantization and the frozen fixed-point fine transform.
    """
    samples = np.asarray(packed_samples)
    weights = np.asarray(packed_weights)
    if samples.ndim != 2:
        raise ValueError(
            f"packed_samples must be 2D (rows, K). Got shape {samples.shape}."
        )
    if weights.ndim != 2:
        raise ValueError(
            f"packed_weights must be 2D (terms, K). Got shape {weights.shape}."
        )
    if weights.shape[0] != REFERENCE_WEIGHT_TERMS:
        raise ValueError("packed_weights must have N=3 (target, ref+, ref-).")
    if samples.shape[1] != weights.shape[1]:
        raise ValueError("packed sample and weight K dimensions must match.")

    x = unpack_packed_complex(samples, bits, dtype=np.int64)
    w = unpack_packed_complex(weights, bits, dtype=np.int64)
    xr = np.asarray(x.real, dtype=np.int64)
    xi = np.asarray(x.imag, dtype=np.int64)
    wr = np.asarray(w.real, dtype=np.int64)
    wi = np.asarray(w.imag, dtype=np.int64)

    # z[n, m] = sum_k x[m, k] * conj(w[n, k]).  Integer inputs keep the
    # einsums exact in int64; the detector's int4/K<=128 bound fits int32.
    real = np.einsum("mk,nk->nm", xr, wr, optimize=True)
    real += np.einsum("mk,nk->nm", xi, wi, optimize=True)
    imag = np.einsum("mk,nk->nm", xi, wr, optimize=True)
    imag -= np.einsum("mk,nk->nm", xr, wi, optimize=True)
    out = np.stack((real, imag), axis=-1)
    info = np.iinfo(np.int32)
    if np.any(out < info.min) or np.any(out > info.max):
        raise OverflowError("matched-filter row projection exceeds int32.")
    return np.ascontiguousarray(out.astype(np.int32))
