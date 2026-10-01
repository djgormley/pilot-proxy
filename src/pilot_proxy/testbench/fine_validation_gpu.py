"""Literal paired fine-detector ablations resident on one CUDA device.

This module performs no threshold fitting or scientific acceptance. Each call
materializes independent Gaussian samples for every stream/window/sample. The
common input realization is complex64 (float32 normal draws and addition), as
in the existing literal sensitivity driver; all unquantized projections, FFTs,
and power sums are complex128/float64. "Floating" therefore describes detector
arithmetic, not an assertion of infinitely precise waveform/noise generation.

The joint quantized floating stage transforms exact integer row sums divided
by the two dequantization scales. This avoids recomputing an algebraically
identical dot product. Audits independently compare the explicitly unpacked
floating dot product, every fixed integer row, every fixed fine bin, and every
coarse marginal. No finite-stream pool or modeled aggregate is used.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from pilot_proxy.detector_reference import (
    matched_filter_row_projections_cpu_reference_packed,
    quantize_complex_numpy,
    unpack_packed_complex,
)
from pilot_proxy.fine_decision import fine_mask_decision, pack_bulk_mask
from pilot_proxy.fxfft import TWIDDLE_Q15, fine_power_fx
from pilot_proxy.testbench.sensitivity_study import (
    STAGE_ATSC_FLOAT,
    STAGE_FIXED_FLOAT_DECISION,
    STAGE_IDEAL_TONE_FLOAT,
    STAGE_INPUT_INT4,
    STAGE_JOINT_INT4_FLOAT,
    STAGE_WEIGHT_INT4,
)

K = 128
WINDOWS = 128
FINE_BINS = 256
BITS = 4
WEIGHT_SCALE = 7.0
DEFAULT_INPUT_SCALE = 7.0 / (3.0 / math.sqrt(2.0))
SCHEMA = "pilotproxy-literal-fine-gpu-v1"
FLOAT_STAGES = (
    STAGE_IDEAL_TONE_FLOAT,
    STAGE_ATSC_FLOAT,
    STAGE_INPUT_INT4,
    STAGE_WEIGHT_INT4,
    STAGE_JOINT_INT4_FLOAT,
)
ALL_STAGES = FLOAT_STAGES + (STAGE_FIXED_FLOAT_DECISION,)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return value


def _positive(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )
    return value


def integer_bounds(num_streams, packed_weights):
    """Conservative integer-only bounds for the actual signed-int4 profile."""
    streams = _integer(num_streams, "num_streams", 1, 2048)
    packed = np.asarray(packed_weights)
    if packed.shape != (3, K) or packed.dtype != np.dtype(np.int8):
        raise ValueError("packed weights must be int8 with shape [3,128]")
    weights = unpack_packed_complex(packed, BITS, dtype=np.float64)
    component = int(max(np.abs(weights.real).max(), np.abs(weights.imag).max()))
    row = 2 * K * 7 * component
    # Per-component butterfly: a + round15(c*b_real-s*b_imag).
    twiddle_l1 = max(abs(c) + abs(s) for c, s in TWIDDLE_Q15)
    transformed = row
    for _ in range(8):
        transformed += (twiddle_l1 * transformed + 16384) // 32768 + 1
    fine_sum = 2 * transformed * transformed * streams
    coarse_sum = 2 * row * row * streams * WINDOWS
    if (
        row > (1 << 20)
        or transformed >= (1 << 31)
        or max(fine_sum, coarse_sum) >= (1 << 63)
    ):
        raise OverflowError(
            "declared geometry exceeds exact signed accumulation bounds"
        )
    return {
        "row_component_abs_bound": row,
        "fft_component_abs_bound": transformed,
        "fine_power_sum_bound": fine_sum,
        "coarse_power_sum_bound": coarse_sum,
        "num_streams": streams,
    }


def arrays_sha256(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(str(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LiteralNoiseDraw:
    """One independent literal draw, reusable across paired stages/profiles.

    Treat samples as read-only. Engine evaluation never mutates them. Sharing a
    draw deliberately pairs profiles/SNRs and does not create independent draws.
    """

    seed: int
    num_streams: int
    device_id: int
    samples: object


def draw_noise(seed, *, num_streams, cp):
    seed = _integer(seed, "seed", 0, (1 << 64) - 1)
    streams = _integer(num_streams, "num_streams", 1, 2048)
    rng = cp.random.default_rng(seed % (1 << 63))
    shape = (streams, WINDOWS, K)
    real = rng.standard_normal(shape, dtype=cp.float32)
    imag = rng.standard_normal(shape, dtype=cp.float32)
    noise = (real + 1j * imag).astype(cp.complex64) * cp.float32(1.0 / math.sqrt(2.0))
    return LiteralNoiseDraw(seed, streams, int(cp.cuda.Device().id), noise)


_PACK_DEQUANTIZE_KERNEL = None


def _quantized_inputs(rows, input_scale, cp):
    """Fused literal int4 quantization, matching NumPy ties-to-even and clipping."""
    global _PACK_DEQUANTIZE_KERNEL
    scale = cp.float32(input_scale)
    if not bool(cp.isfinite(scale)) or not bool(scale > 0):
        raise ValueError("input scale is not finite positive float32")
    if _PACK_DEQUANTIZE_KERNEL is None:
        _PACK_DEQUANTIZE_KERNEL = cp.ElementwiseKernel(
            "complex64 value, float32 quant_scale, float64 dequant_scale",
            "int8 packed, complex128 dequantized, bool clipped",
            r"""
            const float sr = value.real() * quant_scale;
            const float si = value.imag() * quant_scale;
            clipped = (fabsf(sr) > 7.0f) || (fabsf(si) > 7.0f);
            const int re = (int)fminf(7.0f, fmaxf(-7.0f, rintf(sr)));
            const int im = (int)fminf(7.0f, fmaxf(-7.0f, rintf(si)));
            const unsigned int bits = ((unsigned int)(re & 15) << 4) | (unsigned int)(im & 15);
            packed = (signed char)bits;
            dequantized = complex<double>((double)re / dequant_scale, (double)im / dequant_scale);
            """,
            "fine_validation_pack_dequantize_v1",
        )
    packed, dequantized, clipped = _PACK_DEQUANTIZE_KERNEL(
        rows, scale, cp.float64(input_scale)
    )
    clip_count = int(cp.count_nonzero(clipped))
    return packed, dequantized, clip_count


@dataclass(frozen=True)
class PreparedNullInput:
    """One literal H0 input, packed/dequantized/cast once for paired profiles.

    Treat the arrays as read-only. This is the same Gaussian sample array, not
    a sufficient-statistic resampling approximation. Different profiles using
    it are paired and must not be counted as independent null trials.
    """

    noise_draw: LiteralNoiseDraw
    input_scale: float
    float_rows: object
    packed: object
    dequantized: object
    clip_count: int


def prepare_null_input(seed, *, num_streams, cp, input_scale=DEFAULT_INPUT_SCALE):
    scale = _positive(input_scale, "input_scale")
    draw = draw_noise(seed, num_streams=num_streams, cp=cp)
    rows = draw.samples.reshape(-1, K)
    if not bool(cp.all(cp.isfinite(rows))):
        raise ValueError("nonfinite literal null samples")
    packed, dequantized, count = _quantized_inputs(rows, scale, cp)
    return PreparedNullInput(
        draw, scale, rows.astype(cp.complex128), packed, dequantized, count
    )


def _decision(values):
    if values is None:
        return None
    keys = {
        "anchor_bin",
        "designated_half_width",
        "bulk_mask",
        "cfar_rank",
        "multiplier_q16",
    }
    if not isinstance(values, dict) or set(values) != keys:
        raise ValueError(
            "decision must specify exactly anchor, width, bulk, rank and multiplier"
        )
    mask = np.asarray(values["bulk_mask"])
    if mask.shape != (FINE_BINS,) or mask.dtype != np.dtype(bool):
        raise ValueError("decision bulk mask must be Boolean [256]")
    return {
        "anchor_bin": _integer(values["anchor_bin"], "anchor_bin", 0, 255),
        "designated_half_width": _integer(
            values["designated_half_width"], "designated_half_width", 0, 127
        ),
        "bulk_mask": mask.copy(),
        "cfar_rank": _integer(values["cfar_rank"], "cfar_rank", 0, 255),
        "multiplier_q16": _integer(
            values["multiplier_q16"], "multiplier_q16", 1, (1 << 64) - 1
        ),
    }


class FineValidationEngine:
    """Reusable literal engine; one instance fixes weights, signals and M."""

    def __init__(
        self,
        profile,
        atsc_rows,
        tone_rows,
        *,
        gpu,
        num_streams=2048,
        input_scale=DEFAULT_INPUT_SCALE,
    ):
        self.streams = _integer(num_streams, "num_streams", 1, 2048)
        self.input_scale = _positive(input_scale, "input_scale")
        self.packed_weights = np.array(profile.packed_weights, copy=True, order="C")
        self.bounds = integer_bounds(self.streams, self.packed_weights)
        self.ideal_weights = np.array(
            profile.ideal_weights, dtype=np.complex128, copy=True, order="C"
        )
        if (
            self.ideal_weights.shape != (3, K)
            or not np.isfinite(self.ideal_weights).all()
        ):
            raise ValueError("ideal weights must be finite [3,128]")
        self.packed_float_weights = (
            unpack_packed_complex(self.packed_weights, BITS) / WEIGHT_SCALE
        )
        self.signal_rows = {}
        for name, rows in (("atsc", atsc_rows), ("tone", tone_rows)):
            values = np.array(rows, dtype=np.complex64, copy=True, order="C")
            if values.shape != (WINDOWS, K) or not np.isfinite(values).all():
                raise ValueError(f"{name} signal must be finite [128,128]")
            self.signal_rows[name] = values
        self.gpu = gpu
        self.cp = gpu.cp
        self.kernel = gpu.kernel
        if (
            not self.kernel.supports_row_projections()
            or not self.kernel.supports_fine_powers()
        ):
            raise ValueError("kernel lacks exact row/fine APIs")
        specs = self.kernel.get_fine_specs()
        if specs != {
            "windows_per_stream": WINDOWS,
            "pad_factor": 2,
            "fine_bins": FINE_BINS,
        }:
            raise ValueError("kernel fine geometry differs")
        if (
            self.kernel.specs.detector_window_samples != K
            or self.kernel.specs.num_weight_terms != 3
            or self.kernel.specs.sample_bits_per_component != BITS
        ):
            raise ValueError("kernel matched-filter geometry differs")
        cp = self.cp
        self.ideal_device = cp.asarray(self.ideal_weights)
        self.packed_float_device = cp.asarray(self.packed_float_weights)
        self.combined_float_weights = cp.concatenate(
            (self.ideal_device, self.packed_float_device), axis=0
        )
        self.signals_device = {
            name: cp.asarray(rows) for name, rows in self.signal_rows.items()
        }
        self._cached_noise = None
        self.input_identity_sha256 = arrays_sha256(
            {
                "ideal_weights": self.ideal_weights,
                "packed_weights": self.packed_weights,
                "atsc_signal": self.signal_rows["atsc"],
                "tone_signal": self.signal_rows["tone"],
            }
        )

    def _float_powers(self, projection):
        cp = self.cp
        projected = projection.T.reshape(3, self.streams, WINDOWS)
        transformed = cp.fft.fft(projected, n=FINE_BINS, axis=-1)
        fine = cp.sum(
            transformed.real * transformed.real + transformed.imag * transformed.imag,
            axis=1,
            dtype=cp.float64,
        )
        coarse = cp.sum(
            projected.real * projected.real + projected.imag * projected.imag,
            axis=(1, 2),
            dtype=cp.float64,
        )
        return cp.asnumpy(fine), cp.asnumpy(coarse)

    def clear_noise_cache(self):
        """Release only this engine's optional last-draw cache."""
        self._cached_noise = None

    def evaluate(
        self,
        seed,
        amplitude,
        *,
        audit=False,
        noise_draw=None,
        reuse_noise=False,
        decision=None,
        null_input=None,
    ):
        """Return full powers plus exact provenance; audit additionally retains inputs.

        Raw seeds are uint64 identities; CuPy receives seed modulo 2**63.
        The caller must freeze and check uniqueness of effective seeds across
        calibration/evaluation. This engine does not infer independence from
        stage labels or collect old observations.
        """
        seed = _integer(seed, "seed", 0, (1 << 64) - 1)
        amplitude = _positive(amplitude, "amplitude", allow_zero=True)
        if type(audit) is not bool:
            raise TypeError("audit must be Boolean")
        cp = self.cp
        if int(cp.cuda.get_current_stream().ptr) != 0:
            raise ValueError(
                "the exact CUDA wrapper uses the default stream; evaluate there"
            )
        if type(reuse_noise) is not bool:
            raise TypeError("reuse_noise must be Boolean")
        selected_decision = _decision(decision)
        if selected_decision is not None and not self.kernel.supports_fused_fine_mask():
            raise ValueError("selected kernel lacks the device Q16 epilogue")
        if null_input is not None:
            if not isinstance(null_input, PreparedNullInput):
                raise TypeError("null_input must come from prepare_null_input")
            if amplitude != 0 or noise_draw is not None or reuse_noise:
                raise ValueError(
                    "prepared H0 input requires amplitude zero and no other noise source"
                )
            if null_input.input_scale != self.input_scale:
                raise ValueError("prepared H0 quantization scale differs")
            noise_draw = null_input.noise_draw
        if noise_draw is not None and reuse_noise:
            raise ValueError("choose explicit noise_draw or engine cache, not both")
        if noise_draw is None and reuse_noise and self._cached_noise is not None:
            if self._cached_noise.seed == seed:
                noise_draw = self._cached_noise
        if noise_draw is None:
            noise_draw = draw_noise(seed, num_streams=self.streams, cp=cp)
        if not isinstance(noise_draw, LiteralNoiseDraw):
            raise TypeError("noise_draw must come from draw_noise")
        if (
            noise_draw.seed != seed
            or noise_draw.num_streams != self.streams
            or noise_draw.device_id != int(cp.cuda.Device().id)
        ):
            raise ValueError("shared noise identity/geometry/device mismatch")
        noise = noise_draw.samples
        if (
            not isinstance(noise, cp.ndarray)
            or noise.dtype != cp.complex64
            or noise.shape != (self.streams, WINDOWS, K)
        ):
            raise ValueError("shared noise shape/dtype mismatch")
        if reuse_noise:
            self._cached_noise = noise_draw
        signal_amp = cp.float32(amplitude)
        if not bool(cp.isfinite(signal_amp)):
            raise ValueError("amplitude overflows the declared complex64 input model")
        if null_input is None:
            atsc = (
                noise + signal_amp * self.signals_device["atsc"][None, :, :]
            ).reshape(-1, K)
            if not bool(cp.all(cp.isfinite(atsc))):
                raise ValueError("nonfinite generated detector input")
            packed, dequantized, clip_count = _quantized_inputs(
                atsc, self.input_scale, cp
            )
        else:
            atsc = noise.reshape(-1, K)
            packed, dequantized, clip_count = (
                null_input.packed,
                null_input.dequantized,
                null_input.clip_count,
            )
            expected_shape = (self.streams * WINDOWS, K)
            for values, dtype in (
                (packed, cp.int8),
                (dequantized, cp.complex128),
                (null_input.float_rows, cp.complex128),
            ):
                if (
                    not isinstance(values, cp.ndarray)
                    or values.shape != expected_shape
                    or values.dtype != dtype
                ):
                    raise ValueError("prepared H0 array geometry/dtype differs")
        fine, coarse, projections = {}, {}, {}

        def store_projection(stage, projection):
            fine[stage], coarse[stage] = self._float_powers(projection)
            if audit:
                projections[stage] = cp.asnumpy(
                    projection.T.reshape(3, self.streams, WINDOWS)
                )

        def floating(stage, rows, weights):
            projection = (
                rows.astype(cp.complex128, copy=False) @ cp.conjugate(weights).T
            )
            store_projection(stage, projection)

        tone_host = None
        if amplitude != 0.0:
            tone = (
                noise + signal_amp * self.signals_device["tone"][None, :, :]
            ).reshape(-1, K)
            if not bool(cp.all(cp.isfinite(tone))):
                raise ValueError("nonfinite generated tone input")
            floating(STAGE_IDEAL_TONE_FLOAT, tone, self.ideal_device)
            tone_host = cp.asnumpy(tone) if audit else None
            del tone
        del noise
        atsc_float = (
            atsc.astype(cp.complex128) if null_input is None else null_input.float_rows
        )
        combined_projection = atsc_float @ cp.conjugate(self.combined_float_weights).T
        store_projection(STAGE_ATSC_FLOAT, combined_projection[:, :3])
        store_projection(STAGE_WEIGHT_INT4, combined_projection[:, 3:])
        if amplitude == 0.0:
            fine[STAGE_IDEAL_TONE_FLOAT] = fine[STAGE_ATSC_FLOAT].copy()
            coarse[STAGE_IDEAL_TONE_FLOAT] = coarse[STAGE_ATSC_FLOAT].copy()
            if audit:
                projections[STAGE_IDEAL_TONE_FLOAT] = projections[
                    STAGE_ATSC_FLOAT
                ].copy()
                tone_host = cp.asnumpy(atsc)
        del atsc_float, combined_projection
        floating(STAGE_INPUT_INT4, dequantized, self.ideal_device)
        del dequantized

        rows = self.streams * WINDOWS
        diagnostic = cp.zeros(1, dtype=cp.float32)
        fixed_rows = cp.zeros((3, rows, 2), dtype=cp.int32)
        fixed_fine = cp.zeros((3, FINE_BINS), dtype=cp.uint64)
        fixed_coarse = cp.zeros(3, dtype=cp.uint64)
        handle = self.kernel.create_raw(rows, packed.data.ptr, diagnostic.data.ptr)
        device_mask = None
        try:
            if selected_decision is not None:
                d_mask = cp.zeros(1, dtype=cp.int32)
                self.kernel.compute_fused_fine_mask_u64(
                    handle,
                    self.packed_weights.ctypes.data,
                    selected_decision["anchor_bin"],
                    selected_decision["designated_half_width"],
                    pack_bulk_mask(selected_decision["bulk_mask"]),
                    selected_decision["cfar_rank"],
                    selected_decision["multiplier_q16"],
                    fixed_fine.data.ptr,
                    d_mask.data.ptr,
                    fixed_coarse.data.ptr,
                    fixed_rows.data.ptr,
                )
                cp.cuda.get_current_stream().synchronize()
                device_mask = int(cp.asnumpy(d_mask)[0])
                execution = "fused_fine_coarse_device_q16_with_exact_row_tap"
            elif self.kernel.supports_fused_fine():
                self.kernel.compute_fused_fine_u64(
                    handle,
                    self.packed_weights.ctypes.data,
                    fixed_fine.data.ptr,
                    fixed_coarse.data.ptr,
                    fixed_rows.data.ptr,
                )
                execution = "fused_fine_coarse_with_exact_row_tap"
            else:
                self.kernel.compute_row_projections_i32(
                    handle, self.packed_weights.ctypes.data, fixed_rows.data.ptr
                )
                self.kernel.compute_fine_powers_u64(
                    fixed_rows.data.ptr, self.streams, WINDOWS, 1, fixed_fine.data.ptr
                )
                self.kernel.compute_powers_u64(
                    handle, self.packed_weights.ctypes.data, fixed_coarse.data.ptr
                )
                execution = "composed_exact_row_fine_coarse"
            cp.cuda.get_current_stream().synchronize()
            fine[STAGE_FIXED_FLOAT_DECISION] = cp.asnumpy(fixed_fine)
            coarse[STAGE_FIXED_FLOAT_DECISION] = cp.asnumpy(fixed_coarse)
            fixed_float_projection = fixed_rows[..., 0].astype(
                cp.float64
            ) + 1j * fixed_rows[..., 1].astype(cp.float64)
            fixed_float_projection /= self.input_scale * WEIGHT_SCALE
            fine[STAGE_JOINT_INT4_FLOAT], coarse[STAGE_JOINT_INT4_FLOAT] = (
                self._float_powers(fixed_float_projection.T)
            )
            if audit:
                projections[STAGE_JOINT_INT4_FLOAT] = cp.asnumpy(
                    fixed_float_projection.reshape(3, self.streams, WINDOWS)
                )
        finally:
            self.kernel.destroy(handle)
        for stage in ALL_STAGES:
            if fine[stage].shape != (3, FINE_BINS) or coarse[stage].shape != (3,):
                raise AssertionError("power output shape differs")
            if stage == STAGE_FIXED_FLOAT_DECISION:
                if fine[stage].dtype != np.uint64 or coarse[stage].dtype != np.uint64:
                    raise AssertionError(
                        "fixed powers lost exact uint64 representation"
                    )
                if (
                    int(fine[stage].max()) > self.bounds["fine_power_sum_bound"]
                    or int(coarse[stage].max()) > self.bounds["coarse_power_sum_bound"]
                ):
                    raise OverflowError("fixed powers exceed proven bound")
            elif not (
                np.isfinite(fine[stage]).all() and np.isfinite(coarse[stage]).all()
            ):
                raise ValueError("nonfinite floating power output")
        if selected_decision is not None:
            expected_mask = fine_mask_decision(
                fine[STAGE_FIXED_FLOAT_DECISION], **selected_decision
            )
            if expected_mask.mask != device_mask:
                raise AssertionError(
                    "device Q16 epilogue differs from exact host decision"
                )
        result = {
            "schema": SCHEMA,
            "num_streams": self.streams,
            "frame_seed": seed,
            "effective_seed63": seed % (1 << 63),
            "amplitude": amplitude,
            "input_scale": self.input_scale,
            "clip_count": clip_count,
            "sample_count": self.streams * WINDOWS * K,
            "clip_fraction": clip_count / (self.streams * WINDOWS * K),
            "powers_by_stage": fine,
            "coarse_by_stage": coarse,
            "integer_bounds": dict(self.bounds),
            "input_identity_sha256": self.input_identity_sha256,
            "execution_form": execution,
            "device_q16_mask": device_mask,
            "device_q16_checked": selected_decision is not None,
            "input_arithmetic": "complex64 independent Gaussian samples and additive signals; complex128 detector projection/FFT, float64 sums",
            "physical_certification": False,
        }
        result["output_sha256"] = arrays_sha256(
            {
                **{f"fine/{k}": v for k, v in fine.items()},
                **{f"coarse/{k}": v for k, v in coarse.items()},
            }
        )
        if audit:
            result["audit_inputs"] = {
                "atsc": cp.asnumpy(atsc),
                "tone": tone_host,
                "packed": cp.asnumpy(packed),
                "integer_rows": cp.asnumpy(fixed_rows),
                "projections_by_stage": projections,
            }
        return result


def audit_against_cpu(engine, result, *, rtol=2e-11, atol=1e-8):
    """Independent NumPy arithmetic audit of one retained literal realization."""
    if "audit_inputs" not in result:
        raise ValueError("evaluate(..., audit=True) is required")
    data = result["audit_inputs"]
    packed = quantize_complex_numpy(data["atsc"], BITS, engine.input_scale)
    np.testing.assert_array_equal(packed, data["packed"])
    rows = matched_filter_row_projections_cpu_reference_packed(
        packed, engine.packed_weights, BITS
    )
    np.testing.assert_array_equal(rows, data["integer_rows"])
    fixed_fine = fine_power_fx(rows, num_streams=engine.streams)
    fixed_coarse = np.sum(
        rows.astype(np.int64) ** 2, axis=(1, 2), dtype=np.int64
    ).astype(np.uint64)
    np.testing.assert_array_equal(
        fixed_fine, result["powers_by_stage"][STAGE_FIXED_FLOAT_DECISION]
    )
    np.testing.assert_array_equal(
        fixed_coarse, result["coarse_by_stage"][STAGE_FIXED_FLOAT_DECISION]
    )
    dequantized = unpack_packed_complex(packed, BITS) / engine.input_scale
    comparisons = []
    for stage, values, weights in (
        (STAGE_IDEAL_TONE_FLOAT, data["tone"], engine.ideal_weights),
        (STAGE_ATSC_FLOAT, data["atsc"], engine.ideal_weights),
        (STAGE_INPUT_INT4, dequantized, engine.ideal_weights),
        (STAGE_WEIGHT_INT4, data["atsc"], engine.packed_float_weights),
        (STAGE_JOINT_INT4_FLOAT, dequantized, engine.packed_float_weights),
    ):
        projection = values.astype(np.complex128) @ weights.conjugate().T
        z = projection.T.reshape(3, engine.streams, WINDOWS)
        np.testing.assert_allclose(
            z, data["projections_by_stage"][stage], rtol=rtol, atol=atol
        )
        transformed = np.fft.fft(z, n=FINE_BINS, axis=-1)
        fine = np.sum(np.abs(transformed) ** 2, axis=1, dtype=np.float64)
        coarse = np.sum(np.abs(z) ** 2, axis=(1, 2), dtype=np.float64)
        for name, expected, actual in (
            ("fine", fine, result["powers_by_stage"][stage]),
            ("coarse", coarse, result["coarse_by_stage"][stage]),
        ):
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
            comparisons.append(
                {
                    "stage": stage,
                    "quantity": name,
                    "maximum_relative_error": float(
                        np.max(
                            np.abs(actual - expected)
                            / np.maximum(np.abs(expected), np.finfo(float).tiny)
                        )
                    ),
                }
            )
    return {
        "passed": True,
        "schema": SCHEMA,
        "frame_seed": result["frame_seed"],
        "effective_seed63": result["effective_seed63"],
        "num_streams": engine.streams,
        "packing_exact": True,
        "integer_rows_exact": True,
        "fixed_fine_exact": True,
        "fixed_coarse_exact": True,
        "float_comparisons": comparisons,
        "float_rtol": rtol,
        "float_atol": atol,
        "output_sha256": result["output_sha256"],
        "input_identity_sha256": engine.input_identity_sha256,
    }
