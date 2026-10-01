# coding=utf-8
"""Core arithmetic for the current-geometry synthetic sensitivity study.

The command-line driver lives in ``tools/current_geometry_sensitivity.py``.
This module contains the small, testable pieces that define the experiment:
order-independent common-random-number seeds, the float and exact fine
statistics, conservative null calibration, exact Q16 decisions, detection
intervals, crossing brackets, and a paired bootstrap comparison.

The study intentionally stores a *response ratio* rather than a decision in
its resumable shards.  For one frame the ratio is

``max(F2 in designated set) / F2[null-bulk rank]``.

That is sufficient to apply a float multiplier after all null shards have
been collected.  The fixed-point path additionally stores the four exact
integer numerator/denominator fields, so its Q16 comparison is replayed with
Python integers and does not depend on a float value rounded into an NPZ.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pilot_proxy.fine_reduction import (
    WEIGHT_TERM_REF_LOWER,
    WEIGHT_TERM_REF_UPPER,
    WEIGHT_TERM_TARGET,
)

SENSITIVITY_STUDY_SCHEMA = "pilotproxy_current_geometry_sensitivity_v1"
FINE_BINS = 256
Q16_ONE = 1 << 16

STAGE_IDEAL_TONE_FLOAT = "ideal_tone_float"
STAGE_ATSC_FLOAT = "atsc_8vsb_float"
STAGE_INPUT_INT4 = "atsc_8vsb_input_int4_float"
STAGE_WEIGHT_INT4 = "atsc_8vsb_weight_int4_float"
STAGE_JOINT_INT4_FLOAT = "atsc_8vsb_joint_int4_float_transform"
STAGE_FIXED_FLOAT_DECISION = "atsc_8vsb_fixed_transform_float_decision"
STAGE_FIXED_Q16_CPU = "atsc_8vsb_fixed_transform_q16_cpu"
STAGE_FULL_GPU = "atsc_8vsb_packed_gpu_fine_exact_q16"

FLOAT_RESPONSE_STAGES = (
    STAGE_IDEAL_TONE_FLOAT,
    STAGE_ATSC_FLOAT,
    STAGE_INPUT_INT4,
    STAGE_WEIGHT_INT4,
    STAGE_JOINT_INT4_FLOAT,
    STAGE_FIXED_FLOAT_DECISION,
)
REPORT_STAGES = FLOAT_RESPONSE_STAGES + (STAGE_FIXED_Q16_CPU, STAGE_FULL_GPU)

STAGE_DEFINITIONS: dict[str, str] = {
    STAGE_IDEAL_TONE_FLOAT: (
        "Analytic coherent pilot tone with the measured 8-VSB pilot-line "
        "amplitude, unquantized detector input, ideal complex weights, and "
        "a complex128 padded FFT; an upper-bound signal model."
    ),
    STAGE_ATSC_FLOAT: (
        "GNU Radio ATSC 8-VSB waveform passed through the reference PFB, "
        "then unquantized detector input, ideal complex weights, and a "
        "complex128 padded FFT."
    ),
    STAGE_INPUT_INT4: (
        "The ATSC float stage with only the detector input quantized to "
        "signed 4+4 bit and dequantized before float matched filtering."
    ),
    STAGE_WEIGHT_INT4: (
        "The ATSC float stage with only the shipped signed 4+4-bit weight "
        "profile substituted for the ideal steering vectors."
    ),
    STAGE_JOINT_INT4_FLOAT: (
        "ATSC waveform with both input and weight quantization, but float "
        "matched filtering, transform, accumulation, and decision."
    ),
    STAGE_FIXED_FLOAT_DECISION: (
        "Packed ATSC input and shipped packed weights, exact integer matched "
        "filter, frozen fxfft256 transform, exact uint64 accumulation, and "
        "an unquantized float threshold multiplier."
    ),
    STAGE_FIXED_Q16_CPU: (
        "The fixed-transform stage with the null-calibrated threshold rounded "
        "up to Q16 and evaluated by exact cross products on the CPU."
    ),
    STAGE_FULL_GPU: (
        "Packed input and weights through the selected CUDA artifact. Kernel "
        "2.3+ runs fused fine powers and the device Q16 epilogue; kernel 2.1 "
        "runs its exact row-projection/fine-power chain and applies the same "
        "exact Q16 comparison on the host. The report labels the execution "
        "form, and both must be bit-identical to the CPU reference."
    ),
}


@dataclass(frozen=True)
class ExactResponseComponents:
    """Exact rational fields for ``max_designated_F2 / rank_F2``."""

    designated_num: int
    designated_den: int
    rank_num: int
    rank_den: int
    designated_bin: int
    rank_bin: int

    @property
    def valid(self) -> bool:
        return bool(
            self.designated_den > 0
            and self.designated_num >= 0
            and self.rank_num >= 0
            and self.rank_den > 0
        )

    def response_ratio(self) -> float:
        if not self.valid:
            return float("nan")
        if self.rank_num == 0:
            return float("inf") if self.designated_num > 0 else float("nan")
        return float(
            (self.designated_num * self.rank_den)
            / (self.designated_den * self.rank_num)
        )


def canonical_seed(base_seed: int, *coordinates: Any) -> int:
    """Return an order-independent uint64 seed from experiment coordinates.

    The caller must include every coordinate that should create an independent
    noise realization and omit SNR when common random numbers are required
    across an SNR sweep.  JSON canonicalization and SHA-256 make resumption or
    job-array reordering reproduce the same trial exactly.
    """
    payload = json.dumps(
        [int(base_seed), *coordinates],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def stage_seed(
    base_seed: int,
    *,
    purpose: str,
    physical_channel: int,
    offset_fine_bins: float,
    trial_index: int,
) -> int:
    """Common-random-number seed for one channel/offset/trial.

    SNR and ablation stage are deliberately absent: all SNR points and all
    stages see the same standardized noise draw.  ``purpose`` separates null,
    signal, PFB-gain, and bootstrap streams.
    """
    offset_microbins = int(round(float(offset_fine_bins) * 1_000_000.0))
    return canonical_seed(
        base_seed,
        str(purpose),
        int(physical_channel),
        offset_microbins,
        int(trial_index),
    )


def designated_bins(anchor_bin: int, half_width: int) -> np.ndarray:
    anchor = int(anchor_bin)
    width = int(half_width)
    if not 0 <= anchor < FINE_BINS:
        raise ValueError("anchor_bin must be in [0, 256).")
    if not 0 <= width < FINE_BINS // 2:
        raise ValueError("half_width must be in [0, 128).")
    return np.asarray(
        [(anchor + delta) % FINE_BINS for delta in range(-width, width + 1)],
        dtype=np.int64,
    )


def float_fine_power_ratio(
    detector_rows: np.ndarray,
    weights: np.ndarray,
    *,
    num_streams: int,
    windows_per_stream: int = 128,
    pad_factor: int = 2,
) -> np.ndarray:
    """Float reference from detector rows to the 256-bin fine statistic."""
    power = float_fine_powers_by_stream(
        detector_rows,
        weights,
        num_streams=num_streams,
        windows_per_stream=windows_per_stream,
        pad_factor=pad_factor,
    ).sum(axis=1)
    den = power[WEIGHT_TERM_REF_LOWER] + power[WEIGHT_TERM_REF_UPPER]
    return np.divide(
        2.0 * power[WEIGHT_TERM_TARGET],
        den,
        out=np.zeros_like(den, dtype=np.float64),
        where=den > 0,
    )


def float_fine_powers_by_stream(
    detector_rows: np.ndarray,
    weights: np.ndarray,
    *,
    num_streams: int,
    windows_per_stream: int = 128,
    pad_factor: int = 2,
) -> np.ndarray:
    """Return float fine powers before the sufficient sum over streams.

    The result has shape ``[3, num_streams, pad_factor * windows]``.  Keeping
    this additive boundary explicit permits an accelerated current-geometry
    simulator to model the distribution of the 2048-stream sum while a
    stratified audit still traverses the literal packed full-frame path.
    """
    rows = np.asarray(detector_rows)
    w = np.asarray(weights)
    streams = int(num_streams)
    windows = int(windows_per_stream)
    if rows.ndim != 2 or w.ndim != 2:
        raise ValueError("detector_rows and weights must both be 2D.")
    if w.shape[0] != 3 or rows.shape[1] != w.shape[1]:
        raise ValueError("weights must have shape [3, K] matching detector rows.")
    if rows.shape[0] != streams * windows:
        raise ValueError("row count must equal num_streams * windows_per_stream.")
    projection = rows.astype(np.complex128, copy=False) @ np.conjugate(
        w.astype(np.complex128, copy=False)
    ).T
    z = projection.T.reshape(3, streams, windows)
    spectrum = np.fft.fft(z, n=int(pad_factor) * windows, axis=-1)
    return np.asarray(
        spectrum.real * spectrum.real + spectrum.imag * spectrum.imag,
        dtype=np.float64,
    )


def float_response_ratio(
    fine_power_ratio: Sequence[float],
    *,
    designated: Sequence[int],
    bulk_mask: Sequence[bool],
    cfar_rank: int,
) -> float:
    """Return designated maximum divided by the selected null-bulk rank."""
    f2 = np.asarray(fine_power_ratio, dtype=np.float64)
    mask = np.asarray(bulk_mask, dtype=bool)
    designated_arr = np.asarray(list(designated), dtype=np.int64)
    if f2.shape != (FINE_BINS,) or mask.shape != (FINE_BINS,):
        raise ValueError("fine statistic and bulk mask must each have 256 bins.")
    bulk = f2[mask & np.isfinite(f2) & (f2 >= 0)]
    rank = int(cfar_rank)
    if rank < 0 or rank >= bulk.size:
        return float("nan")
    rank_value = float(np.partition(bulk, rank)[rank])
    designated_values = f2[designated_arr]
    if rank_value < 0.0 or not np.any(np.isfinite(designated_values)):
        return float("nan")
    if rank_value == 0.0:
        return (
            float("inf")
            if float(np.nanmax(designated_values)) > 0.0
            else float("nan")
        )
    return float(np.nanmax(designated_values) / rank_value)


def _rational_compare(
    num_a: int,
    den_a: int,
    num_b: int,
    den_b: int,
) -> int:
    left = int(num_a) * int(den_b)
    right = int(num_b) * int(den_a)
    return -1 if left < right else (1 if left > right else 0)


def exact_response_components(
    fine_powers: np.ndarray,
    *,
    designated: Sequence[int],
    bulk_mask: Sequence[bool],
    cfar_rank: int,
) -> ExactResponseComponents:
    """Select exact designated/rank rationals from uint64 fine powers."""
    powers = np.asarray(fine_powers)
    mask = np.asarray(bulk_mask, dtype=bool)
    if powers.shape != (3, FINE_BINS) or powers.dtype != np.dtype(np.uint64):
        raise ValueError("fine_powers must be exact uint64 with shape [3, 256].")
    if mask.shape != (FINE_BINS,):
        raise ValueError("bulk mask must have 256 entries.")
    nums = [2 * int(powers[WEIGHT_TERM_TARGET, b]) for b in range(FINE_BINS)]
    dens = [
        int(powers[WEIGHT_TERM_REF_LOWER, b])
        + int(powers[WEIGHT_TERM_REF_UPPER, b])
        for b in range(FINE_BINS)
    ]
    bulk = [b for b in range(FINE_BINS) if mask[b] and dens[b] > 0]
    rank = int(cfar_rank)
    if rank < 0 or rank >= len(bulk):
        return ExactResponseComponents(0, 0, 0, 0, -1, -1)

    def compare_bins(a: int, b: int) -> int:
        result = _rational_compare(nums[a], dens[a], nums[b], dens[b])
        return result if result else (-1 if a < b else (1 if a > b else 0))

    ranked = sorted(bulk, key=cmp_to_key(compare_bins))
    rank_bin = int(ranked[rank])
    valid_designated = [int(b) for b in designated if dens[int(b)] > 0]
    if not valid_designated:
        return ExactResponseComponents(0, 0, nums[rank_bin], dens[rank_bin], -1, rank_bin)

    designated_bin = valid_designated[0]
    for candidate in valid_designated[1:]:
        if _rational_compare(
            nums[designated_bin],
            dens[designated_bin],
            nums[candidate],
            dens[candidate],
        ) < 0:
            designated_bin = candidate
    return ExactResponseComponents(
        designated_num=nums[designated_bin],
        designated_den=dens[designated_bin],
        rank_num=nums[rank_bin],
        rank_den=dens[rank_bin],
        designated_bin=designated_bin,
        rank_bin=rank_bin,
    )


def order_statistic_threshold(
    null_response_ratios: Sequence[float],
    *,
    p_fa: float,
) -> dict[str, Any]:
    """Calibrate a conservative empirical threshold without interpolation.

    The selected threshold is an observed order statistic.  A detection uses
    strict ``>``; ties at the threshold therefore cannot increase the measured
    null exceedance.  Small null samples may not resolve the requested false
    alarm probability, which is recorded explicitly instead of extrapolated.
    """
    p = float(p_fa)
    if not 0.0 < p < 0.5:
        raise ValueError("p_fa must be in (0, 0.5).")
    values = np.asarray(null_response_ratios, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("null calibration has no finite response ratios.")
    values.sort()
    # The smallest zero-based order index whose empirical CDF is >= 1-Pfa.
    index = min(values.size - 1, max(0, math.ceil((1.0 - p) * values.size) - 1))
    threshold = float(values[index])
    exceed = int(np.count_nonzero(values > threshold))
    return {
        "threshold_multiplier": threshold,
        "order_index": int(index),
        "null_trials": int(values.size),
        "null_exceedances": exceed,
        "empirical_p_fa": float(exceed / values.size),
        "requested_p_fa": p,
        "minimum_resolvable_nonzero_p_fa": float(1.0 / values.size),
        "requested_p_fa_resolved": bool(values.size * p >= 1.0),
        "selection": "observed_order_statistic_strict_greater_than",
    }


def q16_ceil_multiplier(value: float) -> int:
    """Round a positive float threshold upward to the conservative Q16 grid."""
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold multiplier must be positive and finite.")
    q16 = int(math.ceil(threshold * Q16_ONE))
    if not 0 < q16 < (1 << 64):
        raise OverflowError("Q16 threshold does not fit uint64.")
    return q16


def exact_q16_decision(
    components: ExactResponseComponents,
    *,
    multiplier_q16: int,
) -> int:
    """Apply the deployed strict Q16 comparison with arbitrary-precision ints."""
    q16 = int(multiplier_q16)
    if not components.valid:
        return 0
    left = (
        int(components.designated_num)
        * Q16_ONE
        * int(components.rank_den)
    )
    right = (
        q16
        * int(components.rank_num)
        * int(components.designated_den)
    )
    return int(left > right)


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    n = int(trials)
    if n <= 0:
        return float("nan"), float("nan")
    k = int(successes)
    p = k / n
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    radius = float(z) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    radius /= denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def monotone_detection_rates(rates: Sequence[float]) -> np.ndarray:
    """Pool only downward empirical fluctuations for crossing interpolation."""
    values = np.asarray(rates, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rates must be one-dimensional.")
    # A cumulative maximum is intentionally simple and conservative about the
    # first observed crossing.  Raw rates and Wilson intervals remain in the
    # report; this adjusted sequence is used only for locating a bracket.
    return np.maximum.accumulate(values)


def crossing_bracket(
    snr_db: Sequence[float],
    rates: Sequence[float],
    *,
    target: float,
) -> dict[str, Any]:
    """Return an adjacent sampled bracket and an interpolated crossing.

    No extrapolation is performed.  When a target is not bracketed the result
    says so and the crossing estimate is ``None``.
    """
    x = np.asarray(snr_db, dtype=np.float64)
    y_raw = np.asarray(rates, dtype=np.float64)
    if x.ndim != 1 or y_raw.shape != x.shape or x.size == 0:
        raise ValueError("snr_db and rates must be non-empty equal 1D arrays.")
    order = np.argsort(x)
    x = x[order]
    y_raw = y_raw[order]
    y = monotone_detection_rates(y_raw)
    t = float(target)
    if not 0.0 < t < 1.0:
        raise ValueError("target must be in (0, 1).")
    for index in range(1, x.size):
        if y[index - 1] < t <= y[index]:
            lo_x, hi_x = float(x[index - 1]), float(x[index])
            lo_y, hi_y = float(y[index - 1]), float(y[index])
            estimate = (
                lo_x
                if hi_y == lo_y
                else lo_x + (hi_x - lo_x) * (t - lo_y) / (hi_y - lo_y)
            )
            return {
                "bracketed": True,
                "target_probability": t,
                "snr_lo_db": lo_x,
                "snr_hi_db": hi_x,
                "pd_lo": lo_y,
                "pd_hi": hi_y,
                "estimate_db": float(estimate),
                "method": "linear_within_adjacent_sampled_monotone_bracket",
                "raw_monotone_adjustment_max": float(np.max(y - y_raw)),
            }
    return {
        "bracketed": False,
        "target_probability": t,
        "snr_lo_db": None,
        "snr_hi_db": None,
        "pd_lo": None,
        "pd_hi": None,
        "estimate_db": None,
        "method": "no_extrapolation_unbracketed",
        "observed_pd_min": float(np.min(y)),
        "observed_pd_max": float(np.max(y)),
        "observed_snr_min_db": float(np.min(x)),
        "observed_snr_max_db": float(np.max(x)),
        "raw_monotone_adjustment_max": float(np.max(y - y_raw)),
    }


def _percentile_interval(values: Sequence[float]) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    lo, hi = np.quantile(finite, (0.025, 0.975))
    return float(lo), float(hi)


def paired_crossing_bootstrap(
    *,
    snr_db: Sequence[float],
    float_null_ratios: Sequence[float],
    fixed_null_ratios: Sequence[float],
    null_trial_keys: Sequence[Any],
    float_h1_ratios_by_snr: Mapping[float, Sequence[float]],
    fixed_h1_components_by_snr: Mapping[float, Sequence[ExactResponseComponents]],
    trial_keys_by_snr: Mapping[float, Sequence[Any]],
    p_fa: float,
    target_pd: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired null+signal bootstrap of fixed-minus-float crossing loss.

    Null identities are shared across stages. H1 identities are intersected
    across the complete SNR grid, and one resample of those identities is
    shared across SNR and stages. A replicate is retained only when both
    sampled curves bracket ``target_pd``; the valid count is reported so a
    sparse sweep cannot silently acquire an interval by extrapolation.
    """
    x = [float(value) for value in snr_db]
    float_null = np.asarray(float_null_ratios, dtype=np.float64)
    fixed_null = np.asarray(fixed_null_ratios, dtype=np.float64)
    if float_null.shape != fixed_null.shape or float_null.ndim != 1:
        raise ValueError("paired null stage arrays must have identical 1D shape.")
    if float_null.size == 0:
        raise ValueError("paired bootstrap requires null trials.")
    null_keys = list(null_trial_keys)
    if len(null_keys) != float_null.size:
        raise ValueError("null trial identities disagree with paired null arrays.")
    if len(set(null_keys)) != len(null_keys):
        raise ValueError("duplicate null trial identities are not allowed.")
    float_h1 = {
        s: np.asarray(float_h1_ratios_by_snr[s], dtype=np.float64) for s in x
    }
    fixed_h1 = {s: list(fixed_h1_components_by_snr[s]) for s in x}
    for s in x:
        if float_h1[s].ndim != 1 or len(fixed_h1[s]) != float_h1[s].size:
            raise ValueError(f"paired H1 stage arrays disagree at SNR {s}.")
    key_lists = {s: list(trial_keys_by_snr[s]) for s in x}
    for s in x:
        if len(key_lists[s]) != float_h1[s].size:
            raise ValueError(f"trial identities disagree with H1 arrays at SNR {s}.")
        if len(set(key_lists[s])) != len(key_lists[s]):
            raise ValueError(f"duplicate trial identities at SNR {s}.")
    common = set(key_lists[x[0]])
    for s in x[1:]:
        common.intersection_update(key_lists[s])
    if not common:
        raise ValueError("no common H1 trial identities span the SNR grid.")
    common_order = sorted(common, key=repr)
    for s in x:
        lookup = {key: index for index, key in enumerate(key_lists[s])}
        indices = np.asarray([lookup[key] for key in common_order], dtype=np.int64)
        float_h1[s] = float_h1[s][indices]
        fixed_h1[s] = [fixed_h1[s][int(index)] for index in indices]
    sizes = {float_h1[s].size for s in x}
    shared_h1_size = next(iter(sizes)) if len(sizes) == 1 else None
    rng = np.random.default_rng(int(seed))
    float_crossings: list[float] = []
    fixed_crossings: list[float] = []
    losses: list[float] = []
    for _ in range(int(replicates)):
        null_index = rng.integers(0, float_null.size, size=float_null.size)
        float_threshold = order_statistic_threshold(
            float_null[null_index], p_fa=float(p_fa)
        )["threshold_multiplier"]
        fixed_threshold = order_statistic_threshold(
            fixed_null[null_index], p_fa=float(p_fa)
        )["threshold_multiplier"]
        fixed_q16 = q16_ceil_multiplier(float(fixed_threshold))
        float_rates: list[float] = []
        fixed_rates: list[float] = []
        shared_index = (
            None
            if shared_h1_size is None
            else rng.integers(0, shared_h1_size, size=shared_h1_size)
        )
        for s in x:
            n = float_h1[s].size
            if n == 0:
                float_rates.append(float("nan"))
                fixed_rates.append(float("nan"))
                continue
            index = (
                shared_index
                if shared_index is not None
                else rng.integers(0, n, size=n)
            )
            float_rates.append(
                float(np.mean(float_h1[s][index] > float(float_threshold)))
            )
            fixed_rates.append(
                float(
                    np.mean(
                        [
                            exact_q16_decision(
                                fixed_h1[s][int(i)], multiplier_q16=fixed_q16
                            )
                            for i in index
                        ]
                    )
                )
            )
        if not np.all(np.isfinite(float_rates)) or not np.all(
            np.isfinite(fixed_rates)
        ):
            continue
        float_cross = crossing_bracket(x, float_rates, target=float(target_pd))
        fixed_cross = crossing_bracket(x, fixed_rates, target=float(target_pd))
        if not float_cross["bracketed"] or not fixed_cross["bracketed"]:
            continue
        f = float(float_cross["estimate_db"])
        q = float(fixed_cross["estimate_db"])
        float_crossings.append(f)
        fixed_crossings.append(q)
        losses.append(q - f)

    float_lo, float_hi = _percentile_interval(float_crossings)
    fixed_lo, fixed_hi = _percentile_interval(fixed_crossings)
    loss_lo, loss_hi = _percentile_interval(losses)
    valid = len(losses)
    return {
        "target_probability": float(target_pd),
        "bootstrap_seed": int(seed),
        "requested_replicates": int(replicates),
        "valid_bracketed_replicates": valid,
        "valid_fraction": float(valid / int(replicates)) if replicates else 0.0,
        "float_crossing_median_db": (
            None if not float_crossings else float(np.median(float_crossings))
        ),
        "float_crossing_bootstrap95_lo_db": float_lo,
        "float_crossing_bootstrap95_hi_db": float_hi,
        "fixed_q16_crossing_median_db": (
            None if not fixed_crossings else float(np.median(fixed_crossings))
        ),
        "fixed_q16_crossing_bootstrap95_lo_db": fixed_lo,
        "fixed_q16_crossing_bootstrap95_hi_db": fixed_hi,
        "fixed_minus_float_sensitivity_loss_median_db": (
            None if not losses else float(np.median(losses))
        ),
        "fixed_minus_float_sensitivity_loss_bootstrap95_lo_db": loss_lo,
        "fixed_minus_float_sensitivity_loss_bootstrap95_hi_db": loss_hi,
        "interpretation": (
            "Positive loss means the fixed Q16 pipeline needs a larger input "
            "SNR than the 8-VSB ideal-float detector at the same target Pd."
        ),
        "uncertainty_scope": (
            "Paired nonparametric resampling of both null calibration trials "
            "and H1 trials, conditional on the simulated signal/noise model."
        ),
        "null_pairing_mode": "explicit_trial_identities",
        "paired_null_trials": int(float_null.size),
        "h1_pairing_mode": (
            "intersection_of_explicit_trial_identities_across_all_snrs"
        ),
        "paired_h1_trials_per_snr": (
            None if shared_h1_size is None else int(shared_h1_size)
        ),
    }


def components_from_columns(
    columns: Mapping[str, Sequence[int]],
) -> list[ExactResponseComponents]:
    required = (
        "designated_num",
        "designated_den",
        "rank_num",
        "rank_den",
        "designated_bin",
        "rank_bin",
    )
    missing = [key for key in required if key not in columns]
    if missing:
        raise KeyError(f"missing exact response columns: {missing}")
    lengths = {len(columns[key]) for key in required}
    if len(lengths) != 1:
        raise ValueError("exact response columns must have equal lengths.")
    return [
        ExactResponseComponents(
            designated_num=int(columns["designated_num"][i]),
            designated_den=int(columns["designated_den"][i]),
            rank_num=int(columns["rank_num"][i]),
            rank_den=int(columns["rank_den"][i]),
            designated_bin=int(columns["designated_bin"][i]),
            rank_bin=int(columns["rank_bin"][i]),
        )
        for i in range(next(iter(lengths), 0))
    ]


def exact_columns(components: Iterable[ExactResponseComponents]) -> dict[str, np.ndarray]:
    values = list(components)
    return {
        "designated_num": np.asarray(
            [item.designated_num for item in values], dtype=np.uint64
        ),
        "designated_den": np.asarray(
            [item.designated_den for item in values], dtype=np.uint64
        ),
        "rank_num": np.asarray([item.rank_num for item in values], dtype=np.uint64),
        "rank_den": np.asarray([item.rank_den for item in values], dtype=np.uint64),
        "designated_bin": np.asarray(
            [item.designated_bin for item in values], dtype=np.int16
        ),
        "rank_bin": np.asarray([item.rank_bin for item in values], dtype=np.int16),
    }


__all__ = [
    "ExactResponseComponents",
    "FLOAT_RESPONSE_STAGES",
    "REPORT_STAGES",
    "SENSITIVITY_STUDY_SCHEMA",
    "STAGE_ATSC_FLOAT",
    "STAGE_DEFINITIONS",
    "STAGE_FIXED_FLOAT_DECISION",
    "STAGE_FIXED_Q16_CPU",
    "STAGE_FULL_GPU",
    "STAGE_IDEAL_TONE_FLOAT",
    "STAGE_INPUT_INT4",
    "STAGE_JOINT_INT4_FLOAT",
    "STAGE_WEIGHT_INT4",
    "canonical_seed",
    "components_from_columns",
    "crossing_bracket",
    "designated_bins",
    "exact_columns",
    "exact_q16_decision",
    "exact_response_components",
    "float_fine_power_ratio",
    "float_fine_powers_by_stream",
    "float_response_ratio",
    "monotone_detection_rates",
    "order_statistic_threshold",
    "paired_crossing_bootstrap",
    "q16_ceil_multiplier",
    "stage_seed",
    "wilson_interval",
]
