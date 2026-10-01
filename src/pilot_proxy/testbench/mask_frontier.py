# coding=utf-8
"""Mask-versus-residual frontier arithmetic for the synthetic bench.

The command-line driver lives in ``tools/mask_residual_frontier.py``.  This
module holds the small, testable pieces: the archive's exact per-rank Q16
decision boundary, its keep rule, its empirical candidate staircase, the
shelf-or-floor per-frame residual convention, the residual-score histogram of
chapter 6 eq:detection:histogram, and the prefix-sum walk that turns that
histogram into the frontier of eq:detection:frontier.

Two quantities are carried through every bin of the histogram side by side.
``systematic`` is what the archive books for a frame: ``10^(shelf/10)`` when
the frame has a finite shelf estimate, the channel floor otherwise, and never
below the floor.  ``injected`` is what the synthetic bench actually put into
that frame, which the archive can never know.  Both are prefix-summed over the
same bins, so the frontier reports the claim and the truth on identical kept
sets and any gap between them belongs to the residual convention rather than
to the bookkeeping.

The rules here are transcriptions of the deployed selector, not of a
convenient approximation of it:

* the per-rank boundary is ``ceil(max_D F2 * 2^16 / F2_rho)`` on exact
  rationals, with ``ALWAYS_MASKED_Q16`` for a rank whose reference numerator
  vanishes (``pilot_proxy.detectors.narrowband_marker.scores.required_eta_q16_by_rank``);
* a frame is kept when ``required <= eta`` and it is not always-masked
  (``pilot_proxy.characterization.surface.kept_at``);
* the candidate staircase is the sorted distinct deployable requirements plus
  the policy floor of one (``pilot_proxy.detectors.narrowband_marker.scores.candidate_eta_q16``);
* a candidate retaining fewer than ``MIN_RETAINED_FRAMES`` frames is not
  evaluated (``pilot_proxy.characterization.surface.operating_surface``);
* the floor is the 90th percentile of the shelf estimates of a verified off
  population (``pilot_proxy.characterization.nulls.floor_estimate``).
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from functools import cmp_to_key
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MASK_FRONTIER_SCHEMA = "pilotproxy_mask_residual_frontier_v1"

FINE_BINS = 256
Q16_FRACTION_BITS = 16
Q16_SCALE = 1 << Q16_FRACTION_BITS
MAX_MULTIPLIER_Q16 = (1 << 64) - 1
ALWAYS_MASKED_Q16 = 1 << 64
MINIMUM_CANDIDATE_Q16 = 1
MIN_RETAINED_FRAMES = 30
FLOOR_PERCENTILE = 90.0
FLOOR_MIN_FRAMES = 30


def _rational_order(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    left_num, left_den, left_bin = left
    right_num, right_den, right_bin = right
    first = left_num * right_den
    second = right_num * left_den
    if first < second:
        return -1
    if first > second:
        return 1
    return left_bin - right_bin


def required_multipliers_by_rank(
    fine_powers: np.ndarray,
    *,
    designated: Sequence[int],
    bulk_mask: Sequence[bool],
) -> tuple[int, ...]:
    """Return the exact one-based-rank Q16 boundaries of one frame.

    Rank ``rho`` occupies index ``rho - 1``.  The rule is the deployed one:
    ``F2 = 2 p_target / (p_lower + p_upper)`` as an exact rational per bin, the
    declared bulk ordered by that rational with the bin index breaking ties,
    and the boundary rounded upward so that the integer comparison keeps the
    frame at exactly this multiplier and no smaller one.
    """
    powers = np.asarray(fine_powers)
    if powers.shape != (3, FINE_BINS) or powers.dtype != np.dtype(np.uint64):
        raise ValueError("fine_powers must be exact uint64 with shape [3, 256].")
    mask = np.asarray(bulk_mask, dtype=bool)
    if mask.shape != (FINE_BINS,):
        raise ValueError("bulk_mask must have 256 entries.")
    designated_bins = [int(value) for value in designated]
    if not designated_bins:
        raise ValueError("the designated set must not be empty.")
    if any(not 0 <= value < FINE_BINS for value in designated_bins):
        raise ValueError("designated bins must lie in [0, 256).")
    if mask[designated_bins].any():
        raise ValueError("the bulk mask must exclude every designated bin.")

    numerator = [2 * int(value) for value in powers[0]]
    denominator = [
        int(lower) + int(upper) for lower, upper in zip(powers[1], powers[2])
    ]
    ranked = [
        (numerator[index], denominator[index], index)
        for index in np.flatnonzero(mask)
        if denominator[index] > 0
    ]
    if not ranked:
        raise ValueError("the declared bulk has no bin with a positive denominator.")
    ranked.sort(key=cmp_to_key(_rational_order))

    designated_maximum: tuple[int, int] | None = None
    for index in designated_bins:
        if denominator[index] <= 0 or numerator[index] <= 0:
            continue
        value = (numerator[index], denominator[index])
        if (
            designated_maximum is None
            or value[0] * designated_maximum[1] > designated_maximum[0] * value[1]
        ):
            designated_maximum = value

    boundaries: list[int] = []
    for rank_num, rank_den, _ in ranked:
        if designated_maximum is None:
            boundaries.append(1)
            continue
        if rank_num == 0:
            boundaries.append(ALWAYS_MASKED_Q16)
            continue
        designated_num, designated_den = designated_maximum
        numerator_product = designated_num * Q16_SCALE * rank_den
        denominator_product = rank_num * designated_den
        boundary = (
            numerator_product + denominator_product - 1
        ) // denominator_product
        boundaries.append(
            ALWAYS_MASKED_Q16
            if boundary > MAX_MULTIPLIER_Q16
            else max(1, boundary)
        )
    return tuple(boundaries)


def kept_at(required_q16: np.ndarray, eta_q16: int) -> np.ndarray:
    """A frame is kept when the selected multiplier is at least its requirement."""
    required = np.asarray(required_q16, dtype=object)
    eta = int(eta_q16)
    return np.asarray(
        [int(value) <= eta and int(value) != ALWAYS_MASKED_Q16 for value in required],
        dtype=bool,
    )


def candidate_multipliers_q16(required_q16: Iterable[int]) -> tuple[int, ...]:
    """Return the exact deployable empirical decision staircase."""
    candidates = {MINIMUM_CANDIDATE_Q16}
    empty = True
    for raw in required_q16:
        empty = False
        value = int(raw)
        if not 1 <= value <= ALWAYS_MASKED_Q16:
            raise ValueError("a decision boundary is outside the deployable range.")
        if value <= MAX_MULTIPLIER_Q16:
            candidates.add(value)
    if empty:
        raise ValueError("required_q16 must not be empty.")
    return tuple(sorted(candidates))


def coarse_normalized_excess(
    p_target: int,
    p_ref_lower: int,
    p_ref_upper: int,
    *,
    target_norm_sq: int,
    reference_norm_sum_sq: int,
) -> float:
    """Return ``p_target * ref_norm / (p_ref_sum * target_norm) - 1`` exactly.

    This is the archive's per-frame one-bin normalized pilot excess, formed
    from the same exact uint64 marginals and the same exact integer weight
    norms the deployed product carries.
    """
    numerator = int(p_target) * int(reference_norm_sum_sq)
    denominator = (int(p_ref_lower) + int(p_ref_upper)) * int(target_norm_sq)
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator) - 1.0


def shelf_db_from_excess(excess: float, *, offset_db: float) -> float:
    """Normalized pilot excess to data-shelf SNR, NaN for a non-positive excess."""
    value = float(excess)
    if not math.isfinite(value) or value <= 0.0:
        return float("nan")
    return 10.0 * math.log10(value) + float(offset_db)


def measured_floor_db(
    shelf_db: Sequence[float],
    *,
    percentile: float = FLOOR_PERCENTILE,
    minimum_frames: int = FLOOR_MIN_FRAMES,
) -> dict[str, Any]:
    """The off-population floor: a percentile of its finite shelf estimates."""
    values = np.asarray(shelf_db, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size < int(minimum_frames):
        return {
            "floor_db": float("nan"),
            "evidence": "refused",
            "frames": int(values.size),
            "frames_with_shelf_estimate": int(finite.size),
            "percentile": float(percentile),
            "minimum_frames": int(minimum_frames),
        }
    return {
        "floor_db": float(np.percentile(finite, float(percentile))),
        "evidence": "measured",
        "frames": int(values.size),
        "frames_with_shelf_estimate": int(finite.size),
        "percentile": float(percentile),
        "minimum_frames": int(minimum_frames),
    }


def systematic_residuals(
    shelf_db: Sequence[float], *, floor_linear: float
) -> np.ndarray:
    """The shelf-or-floor linear convention, one value per frame.

    A frame with a finite shelf estimate carries ``max(10^(shelf/10), floor)``;
    a frame without one carries the floor.  No frame falls below the floor.
    """
    floor = float(floor_linear)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("the floor must be a finite non-negative linear residual.")
    values = np.asarray(shelf_db, dtype=np.float64)
    out = np.full(values.shape, floor, dtype=np.float64)
    finite = np.isfinite(values)
    out[finite] = np.maximum(10.0 ** (values[finite] / 10.0), floor)
    return out


@dataclass(frozen=True)
class ResidualScoreHistogram:
    """One rank's histogram, with the injected truth carried beside the claim."""

    rho: int
    bulk_size: int
    candidate_multiplier_q16: tuple[int, ...]
    counts: tuple[int, ...]
    systematic_sums: tuple[float, ...]
    injected_sums: tuple[float, ...]

    @property
    def frame_count(self) -> int:
        return int(sum(self.counts))


def residual_score_histogram(
    required_q16: Sequence[int],
    systematic: Sequence[float],
    injected: Sequence[float],
    *,
    candidates: Sequence[int],
    rho: int,
    bulk_size: int,
) -> ResidualScoreHistogram:
    """Bin exact per-frame Q16 boundaries and total both residuals per bin."""
    grid = [int(value) for value in candidates]
    if not grid:
        raise ValueError("the candidate grid must not be empty.")
    if any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("the candidate grid must be strictly increasing.")
    requirements = [int(value) for value in required_q16]
    if not requirements:
        raise ValueError("required_q16 must not be empty.")
    if any(not 1 <= value <= ALWAYS_MASKED_Q16 for value in requirements):
        raise ValueError("a decision boundary is outside the deployable range.")
    claim = np.asarray(systematic, dtype=np.float64)
    truth = np.asarray(injected, dtype=np.float64)
    for values, name in ((claim, "systematic"), (truth, "injected")):
        if values.shape != (len(requirements),):
            raise ValueError(f"{name} must carry one value per frame.")
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{name} must be finite and non-negative.")
    bins = np.fromiter(
        (bisect_left(grid, value) for value in requirements),
        dtype=np.int64,
        count=len(requirements),
    )
    size = len(grid) + 1
    counts = np.bincount(bins, minlength=size)
    claim_sums = np.bincount(bins, weights=claim, minlength=size)
    truth_sums = np.bincount(bins, weights=truth, minlength=size)
    return ResidualScoreHistogram(
        rho=int(rho),
        bulk_size=int(bulk_size),
        candidate_multiplier_q16=tuple(grid),
        counts=tuple(int(value) for value in counts),
        systematic_sums=tuple(float(value) for value in claim_sums),
        injected_sums=tuple(float(value) for value in truth_sums),
    )


def frontier_points(
    histogram: ResidualScoreHistogram,
    *,
    minimum_retained: int = MIN_RETAINED_FRAMES,
) -> list[dict[str, Any]]:
    """Walk the histogram's prefix sums into unsmoothed frontier points.

    ``r_sys`` is what the cumulative sums claim survives in the kept frames;
    ``r_injected`` is the mean of what the bench actually injected into the
    same kept frames.  Both come from the same prefix sums, so they describe
    identical kept sets.
    """
    frames = histogram.frame_count
    if frames <= 0:
        raise ValueError("the histogram must describe at least one frame.")
    kept = 0
    claim_sum = 0.0
    truth_sum = 0.0
    points: list[dict[str, Any]] = []
    for index, multiplier in enumerate(histogram.candidate_multiplier_q16):
        kept += histogram.counts[index]
        claim_sum += histogram.systematic_sums[index]
        truth_sum += histogram.injected_sums[index]
        if kept < int(minimum_retained):
            continue
        masked_fraction = (frames - kept) / frames
        points.append(
            {
                "rho": histogram.rho,
                "eta_q16": int(multiplier),
                "eta": int(multiplier) / Q16_SCALE,
                "frames": frames,
                "kept": kept,
                "masked_fraction": masked_fraction,
                "r_sys": claim_sum / kept,
                "r_injected": truth_sum / kept,
            }
        )
    return points


def minimum_mask_envelope(
    points: Sequence[Mapping[str, Any]], tolerance: float, *, key: str = "r_sys"
) -> dict[str, Any] | None:
    """Eq:detection:frontier: the least mask whose residual is within tolerance."""
    tau = float(tolerance)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("the residual allowance must be positive and finite.")
    feasible = [point for point in points if float(point[key]) <= tau]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda point: (
            float(point["masked_fraction"]),
            float(point[key]),
            int(point["rho"]),
            int(point["eta_q16"]),
        ),
    )


def bootstrap_frontier(
    required_q16: Sequence[int],
    systematic: Sequence[float],
    injected: Sequence[float],
    *,
    candidates: Sequence[int],
    rho: int,
    bulk_size: int,
    replicates: int,
    seed: int,
    minimum_retained: int = MIN_RETAINED_FRAMES,
    quantiles: Sequence[float] = (0.16, 0.84),
) -> dict[int, dict[str, tuple[float, float]]]:
    """Resample frames with replacement at a fixed candidate grid.

    The grid is held at the value the full sample produced, so every replicate
    reports the same candidates and the interval describes trial noise rather
    than a moving staircase.  Candidates that fall below the retained-frame
    floor in a replicate contribute nothing to that candidate's interval.
    """
    grid = [int(value) for value in candidates]
    boundaries = [int(value) for value in required_q16]
    requirements = np.fromiter(
        (bisect_left(grid, value) for value in boundaries),
        dtype=np.int64,
        count=len(boundaries),
    )
    claim = np.asarray(systematic, dtype=np.float64)
    truth = np.asarray(injected, dtype=np.float64)
    frames = requirements.size
    size = len(grid) + 1
    rng = np.random.default_rng(int(seed))
    replicate_count = int(replicates)
    if replicate_count <= 0:
        raise ValueError("replicates must be positive.")
    masked = np.empty((replicate_count, len(grid)), dtype=np.float64)
    claimed = np.empty((replicate_count, len(grid)), dtype=np.float64)
    injected_mean = np.empty((replicate_count, len(grid)), dtype=np.float64)
    for replicate in range(replicate_count):
        draw = rng.integers(0, frames, size=frames)
        bins = requirements[draw]
        counts = np.bincount(bins, minlength=size)[: len(grid)]
        claim_sums = np.bincount(bins, weights=claim[draw], minlength=size)[: len(grid)]
        truth_sums = np.bincount(bins, weights=truth[draw], minlength=size)[: len(grid)]
        kept = np.cumsum(counts)
        with np.errstate(invalid="ignore", divide="ignore"):
            masked[replicate] = (frames - kept) / frames
            claimed[replicate] = np.cumsum(claim_sums) / kept
            injected_mean[replicate] = np.cumsum(truth_sums) / kept
        below = kept < int(minimum_retained)
        masked[replicate][below] = np.nan
        claimed[replicate][below] = np.nan
        injected_mean[replicate][below] = np.nan
    low, high = float(quantiles[0]), float(quantiles[1])
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for index, multiplier in enumerate(grid):
        entry: dict[str, tuple[float, float]] = {}
        for name, values in (
            ("masked_fraction", masked[:, index]),
            ("r_sys", claimed[:, index]),
            ("r_injected", injected_mean[:, index]),
        ):
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                entry[name] = (float("nan"), float("nan"))
                continue
            entry[name] = (
                float(np.quantile(finite, low)),
                float(np.quantile(finite, high)),
            )
        entry["replicates"] = replicate_count
        out[int(multiplier)] = entry
    return out


def kept_frame_means(
    required_q16: Sequence[int],
    values: Sequence[float],
    *,
    eta_q16: int,
) -> dict[str, Any]:
    """The direct kept-frame mean, for auditing the prefix-sum bookkeeping."""
    keep = kept_at(np.asarray(list(required_q16), dtype=object), int(eta_q16))
    data = np.asarray(values, dtype=np.float64)
    if data.shape != keep.shape:
        raise ValueError("values must carry one entry per frame.")
    kept = int(np.count_nonzero(keep))
    return {
        "kept": kept,
        "mean": float(data[keep].mean()) if kept else float("nan"),
    }
