"""Within-era stability of a candidate family: era-half support, the drift
screen, block resampling, and the prepared family itself.

:func:`prepare_family` takes the exact per-frame Q16 keep boundaries of every
rank (the detector's scores), the per-frame residuals and the frames' times
and acquisitions; it derives the candidate grids, the calendar split of the
era at its midpoint, the support of each half, the pooled and half histograms,
and the drift evidence (:func:`assess_histogram_stability`). A family whose
screen does not pass is refused for selection (:class:`StabilityRefused`, a
*detector refusal*); the surface itself is still reported. The screen is
independent of any science tolerance.

:func:`drift_diagnostic` records why the screen refused, measured on the
candidates a selector could choose.

Register values (``era.*``, ``preparation.*``, ``stability.*``) are the
project's detector register; a prepared family records the register's
canonical JSON and digest.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
import datetime as dt
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Sequence

import numpy as np

from pilot_proxy.config.project import default_project
from pilot_proxy.detectors.narrowband_marker.scores import candidate_eta_q16, candidate_rho_values

from .surface import (ALWAYS_MASKED_Q16, MIN_RETAINED_FRAMES, CalibrationEvidence, ScoreHistogram,
                      build_score_histogram)

_REGISTER = default_project().register
# the detector register's operational entries (a claim above screening needs every one resolved)
OPERATIONAL_PREFIXES = ("era.", "floor.", "correlation.", "preparation.", "stability.", "selection.", "detection.")
OPERATIONAL_REQUIRED_IDS = tuple(sorted(
    i for i in _REGISTER.ids()
    if i.startswith(OPERATIONAL_PREFIXES) and _REGISTER.decision(i).status != "historical"))


def _blockers(ids) -> tuple:
    """Register decisions that cannot support an operational claim."""
    return tuple(_REGISTER.decision(i) for i in sorted(ids)
                 if _REGISTER.decision(i).status not in {"derived", "locked"})


class StabilityRefused(RuntimeError):
    """Raised when a prepared family lacks required evidence."""


@dataclass(frozen=True)
class EraHalfSupport:
    """Independent support recorded for one calendar half of an era."""

    frame_count: int
    acquisition_count: int
    observed_months: int
    span_days: float

    def __post_init__(self):
        for name in ("frame_count", "acquisition_count", "observed_months"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))
        if (isinstance(self.span_days, bool)
                or not isinstance(self.span_days, Real)):
            raise TypeError("span_days must be a number")
        span = float(self.span_days)
        if not math.isfinite(span) or span < 0.0:
            raise ValueError("span_days must be non-negative and finite")
        object.__setattr__(self, "span_days", span)


@dataclass(frozen=True)
class BlockResamplingPlan:
    """Caller-declared block resampling controls."""

    block_unit: str
    seed: int
    replicates: int
    interval_coverage: float
    minimum_blocks_per_half: int

    def __post_init__(self):
        if self.block_unit not in {"acquisition", "sidereal_day"}:
            raise ValueError(
                "block_unit must be 'acquisition' or 'sidereal_day'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if (isinstance(self.replicates, bool)
                or not isinstance(self.replicates, Integral)):
            raise TypeError("replicates must be an integer")
        if int(self.replicates) <= 0:
            raise ValueError("replicates must be positive")
        if (isinstance(self.minimum_blocks_per_half, bool)
                or not isinstance(self.minimum_blocks_per_half, Integral)):
            raise TypeError("minimum_blocks_per_half must be an integer")
        if int(self.minimum_blocks_per_half) < 2:
            raise ValueError("minimum_blocks_per_half must be at least two")
        if (isinstance(self.interval_coverage, bool)
                or not isinstance(self.interval_coverage, Real)):
            raise TypeError("interval_coverage must be a number")
        coverage = float(self.interval_coverage)
        if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
            raise ValueError("interval_coverage must be finite and in (0, 1)")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "replicates", int(self.replicates))
        object.__setattr__(self, "minimum_blocks_per_half",
                           int(self.minimum_blocks_per_half))
        object.__setattr__(self, "interval_coverage", coverage)
        if self.replicates < self.minimum_tail_replicates:
            raise ValueError(
                "replicates do not resolve the requested interval coverage")

    @property
    def minimum_tail_replicates(self) -> int:
        tail_resolution = math.ceil(
            1.0 / (1.0 - self.interval_coverage) - 1e-12)
        return max(2, int(tail_resolution))

    @property
    def minimum_successful_replicates(self) -> int:
        return int(math.ceil(self.interval_coverage * self.replicates))


@dataclass(frozen=True)
class BlockStabilityAssessment:
    """Surface-wide upper bounds from stratified whole-block resampling."""

    status: str
    reason: str
    method: str
    block_unit: str
    seed: int
    replicates_requested: int
    replicates_succeeded: int
    minimum_successful_replicates: int
    interval_coverage: float
    minimum_blocks_per_half: int
    early_blocks: int
    late_blocks: int
    points_checked: int
    maximum_cost_ratio_upper_bound: float | None
    maximum_systematic_residual_ratio_upper_bound: float | None

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True)
class StabilityAssessment:
    """Deterministic early/late drift screen for a candidate surface."""

    status: str
    reason: str
    early_support: EraHalfSupport
    late_support: EraHalfSupport
    points_checked: int
    points_skipped: int
    maximum_cost_ratio: float | None
    maximum_systematic_residual_ratio: float | None
    maximum_masked_fraction_difference: float | None
    worst_cost_point: tuple[int, float] | None
    worst_systematic_point: tuple[int, float] | None
    worst_cost_multiplier_q16: int | None
    worst_systematic_multiplier_q16: int | None
    cost_ratio_limit: float | None
    systematic_residual_ratio_limit: float | None
    minimum_observed_months: int
    minimum_span_days: float
    minimum_half_retained_frames: int | None
    block_uncertainty: BlockStabilityAssessment | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"




def _ratio(left: float, right: float) -> float:
    if left == 0.0 and right == 0.0:
        return 1.0
    if left <= 0.0 or right <= 0.0:
        return math.inf
    return max(left, right) / min(left, right)


def _limit(value, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number or None")
    result = float(value)
    if not math.isfinite(result) or result < 1.0:
        raise ValueError(f"{name} must be finite and at least one")
    return result


def _ranked(histograms, name: str) -> dict[int, ScoreHistogram]:
    if not isinstance(histograms, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if not histograms:
        raise ValueError(f"{name} must not be empty")
    out = {}
    for rho, histogram in histograms.items():
        if isinstance(rho, bool) or not isinstance(rho, Integral):
            raise TypeError(f"{name} keys must be integers")
        if int(rho) <= 0:
            raise ValueError(f"{name} keys must be positive")
        if not isinstance(histogram, ScoreHistogram):
            raise TypeError(f"{name} values must be ScoreHistogram")
        out[int(rho)] = histogram
    return out


def _surface(histogram: ScoreHistogram):
    has_variance = histogram.variance_residual_sums is not None
    kept = 0
    systematic = 0.0
    variance = 0.0
    out = []
    total = histogram.frame_count
    for index, eta in enumerate(histogram.candidate_eta):
        kept += histogram.counts[index]
        systematic += histogram.systematic_residual_sums[index]
        if has_variance:
            variance += histogram.variance_residual_sums[index]
        if kept == 0:
            out.append(None)
            continue
        kept_fraction = kept / total
        systematic_mean = systematic / kept
        variance_mean = variance / kept if has_variance else None
        cost = ((1.0 + (0.0 if variance_mean is None else variance_mean))
                / kept_fraction)
        out.append((kept, kept_fraction, systematic_mean, variance_mean, cost))
    return tuple(out)


def assess_histogram_stability(
        early_histograms: Mapping[int, ScoreHistogram],
        late_histograms: Mapping[int, ScoreHistogram],
        *,
        early_support: EraHalfSupport,
        late_support: EraHalfSupport,
        max_cost_ratio: float | None,
        max_systematic_residual_ratio: float | None,
        minimum_observed_months: int | None = None,
        minimum_span_days: float | None = None,
        minimum_half_retained_frames: int | None = None,
) -> StabilityAssessment:
    """Screen early/late point estimates using one frozen calibration.

    The screen is independent of science tolerance. It is not a statistical
    equivalence test; operational use also needs block-based uncertainty.
    """
    early = _ranked(early_histograms, "early_histograms")
    late = _ranked(late_histograms, "late_histograms")
    if set(early) != set(late):
        raise ValueError("early and late histograms must use the same ranks")
    if not isinstance(early_support, EraHalfSupport):
        raise TypeError("early_support must be EraHalfSupport")
    if not isinstance(late_support, EraHalfSupport):
        raise TypeError("late_support must be EraHalfSupport")
    if any(hist.frame_count != early_support.frame_count
           for hist in early.values()):
        raise ValueError("early support frame count does not match histograms")
    if any(hist.frame_count != late_support.frame_count
           for hist in late.values()):
        raise ValueError("late support frame count does not match histograms")
    for rho in sorted(early):
        early_hist = early[rho]
        late_hist = late[rho]
        if early_hist.bulk_size != late_hist.bulk_size:
            raise ValueError("early and late histograms must use the same bulk")
        if (early_hist.candidate_multiplier_q16
                != late_hist.candidate_multiplier_q16):
            raise ValueError("early and late histograms must use the same Q16 grid")
        if ((early_hist.variance_residual_sums is None)
                != (late_hist.variance_residual_sums is None)):
            raise ValueError("early and late histograms need the same variance basis")

    min_months = int(
        _REGISTER.value("era.minimum_observed_months")
        if minimum_observed_months is None else minimum_observed_months)
    min_days = float(
        _REGISTER.value("era.minimum_span_days")
        if minimum_span_days is None else minimum_span_days)
    if min_months <= 0:
        raise ValueError("minimum_observed_months must be positive")
    if not math.isfinite(min_days) or min_days <= 0.0:
        raise ValueError("minimum_span_days must be positive and finite")
    if minimum_half_retained_frames is None:
        min_half = None
    else:
        if (isinstance(minimum_half_retained_frames, bool)
                or not isinstance(minimum_half_retained_frames, Integral)):
            raise TypeError("minimum_half_retained_frames must be an integer")
        min_half = int(minimum_half_retained_frames)
        if min_half <= 0:
            raise ValueError("minimum_half_retained_frames must be positive")
    cost_limit = _limit(max_cost_ratio, "max_cost_ratio")
    systematic_limit = _limit(
        max_systematic_residual_ratio,
        "max_systematic_residual_ratio")

    base = dict(
        early_support=early_support,
        late_support=late_support,
        points_checked=0,
        points_skipped=0,
        maximum_cost_ratio=None,
        maximum_systematic_residual_ratio=None,
        maximum_masked_fraction_difference=None,
        worst_cost_point=None,
        worst_systematic_point=None,
        worst_cost_multiplier_q16=None,
        worst_systematic_multiplier_q16=None,
        cost_ratio_limit=cost_limit,
        systematic_residual_ratio_limit=systematic_limit,
        minimum_observed_months=min_months,
        minimum_span_days=min_days,
        minimum_half_retained_frames=min_half,
    )
    for label, support in (("early", early_support), ("late", late_support)):
        if support.observed_months < min_months:
            return StabilityAssessment(
                status="refused_insufficient_support",
                reason=(f"{label} half has {support.observed_months} observed "
                        f"months; need {min_months}"), **base)
        if support.span_days < min_days:
            return StabilityAssessment(
                status="refused_insufficient_support",
                reason=(f"{label} half spans {support.span_days:g} days; "
                        f"need {min_days:g}"), **base)
    if cost_limit is None or systematic_limit is None or min_half is None:
        return StabilityAssessment(
            status="refused_unconfigured",
            reason=("both drift limits and the per-half retained-frame floor "
                    "must be declared"),
            **base)

    cost_rows = []
    systematic_rows = []
    mask_rows = []
    skipped = 0
    for rho in sorted(early):
        early_hist = early[rho]
        late_hist = late[rho]
        early_surface = _surface(early_hist)
        late_surface = _surface(late_hist)
        for index, eta in enumerate(early_hist.candidate_eta):
            epoint = early_surface[index]
            lpoint = late_surface[index]
            early_kept = 0 if epoint is None else epoint[0]
            late_kept = 0 if lpoint is None else lpoint[0]
            if early_kept + late_kept < MIN_RETAINED_FRAMES:
                skipped += 1
                continue
            if (epoint is None or lpoint is None
                    or early_kept < min_half or late_kept < min_half):
                skipped += 1
                continue
            key = (rho, eta)
            q16 = early_hist.candidate_multiplier_q16[index]
            cost_rows.append((_ratio(epoint[4], lpoint[4]), key, q16))
            systematic_rows.append(
                (_ratio(epoint[2], lpoint[2]), key, q16))
            mask_rows.append((abs(epoint[1] - lpoint[1]), key))

    if not cost_rows:
        base["points_skipped"] = skipped
        return StabilityAssessment(
            status="refused_insufficient_support",
            reason="no selector-evaluable candidate has supported era halves",
            **base)
    worst_cost, worst_cost_point, worst_cost_q16 = max(
        cost_rows, key=lambda item: item[0])
    worst_systematic, worst_systematic_point, worst_systematic_q16 = max(
        systematic_rows, key=lambda item: item[0])
    worst_mask, _ = max(mask_rows, key=lambda item: item[0])
    measured = dict(base)
    measured.update(
        points_checked=len(cost_rows),
        points_skipped=skipped,
        maximum_cost_ratio=float(worst_cost),
        maximum_systematic_residual_ratio=float(worst_systematic),
        maximum_masked_fraction_difference=float(worst_mask),
        worst_cost_point=worst_cost_point,
        worst_systematic_point=worst_systematic_point,
        worst_cost_multiplier_q16=worst_cost_q16,
        worst_systematic_multiplier_q16=worst_systematic_q16,
    )
    failures = []
    if worst_cost > cost_limit:
        failures.append(
            f"cost ratio {worst_cost:.6g} exceeds {cost_limit:.6g} at "
            f"rho={worst_cost_point[0]}, eta={worst_cost_point[1]:g}")
    if worst_systematic > systematic_limit:
        failures.append(
            f"systematic-residual ratio {worst_systematic:.6g} exceeds "
            f"{systematic_limit:.6g} at rho={worst_systematic_point[0]}, "
            f"eta={worst_systematic_point[1]:g}")
    if failures:
        return StabilityAssessment(
            status="refused_drift", reason="; ".join(failures), **measured)
    return StabilityAssessment(
        status="passed",
        reason="supported point estimates are within both declared drift limits",
        **measured)


def _one_dimensional_numeric(values, name: str, *, positive=False):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain only numbers")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain only real numbers")
    array = array.astype(float, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if positive and (array <= 0.0).any():
        raise ValueError(f"{name} must be positive")
    return array


def _half_support(times, acquisitions) -> EraHalfSupport:
    months = {
        (stamp.year, stamp.month)
        for stamp in (dt.datetime.fromtimestamp(float(value), dt.timezone.utc)
                      for value in times)
    }
    return EraHalfSupport(
        frame_count=int(times.size),
        acquisition_count=len(set(acquisitions)),
        observed_months=len(months),
        span_days=float((times.max() - times.min()) / 86400.0),
    )


def _frame_block_ids(values, size: int):
    if isinstance(values, (str, bytes)):
        raise TypeError("stability_block_ids must be a sequence")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError("stability_block_ids must be a sequence") from exc
    if len(items) != size:
        raise ValueError("stability_block_ids must match systematic_residuals")
    checked = []
    for value in items:
        if isinstance(value, bool):
            raise TypeError("stability_block_ids must contain strings or integers")
        if isinstance(value, Integral):
            checked.append(("integer", int(value)))
        elif isinstance(value, str):
            checked.append(("string", value))
        else:
            raise TypeError(
                "stability_block_ids must contain strings or integers")
    return tuple(checked)


def _same_partition(left, right) -> bool:
    left_to_right = {}
    right_to_left = {}
    for left_value, right_value in zip(left, right):
        if (left_value in left_to_right
                and left_to_right[left_value] != right_value):
            return False
        if (right_value in right_to_left
                and right_to_left[right_value] != left_value):
            return False
        left_to_right[left_value] = right_value
        right_to_left[right_value] = left_value
    return True


def _block_labels(block_ids, mask):
    return tuple(sorted({value for value, keep in zip(block_ids, mask) if keep}))


def _sorted_half_rows(
        requirements, candidates, systematic, variance, mask,
        block_ids, labels,
):
    label_index = {value: index for index, value in enumerate(labels)}
    rows = np.flatnonzero(mask)
    ordered_rows = np.asarray(sorted(
        rows,
        key=lambda index: (
            requirements[index], block_ids[index], systematic[index],
            -1.0 if variance is None else variance[index])),
        dtype=np.int64)
    ordered_requirements = tuple(requirements[index] for index in ordered_rows)
    groups = np.fromiter(
        (label_index[block_ids[index]] for index in ordered_rows),
        dtype=np.int64, count=len(ordered_rows))
    stops = np.fromiter(
        (bisect_right(ordered_requirements, candidate)
         for candidate in candidates),
        dtype=np.int64, count=len(candidates))
    return (
        groups,
        systematic[ordered_rows],
        None if variance is None else variance[ordered_rows],
        stops,
    )


def _weighted_prefixes(draws, groups, systematic, variance):
    weights = draws[:, groups]
    counts = np.cumsum(weights, axis=1)
    systematic_sums = np.cumsum(weights * systematic[None, :], axis=1)
    variance_sums = (None if variance is None else np.cumsum(
        weights * variance[None, :], axis=1))
    return counts, systematic_sums, variance_sums


def _prefix_metrics(prefixes, stops, minimum_kept):
    counts, systematic_sums, variance_sums = prefixes
    columns = stops - 1
    total = counts[:, -1]
    kept = counts[:, columns]
    systematic = systematic_sums[:, columns]
    variance = (None if variance_sums is None else
                variance_sums[:, columns])
    valid = kept >= minimum_kept
    kept_safe = np.where(valid, kept, 1)
    kept_fraction = kept_safe / total[:, None]
    systematic_mean = systematic / kept_safe
    variance_mean = (None if variance is None else variance / kept_safe)
    cost = ((1.0 if variance_mean is None else 1.0 + variance_mean)
            / kept_fraction)
    systematic_mean = np.where(valid, systematic_mean, np.nan)
    cost = np.where(valid, cost, np.nan)
    return systematic_mean, cost, valid


def _symmetric_ratio_array(left, right):
    valid = ~np.isnan(left) & ~np.isnan(right)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    ratio = np.full(left.shape, np.nan, dtype=float)
    both_zero = valid & (high == 0.0)
    one_zero = valid & (low == 0.0) & ~both_zero
    positive = valid & (low > 0.0)
    ratio[both_zero] = 1.0
    ratio[one_zero] = math.inf
    ratio[positive] = high[positive] / low[positive]
    return ratio, valid


def _upper_order_statistic(values, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    index = max(0, int(math.ceil(coverage * ordered.size)) - 1)
    return float(ordered[index])


def _block_stability_assessment(
        requirements, systematic, variance, early_mask, block_ids,
        plan: BlockResamplingPlan, stability: StabilityAssessment,
) -> BlockStabilityAssessment:
    method = "stratified_whole_block_surface_maximum_upper_percentile"
    early_labels = _block_labels(block_ids, early_mask)
    late_labels = _block_labels(block_ids, ~early_mask)
    common = dict(
        method=method,
        block_unit=plan.block_unit,
        seed=plan.seed,
        replicates_requested=plan.replicates,
        minimum_successful_replicates=plan.minimum_successful_replicates,
        interval_coverage=plan.interval_coverage,
        minimum_blocks_per_half=plan.minimum_blocks_per_half,
        early_blocks=len(early_labels),
        late_blocks=len(late_labels),
        points_checked=stability.points_checked,
        maximum_cost_ratio_upper_bound=None,
        maximum_systematic_residual_ratio_upper_bound=None,
    )
    if set(early_labels) & set(late_labels):
        return BlockStabilityAssessment(
            status="refused_split_blocks",
            reason="a resampling block crosses the calendar split",
            replicates_succeeded=0,
            **common,
        )
    if (len(early_labels) < plan.minimum_blocks_per_half
            or len(late_labels) < plan.minimum_blocks_per_half):
        return BlockStabilityAssessment(
            status="refused_insufficient_blocks",
            reason=(f"each era half needs {plan.minimum_blocks_per_half} "
                    f"{plan.block_unit} blocks by caller declaration"),
            replicates_succeeded=0,
            **common,
        )

    candidates_by_rho = {}
    for rho, values in requirements.items():
        candidates = candidate_eta_q16(values)
        ordered = sorted(values)
        early_ordered = sorted(v for v, keep in zip(values, early_mask) if keep)
        late_ordered = sorted(v for v, keep in zip(values, early_mask) if not keep)
        selected = [
            candidate for candidate in candidates
            if bisect_right(ordered, candidate) >= MIN_RETAINED_FRAMES
            and bisect_right(early_ordered, candidate) >= stability.minimum_half_retained_frames
            and bisect_right(late_ordered, candidate) >= stability.minimum_half_retained_frames
        ]
        if selected:
            candidates_by_rho[rho] = tuple(selected)
    points = sum(len(values) for values in candidates_by_rho.values())
    if points != stability.points_checked:
        raise RuntimeError("block resampling candidate surface does not match screen")

    rng = np.random.default_rng(plan.seed)
    early_draws = rng.multinomial(
        len(early_labels), np.full(len(early_labels), 1.0 / len(early_labels)),
        size=plan.replicates)
    late_draws = rng.multinomial(
        len(late_labels), np.full(len(late_labels), 1.0 / len(late_labels)),
        size=plan.replicates)
    successful = np.ones(plan.replicates, dtype=bool)
    maximum_cost = np.zeros(plan.replicates, dtype=float)
    maximum_systematic = np.zeros(plan.replicates, dtype=float)

    maximum_prefix_cells = 1_000_000
    minimum_kept = int(stability.minimum_half_retained_frames)
    for rho in sorted(candidates_by_rho):
        rank_candidates = candidates_by_rho[rho]
        early_rows = _sorted_half_rows(
            requirements[rho], rank_candidates, systematic, variance,
            early_mask, block_ids, early_labels)
        late_rows = _sorted_half_rows(
            requirements[rho], rank_candidates, systematic, variance,
            ~early_mask, block_ids, late_labels)
        frame_columns = len(early_rows[0]) + len(late_rows[0])
        replicate_batch = max(
            1, min(plan.replicates,
                   maximum_prefix_cells // frame_columns))
        for replica_start in range(0, plan.replicates, replicate_batch):
            replica_stop = min(
                plan.replicates, replica_start + replicate_batch)
            replica_slice = slice(replica_start, replica_stop)
            early_prefixes = _weighted_prefixes(
                early_draws[replica_slice], *early_rows[:3])
            late_prefixes = _weighted_prefixes(
                late_draws[replica_slice], *late_rows[:3])
            candidate_batch = max(
                1, min(256, maximum_prefix_cells
                       // (replica_stop - replica_start)))
            for candidate_start in range(
                    0, len(rank_candidates), candidate_batch):
                candidate_stop = min(
                    len(rank_candidates), candidate_start + candidate_batch)
                candidate_slice = slice(candidate_start, candidate_stop)
                early_systematic, early_cost, early_valid = _prefix_metrics(
                    early_prefixes, early_rows[3][candidate_slice],
                    minimum_kept)
                late_systematic, late_cost, late_valid = _prefix_metrics(
                    late_prefixes, late_rows[3][candidate_slice],
                    minimum_kept)
                cost_ratio, cost_valid = _symmetric_ratio_array(
                    early_cost, late_cost)
                systematic_ratio, systematic_valid = _symmetric_ratio_array(
                    early_systematic, late_systematic)
                chunk_valid = (early_valid & late_valid & cost_valid
                               & systematic_valid).all(axis=1)
                successful[replica_slice] &= chunk_valid
                maximum_cost[replica_slice] = np.maximum(
                    maximum_cost[replica_slice],
                    np.max(np.where(
                        cost_valid, cost_ratio, -math.inf), axis=1))
                maximum_systematic[replica_slice] = np.maximum(
                    maximum_systematic[replica_slice],
                    np.max(np.where(
                        systematic_valid, systematic_ratio, -math.inf),
                        axis=1))

    succeeded = int(successful.sum())
    if succeeded < plan.minimum_successful_replicates:
        return BlockStabilityAssessment(
            status="refused_insufficient_resamples",
            reason=(f"{succeeded} of {plan.replicates} resamples retain every "
                    f"candidate; need {plan.minimum_successful_replicates}"),
            replicates_succeeded=succeeded,
            **common,
        )

    maximum_cost[~successful] = math.inf
    maximum_systematic[~successful] = math.inf
    cost_bound = max(
        float(stability.maximum_cost_ratio),
        _upper_order_statistic(maximum_cost, plan.interval_coverage))
    systematic_bound = max(
        float(stability.maximum_systematic_residual_ratio),
        _upper_order_statistic(maximum_systematic, plan.interval_coverage))
    measured = dict(common)
    measured.update(
        replicates_succeeded=succeeded,
        maximum_cost_ratio_upper_bound=cost_bound,
        maximum_systematic_residual_ratio_upper_bound=systematic_bound,
    )
    if not math.isfinite(cost_bound) or not math.isfinite(systematic_bound):
        unbounded = dict(measured)
        if not math.isfinite(cost_bound):
            unbounded["maximum_cost_ratio_upper_bound"] = None
        if not math.isfinite(systematic_bound):
            unbounded["maximum_systematic_residual_ratio_upper_bound"] = None
        return BlockStabilityAssessment(
            status="refused_unbounded",
            reason="the requested percentile has no finite surface-wide bound",
            **unbounded,
        )
    failures = []
    if cost_bound > float(stability.cost_ratio_limit):
        failures.append(
            f"cost-ratio upper bound {cost_bound:.6g} exceeds "
            f"{stability.cost_ratio_limit:.6g}")
    if systematic_bound > float(stability.systematic_residual_ratio_limit):
        failures.append(
            f"systematic-residual-ratio upper bound {systematic_bound:.6g} "
            f"exceeds {stability.systematic_residual_ratio_limit:.6g}")
    if failures:
        return BlockStabilityAssessment(
            status="refused_uncertainty", reason="; ".join(failures),
            **measured)
    return BlockStabilityAssessment(
        status="passed",
        reason="surface-wide upper bounds are within both declared limits",
        **measured,
    )


def prepare_family(
        required_multiplier_q16_by_rho,
        systematic_residuals,
        *,
        frame_times,
        acquisition_ids,
        exposure_seconds,
        source_id: str,
        era_label: str,
        latest_era: bool,
        additive_residuals: bool,
        score: CalibrationEvidence,
        correlation: CalibrationEvidence,
        transfer: CalibrationEvidence,
        max_cost_ratio: float | None,
        max_systematic_residual_ratio: float | None,
        minimum_half_retained_frames: int | None,
        variance_residuals=None,
        minimum_observed_months: int | None = None,
        minimum_span_days: float | None = None,
        stability_block_ids=None,
        stability_resampling: BlockResamplingPlan | None = None,
        incoherent_residuals=None,
        floor_dominated=None,
) -> PreparedFamily:
    """Build a complete family from accepted latest-era Q16 boundaries.

    Era discovery and invalid-frame bookkeeping stay outside this function.
    This boundary derives the candidate grids, calendar split, support counts,
    histograms, and drift evidence from the supplied frame rows.

    ``incoherent_residuals`` (the per-frame residual at G = 1) and
    ``floor_dominated`` (whether the floor term sets the frame's residual) are
    optional; with them the pooled histograms also carry the sums from which
    the surface reports ``r_sys_incoherent`` and ``floor_share``. They do not
    enter the drift screen.
    """
    requirements = _ranked_requirements(required_multiplier_q16_by_rho)
    bulk_size = max(requirements)
    if tuple(sorted(requirements)) != candidate_rho_values(bulk_size):
        raise ValueError("requirements must contain every supported rank")
    systematic = _one_dimensional_numeric(
        systematic_residuals, "systematic_residuals")
    if (systematic < 0.0).any():
        raise ValueError("systematic_residuals must be non-negative")
    variance = (None if variance_residuals is None else
                _one_dimensional_numeric(
                    variance_residuals, "variance_residuals"))
    if variance is not None and (variance < 0.0).any():
        raise ValueError("variance_residuals must be non-negative")
    times = _one_dimensional_numeric(frame_times, "frame_times")
    exposure = _one_dimensional_numeric(
        exposure_seconds, "exposure_seconds", positive=True)
    try:
        acquisitions = tuple(acquisition_ids)
    except TypeError as exc:
        raise TypeError("acquisition_ids must be a sequence") from exc
    size = systematic.size
    if size == 0:
        raise ValueError("frame rows must not be empty")
    for values, name in ((variance, "variance_residuals"),
                         (times, "frame_times"),
                         (exposure, "exposure_seconds")):
        if values is not None and values.size != size:
            raise ValueError(f"{name} must match systematic_residuals")
    if len(acquisitions) != size:
        raise ValueError("acquisition_ids must match systematic_residuals")
    try:
        set(acquisitions)
    except TypeError as exc:
        raise TypeError("acquisition_ids must contain hashable values") from exc
    if any(len(values) != size for values in requirements.values()):
        raise ValueError("all ranks must describe the same frame population")
    if (stability_block_ids is None) != (stability_resampling is None):
        raise ValueError(
            "stability_block_ids and stability_resampling must be supplied together")
    if (stability_resampling is not None
            and not isinstance(stability_resampling, BlockResamplingPlan)):
        raise TypeError("stability_resampling must be BlockResamplingPlan")
    block_ids = (None if stability_block_ids is None else
                 _frame_block_ids(stability_block_ids, size))
    if (block_ids is not None
            and stability_resampling.block_unit == "acquisition"
            and not _same_partition(acquisitions, block_ids)):
        raise ValueError(
            "acquisition stability blocks must match acquisition_ids")
    if not np.equal(exposure, exposure[0]).all():
        raise ValueError("accepted frames must have equal exposure")
    start = float(times.min())
    stop = float(times.max())
    if stop <= start:
        raise ValueError("frame_times must span more than one instant")
    early_mask = times <= 0.5 * (start + stop)
    late_mask = ~early_mask
    if not early_mask.any() or not late_mask.any():
        raise ValueError("calendar split must leave frames in both halves")

    incoherent = (None if incoherent_residuals is None else
                  _one_dimensional_numeric(incoherent_residuals, "incoherent_residuals"))
    dominated = (None if floor_dominated is None else
                 np.asarray(floor_dominated, dtype=bool))
    for values, name in ((incoherent, "incoherent_residuals"), (dominated, "floor_dominated")):
        if values is not None and values.shape != (size,):
            raise ValueError(f"{name} must match systematic_residuals")

    pooled = {}
    early = {}
    late = {}
    for rho, values in requirements.items():
        q16 = candidate_eta_q16(values)
        pooled[rho] = build_score_histogram(
            values, systematic, q16, bulk_size=bulk_size,
            variance_residuals=variance, incoherent_residuals=incoherent,
            floor_dominated=dominated)
        early[rho] = build_score_histogram(
            tuple(value for value, keep in zip(values, early_mask) if keep),
            systematic[early_mask], q16, bulk_size=bulk_size,
            variance_residuals=(None if variance is None else
                                variance[early_mask]))
        late[rho] = build_score_histogram(
            tuple(value for value, keep in zip(values, late_mask) if keep),
            systematic[late_mask], q16, bulk_size=bulk_size,
            variance_residuals=(None if variance is None else
                                variance[late_mask]))
        if minimum_half_retained_frames is not None:
            # The optimizer must use exactly the supported surface assessed
            # below, while retaining all bins for exact cumulative sums.
            ec = np.cumsum(early[rho].counts[:-1])
            lc = np.cumsum(late[rho].counts[:-1])
            pooled[rho] = replace(pooled[rho], candidate_eligible=tuple(
                bool(e >= minimum_half_retained_frames and l >= minimum_half_retained_frames)
                for e, l in zip(ec, lc)))

    early_support = _half_support(
        times[early_mask], tuple(value for value, keep
                                 in zip(acquisitions, early_mask) if keep))
    late_support = _half_support(
        times[late_mask], tuple(value for value, keep
                                in zip(acquisitions, late_mask) if keep))
    stability = assess_histogram_stability(
        early, late,
        early_support=early_support,
        late_support=late_support,
        max_cost_ratio=max_cost_ratio,
        max_systematic_residual_ratio=max_systematic_residual_ratio,
        minimum_observed_months=minimum_observed_months,
        minimum_span_days=minimum_span_days,
        minimum_half_retained_frames=minimum_half_retained_frames,
    )
    if block_ids is not None and stability.passed:
        stability = replace(
            stability,
            block_uncertainty=_block_stability_assessment(
                requirements, systematic, variance, early_mask, block_ids,
                stability_resampling, stability),
        )
    return PreparedFamily(
        histograms_by_rho=pooled,
        source_id=source_id,
        era_label=era_label,
        latest_era=latest_era,
        valid_frames_only=True,
        equal_exposure_frames=True,
        additive_residuals=additive_residuals,
        stability=stability,
        score=score,
        correlation=correlation,
        transfer=transfer,
    )


def _ranked_requirements(values):
    if not isinstance(values, Mapping):
        raise TypeError("required_multiplier_q16_by_rho must be a mapping")
    if not values:
        raise ValueError("required_multiplier_q16_by_rho must not be empty")
    out = {}
    for rho, requirements in values.items():
        if isinstance(rho, bool) or not isinstance(rho, Integral):
            raise TypeError("requirement keys must be integers")
        if int(rho) <= 0:
            raise ValueError("requirement keys must be positive")
        try:
            items = tuple(requirements)
        except TypeError as exc:
            raise TypeError("rank requirements must be sequences") from exc
        candidate_eta_q16(items)
        out[int(rho)] = items
    return out


@dataclass(frozen=True)
class PreparedFamily:
    """Histogram family plus the evidence required to interpret it."""

    histograms_by_rho: Mapping[int, ScoreHistogram]
    source_id: str
    era_label: str
    latest_era: bool
    valid_frames_only: bool
    equal_exposure_frames: bool
    additive_residuals: bool
    stability: StabilityAssessment
    score: CalibrationEvidence
    correlation: CalibrationEvidence
    transfer: CalibrationEvidence
    policy_json: str = field(default_factory=_REGISTER.canonical_json)
    policy_sha256: str = field(default_factory=_REGISTER.sha256)

    def __post_init__(self):
        histograms = _ranked(self.histograms_by_rho, "histograms_by_rho")
        bulk_size = next(iter(histograms.values())).bulk_size
        expected_ranks = candidate_rho_values(bulk_size)
        if tuple(sorted(histograms)) != expected_ranks:
            raise ValueError(
                "prepared families must contain every supported rank")
        for histogram in histograms.values():
            if histogram.bulk_size != bulk_size:
                raise ValueError("prepared histograms need one common bulk size")
            q16 = histogram.candidate_multiplier_q16
            if q16[0] != int(_REGISTER.value(
                    "preparation.minimum_multiplier_q16")):
                raise ValueError("prepared multiplier grids must start at Q16 one")
        object.__setattr__(self, "histograms_by_rho",
                           MappingProxyType(dict(sorted(histograms.items()))))
        if not self.source_id.startswith("sha256:"):
            raise ValueError("source_id must be a sha256 content identifier")
        source_digest = self.source_id.removeprefix("sha256:")
        if (len(source_digest) != 64
                or any(ch not in "0123456789abcdef" for ch in source_digest)):
            raise ValueError("source_id must contain a lowercase SHA-256 digest")
        if not self.era_label.strip():
            raise ValueError("era_label must not be empty")
        for name in ("latest_era", "valid_frames_only",
                     "equal_exposure_frames", "additive_residuals"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if not isinstance(self.stability, StabilityAssessment):
            raise TypeError("stability must be StabilityAssessment")
        for name in ("score", "correlation", "transfer"):
            if not isinstance(getattr(self, name), CalibrationEvidence):
                raise TypeError(f"{name} must be CalibrationEvidence")
        if not isinstance(self.policy_json, str):
            raise TypeError("policy_json must be a string")
        try:
            policy = json.loads(self.policy_json)
        except json.JSONDecodeError as exc:
            raise ValueError("policy_json must contain valid JSON") from exc
        if policy.get("schema") != _REGISTER.schema:
            raise ValueError("policy_json uses an unknown schema")
        if (len(self.policy_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in self.policy_sha256)):
            raise ValueError("policy_sha256 must be a lowercase SHA-256 digest")
        stored_sha256 = hashlib.sha256(self.policy_json.encode("utf-8")).hexdigest()
        if self.policy_sha256 != stored_sha256:
            raise ValueError("policy_sha256 does not match policy_json")

    def refusals(self, *, allow_screening: bool = False) -> tuple[str, ...]:
        """Return every reason this family cannot support selection."""
        out = []
        if (self.policy_sha256 != _REGISTER.sha256()
                or self.policy_json != _REGISTER.canonical_json()):
            out.append("threshold decision snapshot does not match this release")
        if not self.latest_era:
            out.append("the histogram does not describe the latest era")
        if not self.valid_frames_only:
            out.append("invalid frames were not removed during preparation")
        if not self.equal_exposure_frames:
            out.append("frame counts are not equal-exposure counts")
        if not self.additive_residuals:
            out.append("residual calibration is mask-dependent and non-additive")
        if not self.stability.passed:
            out.append(f"within-era stability {self.stability.status}: "
                       f"{self.stability.reason}")
        elif (self.stability.block_uncertainty is not None
              and not self.stability.block_uncertainty.passed):
            uncertainty = self.stability.block_uncertainty
            out.append(f"within-era block uncertainty {uncertainty.status}: "
                       f"{uncertainty.reason}")
        elif (not allow_screening
              and self.stability.block_uncertainty is None):
            out.append("within-era stability has no block-resampled upper bound")
        if not allow_screening:
            unresolved = _blockers(OPERATIONAL_REQUIRED_IDS)
            if unresolved:
                joined = ", ".join(
                    f"{item.id} ({item.status})" for item in unresolved)
                out.append(f"unresolved operating decisions: {joined}")
        accepted = ({"measured", "bounded", "conditional"}
                    if allow_screening else {"measured", "bounded"})
        if self.score.state not in accepted:
            out.append(f"score evidence is {self.score.state}")
        if self.correlation.state not in accepted:
            out.append(f"correlation evidence is {self.correlation.state}")
        if self.transfer.state not in accepted:
            out.append(f"science transfer evidence is {self.transfer.state}")
        return tuple(out)

    def require_screening(self) -> None:
        """Raise :class:`StabilityRefused` when the family cannot support even a screening claim."""
        reasons = self.refusals(allow_screening=True)
        if reasons:
            raise StabilityRefused("; ".join(reasons))

    @property
    def status(self) -> str:
        if not self.refusals():
            return "operational"
        if not self.refusals(allow_screening=True):
            return "screening"
        return "refused"

    def metadata(self) -> dict:
        """Return the policy identity and evidence state for serialization."""
        return {
            "source_id": self.source_id,
            "era_label": self.era_label,
            "status": self.status,
            "latest_era": self.latest_era,
            "valid_frames_only": self.valid_frames_only,
            "equal_exposure_frames": self.equal_exposure_frames,
            "additive_residuals": self.additive_residuals,
            "policy_sha256": self.policy_sha256,
            "policy": json.loads(self.policy_json),
            "stability": asdict(self.stability),
            "score": asdict(self.score),
            "correlation": asdict(self.correlation),
            "transfer": asdict(self.transfer),
        }


# provisional drift-screen values (register ids stability.*, open): recorded on every result
PROVISIONAL_MIN_HALF_RETAINED = 30
PROVISIONAL_MAX_COST_RATIO = 1.05
PROVISIONAL_MAX_SYSTEMATIC_RATIO = 1.10

# where a real operating point lives: the fraction of each era half a candidate must keep for its
# early/late ratios to describe the drift a selected point would see, rather than a sparse tail
DRIFT_KEPT_FRACTIONS = (0.0, 0.01, 0.05, 0.2)


def drift_diagnostic(bundle, residuals: np.ndarray, frame_time: np.ndarray, *,
                     min_half: int = PROVISIONAL_MIN_HALF_RETAINED,
                     kept_fractions: Sequence[float] = DRIFT_KEPT_FRACTIONS) -> dict:
    """Why the within-era drift screen refused, measured on the candidates a selector could choose.

    :func:`assess_histogram_stability` refuses the whole
    family at the first candidate whose pooled kept count reaches the
    thirty-frame floor while one calendar half falls below ``min_half``. Every
    eta sweep crosses that band, so the refusal names a point keeping a few
    tenths of a percent of the block, which no selector would choose, and its
    message says "insufficient support" whatever the data does.

    This records both facts: the candidate the screen refused on (with the
    frames it keeps in each half), and the worst early/late cost and
    systematic-residual ratios over the candidates that keep at least a stated
    fraction of *each* half. The second is the drift a selected point would
    actually see; where it exceeds the declared limits the refusal is the
    right verdict for the wrong stated reason.
    """
    times = np.asarray(frame_time, dtype=float)
    if times.size == 0 or not np.isfinite(times).any():
        return {"drift_status": "no timed frames"}
    mid = 0.5 * (np.nanmin(times) + np.nanmax(times))
    early_mask, late_mask = times <= mid, times > mid
    n_early, n_late = int(early_mask.sum()), int(late_mask.sum())
    if not (n_early and n_late):
        return {"drift_status": "calendar split leaves one half empty"}
    requirements = bundle.requirements_by_rho()
    bulk_size = max(requirements)
    refused = None
    rows: list[tuple[float, float, float]] = []
    for rho in sorted(requirements):
        values = tuple(requirements[rho])
        grid = tuple(sorted({int(v) for v in values if int(v) != ALWAYS_MASKED_Q16}))
        if not grid:
            continue
        early = build_score_histogram(
            tuple(v for v, k in zip(values, early_mask) if k), residuals[early_mask], grid, bulk_size=bulk_size)
        late = build_score_histogram(
            tuple(v for v, k in zip(values, late_mask) if k), residuals[late_mask], grid, bulk_size=bulk_size)
        e_surface, l_surface = _surface(early), _surface(late)
        for index, eta in enumerate(early.candidate_eta):
            e, l = e_surface[index], l_surface[index]
            e_kept = 0 if e is None else e[0]
            l_kept = 0 if l is None else l[0]
            if e_kept + l_kept < MIN_RETAINED_FRAMES:
                continue
            if e is None or l is None or e_kept < min_half or l_kept < min_half:
                if refused is None:
                    refused = {"drift_refused_rho": int(rho), "drift_refused_eta": float(eta),
                               "drift_refused_early_kept": int(e_kept), "drift_refused_late_kept": int(l_kept),
                               "drift_refused_kept_fraction": (e_kept + l_kept) / (n_early + n_late)}
                continue
            rows.append((min(e_kept / n_early, l_kept / n_late), _ratio(e[4], l[4]), _ratio(e[2], l[2])))
    out = {"drift_status": "measured" if rows else "no evaluable candidate",
           "drift_early_frames": n_early, "drift_late_frames": n_late, **(refused or {})}
    if rows:
        arr = np.asarray(rows, dtype=float)
        for fraction in kept_fractions:
            keep = arr[arr[:, 0] >= float(fraction)]
            tag = f"{float(fraction):g}".replace(".", "p")
            if keep.size:
                out[f"drift_candidates_at_{tag}"] = int(keep.shape[0])
                out[f"drift_max_cost_ratio_at_{tag}"] = float(keep[:, 1].max())
                out[f"drift_max_systematic_ratio_at_{tag}"] = float(keep[:, 2].max())
    return out


__all__ = [
    "StabilityRefused", "EraHalfSupport", "BlockResamplingPlan", "BlockStabilityAssessment",
    "StabilityAssessment", "PreparedFamily", "assess_histogram_stability", "prepare_family",
    "drift_diagnostic", "DRIFT_KEPT_FRACTIONS", "OPERATIONAL_REQUIRED_IDS",
    "PROVISIONAL_MIN_HALF_RETAINED", "PROVISIONAL_MAX_COST_RATIO", "PROVISIONAL_MAX_SYSTEMATIC_RATIO",
]
