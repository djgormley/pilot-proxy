"""The operating surface: every candidate threshold's masked fraction and kept residual.

Every candidate ``(rho, eta)`` of a band's calibration block gives a masked
fraction ``f`` and a kept-frame residual ``r_sys``. This module enumerates
them (:func:`operating_surface`) from the prepared score histograms, marks
the two notable points (the least-residual point and the Pareto *knee*),
replays a point on another block (:func:`replay`), and characterizes one
block end to end (:func:`characterize_block`). It never reads a science
tolerance: which candidates are acceptable, and at what cost, is the science
side's (its verdict).

Histograms. ``counts[j]`` covers exact scores above the previous boundary and
at or below ``candidate_multiplier_q16[j]``; the final count is the overflow.
Residual arrays hold calibrated additive totals in the same bins, so a prefix
sum over the retained count is the retained residual. Besides ``r_sys`` (with
the chain gain G) the surface carries, computed in the same loop:

- ``r_sys_incoherent``: the same kept-frame mean at G = 1;
- ``floor_share``: the fraction of kept frames whose residual is the floor
  (the floor term dominates the shelf, or the frame has no shelf estimate);
- ``exposure_cost_uniform_loss``: ``1 / (1 - f)``, observing time per unit of
  kept time if the lost frames are spread evenly;
- ``candidate_order``: the enumeration order (rank, then multiplier).

Mask rule. A frame is kept when the multiplier is at least its keep boundary
(:func:`kept_at`): with ``eta = eta_q16 / 2^16`` and the frame's boundary
``b``, the frame is masked if and only if ``b > eta``; equality keeps it.

Ties. Candidates that keep the same frames (one kept set reached at several
ranks, or a plateau where every kept frame sits at the floor) have residuals
equal up to summation-order rounding. A :class:`TieRule` says how a
least-residual choice breaks them. The default :data:`SELECTOR_ORDER` treats
residuals within :data:`TIE_REL_TOL` (relative) as equal and breaks the tie by
the smallest masked fraction, then the lowest rank, then the lowest
multiplier; the frontier represents each masked fraction by its least residual
and a larger fraction joins only when its residual is lower by more than the
tolerance, so a flat stretch is represented by its least-masking point. The
first-minimum rule of the author-dated-eras release is a record configuration
(``pilot_proxy.records.chime_atsc_2026.archive_releases``).

The knee (a declared policy, with its sensitivities recorded): cost
``1/(1-f)`` and residual ``r`` are each mapped to ``[0, 1]`` across the
frontier's own span in log space, and the point chosen is the one closest to
the corner where both are least. The alternative, reported beside it, is the
smallest mask within a margin of the frontier's floor,
``f = min{f : r(f) <= (1 + margin) r_floor}`` for ``margin`` in
``(0.05, 0.10, 0.25, 0.50)``.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Iterable, Protocol, TypeVar

import numpy as np

from pilot_proxy.config.project import default_project

from . import blocks
from .residual_chain import floor_dominated, frame_residuals

_REGISTER = default_project().register
MIN_RETAINED_FRAMES = int(
    _REGISTER.value("selection.minimum_retained_frames"))
Q16_FRACTION_BITS = 16
Q16_SCALE = 1 << Q16_FRACTION_BITS
MAX_MULTIPLIER_Q16 = (1 << 64) - 1
ALWAYS_MASKED_Q16 = 1 << 64
DESIGNATED_HALF_WIDTH = 2               # D = {(f_a + k) mod 256 : |k| <= 2}


def _float_tuple(values, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of numbers") from exc
    out = []
    for value in items:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain only numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite values")
        if number < 0.0:
            raise ValueError(f"{name} must contain only non-negative values")
        out.append(number)
    return tuple(out)


def _count_tuple(values) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("counts must be a sequence of integers")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError("counts must be a sequence of integers") from exc
    out = []
    for value in items:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("counts must contain only integers")
        count = int(value)
        if count < 0:
            raise ValueError("counts must be non-negative")
        out.append(count)
    return tuple(out)


def _q16_tuple(values, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc
    out = []
    for value in items:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain only integers")
        q16 = int(value)
        if not 1 <= q16 <= MAX_MULTIPLIER_Q16:
            raise ValueError(
                f"{name} must be between 1 and {MAX_MULTIPLIER_Q16}")
        out.append(q16)
    return tuple(out)


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


@dataclass(frozen=True)
class ScoreHistogram:
    """A complete score histogram for one candidate rank.

    Ranks are one-based. Every accepted frame must have the same duration and
    exposure, so count fractions are also exposure fractions. All ranks in one
    optimization must describe the same accepted frame population and common
    usable bulk.

    ``counts[j]`` covers exact scores above the previous boundary and at or
    below ``candidate_multiplier_q16[j]``. ``candidate_eta`` is the derived
    floating display of that exact grid. The final count is the overflow above
    the last boundary. Residual arrays contain calibrated additive totals in
    the same bins. Their prefix sum divided by the retained count must be the
    calibrated residual for that retained population. A calibration that must
    be rerun after masking cannot use this compressed form.

    ``incoherent_residual_sums`` (the residual at G = 1) and ``floor_counts``
    (frames whose residual is the floor) are optional bins of the same form;
    they feed ``r_sys_incoherent`` and ``floor_share`` and nothing else.
    """

    bulk_size: int
    candidate_multiplier_q16: tuple[int, ...]
    counts: tuple[int, ...]
    systematic_residual_sums: tuple[float, ...]
    variance_residual_sums: tuple[float, ...] | None = None
    candidate_eligible: tuple[bool, ...] | None = None
    incoherent_residual_sums: tuple[float, ...] | None = None
    floor_counts: tuple[int, ...] | None = None
    candidate_eta: tuple[float, ...] = field(init=False)

    def __post_init__(self):
        bulk_size = _positive_integer(self.bulk_size, "bulk_size")
        q16 = _q16_tuple(
            self.candidate_multiplier_q16, "candidate_multiplier_q16")
        counts = _count_tuple(self.counts)
        systematic = _float_tuple(
            self.systematic_residual_sums, "systematic_residual_sums")
        variance = (None if self.variance_residual_sums is None else
                    _float_tuple(self.variance_residual_sums,
                                 "variance_residual_sums"))
        if not q16:
            raise ValueError("candidate_multiplier_q16 must not be empty")
        if self.candidate_eligible is not None:
            eligible = tuple(self.candidate_eligible)
            if len(eligible) != len(q16) or any(type(v) is not bool for v in eligible):
                raise ValueError("candidate_eligible must contain one boolean per candidate")
            object.__setattr__(self, "candidate_eligible", eligible)
        if len(counts) != len(q16) + 1:
            raise ValueError("counts must include one overflow bin")
        if len(systematic) != len(counts):
            raise ValueError("systematic_residual_sums must match counts")
        if variance is not None and len(variance) != len(counts):
            raise ValueError("variance_residual_sums must match counts")
        incoherent = (None if self.incoherent_residual_sums is None else
                      _float_tuple(self.incoherent_residual_sums,
                                   "incoherent_residual_sums"))
        floor_counts = (None if self.floor_counts is None else
                        _count_tuple(self.floor_counts))
        if incoherent is not None and len(incoherent) != len(counts):
            raise ValueError("incoherent_residual_sums must match counts")
        if floor_counts is not None and (
                len(floor_counts) != len(counts)
                or any(f > c for f, c in zip(floor_counts, counts))):
            raise ValueError("floor_counts must match counts")
        if any(right <= left for left, right in zip(q16, q16[1:])):
            raise ValueError(
                "candidate_multiplier_q16 must be strictly increasing")
        if sum(counts) == 0:
            raise ValueError("the histogram must contain at least one frame")
        for index, count in enumerate(counts):
            if count == 0 and systematic[index] != 0.0:
                raise ValueError("an empty bin cannot carry systematic residual")
            if count == 0 and variance is not None and variance[index] != 0.0:
                raise ValueError("an empty bin cannot carry variance residual")

        object.__setattr__(self, "bulk_size", bulk_size)
        object.__setattr__(self, "candidate_multiplier_q16", q16)
        object.__setattr__(
            self, "candidate_eta", tuple(value / Q16_SCALE for value in q16))
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "systematic_residual_sums", systematic)
        object.__setattr__(self, "variance_residual_sums", variance)
        object.__setattr__(self, "incoherent_residual_sums", incoherent)
        object.__setattr__(self, "floor_counts", floor_counts)

    @property
    def frame_count(self) -> int:
        return sum(self.counts)


@dataclass(frozen=True)
class SurfacePoint:
    """One evaluated ``(rho, eta)`` pair of the operating surface."""

    rho: int
    bulk_size: int
    rank_fraction: float
    eta_q16: int
    frame_count: int
    kept_frames: int
    masked_frames: int
    masked_fraction: float
    r_sys: float
    r_var: float | None
    exposure_cost_uniform_loss: float
    candidate_order: int
    r_sys_incoherent: float = math.nan
    floor_share: float = math.nan
    eta: float = field(init=False)

    def __post_init__(self):
        q16 = _q16_tuple((self.eta_q16,), "eta_q16")[0]
        object.__setattr__(self, "eta_q16", q16)
        object.__setattr__(self, "eta", q16 / Q16_SCALE)


def build_score_histogram(
        required_multiplier_q16: Sequence[int],
        systematic_residuals: Sequence[float],
        candidate_multiplier_q16: Sequence[int],
        *,
        bulk_size: int,
        variance_residuals: Sequence[float] | None = None,
        incoherent_residuals: Sequence[float] | None = None,
        floor_dominated: Sequence[bool] | None = None,
) -> ScoreHistogram:
    """Bin exact per-frame Q16 decision boundaries.

    ``ALWAYS_MASKED_Q16`` is the overflow sentinel for a frame that cannot be
    kept by any deployable multiplier. A frame is kept when the selected
    multiplier is at least its required value.
    """
    bulk_size = _positive_integer(bulk_size, "bulk_size")
    candidates = _q16_tuple(
        candidate_multiplier_q16, "candidate_multiplier_q16")
    if not candidates:
        raise ValueError("candidate_multiplier_q16 must not be empty")
    if any(right <= left for left, right in zip(candidates, candidates[1:])):
        raise ValueError("candidate_multiplier_q16 must be strictly increasing")
    if isinstance(required_multiplier_q16, (str, bytes)):
        raise TypeError("required_multiplier_q16 must be a sequence of integers")
    try:
        requirements = tuple(required_multiplier_q16)
    except TypeError as exc:
        raise TypeError(
            "required_multiplier_q16 must be a sequence of integers") from exc
    if not requirements:
        raise ValueError("required_multiplier_q16 must not be empty")
    checked = []
    for value in requirements:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("required_multiplier_q16 must contain only integers")
        requirement = int(value)
        if not 1 <= requirement <= ALWAYS_MASKED_Q16:
            raise ValueError(
                "required_multiplier_q16 contains an invalid decision boundary")
        checked.append(requirement)

    systematic = np.asarray(systematic_residuals)
    variance = (None if variance_residuals is None else
                np.asarray(variance_residuals))
    incoherent = (None if incoherent_residuals is None else
                  np.asarray(incoherent_residuals))
    for values, name in ((systematic, "systematic_residuals"),
                         (variance, "variance_residuals"),
                         (incoherent, "incoherent_residuals")):
        if values is None:
            continue
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"{name} must contain only numbers")
        if np.issubdtype(values.dtype, np.complexfloating):
            raise TypeError(f"{name} must contain only real numbers")
        if values.size != len(checked):
            raise ValueError(f"{name} must match required_multiplier_q16")
        values = values.astype(float, copy=False)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{name} must be non-negative and finite")
        if name == "systematic_residuals":
            systematic = values
        elif name == "variance_residuals":
            variance = values
        else:
            incoherent = values
    dominated = None
    if floor_dominated is not None:
        dominated = np.asarray(floor_dominated, dtype=bool)
        if dominated.shape != (len(checked),):
            raise ValueError("floor_dominated must match required_multiplier_q16")

    bins = np.fromiter(
        (bisect_left(candidates, value) for value in checked),
        dtype=np.int64, count=len(checked))
    size = len(candidates) + 1
    counts = np.bincount(bins, minlength=size)
    systematic_sums = np.bincount(
        bins, weights=systematic, minlength=size)
    variance_sums = (None if variance is None else
                     np.bincount(bins, weights=variance, minlength=size))
    incoherent_sums = (None if incoherent is None else
                       np.bincount(bins, weights=incoherent, minlength=size))
    floor_counts = (None if dominated is None else
                    np.bincount(bins[dominated], minlength=size))
    return ScoreHistogram(
        bulk_size=bulk_size,
        candidate_multiplier_q16=candidates,
        counts=tuple(int(value) for value in counts),
        systematic_residual_sums=tuple(float(value)
                                       for value in systematic_sums),
        variance_residual_sums=(
            None if variance_sums is None else
            tuple(float(value) for value in variance_sums)),
        incoherent_residual_sums=(
            None if incoherent_sums is None else
            tuple(float(value) for value in incoherent_sums)),
        floor_counts=(
            None if floor_counts is None else
            tuple(int(value) for value in floor_counts)),
    )


def _same_total(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12 * scale)


def operating_surface(
        histograms_by_rho: Mapping[int, ScoreHistogram],
) -> tuple[SurfacePoint, ...]:
    """Every evaluable ``(rho, eta)`` pair of the prepared histograms, in enumeration order.

    Preparation owns frame rejection, era selection, stability, correlation,
    residual calibration, and equal-exposure frame construction. Mapping keys
    are one-based ranks. This function derives masking and retained residuals
    only from the prepared histograms. Each rank may use its own candidate
    multiplier grid. A candidate is evaluable when it keeps at least
    :data:`MIN_RETAINED_FRAMES` frames and is eligible (both calendar halves
    of the era keep enough frames); the enumeration order is rank, then
    multiplier, and is the point's ``candidate_order``.
    """
    if not isinstance(histograms_by_rho, Mapping):
        raise TypeError("histograms_by_rho must be a mapping")
    if not histograms_by_rho:
        raise ValueError("histograms_by_rho must not be empty")

    histograms = []
    for rho, histogram in histograms_by_rho.items():
        if isinstance(rho, bool) or not isinstance(rho, Integral):
            raise TypeError("rho keys must be integers")
        if int(rho) <= 0:
            raise ValueError("rho keys must be positive")
        if not isinstance(histogram, ScoreHistogram):
            raise TypeError("mapping values must be ScoreHistogram")
        histograms.append((int(rho), histogram))
    histograms.sort(key=lambda item: item[0])

    frame_count = histograms[0][1].frame_count
    bulk_size = histograms[0][1].bulk_size
    has_variance = histograms[0][1].variance_residual_sums is not None
    systematic_total = sum(histograms[0][1].systematic_residual_sums)
    variance_total = (None if not has_variance else
                      sum(histograms[0][1].variance_residual_sums))
    for _, histogram in histograms[1:]:
        if histogram.frame_count != frame_count:
            raise ValueError("all ranks must describe the same frame population")
        if histogram.bulk_size != bulk_size:
            raise ValueError("all ranks must use the same bulk size")
        if (histogram.variance_residual_sums is not None) != has_variance:
            raise ValueError("all ranks must use the same variance-residual basis")
        if not _same_total(sum(histogram.systematic_residual_sums),
                           systematic_total):
            raise ValueError("all ranks must have the same systematic residual total")
        if has_variance and not _same_total(
                sum(histogram.variance_residual_sums), variance_total):
            raise ValueError("all ranks must have the same variance residual total")

    points = []
    for rho, histogram in histograms:
        if rho > histogram.bulk_size:
            raise ValueError("rho cannot exceed the histogram bulk size")
        has_incoherent = histogram.incoherent_residual_sums is not None
        has_floor = histogram.floor_counts is not None
        kept = 0
        systematic_sum = 0.0
        variance_sum = 0.0
        incoherent_sum = 0.0
        floor_count = 0
        for index, multiplier_q16 in enumerate(
                histogram.candidate_multiplier_q16):
            kept += histogram.counts[index]
            systematic_sum += histogram.systematic_residual_sums[index]
            if has_variance:
                variance_sum += histogram.variance_residual_sums[index]
            if has_incoherent:
                incoherent_sum += histogram.incoherent_residual_sums[index]
            if has_floor:
                floor_count += histogram.floor_counts[index]
            if kept < MIN_RETAINED_FRAMES:
                continue
            if histogram.candidate_eligible is not None and not histogram.candidate_eligible[index]:
                continue
            masked = frame_count - kept
            masked_fraction = masked / frame_count
            systematic_residual = systematic_sum / kept
            variance_residual = (variance_sum / kept if has_variance else None)
            points.append(SurfacePoint(
                rho=rho,
                bulk_size=bulk_size,
                rank_fraction=rho / (bulk_size + 1),
                eta_q16=multiplier_q16,
                frame_count=frame_count,
                kept_frames=kept,
                masked_frames=masked,
                masked_fraction=masked_fraction,
                r_sys=systematic_residual,
                r_var=variance_residual,
                exposure_cost_uniform_loss=1.0 / (1.0 - masked_fraction),
                candidate_order=len(points),
                r_sys_incoherent=(incoherent_sum / kept if has_incoherent else math.nan),
                floor_share=(floor_count / kept if has_floor else math.nan),
            ))
    return tuple(points)


@dataclass(frozen=True)
class CalibrationEvidence:
    """Evidence state for one conditioning operation."""

    state: str
    method: str
    source: str
    detail: str = ""
    artifact_sha256: str | None = None
    bounds: tuple[float | None, float | None] | None = None
    units: str | None = None

    def __post_init__(self):
        allowed = {"measured", "bounded", "conditional", "unmeasured",
                   "refused"}
        if self.state not in allowed:
            raise ValueError(f"unknown calibration evidence state {self.state!r}")
        if not self.method.strip():
            raise ValueError("calibration evidence needs a method")
        if not self.source.strip():
            raise ValueError("calibration evidence needs a source")
        digest = self.artifact_sha256
        if digest is not None and (
                len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        if self.state in {"measured", "bounded"} and digest is None:
            raise ValueError(
                f"{self.state} evidence needs an artifact SHA-256 digest")
        if self.bounds is not None:
            if len(self.bounds) != 2:
                raise ValueError("bounds must contain lower and upper values")
            checked = []
            for value in self.bounds:
                if value is None:
                    checked.append(None)
                elif isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError("bounds must contain numbers or None")
                elif not math.isfinite(float(value)):
                    raise ValueError("bounds must be finite")
                else:
                    checked.append(float(value))
            if checked == [None, None]:
                raise ValueError("at least one bound must be supplied")
            if (checked[0] is not None and checked[1] is not None
                    and checked[0] > checked[1]):
                raise ValueError("lower bound cannot exceed upper bound")
            object.__setattr__(self, "bounds", tuple(checked))
        if self.state == "bounded":
            if self.bounds is None or not self.units:
                raise ValueError("bounded evidence needs bounds and units")



# ------------------------------------------------------------------ ties
# Two candidates that keep the same frames (one kept set reached at several
# ranks, or a plateau on which every kept frame sits at the band floor) have
# residuals that differ only by summation-order rounding, about 1e-15
# relative, so an exact comparison chooses among them by the last digit.
# TIE_REL_TOL = 1e-9 is well above the rounding of a mean over a million
# frames (about 1e-10 at worst) and well below any real difference between two
# kept sets.
T = TypeVar("T")

TIE_REL_TOL = 1e-9


def tied(a: float, b: float, *, rel_tol: float = TIE_REL_TOL) -> bool:
    """Whether two residuals are equal to within ``rel_tol`` of the larger magnitude."""
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=0.0)


def least(rows: Iterable[T], residual: Callable[[T], float], tie_key: Callable[[T], object], *,
          rel_tol: float = TIE_REL_TOL) -> T | None:
    """The row of least ``residual``; rows tied with it (:func:`tied`) go to the smallest ``tie_key``.

    ``None`` for no rows. Every residual must be finite; the callers filter first.
    """
    rows = list(rows)
    if not rows:
        return None
    low = min(float(residual(r)) for r in rows)
    return min((r for r in rows if tied(residual(r), low, rel_tol=rel_tol)), key=tie_key)

class TieRule(Protocol):
    """How a least-residual choice breaks ties, and the frontier built with it."""

    name: str

    def least(self, rows, residual, tie_key): ...

    def frontier(self, points, min_kept: int) -> list: ...


class SelectorOrder:
    """Residuals within :data:`TIE_REL_TOL` are tied; the tie goes to the selector's order."""

    name = "selector_order"

    def least(self, rows, residual, tie_key):
        return least(rows, residual, tie_key)

    def frontier(self, points, min_kept: int) -> list:
        return _frontier_selector_order(points, min_kept=min_kept)


SELECTOR_ORDER = SelectorOrder()


@dataclass(frozen=True)
class Floor:
    """The channel floor the residual convention falls back to on kept frames."""

    db: float
    evidence: str            # 'measured' | 'stated' | 'refused'
    population: str          # e.g. 'off era 2024-12..2026-04, 11168 frames' or 'sigma-implied substitute'

    @property
    def linear(self) -> float:
        return 10.0 ** (self.db / 10.0) if math.isfinite(self.db) else math.nan



@dataclass(frozen=True)
class Replay:
    """The selected point replayed on a frame block."""

    frames: int
    kept: int
    masked_fraction: float
    retained_residual: float          # kept-frame mean systematic residual
    survey_flag_rate: float           # occupancy indicator: fraction with F > mu_0
    masked_fraction_bootstrap: dict | None
    retained_residual_bootstrap: dict | None
    false_alarm_rate: float           # masked fraction when the block is a verified off era; NaN otherwise
    false_alarm_basis: str
    unmasked_residual: float = math.nan   # mean systematic residual over every frame of the block (keep-everything)
    retained_residual_incoherent: float = math.nan   # the kept-frame mean at G = 1
    floor_share: float = math.nan         # fraction of kept frames whose residual is the floor



def kept_at(required_q16: np.ndarray, eta_q16: int) -> np.ndarray:
    """A frame is kept when the selected multiplier is at least its requirement."""
    req = np.asarray(required_q16)
    return (req <= int(eta_q16)) & (req != ALWAYS_MASKED_Q16)


def replay(product, block: np.ndarray, *, anchor_bin: int, bulk_mask, rho: int, eta_q16: int,
           floor: Floor, off_era: bool, replicates: int, seed: int, gain: float = 1.0) -> Replay:
    """Apply a ``(rho, eta_q16)`` to another block of the same band."""
    from pilot_proxy.detectors.narrowband_marker.scores import build_score_bundle

    bundle = build_score_bundle(product.path, np.asarray(block, dtype=bool), anchor_bin=int(anchor_bin),
                                designated_half_width=DESIGNATED_HALF_WIDTH, bulk_mask=np.asarray(bulk_mask, dtype=bool))
    rows = bundle.source_row_index
    if int(rho) > bundle.supported_rho_count:
        raise ValueError(f"rank {rho} is not supported on the evaluation block (bulk {bundle.supported_rho_count})")
    required = np.asarray(bundle.requirements_by_rho()[int(rho)], dtype=object)
    kept = kept_at(required, int(eta_q16))
    residual = frame_residuals(product, rows, floor, gain)
    incoherent = frame_residuals(product, rows, floor, 1.0)
    dominated = floor_dominated(product, rows, floor)
    n = rows.size
    f = 1.0 - kept.sum() / n
    r_sys = float(residual[kept].mean()) if kept.any() else math.nan
    flag = product.rejected[rows]
    units = product.frame_unit_index[rows]
    sel = np.ones(n, dtype=bool)
    mf_bs = rr_bs = None
    if np.unique(units).size >= blocks.MIN_BLOCKS_PER_HALF:
        mf_bs = blocks.block_bootstrap(units, sel, lambda w: blocks.weighted_fraction(w, ~kept),
                                       replicates=replicates, seed=seed).as_dict()

        def kept_mean(w):
            ww = w * kept
            return float((ww * residual).sum() / ww.sum()) if ww.sum() > 0 else math.nan
        rr_bs = blocks.block_bootstrap(units, sel, kept_mean, replicates=replicates, seed=seed).as_dict()
    return Replay(frames=int(n), kept=int(kept.sum()), masked_fraction=float(f), retained_residual=r_sys,
                  survey_flag_rate=float(flag.mean()), masked_fraction_bootstrap=mf_bs, retained_residual_bootstrap=rr_bs,
                  false_alarm_rate=float(f) if off_era else math.nan,
                  false_alarm_basis="verified off era: every frame is null" if off_era else "not measurable: no verified off state in this block",
                  unmasked_residual=float(residual.mean()) if n else math.nan,
                  retained_residual_incoherent=float(incoherent[kept].mean()) if kept.any() else math.nan,
                  floor_share=float(dominated[kept].mean()) if kept.any() else math.nan)


def point_row(pt: SurfacePoint) -> dict:
    """A surface point as the flat row the frontier, the thinning and the OC table read."""
    return {"rho": int(pt.rho), "rank_fraction": float(pt.rank_fraction), "eta_q16": int(pt.eta_q16), "eta": float(pt.eta),
            "frames": int(pt.frame_count), "kept": int(pt.kept_frames), "masked_fraction": float(pt.masked_fraction),
            "r_sys": float(pt.r_sys), "r_sys_incoherent": float(pt.r_sys_incoherent), "floor_share": float(pt.floor_share),
            "exposure_cost_uniform_loss": float(pt.exposure_cost_uniform_loss), "candidate_order": int(pt.candidate_order)}


def least_residual_point(points, rule: TieRule = SELECTOR_ORDER):
    """The least-residual point of a surface (``None`` when no residual is finite).

    Under the default rule residuals within :data:`TIE_REL_TOL` of the least
    (one kept set reached at several ranks) are tied and go to the selector's
    order: the smallest masked fraction, the lowest rho, the lowest multiplier.
    """
    evaluable = [pt for pt in points if math.isfinite(pt.r_sys)]
    return rule.least(evaluable, lambda pt: pt.r_sys, lambda pt: (pt.masked_fraction, pt.rho, pt.eta_q16))


MAX_POINTS_PER_RHO = 200


def thin_points(points: Sequence[dict], selected: tuple | Sequence[tuple] | None = None, *,
                max_per_rho: int = MAX_POINTS_PER_RHO, rule: TieRule = SELECTOR_ORDER) -> list[dict]:
    """At most ``max_per_rho`` points per rank: evenly spaced in eta, always keeping the first, the last,
    the least-residual point of the rank and the marked points (``selected``: one ``(rho, eta_q16)`` pair
    or several). The full surface (about 1.3 million pairs per band) stays in memory for the summary; only
    this is written."""
    if selected is None:
        marks = []
    elif selected and isinstance(selected[0], (tuple, list)):
        marks = [tuple(int(v) for v in m) for m in selected]
    else:
        marks = [tuple(selected)]
    by_rho: dict[int, list[dict]] = {}
    for row in points:
        by_rho.setdefault(int(row["rho"]), []).append(row)
    out: list[dict] = []
    for rho in sorted(by_rho):
        rows = sorted(by_rho[rho], key=lambda r: r["eta_q16"])
        keep = set()
        n = len(rows)
        if n <= max_per_rho:
            keep = set(range(n))
        else:
            keep.update(int(round(i * (n - 1) / (max_per_rho - 1))) for i in range(max_per_rho))
        finite = [i for i, r in enumerate(rows) if math.isfinite(r["r_sys"])]
        if finite:
            keep.add(rule.least(finite, lambda i: rows[i]["r_sys"],
                                lambda i: (rows[i].get("masked_fraction", 0.0), rows[i]["eta_q16"])))
        for mark in marks:
            keep.update(i for i, r in enumerate(rows) if (r["rho"], r["eta_q16"]) == mark)
        out.extend(rows[i] for i in sorted(keep))
    return out


def surface_summary(points: Sequence[dict], rule: TieRule = SELECTOR_ORDER) -> dict:
    """The least residual on the evaluated surface and where it lies.

    Under the default rule residuals within :data:`TIE_REL_TOL` of the least
    are tied and go to the selector's order: the smallest masked fraction,
    then the lowest rho, then the lowest multiplier.
    """
    pts = [p for p in points if math.isfinite(p["r_sys"])]
    if not pts:
        return {"surface_points": len(points), "min_r_sys": math.nan, "min_r_sys_rho": None, "min_r_sys_eta": math.nan,
                "min_r_sys_masked_fraction": math.nan}
    best = rule.least(pts, lambda p: p["r_sys"], lambda p: (p["masked_fraction"], p["rho"], p["eta_q16"]))
    return {"surface_points": len(points), "min_r_sys": best["r_sys"], "min_r_sys_rho": best["rho"], "min_r_sys_eta": best["eta"],
            "min_r_sys_masked_fraction": best["masked_fraction"]}



# ------------------------------------------------------------------ frontier and knee
MARGIN = 0.10
MARGIN_SENSITIVITY = (0.05, 0.10, 0.25, 0.50)
MIN_KEPT = 30                      # the selector's own support floor: a point keeping less is not a point


@dataclass(frozen=True)
class FrontierPoint:
    rho: int
    eta_q16: int
    eta: float
    masked_fraction: float
    kept: int
    r_sys: float
    cost: float

    def as_dict(self, prefix: str) -> dict:
        return {f"{prefix}_rho": self.rho, f"{prefix}_eta_q16": self.eta_q16, f"{prefix}_eta": self.eta,
                f"{prefix}_masked_fraction": self.masked_fraction, f"{prefix}_kept": self.kept,
                f"{prefix}_r_sys": self.r_sys, f"{prefix}_cost": self.cost}


@dataclass(frozen=True)
class OperatingPoint:
    """One channel's frontier and the point chosen on it."""

    channel: int
    frontier: tuple[FrontierPoint, ...]
    r_floor: float
    margin: float
    point: FrontierPoint | None            # the rule's choice
    knee: FrontierPoint | None             # maximum curvature, as a check
    keep_everything_r: float               # r at f = 0: what the mask is measured against
    sensitivity: dict = field(default_factory=dict)   # margin -> (f, r, rho, eta_q16)
    status: str = "measured"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def suppression(self) -> float:
        """Keep-everything residual over the operating point's, the factor the mask buys."""
        if self.point is None or not (self.point.r_sys > 0 and math.isfinite(self.keep_everything_r)):
            return math.nan
        return self.keep_everything_r / self.point.r_sys

    def as_row(self) -> dict:
        row = {"channel": self.channel, "frontier_points": len(self.frontier), "r_floor": self.r_floor,
               "margin": self.margin, "keep_everything_r_sys": self.keep_everything_r,
               "suppression_factor": self.suppression,
               "suppression_db": 10.0 * math.log10(self.suppression) if self.suppression > 0 and math.isfinite(self.suppression) else math.nan,
               "status": self.status, "notes": "; ".join(self.notes)}
        row.update((self.point or _nan_point()).as_dict("operating"))
        row.update((self.knee or _nan_point()).as_dict("knee"))
        for margin, value in sorted(self.sensitivity.items()):
            tag = f"{margin:g}".replace(".", "p")
            row[f"sensitivity_{tag}_masked_fraction"] = value[0]
            row[f"sensitivity_{tag}_r_sys"] = value[1]
            row[f"sensitivity_{tag}_rho"] = value[2]
            row[f"sensitivity_{tag}_eta_q16"] = value[3]
        return row


def _nan_point() -> FrontierPoint:
    return FrontierPoint(-1, -1, math.nan, math.nan, 0, math.nan, math.nan)


def frontier(points: Sequence[dict], *, min_kept: int = MIN_KEPT, rule: TieRule = SELECTOR_ORDER) -> list[FrontierPoint]:
    """The lower envelope of ``r_sys`` against masked fraction, ascending in ``f``.

    A point keeping fewer than ``min_kept`` frames is not on the frontier: the
    selector will not choose one, and its residual is an average over too few
    frames to mean anything. The tie rule decides how equal residuals are
    represented (:class:`TieRule`).
    """
    return rule.frontier(points, min_kept)


def _frontier_selector_order(points: Sequence[dict], *, min_kept: int = MIN_KEPT) -> list[FrontierPoint]:
    """The frontier under :data:`SELECTOR_ORDER`.

    Residuals within :data:`TIE_REL_TOL` are equal: each masked fraction is
    represented by its least residual, ties to the lowest ``(rho, eta_q16)``,
    and a fraction joins the frontier only when that residual is below every
    smaller fraction's by more than the tolerance.
    """
    rows = [p for p in points if p.get("kept", 0) >= min_kept and math.isfinite(p.get("r_sys", math.nan))]
    if not rows:
        return []
    by_fraction: dict[float, list[dict]] = {}
    for p in rows:
        by_fraction.setdefault(float(p["masked_fraction"]), []).append(p)
    out: list[FrontierPoint] = []
    best = math.inf
    for fraction in sorted(by_fraction):
        p = least(by_fraction[fraction], lambda q: q["r_sys"], lambda q: (int(q["rho"]), int(q["eta_q16"])))
        if p["r_sys"] < best and not tied(p["r_sys"], best):
            best = p["r_sys"]
            out.append(FrontierPoint(int(p["rho"]), int(p["eta_q16"]), float(p["eta"]), float(p["masked_fraction"]),
                                     int(p["kept"]), float(p["r_sys"]), float(p.get("exposure_cost_uniform_loss", math.nan))))
    return out


def _at_margin(frontier: Sequence[FrontierPoint], r_floor: float, margin: float) -> FrontierPoint | None:
    """The smallest-``f`` frontier point within ``margin`` of the floor."""
    target = (1.0 + margin) * r_floor
    return next((p for p in frontier if p.r_sys <= target), None)


def _knee(frontier: Sequence[FrontierPoint]) -> FrontierPoint | None:
    """The frontier's point of greatest curvature, on normalised log axes.

    Cost and residual are made commensurate by mapping each to [0, 1] over the
    frontier's own span in log space, so the knee is where the corner is and
    not where the units happen to be.
    """
    if len(frontier) < 3:
        return None
    cost = np.array([1.0 / (1.0 - p.masked_fraction) if p.masked_fraction < 1.0 else np.inf for p in frontier])
    r = np.array([p.r_sys for p in frontier])
    ok = np.isfinite(cost) & np.isfinite(r) & (cost > 0) & (r > 0)
    if ok.sum() < 3:
        return None
    x, y = np.log10(cost[ok]), np.log10(r[ok])
    x = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0
    y = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0
    # the utopia corner is (0, 0): the knee is the frontier point closest to it
    index = int(np.argmin(x ** 2 + y ** 2))
    return [p for p, keep in zip(frontier, ok) if keep][index]


def knee(channel: int, points: Sequence[dict], *, margin: float = MARGIN,
         sensitivity: Sequence[float] = MARGIN_SENSITIVITY, min_kept: int = MIN_KEPT,
         rule: TieRule = SELECTOR_ORDER) -> OperatingPoint:
    """The knee of one band's evaluated surface, with the margin points beside it."""
    front = rule.frontier(points, min_kept)
    notes: list[str] = []
    if not front:
        return OperatingPoint(channel, (), math.nan, margin, None, None, math.nan, {}, "no frontier",
                              (f"no evaluated point keeps {min_kept} frames with a finite residual",))
    r_floor = min(p.r_sys for p in front)
    keep_all = next((p.r_sys for p in front if p.masked_fraction <= 0.0), math.nan)
    if not math.isfinite(keep_all):
        keep_all = max(p.r_sys for p in front)
        notes.append("no point masks nothing: the reference residual is the frontier's largest")
    corner = _knee(front)
    sens = {}
    for m in sensitivity:
        p = _at_margin(front, r_floor, m)
        sens[m] = (p.masked_fraction, p.r_sys, p.rho, p.eta_q16) if p else (math.nan, math.nan, -1, -1)
    if corner is None:
        notes.append("the frontier has too few points to have a knee")
    elif corner.masked_fraction >= 0.99:
        notes.append(f"the knee itself masks {corner.masked_fraction:.3f}: this channel has no cheap mask, "
                     "and the point is reported for completeness rather than as an operating instruction")
    return OperatingPoint(channel, tuple(front), r_floor, margin, corner, corner, keep_all, sens,
                          "measured" if corner else "undefined", tuple(notes))


OPERATING_COLUMNS = ("channel", "frontier_points", "r_floor", "margin", "keep_everything_r_sys",
                     "suppression_factor", "suppression_db", "operating_rho", "operating_eta_q16", "operating_eta",
                     "operating_masked_fraction", "operating_kept", "operating_r_sys", "operating_cost",
                     "knee_rho", "knee_eta_q16", "knee_eta", "knee_masked_fraction", "knee_kept", "knee_r_sys",
                     "knee_cost", "status", "notes")


def write_knee(results: Sequence[OperatingPoint], path: Path | str) -> Path:
    """One row per band: the knee and what it buys (the ``operating_point.csv`` layout)."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.as_row() for r in results]
    keys = list(OPERATING_COLUMNS) + [k for r in rows for k in r if k not in OPERATING_COLUMNS]
    seen, fields = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (repr(v) if isinstance(v, float) else row.get(k, "")) for k, v in
                             ((k, row.get(k, "")) for k in fields)})
    return path


# ------------------------------------------------------------------ the coarse rule
COARSE_ETA_GRID = tuple(np.round(np.arange(1.0, 1.5001, 0.005), 3)) + (1.6, 1.8, 2.0, 3.0, 5.0, 10.0, 100.0)


def coarse_surface(product, block, floor: Floor, gain: float) -> list[dict]:
    """The coarse rule Q > eta_c on a block: masked fraction and floor-bounded residual per eta_c.

    Every grid value is listed; ``evaluable`` marks the rows that keep at least
    :data:`MIN_RETAINED_FRAMES` frames.
    """
    rows = np.flatnonzero(np.asarray(block, dtype=bool))
    if rows.size == 0 or not math.isfinite(floor.linear):
        return []
    q = product.statistic[rows]
    residual = frame_residuals(product, rows, floor, gain)
    incoherent = frame_residuals(product, rows, floor, 1.0)
    dominated = floor_dominated(product, rows, floor)
    out = []
    for eta in COARSE_ETA_GRID:
        kept = q <= float(eta)
        n_kept = int(kept.sum())
        r = float(residual[kept].mean()) if n_kept else math.nan
        out.append({"eta_c": float(eta), "frames": int(rows.size), "kept": n_kept, "masked_fraction": 1.0 - n_kept / rows.size,
                    "r_sys": r, "r_sys_incoherent": float(incoherent[kept].mean()) if n_kept else math.nan,
                    "floor_share": float(dominated[kept].mean()) if n_kept else math.nan,
                    "evaluable": n_kept >= MIN_RETAINED_FRAMES})
    return out


def coarse_least(frontier_rows: list[dict], rule: TieRule = SELECTOR_ORDER) -> dict:
    """The coarse rule's least residual and where it lies.

    On a plateau (every kept frame at the floor, so ``r_sys`` is the same over
    a range of ``eta_c`` up to rounding) the rows tied with the least go, under
    the default rule, to the one masking least (then the lowest ``eta_c``).
    """
    ok = [f for f in frontier_rows if f["evaluable"] and math.isfinite(f["r_sys"])]
    if not ok:
        return {"coarse_min_r_sys": math.nan, "coarse_min_r_sys_eta": math.nan,
                "coarse_min_r_sys_masked_fraction": math.nan, "coarse_r_sys_at_flag": math.nan}
    best = rule.least(ok, lambda f: f["r_sys"], lambda f: (f["masked_fraction"], f["eta_c"]))
    at_flag = next((f for f in frontier_rows if f["eta_c"] == 1.0), None)
    return {"coarse_min_r_sys": best["r_sys"], "coarse_min_r_sys_eta": best["eta_c"],
            "coarse_min_r_sys_masked_fraction": best["masked_fraction"],
            "coarse_r_sys_at_flag": at_flag["r_sys"] if at_flag else math.nan}


# ------------------------------------------------------------------ one block, end to end
def _evidence(state: str, method: str, source: str, artifact_sha256: str | None = None, detail: str = "") -> CalibrationEvidence:
    return CalibrationEvidence(state=state, method=method, source=source, detail=detail,
                               artifact_sha256=artifact_sha256 if state in ("measured", "bounded") else None)


LEAST_RESIDUAL_BASIS = "least residual on the calibration surface"


@dataclass(frozen=True)
class BlockCharacterization:
    """One band's calibration-block surface, its notable points, and the least-residual replay.

    ``status`` is ``characterized`` (a surface with evaluable points),
    ``no evaluable point``, or ``refused`` with the reason in ``refusal`` (the
    block has no usable frames, no floor for frames without a shelf estimate,
    the score bundle refused, or the preparation refused). A family whose
    within-era stability screen does not pass is still characterized: its
    ``claim_status`` is ``diagnostic`` and the screen's refusal is in
    ``refusal`` (a detector refusal). Nothing here reads a tolerance.
    """

    channel: int
    freq_id: int
    era_label: str
    anchor_bin: int
    bulk_size: int
    floor: Floor
    calibration_frames: int
    evaluation_frames: int
    frames_without_time: int
    status: str
    refusal: str = ""
    claim_status: str = ""
    points: tuple = ()                    # every evaluated (rho, eta) pair of the calibration surface, as rows
    least_residual: dict = field(default_factory=dict)   # the least-residual point, with its basis
    evaluation: Replay | None = None      # the least-residual point replayed on the evaluation block
    unmasked_residual: float = math.nan   # calibration block, keep-everything residual
    unmasked_residual_incoherent: float = math.nan
    stability: dict = field(default_factory=dict)
    provisional: dict = field(default_factory=dict)
    candidates_by_rho: dict = field(default_factory=dict)   # rho -> number of candidate multipliers
    source_id: str = ""
    policy_sha256: str = ""
    anchor_sentinel: str = ""

    def as_row(self) -> dict:
        """The detector columns of the block, flat (the ledger's ``surface`` section)."""
        row = {
            "channel": self.channel, "freq_id": self.freq_id, "era": self.era_label,
            "anchor_bin": self.anchor_bin, "bulk_size": self.bulk_size,
            "floor_db": self.floor.db, "floor_evidence": self.floor.evidence, "floor_population": self.floor.population,
            "calibration_frames": self.calibration_frames, "evaluation_frames": self.evaluation_frames,
            "frames_without_time": self.frames_without_time,
            "status": self.status, "refusal": self.refusal, "claim_status": self.claim_status,
        }
        d = self.least_residual
        row.update({
            "least_residual_basis": d.get("basis", ""), "least_residual_rho": d.get("rho"),
            "least_residual_eta_q16": d.get("eta_q16"), "least_residual_eta": d.get("eta", math.nan),
            "least_residual_masked_fraction": d.get("masked_fraction", math.nan),
            "least_residual_r_sys": d.get("r_sys", math.nan),
            "least_residual_r_sys_incoherent": d.get("r_sys_incoherent", math.nan),
            "least_residual_floor_share": d.get("floor_share", math.nan),
            "least_residual_exposure_cost_uniform_loss": d.get("exposure_cost_uniform_loss", math.nan),
            "anchor_sentinel": self.anchor_sentinel,
            "stability_status": self.stability.get("status", ""), "stability_reason": self.stability.get("reason", ""),
            "stability_points_checked": self.stability.get("points_checked"),
            "stability_points_skipped": self.stability.get("points_skipped"),
            "chain_gain": self.provisional.get("chain_gain", math.nan),
            "residual_convention": self.provisional.get("residual_convention", ""),
            "r_sys_unmasked_calibration": self.unmasked_residual,
            "r_sys_incoherent_unmasked_calibration": self.unmasked_residual_incoherent,
        })
        ev = self.evaluation
        mf = (ev.masked_fraction_bootstrap or {}) if ev else {}
        rr = (ev.retained_residual_bootstrap or {}) if ev else {}
        row.update({
            "masked_fraction_evaluation": ev.masked_fraction if ev else math.nan,
            "masked_fraction_evaluation_q16": mf.get("q0.16", math.nan), "masked_fraction_evaluation_q84": mf.get("q0.84", math.nan),
            "r_sys_evaluation": ev.retained_residual if ev else math.nan,
            "r_sys_evaluation_q16": rr.get("q0.16", math.nan), "r_sys_evaluation_q84": rr.get("q0.84", math.nan),
            "bootstrap_blocks_evaluation": mf.get("blocks", 0),
            "r_sys_unmasked_evaluation": ev.unmasked_residual if ev else math.nan,
            "r_sys_incoherent_evaluation": ev.retained_residual_incoherent if ev else math.nan,
            "floor_share_evaluation": ev.floor_share if ev else math.nan,
            "kept_evaluation": ev.kept if ev else 0,
            "survey_flag_rate_evaluation": ev.survey_flag_rate if ev else math.nan,
            "false_alarm_rate": ev.false_alarm_rate if ev else math.nan,
            "false_alarm_basis": ev.false_alarm_basis if ev else "",
        })
        return row


def characterize_block(product, calibration: np.ndarray, evaluation: np.ndarray, *,
                       anchor_bin: int, bulk_mask: np.ndarray, floor: Floor, era_label: str,
                       gain: float = 1.0, latest_era: bool = True, off_era: bool = False,
                       correlation: CalibrationEvidence | None = None,
                       transfer: CalibrationEvidence | None = None,
                       min_half_retained: int | None = None,
                       max_cost_ratio: float | None = None,
                       max_systematic_ratio: float | None = None,
                       minimum_observed_months: int | None = None, minimum_span_days: float | None = None,
                       bootstrap_replicates: int = blocks.DEFAULT_REPLICATES,
                       bootstrap_seed: int = blocks.DEFAULT_SEED,
                       rule: TieRule = SELECTOR_ORDER) -> BlockCharacterization:
    """The operating surface on the calibration block; the least-residual point replayed on the evaluation block.

    This is the detector half of the release's operating-point selection
    (``select_operating_point``): the same bundle, residuals, prepared family
    and enumeration, with the within-era stability screen applied at the
    screening level; the feasibility against a tolerance and the choice inside
    the deployable set are the science side's.
    """
    from pilot_proxy.detectors.narrowband_marker.scores import ScoreRefused, build_score_bundle
    from pilot_proxy.products.reader import sha256_of

    from .stability import (PROVISIONAL_MAX_COST_RATIO, PROVISIONAL_MAX_SYSTEMATIC_RATIO,
                            PROVISIONAL_MIN_HALF_RETAINED)

    min_half_retained = PROVISIONAL_MIN_HALF_RETAINED if min_half_retained is None else min_half_retained
    max_cost_ratio = PROVISIONAL_MAX_COST_RATIO if max_cost_ratio is None else max_cost_ratio
    max_systematic_ratio = PROVISIONAL_MAX_SYSTEMATIC_RATIO if max_systematic_ratio is None else max_systematic_ratio
    cal = np.asarray(calibration, dtype=bool) & product.selected
    eva = np.asarray(evaluation, dtype=bool) & product.selected
    timed = np.isfinite(product.frame_time)
    without_time = int(((cal | eva) & ~timed).sum())
    cal &= timed
    eva &= timed
    product_sha = sha256_of(product.path)
    provisional = {"stability.minimum_half_retained_frames": min_half_retained,
                   "stability.maximum_cost_ratio": max_cost_ratio,
                   "stability.maximum_systematic_residual_ratio": max_systematic_ratio,
                   "residual_convention": "floor-bounded shelf linear x chain gain, variance 0", "chain_gain": float(gain)}
    provisional.update({
        "residual_evidence": "conditional screening allowance; not measured contamination",
        "floor_is_physical_lower_bound": False,
        "floor_only_r_sys": floor.linear * gain,
    })
    base = dict(channel=product.geometry.physical_channel, freq_id=product.geometry.freq_id, era_label=era_label,
                anchor_bin=int(anchor_bin), bulk_size=int(np.asarray(bulk_mask, dtype=bool).sum()),
                floor=floor, calibration_frames=int(cal.sum()), evaluation_frames=int(eva.sum()),
                frames_without_time=without_time, provisional=provisional)
    if not cal.any():
        return BlockCharacterization(status="refused", refusal="calibration block has no usable frames", **base)
    if not math.isfinite(floor.linear) and not np.isfinite(product.shelf_db[cal]).all():
        return BlockCharacterization(status="refused", refusal="no floor for frames without a shelf estimate", **base)
    try:
        bundle = build_score_bundle(product.path, cal, anchor_bin=int(anchor_bin),
                                    designated_half_width=DESIGNATED_HALF_WIDTH, bulk_mask=np.asarray(bulk_mask, dtype=bool))
    except ScoreRefused as exc:
        return BlockCharacterization(status="refused", refusal=f"bundle: {exc}", **base)
    rows = bundle.source_row_index
    residuals = frame_residuals(product, rows, floor, gain)
    incoherent = frame_residuals(product, rows, floor, 1.0)
    dominated = floor_dominated(product, rows, floor)
    base["unmasked_residual"] = float(residuals.mean()) if residuals.size else math.nan
    base["unmasked_residual_incoherent"] = float(incoherent.mean()) if incoherent.size else math.nan
    score = _evidence("measured", "exact fine-power terms", product.path.name, product_sha,
                      "Q16 requirements from fine_power_u64")
    correlation = correlation or _evidence("conditional", "correlation time pending", "chapter 8 tau_c estimator not yet run on v5")
    transfer = transfer or _evidence("conditional", "unity transfer closure", "g_var = g_sys = 1 (chapter 9)")
    try:
        family = bundle.prepare_threshold_family(
            residuals, variance_residuals=np.zeros(bundle.frame_count), era_label=era_label, latest_era=latest_era,
            additive_residuals=True, score=score, correlation=correlation, transfer=transfer,
            max_cost_ratio=max_cost_ratio, max_systematic_residual_ratio=max_systematic_ratio,
            minimum_half_retained_frames=min_half_retained,
            minimum_observed_months=minimum_observed_months, minimum_span_days=minimum_span_days,
            incoherent_residuals=incoherent, floor_dominated=dominated)
    except (ValueError, TypeError) as exc:
        return BlockCharacterization(status="refused", refusal=f"preparation: {exc}", **base)
    st = family.stability
    stability = {"status": st.status, "reason": st.reason, "points_checked": st.points_checked,
                 "points_skipped": st.points_skipped, "maximum_cost_ratio": st.maximum_cost_ratio,
                 "maximum_systematic_residual_ratio": st.maximum_systematic_residual_ratio}
    reasons = family.refusals(allow_screening=True)
    surface = operating_surface(family.histograms_by_rho)
    points = tuple(point_row(pt) for pt in surface)
    least = least_residual_point(surface, rule)
    least_row, evaluation_replay = {}, None
    if least is not None:
        least_row = {"basis": LEAST_RESIDUAL_BASIS, **point_row(least)}
        if eva.any():
            evaluation_replay = replay(product, eva, anchor_bin=int(anchor_bin), bulk_mask=bulk_mask, rho=least.rho,
                                       eta_q16=least.eta_q16, floor=floor, gain=gain, off_era=off_era,
                                       replicates=bootstrap_replicates, seed=bootstrap_seed)
    return BlockCharacterization(
        status="characterized" if points else "no evaluable point", refusal="; ".join(reasons),
        claim_status="diagnostic" if reasons else "screening", points=points, least_residual=least_row,
        evaluation=evaluation_replay, stability=stability,
        candidates_by_rho={int(rho): len(h.candidate_multiplier_q16) for rho, h in family.histograms_by_rho.items()},
        source_id=bundle.source_id, policy_sha256=family.policy_sha256, **base)


__all__ = [
    "ALWAYS_MASKED_Q16", "BlockCharacterization", "COARSE_ETA_GRID", "CalibrationEvidence", "DESIGNATED_HALF_WIDTH",
    "Floor", "FrontierPoint", "LEAST_RESIDUAL_BASIS", "MARGIN", "MARGIN_SENSITIVITY", "MAX_MULTIPLIER_Q16",
    "MAX_POINTS_PER_RHO", "MIN_KEPT", "MIN_RETAINED_FRAMES", "OPERATING_COLUMNS", "OperatingPoint",
    "Q16_FRACTION_BITS", "Q16_SCALE", "Replay", "SELECTOR_ORDER", "ScoreHistogram", "SelectorOrder",
    "SurfacePoint", "TIE_REL_TOL", "TieRule", "build_score_histogram", "characterize_block", "coarse_least",
    "coarse_surface", "frontier", "kept_at", "knee", "least", "least_residual_point", "operating_surface",
    "point_row", "replay", "surface_summary", "thin_points", "tied", "write_knee",
]
