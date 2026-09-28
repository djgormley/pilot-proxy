"""Calibration/evaluation blocks and the acquisition block bootstrap.

Consecutive 42 ms frames of one acquisition are strongly correlated and the
archive has diurnal and seasonal structure, so nothing here ever resamples
or splits frames individually. The unit is the acquisition (the product's
``frame_unit_index``): a current era is split into a calibration block and an
evaluation block on whole acquisitions in chronological order, at the
acquisition that balances the frame counts, so that calibration precedes
evaluation the way offline calibration precedes online use; and every
uncertainty is a bootstrap that resamples whole acquisitions with a recorded
seed.

Month support follows the dissertation's section 8.1 aggregation rule: a
UTC calendar month is populated only with at least 30 valid frames drawn
from at least 5 distinct acquisitions on at least 3 distinct days (policy
values; 3/5/10 acquisitions are the registered sensitivity values). Each
block must satisfy the era's month minimums on its own or the selector
reports insufficient support instead of thinning the split.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

MONTH_MIN_FRAMES = 30
MONTH_MIN_UNITS = 5
MONTH_MIN_DAYS = 3
MONTH_MIN_UNITS_SENSITIVITY = (3, 5, 10)

DEFAULT_REPLICATES = 1000
DEFAULT_SEED = 20260907
DEFAULT_QUANTILES = (0.16, 0.84)
WIDE_QUANTILES = (0.025, 0.975)
MIN_BLOCKS_PER_HALF = 8


def month_index(unix_seconds) -> np.ndarray:
    """UTC calendar month as ``year * 12 + (month - 1)``; -1 where the time is NaN."""
    t = np.asarray(unix_seconds, dtype=float)
    out = np.full(t.shape, -1, dtype=np.int64)
    for i in np.flatnonzero(np.isfinite(t)):
        d = dt.datetime.fromtimestamp(float(t[i]), dt.timezone.utc)
        out[i] = d.year * 12 + d.month - 1
    return out


def month_label(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def day_index(unix_seconds) -> np.ndarray:
    """UTC day number; -1 where NaN."""
    t = np.asarray(unix_seconds, dtype=float)
    out = np.full(t.shape, -1, dtype=np.int64)
    ok = np.isfinite(t)
    out[ok] = np.floor(t[ok] / 86400.0).astype(np.int64)
    return out


@dataclass(frozen=True)
class MonthSupport:
    """Per populated month: frames, acquisitions, days."""

    month: int
    frames: int
    units: int
    days: int

    @property
    def label(self) -> str:
        return month_label(self.month)


def month_support(frame_time, unit_index, selected, *, min_frames: int = MONTH_MIN_FRAMES,
                  min_units: int = MONTH_MIN_UNITS, min_days: int = MONTH_MIN_DAYS) -> list[MonthSupport]:
    """Populated months among the selected frames, in calendar order.

    Frames whose time is unrecorded (NaN) cannot be placed in a month and are
    left out; callers report the reduced denominator.
    """
    t = np.asarray(frame_time, dtype=float)
    u = np.asarray(unit_index)
    sel = np.asarray(selected, dtype=bool) & np.isfinite(t)
    months = month_index(t[sel])
    units = u[sel]
    days = day_index(t[sel])
    out = []
    for m in np.unique(months):
        here = months == m
        n_frames = int(here.sum())
        n_units = int(np.unique(units[here]).size)
        n_days = int(np.unique(days[here]).size)
        if n_frames >= min_frames and n_units >= min_units and n_days >= min_days:
            out.append(MonthSupport(int(m), n_frames, n_units, n_days))
    return out


@dataclass(frozen=True)
class BlockSplit:
    """A chronological split of selected frames into two blocks of whole acquisitions."""

    calibration: np.ndarray      # bool (N,): frames in the calibration block
    evaluation: np.ndarray       # bool (N,): frames in the evaluation block
    boundary_time: float         # unit time of the first evaluation acquisition
    calibration_units: int
    evaluation_units: int
    calibration_months: tuple[MonthSupport, ...]
    evaluation_months: tuple[MonthSupport, ...]
    minimum_months: int
    status: str                  # 'supported' | 'insufficient_support' | 'empty'
    detail: str = ""

    @property
    def calibration_frames(self) -> int:
        return int(self.calibration.sum())

    @property
    def evaluation_frames(self) -> int:
        return int(self.evaluation.sum())


def split_blocks(unit_index, unit_time, selected, *, frame_time=None, minimum_months: int = 1,
                 month_kwargs: dict | None = None) -> BlockSplit:
    """Split selected frames into calibration and evaluation blocks.

    Acquisitions are ordered by their first-sample time and assigned whole to
    the calibration block until the cumulative frame count reaches half of
    the selected frames; the remainder is the evaluation block. ``frame_time``
    (default: the unit time of each frame) is used only to count populated
    months per block; a block with fewer than ``minimum_months`` populated
    months makes the split ``insufficient_support``.
    """
    u = np.asarray(unit_index)
    sel = np.asarray(selected, dtype=bool)
    ut = np.asarray(unit_time, dtype=float)
    ft = ut if frame_time is None else np.asarray(frame_time, dtype=float)
    n = sel.size
    empty = np.zeros(n, dtype=bool)
    if not sel.any():
        return BlockSplit(empty, empty.copy(), float("nan"), 0, 0, (), (), minimum_months, "empty", "no selected frames")
    units, inverse = np.unique(u[sel], return_inverse=True)
    frames_per_unit = np.bincount(inverse, minlength=units.size)
    first_time = np.full(units.size, np.inf)
    np.minimum.at(first_time, inverse, ut[sel])
    order = np.argsort(first_time, kind="stable")
    cumulative = np.cumsum(frames_per_unit[order])
    half = cumulative[-1] / 2.0
    # the acquisition that carries the cumulative count across half joins the
    # calibration block; the evaluation block always keeps at least one unit
    cut = int(np.searchsorted(cumulative, half, side="left"))
    cut = min(cut, units.size - 2) if units.size > 1 else -1
    cal_units = set(units[order[: cut + 1]].tolist())
    in_cal_unit = np.isin(u, list(cal_units))
    calibration = sel & in_cal_unit
    evaluation = sel & ~in_cal_unit
    boundary = float(np.nanmin(ut[evaluation])) if evaluation.any() else float("nan")
    kw = dict(month_kwargs or {})
    cal_months = tuple(month_support(ft, u, calibration, **kw))
    eva_months = tuple(month_support(ft, u, evaluation, **kw))
    status = "supported"
    detail = ""
    if len(cal_months) < minimum_months or len(eva_months) < minimum_months:
        status = "insufficient_support"
        detail = (f"calibration block has {len(cal_months)} populated months, evaluation block "
                  f"{len(eva_months)}; the era procedure needs {minimum_months} in each")
    return BlockSplit(calibration, evaluation, boundary, len(cal_units), int(units.size - len(cal_units)),
                      cal_months, eva_months, minimum_months, status, detail)


@dataclass(frozen=True)
class Bootstrap:
    """A block-bootstrap estimate: the point value on the full block and its quantiles."""

    estimate: float
    quantiles: tuple[float, ...]
    values: tuple[float, ...]            # the statistic at each quantile
    replicates: int
    seed: int
    blocks: int
    samples: np.ndarray = field(repr=False)   # the replicate statistics

    @property
    def low(self) -> float:
        return self.values[0]

    @property
    def high(self) -> float:
        return self.values[-1]

    def as_dict(self) -> dict:
        return {"estimate": self.estimate, "replicates": self.replicates, "seed": self.seed, "blocks": self.blocks,
                **{f"q{q:g}": v for q, v in zip(self.quantiles, self.values)}}


def block_bootstrap(unit_index, selected, statistic: Callable[[np.ndarray], float], *,
                    replicates: int = DEFAULT_REPLICATES, seed: int = DEFAULT_SEED,
                    quantiles: Sequence[float] = DEFAULT_QUANTILES, min_blocks: int = MIN_BLOCKS_PER_HALF) -> Bootstrap:
    """Resample whole acquisitions with replacement and re-evaluate a statistic.

    ``statistic`` receives an integer weight per frame (how many times the
    frame's acquisition was drawn; zero for frames outside the selection) and
    returns a float; the point estimate uses unit weights on the selection.
    Weighted statistics keep every replicate a single vectorised pass, so a
    1000-replicate bootstrap over 40k frames costs one pass per replicate,
    not a copy of the product.
    """
    u = np.asarray(unit_index)
    sel = np.asarray(selected, dtype=bool)
    units, inverse = np.unique(u[sel], return_inverse=True)
    if units.size < min_blocks:
        raise ValueError(f"block bootstrap needs at least {min_blocks} acquisitions; got {units.size}")
    base = np.zeros(u.shape, dtype=np.int64)
    base[sel] = 1
    estimate = float(statistic(base))
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=float)
    sel_idx = np.flatnonzero(sel)
    for r in range(int(replicates)):
        draw = rng.integers(0, units.size, size=units.size)
        counts = np.bincount(draw, minlength=units.size)
        weights = np.zeros(u.shape, dtype=np.int64)
        weights[sel_idx] = counts[inverse]
        samples[r] = statistic(weights)
    finite = samples[np.isfinite(samples)]
    values = tuple(float(v) for v in np.quantile(finite, list(quantiles))) if finite.size else tuple(float("nan") for _ in quantiles)
    return Bootstrap(estimate, tuple(float(q) for q in quantiles), values, int(replicates), int(seed), int(units.size), samples)


def weighted_fraction(weights: np.ndarray, flag: np.ndarray) -> float:
    """Fraction of (weighted) frames with ``flag`` set; NaN when no weight."""
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    return float((w * np.asarray(flag, dtype=float)).sum() / total) if total > 0 else float("nan")


def weighted_quantile(weights: np.ndarray, values: np.ndarray, q: float) -> float:
    """Quantile of ``values`` under integer frame weights (NaN values ignored)."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = (w > 0) & np.isfinite(v)
    if not ok.any():
        return float("nan")
    order = np.argsort(v[ok], kind="stable")
    vs, ws = v[ok][order], w[ok][order]
    cdf = np.cumsum(ws) / ws.sum()
    return float(vs[int(np.searchsorted(cdf, q, side="left").clip(0, vs.size - 1))])
