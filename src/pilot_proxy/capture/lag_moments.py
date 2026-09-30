"""Support-matched descriptive temporal moments, without circular wrapping.

Positive lag is late * conjugate(early). Inputs are contiguous native time
series; call once on the complete interval to retain storage-block crossings.
No confidence interval, stationarity, calibration, or effective sample count
is implied by the returned population moments.
"""
from __future__ import annotations

import numpy as np


def _samples_and_support(samples, support):
    x = np.asarray(samples)
    if x.ndim != 2 or x.shape[1] == 0 or not np.iscomplexobj(x):
        raise ValueError("samples must be complex [time,series]")
    valid = np.isfinite(x)
    if support is not None:
        support = np.asarray(support)
        if support.dtype != np.bool_ or support.shape != x.shape:
            raise ValueError("support must be boolean with sample shape")
        valid &= support
    return x.astype(np.complex128), valid


def lag_moments(samples, lags, support=None):
    """Return additive sums on each series' matched early/late support.

    Arrays have axes [lag,series]. Support excludes nonfinite values. No
    temporal compression, circular wrap, or averaging across blocks occurs.
    Centering uses separate means on the two overlapping windows at each lag.
    """
    x, valid = _samples_and_support(samples, support)
    lags = np.asarray(lags)
    if (lags.ndim != 1 or not len(lags) or lags.dtype.kind not in "iu"
            or np.any(lags < 0) or np.any(lags >= len(x))
            or len(np.unique(lags)) != len(lags)):
        raise ValueError("lags must be distinct integers in [0, samples)")
    result = {key: [] for key in
              ("count", "sum_late", "sum_early", "power_late", "power_early", "cross")}
    for lag in lags:
        n = len(x) - int(lag)
        common = valid[lag:] & valid[:n]
        late = np.where(common, x[lag:], 0)
        early = np.where(common, x[:n], 0)
        result["count"].append(np.sum(common, axis=0, dtype=np.int64))
        result["sum_late"].append(np.sum(late, axis=0))
        result["sum_early"].append(np.sum(early, axis=0))
        result["power_late"].append(np.sum(late.real**2 + late.imag**2, axis=0))
        result["power_early"].append(np.sum(early.real**2 + early.imag**2, axis=0))
        result["cross"].append(np.sum(late * early.conj(), axis=0))
    return {key: np.stack(value) for key, value in result.items()}


def summarize_lags(moments):
    """Population autocorrelation and normalized centered/uncentered forms.

    Native units are squared units of the supplied series. Supplying complex
    voltage cross-products therefore returns fourth-power voltage-code units.
    Undefined normalized values are NaN, including centered constant series.
    """
    n = np.asarray(moments["count"])
    keys = ("sum_late", "sum_early", "power_late", "power_early", "cross")
    sl, se, pl, pe, cross = [np.asarray(moments[k]) for k in keys]
    if (n.ndim != 2 or n.dtype.kind not in "iu" or np.any(n < 0)
            or any(a.shape != n.shape for a in (sl, se, pl, pe, cross))
            or any(not np.isfinite(a).all() for a in (sl, se, pl, pe, cross))
            or np.any(pl < 0) or np.any(pe < 0)):
        raise ValueError("invalid lag moments")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom_n = np.where(n > 0, n, np.nan)
        ml, me = sl / denom_n, se / denom_n
        raw = cross / denom_n
        centered = raw - ml * me.conj()
        vl, ve = pl / denom_n - np.abs(ml)**2, pe / denom_n - np.abs(me)**2
        for variance, power in ((vl, pl), (ve, pe)):
            if np.any(variance < -64*np.finfo(float).eps*np.maximum(power/denom_n, 1)):
                raise ValueError("negative centered power")
        vl, ve = np.maximum(vl, 0), np.maximum(ve, 0)
        d_raw, d_centered = np.sqrt(pl * pe), np.sqrt(vl * ve)
        return {"mean_late": ml, "mean_early": me,
                "raw": raw, "centered": centered,
                "variance_late": vl, "variance_early": ve,
                "normalized_raw": np.where(d_raw > 0, cross/d_raw, np.nan+1j*np.nan),
                "normalized_centered": np.where(d_centered > 0, centered/d_centered,
                                                 np.nan+1j*np.nan)}


def frame_moments(samples, frame_samples, support=None):
    """Per-series frame counts/sums/powers including a separately marked tail.

    Frame index zero starts at the supplied interval's first sample; no claim
    of alignment with an observatory's operational frame boundaries is made.
    """
    x, valid = _samples_and_support(samples, support)
    if isinstance(frame_samples, bool) or not isinstance(frame_samples, (int, np.integer)) or frame_samples <= 0:
        raise ValueError("frame_samples must be a positive integer")
    blocks = np.array([(start, min(start+frame_samples, len(x)))
                       for start in range(0, len(x), frame_samples)], dtype=np.int64)
    if len(blocks) == 0:
        raise ValueError("at least one sample is required")
    result = {key: [] for key in ("count", "sum", "power")}
    for start, stop in blocks:
        values = np.where(valid[start:stop], x[start:stop], 0)
        result["count"].append(valid[start:stop].sum(axis=0, dtype=np.int64))
        result["sum"].append(values.sum(axis=0))
        result["power"].append((values.real**2+values.imag**2).sum(axis=0))
    return {"blocks": blocks, "complete": np.diff(blocks, axis=1).ravel() == frame_samples,
            **{key: np.stack(value) for key, value in result.items()}}
