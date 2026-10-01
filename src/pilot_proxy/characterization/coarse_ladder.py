"""The coarse retention ladder: cal_q rungs of the coarse statistic over a declared population.

A rung keeps the frames whose coarse statistic ``Q = F / mu_0`` is at or
below a threshold read off the population itself: ``cal_q0.5`` keeps the
frames at or below the population's median of Q (the "higher" quantile, an
observed value, so the rung is deployable), ``cal_q0.1`` and ``cal_q0.9`` the
other declared quantiles; ``keep_all`` keeps every frame. The threshold is
replayed exactly on the product's integer powers
(``ProductView.rejected_at_multiplier``), not re-estimated, and each rung is
reported on the calibration and evaluation blocks and on the calibration
block's two calendar halves.

``population`` names the frames the thresholds are read from:
``current_era_calibration_block`` (the archive: the current era's calibration
block, as the release's coarse retention diagnostic read it) or
``all_valid_frames`` (every valid, finite frame of the band's product: the
ladder a capture dump is placed on, :func:`product_thresholds`). The residual of
a rung is the kept-frame mean of the floor-bounded shelf at G = 1
(``G1_allowance``) and that mean times the chain gain (``chain_allowance``);
a missing floor stays missing. Pricing against a tolerance is the science
side's.

The rule and its columns reproduce ``coarse_retention_diagnostic_v1.py`` of
``results/archive_author_eras_2026-09-23`` (the frozen tool that wrote
``coarse/tradeoffs.csv``), whose detector half this is.
"""
from __future__ import annotations

import math

import numpy as np

from . import blocks
from .residual_chain import frame_residuals

QUANTILES = (0.1, 0.5, 0.9)
MINIMUM = 30
POPULATIONS = ("current_era_calibration_block", "all_valid_frames")


def threshold(q, quantile):
    q = np.asarray(q, dtype=float)
    if not q.size or not np.isfinite(q).all():
        raise ValueError('Calibration Q must be nonempty and finite')
    eta = float(np.quantile(q, quantile, method='higher'))
    if eta <= 0:
        raise ValueError('Chosen calibration threshold is not positive/deployable')
    return eta


def mean_allowance(product, kept, floor):
    """Reuse released convention; no replacement for a missing floor."""
    rows = np.flatnonzero(kept)
    if not len(rows):
        return math.nan, 'no retained frames'
    try:
        return float(frame_residuals(product, rows, floor, 1.0).mean()), ''
    except ValueError as exc:
        return math.nan, str(exc)


def block_record(product, frames, mask, floor, gain):
    count = int(frames.sum()); kept = frames & mask; n = int(kept.sum())
    allowance, error = mean_allowance(product, kept, floor)
    support = blocks.month_support(product.frame_time, product.frame_unit_index, kept)
    return {'frames': count, 'kept': n,
            'retention': n / count if count else math.nan,
            'mask_only_cost': count / n if n else math.nan,
            'minimum_30_met': n >= MINIMUM,
            'kept_acquisitions': int(np.unique(product.frame_unit_index[kept]).size),
            'kept_days': int(np.unique(np.floor(product.frame_time[kept] / 86400)).size),
            'kept_supported_months': len(support),
            'missing_shelf_kept': int((kept & ~np.isfinite(product.shelf_db)).sum()),
            'G1_allowance': allowance,
            'chain_allowance': allowance * gain if math.isfinite(gain) and gain > 0 else math.nan,
            'residual_error': error}


def symmetric_ratio(left, right):
    if not (math.isfinite(left) and math.isfinite(right)):
        return math.nan
    if left == right == 0:
        return 1.0
    return max(left, right) / min(left, right) if min(left, right) > 0 else math.inf


def calendar_halves(times, calibration):
    t = np.asarray(times, dtype=float)
    cal = np.asarray(calibration, dtype=bool)
    if not cal.any():
        return cal.copy(), cal.copy(), math.nan
    midpoint = 0.5 * (float(t[cal].min()) + float(t[cal].max()))
    return cal & (t <= midpoint), cal & (t > midpoint), midpoint


def half_diagnostics(product, calibration, keep, floor, gain):
    early, late, midpoint = calendar_halves(product.frame_time, calibration)
    a = block_record(product, early, keep, floor, gain)
    b = block_record(product, late, keep, floor, gain)
    return {'early': a, 'late': b, 'midpoint_time': midpoint,
            'both_halves_30_kept': a['kept'] >= MINIMUM and b['kept'] >= MINIMUM,
            'mask_only_cost_ratio': symmetric_ratio(a['mask_only_cost'], b['mask_only_cost']),
            'G1_residual_ratio': symmetric_ratio(a['G1_allowance'], b['G1_allowance']),
            'chain_residual_ratio': symmetric_ratio(a['chain_allowance'], b['chain_allowance']),
            'scope': 'Fixed-policy point diagnostic only; no equivalence test or family gate change'}


def ladder(product, calibration: np.ndarray, evaluation: np.ndarray, floor, gain: float, *,
           population: str = "current_era_calibration_block", quantiles=QUANTILES) -> list[dict]:
    """Every rung of the ladder on one band: its threshold, and the two blocks and the calibration halves.

    ``calibration`` and ``evaluation`` are the frame masks of the blocks; the
    thresholds are read from ``Q`` on the calibration block.
    """
    if population not in POPULATIONS:
        raise ValueError(f"population must be one of {POPULATIONS}")
    view = product.view
    rows = []
    for q in (*quantiles, None):
        error = ''; eta = None
        try:
            if q is None:
                keep = np.ones(product.n_frames, dtype=bool)
            else:
                eta = threshold(view.statistic[calibration], q)
                keep = ~view.rejected_at_multiplier(eta)
        except ValueError as exc:
            keep = np.zeros(product.n_frames, dtype=bool); error = str(exc)
        cal = block_record(product, calibration, keep, floor, gain)
        ev = block_record(product, evaluation, keep, floor, gain)
        halves = half_diagnostics(product, calibration, keep, floor, gain)
        rows.append({'policy': 'keep_all' if q is None else f'cal_q{q:g}',
                     'nominal_calibration_retention': q if q is not None else 1.0,
                     'eta': eta, 'eta_integer_ratio': list(eta.as_integer_ratio()) if eta else None,
                     'policy_error': error, 'population': population, 'calibration': cal, 'evaluation': ev,
                     'calibration_halves': halves})
    return rows


def product_thresholds(path, quantiles=QUANTILES):
    """The ladder over ``all_valid_frames`` of a band's archive product (``<freq_id>.npz``).

    eta_q is the q-quantile (method 'higher') of Q = F / mu0 over every valid,
    finite frame, with mu0 = 2 |t|^2 / sum |r|^2, the bank's null power ratio
    from its norms. Returns ``{q: eta_q}`` with the frame count ``"n"`` and the
    median ``"median"`` of Q beside them.
    """
    z = np.load(path, allow_pickle=True); v = z["valid"][:, 0].astype(bool)
    mu0 = 2.0 * float(z["target_norm_sq"][0]) / float(z["reference_norm_sum_sq"][0])   # null_power_ratio of the bank
    Q = z["coarse_power_ratio"][:, 0].astype(float) / mu0; Q = Q[v & np.isfinite(Q)]
    out = {q: float(np.quantile(Q, q, method="higher")) for q in quantiles}; out["n"] = int(Q.size); out["median"] = float(np.median(Q))
    return out


def run_thresholds(path, quantiles=QUANTILES, band=None):
    """The same ladder over a detector run's outputs (``chime_detector_outputs.npz``), with mu0 the run's own
    ``null_power_ratio``: the ladder of a rescan with another bank. ``band`` reads the run's column of that
    physical channel and is refused when the run does not hold it exactly once; without it the first column is read."""
    z = np.load(path, allow_pickle=True); j = 0
    if band is not None:
        hits = np.flatnonzero(np.asarray(z["physical_channel"]) == int(band))
        if hits.size != 1:
            raise ValueError(f"{path}: band {band} is not one column of the run "
                             f"(physical_channel {[int(c) for c in z['physical_channel']]})")
        j = int(hits[0])
    v = z["valid"][:, j].astype(bool)
    Q = z["coarse_power_ratio"][:, j].astype(float) / float(z["null_power_ratio"][j]); Q = Q[v & np.isfinite(Q)]
    out = {q: float(np.quantile(Q, q, method="higher")) for q in quantiles}; out["n"] = int(Q.size); out["median"] = float(np.median(Q))
    return out


def place_dumps(path, eta=None):
    """Each band's placement of one dump (a detector run's ``chime_detector_outputs.npz``) on a ladder.

    Returns ``{band: {"q_median": median Q of the dump's valid frames, "kept_fraction": {q: fraction of those
    frames with Q <= eta_q}}}``; ``kept_fraction`` only for the bands ``eta`` (``{band: {q: eta_q}}``) holds.
    """
    d = np.load(path, allow_pickle=True)
    ch_ = d["physical_channel"]; F = d["coarse_power_ratio"]; mu0 = d["null_power_ratio"]; v = d["valid"].astype(bool)
    out = {}
    for j, c in enumerate(ch_):
        Q = F[:, j][v[:, j]] / float(mu0[j])
        row = {"q_median": float(np.median(Q))}
        if eta is not None and int(c) in eta:
            row["kept_fraction"] = {q: float(np.mean(Q <= eta[int(c)][q])) for q in eta[int(c)] if not isinstance(q, str)}
        out[int(c)] = row
    return out


__all__ = ["MINIMUM", "POPULATIONS", "QUANTILES", "block_record", "calendar_halves", "half_diagnostics",
           "ladder", "mean_allowance", "place_dumps", "product_thresholds", "run_thresholds", "symmetric_ratio",
           "threshold"]
