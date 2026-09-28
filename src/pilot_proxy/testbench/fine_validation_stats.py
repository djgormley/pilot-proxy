"""Statistical bookkeeping for separately frozen fine-detector experiments.

These helpers do not certify a detector, physical residual or telescope policy.
Trials must be independent across IDs; stages/SNRs may deliberately be paired.
The single-look ideal fine F(2,4) has infinite variance: a finite sample standard
deviation at M=1 is descriptive, not a finite theoretical scaling benchmark.
SciPy is required for exact binomial intervals (available in the test extra).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math

import numpy as np

from pilot_proxy.characterization.false_alarm import (  # noqa: F401  (re-exported under their former names)
    ALWAYS_MASKED_Q16, MAX_Q16, _integer, _probability, _real_vector, _requirements,
    empirical_threshold as empirical_null_threshold,
    exact_q16_threshold as exact_q16_null_threshold,
    higher_index as _higher_index,
)


def binomial_interval(successes, trials, *, confidence=0.95, side="two-sided"):
    """Exact Clopper-Pearson bounds; k counts events, which may be false alarms."""
    from scipy.stats import beta

    n = _integer(trials, "trials", 1)
    k = _integer(successes, "successes")
    if k > n:
        raise ValueError("successes cannot exceed trials")
    level = float(_probability(confidence, "confidence"))
    if side not in {"two-sided", "upper", "lower"}:
        raise ValueError("side must be two-sided, upper or lower")
    alpha = 1.0-level
    tail = alpha/2 if side == "two-sided" else alpha
    lo = 0.0 if k == 0 or side == "upper" else float(beta.ppf(tail, k, n-k+1))
    hi = 1.0 if k == n or side == "lower" else float(beta.isf(tail, k+1, n-k))
    return {"successes": k, "trials": n, "estimate": k/n, "lower": lo,
            "upper": hi, "confidence": level, "side": side,
            "method": "Clopper-Pearson exact binomial", "independent_trial_assumption": True}


def false_alarm_validation(exceedances, trials, *, nominal_pfa, cap_pfa,
                           family_tests=1, confidence=0.95):
    """Pointwise precision plus a separate Bonferroni family upper-cap check.

    The cap is an explicit engineering tolerance, not the nominal calibration
    target. Dependence between policies is allowed; trials within a policy must
    be independent. Call with the complete predeclared family size.
    """
    nominal = float(_probability(nominal_pfa, "nominal_pfa"))
    cap = float(_probability(cap_pfa, "cap_pfa"))
    family = _integer(family_tests, "family_tests", 1)
    level = float(_probability(confidence, "confidence"))
    pointwise = binomial_interval(exceedances, trials, confidence=level)
    per_test = 1-(1-level)/family
    if per_test == 1.0:
        raise ValueError("family size exceeds floating confidence resolution")
    simultaneous = binomial_interval(exceedances, trials, confidence=per_test, side="upper")
    return {"pointwise": pointwise, "simultaneous_upper": simultaneous,
            "family_tests": family, "family_confidence": level,
            "nominal_pfa": nominal, "cap_pfa": cap,
            "cap_demonstrated": simultaneous["upper"] <= cap,
            "pointwise_interval_width": pointwise["upper"]-pointwise["lower"],
            "pointwise_width_at_most_nominal": pointwise["upper"]-pointwise["lower"] <= nominal,
            "physical_acceptance": False}


def raw_crossing_brackets(snr_db, rates, *, target):
    """Report every raw adjacent crossing and target plateau, without smoothing.

    A unique result needs one upward crossing, no downward crossing and no
    multi-point target plateau. Endpoints outside the sampled range are never
    extrapolated. Linear interpolation is labeled as a display estimate.
    """
    x = np.asarray(snr_db, dtype=float)
    y = np.asarray(rates, dtype=float)
    t = float(_probability(target, "target"))
    if x.ndim != 1 or x.size < 2 or y.shape != x.shape:
        raise ValueError("at least two aligned SNR/rate points are required")
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("SNR grid must be finite and strictly increasing")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("rates must be finite probabilities; no rows are dropped")
    upward, downward, plateau = [], [], []
    for i in range(len(x)-1):
        if y[i] == t == y[i+1]:
            plateau.append([i, i+1])
        direction = None
        if y[i] < t <= y[i+1]:
            direction = upward
        elif y[i] > t >= y[i+1]:
            direction = downward
        if direction is not None:
            estimate = x[i]+(x[i+1]-x[i])*(t-y[i])/(y[i+1]-y[i])
            direction.append({"indices": [i, i+1], "snr_lo_db": float(x[i]),
                              "snr_hi_db": float(x[i+1]), "rate_lo": float(y[i]),
                              "rate_hi": float(y[i+1]), "estimate_db": float(estimate)})
    # A curve that touches the target and turns back is not an identified crossing.
    touches = [i for i in range(1, len(y)-1) if y[i] == t and
               ((y[i-1] < t and y[i+1] < t) or (y[i-1] > t and y[i+1] > t))]
    ambiguous = len(upward) > 1 or bool(downward or plateau or touches)
    unique = len(upward) == 1 and not ambiguous
    return {"target": t, "upward_brackets": upward, "downward_brackets": downward,
            "plateau_segments": plateau, "touch_indices": touches,
            "exact_target_indices": np.flatnonzero(y == t).tolist(),
            "ambiguous": ambiguous, "status": "unique" if unique else "ambiguous" if ambiguous else "unbracketed",
            "estimate_db": upward[0]["estimate_db"] if unique else None,
            "raw_rates": y.tolist(), "snr_db": x.tolist(),
            "method": "raw adjacent sampled crossings; linear display interpolation; no extrapolation"}


def _identities(values, expected, name):
    ids = list(values)
    if len(ids) != expected:
        raise ValueError(f"{name} do not match the trial axis")
    try:
        if len(set(ids)) != expected:
            raise ValueError(f"{name} must be unique")
    except TypeError as exc:
        raise TypeError(f"{name} must be hashable") from exc
    return ids


def paired_crossing_loss_bootstrap(*, snr_db, null_by_stage, h1_by_stage,
                                   trial_ids, null_ids, pfa, target,
                                   replicates, seed, stage_kinds=None, pairs=None,
                                   min_valid_fraction=0.95):
    """Paired recalibration+H1 bootstrap, retaining every censored replicate.

    Each H1 stage is [SNR, trial], with one explicit shared trial-ID vector.
    Null IDs are disjoint from H1 IDs; both sets are paired across stages.
    Stage kinds are 'float' responses or decoded 'q16' keep requirements.
    Loss is second-stage minus first-stage crossing. Returned percentile
    intervals are withheld below min_valid_fraction, or if the original curve
    pair is ambiguous/unbracketed. Valid intervals remain explicitly conditional
    on the small uncensored fraction; censor counts and every replicate remain.
    """
    nrep = _integer(replicates, "replicates", 1)
    seed = _integer(seed, "seed")
    minimum = float(min_valid_fraction)
    if not math.isfinite(minimum) or not 0.95 <= minimum <= 1:
        raise ValueError("min_valid_fraction must lie in [0.95,1]")
    if not isinstance(null_by_stage, Mapping) or set(null_by_stage) != set(h1_by_stage) or len(null_by_stage) < 2:
        raise ValueError("at least two identical stage sets are required")
    stages = list(null_by_stage)
    kinds = dict(stage_kinds or {s: "float" for s in stages})
    if set(kinds) != set(stages) or any(k not in {"float", "q16"} for k in kinds.values()):
        raise ValueError("each stage kind must be float or q16")
    comparisons = list(pairs or zip(stages[:-1], stages[1:]))
    if not comparisons or any(len(p) != 2 or p[0] == p[1] or any(s not in stages for s in p) for p in comparisons):
        raise ValueError("pairs must name distinct existing stages")
    if len(set(tuple(p) for p in comparisons)) != len(comparisons):
        raise ValueError("comparison pairs must be unique")
    null, h1 = {}, {}
    for s in stages:
        check = _requirements if kinds[s] == "q16" else lambda v: _real_vector(v, "responses")
        null[s] = np.asarray(check(null_by_stage[s]), dtype=object if kinds[s] == "q16" else float)
        raw = np.asarray(h1_by_stage[s], dtype=object)
        if raw.ndim != 2 or raw.shape[0] != len(snr_db):
            raise ValueError("each H1 stage must have shape [SNR,trial]")
        h1[s] = np.asarray([check(row) for row in raw], dtype=object if kinds[s] == "q16" else float)
    nnull, nh1 = len(null[stages[0]]), h1[stages[0]].shape[1]
    if any(len(null[s]) != nnull or h1[s].shape[1] != nh1 for s in stages):
        raise ValueError("paired stages must have identical complete trial axes")
    null_keys = _identities(null_ids, nnull, "null IDs")
    h1_keys = _identities(trial_ids, nh1, "H1 IDs")
    if set(null_keys) & set(h1_keys):
        raise ValueError("calibration-null and H1 trial IDs overlap")
    raw_crossing_brackets(snr_db, np.zeros(len(snr_db)), target=target)
    _probability(pfa, "pfa")

    def crossings(ni, hi):
        out = {}
        for s in stages:
            calibrator = exact_q16_null_threshold if kinds[s] == "q16" else empirical_null_threshold
            threshold = calibrator(null[s][ni], pfa=pfa)
            if threshold["status"] != "available":
                out[s] = {"status": "threshold_unavailable", "estimate_db": None}
                continue
            rates = np.asarray(h1[s][:, hi] > threshold["threshold"], dtype=bool).mean(axis=1)
            out[s] = raw_crossing_brackets(snr_db, rates, target=target)
        return out

    original = crossings(np.arange(nnull), np.arange(nh1))
    loss = {tuple(pair): [] for pair in comparisons}
    censor = {tuple(pair): Counter() for pair in comparisons}
    rng = np.random.default_rng(seed)
    for _ in range(nrep):
        curves = crossings(rng.integers(nnull, size=nnull), rng.integers(nh1, size=nh1))
        for pair in loss:
            a, b = pair
            if curves[a]["status"] == curves[b]["status"] == "unique":
                loss[pair].append(curves[b]["estimate_db"]-curves[a]["estimate_db"])
            else:
                loss[pair].append(None)
                censor[pair][f"{a}:{curves[a]['status']}|{b}:{curves[b]['status']}"] += 1
    report = []
    for pair, samples in loss.items():
        finite = [v for v in samples if v is not None]
        fraction = len(finite)/nrep
        point_unique = all(original[s]["status"] == "unique" for s in pair)
        eligible = fraction >= minimum and point_unique
        interval = np.quantile(finite, [0.025, 0.5, 0.975]).tolist() if eligible else [None]*3
        report.append({"stages": list(pair), "requested_replicates": nrep,
                       "valid_replicates": len(finite), "censored_replicates": nrep-len(finite),
                       "valid_fraction": fraction, "minimum_valid_fraction": minimum,
                       "censoring_reasons": dict(censor[pair]), "replicate_losses_db": samples,
                       "interval_reportable": eligible,
                       "point_loss_db": original[pair[1]]["estimate_db"]-original[pair[0]]["estimate_db"] if point_unique else None,
                       "loss_lo_db": interval[0], "loss_median_db": interval[1], "loss_hi_db": interval[2],
                       "interval_scope": "paired percentile interval conditional on bracketed replicates; censoring remains explicit"})
    return {"seed": seed, "null_trials": nnull, "h1_trials_per_snr": nh1,
            "nominal_pfa": float(pfa), "target_pd": float(target),
            "point_crossings": original, "comparisons": report,
            "pairing": "shared null and H1 index resamples across all stages and SNRs",
            "physical_acceptance": False}


def exact_rank_frontier(requirements, truth, *, exposure=None, min_kept=30,
                        variance=None):
    """Every empirical Q16 step on each rank, retaining nonmonotone truth means.

    `requirements` maps one-based ranks to full logical requirement vectors.
    Known injected truth and optional variance allowance are in linear units.
    Exposure cost restores the supplied exposure only; no sky covariance or
    physical transfer is inferred. Support failures remain rows in the result.
    """
    if not isinstance(requirements, Mapping) or not requirements:
        raise ValueError("requirements must map ranks to vectors")
    minimum = _integer(min_kept, "min_kept", 1)
    signal = _real_vector(truth, "truth")
    n = len(signal)
    weights = np.ones(n) if exposure is None else _real_vector(exposure, "exposure", positive=True)
    noise = None if variance is None else _real_vector(variance, "variance")
    if len(weights) != n or (noise is not None and len(noise) != n):
        raise ValueError("all per-frame arrays must share the same complete population")
    total = math.fsum(weights)
    if not math.isfinite(total):
        raise ValueError("total exposure is not finite")
    rows, sentinels = [], {}
    for rank in sorted(requirements):
        rho = _integer(rank, "rank", 1)
        values = _requirements(requirements[rank])
        if len(values) != n:
            raise ValueError("all ranks must share the same complete frame population")
        sentinels[rho] = values.count(ALWAYS_MASKED_Q16)
        groups = {}
        for index, value in enumerate(values):
            if value <= MAX_Q16:
                groups.setdefault(value, []).append(index)
        thresholds = sorted({1, *groups})
        kept = 0
        sums = [0.0, 0.0, 0.0, 0.0]
        for eta in thresholds:
            indices = groups.get(eta, [])
            kept += len(indices)
            terms = [math.fsum(weights[indices]), math.fsum(signal[indices]),
                     math.fsum(weights[indices]*signal[indices]),
                     0.0 if noise is None else math.fsum(weights[indices]*noise[indices])]
            sums = [math.fsum((a, b)) for a, b in zip(sums, terms)]
            if not all(math.isfinite(v) for v in sums):
                raise ValueError("frontier sums are not finite")
            retained, signal_sum, weighted_signal, weighted_noise = sums
            mean = weighted_signal/retained if retained else None
            cost = total/retained if retained else None
            variance_mean = weighted_noise/retained if retained and noise is not None else None
            rows.append({"rank": rho, "eta_q16": eta, "frames": n, "kept": kept,
                         "masked_fraction": 1-kept/n, "retention": kept/n,
                         "total_exposure": total, "retained_exposure": retained,
                         "exposure_retention": retained/total,
                         "truth_sum": signal_sum, "retained_truth_mean": mean,
                         "retained_truth_mean_unweighted": signal_sum/kept if kept else None,
                         "mask_only_cost": cost, "variance_mean": variance_mean,
                         "total_cost": (1+variance_mean)*cost if variance_mean is not None else None,
                         "supported": kept >= minimum})
    return {"frames": n, "min_kept": minimum, "sentinel_counts": sentinels,
            "rows": rows, "physical_acceptance": False,
            "scope": "Exact empirical steps on supplied data; choosing thresholds here is development selection, not independent evaluation"}


def invert_frontier(frontier, tolerances, *, objective="minimum_mask"):
    """Unsmoothed per-rank and all-rank tolerance envelopes, with explicit objective.

    Minimum mask maximizes retained exposure. Total cost requires supplied
    variance values and minimizes (1+retained variance)*exposure cost. Neither
    output is an accepted operating policy or an independent test result.
    """
    if objective not in {"minimum_mask", "total_cost"}:
        raise ValueError("objective must be minimum_mask or total_cost")
    limits = _real_vector(tolerances, "tolerances")
    rows = frontier["rows"]
    ranks = sorted({r["rank"] for r in rows})
    out = []
    for limit in limits:
        feasible = [r for r in rows if r["supported"] and r["retained_truth_mean"] is not None
                    and r["retained_truth_mean"] <= limit]
        if objective == "total_cost" and any(r["total_cost"] is None for r in feasible):
            raise ValueError("total-cost inversion needs an explicit variance allowance")
        key = (lambda r: (r["mask_only_cost"], r["retained_truth_mean"], r["rank"], r["eta_q16"])) if objective == "minimum_mask" else (lambda r: (r["total_cost"], r["retained_truth_mean"], r["rank"], r["eta_q16"]))
        per_rank = {}
        for rank in ranks:
            candidates = [r for r in feasible if r["rank"] == rank]
            per_rank[rank] = dict(min(candidates, key=key)) if candidates else None
        out.append({"tolerance": float(limit), "per_rank": per_rank,
                    "envelope": dict(min(feasible, key=key)) if feasible else None,
                    "status": "feasible_on_supplied_data" if feasible else "no_feasible_step"})
    return {"objective": objective, "tolerances": out, "physical_acceptance": False}
