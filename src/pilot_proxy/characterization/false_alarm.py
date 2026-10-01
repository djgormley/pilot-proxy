"""The false-alarm limit eta_Pfa: the threshold whose false-alarm rate on a
verified signal-free population does not exceed alpha, with its effective
sample count, its bound and its availability.

Definition (DESIGN C.4 with addendum 1 section 2). For a declared alpha and a
verified signal-free population of ``n`` frames with per-frame statistics
``b_1..b_n``, eta_Pfa is the "higher" order statistic ``b_(k)`` with
``k = ceil((1 - alpha)(n - 1))`` in exact rational arithmetic
(:func:`higher_index`; ties are kept): the tightest threshold whose empirical
rate on the population does not exceed alpha. Frames are correlated, so the
estimator is the predeclared null's (:func:`eta_pfa_block`): a day-block
bootstrap (``B`` replicates, a recorded seed) gives the design effect DEFF of
the exceedance at eta_Pfa, ``n_eff = n / max(DEFF, 1)``, and a one-sided 95%
Clopper-Pearson upper bound on the achieved rate with ``n_eff`` trials.
eta_Pfa is ``available`` when ``n_eff >= minimum_frames_per_false_alarm /
alpha`` (the detector register's value, 10), and ``unsupported: n_eff = N``
below it, with the point value still reported as a diagnostic.

Which population. Only a band's own transmitter-off population that is
independently verified and null-like, inside the band's current era (the
latest-era principle), sets eta_Pfa (:func:`false_alarm_limit`). The status
strings keep a missing specification apart from a data gap:

- ``available`` (the only status with a value in ``eta_pfa``);
- ``unsupported: n_eff = N`` (a population, too few effective samples; its
  point value is a diagnostic, in ``eta_pfa_diagnostic``);
- ``unavailable: <reason>`` (alpha declared, the data do not support it);
- ``undefined: no alpha declared``;
- ``not defined: control band ...`` (a band with no transmitter has no
  eta_Pfa of its own);
- ``not computed: statistic Z_rho`` (a fine row: eta_Pfa and the per-row
  P_fa are estimated on the coarse statistic Q only).

Two reference values are reported beside it and never used as eta_Pfa: the
ideal-noise model's quantile (:func:`ideal_model_threshold`, the F(2P, 4P)
``(1 - alpha)`` quantile divided by the law's mean) and the product's OS-CFAR
design value ``fine_p_fa`` (model-conditional, not verified).

The order-statistic calibration and fixed-sample binomial acceptance
diagnostics of the fine-detector validation (:func:`higher_rank` to
:func:`family_power_lower_bound`) are design and reference calculations, not
permission to extend or retune an experiment after evaluation; policies may
share trials, and no independence between policies is assumed by the family
lower bound.
"""
from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral, Real
from typing import Mapping

import numpy as np
from scipy import stats
from scipy.special import logsumexp
from scipy.stats import beta, betabinom, binom

from pilot_proxy.config.project import default_project

from . import oc_table

_PROJECT = default_project()
_REGISTER = _PROJECT.register
MAX_Q16 = (1 << 64) - 1
ALWAYS_MASKED_Q16 = 1 << 64


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _probability(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a probability")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        result = Fraction(str(value))
    else:
        raise TypeError(f"{name} must be a real probability or Fraction")
    if not 0 < result < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def _real_vector(values, name, *, positive=False):
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw):
        raise TypeError(f"{name} must contain real numbers")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all() or np.any(result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must contain finite {'positive' if positive else 'nonnegative'} values; no rows are dropped")
    return result


def _requirements(values):
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("logical Q16 requirements must be a nonempty vector")
    result = [_integer(v, "requirement", 1) for v in raw]
    if any(v > ALWAYS_MASKED_Q16 for v in result):
        raise ValueError("logical Q16 requirements exceed the sentinel")
    return result


def higher_index(n, pfa):
    q = 1-_probability(pfa, "pfa")
    return (q.numerator*(n-1)+q.denominator-1)//q.denominator


def empirical_threshold(responses, *, pfa):
    """Observed higher (1-pfa) quantile; no interpolation or finite-row trimming."""
    values = _real_vector(responses, "null responses")
    index = higher_index(len(values), pfa)
    threshold = float(np.partition(values, index)[index])
    exceeded = int(np.count_nonzero(values > threshold))
    return {"status": "available", "threshold": threshold,
            "index": index, "trials": len(values), "exceedances": exceeded,
            "empirical_pfa": exceeded/len(values), "nominal_pfa": float(pfa),
            "method": "observed higher quantile; strict response > threshold",
            "independent_validation": False}


def exact_q16_threshold(requirements, *, pfa):
    """Higher quantile of decoded exact keep boundaries, with ties kept.

    Sentinel 2**64 remains in every denominator and in sorting. If selected,
    there is no legal threshold; it is not replaced with the uint64 maximum.
    Serialized zero+bitset encodings must be decoded before calling.
    """
    values = _requirements(requirements)
    index = higher_index(len(values), pfa)
    threshold = sorted(values)[index]
    available = threshold <= MAX_Q16
    exceeded = sum(v > threshold for v in values) if available else None
    return {"status": "available" if available else "unavailable",
            "threshold": threshold if available else None,
            "selected_requirement": threshold, "index": index,
            "trials": len(values), "sentinel_trials": values.count(ALWAYS_MASKED_Q16),
            "exceedances": exceeded,
            "empirical_pfa": exceeded/len(values) if available else None,
            "nominal_pfa": float(pfa), "independent_validation": False,
            "method": "exact higher keep-boundary quantile; strict requirement > threshold"}


def higher_rank(trials: int, nominal_pfa: float) -> int:
    """One-based rank for ceil((1-p)*(n-1)), calculated without float rounding."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    p = Fraction(str(nominal_pfa))
    if not 0 < p < 1:
        raise ValueError("nominal_pfa must lie strictly between zero and one")
    v = (1-p)*(trials-1)
    return (v.numerator + v.denominator - 1)//v.denominator + 1


def calibration_reference(trials: int, nominal_pfa: float) -> dict:
    """Continuous-IID repeated-calibration distribution of strict exceedance p.

    If T is order statistic r from n draws, 1-F(T) ~ Beta(n-r+1,r).
    This is a sampling distribution, not a posterior or a confidence interval
    conditional on one observed threshold. For atoms and a strict comparison,
    it provides an upper stochastic reference under the same IID null law.
    """
    rank = higher_rank(trials, nominal_pfa)
    a, b = trials-rank+1, rank
    return {"trials": trials, "rank": rank, "beta_a": a, "beta_b": b,
            "mean_pfa": float(beta.mean(a, b)),
            "sd_pfa": float(beta.std(a, b)),
            "central_95_pfa": beta.ppf([.025, .975], a, b).tolist(),
            "assumption": "continuous IID scores; calibration and validation share the same null law"}


def accepted_count_limit(trials: int, cap: float, *, family_tests: int = 1334,
                         confidence: float = .95) -> int:
    """Largest k whose one-sided Bonferroni Clopper--Pearson upper <= cap.

    CP inversion is equivalent to BinomialCDF(k; n, cap) <= alpha/family.
    Returns -1 if even zero observed events cannot demonstrate the cap.
    """
    if trials < 1 or int(trials) != trials or family_tests < 1 or int(family_tests) != family_tests:
        raise ValueError("positive integer trial and family counts required")
    if not 0 < cap < 1 or not 0 < confidence < 1:
        raise ValueError("cap and confidence must lie strictly between zero and one")
    alpha = (1-confidence)/family_tests
    lo, hi = -1, trials
    while hi-lo > 1:
        k = (lo+hi)//2
        if binom.cdf(k, trials, cap) <= alpha:
            lo = k
        else:
            hi = k
    return lo


def gate_count_limit(trials: int, nominal_pfa: float, *, family_tests: int = 1334,
                     confidence: float = .95, cap_multiple: float = 2.) -> int:
    """Full low-tail acceptance region, including pointwise CP width.

    Refuses a noncontiguous region instead of presuming cap-only acceptance.
    """
    end = accepted_count_limit(trials, cap_multiple*nominal_pfa,
                               family_tests=family_tests, confidence=confidence)
    if end < 0:
        return -1
    k = np.arange(end+1)
    alpha = (1-confidence)/2
    lower = np.zeros(end+1)
    lower[1:] = beta.ppf(alpha, k[1:], trials-k[1:]+1)
    upper = beta.isf(alpha, k+1, trials-k)
    accepted = upper-lower <= nominal_pfa
    indices = np.flatnonzero(accepted)
    if not len(indices):
        return -1
    last = int(indices[-1])
    if not np.all(accepted[:last+1]):
        raise ValueError("acceptance region is not a contiguous lower tail")
    return last


def prospective_power(calibration_trials: int, validation_trials: int,
                      nominal_pfa: float, *, family_tests: int = 1334) -> dict:
    """Reference power for fresh independent calibration and validation."""
    ref = calibration_reference(calibration_trials, nominal_pfa)
    limit = gate_count_limit(validation_trials, nominal_pfa, family_tests=family_tests)
    # Sum the rejection tail directly: 1-CDF loses precision for large designs.
    failure = float(np.exp(logsumexp(betabinom.logpmf(
        np.arange(limit+1, validation_trials+1), validation_trials,
        ref['beta_a'], ref['beta_b'])))) if limit >= 0 else 1.
    failure = min(1., max(0., failure))
    power = 1-failure
    return {"calibration_trials": calibration_trials, "validation_trials": validation_trials,
            "nominal_pfa": nominal_pfa, "accepted_exceedances_max": limit,
            "single_policy_power": power, "single_policy_failure_probability": failure,
            "reference": "continuous same-null IID order-statistic calibration and fresh fixed-size validation"}


def family_power_lower_bound(policy_failure_probabilities: list[float]) -> float:
    """Union-bound lower bound; valid without independence among policies."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in policy_failure_probabilities):
        raise ValueError("failure probabilities must be finite and within [0,1]")
    return max(0., 1-math.fsum(policy_failure_probabilities))


# ---------------------------------------------------------------- the predeclared null's estimator
# The ideal coarse null law and its constants (Appendix C; the predeclared null's section 1).
DOF = _PROJECT.detector_config.coarse_null_dof(_PROJECT.instrument)
LAW = stats.f(*DOF)
MEAN0, VAR0 = (float(v) for v in LAW.stats(moments="mv"))
SIGMA0 = math.sqrt(VAR0)
M0 = float(LAW.ppf(0.5))
P16, P84 = float(stats.norm.cdf(-1.0)), float(stats.norm.cdf(1.0))
Q16_0, Q84_0 = float(LAW.ppf(P16)), float(LAW.ppf(P84))
HALF68_0 = (Q84_0 - Q16_0) / 2.0
ALPHA = Fraction(str(_REGISTER.value("detection.false_alarm_target")))
Q999_0 = float(LAW.ppf(1 - float(ALPHA)))
B = 2000
SEED = 20260925
MIN_FRAMES_PER_FALSE_ALARM = float(_REGISTER.value("detection.minimum_frames_per_false_alarm"))
SUPPORT_N = MIN_FRAMES_PER_FALSE_ALARM / float(ALPHA)


def higher_eta(sorted_q, alpha=None):
    """The higher (1 - alpha) order statistic of a sorted sample (alpha: the register's by default)."""
    n = len(sorted_q)
    return float(sorted_q[higher_index(n, ALPHA if alpha is None else alpha)])


def ideal_quantile(alpha=None) -> float:
    """The ideal law's (1 - alpha) quantile of Q (``Q999_0`` at the register's alpha)."""
    return Q999_0 if alpha is None else float(LAW.ppf(1 - float(alpha)))


def support_frames(alpha=None) -> float:
    """The support rule's effective-sample minimum, minimum_frames_per_false_alarm / alpha."""
    return SUPPORT_N if alpha is None else MIN_FRAMES_PER_FALSE_ALARM / float(alpha)


class DayBootstrap:
    """Day-block bootstrap over the frames of one selection."""

    def __init__(self, q, day, seed=SEED, reps=B):
        self.q = np.asarray(q, dtype=float)
        days, inv = np.unique(day, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv, minlength=days.size)
        self.starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.counts = counts
        self.order = order
        self.ndays = days.size
        rng = np.random.default_rng(seed)
        self.draws = [rng.integers(0, self.ndays, size=self.ndays) for _ in range(reps)]

    def samples(self):
        for d in self.draws:
            idx = np.concatenate([self.order[self.starts[k]:self.starts[k] + self.counts[k]] for k in d])
            yield self.q[idx]


def boot_stats(q, day, thresholds, seed=SEED, reps=B, alpha=None):
    """Replicates of median, r68, p999, eta_Pfa and the exceedance at fixed thresholds."""
    q999 = ideal_quantile(alpha)
    bs = DayBootstrap(q, day, seed=seed, reps=reps)
    rep = {"z_L": [], "r68": [], "p999": [], "eta_pfa": [], "n": []}
    for name in thresholds:
        rep[f"exceed:{name}"] = []
    for x in bs.samples():
        s = np.sort(x)
        med = float(np.quantile(s, 0.5))
        q16, q84 = float(np.quantile(s, P16)), float(np.quantile(s, P84))
        rep["z_L"].append((med - M0) / SIGMA0)
        rep["r68"].append((q84 - q16) / 2.0 / HALF68_0)
        rep["p999"].append(float(np.mean(x > q999)))
        rep["eta_pfa"].append(higher_eta(s, alpha))
        rep["n"].append(x.size)
        for name, t in thresholds.items():
            rep[f"exceed:{name}"].append(float(np.mean(x > t)))
    return {k: np.asarray(v, dtype=float) for k, v in rep.items()}, bs.ndays


def ci(v):
    v = np.asarray(v, dtype=float)
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def eta_pfa_block(q, day, label, seed=SEED, alpha=None):
    """eta_Pfa(alpha) by the design definition, with its day bootstrap, n_eff, support and CP bound.

    ``alpha`` is the register's (``detection.false_alarm_target``) unless one
    is given; the quantile, the ideal-law reference and the support rule all
    follow it.
    """
    a = ALPHA if alpha is None else alpha
    q = np.asarray(q, dtype=float)
    n = q.size
    if n < 2:
        return {"label": label, "n": n, "status": "unsupported: too few frames"}
    point = empirical_threshold(q, pfa=a) if n <= 200000 else None
    s = np.sort(q)
    eta = higher_eta(s, a)
    if point is not None and point["threshold"] != eta:
        raise SystemExit(f"{label}: higher-quantile mismatch {point['threshold']} vs {eta}")
    k = int(np.sum(q > eta))
    p_hat = k / n
    q999, support = ideal_quantile(alpha), support_frames(alpha)
    rep, ndays = boot_stats(q, day, {"eta_pfa_point": eta}, seed=seed, alpha=alpha)
    var_boot = float(np.var(rep["exceed:eta_pfa_point"], ddof=1))
    binom = p_hat * (1 - p_hat) / n
    deff = var_boot / binom if binom > 0 else math.nan
    n_eff = n / max(deff, 1.0) if math.isfinite(deff) else math.nan
    k_eff = p_hat * n_eff if math.isfinite(n_eff) else math.nan
    cp_upper = float(stats.beta.ppf(0.95, k_eff + 1, n_eff - k_eff)) if math.isfinite(n_eff) else math.nan
    supported = math.isfinite(n_eff) and n_eff >= support
    return {
        "label": label, "n": n, "days": ndays,
        "eta_pfa": eta, "eta_pfa_z": (eta - M0) / SIGMA0,
        "eta_minus_q999_sigma0": (eta - q999) / SIGMA0,
        "index": int(higher_index(n, a)), "exceedances": k, "achieved_pfa": p_hat,
        "boot_ci_eta": ci(rep["eta_pfa"]), "boot_ci_eta_z": [(v - M0) / SIGMA0 for v in ci(rep["eta_pfa"])],
        "deff": deff, "n_eff": n_eff, "cp95_upper_pfa": cp_upper,
        "status": "available" if supported else f"unsupported: n_eff = {n_eff:.0f}",
        "support_rule": f"n_eff >= {support:.0f}",
    }

# ---------------------------------------------------------------- the band-level limit
UNDEFINED = "undefined: no alpha declared"
AVAILABLE = "available"
MEASURED = "measured"
# eta_Pfa and P_fa are estimated on the coarse statistic Q only; a fine (Z_rho) row says so instead of carrying a
# Q value (the per-rank keep boundaries of a verified population would be needed, and none is verified today)
NOT_COMPUTED_FINE = oc_table.NOT_COMPUTED_FINE
PFA_DESIGN_MODEL_NOTE = "OS-CFAR design value under i.i.d. bulk; model-conditional; not verified"
_REGISTER_ALPHA = object()


def ideal_model_threshold(alpha=None) -> float:
    """The ideal-noise law's (1 - alpha) quantile of Q, divided by the law's mean (reference only, never eta_Pfa)."""
    a = float(ALPHA if alpha is None else alpha)
    return float(stats.f.isf(a, *DOF) / stats.f.mean(*DOF))


def band_seed(band_id) -> int:
    """The bootstrap seed of a band's own population: the predeclared null's convention (seed + 100 + band)."""
    return SEED + 100 + int(band_id)


def verified_population(off_population, current_era) -> np.ndarray | None:
    """The frames eta_Pfa and P_fa are read on: the verified signal-free off population inside the current era.

    None when the band has none (no population, not signal-free, or not in
    the current era: the latest-era principle).
    """
    if off_population is None or not off_population.signal_free or current_era is None:
        return None
    in_era = np.asarray(off_population.mask, dtype=bool) & np.asarray(current_era, dtype=bool)
    return in_era if in_era.any() else None


def exceedance(values, eta) -> float:
    """Empirical P_fa of one threshold on a population: the fraction of its frames with ``statistic > eta``."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan
    if eta is None:
        return 0.0
    return float(np.count_nonzero(x > float(eta)) / x.size)


def false_alarm_limit(band_id, *, role: str, off_population=None, q=None, frame_time=None, current_era=None,
                      alpha=_REGISTER_ALPHA) -> dict:
    """eta_Pfa of one band, or the reason it has none, as ``oc_summary`` columns of its Q rows.

    ``off_population`` is the band's :class:`~pilot_proxy.characterization.nulls.OffPopulation`;
    ``q`` and ``frame_time`` are the band's per-frame statistic and time, and
    ``current_era`` the frame mask of its current era. ``alpha`` is the
    register's ``detection.false_alarm_target`` unless one is given, and
    ``None`` declares none (``undefined: no alpha declared``).

    ``eta_pfa`` is filled only when its status is ``available``. An
    ``unsupported`` point, and the estimator run on the current era of a band
    whose independently verified population is not signal-free, go to the
    ``eta_pfa_diagnostic*`` columns; ``null_frames`` counts the verified
    signal-free population only, and ``eta_pfa_diagnostic_frames`` the frames
    a diagnostic read.
    """
    a = ALPHA if alpha is _REGISTER_ALPHA else alpha
    out = {"pfa_target": (float(a) if a is not None else "not declared"), "eta_pfa": math.nan,
           "eta_pfa_status": "", "null_source": "", "null_frames": 0, "pfa_effective_samples": math.nan,
           "pfa_upper_95": math.nan, "null_rejection_reason": "",
           "eta_pfa_diagnostic": math.nan, "eta_pfa_diagnostic_boot_low": math.nan,
           "eta_pfa_diagnostic_boot_high": math.nan, "eta_pfa_diagnostic_n_eff": math.nan,
           "eta_pfa_diagnostic_status": "", "eta_pfa_diagnostic_population": "", "eta_pfa_diagnostic_frames": 0}
    if a is None:
        out["eta_pfa_status"] = UNDEFINED
        return out
    estimator_alpha = None if a == ALPHA else a
    if role == "control":
        out["eta_pfa_status"] = ("not defined: control band (no transmitter); its 1e-3 quantile is the surrogate "
                                 "Q37(1e-3) of the predeclared null, not an eta_Pfa")
        out["null_source"] = "control band (no transmitter)"
        return out
    reason = off_population.rejection_reason if off_population is not None else ""
    if off_population is None or (off_population.off_from is None and off_population.off_through is None):
        from .nulls import NO_OFF_EPOCH
        out["null_rejection_reason"] = NO_OFF_EPOCH
        out["null_source"] = "bulk of the mixture (declared)"
        out["eta_pfa_status"] = f"unavailable: {NO_OFF_EPOCH}"
        return out
    out["null_source"] = f"transmitter-off epoch {off_population.dated}"
    in_era = verified_population(off_population, current_era)
    latest = bool(off_population.mask is not None and current_era is not None
                  and (np.asarray(off_population.mask, dtype=bool) & np.asarray(current_era, dtype=bool)).any())
    if off_population.signal_free and not latest:
        reason = (f"transmitter-off epoch {off_population.dated} is not the current era "
                  "(latest-era principle)")
    elif not off_population.signal_free and not latest and off_population.off_through is not None:
        reason = reason + f"; the off epoch {off_population.dated} is not the current era (latest-era principle)"
    out["null_rejection_reason"] = reason
    if in_era is not None:
        values = np.asarray(q, dtype=float)[in_era]
        times = np.asarray(frame_time, dtype=float)[in_era]
        keep = np.isfinite(values) & np.isfinite(times)
        block = eta_pfa_block(values[keep], np.floor(times[keep] / 86400.0).astype(np.int64),
                              f"band {band_id} verified off population", seed=band_seed(band_id),
                              alpha=estimator_alpha)
        out.update({"eta_pfa_status": block["status"],
                    "null_source": f"verified signal-free transmitter-off epoch {off_population.dated}",
                    "null_frames": int(block["n"]), "pfa_effective_samples": block.get("n_eff", math.nan)})
        if block["status"] == AVAILABLE:
            out.update({"eta_pfa": block.get("eta_pfa", math.nan), "pfa_upper_95": block.get("cp95_upper_pfa", math.nan)})
        else:
            out.update({"eta_pfa_diagnostic": block.get("eta_pfa", math.nan),
                        "eta_pfa_diagnostic_boot_low": block.get("boot_ci_eta", [math.nan, math.nan])[0],
                        "eta_pfa_diagnostic_boot_high": block.get("boot_ci_eta", [math.nan, math.nan])[1],
                        "eta_pfa_diagnostic_n_eff": block.get("n_eff", math.nan),
                        "eta_pfa_diagnostic_status": block["status"],
                        "eta_pfa_diagnostic_population": ("verified signal-free population, current era "
                                                          "(below the support rule: a diagnostic, not eta_Pfa)"),
                        "eta_pfa_diagnostic_frames": int(block["n"])})
        return out
    out["eta_pfa_status"] = (f"unavailable: {reason}" if off_population.independently_verified
                             else f"unavailable: no verified signal-free population ({reason})")
    if off_population.independently_verified and current_era is not None and q is not None:
        era = np.asarray(current_era, dtype=bool)
        values = np.asarray(q, dtype=float)[era]
        times = np.asarray(frame_time, dtype=float)[era]
        keep = np.isfinite(values) & np.isfinite(times)
        block = eta_pfa_block(values[keep], np.floor(times[keep] / 86400.0).astype(np.int64),
                              f"band {band_id} current era (diagnostic)", seed=band_seed(band_id),
                              alpha=estimator_alpha)
        out.update({"eta_pfa_diagnostic": block.get("eta_pfa", math.nan),
                    "eta_pfa_diagnostic_boot_low": block.get("boot_ci_eta", [math.nan, math.nan])[0],
                    "eta_pfa_diagnostic_boot_high": block.get("boot_ci_eta", [math.nan, math.nan])[1],
                    "eta_pfa_diagnostic_n_eff": block.get("n_eff", math.nan),
                    "eta_pfa_diagnostic_status": block["status"],
                    "eta_pfa_diagnostic_population": "current era, finite Q and time (not signal-free)",
                    "eta_pfa_diagnostic_frames": int(block["n"])})
    return out


def fine_row_columns(limit: Mapping) -> dict:
    """The false-alarm columns of a fine (Z_rho) summary row: no Q value, the population's description kept."""
    return {**limit, "eta_pfa": math.nan, "eta_pfa_status": NOT_COMPUTED_FINE, "null_frames": 0,
            "pfa_effective_samples": math.nan, "pfa_upper_95": math.nan,
            "eta_pfa_diagnostic": math.nan, "eta_pfa_diagnostic_boot_low": math.nan,
            "eta_pfa_diagnostic_boot_high": math.nan, "eta_pfa_diagnostic_n_eff": math.nan,
            "eta_pfa_diagnostic_status": "", "eta_pfa_diagnostic_population": "", "eta_pfa_diagnostic_frames": 0}


def row_pfa_status(limit: Mapping, statistic: str, population_available: bool) -> str:
    """``pfa_status`` of one oc_table row: measured (Q, on a verified population), not computed (Z_rho), or the band's reason."""
    status = limit["eta_pfa_status"]
    if statistic != "Q":
        return NOT_COMPUTED_FINE
    if population_available:
        return MEASURED
    return status


__all__ = ["ALPHA", "ALWAYS_MASKED_Q16", "AVAILABLE", "B", "DOF", "DayBootstrap", "LAW", "M0", "MAX_Q16",
           "MEASURED", "MIN_FRAMES_PER_FALSE_ALARM", "NOT_COMPUTED_FINE", "PFA_DESIGN_MODEL_NOTE", "Q999_0", "SEED",
           "SIGMA0", "SUPPORT_N", "UNDEFINED", "accepted_count_limit", "band_seed", "boot_stats",
           "calibration_reference", "ci", "empirical_threshold", "eta_pfa_block", "exact_q16_threshold", "exceedance",
           "false_alarm_limit", "family_power_lower_bound", "fine_row_columns", "gate_count_limit", "higher_eta",
           "higher_index", "higher_rank", "ideal_model_threshold", "ideal_quantile", "prospective_power",
           "row_pfa_status", "support_frames", "verified_population"]
