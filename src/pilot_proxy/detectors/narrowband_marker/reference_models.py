"""Reference laws of the narrowband-marker detector's coarse statistic.

The coarse statistic ``Q = F / mu_0`` sums ``P`` independent projections in
its target and reference branches (``P = n_inputs x K``, the detector
configuration's summed terms; 262144 on CHIME). Under ideal independent
Gaussian noise with equal, signal-free references it follows the central
``F(2P, 4P)`` law: the null a detector designer expects (H0,
:data:`NULL`). A fixed received signal of projected power ratio ``gamma``
makes the numerator noncentral with ``lambda = 2P gamma`` (the steady-emitter
law, :func:`steady_law`, H1).

- :func:`median_matched`: the noncentral law whose median equals an observed
  median. It is fitted to each era, so it describes the data rather than
  giving independent evidence about the null.
- :func:`moments`: its exact mean and standard deviation.
- :func:`bin_masses`: a law's probability mass in histogram bins, from the
  lower-tail CDF below the median and the upper-tail survival function above
  (the numerically stable side of each bin).
- :func:`power_response`: the median, mean and spread of ``Q`` as a function
  of ``gamma``, with the large-``P`` approximations.
- :func:`row_ratio_law`: the single-row ratio ``F(2, 4)`` (noncentral with
  ``lambda``) of one detector row, for single-row bench references.

Every law is ideal: independent proper complex Gaussian noise, equal noise
variances, independent target and reference branches, signal-free references.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from scipy.stats import f, ncf

from pilot_proxy.config.project import default_project

_PROJECT = default_project()
P = _PROJECT.detector_config.summed_terms(_PROJECT.instrument)
A, B = _PROJECT.detector_config.coarse_null_dof(_PROJECT.instrument)
DF1, DF2 = A, B
NULL = stats.f(A, B)
NULL_MEDIAN = float(NULL.ppf(.5))
NULL_SD = float(NULL.std())


def median_matched(median):
    if median < NULL_MEDIAN:
        return NULL, 0.0, "below_null_median_no_nonnegative_match", float(NULL.cdf(median) - .5)
    if median == NULL_MEDIAN:
        return NULL, 0.0, "median_matched", 0.0
    def objective(gamma):
        return float(NULL.cdf(median) if gamma == 0 else stats.ncf.cdf(median, A, B, A * gamma)) - .5
    high = max(.001, 2 * max(median - 1, 0))
    while objective(high) > 0:
        high *= 2
        if high > 1e12:
            raise ValueError("Could not bracket median-equivalent noncentrality")
    gamma = float(optimize.brentq(objective, 0.0, high, xtol=1e-13, rtol=1e-13))
    error = objective(gamma)
    if abs(error) > 1e-7:
        raise ValueError(f"Median CDF inversion inaccurate: {error}")
    return stats.ncf(A, B, A * gamma), gamma, "median_matched", error


def moments(gamma):
    nc = A * gamma
    mean = B * (A + nc) / (A * (B - 2))
    var = 2 * (B / A)**2 * ((A + nc)**2 + (A + 2 * nc) * (B - 2)) / ((B - 2)**2 * (B - 4))
    return mean, np.sqrt(var)


def bin_masses(law, edges):
    lo, hi = edges[:-1], edges[1:]
    c_lo, c_hi = law.cdf(lo), law.cdf(hi)
    s_lo, s_hi = law.sf(lo), law.sf(hi)
    mass = np.where(c_lo > .5, s_lo-s_hi, c_hi-c_lo)
    if np.any(mass < -1e-9) or not np.isfinite(mass).all():
        raise ValueError("Invalid model bin mass")
    return np.clip(mass, 0, 1)


def steady_law(gamma):
    return stats.f(2 * P, 4 * P) if gamma == 0 else stats.ncf(2 * P, 4 * P, 2 * P * gamma)


def power_response(gamma):
    gamma = np.atleast_1d(np.asarray(gamma, dtype=np.float64))
    if not np.isfinite(gamma).all() or np.any(gamma < 0):
        raise ValueError("Signal/noise power must be finite and nonnegative")
    nc = DF1 * gamma
    positive = gamma > 0
    median = np.full(gamma.shape, stats.f.ppf(0.5, DF1, DF2))
    median[positive] = stats.ncf.ppf(0.5, DF1, DF2, nc[positive])
    mean = DF2 * (DF1 + nc) / (DF1 * (DF2 - 2))
    # Exact variance of (X/DF1)/(Y/DF2), X noncentral chi-square and Y central.
    variance = (2 * (DF2 / DF1) ** 2
                * ((DF1 + nc) ** 2 + (DF1 + 2 * nc) * (DF2 - 2))
                / ((DF2 - 2) ** 2 * (DF2 - 4)))
    sd = np.sqrt(variance)
    approximate_sd = np.sqrt((gamma**2 + 6 * gamma + 3) / (2 * P))
    return {
        "gamma": gamma, "lambda": nc, "median": median, "mean": mean,
        "std": sd, "relative_std_over_mean": sd / mean,
        "median_large_P_approximation": 1 + gamma,
        "std_large_P_approximation": approximate_sd,
    }


def row_ratio_law(lam):
    # Use central f explicitly at lambda0 (some SciPy ncf versions have sf edge behavior).
    return f(2,4) if lam == 0 else ncf(2,4,lam)


__all__ = ["A", "B", "DF1", "DF2", "NULL", "NULL_MEDIAN", "NULL_SD", "P", "bin_masses", "median_matched",
           "moments", "power_response", "row_ratio_law", "steady_law"]
