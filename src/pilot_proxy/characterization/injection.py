"""Recovery statistics for truth-labelled injections: detection probability at
a matched false-alarm rate, and the recovery-linearity fit.

An injection ladder puts a known tone into stored data at several amplitudes,
with one ``a = 0`` control point that is byte-identical to the source. The
statistics here need only per-frame statistics and the ladder's truth labels:

- :func:`matched_thresholds`: for each requested false-alarm rate, the
  empirical ``(1 - P_fa)`` quantile of each statistic on the control, used
  only where the control holds at least
  :data:`MIN_FRAMES_PER_FALSE_ALARM` / ``P_fa`` valid frames (the detector
  register's ``detection.minimum_frames_per_false_alarm``);
- :func:`detection_rates`: the detected fraction of a point at each threshold,
  with Wilson 95% intervals;
- :func:`weighted_linear_fit`: ``rho_hat = floor + gain * a^2`` with
  ``1 / sem^2`` weights;
- :func:`signal_dominated_log_slope`: the slope-one check in log space over the
  points where the injected term dominates the floor.

The orchestration (reading ladder points, writing the report and figures)
stays with the instrument reader (``pilot_proxy.chime.injection_recovery``).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pilot_proxy.config.project import default_project
from pilot_proxy.testbench.evaluate_snr import wilson_interval

# An empirical quantile at P_fa needs enough H0 frames to be meaningful: at
# least this many expected false alarms (detector register entry
# detection.minimum_frames_per_false_alarm).
MIN_FRAMES_PER_FALSE_ALARM = default_project().register.value(
    "detection.minimum_frames_per_false_alarm"
)


def weighted_linear_fit(
    x: np.ndarray, y: np.ndarray, sem: np.ndarray
) -> dict[str, float]:
    """Weighted least squares y = floor + gain * x with 1/sem^2 weights."""
    weights = 1.0 / np.square(sem)
    design = np.stack([np.ones_like(x), x], axis=1)
    wd = design * weights[:, np.newaxis]
    normal = design.T @ wd
    covariance = np.linalg.inv(normal)
    beta = covariance @ (wd.T @ y)
    return {
        "floor": float(beta[0]),
        "gain_per_lsb2": float(beta[1]),
        "floor_err": float(np.sqrt(covariance[0, 0])),
        "gain_err": float(np.sqrt(covariance[1, 1])),
    }


def matched_thresholds(control: dict[str, Any], false_alarm_rates: Sequence[float], *,
                       statistics: dict[str, str]) -> tuple[dict[str, dict[str, float]], list[float]]:
    """Thresholds at matched P_fa from the control's empirical quantiles, and the rates it supports.

    ``statistics`` maps an output name to the control's key for that
    statistic (for example ``{"fstat": "fstat", "radiometer": "power"}``).
    """
    thresholds: dict[str, dict[str, float]] = {name: {} for name in statistics}
    usable_pfa: list[float] = []
    for pfa in sorted(set(float(p) for p in false_alarm_rates), reverse=True):
        if control["n_valid"] < MIN_FRAMES_PER_FALSE_ALARM / pfa:
            continue
        usable_pfa.append(pfa)
        quantile = 1.0 - pfa
        for name, key in statistics.items():
            thresholds[name][f"{pfa:g}"] = float(
                np.quantile(control[key], quantile)
            )
    return thresholds, usable_pfa


def detection_rates(point: dict[str, Any], thresholds: dict[str, dict[str, float]], usable_pfa: Sequence[float], *,
                    statistics: dict[str, str]) -> dict[str, float]:
    """The detected fraction of one point at each matched threshold, with Wilson 95% intervals."""
    row: dict[str, float] = {}
    for pfa in usable_pfa:
        key = f"{pfa:g}"
        for name, source in statistics.items():
            stat = point[source]
            detected = int(np.count_nonzero(stat > thresholds[name][key]))
            lo, hi = wilson_interval(detected, point["n_valid"])
            row[f"pd_{name}_pfa{key}"] = detected / point["n_valid"]
            row[f"pd_{name}_pfa{key}_wilson95_lo"] = lo
            row[f"pd_{name}_pfa{key}_wilson95_hi"] = hi
    return row


def signal_dominated_log_slope(x: np.ndarray, y: np.ndarray, fit: dict[str, float]) -> float | None:
    """Slope of log(rho_hat - floor) against log(a^2) over the signal-dominated points (None with fewer than two)."""
    dominated = x * fit["gain_per_lsb2"] > 3.0 * abs(fit["floor"])
    log_slope = None
    if np.count_nonzero(dominated) >= 2:
        lx = np.log10(x[dominated])
        ly = np.log10(y[dominated] - fit["floor"])
        log_slope = float(np.polyfit(lx, ly, 1)[0])
    return log_slope


__all__ = ["MIN_FRAMES_PER_FALSE_ALARM", "detection_rates", "matched_thresholds", "signal_dominated_log_slope",
           "weighted_linear_fit"]
