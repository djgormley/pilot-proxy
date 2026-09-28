"""The injection-recovery statistics: the weighted fit, matched thresholds and the detected fractions."""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.characterization import injection
from pilot_proxy.chime import injection_recovery


def test_the_orchestration_uses_these_statistics():
    assert injection_recovery.MIN_FRAMES_PER_FALSE_ALARM == injection.MIN_FRAMES_PER_FALSE_ALARM == 10.0


def test_the_weighted_fit_recovers_an_exact_line_and_weights_by_sem():
    x = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    y = 0.5 + 3.0 * x
    fit = injection.weighted_linear_fit(x, y, np.full(x.size, 0.1))
    assert fit["floor"] == pytest.approx(0.5) and fit["gain_per_lsb2"] == pytest.approx(3.0)
    # a point with a huge sem does not move the fit
    y_bad = y.copy()
    y_bad[2] += 100.0
    sem = np.full(x.size, 0.1)
    sem[2] = 1e6
    fit = injection.weighted_linear_fit(x, y_bad, sem)
    assert fit["gain_per_lsb2"] == pytest.approx(3.0, rel=1e-6)
    assert fit["floor_err"] > 0 and fit["gain_err"] > 0


def test_thresholds_are_the_controls_quantiles_at_supported_rates_only():
    rng = np.random.default_rng(4)
    control = {"n_valid": 5000, "f": rng.standard_normal(5000)}
    thresholds, usable = injection.matched_thresholds(control, [1e-3, 1e-2, 0.1], statistics={"fstat": "f"})
    assert usable == [0.1, 0.01]                      # 1e-3 would need 10,000 frames
    assert thresholds["fstat"]["0.1"] == float(np.quantile(control["f"], 0.9))
    point = {"n_valid": 4, "f": np.array([-10.0, 10.0, 10.0, thresholds["fstat"]["0.1"]])}
    row = injection.detection_rates(point, thresholds, usable, statistics={"fstat": "f"})
    assert row["pd_fstat_pfa0.1"] == 0.5              # equality is not a detection
    assert row["pd_fstat_pfa0.1_wilson95_lo"] < 0.5 < row["pd_fstat_pfa0.1_wilson95_hi"]


def test_the_log_slope_reads_only_signal_dominated_points():
    x = np.logspace(0, 3, 10)
    fit = {"floor": 1.0, "gain_per_lsb2": 2.0}
    y = fit["floor"] + fit["gain_per_lsb2"] * x
    assert injection.signal_dominated_log_slope(x, y, fit) == pytest.approx(1.0)
    assert injection.signal_dominated_log_slope(x[:1], y[:1], fit) is None
