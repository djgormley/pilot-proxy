"""T13: an off population is null-like on both sides; a heavy upper tail is refused (author ruling of 2026-09-28)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pilot_proxy.characterization import false_alarm as fa
from pilot_proxy.characterization import nulls


def _ideal(n=20_000, seed=20260928):
    rng = np.random.default_rng(seed)
    return fa.LAW.rvs(size=n, random_state=rng) / fa.LAW.mean()


def _product(q):
    q = np.asarray(q, dtype=float)
    return SimpleNamespace(statistic=q, selected=np.ones(q.size, dtype=bool))


def _stretch_upper(q, factor):
    """The same population with every frame above its median moved `factor` times further up: the lower side,
    the centre and so the left-side core width are untouched."""
    q = np.asarray(q, dtype=float).copy()
    centre = np.median(q)
    up = q > centre
    q[up] = centre + (q[up] - centre) * factor
    return q


def test_an_ideal_population_passes_both_sides():
    q = _ideal()
    assert nulls.central68_ratio(q) == pytest.approx(1.0, abs=0.03)
    ok, widths, reason = nulls.off_population_check(_product(q), np.ones(q.size, dtype=bool))
    assert ok and widths is not None
    assert reason.startswith(f"centre {widths.centre:.4f}, core width factor {widths.core_width_factor:.2f}, r68 ")


def test_a_symmetric_wider_population_within_the_limit_passes():
    q = _ideal()
    centre = np.median(q)
    wide = centre + (q - centre) * 3.0              # both sides three ideal widths: inside OFF_WIDTH_LIMIT
    ok, widths, _ = nulls.off_population_check(_product(wide), np.ones(wide.size, dtype=bool))
    assert ok and widths.core_width_factor == pytest.approx(3.0, rel=0.05)
    assert nulls.central68_ratio(wide) == pytest.approx(3.0, rel=0.05)


def test_a_heavy_upper_tail_with_a_clean_lower_side_is_refused():
    q = _stretch_upper(_ideal(), 20.0)
    widths = nulls.describe_null(q, nulls.COARSE_DOF)
    # the release's one-sided test reads this population as a null ...
    assert nulls.null_like(widths)[0]
    assert widths.core_width_factor < nulls.OFF_WIDTH_LIMIT and abs(widths.centre - 1.0) < nulls.OFF_CENTRE_TOLERANCE
    # ... and the two-sided test refuses it on the upper side
    r68 = nulls.central68_ratio(q)
    assert r68 > nulls.OFF_WIDTH_LIMIT
    ok, _, reason = nulls.off_population_check(_product(q), np.ones(q.size, dtype=bool))
    assert not ok
    assert reason == f"central-68 width r68 {r68:.1f} exceeds {nulls.OFF_WIDTH_LIMIT:g} (upper tail not null-like)"


def test_every_failing_clause_is_named():
    q = _stretch_upper(_ideal() * 1.1, 20.0)       # centre off mu_0 as well
    ok, _, reason = nulls.off_population_check(_product(q), np.ones(q.size, dtype=bool))
    assert not ok
    assert reason.startswith("centre 1.1") and "is more than 0.02 from mu_0" in reason
    assert reason.endswith("(upper tail not null-like)") and "; central-68 width r68 " in reason


def test_too_few_frames_are_not_described():
    q = _ideal(n=nulls.MIN_NULL_FRAMES - 1)
    assert math.isnan(nulls.central68_ratio(q))
    ok, widths, reason = nulls.off_population_check(_product(q), np.ones(q.size, dtype=bool))
    assert not ok and widths is None and reason.startswith("off population has ")


def test_the_refusal_reaches_the_status_string():
    q = _stretch_upper(_ideal(), 20.0)
    ok, _, check = nulls.off_population_check(_product(q), np.ones(q.size, dtype=bool))
    assert not ok
    mask = np.ones(q.size, dtype=bool)
    off = nulls.OffPopulation(mask=mask, off_through=None, off_from="2023-02", record_frames=q.size,
                              off_frames=q.size, note="", independently_verified=True, external_record="confirmed",
                              null_like=False, null_check=check, null_r68=nulls.central68_ratio(q))
    assert not off.signal_free
    assert off.rejection_reason == f"off-state, not signal-free: {check}"
    time = np.repeat(np.arange(q.size // 50), 50) * 86400.0 + 3600.0
    limit = fa.false_alarm_limit(32, role="screened", off_population=off, q=q, frame_time=time, current_era=mask)
    assert limit["eta_pfa_status"] == f"unavailable: off-state, not signal-free: {check}"
    assert "(upper tail not null-like)" in limit["eta_pfa_status"]
    assert math.isnan(limit["eta_pfa"]) and limit["null_frames"] == 0
    assert math.isfinite(limit["eta_pfa_diagnostic"]) and limit["eta_pfa_diagnostic_frames"] == q.size


def test_the_limit_is_the_declared_width_limit():
    """No new constant: the upper clause reads r68 against the release's OFF_WIDTH_LIMIT."""
    assert nulls.OFF_WIDTH_LIMIT == 5.0
    widths = nulls.describe_null(_ideal(), nulls.COARSE_DOF)
    assert nulls.off_null_like(widths, 5.0)[0] and not nulls.off_null_like(widths, 5.0 + 1e-9)[0]
    assert not nulls.off_null_like(widths, math.nan)[0]
