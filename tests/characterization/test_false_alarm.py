"""eta_Pfa: the predeclared null estimator, its support rule, the status forms and the signal-free gate."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats

from pilot_proxy.characterization import false_alarm as fa
from pilot_proxy.characterization import nulls


def test_the_higher_index_is_the_exact_ceiling():
    for n in (1, 2, 999, 1000, 1001, 10_000, 123_457):
        for pfa in (Fraction(1, 1000), 0.001, Fraction(1, 3), 0.25):
            q = 1 - Fraction(str(pfa)) if not isinstance(pfa, Fraction) else 1 - pfa
            expected = -((-q.numerator * (n - 1)) // q.denominator)          # ceil((1 - pfa)(n - 1)) exactly
            assert fa.higher_index(n, pfa) == expected
    # the float 0.001 is read as its decimal, not as its binary expansion
    assert fa.higher_index(1001, 0.001) == 999


def test_the_support_rule_is_ten_false_alarms():
    assert fa.ALPHA == Fraction(1, 1000)
    assert fa.MIN_FRAMES_PER_FALSE_ALARM == 10.0
    assert fa.SUPPORT_N == 10.0 / float(fa.ALPHA) == 10_000.0


def _law_sample(rng, days, per_day):
    q = fa.LAW.rvs(size=days * per_day, random_state=rng) / fa.LAW.mean()
    return q, np.repeat(np.arange(days), per_day)


def test_an_independent_population_has_unit_design_effect_and_is_supported():
    rng = np.random.default_rng(20260927)
    q, day = _law_sample(rng, 400, 60)
    block = fa.eta_pfa_block(q, day, "iid", seed=fa.band_seed(1))
    assert block["eta_pfa"] == float(np.sort(q)[fa.higher_index(q.size, fa.ALPHA)])
    assert block["exceedances"] <= q.size * float(fa.ALPHA) + 1
    assert 0.5 < block["deff"] < 2.0
    assert block["n_eff"] == pytest.approx(q.size / max(block["deff"], 1.0))
    assert block["status"] == "available" and block["n_eff"] >= fa.SUPPORT_N
    k_eff = block["achieved_pfa"] * block["n_eff"]
    assert block["cp95_upper_pfa"] == pytest.approx(stats.beta.ppf(0.95, k_eff + 1, block["n_eff"] - k_eff))
    low, high = block["boot_ci_eta"]
    assert low <= block["eta_pfa"] <= high


def test_day_correlated_exceedances_shrink_n_eff_below_support():
    rng = np.random.default_rng(7)
    q, day = _law_sample(rng, 400, 60)
    loud = rng.choice(400, size=4, replace=False)
    q = q.copy()
    q[np.isin(day, loud)] *= 3.0                  # whole days above the tail: exceedances cluster by day
    block = fa.eta_pfa_block(q, day, "clustered", seed=3)
    assert block["deff"] > 1.0 and block["n_eff"] < q.size
    assert block["status"] == f"unsupported: n_eff = {block['n_eff']:.0f}"


def test_the_bootstrap_is_seeded():
    rng = np.random.default_rng(1)
    q, day = _law_sample(rng, 50, 40)
    assert fa.eta_pfa_block(q, day, "a", seed=11) == {**fa.eta_pfa_block(q, day, "a", seed=11)}
    assert fa.band_seed(20) == fa.SEED + 120


def _population(*, verified=True, null_like=True, mask=None, off_from="2022-09", off_through=None, check=""):
    return nulls.OffPopulation(mask=mask, off_through=off_through, off_from=off_from, record_frames=0, off_frames=0,
                               note="", independently_verified=verified, external_record="confirmed" if verified else "",
                               null_like=null_like, null_check=check)


def test_signal_free_needs_both_tests():
    mask = np.ones(4, dtype=bool)
    assert _population(mask=mask).signal_free
    assert not _population(mask=mask, verified=False).signal_free
    assert not _population(mask=mask, null_like=False).signal_free
    assert not _population(mask=None).signal_free
    assert _population(mask=mask, null_like=False, check="core width factor 20.9 exceeds 5").rejection_reason == \
        "off-state, not signal-free: core width factor 20.9 exceeds 5"
    assert "fails independent_verification" in _population(mask=mask, verified=False).rejection_reason


def test_the_five_status_forms(monkeypatch):
    rng = np.random.default_rng(3)
    n = 24_000
    q = fa.LAW.rvs(size=n, random_state=rng) / fa.LAW.mean()
    time = np.repeat(np.arange(400), 60) * 86400.0 + 3600.0
    era = np.ones(n, dtype=bool)
    # available: a verified, null-like population in the current era
    ok = fa.false_alarm_limit(20, role="screened", off_population=_population(mask=era), q=q, frame_time=time,
                              current_era=era)
    assert ok["eta_pfa_status"] == fa.AVAILABLE and math.isfinite(ok["eta_pfa"])
    assert ok["null_source"].startswith("verified signal-free transmitter-off epoch")
    # unsupported: too few frames for 10 false alarms
    few = fa.false_alarm_limit(20, role="screened", off_population=_population(mask=era[:2000]), q=q[:2000],
                               frame_time=time[:2000], current_era=era[:2000])
    assert few["eta_pfa_status"].startswith("unsupported: n_eff = ")
    # unavailable: verified but not null-like; the current era is still measured, as a diagnostic
    bad = fa.false_alarm_limit(20, role="screened", off_population=_population(mask=era, null_like=False, check="c"),
                               q=q, frame_time=time, current_era=era)
    assert bad["eta_pfa_status"] == "unavailable: off-state, not signal-free: c" and math.isnan(bad["eta_pfa"])
    assert math.isfinite(bad["eta_pfa_diagnostic"]) and bad["eta_pfa_diagnostic_status"]
    # unavailable with no epoch at all
    none = fa.false_alarm_limit(14, role="screened", off_population=None)
    assert none["eta_pfa_status"] == f"unavailable: {nulls.NO_OFF_EPOCH}"
    assert none["null_source"] == "bulk of the mixture (declared)"
    # not defined: the control band
    control = fa.false_alarm_limit(37, role="control")
    assert control["eta_pfa_status"].startswith("not defined: control band (no transmitter)")
    assert "Q37(1e-3)" in control["eta_pfa_status"]
    # undefined: no alpha declared
    monkeypatch.setattr(fa, "ALPHA", None)
    assert fa.false_alarm_limit(14, role="screened")["eta_pfa_status"] == fa.UNDEFINED == "undefined: no alpha declared"


def test_the_latest_era_principle():
    """A signal-free population that is not in the current era has no eta_Pfa."""
    n = 1000
    old = np.zeros(n, dtype=bool)
    old[:400] = True
    current = ~old
    out = fa.false_alarm_limit(35, role="screened", off_population=_population(mask=old, off_from=None, off_through="2021-10"),
                               q=np.ones(n), frame_time=np.arange(n) * 60.0, current_era=current)
    assert out["eta_pfa_status"] == ("unavailable: transmitter-off epoch through 2021-10 is not the current era "
                                     "(latest-era principle)")
    unverified = _population(mask=old, off_from=None, off_through="2021-10", verified=False)
    out = fa.false_alarm_limit(35, role="screened", off_population=unverified, q=np.ones(n),
                               frame_time=np.arange(n) * 60.0, current_era=current)
    assert out["eta_pfa_status"].startswith("unavailable: no verified signal-free population (")
    assert "not the current era (latest-era principle)" in out["eta_pfa_status"]
