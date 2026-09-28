"""The incumbent flaggers, and the blind spot the detector exists to cover.

The claim these tests hold in place is narrow and mechanical: a flagger that
estimates its own reference from the data cannot see an interferer that is
always on, because the interferer *is* the reference. Everything the policy
comparison concludes rests on that, so it is tested against synthetic data
where the answer is known rather than only against the survey products.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.characterization import flaggers as incumbent


def synth(n_acq=200, frames=16, rel_sigma=1e-3, seed=0):
    """Clean power: flat mean, radiometric scatter, split into acquisitions."""
    rng = np.random.default_rng(seed)
    n = n_acq * frames
    power = rng.normal(1.0, rel_sigma, n)
    unit = np.repeat(np.arange(n_acq), frames)
    return power, unit


def test_blocks_never_cross_acquisitions():
    _, unit = synth(n_acq=5, frames=7)
    blocks = incumbent.acquisition_blocks(unit, min_frames=2)
    assert len(blocks) == 5
    for a, b in blocks:
        assert len(np.unique(unit[a:b])) == 1
    # short acquisitions drop out rather than merging with a neighbour
    assert incumbent.acquisition_blocks(unit, min_frames=8) == []


def test_ragged_acquisitions_are_kept_separate():
    unit = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2])
    blocks = incumbent.acquisition_blocks(unit, min_frames=3)
    assert blocks == [(0, 3), (3, 8)]


def test_steady_interferer_is_invisible_to_both_incumbents():
    """The blind spot. A constant 10% excess in every frame passes clean."""
    power, unit = synth()
    dirty = power * 1.10                      # on in 100% of frames

    mad = incumbent.mad_flag(dirty, unit)
    sk, _, _ = incumbent.sk_flag(dirty, unit, min_frames=8)

    # both flag at their false-alarm rate and no more
    assert mad.mean() < 0.15
    assert sk.mean() < 0.15
    # and what survives still carries the interferer
    assert dirty[~mad].mean() == pytest.approx(1.10, rel=1e-3)
    assert dirty[~sk].mean() == pytest.approx(1.10, rel=1e-3)


def test_bursty_interferer_is_caught():
    """The regime where the incumbent already works, so the detector need not."""
    power, unit = synth(seed=1)
    dirty = power.copy()
    rng = np.random.default_rng(2)
    hot = rng.random(len(dirty)) < 0.02       # 2% duty, large amplitude
    dirty[hot] *= 3.0

    mad = incumbent.mad_flag(dirty, unit)
    assert mad[hot].mean() > 0.9              # nearly every burst flagged
    assert dirty[~mad].mean() == pytest.approx(1.0, rel=5e-3)


def test_duty_cycle_is_what_separates_the_two_regimes():
    """MAD survives a minority duty cycle and fails a majority one."""
    power, unit = synth(seed=3)
    rng = np.random.default_rng(4)
    caught = {}
    for duty in (0.1, 0.4, 0.7, 0.95):
        dirty = power.copy()
        on = rng.random(len(dirty)) < duty
        dirty[on] *= 1.10
        mad = incumbent.mad_flag(dirty, unit)
        caught[duty] = mad[on].mean() if on.sum() else 0.0
    assert caught[0.1] > 0.5                  # minority: flagged
    assert caught[0.95] < 0.1                 # majority: absorbed into baseline


@pytest.mark.parametrize("M,N", [(8, 1), (16, 2)])
def test_sk_sigma_matches_the_exact_null_variance(M, N):
    """Nita & Gary (2010): Var = 2 M^2 N (N+1) / ((M-1)(MN+2)(MN+3)).

    At these accumulation counts the large-N shortcut (M*N treated as one
    accumulation) sits well below the exact value, so this pins the exact
    expression rather than the limit.
    """
    exact = np.sqrt(2.0 * M**2 * N * (N + 1)
                    / ((M - 1) * (M * N + 2) * (M * N + 3)))
    assert incumbent.sk_sigma(M, N) == pytest.approx(exact, rel=1e-12)
    nm = M * N
    shortcut = np.sqrt(2.0 * nm * (nm + 1) / ((M - 1) * (nm + 2) * (nm + 3)))
    assert incumbent.sk_sigma(M, N) > shortcut


def test_sk_sigma_large_n_limit_matches_the_old_shortcut():
    """The calibrated path runs at large effective n_accum, where the exact
    expression and the old shortcut agree to O(1/N), so calibrated behavior
    is unchanged."""
    M, N = 8, 1e6
    nm = M * N
    shortcut = np.sqrt(2.0 * nm * (nm + 1) / ((M - 1) * (nm + 2) * (nm + 3)))
    assert incumbent.sk_sigma(M, N) == pytest.approx(shortcut, rel=1e-5)
    assert incumbent.sk_sigma(M, N) == pytest.approx(np.sqrt(2.0 / (M - 1)),
                                                    rel=1e-5)


def test_sk_null_calibration_centers_the_statistic():
    power, unit = synth(rel_sigma=5e-4, seed=5)
    n_accum = incumbent.calibrate_sk_null(power, unit, min_frames=8)
    sks = [incumbent.spectral_kurtosis(power[a:b], n_accum)
           for a, b in incumbent.acquisition_blocks(unit, 8)]
    assert np.median(sks) == pytest.approx(1.0, abs=0.02)


def test_sk_calibration_needs_something_to_calibrate_on():
    _, unit = synth(n_acq=3, frames=2)
    with pytest.raises(ValueError):
        incumbent.calibrate_sk_null(np.ones(6), unit, min_frames=8)


def test_scoring_charges_for_frames_removed():
    lin = np.full(100, 0.05)
    base = np.ones(100, bool)
    flag = np.zeros(100, bool)
    flag[:50] = True
    lin[:50] = 0.09                           # the flagger takes the dirty half
    r = incumbent.score_flagger("half", flag, lin, base)
    assert r.f == pytest.approx(0.5)
    assert r.r == pytest.approx(0.05)
    assert r.reduction_db == pytest.approx(10 * np.log10(0.07 / 0.05), rel=1e-6)


def test_scoring_handles_a_flagger_that_takes_everything():
    lin = np.full(10, 0.05)
    base = np.ones(10, bool)
    r = incumbent.score_flagger("all", np.ones(10, bool), lin, base)
    assert r.f == 1.0
    assert np.isinf(r.r)
    assert r.n_kept == 0
