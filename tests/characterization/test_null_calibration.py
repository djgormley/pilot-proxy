"""Independent identities and boundary tests for prospective null diagnostics."""
import numpy as np
import pytest
from scipy.stats import beta, binom

from pilot_proxy.characterization.false_alarm import (
    accepted_count_limit, calibration_reference, family_power_lower_bound,
    gate_count_limit, higher_rank, prospective_power,
)


def test_exact_order_rank_at_decimal_boundary():
    assert higher_rank(4000, .01) == 3961
    assert higher_rank(4000, .05) == 3801
    assert higher_rank(4000, .001) == 3997
    assert higher_rank(1, .01) == 1
    with pytest.raises(ValueError):
        higher_rank(0, .01)


def test_order_stat_distribution_matches_binomial_event():
    ref = calibration_reference(4000, .01)
    assert (ref['beta_a'], ref['beta_b']) == (40, 3961)
    for p in [.005, .01, .02]:
        # P(p_true > p) = P(at least r calibration values <= F^{-1}(1-p)).
        assert np.isclose(beta.sf(p, 40, 3961), binom.sf(3960, 4000, 1-p), atol=1e-13)


def test_count_limit_straddles_the_exact_upper_bound():
    k = accepted_count_limit(10000, .02)
    assert k == 146
    assert beta.isf(.05/1334, k+1, 10000-k) <= .02
    assert beta.isf(.05/1334, k+2, 9999-k) > .02
    assert gate_count_limit(10000, .01) == k
    assert accepted_count_limit(10, .02) == -1


def test_prediction_includes_calibration_randomness():
    power = prospective_power(4000, 10000, .01)['single_policy_power']
    assert .9887 < power < .9889
    fixed_p_power = binom.cdf(146, 10000, .01)
    assert fixed_p_power > .99999
    assert power < fixed_p_power


def test_family_bound_does_not_multiply_marginal_powers():
    assert family_power_lower_bound([.01, .02]) == pytest.approx(.97)
    assert family_power_lower_bound([.1]*11) == 0
    assert family_power_lower_bound([]) == 1
    with pytest.raises(ValueError):
        family_power_lower_bound([float('nan')])
