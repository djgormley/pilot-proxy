import numpy as np
import pytest

from pilot_proxy.characterization import surface as thresholds

Q16 = thresholds.Q16_SCALE


def _hist(counts=(32, 32, 36), systematic=(2.0, 6.0, 43.0),
          variance=None, bulk_size=10):
    return thresholds.ScoreHistogram(
        bulk_size=bulk_size,
        candidate_multiplier_q16=(Q16, 2 * Q16),
        counts=counts,
        systematic_residual_sums=systematic,
        variance_residual_sums=variance,
    )


def test_q16_builder_keeps_frames_at_the_exact_boundary():
    histogram = thresholds.build_score_histogram(
        [1, Q16, 2 * Q16, thresholds.ALWAYS_MASKED_Q16],
        [1.0, 2.0, 3.0, 4.0],
        [1, Q16, 2 * Q16],
        bulk_size=1,
        variance_residuals=[4.0, 3.0, 2.0, 1.0],
    )
    assert histogram.candidate_multiplier_q16 == (1, Q16, 2 * Q16)
    assert histogram.counts == (1, 1, 1, 1)
    assert histogram.systematic_residual_sums == (1.0, 2.0, 3.0, 4.0)
    assert histogram.variance_residual_sums == (4.0, 3.0, 2.0, 1.0)


@pytest.mark.parametrize("field", ["systematic_residuals",
                                    "variance_residuals"])
def test_builder_rejects_complex_values(field):
    kwargs = dict(
        required_multiplier_q16=np.array([Q16, 2 * Q16]),
        systematic_residuals=np.array([1.0, 2.0]),
        variance_residuals=np.array([1.0, 2.0]),
        candidate_multiplier_q16=(Q16, 2 * Q16),
        bulk_size=10,
    )
    kwargs[field] = np.array([1.0 + 1.0j, 2.0 + 0.0j])
    with pytest.raises(TypeError, match="real numbers"):
        thresholds.build_score_histogram(**kwargs)


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"systematic_residuals": np.array([-1.0, 1.0])},
         "non-negative and finite"),
        ({"variance_residuals": np.array([np.nan, 1.0])},
         "non-negative and finite"),
        ({"systematic_residuals": np.array([1.0])},
         "match required_multiplier_q16"),
        ({"systematic_residuals": np.array([[1.0, 2.0]])},
         "one-dimensional"),
    ],
)
def test_builder_rejects_invalid_arrays(updates, message):
    kwargs = dict(
        required_multiplier_q16=np.array([Q16, 2 * Q16]),
        systematic_residuals=np.array([1.0, 2.0]),
        variance_residuals=np.array([1.0, 2.0]),
        candidate_multiplier_q16=(Q16, 2 * Q16),
        bulk_size=10,
    )
    kwargs.update(updates)
    with pytest.raises((TypeError, ValueError), match=message):
        thresholds.build_score_histogram(**kwargs)


@pytest.mark.parametrize(
    "requirements,message",
    [([], "must not be empty"), ([0], "invalid decision boundary"),
     ([thresholds.ALWAYS_MASKED_Q16 + 1], "invalid decision boundary"),
     ([1.5], "integers"), ([True], "integers")],
)
def test_builder_rejects_invalid_exact_boundaries(requirements, message):
    with pytest.raises((TypeError, ValueError), match=message):
        thresholds.build_score_histogram(
            requirements, [1.0] * len(requirements), (Q16,), bulk_size=10)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"candidate_multiplier_q16": ()}, "must not be empty"),
        ({"bulk_size": 0}, "positive"),
        ({"bulk_size": True}, "integer"),
        ({"candidate_multiplier_q16": (Q16, Q16)}, "strictly increasing"),
        ({"candidate_multiplier_q16": (0, Q16)}, "between 1"),
        ({"counts": (1, 2)}, "overflow"),
        ({"counts": (1.0, 2, 3)}, "integers"),
        ({"counts": (0, 0, 0)}, "at least one frame"),
        ({"systematic_residual_sums": (1.0, 2.0)}, "match counts"),
        ({"systematic_residual_sums": (1.0, -1.0, 2.0)}, "non-negative"),
        ({"variance_residual_sums": (1.0, 2.0)}, "match counts"),
        ({"counts": (0, 3, 7),
          "systematic_residual_sums": (0.1, 0.9, 4.1)}, "empty bin"),
        ({"counts": (0, 3, 7),
          "systematic_residual_sums": (0.0, 0.9, 4.1),
          "variance_residual_sums": (0.1, 0.9, 4.1)}, "empty bin"),
    ],
)
def test_histogram_rejects_malformed_inputs(kwargs, message):
    base = dict(
        bulk_size=10,
        candidate_multiplier_q16=(Q16, 2 * Q16),
        counts=(40, 20, 40),
        systematic_residual_sums=(4.0, 9.2, 37.8),
    )
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        thresholds.ScoreHistogram(**base)
