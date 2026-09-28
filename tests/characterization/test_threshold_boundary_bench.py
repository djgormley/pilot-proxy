"""Malformed container, exact eligibility and cumulative-population boundaries."""
import numpy as np
import pytest

from pilot_proxy.characterization import surface as t


def _hist(**changes):
    values = dict(bulk_size=2, candidate_multiplier_q16=(1, 65536), counts=(30, 30, 40),
                   systematic_residual_sums=(3.0, 9.0, 20.0), variance_residual_sums=(6.0, 12.0, 20.0))
    values.update(changes)
    return t.ScoreHistogram(**values)


@pytest.mark.parametrize("field,value", [
    ("counts", "100"), ("counts", None), ("counts", (-1, 61, 40)),
    ("systematic_residual_sums", "123"), ("systematic_residual_sums", None),
    ("systematic_residual_sums", (True, 1, 2)), ("systematic_residual_sums", (1, np.inf, 2)),
    ("candidate_multiplier_q16", "123"), ("candidate_multiplier_q16", None),
    ("candidate_multiplier_q16", (1.0, 2)),
    ("candidate_eligible", (True,)), ("candidate_eligible", (1, 0)),
])
def test_histogram_refuses_malformed_containers_without_coercion(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        _hist(**{field: value})


@pytest.mark.parametrize("requirements,candidates,scores", [
    ("1", [1], [1]), (None, [1], [1]), ([1], [], [1]),
    ([1], [2, 1], [1]), ([1], [1], ["1"]),
])
def test_builder_refuses_invalid_boundaries_and_text_scores(requirements, candidates, scores):
    with pytest.raises((TypeError, ValueError)):
        t.build_score_histogram(requirements, scores, candidates, bulk_size=1)
