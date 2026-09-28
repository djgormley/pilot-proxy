"""Independent acquisition-level split, support, and resampling benches."""
import datetime as dt

import numpy as np
import pytest

from pilot_proxy.characterization import blocks


def _unix(text):
    return dt.datetime.fromisoformat(text).replace(tzinfo=dt.timezone.utc).timestamp()


def test_calendar_indices_cover_rollovers_leap_day_and_missing_times():
    times = [_unix("2024-02-29T23:59:59"), _unix("2024-03-01"),
             _unix("2025-01-01"), np.nan, np.inf, -np.inf]
    months = blocks.month_index(times)
    assert [blocks.month_label(m) for m in months[:3]] == ["2024-02", "2024-03", "2025-01"]
    np.testing.assert_array_equal(months[3:], [-1, -1, -1])
    days = blocks.day_index(times)
    assert days[1] - days[0] == 1
    np.testing.assert_array_equal(days[3:], [-1, -1, -1])
    assert blocks.day_index([-0.1])[0] == -1


def test_month_support_requires_all_three_denominators():
    times = _unix("2024-02-01") + np.repeat([0, 86400, 2 * 86400, 0, 86400], 6)
    units = np.repeat(np.arange(5), 6)
    selected = np.ones(30, dtype=bool)
    support = blocks.month_support(times, units, selected)
    assert [(s.label, s.frames, s.units, s.days) for s in support] == [("2024-02", 30, 5, 3)]
    selected[0] = False
    assert blocks.month_support(times, units, selected) == []
    assert blocks.month_support(times, np.zeros(30), np.ones(30, bool)) == []
    assert blocks.month_support(np.full(30, times[0]), units, np.ones(30, bool)) == []
    assert blocks.month_support(np.full(30, np.nan), units, np.ones(30, bool)) == []


@pytest.mark.parametrize("seed", range(10))
def test_split_preserves_acquisitions_selection_and_chronology(seed):
    rng = np.random.default_rng(seed)
    units = np.repeat([80, 2, 40, 9], [4, 13, 7, 5])
    times = np.repeat([400, 100, 300, 200], [4, 13, 7, 5]) + _unix("2024-01-01")
    selected = np.ones(units.size, dtype=bool)
    selected[-1] = False
    order = rng.permutation(units.size)
    units, times, selected = units[order], times[order], selected[order]
    result = blocks.split_blocks(units, times, selected, minimum_months=1,
                                  month_kwargs=dict(min_frames=1, min_units=1, min_days=1))
    assert result.status == "supported"
    assert set(units[result.calibration]) == {2, 9}
    assert set(units[result.evaluation]) == {40, 80}
    np.testing.assert_array_equal(result.calibration | result.evaluation, selected)
    assert not np.any(result.calibration & result.evaluation)
    assert result.calibration_frames + result.evaluation_frames == selected.sum()
    assert result.calibration_units == result.evaluation_units == 2
    assert times[result.calibration].max() < result.boundary_time == times[result.evaluation].min()


def test_empty_single_acquisition_and_under_supported_splits():
    empty = blocks.split_blocks([], [], [])
    assert empty.status == "empty" and empty.calibration_frames == empty.evaluation_frames == 0
    single = blocks.split_blocks([7] * 40, [0] * 40, [True] * 40)
    assert single.status == "insufficient_support"
    assert single.calibration_units == 0 and single.evaluation_units == 1
    assert single.evaluation_frames == 40
    sparse = blocks.split_blocks([7, 8], [0, 1], [True, True])
    assert sparse.status == "insufficient_support" and "populated months" in sparse.detail


def test_bootstrap_resamples_whole_acquisitions_and_records_reproducible_intervals():
    units = np.repeat([9, 42, 100], [2, 3, 4])
    selected = np.array([True] * 8 + [False])
    seen = []

    def statistic(weights):
        assert np.issubdtype(weights.dtype, np.integer)
        assert weights[-1] == 0
        multiplicities = [np.unique(weights[(units == u) & selected]) for u in [9, 42, 100]]
        assert all(len(m) == 1 for m in multiplicities)
        assert sum(int(m[0]) for m in multiplicities) == 3
        seen.append(weights.copy())
        return blocks.weighted_fraction(weights, units == 42)

    first = blocks.block_bootstrap(units, selected, statistic, min_blocks=3, replicates=60, seed=12,
                                    quantiles=(0.1, 0.5, 0.9))
    second = blocks.block_bootstrap(units, selected, statistic, min_blocks=3, replicates=60, seed=12,
                                     quantiles=(0.1, 0.5, 0.9))
    np.testing.assert_array_equal(seen[0], selected.astype(int))
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.estimate == 3 / 8
    np.testing.assert_allclose(first.values, np.quantile(first.samples, [0.1, 0.5, 0.9]))
    assert first.low == first.values[0] and first.high == first.values[-1]
    assert first.as_dict() == {"estimate": 3 / 8, "replicates": 60, "seed": 12, "blocks": 3,
                               "q0.1": first.values[0], "q0.5": first.values[1], "q0.9": first.values[2]}
    assert np.unique(first.samples).size > 1


def test_bootstrap_refuses_insufficient_independent_acquisitions():
    with pytest.raises(ValueError, match="at least 8 acquisitions; got 1"):
        blocks.block_bootstrap([1] * 100, [True] * 100, lambda w: w.sum())


def test_bootstrap_with_undefined_statistics_does_not_invent_bounds():
    result = blocks.block_bootstrap(range(8), [True] * 8, lambda w: np.nan, replicates=8)
    assert np.isnan(result.estimate) and np.isnan(result.samples).all()
    assert np.isnan(result.low) and np.isnan(result.high)


@pytest.mark.parametrize("seed", range(10))
def test_weighted_quantiles_match_an_expanded_sample_inverse_cdf(seed):
    rng = np.random.default_rng(seed)
    values = rng.integers(-5, 6, size=20).astype(float)
    weights = rng.integers(0, 6, size=20)
    values[0] = np.nan
    expanded = sorted(v for w, v in zip(weights, values) if np.isfinite(v) for _ in range(w))
    for q in (0.0, 0.1, 0.5, 0.9, 1.0):
        expected = expanded[max(0, int(np.ceil(q * len(expanded))) - 1)]
        assert blocks.weighted_quantile(weights, values, q) == expected
    assert np.isnan(blocks.weighted_quantile([0], [10], 0.5))
    assert np.isnan(blocks.weighted_fraction([0], [True]))
