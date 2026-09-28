import datetime as dt
import json
import time

import numpy as np
import pytest

from pilot_proxy.characterization import stability as preparation
from pilot_proxy.characterization import surface as thresholds
from pilot_proxy.detectors.narrowband_marker import scores


Q16 = thresholds.Q16_SCALE
Q_GRID = (1, 2 * Q16)
ETA_GRID = tuple(value / Q16 for value in Q_GRID)
EVIDENCE_SHA = "1" * 64
SOURCE_ID = "sha256:" + "2" * 64


def _hist(counts=(40, 40, 20), systematic=(4.0, 4.0, 12.0),
          variance=None, bulk_size=1, q16=Q_GRID):
    return thresholds.ScoreHistogram(
        bulk_size=bulk_size,
        candidate_multiplier_q16=q16,
        counts=counts,
        systematic_residual_sums=systematic,
        variance_residual_sums=variance,
    )


def _support(frames=100, months=12, days=365.0):
    return preparation.EraHalfSupport(
        frame_count=frames,
        acquisition_count=50,
        observed_months=months,
        span_days=days,
    )


def _assessment(early=None, late=None, **kwargs):
    return preparation.assess_histogram_stability(
        {1: _hist() if early is None else early},
        {1: _hist() if late is None else late},
        early_support=_support(),
        late_support=_support(),
        max_cost_ratio=kwargs.pop("max_cost_ratio", 1.0),
        max_systematic_residual_ratio=kwargs.pop(
            "max_systematic_residual_ratio", 1.0),
        minimum_half_retained_frames=kwargs.pop(
            "minimum_half_retained_frames", 30),
        **kwargs,
    )


def _evidence(state="measured"):
    values = dict(state=state, method="test method", source="test source")
    if state in {"measured", "bounded"}:
        values["artifact_sha256"] = EVIDENCE_SHA
    if state == "bounded":
        values.update(bounds=(None, 1.0), units="ratio")
    return preparation.CalibrationEvidence(**values)


def _family(stability=None, transfer="measured", correlation="measured",
            score="measured", **kwargs):
    values = dict(
        histograms_by_rho={1: _hist()},
        source_id=SOURCE_ID,
        era_label="2024-01..2026-07",
        latest_era=True,
        valid_frames_only=True,
        equal_exposure_frames=True,
        additive_residuals=True,
        stability=_assessment() if stability is None else stability,
        score=_evidence(score),
        correlation=_evidence(correlation),
        transfer=_evidence(transfer),
    )
    values.update(kwargs)
    return preparation.PreparedFamily(**values)


def _block_family(*, block_size=4, keep_blocks=5, block_count=10,
                  systematic=1.0, cost_limit=10.0, residual_limit=10.0,
                  seed=31, replicates=200, coverage=0.8,
                  singleton_blocks=False, order=None,
                  masked_requirement=2 * Q16, block_unit="acquisition",
                  shared_block_labels=False):
    requirements = []
    times = []
    blocks = []
    for half, offset in (("early", 0), ("late", 100)):
        for block in range(block_count):
            for row in range(block_size):
                requirements.append(
                    1 if block < keep_blocks else masked_requirement)
                times.append((offset + block * block_size + row) * 86400.0)
                suffix = f"-{row}" if singleton_blocks else ""
                prefix = "" if shared_block_labels else f"{half}-"
                blocks.append(f"{prefix}{block}{suffix}")
    n_rows = len(requirements)
    if order is None:
        order = np.arange(n_rows)
    order = np.asarray(order)
    requirements = np.asarray(requirements, dtype=object)[order].tolist()
    times = np.asarray(times)[order]
    blocks = np.asarray(blocks)[order].tolist()
    residuals = (np.full(n_rows, systematic) if np.isscalar(systematic)
                 else np.asarray(systematic))[order]
    acquisitions = (blocks if block_unit == "acquisition" else
                    np.arange(n_rows)[order])
    return preparation.prepare_family(
        {1: requirements}, residuals,
        frame_times=times,
        acquisition_ids=acquisitions,
        exposure_seconds=np.ones(n_rows),
        source_id=SOURCE_ID,
        era_label="test era",
        latest_era=True,
        additive_residuals=True,
        score=_evidence(),
        correlation=_evidence(),
        transfer=_evidence(),
        max_cost_ratio=cost_limit,
        max_systematic_residual_ratio=residual_limit,
        minimum_half_retained_frames=10,
        minimum_observed_months=1,
        minimum_span_days=10.0,
        stability_block_ids=blocks,
        stability_resampling=preparation.BlockResamplingPlan(
            block_unit=block_unit,
            seed=seed,
            replicates=replicates,
            interval_coverage=coverage,
            minimum_blocks_per_half=5,
        ),
    )


def test_candidate_rank_grid_covers_every_supported_order_statistic():
    assert preparation.candidate_rho_values(4) == (1, 2, 3, 4)
    with pytest.raises(ValueError, match="positive"):
        preparation.candidate_rho_values(0)


def test_detector_rank_maps_to_one_based_rho():
    assert scores.rank_from_index(0) == 1
    assert scores.rank_from_index(63) == 64
    with pytest.raises(TypeError, match="cfar_rank must be an integer"):
        scores.rank_from_index(True)
    with pytest.raises(ValueError, match="cfar_rank must be non-negative"):
        scores.rank_from_index(-1)


def test_candidate_multiplier_grid_is_the_exact_q16_staircase():
    requirements = [1, Q16, Q16, 3 * Q16,
                    thresholds.ALWAYS_MASKED_Q16]
    assert preparation.candidate_eta_q16(requirements) == (
        1, Q16, 3 * Q16)


@pytest.mark.parametrize(
    "requirements",
    [[], [1, 0], [1, thresholds.ALWAYS_MASKED_Q16 + 1], [1, 1.5]],
)
def test_candidate_multiplier_grid_rejects_invalid_boundaries(requirements):
    with pytest.raises((TypeError, ValueError)):
        preparation.candidate_eta_q16(requirements)


def test_identical_candidate_surfaces_pass_the_drift_screen():
    result = _assessment()
    assert result.status == "passed"
    assert result.points_checked == 2
    assert result.points_skipped == 0
    assert result.maximum_cost_ratio == pytest.approx(1.0)
    assert result.maximum_systematic_residual_ratio == pytest.approx(1.0)
    assert result.maximum_masked_fraction_difference == pytest.approx(0.0)


def test_whole_blocks_preserve_dependence_in_the_upper_bound():
    grouped = _block_family()
    independent = _block_family(singleton_blocks=True)
    grouped_bound = grouped.stability.block_uncertainty
    independent_bound = independent.stability.block_uncertainty
    assert grouped_bound.passed
    assert independent_bound.passed
    assert (grouped_bound.maximum_cost_ratio_upper_bound
            > independent_bound.maximum_cost_ratio_upper_bound)


def test_block_surface_matches_direct_frame_resampling():
    q2 = 2 * Q16
    q3 = 3 * Q16
    patterns = {
        1: (1, 1, 1, q2, q2, q3),
        2: (1, 1, 1, q2, q3, q3),
    }
    requirements = {1: [], 2: []}
    blocks = []
    times = []
    for half, offset in (("early", 0), ("late", 100)):
        for block in range(6):
            blocks.extend([f"{half}-{block}"] * 6)
            times.extend((offset + block * 6 + row) * 86400.0
                         for row in range(6))
            for rho, pattern in patterns.items():
                requirements[rho].extend(pattern)
    n_rows = len(blocks)
    systematic = 0.5 + 0.1 * (np.arange(n_rows) % 7)
    variance = 0.2 + 0.05 * (np.arange(n_rows) % 5)
    plan = preparation.BlockResamplingPlan(
        "acquisition", seed=17, replicates=40,
        interval_coverage=0.5, minimum_blocks_per_half=5)
    family = preparation.prepare_family(
        requirements, systematic,
        variance_residuals=variance,
        frame_times=times,
        acquisition_ids=blocks,
        exposure_seconds=np.ones(n_rows),
        source_id=SOURCE_ID,
        era_label="test era",
        latest_era=True,
        additive_residuals=True,
        score=_evidence(),
        correlation=_evidence(),
        transfer=_evidence(),
        max_cost_ratio=100.0,
        max_systematic_residual_ratio=100.0,
        minimum_half_retained_frames=5,
        minimum_observed_months=1,
        minimum_span_days=20.0,
        stability_block_ids=blocks,
        stability_resampling=plan,
    )

    early_labels = [f"early-{index}" for index in range(6)]
    late_labels = [f"late-{index}" for index in range(6)]
    block_rows = {
        label: np.flatnonzero(np.asarray(blocks) == label)
        for label in early_labels + late_labels
    }
    rng = np.random.default_rng(plan.seed)
    probability = np.full(6, 1.0 / 6.0)
    early_draws = rng.multinomial(6, probability, size=plan.replicates)
    late_draws = rng.multinomial(6, probability, size=plan.replicates)
    candidates = []
    for rho, values in requirements.items():
        for candidate in preparation.candidate_eta_q16(values):
            if sum(value <= candidate for value in values) >= 30:
                candidates.append((rho, candidate))

    cost_maxima = []
    systematic_maxima = []
    for early_draw, late_draw in zip(early_draws, late_draws):
        early_rows = np.concatenate([
            np.tile(block_rows[label], count)
            for label, count in zip(early_labels, early_draw) if count
        ])
        late_rows = np.concatenate([
            np.tile(block_rows[label], count)
            for label, count in zip(late_labels, late_draw) if count
        ])
        cost_ratios = []
        systematic_ratios = []
        for rho, candidate in candidates:
            requirement = np.asarray(requirements[rho], dtype=object)
            early_kept = early_rows[requirement[early_rows] <= candidate]
            late_kept = late_rows[requirement[late_rows] <= candidate]
            assert len(early_kept) >= 5
            assert len(late_kept) >= 5
            early_cost = ((1.0 + variance[early_kept].mean())
                          / (len(early_kept) / len(early_rows)))
            late_cost = ((1.0 + variance[late_kept].mean())
                         / (len(late_kept) / len(late_rows)))
            early_systematic = systematic[early_kept].mean()
            late_systematic = systematic[late_kept].mean()
            cost_ratios.append(max(early_cost, late_cost)
                               / min(early_cost, late_cost))
            systematic_ratios.append(
                max(early_systematic, late_systematic)
                / min(early_systematic, late_systematic))
        cost_maxima.append(max(cost_ratios))
        systematic_maxima.append(max(systematic_ratios))

    index = int(np.ceil(plan.interval_coverage * plan.replicates)) - 1
    expected_cost = max(
        family.stability.maximum_cost_ratio,
        sorted(cost_maxima)[index])
    expected_systematic = max(
        family.stability.maximum_systematic_residual_ratio,
        sorted(systematic_maxima)[index])
    result = family.stability.block_uncertainty
    assert result.method.endswith("surface_maximum_upper_percentile")
    assert result.points_checked == len(candidates)
    assert result.maximum_cost_ratio_upper_bound == pytest.approx(expected_cost)
    assert (result.maximum_systematic_residual_ratio_upper_bound
            == pytest.approx(expected_systematic))


def test_block_sweep_scaling_is_subquadratic():
    def run(candidate_count):
        n_half = 4096
        half_requirements = np.arange(n_half) % candidate_count + 1
        requirements = np.concatenate(
            (half_requirements, half_requirements)).tolist()
        frame_times = np.concatenate((
            np.arange(n_half, dtype=float),
            2 * n_half + np.arange(n_half, dtype=float),
        )) * 86400.0
        blocks = []
        for half in ("early", "late"):
            blocks.extend(
                f"{half}-{index // 32}" for index in range(n_half))
        rows = np.arange(2 * n_half)
        return preparation.prepare_family(
            {1: requirements}, 0.5 + 0.01 * (rows % 7),
            variance_residuals=0.2 + 0.01 * (rows % 5),
            frame_times=frame_times,
            acquisition_ids=blocks,
            exposure_seconds=np.ones(2 * n_half),
            source_id=SOURCE_ID,
            era_label="scaling test",
            latest_era=True,
            additive_residuals=True,
            score=_evidence(),
            correlation=_evidence(),
            transfer=_evidence(),
            max_cost_ratio=100.0,
            max_systematic_residual_ratio=100.0,
            minimum_half_retained_frames=5,
            minimum_observed_months=1,
            minimum_span_days=1.0,
            stability_block_ids=blocks,
            stability_resampling=preparation.BlockResamplingPlan(
                "acquisition", seed=12, replicates=16,
                interval_coverage=0.5, minimum_blocks_per_half=5),
        )

    run(32)
    elapsed = {128: [], 4096: []}
    points_checked = {}
    for _ in range(3):
        for candidate_count in elapsed:
            started = time.perf_counter()
            family = run(candidate_count)
            elapsed[candidate_count].append(time.perf_counter() - started)
            points_checked[candidate_count] = (
                family.stability.block_uncertainty.points_checked)
    assert points_checked[128] == 128
    assert points_checked[4096] > 4000
    ratio = float(np.median(elapsed[4096]) / np.median(elapsed[128]))
    assert ratio < 4.0, f"32x candidates took {ratio:.3f}x"


def test_block_resampling_refuses_insufficient_blocks():
    family = _block_family(block_count=4, keep_blocks=2, block_size=10)
    result = family.stability.block_uncertainty
    assert result.status == "refused_insufficient_blocks"
    assert result.early_blocks == 4
    assert result.minimum_blocks_per_half == 5


def test_block_plan_has_no_hidden_tail_or_support_default():
    with pytest.raises(ValueError, match="at least two"):
        preparation.BlockResamplingPlan(
            "acquisition", seed=1, replicates=100,
            interval_coverage=0.8, minimum_blocks_per_half=1)
    with pytest.raises(ValueError, match="do not resolve"):
        preparation.BlockResamplingPlan(
            "acquisition", seed=1, replicates=10,
            interval_coverage=0.95, minimum_blocks_per_half=5)


def test_block_resampling_refuses_insufficient_successes():
    requirements = []
    times = []
    blocks = []
    for half, offset in (("early", 0), ("late", 100)):
        for block in range(10):
            rows = 20 if block == 0 else 2
            for row in range(rows):
                requirements.append(1 if block == 0 else 2 * Q16)
                times.append((offset + block + row / 100) * 86400.0)
                blocks.append(f"{half}-{block}")
    n_rows = len(requirements)
    family = preparation.prepare_family(
        {1: requirements}, np.ones(n_rows),
        frame_times=times,
        acquisition_ids=blocks,
        exposure_seconds=np.ones(n_rows),
        source_id=SOURCE_ID,
        era_label="test era",
        latest_era=True,
        additive_residuals=True,
        score=_evidence(),
        correlation=_evidence(),
        transfer=_evidence(),
        max_cost_ratio=100.0,
        max_systematic_residual_ratio=100.0,
        minimum_half_retained_frames=10,
        minimum_observed_months=1,
        minimum_span_days=5.0,
        stability_block_ids=blocks,
        stability_resampling=preparation.BlockResamplingPlan(
            "acquisition", seed=8, replicates=100,
            interval_coverage=0.8, minimum_blocks_per_half=5),
    )
    result = family.stability.block_uncertainty
    assert result.status == "refused_insufficient_resamples"
    assert result.replicates_succeeded == 45
    assert result.minimum_successful_replicates == 80


def test_block_resampling_is_reproducible_and_row_order_independent():
    first = _block_family(seed=61)
    order = np.random.default_rng(4).permutation(80)
    shuffled = _block_family(seed=61, order=order)
    assert first.stability.block_uncertainty == shuffled.stability.block_uncertainty


def test_zero_systematic_residual_has_unit_ratio_bound():
    family = _block_family(
        keep_blocks=10, systematic=0.0,
        cost_limit=1.0, residual_limit=1.0)
    result = family.stability.block_uncertainty
    assert result.passed
    assert result.maximum_cost_ratio_upper_bound == pytest.approx(1.0)
    assert (result.maximum_systematic_residual_ratio_upper_bound
            == pytest.approx(1.0))


def test_one_sided_zero_residual_draw_refuses_a_finite_bound():
    residuals = ([10.0] * 4 + [0.0] * 36) * 2
    family = _block_family(
        keep_blocks=10, systematic=residuals,
        cost_limit=2.0, residual_limit=100.0,
        seed=9, replicates=100)
    result = family.stability.block_uncertainty
    assert result.status == "refused_unbounded"
    assert result.maximum_systematic_residual_ratio_upper_bound is None
    json.dumps(family.metadata(), allow_nan=False)


def test_block_resampling_keeps_the_always_masked_sentinel():
    family = _block_family(
        masked_requirement=thresholds.ALWAYS_MASKED_Q16)
    assert family.stability.block_uncertainty.passed


def test_acquisition_blocks_must_match_acquisition_partition():
    with pytest.raises(ValueError, match="must match acquisition_ids"):
        preparation.prepare_family(
            {1: [1] * 40}, np.zeros(40),
            frame_times=np.arange(40) * 86400.0,
            acquisition_ids=range(40),
            exposure_seconds=np.ones(40),
            source_id=SOURCE_ID,
            era_label="test era",
            latest_era=True,
            additive_residuals=True,
            score=_evidence(),
            correlation=_evidence(),
            transfer=_evidence(),
            max_cost_ratio=1.0,
            max_systematic_residual_ratio=1.0,
            minimum_half_retained_frames=10,
            minimum_observed_months=1,
            minimum_span_days=5.0,
            stability_block_ids=[index // 2 for index in range(40)],
            stability_resampling=preparation.BlockResamplingPlan(
                "acquisition", seed=8, replicates=100,
                interval_coverage=0.8, minimum_blocks_per_half=5),
        )


def test_sidereal_day_blocks_are_supplied_independently():
    family = _block_family(block_unit="sidereal_day")
    assert family.stability.block_uncertainty.block_unit == "sidereal_day"
    assert family.stability.block_uncertainty.passed


def test_resampling_block_must_not_cross_the_calendar_split():
    family = _block_family(shared_block_labels=True)
    result = family.stability.block_uncertainty
    assert result.status == "refused_split_blocks"


def test_block_surface_uses_selector_support_ordering():
    family = _block_family(keep_blocks=1)
    assert family.stability.points_skipped == 1
    assert family.stability.points_checked == 1
    assert family.stability.block_uncertainty.points_checked == 1
    assert family.stability.block_uncertainty.passed


def test_mask_and_cost_drift_refuses_the_surface():
    result = _assessment(
        late=_hist(counts=(30, 40, 30), systematic=(3.0, 4.0, 13.0)),
        max_cost_ratio=1.1,
        max_systematic_residual_ratio=1.1,
    )
    assert result.status == "refused_drift"
    assert result.maximum_cost_ratio > 1.1
    assert result.worst_cost_point == (1, ETA_GRID[0])


def test_residual_only_drift_refuses_with_unchanged_counts():
    result = _assessment(
        late=_hist(systematic=(8.0, 8.0, 4.0)),
        max_cost_ratio=1.0,
        max_systematic_residual_ratio=1.5,
    )
    assert result.status == "refused_drift"
    assert result.maximum_masked_fraction_difference == pytest.approx(0.0)
    assert result.maximum_systematic_residual_ratio == pytest.approx(2.0)


@pytest.mark.parametrize(
    "updates",
    [{"max_cost_ratio": None},
     {"max_systematic_residual_ratio": None},
     {"minimum_half_retained_frames": None}],
)
def test_missing_drift_choice_is_a_structured_refusal(updates):
    result = _assessment(**updates)
    assert result.status == "refused_unconfigured"
    assert "must be declared" in result.reason


@pytest.mark.parametrize(
    "support, expected",
    [(_support(months=5), "observed months"),
     (_support(days=269.0), "spans")],
)
def test_sparse_era_half_refuses_before_surface_comparison(support, expected):
    result = preparation.assess_histogram_stability(
        {1: _hist()}, {1: _hist()},
        early_support=support, late_support=_support(),
        max_cost_ratio=1.1, max_systematic_residual_ratio=1.1,
        minimum_half_retained_frames=30)
    assert result.status == "refused_insufficient_support"
    assert expected in result.reason


def test_selector_unsupported_points_are_skipped_before_half_support_gate():
    thin = _hist(counts=(10, 30, 60), systematic=(1.0, 3.0, 16.0))
    result = _assessment(early=thin, late=thin)
    assert result.status == "passed"
    assert result.points_skipped == 1
    assert result.points_checked == 1


def test_evaluable_candidate_needs_the_declared_half_support():
    thin = _hist(counts=(20, 20, 60), systematic=(2.0, 2.0, 16.0))
    result = _assessment(early=thin, late=thin,
                         minimum_half_retained_frames=30)
    assert result.status == "passed"
    assert result.points_skipped == 1
    assert result.points_checked == 1


def test_identical_fine_staircases_do_not_refuse_supported_tail():
    hist = thresholds.ScoreHistogram(
        bulk_size=1, candidate_multiplier_q16=tuple(range(1, 101)),
        counts=(1,) * 100 + (0,), systematic_residual_sums=(0.1,) * 100 + (0.0,))
    result = _assessment(early=hist, late=hist)
    assert result.passed
    assert result.points_checked == 71
    assert result.maximum_cost_ratio == pytest.approx(1.0)
    assert result.maximum_systematic_residual_ratio == pytest.approx(1.0)


def test_stability_requires_matching_ranks_and_grids():
    with pytest.raises(ValueError, match="same ranks"):
        preparation.assess_histogram_stability(
            {1: _hist()}, {2: _hist()},
            early_support=_support(), late_support=_support(),
            max_cost_ratio=1.1, max_systematic_residual_ratio=1.1,
            minimum_half_retained_frames=30)
    other_grid = _hist(q16=(1, 3 * Q16))
    with pytest.raises(ValueError, match="same Q16 grid"):
        preparation.assess_histogram_stability(
            {1: _hist()}, {1: other_grid},
            early_support=_support(), late_support=_support(),
            max_cost_ratio=1.1, max_systematic_residual_ratio=1.1,
            minimum_half_retained_frames=30)
    first = (1 << 63) - 1
    second = 1 << 63
    same_float_a = _hist(q16=(first, thresholds.MAX_MULTIPLIER_Q16))
    same_float_b = _hist(q16=(second, thresholds.MAX_MULTIPLIER_Q16))
    assert same_float_a.candidate_eta == same_float_b.candidate_eta
    with pytest.raises(ValueError, match="same Q16 grid"):
        preparation.assess_histogram_stability(
            {1: same_float_a}, {1: same_float_b},
            early_support=_support(), late_support=_support(),
            max_cost_ratio=1.1, max_systematic_residual_ratio=1.1,
            minimum_half_retained_frames=30)


def test_prepared_family_requires_every_rank():
    with pytest.raises(ValueError, match="every supported rank"):
        _family(histograms_by_rho={1: _hist(bulk_size=2)})


def test_bounded_evidence_requires_a_structured_bound():
    with pytest.raises(ValueError, match="bounds and units"):
        preparation.CalibrationEvidence(
            "bounded", "test", "test", artifact_sha256=EVIDENCE_SHA)
    assert _evidence("bounded").bounds == (None, 1.0)


def test_factory_derives_grids_split_support_and_histograms():
    stamps = []
    for year in (2024, 2025):
        for month in range(1, 13):
            stamp = dt.datetime(
                year, month, 1, tzinfo=dt.timezone.utc).timestamp()
            stamps.extend([stamp] * 5)
    n = len(stamps)
    pattern = [1, Q16, 2 * Q16, thresholds.ALWAYS_MASKED_Q16, Q16]
    requirements = pattern * (n // len(pattern))
    family = preparation.prepare_family(
        {1: requirements, 2: requirements},
        np.full(n, 0.02),
        frame_times=stamps,
        acquisition_ids=range(n),
        exposure_seconds=np.full(n, 41.0),
        source_id=SOURCE_ID,
        era_label="2024-01..2025-12",
        latest_era=True,
        additive_residuals=True,
        score=_evidence(),
        correlation=_evidence(),
        transfer=_evidence("conditional"),
        max_cost_ratio=1.0,
        max_systematic_residual_ratio=1.0,
        minimum_half_retained_frames=10,
    )
    assert tuple(family.histograms_by_rho) == (1, 2)
    assert family.histograms_by_rho[1].candidate_multiplier_q16 == (
        1, Q16, 2 * Q16)
    assert family.stability.status == "passed"
    assert family.stability.early_support.observed_months == 12
    assert family.stability.late_support.observed_months == 12
    assert family.status == "screening"


def test_factory_rejects_unequal_exposure():
    with pytest.raises(ValueError, match="equal exposure"):
        preparation.prepare_family(
            {1: [1, Q16]}, [0.1, 0.1],
            frame_times=[0.0, 86400.0], acquisition_ids=[1, 2],
            exposure_seconds=[1.0, 2.0], source_id="test",
            era_label="test", latest_era=True, additive_residuals=True,
            score=_evidence(), correlation=_evidence(), transfer=_evidence(),
            max_cost_ratio=1.0, max_systematic_residual_ratio=1.0,
            minimum_half_retained_frames=1,
            minimum_observed_months=1, minimum_span_days=0.5)


def test_prepared_family_requires_content_pinned_source_rows():
    with pytest.raises(ValueError, match="sha256 content identifier"):
        _family(source_id="latest-era rows")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _family(source_id="sha256:not-a-digest")
