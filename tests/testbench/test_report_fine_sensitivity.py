import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import report_fine_sensitivity as R
from pilot_proxy.testbench.fine_validation_stats import empirical_null_threshold, exact_q16_null_threshold


def test_multiplicities_reproduce_the_same_explicit_bootstrap_draws():
    counts = R.resampling(np.arange(7), np.arange(100, 111), replicates=20, seed=214)
    rng = np.random.default_rng(214)
    for i in range(20):
        np.testing.assert_array_equal(counts["null_counts"][i], np.bincount(rng.integers(7, size=7), minlength=7))
        np.testing.assert_array_equal(counts["h1_counts"][i], np.bincount(rng.integers(11, size=11), minlength=11))
    same = R.resampling(np.arange(7), np.arange(100, 111), replicates=20, seed=214)
    np.testing.assert_array_equal(counts["h1_counts"], same["h1_counts"])


@pytest.mark.parametrize("exact", [False, True])
def test_weighted_thresholds_and_rates_equal_expanded_multisets(exact):
    weights = R.resampling(np.arange(8), np.arange(100, 112), replicates=40, seed=9)
    null = np.array([7, 2, 2, 3, 8, 5, 1, 4], dtype=object if exact else float)
    sample = np.array([np.arange(1, 13), np.arange(3, 15)], dtype=object if exact else float)
    if exact:
        null[-1] = R.SENTINEL
        sample[0, 0] = R.SENTINEL
        null[0] = (1 << 63) + 1
    thresholds, available = R.weighted_thresholds(null, weights["null_counts"], pfa=.25, exact=exact)
    rates = R.bootstrap_rates(sample, thresholds, available, weights["h1_counts"], exact=exact, batch=7)
    calibrator = exact_q16_null_threshold if exact else empirical_null_threshold
    for b in range(40):
        expanded_null = np.repeat(null, weights["null_counts"][b])
        record = calibrator(expanded_null, pfa=.25)
        assert available[b] == (record["status"] == "available")
        if available[b]:
            assert thresholds[b] == record["threshold"]
            expected = np.mean(np.repeat(sample, weights["h1_counts"][b], axis=1) > record["threshold"], axis=1)
            np.testing.assert_array_equal(rates[b], expected)
        else:
            assert np.isnan(rates[b]).all()


def test_uint64_adjacent_threshold_ties_and_always_masked_decisions():
    samples = np.array([[2**63, 0], [2**63 + 1, 0]], dtype=np.uint64)
    valid = np.array([[True, False], [True, True]])
    always = np.array([[False, False], [False, True]])
    result = R.curve_summary(samples, valid, 2**63, [-50, -49], exact=True, always=always)
    assert result["counts"] == [1, 1]
    assert result["invalid_trials"] == [0, 1]
    assert result["trials"] == 2 and not result["response_qualified"]


def test_pairing_retains_identical_stage_zero_loss_and_censoring():
    snrs = np.arange(-3, 4, dtype=float)
    null = np.linspace(1, 2, 20)
    h1 = np.tile((snrs + 4)[:, None], (1, 32))
    weights = R.resampling(np.arange(20), np.arange(100, 132), replicates=30, seed=5)
    samples0, samples1, qualification, points = {}, {}, {}, {"0.05": {}}
    for stage in R.STAGES:
        exact = stage == R.Q16_STAGE
        samples0[stage] = np.array(np.ceil(null * 65536), dtype=object) if exact else null
        if exact:
            samples0[stage] = np.array([int(v) for v in samples0[stage]], dtype=object)
            samples1[stage] = np.array([[int(v) for v in row] for row in np.ceil(h1 * 65536)], dtype=object)
        else:
            samples1[stage] = h1
        qualification[stage] = True
        rates = np.mean(h1 > 2, axis=1)
        points["0.05"][stage] = {str(t): R.raw_crossing_brackets(snrs, rates, target=t) for t in (.5, .9)}
    arrays, summaries = R.paired_ladder(samples0, samples1, qualification, points, snrs, weights, [.05])
    assert np.all(arrays["loss_valid"])
    assert np.nanmax(np.abs(arrays["loss_db"])) == 0
    assert all(v["pointwise_percentile95_db"] == [0, 0] for v in summaries)
    qualification[R.STAGES[1]] = False
    arrays2, summaries2 = R.paired_ladder(samples0, samples1, qualification, points, snrs, weights, [.05])
    assert not arrays2["loss_valid"][:, :, 7].any()  # total ATSC-to-Q16
    total = [v for v in summaries2 if v["pair"] == "total_atsc_to_q16"]
    assert all(v["pointwise_percentile95_db"] is None and v["valid_replicates"] == 0 for v in total)


def test_ambiguous_and_unbracketed_bootstrap_rows_remain_censored():
    values = np.array([[0, .8, .2, 1], [0, .1, .2, .3], [0, .2, .8, 1]])
    estimates, status = R.crossing_arrays([0, 1, 2, 3], values, .5)
    assert status.tolist() == [2, 1, 0]
    assert np.isnan(estimates[:2]).all() and estimates[2] == pytest.approx(1.5)


def test_raw_grid_discretization_is_outer_difference_not_point_difference():
    a = R.raw_crossing_brackets([-3, 0, 3], [0, .7, 1], target=.5)
    b = R.raw_crossing_brackets([-3, 0, 3], [0, .3, 1], target=.5)
    bound = R.loss_bounds(a, b)
    assert bound["lower_db"] == 0 and bound["upper_db"] == 6
    assert 0 < bound["point_loss_db"] < 6


def engineering_cohort(tmp_path):
    spec = importlib.util.spec_from_file_location("sensitivity_fixture_builder", ROOT / "tests/testbench/test_reduce_fine_validation.py")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    out = helper.fixture(tmp_path)
    plan = R.read_json(out / "plan.json")
    ids = np.load(out / "trial-identities.npy", allow_pickle=False)
    extra = []
    for i in range(2):
        identity = f"literal-fine-validation-campaign-v1:fixture-choice:evaluation:{i}".encode()
        payload = 2 + (R.hashlib.sha256(identity).digest()[0] & 1)
        extra.append(("evaluation", 2048, i, 300 + i, 300 + i, payload))
    ids = np.concatenate([ids, np.array(extra, dtype=ids.dtype)])
    np.save(out / "trial-identities.npy", ids, allow_pickle=False)
    plan["identities_sha256"] = R.sha(out / "trial-identities.npy")
    plan["counts"]["evaluation"] = 2
    plan["loss_rule"] = {"bootstrap_replicates": 12}
    helper.write(out / "plan.json", plan)
    # These are unsealed test fixtures, not scientific artifacts.
    for path in (out / "raw").rglob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files if k != "meta_json"}
            metadata = json.loads(str(data["meta_json"].item()))
        metadata["plan_sha256"] = R.sha(out / "plan.json")
        np.savez(path, **arrays, meta_json=np.asarray(json.dumps(metadata)))
        receipt = R.read_json(path.with_suffix(".json"))
        receipt.update(metadata=metadata, sha256=R.sha(path))
        helper.write(path.with_suffix(".json"), receipt)
    D = helper.REDUCER
    D.score_stage(out, "null_calibration")
    D.calibrate(out)
    D.score_stage(out, "discovery")
    grid = D.freeze_grid(out)
    for case in range(7):
        snrs = grid["grids"][f"ch14/case{case}"]
        fine = np.full((2, len(snrs), 3, 256), 100, dtype=np.uint64)
        coarse = np.full((2, len(snrs), 3), 100, dtype=np.uint64)
        for j, snr in enumerate(snrs):
            fine[:, j, 0, 202] = 100 if snr < -49.5 else 150
            coarse[:, j, 0] = 100 if snr < -49.5 else 150
        metadata = {"plan_sha256": R.sha(out / "plan.json"), "phase": "evaluation", "channel": 14,
            "case_index": case, "streams": 2048, "start": 0, "stop": 2, "snr_db": snrs,
            "float_stages": plan["float_stages"], "raw_seeds": [300, 301], "payloads": [v[-1] for v in extra],
            "axes": ["trial", "snr", "stage_if_float", "term", "bin_if_fine"],
            "evaluation_grid_sha256": R.sha(out / "evaluation-grid.json")}
        path = out / "raw/evaluation" / f"ch14_case{case}_M2048_000000.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, fixed_fine=fine, fixed_coarse=coarse,
                 float_fine=np.repeat(fine[:, :, None], 5, axis=2).astype(float),
                 float_coarse=np.repeat(coarse[:, :, None], 5, axis=2).astype(float),
                 clip_count=np.zeros((2, len(snrs)), dtype=np.uint32), meta_json=np.asarray(json.dumps(metadata)))
        helper.write(path.with_suffix(".json"), {"metadata": metadata, "sha256": R.sha(path), "completed_utc": R.utc()})
    return out


def test_complete_small_native_cohort_end_to_end(tmp_path):
    out = engineering_cohort(tmp_path)
    report = R.run(out, "evaluation")
    assert report["coverage_complete"] and len(report["cells"]) == 7
    assert report["all_profile_equivalence"] == "not established"
    assert report["trials_per_cell_snr"] == 2
    assert not report["thresholds_changed"] and not report["physical_certification"]
    seeds = set()
    for cell in report["cells"]:
        saved = R.read_json(out / cell["report"])
        assert len(saved["fine_curves"]) == 168 and len(saved["coarse_curves"]) == 18
        assert len(saved["paired_contrasts"]) == 32
        seeds.add(saved["bootstrap_seed"])
        assert all(not item["equivalence_demonstrated"] for item in saved["paired_contrasts"])
    assert len(seeds) == 1
    R.rehash(out, report["outputs_sha256"])


def test_missing_heldout_cell_refused_before_report_creation(tmp_path):
    out = engineering_cohort(tmp_path)
    (out / "raw/evaluation/ch14_case6_M2048_000000.json").unlink()
    with pytest.raises(FileNotFoundError):
        R.run(out, "evaluation")
    assert not (out / "reports/evaluation").exists()


def test_shared_or_duplicated_noise_ids_cannot_be_independent_bootstrap():
    with pytest.raises(ValueError, match="Unique disjoint"):
        R.resampling(np.array([1, 2]), np.array([2, 3]), replicates=3, seed=1)
    with pytest.raises(ValueError, match="Unique disjoint"):
        R.resampling(np.array([1, 1]), np.array([2, 3]), replicates=3, seed=1)


def test_authenticated_wrong_raw_dtype_refused_before_any_report(tmp_path):
    out = engineering_cohort(tmp_path)
    path = out / "raw/evaluation/ch14_case6_M2048_000000.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    data["fixed_fine"] = data["fixed_fine"].astype(np.float64)
    np.savez(path, **data)
    receipt = R.read_json(path.with_suffix(".json"))
    receipt["sha256"] = R.sha(path)
    path.with_suffix(".json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="header shape/dtype"):
        R.run(out, "evaluation")
    assert not (out / "reports/evaluation").exists()
