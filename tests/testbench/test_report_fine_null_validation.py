"""Frozen calibration to independent H0 report, plus exact boundary regressions."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPORT = load("report_fine_null_validation_tested", ROOT / "tools/report_fine_null_validation.py")
FIXTURES = load("fine_null_report_reducer_fixtures", ROOT / "tests/testbench/test_reduce_fine_validation.py")
REDUCER = FIXTURES.REDUCER


def write(path, value):
    path.write_text(json.dumps(value))


def study(tmp_path):
    """Two deterministic fake-power observations, with actual reducer/calibrator."""
    out = FIXTURES.fixture(tmp_path)
    path = out / "trial-identities.npy"
    ids = np.load(path, allow_pickle=False)
    new = np.array([("null_validation", 2048, i, 120 + i, 120 + i,
        2 + (hashlib.sha256(f"{REDUCER.PLAN_SCHEMA}:fixture-choice:null_validation:{i}".encode()).digest()[0] & 1))
        for i in range(2)], dtype=ids.dtype)
    np.save(path, np.concatenate((ids, new)), allow_pickle=False)
    plan = REDUCER.read_json(out / "plan.json")
    plan["identities_sha256"] = REDUCER.sha(path)
    plan["counts"]["null_validation"] = 2
    plan["false_alarm_gate"] = {"confidence": .95, "family_tests": 1334,
        "cap_multiple": 2., "pointwise_width_max_multiple_nominal": 1.}
    write(out / "plan.json", plan)
    # Rebind this synthetic fixture before calibration, never after its freeze.
    for path in (out / "raw").rglob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files if name != "meta_json"}
            metadata = json.loads(str(data["meta_json"].item()))
        metadata["plan_sha256"] = REDUCER.sha(out / "plan.json")
        np.savez(path, **arrays, meta_json=np.asarray(json.dumps(metadata)))
        receipt = REDUCER.read_json(path.with_suffix(".json"))
        receipt.update(metadata=metadata, sha256=REDUCER.sha(path))
        write(path.with_suffix(".json"), receipt)
    REDUCER.score_stage(out, "null_calibration")
    REDUCER.calibrate(out)
    return out, plan


def validation(out, plan, *, special=False, no_exceedances=False):
    fine = np.full((2, 1, 3, 256), 100, dtype=np.uint64)
    coarse = np.full((2, 1, 3), 100, dtype=np.uint64)
    if not no_exceedances:
        fine[0, :, 0, 202] = 150
        coarse[0, :, 0] = 150
    if special:
        fine[0, :, 0, :] = 0
        fine[0, :, 0, 202] = 100  # positive designated / zero bulk = sentinel
        fine[1, :, 1:, :] = 0    # undefined rank, retained in denominator
        coarse[1, :, 1:] = 0
    arrays = {"fixed_fine": fine, "fixed_coarse": coarse,
        "float_fine": np.repeat(fine[:, :, None], 5, axis=2).astype(np.float64),
        "float_coarse": np.repeat(coarse[:, :, None], 5, axis=2).astype(np.float64),
        "clip_count": np.zeros((2, 1), dtype=np.uint32)}
    metadata = {"plan_sha256": REDUCER.sha(out / "plan.json"), "phase": "null_validation",
        "start": 0, "stop": 2, "channels": [14], "streams": 2048,
        "float_stages": plan["float_stages"], "raw_seeds": [120, 121],
        "axes": ["trial", "channel", "stage_if_float", "term", "bin_if_fine"]}
    path = out / "raw/null_validation/000000.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, **arrays, meta_json=np.asarray(json.dumps(metadata)))
    write(path.with_suffix(".json"), {"metadata": metadata, "sha256": REDUCER.sha(path),
                                    "completed_utc": REDUCER.utc()})
    return path


def test_two_trial_engineering_report_has_full_denominators_and_failing_gates(tmp_path):
    out, plan = study(tmp_path)
    validation(out, plan)
    result = REPORT.run(out)
    assert result["trials_per_policy"] == 2
    assert result["primary_family_size"] == 1334
    assert result["primary_policy_rows"] == result["failed_primary_rows"] == 58
    assert result["passed_primary_rows"] == 0
    assert result["all_primary_gates_passed"] is False
    assert 'ignores those labels' in result["null_fixture_labels"]
    assert len(result["rows"]) == 8 * 7 * 4 * 3 + 6 * 3
    for row in result["rows"]:
        assert row["exceedances"] == 1
        assert row["invalid_trials"] == 0
        assert row["valid_nonexceedances"] == 1
        assert row["statistics"]["pointwise"]["trials"] == 2
        if row["primary_family"]:
            assert row["gate_status"] == "failed"
            assert "pointwise_precision_not_met" in row["gate_failure_reasons"]
            assert "simultaneous_cap_not_demonstrated" in row["gate_failure_reasons"]
    path = out / "reports/null_validation/report.json"
    assert path.with_suffix(".sha256").read_text() == REDUCER.sha(path) + "\n"
    for key in ("inputs_sha256", "outputs_sha256", "source_sha256", "snapshots_sha256"):
        REDUCER.rehash(out, result[key])
    with pytest.raises(ValueError, match="overwrite"):
        REPORT.run(out)


def test_sentinel_and_invalid_observations_are_reported_without_science_exception(tmp_path):
    out, plan = study(tmp_path)
    validation(out, plan, special=True)
    result = REPORT.run(out)
    row = next(r for r in result["rows"] if r["kind"] == "fine" and r["stage"] == REPORT.Q16_STAGE)
    assert row["exceedances"] == 1
    assert row["always_masked_trials"] == 1
    assert row["invalid_trials"] == 1
    assert row["valid_nonexceedances"] == 0
    assert row["statistics"]["pointwise"]["trials"] == 2
    assert not result["all_primary_gates_passed"]


def test_zero_exceedances_remain_explicit_and_do_not_pass_two_trial_gates(tmp_path):
    out, plan = study(tmp_path)
    validation(out, plan, no_exceedances=True)
    result = REPORT.run(out)
    assert result["passed_primary_rows"] == 0
    for row in result["rows"]:
        assert row["exceedances"] == row["invalid_trials"] == 0
        assert row["valid_nonexceedances"] == 2
        assert row["statistics"]["pointwise"]["successes"] == 0
        assert row["statistics"]["pointwise"]["trials"] == 2
        assert row["statistics"]["pointwise"]["lower"] == 0
        assert row["statistics"]["pointwise"]["upper"] > 0


@pytest.mark.parametrize("threshold", [(1 << 53) + 1, (1 << 63) + 1, (1 << 64) - 2])
def test_exact_adjacent_uint64_boundaries_never_pass_through_float(threshold):
    samples = np.array([threshold - 1, threshold, threshold + 1, 0, 0], dtype=np.uint64)
    result = REPORT.count_outcomes(samples, np.array([True, True, True, True, False]),
                                  threshold, always=np.array([False, False, False, True, False]))
    assert result["exceedances"] == 2
    assert result["always_masked_trials"] == 1
    assert result["valid_nonexceedances"] == 2
    assert result["invalid_trials"] == 1


def test_largest_legal_threshold_keeps_its_tie_but_masks_logical_2_to_64():
    result = REPORT.count_outcomes(np.array([2**64-1, 0], dtype=np.uint64),
        np.ones(2, dtype=bool), 2**64-1, always=np.array([False, True]))
    assert result["exceedances"] == 1
    assert result["valid_nonexceedances"] == 1


@pytest.mark.parametrize("threshold", [float(2**64-1), True, 0, 2**64])
def test_noninteger_or_undeployable_q16_threshold_refused(threshold):
    with pytest.raises(ValueError, match="Q16 calibration threshold"):
        REPORT.count_outcomes(np.array([1], dtype=np.uint64), np.array([True]),
                              threshold, always=np.array([False]))


def test_zero_float_response_is_kept_nan_is_invalid_and_infinity_masks():
    result = REPORT.count_outcomes(np.array([0., np.nan, np.inf]), np.ones(3, dtype=bool), 0.)
    assert result["exceedances"] == result["valid_nonexceedances"] == result["invalid_trials"] == 1
    assert result["valid_zero_response_trials"] == result["undefined_float_trials"] == 1


@pytest.mark.parametrize("change", ["future", "naive", "early", "wrong_seed", "extra", "marker", "calibration_scope", "omitted_dependency", "source"])
def test_integrity_failures_refuse_before_report_creation(tmp_path, change):
    out, plan = study(tmp_path)
    raw = validation(out, plan)
    receipt = raw.with_suffix(".json")
    saved = REDUCER.read_json(receipt)
    if change in ("future", "naive", "early"):
        saved["completed_utc"] = {"future": "2099-01-01T00:00:00+00:00", "naive": "2021-01-01T00:00:00", "early": "2021-01-01T00:00:00+00:00"}[change]
        write(receipt, saved)
    elif change == "wrong_seed":
        saved["metadata"]["raw_seeds"][0] = 999
        write(receipt, saved)
    elif change == "extra":
        (raw.parent / "undeclared.npz").write_bytes(b"not science")
    elif change == "marker":
        (out / "calibration.sha256").write_text("0" * 64 + "\n")
    elif change in ("calibration_scope", "omitted_dependency"):
        calibration = REDUCER.read_json(out / "calibration.json")
        if change == "calibration_scope":
            calibration["coordinate_system"] = "normalized-reference"
        else:
            del calibration["calibration_inputs_sha256"]["scores/null_calibration/ch14.npz"]
        write(out / "calibration.json", calibration)
        (out / "calibration.sha256").write_text(REDUCER.sha(out / "calibration.json") + "\n")
    elif change == "source":
        (tmp_path / "generator-source.py").write_text("# altered after freeze\n")
    with pytest.raises(ValueError):
        REPORT.run(out)
    assert not (out / "reports/null_validation").exists()
