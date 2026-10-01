import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("reduce_fine_validation_v1_tested", ROOT / "tools/reduce_fine_validation.py")
REDUCER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REDUCER
SPEC.loader.exec_module(REDUCER)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, *, invalid_null=False):
    out = tmp_path / "study"
    out.mkdir()
    cache = tmp_path / "cache"
    profile = {"physical_channel": 14, "layout": {"target_norm_sq": 100, "reference_norm_sum_sq": 200},
               "cases": [{"calibrated_anchor_bin_normal": 54, "nominal_anchor_bin_normal": 54,
                          "calibrated_anchor_bin_inverted": 202,
                          "nominal_anchor_bin_inverted": 202} for _ in range(7)]}
    write(cache / "plan.json", {"profiles": [profile]})
    source = tmp_path / "generator-source.py"
    source.write_text("# Explicit engineering fake-power generator fixture\n")
    dtype = [("phase", "U24"), ("streams", "u4"), ("trial", "u4"),
             ("raw_seed", "u8"), ("effective_seed63", "u8"), ("payload", "u1")]
    identities = [(phase, 2048, i, 100 + p * 10 + i, 100 + p * 10 + i, 0)
                  for p, phase in enumerate(("null_calibration", "discovery")) for i in range(2)]
    np.save(out / "trial-identities.npy", np.array(identities, dtype=dtype), allow_pickle=False)
    plan = {"schema": REDUCER.PLAN_SCHEMA, "scope": "engineering_only", "frozen_utc": "2020-01-01T00:00:00+00:00",
            "coordinate_system": "native-inverted-adapted",
            "channels": [14], "streams": 2048, "windows": 128, "window_samples": 128, "fine_bins": 256,
            "float_stages": list(REDUCER.FLOAT_STAGES[:-1]), "fixed_stage": REDUCER.FLOAT_STAGES[-1],
            "counts": {"null_calibration": 2, "discovery": 2},
            "primary_pfa": [0.01, 0.05], "diagnostic_pfa": [0.001],
            "geometry": {"designated_half_width": 2, "guard_native_bins": 1, "rank_fractions": [.25, .5, .75, 1.]},
            "evaluation_grid_rule": {"target_probabilities": [.5, .9], "pfa": [.01, .05], "max_points": 61},
            "discovery_snr_db": list(range(-66, -20, 3)), "source_sha256": {str(source): REDUCER.sha(source)},
            "identities_sha256": REDUCER.sha(out / "trial-identities.npy"), "cache_path": str(cache),
            "cache_sha256": {"plan.json": REDUCER.sha(cache / "plan.json")}}
    write(out / "plan.json", plan)
    for phase in ("null_calibration", "discovery"):
        ids = [i for i in identities if i[0] == phase]
        for case in [None] if phase == "null_calibration" else range(7):
            second = 1 if case is None else 16
            fine = np.full((2, second, 3, 256), 100, dtype=np.uint64)
            coarse = np.full((2, second, 3), 100, dtype=np.uint64)
            if case is None:
                fine[1, :, 0, 202] = 120
                coarse[1, :, 0] = 120
                if invalid_null:
                    fine[1, :, 1:, :] = 0
            else:
                for snr in range(second):
                    # Independent fixture observations at each SNR remain paired.
                    target = 100 if snr < 6 else 140
                    fine[:, snr, 0, 202] = target
                    coarse[:, snr, 0] = target
            arrays = {"fixed_fine": fine, "fixed_coarse": coarse,
                      "float_fine": np.repeat(fine[:, :, None], 5, axis=2).astype(np.float64),
                      "float_coarse": np.repeat(coarse[:, :, None], 5, axis=2).astype(np.float64),
                      "clip_count": np.zeros((2, second), dtype=np.uint32)}
            meta = {"plan_sha256": REDUCER.sha(out / "plan.json"), "phase": phase, "start": 0, "stop": 2,
                    "streams": 2048, "float_stages": plan["float_stages"], "raw_seeds": [i[3] for i in ids]}
            if case is None:
                name = "000000.npz"
                meta.update(channels=[14], axes=["trial", "channel", "stage_if_float", "term", "bin_if_fine"])
            else:
                name = f"ch14_case{case}_M2048_000000.npz"
                meta.update(channel=14, case_index=case, snr_db=plan["discovery_snr_db"], payloads=[0, 0],
                            axes=["trial", "snr", "stage_if_float", "term", "bin_if_fine"])
            path = out / "raw" / phase / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **arrays, meta_json=np.asarray(json.dumps(meta)))
            write(path.with_suffix(".json"), {"metadata": meta, "sha256": REDUCER.sha(path),
                                            "completed_utc": "2021-01-01T00:00:00+00:00"})
    return out


def test_engineering_power_files_end_to_end_complete_frozen_grid(tmp_path):
    out = fixture(tmp_path)
    null = REDUCER.score_stage(out, "null_calibration")
    assert null["coverage_complete"] and null["trials"] == 2
    calibration = REDUCER.calibrate(out)
    assert calibration["status"] == "calibrated"
    fine = calibration["fine"]["ch14"]["202"]
    assert fine[REDUCER.FLOAT_STAGES[0]]["62"]["0.01"]["threshold"] == pytest.approx(1.2)
    assert fine[REDUCER.Q16_STAGE]["62"]["0.01"]["threshold"] == 78644
    assert (out / "calibration.sha256").read_text().strip() == REDUCER.sha(out / "calibration.json")
    REDUCER.score_stage(out, "discovery")
    frozen = REDUCER.freeze_grid(out)
    assert frozen["status"] == "frozen" and frozen["expected_cells"] == 7
    assert frozen["evaluation_outcomes_used"] is False
    assert set(frozen["grids"]) == {f"ch14/case{i}" for i in range(7)}
    assert frozen["grids"]["ch14/case0"] == list(np.arange(-52.5, -46.49, .75))
    assert len(frozen["details"]["ch14/case0"]["curves"]) == 52
    assert (out / "evaluation-grid.sha256").read_text().strip() == REDUCER.sha(out / "evaluation-grid.json")
    REDUCER.rehash(out, frozen["calibration_inputs_sha256"])
    with pytest.raises(ValueError, match="overwrite"):
        REDUCER.score_stage(out, "null_calibration")


def test_invalid_null_frame_refuses_instead_of_trimming(tmp_path):
    out = fixture(tmp_path, invalid_null=True)
    REDUCER.score_stage(out, "null_calibration")
    with pytest.raises(ValueError, match="Calibration refused"):
        REDUCER.calibrate(out)
    report = REDUCER.read_json(out / "calibration-refused.json")
    assert report["null_trials"] == 2 and report["refusals"]
    assert not (out / "calibration.json").exists()
    assert not (out / "calibration.sha256").exists()


def test_missing_cell_refused_before_any_derived_discovery_output(tmp_path):
    out = fixture(tmp_path)
    (out / "raw/discovery/ch14_case6_M2048_000000.json").unlink()
    with pytest.raises(FileNotFoundError):
        REDUCER.score_stage(out, "discovery")
    assert not (out / "scores/discovery").exists()


def test_tampered_raw_and_wrong_seed_receipt_refused(tmp_path):
    out = fixture(tmp_path)
    receipt = out / "raw/null_calibration/000000.json"
    data = REDUCER.read_json(receipt)
    data["metadata"]["raw_seeds"][0] += 1
    write(receipt, data)
    with pytest.raises(ValueError, match="Raw identity differs"):
        REDUCER.score_stage(out, "null_calibration")


def test_source_mutation_refuses_later_calibration(tmp_path):
    out = fixture(tmp_path)
    REDUCER.score_stage(out, "null_calibration")
    (tmp_path / "generator-source.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="Bound input changed"):
        REDUCER.calibrate(out)


def test_grid_cannot_be_frozen_after_evaluation_begins(tmp_path):
    out = fixture(tmp_path)
    (out / "raw/evaluation").mkdir()
    (out / "raw/evaluation/uninspected.npz").write_bytes(b"metadata-only presence check")
    with pytest.raises(ValueError, match="before any evaluation"):
        REDUCER.freeze_grid(out)


def test_grid_preserves_downward_and_plateau_crossings_and_fallback():
    snrs = list(range(-66, -20, 3))
    rates = np.array([0, 1, .5, .5, .2] + [1] * 11)
    grid, details = REDUCER.curve_grid(snrs, {"oscillating": rates})
    records = details["curves"]
    half = next(r for r in records if r["target"] == .5)
    assert half["downward_brackets"] and half["plateau_segments"]
    assert -66 in grid and -52.5 in grid
    assert not details["fallback_discovery_grid"]
    grid2, details2 = REDUCER.curve_grid(snrs, {"oscillating": rates, "unbracketed": np.zeros(16)})
    assert details2["fallback_discovery_grid"]
    assert set(snrs) <= set(grid2)
    assert set(grid) <= set(grid2)


def test_malformed_score_manifest_cannot_omit_a_declared_cell(tmp_path):
    out = fixture(tmp_path)
    REDUCER.score_stage(out, "null_calibration")
    path = out / "scores/null_calibration/manifest.json"
    manifest = REDUCER.read_json(path)
    manifest["outputs_sha256"].pop("scores/null_calibration/ch14.npz")
    write(path, manifest)
    with pytest.raises(ValueError, match="omits or adds"):
        REDUCER.calibrate(out)


def test_actual_fixed_float_and_q16_score_shapes_and_coarse_norms():
    fine = np.ones((3, 3, 256), dtype=np.uint64)
    floating = np.repeat(fine[:, None], 5, axis=1).astype(float)
    coarse = np.ones((3, 3), dtype=np.uint64)
    float_coarse = np.repeat(coarse[:, None], 5, axis=1).astype(float)
    values = REDUCER.reduce_powers(floating, fine, float_coarse, coarse, [202, 203], 2.)
    assert values["fine_float"].shape == (3, 2, 6, 4)
    assert values["required_q16"].dtype == np.uint64
    np.testing.assert_array_equal(values["coarse_float"], [[1, 1, 1, .5, .5, .5]] * 3)
    assert np.all(values["fine_float"] == 1)


def test_normalized_primary_coordinate_refused_before_score_reads(tmp_path):
    out = fixture(tmp_path)
    plan = REDUCER.read_json(out / "plan.json")
    plan["coordinate_system"] = "normalized-reference"
    write(out / "plan.json", plan)
    with pytest.raises(ValueError, match="native-inverted-adapted"):
        REDUCER.score_stage(out, "null_calibration")
