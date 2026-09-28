"""The results layer's evaluation reader: dtv_snr_eval directories, pooling, release reproduction."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.testbench.harness import evaluations as ev
from pilot_proxy.testbench.harness.estimator_transfer import Calibration

CAL = Calibration(pilot_below_data_db=11.918446870168612, bin_enbw_hz=3051.7578125, dtv_bandwidth_hz=6.0e6)


def _write_evaluation(root: Path, snr_db: tuple[float, ...], *, trials: int = 6, seed: int = 1) -> Path:
    """A small evaluate-snr directory: exact power terms per trial, pooled summary, report."""
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    target_norm, ref_norm = 6338, 12650
    trial_rows, summary_rows = [], []
    for snr in snr_db:
        excess = 10 ** (snr / 10) * 30.0           # a made-up but monotone transfer
        pt_sum = pr_sum = 0
        for t in range(trials):
            p_ref = int(rng.integers(40_000_000, 50_000_000))
            p_target = int(p_ref * target_norm / ref_norm * (1.0 + excess) * (1 + 0.05 * rng.standard_normal()))
            lower = p_ref // 2
            q = p_target * ref_norm / (p_ref * target_norm)
            trial_rows.append({
                "requested_data_shelf_snr_db": snr, "frequency_offset_hz": 0.0, "trial": t,
                "coarse_power_ratio": 2 * p_target / p_ref, "normalized_coarse_power_ratio": q,
                "p_target_u64": p_target, "p_ref_lower_u64": lower, "p_ref_upper_u64": p_ref - lower,
                "p_ref_sum_u64": p_ref,
                "cpu_float_coarse_power_ratio": 2 * p_target / p_ref, "cpu_float_normalized_coarse_power_ratio": q,
                "cpu_float_p_ref_sum": p_ref, "cpu_packed_normalized_coarse_power_ratio": q,
                "cpu_packed_p_ref_sum": p_ref, "noise_seeds": "[1, 2]",
            })
            pt_sum += p_target; pr_sum += p_ref
        pooled = pt_sum * ref_norm / (pr_sum * target_norm)
        summary_rows.append({
            "requested_data_shelf_snr_db": snr, "frequency_offset_hz": 0.0, "trials": trials,
            "gpu_pooled_normalized_coarse_power_ratio": pooled, "gpu_pooled_p_ref_sum": pr_sum,
            "cpu_float_pooled_normalized_coarse_power_ratio": pooled, "cpu_float_pooled_p_ref_sum": pr_sum,
            "cpu_packed_pooled_normalized_coarse_power_ratio": pooled, "cpu_packed_pooled_p_ref_sum": pr_sum,
        })
    for name, rows in ((ev.TRIALS_NAME, trial_rows), (ev.SUMMARY_NAME, summary_rows)):
        with (root / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (root / ev.REPORT_NAME).write_text(json.dumps({
        "schema_version": ev.REPORT_SCHEMA,
        "calibration": {"pilot_below_data_db_used": CAL.pilot_below_data_db, "bin_enbw_hz_assumed": CAL.bin_enbw_hz,
                        "pilot_capture_efficiency_assumed": 1.0},
        "detector_geometry": {"bin_enbw_hz": CAL.bin_enbw_hz, "dtv_bandwidth_hz": CAL.dtv_bandwidth_hz,
                              "pilot_capture_efficiency": 1.0},
    }))
    return root


def test_reader_loads_report_trials_and_summary(tmp_path):
    e = ev.load_evaluation(_write_evaluation(tmp_path / "e", (-30.0, -27.0)))
    assert e.requested_snr_db == (-30.0, -27.0) and len(e.trials) == 12 and not e.is_radio
    assert e.calibration == CAL
    assert math.isnan(e.trials[0]["noise_seeds"])          # non-numeric cells read as NaN, not errors


def test_reader_refuses_an_unknown_report_schema(tmp_path):
    root = _write_evaluation(tmp_path / "e", (-30.0,))
    (root / ev.REPORT_NAME).write_text(json.dumps({"schema_version": "something_else_v9"}))
    with pytest.raises(ValueError, match="report schema"):
        ev.load_evaluation(root)


def test_pooled_ratio_sums_powers_before_dividing(tmp_path):
    e = ev.load_evaluation(_write_evaluation(tmp_path / "e", (-30.0,)))
    pt = sum(r["p_target_u64"] for r in e.trials); pr = sum(r["p_ref_sum_u64"] for r in e.trials)
    pooled = ev.pooled_ratio(list(e.trials), "gpu")
    assert pooled == pytest.approx(pt * 12650 / (pr * 6338), rel=1e-12)
    assert ev.summary_ratio(list(e.summary), "gpu") == pytest.approx(pooled, rel=1e-12)


def test_bootstrap_interval_is_seeded_and_brackets_the_point(tmp_path):
    e = ev.load_evaluation(_write_evaluation(tmp_path / "e", (-24.0,), trials=40))
    rows = list(e.trials)
    seed = ev.point_seed("gpu", 0.0, -24.0)
    a = ev.bootstrap_excess_interval(rows, "gpu", samples=500, seed=seed)
    b = ev.bootstrap_excess_interval(rows, "gpu", samples=500, seed=seed)
    assert a == b
    excess = ev.pooled_ratio(rows, "gpu") - 1.0
    assert a[0] < excess < a[1]
    assert ev.point_seed("gpu", 0.0, -24.0) != ev.point_seed("cpu_float", 0.0, -24.0)


def test_shelf_conversion_matches_the_calibration_and_censors_nonpositive_excess():
    assert math.isnan(ev.excess_to_shelf_db(0.0, CAL)) and math.isnan(ev.excess_to_shelf_db(-0.2, CAL))
    excess = 10 ** (-5.0 / 10)                       # -5 dB of one-bin pilot excess
    assert ev.excess_to_shelf_db(excess, CAL) == pytest.approx(-5.0 - CAL.offset_db)
    x = np.array([-60.0, 0.0, 30.0])
    np.testing.assert_allclose(ev.ideal_transfer_db(x), x - 10 * np.log10(1 + 10 ** (x / 10)))


def test_conditioning_model_follows_the_recorded_formula(tmp_path):
    doc = {"coefficients": {"C_db": -21.0, "delta": 0.002, "a": 136.0, "b": 1.1}}
    (tmp_path / "conditioning.json").write_text(json.dumps(doc))
    model = ev.Conditioning.from_json(tmp_path / "conditioning.json")
    x = -12.0
    lin = 10 ** (x / 10)
    assert model.expected_db([x])[0] == pytest.approx(-21.0 + 10 * math.log10((0.002 + 136.0 * lin) / (1 + 1.1 * lin)))


def test_transfer_points_pool_shards_in_order_and_count_positive_excess(tmp_path):
    shards = ev.load_evaluations([_write_evaluation(tmp_path / "a", (-30.0, -27.0), seed=1),
                                  _write_evaluation(tmp_path / "b", (-24.0,), seed=2)])
    rows = ev.transfer_points(shards, bootstrap_samples=200)
    assert [r["requested_data_shelf_snr_db"] for r in rows] == [-30.0, -27.0, -24.0]
    assert all(r["trials"] == 6 for r in rows)
    assert rows[-1]["gpu_fixed_db"] == pytest.approx(rows[-1]["cpu_packed_db"])
    assert rows[-1]["gpu_ci95_low_db"] <= rows[-1]["gpu_fixed_db"] <= rows[-1]["gpu_ci95_high_db"]
    assert all(math.isnan(r["waveform_conditioned_expected_db"]) for r in rows)
    out = ev.write_points(rows, tmp_path / "plot_points.csv")
    with out.open(newline="") as fh:
        back = list(csv.DictReader(fh))
    assert list(back[0]) == list(ev.DIGITAL_COLUMNS) and back[0]["waveform_conditioned_expected_db"] == ""


def test_transfer_points_refuse_shards_with_different_calibrations(tmp_path):
    a = _write_evaluation(tmp_path / "a", (-30.0,))
    b = _write_evaluation(tmp_path / "b", (-27.0,))
    doc = json.loads((b / ev.REPORT_NAME).read_text()); doc["calibration"]["pilot_below_data_db_used"] = 11.3
    (b / ev.REPORT_NAME).write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="calibration differs"):
        ev.transfer_points(ev.load_evaluations([a, b]), bootstrap_samples=10)


def test_digital_sweep_layout_names_forty_shards(tmp_path):
    paths = ev.digital_sweep_layout(tmp_path)
    assert len(paths) == 40 and len(set(paths)) == 40
    assert paths[0] == tmp_path / "extreme_low" / "snr_m60" and paths[-1] == tmp_path / "high" / "snr_p60"


SWEEP = os.environ.get("RFISHER_TRANSFER_SWEEP")
RELEASE = os.environ.get("RFISHER_TRANSFER_RELEASE")


@pytest.mark.skipif(not (SWEEP and RELEASE and Path(SWEEP).is_dir() and Path(RELEASE).is_dir()),
                    reason="set RFISHER_TRANSFER_SWEEP (raw shard root) and RFISHER_TRANSFER_RELEASE (release dir)")
def test_the_frozen_digital_release_is_reproduced_from_its_raw_shards():
    shards = ev.load_evaluations(ev.digital_sweep_layout(SWEEP))
    conditioning = ev.Conditioning.from_json(Path(RELEASE) / "run" / "conditioning.json")
    rows = ev.transfer_points(shards, conditioning=conditioning)
    with (Path(RELEASE) / "data" / "plot_points.csv").open(newline="") as fh:
        released = list(csv.DictReader(fh))
    assert len(released) == len(rows) == 41
    for theirs, ours in zip(released, rows):
        for name in ev.DIGITAL_COLUMNS:
            a = math.nan if theirs[name] == "" else float(theirs[name])
            b = float(ours[name])
            assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-9), (name, theirs, ours)
