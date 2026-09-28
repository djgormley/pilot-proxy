"""The LimeSDR detection campaign: paired crossings, their intervals, and the
rows the products cannot fill."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.characterization.report import core, detection

REAL_SWEEP = Path("/home/djg/rail/products/estimator_transfer_2026-08-25")
REAL_OTA = Path("/home/djg/rail/products/ota_transfer_2026-08-24_ch35")


# ------------------------------------------------------------------- fixtures
def _run(tmp_path) -> core.Run:
    """A one-channel ledger: the module never reads it, but ``build`` takes a Run."""
    ledger = tmp_path / "ledger" / "channels"
    ledger.mkdir(parents=True)
    record = {"channel": 35, "freq_id": 500, "product": "500.npz", "product_sha256": "c" * 64,
              "notes": [], "sections": {}}
    (ledger / "ch35_fid500.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "ledger" / "run.json").write_text(
        json.dumps({"generated": "2026-09-07", "producer": {"commit": "d" * 40},
                    "channels": ["channels/ch35_fid500.json"]}), encoding="utf-8")
    return core.load_run(tmp_path)


COLUMNS = ("requested_data_shelf_snr_db", "frequency_offset_hz", "normalized_coarse_power_ratio",
           "p_ref_sum_u64", "cpu_float_normalized_coarse_power_ratio", "cpu_float_p_ref_sum",
           "cpu_packed_normalized_coarse_power_ratio", "cpu_packed_p_ref_sum", "cpu_gpu_abs_diff",
           "num_input_streams", "detector_rows_per_frame")


def _shard(directory: Path, points, *, per_point: int = 400, offset_hz: float = 0.0, gain_db: float = 0.0,
           jitter: float = 0.0, seed: int = 7, packed_equals_gpu: bool = True) -> Path:
    """One ``evaluate-snr`` directory whose packed path rescales the float excess and adds ``jitter``.

    The statistic is a chi-square-like ratio around 1 whose mean rises with the
    injected shelf, so the fixed-threshold and positive-excess curves both sweep
    from the null through saturation and a crossing exists for every criterion.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    gain = 10.0 ** (gain_db / 10.0)
    rows = []
    for snr in points:
        excess = 10.0 ** ((snr + 30.0) / 10.0)
        draw = rng.gamma(shape=8.0, scale=1.0 / 8.0, size=per_point)
        q_float = (1.0 + excess) * draw
        q_packed = 1.0 + gain * (q_float - 1.0)
        if jitter:
            q_packed = q_packed + rng.normal(0.0, jitter, size=per_point)
        for a, b in zip(q_float, q_packed):
            rows.append({"requested_data_shelf_snr_db": snr, "frequency_offset_hz": offset_hz,
                         "normalized_coarse_power_ratio": b if packed_equals_gpu else a,
                         "p_ref_sum_u64": 1000.0,
                         "cpu_float_normalized_coarse_power_ratio": a, "cpu_float_p_ref_sum": 1000.0,
                         "cpu_packed_normalized_coarse_power_ratio": b, "cpu_packed_p_ref_sum": 1000.0,
                         "cpu_gpu_abs_diff": 0.0 if packed_equals_gpu else 1.0,
                         "num_input_streams": 4, "detector_rows_per_frame": 512})
    with (directory / "dtv_snr_eval.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return directory


POINTS = [-60.0, -57.0, -54.0, -51.0, -48.0, -45.0, -42.0, -39.0, -36.0, -33.0, -30.0, -27.0, -24.0]


def _capture(tmp_path, **kwargs) -> detection.Capture:
    shard = _shard(tmp_path / kwargs.pop("name", "shard"), POINTS, **kwargs)
    capture = detection.load_capture("synthetic", "Synthetic fixture", [shard])
    assert capture is not None
    return capture


# ---------------------------------------------------------------------- unit
def test_paired_statistics_keeps_only_trials_every_path_carries(tmp_path):
    capture = _capture(tmp_path)
    assert capture.trials == len(POINTS) * 400
    assert set(capture.statistic) == set(("gpu", "cpu_float", "cpu_packed"))
    assert capture.offsets_hz == (0.0,) and capture.input_streams == (4,) and capture.detector_rows == (512,)
    assert capture.cpu_gpu_equal == capture.cpu_gpu_trials == capture.trials
    # the null population is the points at or below the declared ceiling, and nothing else
    assert capture.null_trials == sum(1 for p in POINTS if p <= detection.NULL_CEILING_DB) * 400


def test_crossing_reads_the_last_rise_through_the_level():
    points = np.array([-9.0, -6.0, -3.0, 0.0, 3.0])
    #                         a non-monotone low tail must not be mistaken for the crossing
    pd = np.array([0.2, 0.1, 0.4, 0.6, 1.0])
    snr, slope, index = detection.crossing(points, pd, 0.5)
    assert index == 2 and math.isclose(slope, 0.2 / 3.0)
    assert math.isclose(snr, -3.0 + 3.0 * 0.5)
    # no rise through the level at all, and a level above the curve's end
    assert detection.crossing(points, np.array([0.9] * 5), 0.5)[2] == -1
    assert detection.crossing(points, pd, 1.5)[2] == -1


def test_crossing_shift_recovers_a_known_horizontal_displacement():
    points = np.array([-12.0, -9.0, -6.0, -3.0, 0.0])
    reference = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    # a locally linear curve shifted right by exactly one grid step is a +3 dB shift
    shifted = np.array([0.0, 0.1, 0.3, 0.5, 0.7])
    assert math.isclose(detection.crossing_shift(points, reference, shifted, 0.5), 3.0, rel_tol=1e-9)
    # a curve identical to the reference has no shift, in either direction
    assert detection.crossing_shift(points, reference, reference, 0.5) == 0.0


def test_a_pure_rescaling_of_the_excess_is_invisible_at_a_matched_false_alarm_rate(tmp_path):
    """A monotone rescaling moves both the threshold and the statistic: it is not a loss.

    Every criterion here is rank-based at a matched null rate, so a packed path
    that only scales the excess makes exactly the same decisions as the control.
    What the crossings measure is a change in separation, not a level offset.
    """
    capture = _capture(tmp_path, gain_db=-0.8, name="gain_only")
    assert not np.allclose(capture.statistic["cpu_packed"], capture.statistic["cpu_float"])
    assert [r.shift_db for r in detection.loss_rows(capture, samples=100)] == [0.0, 0.0, 0.0]


def test_added_representation_noise_moves_the_crossing_and_the_bootstrap_brackets_it(tmp_path):
    """A packed path that is genuinely noisier must be seen, signed, by every criterion."""
    capture = _capture(tmp_path, gain_db=-0.4, jitter=0.30, name="lossy")
    rows = detection.loss_rows(capture, samples=300)
    assert [r.criterion for r in rows] == ["pd50", "pd90", "pos90"]
    for row in rows:
        assert row.shift_db > 0.0, row                       # a weaker packed path crosses later
        assert row.low_db <= row.shift_db <= row.high_db
        assert math.isfinite(row.resolution_db) and 0.0 < row.resolution_db < detection.MARGIN_DB
        assert row.verdict == "not established"              # the loss is real and outside the margin
    # a path that is not degraded at all sits on zero and is judged within the margin
    clean = detection.loss_rows(_capture(tmp_path, name="clean"), samples=300)
    assert all(abs(r.shift_db) < detection.MARGIN_DB for r in clean)
    assert [r.verdict for r in clean] == ["within"] * 3


def test_a_capture_too_coarse_to_decide_the_margin_says_so(tmp_path):
    """Few trials per point: one flipped decision outruns the margin, so no verdict is claimed."""
    shard = _shard(tmp_path / "thin", POINTS, per_point=12, seed=3)
    capture = detection.load_capture("ota", "Thin fixture", [shard])
    rows = detection.loss_rows(capture, samples=200)
    assert all(r.resolution_db > detection.MARGIN_DB for r in rows)
    assert {r.verdict for r in rows} == {"unresolved"}


def test_thresholds_hit_the_nominal_false_alarm_rate_on_the_null(tmp_path):
    capture = _capture(tmp_path)
    tau = detection.thresholds(capture)
    null = capture.null_mask
    for path, value in tau.items():
        rate = float((capture.statistic[path][null] > value).mean())
        assert abs(rate - detection.NOMINAL_PFA) < 0.02, (path, rate)
    points, fixed, positive, _ = detection.detection_curves(capture)
    assert points.size == len(POINTS)
    for path in fixed:
        assert fixed[path][0] < 0.2 and fixed[path][-1] > 0.9      # null to saturation
        assert positive[path][0] < 0.7 and positive[path][-1] > 0.9


# --------------------------------------------------------------------- table
def test_table_prints_the_measured_rows_the_absent_ones_and_the_components(tmp_path):
    run = _run(tmp_path)
    caps = [_capture(tmp_path, gain_db=-0.2, name="sweep")]
    frag = detection.build(run, captures=caps, samples=200)
    text = frag.tex
    assert frag.name == "detection" and frag.label == "tab:detection:loss"
    assert text.startswith("\\begin{tabular}{lrrccl}") and text.rstrip().endswith("\\end{tabular}")
    # every criterion, every unmeasured offset, and both components have a row
    for label in (r"Fixed threshold, $P_{\rm d}=0.5$", r"Positive excess, $P_{\rm d}=0.9$",
                  r"Pilot offset $\pm 1.4$ kHz", "Frozen transform only", "Same-seed CPU/GPU"):
        assert label in text, label
    assert text.count(core.DASH) >= 3 * 4                       # the three offsets, dashed across the row
    assert "$-0.0026$" in text and "$5{,}200/5{,}200$" in text
    keys = {n.key for n in frag.numbers}
    for key in ("ch06.detection.shift_db.synthetic_pd50", "ch06.detection.ci_high_db.synthetic_pd90",
                "ch06.detection.resolution_db.synthetic_pos90", "ch06.detection.margin_db",
                "ch06.detection.cpu_gpu_equal", "ch06.detection.transform_only_db",
                "ch06.detection.detector_rows.synthetic", "ch06.detection.null_trials.synthetic"):
        assert key in keys, key
    by_key = {n.key: n for n in frag.numbers}
    assert by_key["ch06.detection.ci_high_db.synthetic_pd50"].status == "bounded"
    assert by_key["ch06.detection.transform_only_db"].value == detection.TRANSFORM_ONLY_DB
    assert by_key["ch06.detection.shift_db.offset_khz1"].renderings == (core.DASH,)
    assert by_key["ch06.detection.shift_db.offset_khz1"].status == "pending"
    assert frag.inputs and all(p.name == "dtv_snr_eval.csv" for p in frag.inputs)
    # the notes must name every measurement the products do not carry
    notes = " ".join(frag.notes)
    for phrase in ("packed minus full-precision", "no M-sweep", "fine statistic", "frequency_offset_hz",
                   "quantized-weight float rung", "duty cycle", "selects nothing"):
        assert phrase in notes, phrase


def test_the_table_stands_with_no_products_at_all(tmp_path):
    """Absent inputs must dash the table, not raise: the report renders on any machine."""
    frag = detection.build(_run(tmp_path), captures=[])
    assert "no campaign products were found" in " ".join(frag.notes)
    assert core.DASH in frag.tex and "Frozen transform only" in frag.tex
    assert frag.tex.count(r"\\") >= 8
    assert {n.key for n in frag.numbers} >= {"ch06.detection.margin_db", "ch06.detection.cpu_gpu_trials"}
    assert next(n for n in frag.numbers if n.key == "ch06.detection.cpu_gpu_trials").value == 0


def test_missing_directories_load_as_no_captures(tmp_path):
    assert detection.load_capture("x", "X", [tmp_path / "nowhere"]) is None
    assert detection.load_captures(sweep=tmp_path / "nowhere", ota=tmp_path / "also-nowhere") == []


def test_locations_follow_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(detection.SWEEP_ENV, str(tmp_path / "sweep"))
    monkeypatch.setenv(detection.OTA_ENV, str(tmp_path / "ota"))
    assert detection.sweep_dir() == tmp_path / "sweep" and detection.ota_dir() == tmp_path / "ota"
    monkeypatch.delenv(detection.SWEEP_ENV)
    with pytest.raises(ValueError, match="needs the sweep"):
        detection.sweep_dir()


# -------------------------------------------------------------------- figure
def test_figure_renders_both_captures(tmp_path):
    from pilot_proxy import figure_style as style

    style.configure(require_tex=False)
    caps = [_capture(tmp_path, gain_db=-0.2, name="sweep"),
            detection.load_capture("ota", "Thin fixture", [_shard(tmp_path / "thin", POINTS, per_point=40)])]
    paths = detection.render(_run(tmp_path), tmp_path / "figures", captures=caps, samples=120)
    assert [p.name for p in paths] == ["fig_detection_crossings.pdf", "fig_detection_crossings.png"]
    assert all(p.is_file() and p.stat().st_size > 5_000 for p in paths)


def test_figure_is_skipped_when_the_campaign_is_absent(tmp_path):
    assert detection.render(_run(tmp_path), tmp_path / "figures", captures=[]) == []
    assert not (tmp_path / "figures").exists()


def test_registered_in_both_registries():
    from pilot_proxy.characterization.report import build

    assert "detection" in build.TABLE_MODULES and "detection" in build.FIGURE_MODULES


# ----------------------------------------------------------------- real data
@pytest.mark.skipif(not (REAL_SWEEP / "upper" / "dtv_snr_eval.csv").is_file(),
                    reason="the 2026-08-25 estimator-transfer sweep is not on this machine")
def test_real_campaign_is_the_512_row_geometry_at_one_offset():
    caps = detection.load_captures()
    assert [c.key for c in caps][0] == "synthetic"
    sweep = caps[0]
    assert sweep.trials == 9000 and sweep.null_trials == 960
    # the facts the table's notes assert about these products
    assert sweep.offsets_hz == (0.0,) and sweep.input_streams == (4,) and sweep.detector_rows == (512,)
    assert sweep.cpu_gpu_equal == sweep.cpu_gpu_trials == 9000
    rows = detection.loss_rows(sweep, samples=200)
    for row in rows:
        assert abs(row.shift_db) < 0.5 and row.low_db < row.shift_db < row.high_db
        assert -36.0 < row.float_crossing_db < -26.0
