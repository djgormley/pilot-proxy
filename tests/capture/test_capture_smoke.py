"""Smoke tests on synthetic products for the capture commands the data tests reach only at the gate: the dump
diagnostics, the control level, the cadence report and the ladder placement."""
from __future__ import annotations

import csv

import numpy as np
import pytest

pytest.importorskip("yaml")

from capture_fixtures import BANDS, write_detector_run, write_dump  # noqa: E402
from pilot_proxy.capture import cadence, cadence_report, control_level, ladder_place  # noqa: E402
from pilot_proxy.capture.diagnostics import band_shape, bearing, first_look, line_check  # noqa: E402

CONTROL = "control bin 491 (nominal ATSC pilot position)"
N_FILES = sum(len(fids) for _, fids in BANDS.values())


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    return write_dump(tmp_path_factory.mktemp("diagnostics") / "pilot_reduce_20260101000000", seed=50, n_frames=4,
                      diagnostics=True)


@pytest.fixture(scope="module")
def epochs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cadence")
    offsets = {"E0": 0.0, "E1": 15.0, "E2": 60.0, "E3": 240.0, "E4": 1200.0, "E5": 7300.0, "E6": 30000.0}
    return [f"{lbl}={write_dump(root / lbl, seed=80 + i, n_frames=5, t0=1.7e9 + dt)}"
            for i, (lbl, dt) in enumerate(offsets.items())]


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _marker_fids():
    return {str(m) for m, _ in BANDS.values()}


def test_band_shape_reads_every_bin_and_names_the_control_bin(dump, tmp_path, capsys):
    out = tmp_path / "band_shape.csv"
    assert band_shape.main([str(dump), str(out)]) == 0
    rows = _rows(out)
    assert len(rows) == N_FILES
    markers = [r for r in rows if r["is_pilot"] == "1"]
    assert {r["freq_id"] for r in markers} == _marker_fids() and all(r["pilot_line_ratio"] for r in markers)
    assert all(r["pilot_line_ratio"] == "" for r in rows if r["is_pilot"] == "0")
    assert CONTROL in capsys.readouterr().out


def test_first_look_prints_one_line_per_marker_file(dump, tmp_path, capsys):
    out = tmp_path / "first_look.csv"
    assert first_look.main([str(dump), str(out)]) == 0
    rows = _rows(out)
    # the CSV's columns are the first file's (frozen); the first file here is not a marker file
    assert len(rows) == N_FILES and "line_ratio_min" not in rows[0]
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("ch")]
    assert len(lines) == len(BANDS) and sum(CONTROL in line for line in lines) == 1
    assert all(" line ratio " in line and " snr med " in line for line in lines)


def test_line_check_reads_the_marker_files(dump, capsys):
    assert line_check.main([str(dump)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(BANDS) and sum(CONTROL in line for line in lines) == 1
    assert all("naive:" in line and "inverted:" in line for line in lines)


def test_bearing_fits_every_marker_file_without_a_gain(dump, capsys):
    assert bearing.main([str(dump), "none", "1.7e9"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("ch  fid") and len([line for line in out[1:] if line[:2].strip().isdigit()]) == len(BANDS)
    assert sum(line.startswith("mean bearing error of the known stations per convention:") for line in out) == 1
    assert [line.split(":")[0].strip() for line in out if line.startswith("  ch")] == ["ch14"]   # the unknown line read


def test_control_level_writes_every_band_class_and_polarisation(tmp_path, capsys):
    for i, ev in enumerate(("E0", "E1")):
        write_dump(tmp_path / f"pilot_reduce_{ev}", seed=60 + i, n_frames=5)
    out = tmp_path / "out"
    assert control_level.main(["--datasets", str(tmp_path), "--out-dir", str(out), "--events", "E0,E1"]) == 0
    rows = _rows(out / "baseline_floor.csv")
    assert list(rows[0]) == ["channel", "ew", "ns", "baseline_m", "pol", "A", "floor_mean", "floor_sd",
                             "A_over_floor", "A_minus_floor_over_sd"]
    assert len(rows) == len(control_level.CLASSES) * 2 * len(BANDS)
    assert {r["channel"] for r in rows} == {str(b) for b in BANDS}
    # the control band's level is the same on every band's row of a class and polarisation
    for c in control_level.CLASSES:
        for p in ("0", "1"):
            assert len({r["floor_mean"] for r in rows if (r["ew"], r["ns"], r["pol"]) == (str(c[0]), str(c[1]), p)}) == 1
    assert "control level (band 37)" in capsys.readouterr().out


def test_cadence_report_renders_the_detector_sections(epochs, tmp_path, capsys):
    tau, alt, lag = tmp_path / "cadence_tau.csv", tmp_path / "cadence_tau_a3.csv", tmp_path / "lag.csv"
    assert cadence.main(["tau", "--level", "polmax", "--trim", "archive", str(tau), *epochs]) == 0
    assert cadence.main(["tau", str(alt), *epochs]) == 0
    assert cadence.main(["lags", "--classes", "three", "--reference-bands", "37", str(lag), *epochs[:4]]) == 0
    out = tmp_path / "CADENCE_DETECTOR.md"
    assert cadence_report.main([str(out), str(tau), str(lag), str(alt)]) == 0
    text = out.read_text()
    for heading in ("## Pilot bin: tau_c and G", "## In-band median: tau_c and G", "## The amendment-3 form beside it",
                    "## Phase coherence of the reference-subtracted excess"):
        assert heading in text
    assert "## Table of record" not in text and text.endswith(" |\n")
    out3 = tmp_path / "CADENCE_DETECTOR_3.md"
    assert cadence_report.main([str(out3), str(tau), str(lag)]) == 0
    assert "## The amendment-3 form beside it" not in out3.read_text()
    assert cadence_report.main([str(out)]) == 2


def test_cadence_report_refuses_the_table_of_record(epochs, tmp_path, capsys):
    tau, alt, lag = tmp_path / "cadence_tau.csv", tmp_path / "cadence_tau_a3.csv", tmp_path / "lag.csv"
    assert cadence.main(["tau", "--level", "polmax", "--trim", "archive", str(tau), *epochs]) == 0
    assert cadence.main(["tau", str(alt), *epochs]) == 0
    assert cadence.main(["lags", "--classes", "three", "--reference-bands", "37", str(lag), *epochs[:4]]) == 0
    tor = tmp_path / "table.csv"
    cols = ["channel", "freq_id", "role", "disposition", "policy_or_reason", "tau_c_s", "tau_c_status", "G_measured",
            "R_none_Gmeasured", "R_deployed_Gmeasured", "R_deployed_G1"]
    with open(tor, "w", newline="") as fh:
        csv.writer(fh).writerow(cols)
    out = tmp_path / "CADENCE_REPORT.md"
    capsys.readouterr()
    assert cadence_report.main([str(out), str(tau), str(lag), str(tor), str(alt)]) == 2
    assert cadence_report.main([str(out), str(tau), str(lag), str(tor)]) == 2
    err = capsys.readouterr().err
    assert "rfisher records cadence-report" in err and not out.exists()


def test_ladder_place_places_each_band_of_a_run(tmp_path, capsys):
    from pilot_proxy.config.project import default_project
    project = default_project(); rng = np.random.default_rng(70)
    arch = tmp_path / "per_band"; arch.mkdir()
    bands = (14, 15, 16)
    for band in project.frequency_plan.bands("screened"):
        if int(band.label) in bands:
            np.savez(arch / f"{int(project.target_freq_id(band))}.npz", valid=rng.random((300, 1)) > 0.05,
                     target_norm_sq=np.array([0.5]), reference_norm_sum_sq=np.array([2.0]),
                     coarse_power_ratio=rng.gamma(40.0, 0.025, (300, 1)) * 0.5)
    run = write_detector_run(tmp_path / "kernel_x", seed=71, bands=bands)
    assert ladder_place.main(["--archive-products", str(arch), str(run)]) == 0
    out = capsys.readouterr().out.splitlines()
    i = out.index("kernel_x: per channel, median dump Q, and the fraction of dump frames kept under each policy")
    placed = [line for line in out[i + 2:] if line.strip()]
    assert len(placed) == len(bands)
    assert all(line.rstrip().endswith(("kept by cal_q0.1", "kept by cal_q0.5", "kept by cal_q0.9",
                                       "rejected by every calibrated policy")) for line in placed)
