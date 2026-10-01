"""The marker-to-in-band correction on the capture reproduces its record (tests/capture/data/pilot_to_inband.csv, the
frozen frame_analysis output). The record was written under numpy 2.3.5; another numpy may move a printed last
digit, so the cells are compared to their printed precision (the gate compares the bytes under the interpreter of
record)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from capture_data import ROOT, dump_dir  # noqa: E402
from pilot_proxy.capture.marker_map import main as marker_map  # noqa: E402
from pilot_proxy.detectors.narrowband_marker import marker_to_inband  # noqa: E402

RECORD = Path(__file__).resolve().parent / "data" / "pilot_to_inband.csv"


def test_the_correction_reproduces_its_record(tmp_path):
    for ev in ("20260916162300", "20260917040230", "20260917090230", "20260917140230"):
        dump_dir(ev)
    spec = tmp_path / "map.json"
    assert marker_map(["--out", str(spec)]) == 0
    out = tmp_path / "pilot_to_inband.csv"
    assert marker_to_inband.main(["--marker-map", str(spec), "--datasets", str(ROOT), "--out", str(out)]) == 0
    got, want = list(csv.DictReader(out.open())), list(csv.DictReader(RECORD.open()))
    assert [r["channel"] for r in got] == [r["channel"] for r in want]
    for g, w in zip(got, want):
        assert (g["usable"], g["strong"]) == (w["usable"], w["strong"])
        for k in ("pilot_A_long", "inband_A_long"):
            assert float(g[k]) == pytest.approx(float(w[k]), rel=2e-6)
        if w["correction_db"]:
            assert abs(float(g["correction_db"]) - float(w["correction_db"])) <= 0.0015
        else:
            assert g["correction_db"] == ""
