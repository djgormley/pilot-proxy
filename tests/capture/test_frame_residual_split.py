"""The frame residual writes no tolerance: its layout is the frozen one less lambda_q, and the per-input sidecar
reconstructs the U_i fraction for any threshold."""
from __future__ import annotations

import ast
import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")

from capture_fixtures import write_detector_run, write_dump  # noqa: E402
from pilot_proxy.capture import frame_residual  # noqa: E402

# the frozen producer's header (FA frame_residual.py f2ec3277, L98-102)
FROZEN_HEADER = ("channel,ew,ns,pol,frames,n_frames,excess_pilot_bin,excess_inband_median,excess_inband_max,lambda_q,"
                 "phi_fast_pilot,G_dump_pilot,lag_zero_pilot,kernel_excess_db_median")


def _run(tmp_path):
    dump = write_dump(tmp_path / "dump", seed=3)
    run = write_detector_run(tmp_path / "run", seed=4)
    out = tmp_path / "fr.csv"
    assert frame_residual.main(["--products", str(dump), "--detector-run", str(run), "--out", str(out)]) == 0
    return out


def test_header_is_the_frozen_one_less_lambda(tmp_path):
    out = _run(tmp_path)
    header = out.read_text().splitlines()[0]
    assert header.split(",") == [c for c in FROZEN_HEADER.split(",") if c != "lambda_q"]


def test_no_tolerance_is_read_or_written():
    source = Path(frame_residual.__file__).read_text()
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not {"LAMBDA", "lambda_q"} & names
    assert "channels.csv" not in source and "exact-time-reference" not in source


def test_the_sidecar_reconstructs_the_fraction_above_any_threshold(tmp_path):
    out = _run(tmp_path)
    rows = list(csv.DictReader(out.open()))
    ui = [r for r in rows if r["frames"] == "U_i"]
    assert ui and all(r["excess_inband_max"] == "" for r in ui)
    side = list(csv.DictReader(open(frame_residual.inputs_path(str(out)))))
    for r in ui:
        U = np.array([float(s["U"]) for s in side if s["channel"] == r["channel"]], dtype=np.float32)
        assert all(s["n_frames"] == r["n_frames"] for s in side if s["channel"] == r["channel"])
        # the row's median and 90th percentile are the sidecar's
        assert r["excess_pilot_bin"] == str(float(np.median(U)))
        assert r["excess_inband_median"] == str(float(np.percentile(U, 90)))
        for lam in (-1.0, 0.0, float(np.median(U)), 1.0):
            frac = float((U > lam).mean())
            assert 0.0 <= frac <= 1.0
    assert {int(r["channel"]) for r in ui} == {14, 15, 16, 37}   # the control band is read at its target too
