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


def test_the_sidecar_reconstructs_the_fraction_above_any_threshold(tmp_path, monkeypatch):
    # the U arrays the producer holds in memory: the first argument of each 90th-percentile call (the U_i rows only)
    held, percentile = [], np.percentile

    def spy(a, q, *args, **kw):
        if q == 90:
            held.append(np.array(a, copy=True))
        return percentile(a, q, *args, **kw)
    monkeypatch.setattr(np, "percentile", spy)
    out = _run(tmp_path)
    monkeypatch.undo()
    rows = list(csv.DictReader(out.open()))
    ui = [r for r in rows if r["frames"] == "U_i"]
    assert ui and all(r["excess_inband_max"] == "" for r in ui) and len(held) == len(ui)
    with open(frame_residual.inputs_path(str(out))) as fh:
        side = list(csv.DictReader(fh))
    for r, mem in zip(ui, held):
        U = np.array([float(s["U"]) for s in side if s["channel"] == r["channel"]], dtype=np.float32)
        assert all(s["n_frames"] == r["n_frames"] for s in side if s["channel"] == r["channel"])
        assert mem.dtype == np.float32 and np.array_equal(U, mem)
        # the row's median and 90th percentile are the sidecar's
        assert r["excess_pilot_bin"] == str(float(np.median(U)))
        assert r["excess_inband_median"] == str(float(np.percentile(U, 90)))
        # the frozen producer's U_i fraction, float((U > lambda).mean()), is the same from the sidecar at any lambda
        for lam in (-1.0, 0.0, float(np.median(mem)), float(mem.max()), 1.0, *map(float, mem[:5])):
            assert float((U > lam).mean()) == float((mem > lam).mean())
    assert {int(r["channel"]) for r in ui} == {14, 15, 16, 37}   # the control band is read at its target too
