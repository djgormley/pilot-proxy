"""The class excess on the XX and YY stacks and on Stokes I, and the control level from the plan's control band."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

pytest.importorskip("yaml")

from capture_fixtures import BANDS, write_dump  # noqa: E402
from pilot_proxy.capture import class_excess, control_level  # noqa: E402


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp("datasets")
    for i, ev in enumerate(("20260101000000", "20260101000100")):
        write_dump(root / f"pilot_reduce_{ev}", seed=30 + i, n_frames=7)
    epochs = root / "epochs.txt"
    epochs.write_text(" ".join(f"E{i}={root}/pilot_reduce_{ev}" for i, ev in
                               enumerate(("20260101000000", "20260101000100"))))
    return root, epochs


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_rows_follow_class_then_polarisation(tmp_path, datasets):
    root, epochs = datasets
    out = tmp_path / "xy"
    assert class_excess.main(["--epochs", str(epochs), "--datasets", str(root), "--products", "xx,yy",
                              "--out-dir", str(out)]) == 0
    rows = _rows(out / "class_excess_epochs.csv")
    bands = sorted(BANDS)
    expect = [(ev, f"{c[0]},{c[1]}", str(p), str(b)) for ev in ("20260101000000", "20260101000100")
              for c in class_excess.CLASSES for p in (0, 1) for b in bands]
    assert [(r["epoch"], f"{r['ew']},{r['ns']}", r["pol"], r["channel"]) for r in rows] == expect
    floor = _rows(out / "class_floor_bins.csv")
    assert {int(r["freq_id"]) for r in floor} == set(BANDS[37][1])     # the control band's bins only


def test_stokes_i_reads_the_complex_mean_of_the_two_stacks(tmp_path):
    # with the YY stacks equal to the XX stacks, (v_XX + v_YY)/2 is v_XX exactly, so Stokes I reads what XX reads
    root = tmp_path / "same"
    for i, ev in enumerate(("20260101000000", "20260101000100")):
        write_dump(root / f"pilot_reduce_{ev}", seed=50 + i, n_frames=7, same_pols=True)
    events = ["20260101000000", "20260101000100"]
    xi, fi = class_excess.measure(events, ("I",), datasets=str(root), nfft=16384, control=[37])
    xp, fp = class_excess.measure(events, (0, 1), datasets=str(root), nfft=16384, control=[37])
    strip = lambda rows: [{k: v for k, v in r.items() if k != "pol"} for r in rows]  # noqa: E731
    assert strip(xi) == strip([r for r in xp if r["pol"] == 0])
    assert strip(fi) == strip([r for r in fp if r["pol"] == 0])
    assert {r["pol"] for r in xi} == {"I"}
    z = np.load(root / "pilot_reduce_20260101000000" / "843.npz")
    assert json.loads(str(z["meta"]))["channel"] == 14


def test_control_level_reads_the_control_band(tmp_path, datasets):
    root, _ = datasets
    out = tmp_path / "cl"
    assert control_level.main(["--datasets", str(root), "--out-dir", str(out),
                               "--events", "20260101000000,20260101000100"]) == 0
    rows = _rows(out / "baseline_floor.csv")
    assert {r["channel"] for r in rows} == {str(b) for b in BANDS}
    by_class = {}
    for r in rows:
        by_class.setdefault((r["ew"], r["ns"], r["pol"]), set()).add((r["floor_mean"], r["floor_sd"]))
    assert all(len(v) == 1 for v in by_class.values())      # one control level per class and polarisation
