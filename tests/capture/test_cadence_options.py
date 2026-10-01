"""One cadence module for the three levels and both lag class sets."""
from __future__ import annotations

import csv

import pytest

pytest.importorskip("yaml")

from capture_fixtures import write_dump  # noqa: E402
from pilot_proxy.capture import cadence  # noqa: E402


@pytest.fixture(scope="module")
def epochs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cadence")
    offsets = {"E0": 0.0, "E1": 15.0, "E2": 60.0, "E3": 240.0, "E4": 1200.0, "E5": 7300.0, "E6": 30000.0}
    return [f"{lbl}={write_dump(root / lbl, seed=10 + i, n_frames=6, t0=1.7e9 + dt)}"
            for i, (lbl, dt) in enumerate(offsets.items())]


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_stokes_i_is_the_polarisation_average_labelled_I(tmp_path, epochs):
    a, b = tmp_path / "avg.csv", tmp_path / "I.csv"
    assert cadence.main(["tau", "--trim", "archive", "--class", "0,32", str(a), *epochs]) == 0
    assert cadence.main(["tau", "--level", "stokesI", "--trim", "archive", "--class", "0,32", str(b), *epochs]) == 0
    ra, rb = _rows(a), _rows(b)
    assert len(ra) == len(rb) > 0
    for x, y in zip(ra, rb):
        assert x["pol"] == "avg" and y["pol"] == "I"
        assert {k: v for k, v in x.items() if k != "pol"} == {k: v for k, v in y.items() if k != "pol"}
    assert (tmp_path / "avg_structure.csv").read_bytes() == (tmp_path / "I_structure.csv").read_bytes()


def test_polmax_forced_polarisation_is_written(tmp_path, epochs):
    out = tmp_path / "p1.csv"
    assert cadence.main(["tau", "--level", "polmax", "--pol", "1", str(out), *epochs]) == 0
    assert {r["pol"] for r in _rows(out)} == {"1"}


def test_both_lag_class_sets_keep_their_orders(tmp_path, epochs):
    three, ten = tmp_path / "three.csv", tmp_path / "ten.csv"
    assert cadence.main(["lags", "--classes", "three", "--reference-bands", "37", str(three), *epochs[:3]]) == 0
    assert cadence.main(["lags", "--classes", "ten", "--reference-bands", "37", str(ten), *epochs[:3]]) == 0
    def order(path):
        seen = []
        for r in _rows(path):
            if r["channel"] == "14" and r["cls"] not in seen:
                seen.append(r["cls"])
        return seen
    assert order(three) == ["ns1", "ns1_x", "ns8_x", "ew1_x"]
    assert order(ten) == [c + "_x" for c in cadence.LAG_CLASS_SETS["ten"]] + ["ns1"]


def test_an_unknown_level_is_refused(tmp_path, epochs):
    assert cadence.main(["tau", "--level", "polmin", str(tmp_path / "x.csv"), *epochs]) == 2


def test_the_reference_bands_default_to_the_record(tmp_path, epochs):
    from pilot_proxy.config.project import default_project
    bands = default_project().record_module("capture_campaign").LAG_REFERENCE_BANDS
    given, default = tmp_path / "given.csv", tmp_path / "default.csv"
    assert cadence.main(["lags", "--classes", "three", "--reference-bands", ",".join(map(str, bands)), str(given),
                         *epochs[:3]]) == 0
    assert cadence.main(["lags", "--classes", "three", str(default), *epochs[:3]]) == 0
    assert default.read_bytes() == given.read_bytes()
