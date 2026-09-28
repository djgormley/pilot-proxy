"""Frozen-layout replay files are written only for the bands the replay file names; the handoff covers every band."""
from __future__ import annotations

import pytest

from characterization_fixtures import characterize
from pilot_proxy.characterization import oc_table, run


def test_replay_only_for_listed_bands(tmp_path):
    summary, out = characterize(tmp_path, replay=(29,))
    assert summary["errors"] == []
    assert (out / "channels" / "ch29" / "held_out_spectra.npz").is_file()
    assert not (out / "channels" / "ch34" / "held_out_spectra.npz").exists()
    held = oc_table.read_rows(out / "tables" / "held_out_spectra.csv")
    assert {r["channel"] for r in held} == {"29"}
    assert {r["band_id"] for r in oc_table.read_rows(out / "oc_table.csv")} == {"29", "34"}
    assert {r["band_id"] for r in oc_table.read_rows(out / "oc_summary.csv")} == {"29", "34"}


def test_a_named_point_that_is_not_reproduced_is_reported(tmp_path):
    summary, out = characterize(tmp_path, replay=(29,), out="first")
    section = next(r for r in oc_table.read_rows(out / "oc_summary.csv")
                   if r["band_id"] == "29" and r["candidate_set"] == "fine_surface")
    assert section["least_residual_rho"]
    good = tmp_path / "good.csv"
    good.write_text(f"band_id,least_residual_rho,least_residual_eta_q16\n29,{section['least_residual_rho']},"
                    f"{section['least_residual_eta_q16']}\n")
    assert run.read_replay_points(good)[29]["least_residual_rho"] == section["least_residual_rho"]
    fake = [{"record": type("R", (), {"channel": 29, "sections": {"surface": {
        "least_residual_rho": 1, "least_residual_eta_q16": 65536}}})()}]
    assert run._check_replay(fake, run.read_replay_points(good))
    assert summary["replay_problems"] == []


def test_a_replay_file_needs_a_band_column(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("channel\n29\n")
    with pytest.raises(ValueError, match="band_id"):
        run.read_replay_points(bad)
    assert run.read_replay_points(None) == {}
