"""The three named record configurations, their era lists, and a tie the two rules break differently."""
from __future__ import annotations

import math
import shutil

import pytest

from pilot_proxy.characterization import surface
from pilot_proxy.config._files import ProfileError
from pilot_proxy.config.project import default_project, default_project_dir, load_project
from pilot_proxy.records.chime_atsc_2026 import archive_releases as ar


def test_the_releases_are_named_with_their_choices():
    assert set(ar.RECORDS) == {"archive_author_eras_2026_09_23", "archive_no_split_2026_09_24_r2",
                               "archive_no_split_2026_09_24_r3"}
    booked = ar.record("archive_author_eras_2026_09_23")
    off = ar.record("archive_no_split_2026_09_24_r2")
    dated = ar.record("archive_no_split_2026_09_24_r3")
    assert (booked.variance_split, booked.ties.name) == ("booked_when_tau_usable", "first_minimum")
    assert (off.variance_split, off.ties.name) == ("off", "selector_order")
    assert (dated.variance_split, dated.ties.name) == ("off", "selector_order")
    base = default_project().integration_model
    assert base.variance_split == "off"
    model = booked.integration_model(base)
    assert model.variance_split == "booked_when_tau_usable"
    assert (model.frame_seconds, model.coherence_cap_seconds) == (base.frame_seconds, base.coherence_cap_seconds)
    with pytest.raises(KeyError, match="unknown archive record"):
        ar.record("archive_v5_2026_09_07")


def test_each_record_names_the_era_list_its_release_ran():
    project = default_project()
    lists = {name: project.era_list(rec.era_list, rec.era_list_sha256) for name, rec in ar.RECORDS.items()}
    old, new = lists["archive_no_split_2026_09_24_r2"], lists["archive_no_split_2026_09_24_r3"]
    assert lists["archive_author_eras_2026_09_23"] == old
    assert (old.version, new.version) == ("author-dated-eras-2026-09-23", "author-dated-eras-2026-09-28")
    # the default is the r3 configuration: the project's own list
    assert new.source_sha256 == project.eras.source_sha256 == ar.ERAS_2026_09_28[1]
    # the two lists differ only in channel 17's dated first month (and the note and criterion that say why)
    changed = {c for c in set(old.overrides) | set(new.overrides) if old.overrides.get(c) != new.overrides.get(c)}
    assert changed == {"17"}
    assert [s.first_month for s in old.overrides["17"]] == ["2018-12", "2022-10"]
    assert [s.first_month for s in new.overrides["17"]] == ["2018-12", "2021-12"]
    assert old.unchanged_bands == new.unchanged_bands
    assert {c for c in old.notes if old.notes[c] != new.notes.get(c)} == {"17"}


def test_a_record_list_with_other_bytes_is_refused(tmp_path):
    project = load_project(shutil.copytree(default_project_dir(), tmp_path / "project"))
    rec = ar.record("archive_no_split_2026_09_24_r2")
    path = project.directory / rec.era_list
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ProfileError, match="is not the recorded"):
        project.era_list(rec.era_list, rec.era_list_sha256)
    with pytest.raises(ProfileError, match="leaves the project directory"):
        project.era_list("../outside.json")


def _row(rho, eta_q16, f, r):
    return {"rho": rho, "eta_q16": eta_q16, "eta": eta_q16 / 65536, "masked_fraction": f, "kept": 1000,
            "r_sys": r, "exposure_cost_uniform_loss": 1.0 / (1.0 - f)}


def test_an_exact_tie_goes_first_in_order_under_one_rule_and_to_the_selector_order_under_the_other():
    r = 10.933749754233736
    rows = [_row(18, 400_000, 0.9, r - 2e-15), _row(4, 450_000, 0.9, r), _row(7, 300_000, 0.95, r + 1e-15)]
    key = lambda p: (p["masked_fraction"], p["rho"], p["eta_q16"])          # noqa: E731
    first = ar.FIRST_MINIMUM.least(rows, lambda p: p["r_sys"], key)
    selector = surface.SELECTOR_ORDER.least(rows, lambda p: p["r_sys"], key)
    assert (first["rho"], first["eta_q16"]) == (18, 400_000)                 # the last digit decides
    assert (selector["rho"], selector["eta_q16"]) == (4, 450_000)            # the least mask, rank, multiplier


def test_the_frontiers_differ_on_a_rounding_plateau():
    floor = 10.933749754233736
    rows = [_row(1, 70_000, 0.6, floor), _row(1, 70_001, 0.8, math.nextafter(floor, 0.0)),
            _row(1, 70_002, 0.95, math.nextafter(math.nextafter(floor, 0.0), 0.0))]
    first = ar.FIRST_MINIMUM.frontier(rows, 30)
    selector = surface.SELECTOR_ORDER.frontier(rows, 30)
    assert [p.masked_fraction for p in first] == [0.6, 0.8, 0.95]
    assert [p.masked_fraction for p in selector] == [0.6]
