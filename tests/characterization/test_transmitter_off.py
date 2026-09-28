"""The transmitter-off record: channels 20 and 32 are independently verified (N1), the others are not."""
from __future__ import annotations

import json

import pytest

from pilot_proxy.config._files import ProfileError
from pilot_proxy.config.eras import EXTERNAL_RECORD_CLASSES, load_era_list
from pilot_proxy.config.project import default_project


@pytest.fixture(scope="module")
def eras():
    return default_project().eras


def test_twenty_and_thirty_two_are_verified_with_their_evidence(eras):
    verified = {label for label, items in eras.transmitter_off.items() if any(i.independently_verified for i in items)}
    assert verified == {"20", "32"}
    for label in ("20", "32"):
        (interval,) = eras.transmitter_off[label]
        assert interval.external_record == "confirmed"
        assert "N1" in interval.evidence and "2020-391" in interval.evidence
        assert "2021-12-07" in interval.evidence and "2022-09-14" in interval.evidence
    assert eras.transmitter_off["20"][0].from_month == "2022-09"
    assert eras.transmitter_off["32"][0].from_month == "2023-02"


def test_the_others_name_their_external_record(eras):
    classes = {label: items[0].external_record for label, items in eras.transmitter_off.items()}
    assert classes == {"19": "not found", "20": "confirmed", "26": "not found", "27": "consistent",
                       "32": "confirmed", "35": "consistent"}
    assert set(classes.values()) <= set(EXTERNAL_RECORD_CLASSES)
    assert eras.transmitter_off["35"][0].through_month == "2021-10"


def test_the_record_round_trips_and_names_its_source(tmp_path):
    project = default_project()
    path = project.directory / "eras" / "transmitter_off.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "transmitter_off_records.md" in raw["source"]
    copy = tmp_path / "transmitter_off.json"
    copy.write_text(json.dumps(raw), encoding="utf-8")
    author = project.directory / "eras" / "author_eras_2026-09-23.json"
    again = load_era_list(author, copy, project.eras.states)
    assert again.transmitter_off == project.eras.transmitter_off


def test_verified_needs_a_confirmed_record(tmp_path):
    project = default_project()
    path = project.directory / "eras" / "transmitter_off.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["bands"]["27"][0]["independently_verified"] = True          # its record is only consistent
    copy = tmp_path / "transmitter_off.json"
    copy.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ProfileError, match="confirmed external record"):
        load_era_list(project.directory / "eras" / "author_eras_2026-09-23.json", copy, project.eras.states)
