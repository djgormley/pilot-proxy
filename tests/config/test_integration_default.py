"""The profile's integration model books no variance split by default (C3)."""
from __future__ import annotations

from pilot_proxy.config.project import default_project


def test_the_default_is_off_and_the_profile_names_the_other_option():
    project = default_project()
    assert project.integration_model.variance_split == "off"
    text = (project.directory / "integration_model.yaml").read_text(encoding="utf-8")
    assert 'variance_split: "off"' in text and "booked_when_tau_usable" in text
