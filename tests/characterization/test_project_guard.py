"""Review M4 and M5: a project whose bound layers differ is refused; the code of record is the project's."""
from __future__ import annotations

import pytest

from characterization_fixtures import characterize, products_dir, project_copy
from pilot_proxy.config._files import ProfileError
from pilot_proxy.config.project import RECORDS_PACKAGE, default_project, load_project


def test_a_project_with_another_register_is_refused(tmp_path):
    project = project_copy(tmp_path)
    register = load_project(project).files["register"]
    register.write_text(register.read_text(encoding="utf-8").rstrip() + "\n\n", encoding="utf-8")
    with pytest.raises(ProfileError, match=r"its register differ from the default project's"):
        characterize(tmp_path, products=products_dir(tmp_path), project_dir=project, out="refused")
    assert not (tmp_path / "refused" / "characterization" / "tables").exists()


def test_a_project_with_another_detector_configuration_is_refused(tmp_path):
    project = project_copy(tmp_path)
    config = load_project(project).files["detector_config"]
    config.write_text(config.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    with pytest.raises(ProfileError, match=r"detector_config"):
        characterize(tmp_path, products=products_dir(tmp_path), project_dir=project, out="refused")


def test_the_records_are_resolved_through_the_project(tmp_path):
    project = default_project()
    assert project.records == f"{RECORDS_PACKAGE}.chime_atsc_2026"
    releases = project.record_module("archive_releases")
    assert releases.__name__ == "pilot_proxy.records.chime_atsc_2026.archive_releases"
    assert project.record_module("roc_populations").POPULATIONS is not None
    bare = project_copy(tmp_path)
    spec = (bare / "project.yaml").read_text(encoding="utf-8")
    (bare / "project.yaml").write_text("\n".join(l for l in spec.splitlines() if not l.startswith("records:")) + "\n",
                                       encoding="utf-8")
    with pytest.raises(ProfileError, match="names no records package"):
        load_project(bare).record_module("archive_releases")
    (bare / "project.yaml").write_text(spec.replace(f"{RECORDS_PACKAGE}.chime_atsc_2026", "os"), encoding="utf-8")
    with pytest.raises(ProfileError, match="is not a package under pilot_proxy.records"):
        load_project(bare)
