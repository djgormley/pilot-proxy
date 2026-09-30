"""The capture and records command groups."""
from __future__ import annotations

import pytest

pytest.importorskip("yaml")

from pilot_proxy import cli  # noqa: E402

CAPTURE = {"frame-residual", "cadence", "cadence-report", "control-level", "class-excess", "marker-map",
           "marker-to-inband", "ladder-place", "tau-bounds", "oc-table"}


def test_the_capture_group_lists_its_steps(capsys):
    assert set(cli.CAPTURE_STEPS) == CAPTURE
    with pytest.raises(SystemExit):
        cli.main(["capture", "--help"])
    out = capsys.readouterr().out
    assert all(step in out for step in CAPTURE)


def test_every_step_module_has_a_main():
    import importlib
    for module in {**cli.CAPTURE_STEPS, **cli.RECORD_STEPS}.values():
        if module.endswith("frame_policy"):
            continue          # a record: run as __main__, not imported for a main
        assert callable(importlib.import_module(module).main), module


def test_a_step_receives_its_arguments(tmp_path, capsys):
    out = tmp_path / "m.json"
    assert cli.main(["capture", "marker-map", "--out", str(out)]) == 0
    assert out.exists() and "wrote" in capsys.readouterr().out


def test_the_records_group(capsys):
    assert set(cli.RECORD_STEPS) == {"frame-policy"}
    with pytest.raises(SystemExit):
        cli.main(["records", "--help"])
    assert "frame-policy" in capsys.readouterr().out
