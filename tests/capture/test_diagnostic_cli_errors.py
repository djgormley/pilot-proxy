"""Capture diagnostics reject missing configuration before loading products."""
from types import SimpleNamespace

import pytest

from pilot_proxy.capture import cadence_report, control_level
from pilot_proxy.capture.diagnostics import bearing


def test_bearing_help_needs_no_project(monkeypatch, capsys):
    def unavailable(*args):
        raise AssertionError("help must not open the project")

    monkeypatch.setattr(bearing, "_project", unavailable)
    assert bearing.main(["--help"]) == 0
    assert "usage: pilot-proxy capture bearing" in capsys.readouterr().out


def test_bearing_selftest_needs_no_campaign_record(monkeypatch):
    project = SimpleNamespace()
    called = []
    monkeypatch.setattr(bearing, "_project", lambda path: project)
    monkeypatch.setattr(bearing, "selftest", called.append)
    assert bearing.main(["--selftest"]) == 0
    assert called == [project]


def test_bearing_project_option_needs_a_value(capsys):
    assert bearing.main(["--project"]) == 2
    assert "--project requires a directory" in capsys.readouterr().err


def test_control_level_rejects_missing_layout_before_products(monkeypatch, tmp_path, capsys):
    project = SimpleNamespace(instrument=SimpleNamespace(feed_layout=None))
    monkeypatch.setattr(control_level, "resolve_project", lambda path: project)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        control_level.main(["--datasets", str(tmp_path / "missing"), "--out-dir", str(out)])
    assert exc.value.code == 2
    assert "the instrument records no feed layout" in capsys.readouterr().err
    assert not out.exists()
    with pytest.raises(ValueError, match="no feed layout"):
        control_level.rows_for({}, [])


def test_cadence_report_missing_optional_input_is_a_usage_error(tmp_path, capsys):
    out = tmp_path / "report.md"
    assert cadence_report.main([str(out), "tau.csv", "lag.csv", str(tmp_path / "missing.csv")]) == 2
    assert "cannot read cadence input" in capsys.readouterr().err
    assert not out.exists()
