"""The command groups: characterize, products and bench dispatch to their modules."""
from __future__ import annotations

import json

import pytest

from characterization_fixtures import products_dir
from pilot_proxy import cli


@pytest.mark.parametrize("group,steps", [
    ("characterize", {"archive", "report", "histogram-frames", "histograms", "spectrogram", "roc"}),
    ("products", {"check-archive"}),
    ("bench", {"estimator-transfer", "transfer-points"}),
])
def test_the_groups_name_their_steps(group, steps):
    table = {"characterize": cli.CHARACTERIZE_STEPS, "products": cli.PRODUCTS_STEPS, "bench": cli.BENCH_STEPS}[group]
    assert set(table) == steps
    for module in table.values():
        assert callable(getattr(__import__(module, fromlist=["main"]), "main"))


@pytest.mark.parametrize("argv", [["characterize", "roc", "--help"], ["characterize", "archive", "--help"],
                                  ["products", "check-archive", "--help"]])
def test_step_help_is_the_steps_own(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 0
    assert "usage: pilot-proxy " + " ".join(argv[:2]) in capsys.readouterr().out


def test_characterize_archive_from_the_command_line(tmp_path, capsys):
    products = products_dir(tmp_path)
    out = tmp_path / "out"
    assert cli.main(["characterize", "archive", "--products", str(products), "--out", str(out), "--workers", "1",
                     "--bootstrap-replicates", "5", "--bootstrap-seed", "3", "--campaign-last-month", "2024-12"]) == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed[printed.index("{"):])
    assert summary["channels"] == [29, 34] and summary["replay_problems"] == []
    assert (out / "characterization" / "manifest.json").is_file()
    assert (out / "characterization" / "oc_summary.csv").is_file()


def test_an_unknown_step_is_refused(capsys):
    with pytest.raises(SystemExit):
        cli.main(["characterize", "verdict"])
