"""The characterization driver end to end on two synthetic bands."""
from __future__ import annotations

import csv
import json

import pytest

from characterization_fixtures import BANDS, characterize
from pilot_proxy.characterization import ledger, oc_table


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    summary, characterization = characterize(tmp)
    return summary, characterization


def test_every_band_is_characterized_and_the_handoff_validates(run):
    summary, out = run
    assert summary["errors"] == [] and summary["replay_problems"] == []
    assert summary["channels"] == list(BANDS)
    assert oc_table.validate_directory(out) == []
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["mask_rule"] == "statistic > eta" and manifest["residual_shape"] == "band_thermal"
    assert [b["band_id"] for b in manifest["bands"]] == [str(b) for b in BANDS]
    assert manifest["null_policy"]["alpha"] == 0.001
    assert manifest["integration_model"]["variance_split"] == "off"
    assert set(manifest["inputs"]["products"]) == {p.name for p in (out.parent.parent / "products").glob("*.npz")}
    assert "commit" in manifest["producer"]


def test_the_tables_keep_the_release_layout_without_science_columns(run):
    _, out = run
    for name in ("eras", "eras_channels", "anchors", "containment", "kstar", "nulls", "chain", "flaggers"):
        assert (out / "tables" / f"{name}.csv").is_file(), name
    with (out / "tables" / "chain.csv").open(newline="") as fh:
        header = next(csv.reader(fh))
    assert not {"delay_key", "delay_suppression_db", "intraday_share", "fast_share", "n_coh_intraday"} & set(header)
    for gone in ("selection.csv", "screening.csv", "worlds.csv", "channel_tolerances.csv"):
        assert not (out / "tables" / gone).exists()
    run_json = json.loads((out / "ledger" / "run.json").read_text())
    assert run_json["schema"] == ledger.SCHEMA and run_json["tie_rule"] == "selector_order"
    assert run_json["bootstrap"] == {"replicates": 8, "seed": 91}
    text = (out / "oc_summary.csv").read_text().lower()
    for word in ("r_tol", "tolerance", "verdict", "excis"):
        assert word not in text, word


def test_the_booked_record_writes_the_split(tmp_path):
    summary, out = characterize(tmp_path, record_name="archive_author_eras_2026_09_23", replay=None)
    assert summary["errors"] == []
    with (out / "tables" / "chain.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {"intraday_share", "fast_share", "ground_filter_db", "n_coh_intraday"} <= set(rows[0])
    run_json = json.loads((out / "ledger" / "run.json").read_text())
    assert run_json["record"] == "archive_author_eras_2026_09_23" and run_json["tie_rule"] == "first_minimum"
    assert json.loads((out / "manifest.json").read_text())["integration_model"]["variance_split"] == \
        "booked_when_tau_usable"
