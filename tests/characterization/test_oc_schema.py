"""The operating-characteristic handoff: writer and schema agree, and the golden handoff regenerates."""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import pytest

from characterization_fixtures import GOLDEN, characterize, write_golden
from pilot_proxy.characterization import oc_table

ROOT = Path(__file__).resolve().parents[2]
DOCS_SCHEMA = ROOT / "docs" / "schemas" / f"{oc_table.SCHEMA_ID}.schema.json"


def test_schema_version_is_pinned():
    assert (oc_table.SCHEMA_NAME, oc_table.SCHEMA_VERSION) == ("operating_characteristic", 1)
    assert oc_table.SCHEMA_ID == "operating_characteristic_v1"
    assert oc_table.MASK_RULE == "statistic > eta"


def test_the_package_schema_is_the_writer_schema_and_the_docs_copy_is_identical():
    assert oc_table.SCHEMA_FILE.read_text(encoding="utf-8") == oc_table.schema_text()
    assert DOCS_SCHEMA.read_bytes() == oc_table.SCHEMA_FILE.read_bytes()
    spec = json.loads(oc_table.SCHEMA_FILE.read_text(encoding="utf-8"))
    assert [c["name"] for c in spec["x-oc-table-columns"]] == list(oc_table.column_names())
    assert [c["name"] for c in spec["x-oc-summary-columns"]] == list(oc_table.column_names(oc_table.SUMMARY_COLUMNS))
    assert spec["properties"]["mask_rule"]["const"] == oc_table.MASK_RULE


def test_no_column_carries_a_science_quantity():
    names = set(oc_table.column_names()) | set(oc_table.column_names(oc_table.SUMMARY_COLUMNS))
    assert "r_sys_no_split" not in names
    assert not {n for n in names if n.startswith(("r_tol", "R_")) or n in ("R", "feasible", "verdict")}


def test_writer_keeps_the_declared_order_and_round_trips_floats(tmp_path):
    values = [0.1, 1 / 3, 2.0 ** -1074, 1e300, -0.0, math.nan]
    rows = [{"band_id": "7", "eta": v, "kept": 3, "mark": None} for v in values]
    path = oc_table.write_rows(rows, tmp_path / "oc_table.csv")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        assert tuple(next(reader)) == oc_table.column_names()
        cells = [row for row in reader]
    eta = oc_table.column_names().index("eta")
    for value, row in zip(values, cells):
        if math.isnan(value):
            assert row[eta] == ""
        else:
            assert float(row[eta]) == value and row[eta] == repr(value)
    assert all(row[oc_table.column_names().index("mark")] == "" for row in cells)


def test_an_undeclared_column_is_refused(tmp_path):
    with pytest.raises(ValueError, match="undeclared columns"):
        oc_table.write_rows([{"band_id": "1", "tolerance": 1.0}], tmp_path / "oc_table.csv")


def test_the_gzip_copy_is_deterministic(tmp_path):
    path = oc_table.write_rows([{"band_id": "1", "eta": 1.5}], tmp_path / "oc_table.csv")
    first = oc_table.write_gzip_copy(path).read_bytes()
    second = oc_table.write_gzip_copy(path).read_bytes()
    assert first == second and gzip.decompress(first) == path.read_bytes()


def test_a_float_not_in_repr_form_fails_the_check(tmp_path):
    path = oc_table.write_rows([{"band_id": "1", "eta": 1.5}], tmp_path / "oc_table.csv")
    text = path.read_text(encoding="utf-8").replace(",1.5,", ",1.50,")
    path.write_text(text, encoding="utf-8")
    assert any("repr form" in p for p in oc_table.check_table(path, oc_table.OC_COLUMNS))


def test_the_golden_handoff_validates_and_regenerates_byte_identical(tmp_path):
    assert oc_table.validate_directory(GOLDEN) == []
    summary, characterization = characterize(tmp_path)
    assert summary["errors"] == []
    assert oc_table.validate_directory(characterization) == []
    regenerated = write_golden(characterization, tmp_path / "golden")
    for name in ("oc_table.csv", "oc_summary.csv", "manifest.json"):
        assert (regenerated / name).read_bytes() == (GOLDEN / name).read_bytes(), name
