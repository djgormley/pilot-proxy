"""The capture handoff: the writer and the schema agree, and the golden handoff regenerates byte identical."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from capture_handoff_fixtures import GOLDEN, characterize, write_golden  # noqa: E402
from pilot_proxy.capture import oc_table  # noqa: E402
from pilot_proxy.characterization.oc_table import check_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def handoff(tmp_path_factory):
    return characterize(tmp_path_factory.mktemp("capture_handoff"))


def test_the_handoff_validates(handoff):
    assert oc_table.validate_directory(handoff) == []
    for name, cols in ((oc_table.TABLE, oc_table.TABLE_COLUMNS), (oc_table.SUMMARY, oc_table.SUMMARY_COLUMNS)):
        with open(handoff / name, newline="") as fh:
            header = next(csv.reader(fh))
        assert header == [c[0] for c in cols]


def test_the_manifest_states_no_capture_eta_pfa(handoff):
    m = json.loads((handoff / oc_table.MANIFEST).read_text())
    assert m["schema"] == "capture_operating_characteristic" and m["schema_version"] == 1
    assert m["detection_gate_sigma"] == 3.0 and m["measurement_allowance_db"] == 3.0
    assert m["null_policy"]["alpha"] == 0.001
    st = m["null_policy"]["eta_pfa_status"]
    assert st["37"] == "not defined: control band"
    assert all(v.startswith("unavailable: ") for k, v in st.items() if k != "37")
    assert m["integration"] == {"frame_seconds": 0.04194304, "coherence_cap_seconds": 86164.0905, "n_frame": 16384}
    assert m["classes"]["ns32"] == {"ew": 0, "ns": 32, "n_b": 896}


def test_floats_are_repr_and_round_trip(handoff, tmp_path):
    rows = oc_table.read_table(handoff)
    kept = [r for r in rows if r["event"] == oc_table.KEPT and r["A_net"]]
    assert kept
    for r in kept[:50]:
        assert repr(float(r["A_net"])) == r["A_net"]
        assert float(r["A_net"]) == float(r["A"]) - float(r["control_mean"])
    bad = tmp_path / oc_table.TABLE
    text = (handoff / oc_table.TABLE).read_text().replace(kept[0]["A"], kept[0]["A"] + "0", 1)
    bad.write_text(text)
    assert any("repr form" in p for p in check_table(bad, oc_table.TABLE_COLUMNS))


def test_the_power_form_and_the_gate(handoff):
    for r in oc_table.read_table(handoff):
        if r["event"] == oc_table.KEPT and r["A_net"]:
            a = float(r["A_net"]); nb = int(r["n_b"]); nf = int(r["n_frame"])
            assert float(r["x_power"]) == (a * a * nf * nb if a > 0 else a)
            assert r["measured"] == str(float(r["A"]) > float(r["control_mean"]) + 3.0 * float(r["control_sd"]))
            assert float(r["gate_power"]) == (3.0 * float(r["control_sd"])) ** 2 * nf * nb


def test_the_measured_marker_bank_is_the_rescanned_band_only(handoff):
    rows = oc_table.read_table(handoff)
    assert {r["band_id"] for r in rows if r["bank"] == "measured_marker"} == {"33"}
    assert {r["policy"] for r in rows if r["product"] == "polmax"} == {""}


def test_the_schema_file_and_its_docs_copy_are_the_code_schema():
    text = oc_table.schema_text()
    assert oc_table.SCHEMA_FILE.read_text() == text
    assert (ROOT / "docs" / "schemas" / f"{oc_table.SCHEMA_ID}.schema.json").read_text() == text
    assert oc_table.SCHEMA_ID == "capture_operating_characteristic_v1"


def test_the_manifest_validates_against_the_json_schema(handoff):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(json.loads((handoff / oc_table.MANIFEST).read_text()), oc_table.schema())


def test_the_golden_handoff_validates_and_regenerates_byte_identical(handoff, tmp_path):
    assert oc_table.validate_directory(GOLDEN) == []
    regenerated = write_golden(handoff, tmp_path / "golden")
    for name in (oc_table.TABLE, oc_table.SUMMARY, oc_table.TAU_BOUNDS, oc_table.MANIFEST):
        assert (regenerated / name).read_bytes() == (GOLDEN / name).read_bytes(), name


def test_dumps_whose_baseline_counts_differ_are_refused(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    keys = np.array([(ew, ns, p, p) for (ew, ns) in oc_table.CLASSES for p in (0, 1)])
    for i, ev in enumerate(("E0", "E1")):
        d = tmp_path / f"pilot_reduce_{ev}"; d.mkdir()
        np.savez(d / "487.npz", keys=keys, count=np.full(len(keys), 100 + i),
                 stacks=np.zeros((4 + i, len(keys)), np.complex64))
    record = SimpleNamespace(dataset_dir=lambda root, ev: str(Path(root) / f"pilot_reduce_{ev}"))
    frames, n_b = oc_table._frames_and_counts(str(tmp_path), ["E0"], record, oc_table.Inputs())
    assert frames == {"E0": 4} and set(n_b) == set(oc_table.CLASSES) and set(n_b.values()) == {100}
    with pytest.raises(ValueError, match="differ from those of the first dump"):
        oc_table._frames_and_counts(str(tmp_path), ["E0", "E1"], record, oc_table.Inputs())
