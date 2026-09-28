"""The per-channel ledger: prefixes, absent sections, JSON and CSV views."""
from __future__ import annotations

import json
import math

import numpy as np

from pilot_proxy.characterization import ledger as lg


def test_ledger_merges_sections_and_writes_json_and_csv(tmp_path):
    book = lg.Ledger(run={"commit": "abc", "configs": {"eras": "f8e1"}, "products": str(tmp_path)})
    rec = lg.ChannelRecord(36, 506, "506.npz", "0" * 64)
    rec.add("era", {"channel": 36, "freq_id": 506, "current_first_month": "2021-08", "frames": np.int64(28945), "coverage": 0.98})
    rec.add("selection", {"status": "feasible", "eta": 1.5, "R": math.nan, "plateau": (1, 2)})
    rec.add("containment", None)
    rec.notes.append("fine")
    book.add(rec)
    rec2 = lg.ChannelRecord(14, 844, "844.npz", "1" * 64)
    rec2.add("era", {"current_first_month": "2018-12", "frames": 36162, "coverage": 1.0, "extra": "x"})
    book.add(rec2)
    paths = book.write(tmp_path / "ledger")
    run = json.loads(paths["run"].read_text())
    assert run["schema"] == lg.SCHEMA and run["channels"][0].endswith("ch14_fid844.json")
    ch36 = json.loads(paths["ch36"].read_text())
    assert ch36["sections"]["era"]["frames"] == 28945 and ch36["sections"]["selection"]["R"] is None
    assert ch36["sections"]["containment"] is None and ch36["sections"]["selection"]["plateau"] == [1, 2]
    rows = lg.load_ledger_rows(paths["ledger"])
    assert [r["channel"] for r in rows] == ["14", "36"]
    by = {r["channel"]: r for r in rows}
    assert by["36"]["era_current_first_month"] == "2021-08" and by["36"]["selection_R"] == "" and by["36"]["containment_present"] == "False"
    assert by["14"]["era_extra"] == "x" and by["14"]["selection_status"] == ""
    assert by["36"]["notes"] == "fine"
