"""The report core: ledger loading, rendering conventions, the writer."""
from __future__ import annotations

import json
import math

import pytest

from pilot_proxy.characterization.report import core


def test_rendering_conventions():
    assert core.fmt(0.47472, 3) == "0.475" and core.fmt(-1.5, 1) == "-1.5" and core.fmt(float("nan")) == "--"
    assert core.fmt(None, dash="n/a") == "n/a" and core.fmt(0.000123456, 3, sig=True) == "0.000123"
    assert core.fmt(1234.6, 2, sig=True) == "1235" and core.fmt(0.0, 2, sig=True) == "0" and core.fmt(2.0, 1, plus=True) == "+2.0"
    assert core.fmt_int(1566) == "1{,}566" and core.fmt_int(12) == "12" and core.fmt_int(-2000) == "-2{,}000"
    assert core.fmt_int(1566, thousands=False) == "1566" and core.fmt_int(float("nan")) == "--"
    assert core.fmt_month("2024-12") == "2024-12" and core.fmt_month("") == "--"
    assert core.fmt_range(1.234, 2.345) == "1.23--2.35" and core.fmt_range(math.nan, 1) == "--"
    assert core.tex("a_b & 5% #1") == r"a\_b \& 5\% \#1"


def test_booktabs_checks_its_shape():
    t = core.booktabs(["ch", "$f$"], [["14", "0.5"], ["15", "0.6"]], "lr", midrules=[1])
    assert t.splitlines() == [r"\begin{tabular}{lr}", r"\toprule", r"ch & $f$ \\", r"\midrule", r"14 & 0.5 \\", r"\midrule",
                              r"15 & 0.6 \\", r"\bottomrule", r"\end{tabular}"]
    with pytest.raises(ValueError, match="alignment"):
        core.booktabs(["a", "b"], [], "l")
    with pytest.raises(ValueError, match="cells"):
        core.booktabs(["a", "b"], [["1"]], "ll")


def _ledger(tmp_path):
    ledger = tmp_path / "ledger"
    (ledger / "channels").mkdir(parents=True)
    run = {"schema": {"name": "rfisher-archive-ledger", "version": 1}, "generated": "2026-09-07T00:00:00+00:00",
           "producer": {"commit": "a" * 40, "dirty": False, "source_digest": "c" * 64}, "products": {"552.npz": "b" * 64},
           "channels": ["channels/ch33_fid552.json"],
           "era_config_digest": "c" * 64}
    (ledger / "run.json").write_text(json.dumps(run))
    rec = {"channel": 33, "freq_id": 552, "product": "552.npz", "product_sha256": "b" * 64, "notes": ["n1"],
           "sections": {"era": {"current_first_month": "2023-12", "current_frames": 20000}, "selection": None,
                        "null": {"floor_db": -46.1, "floor_evidence": "stated"}}}
    (ledger / "channels" / "ch33_fid552.json").write_text(json.dumps(rec))
    return tmp_path


def test_load_run_and_channel_access(tmp_path):
    run = core.load_run(_ledger(tmp_path))
    assert run.commit == "a" * 40 and run.generated.startswith("2026-09-07") and [c.channel for c in run.channels] == [33]
    c = run.by_channel()[33]
    assert c.era["current_first_month"] == "2023-12" and c.get("era", "current_frames") == 20000
    assert not c.has("selection") and c.selection == {} and c.get("selection", "rho", "absent") == "absent"
    assert c.null["floor_db"] == -46.1 and c.notes == ("n1",) and c.freq_id == 552
    assert len(run.inputs()) == 2


def test_write_report_emits_fragments_numbers_and_manifest(tmp_path):
    run = core.load_run(_ledger(tmp_path))

    def build(run):
        c = run.channels[0]
        frag = core.Fragment("demo_table", "tab:demo", core.booktabs(["ch", "floor"], [[str(c.channel), core.fmt(c.null["floor_db"], 1)]], "lr"))
        frag.add(f"demo.floor_db.ch{c.channel}", c.null["floor_db"], precision=1, row={"channel": c.channel}, column="floor")
        frag.add("demo.evidence.ch33", "stated", kind="text", renderings=("stated",))
        frag.notes.append("one channel")
        return frag

    manifest = core.write_report(run, tmp_path / "dissertation", [build], commit="d" * 40, generated="2026-09-07T01:00:00+00:00")
    out = tmp_path / "dissertation"
    assert (out / "tables" / "demo_table.tex").read_text().startswith(r"\begin{tabular}{lr}")
    doc = json.loads((out / "numbers" / "demo_table.numbers.json").read_text())
    assert doc["producer"]["commit"] == "d" * 40 and [n["key"] for n in doc["numbers"]] == ["demo.floor_db.ch33", "demo.evidence.ch33"]
    assert doc["numbers"][0]["source"] == {"table": "demo_table.tex", "row": {"channel": 33}, "column": "floor"}
    assert len(doc["inputs"]) == 2
    m = json.loads((out / "export_manifest.json").read_text())
    assert m == manifest and m["schema"]["name"] == "rfisher-archive-report" and m["artifacts"][0]["label"] == "tab:demo"
    # the run's own producer travels with the report: a report is only as clean as the run it renders
    assert m["producer"] == {"commit": "a" * 40, "dirty": False, "source_digest": "c" * 64}
    assert m["artifacts"][0]["count"] == 2 and m["artifacts"][0]["notes"] == ["one channel"] and m["run"]["channels"] == [33]
