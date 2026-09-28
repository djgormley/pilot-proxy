"""C4: a verified off population that is not null-like calibrates nothing; the run is as with no population."""
from __future__ import annotations

import json

from characterization_fixtures import characterize, products_dir, project_copy
from pilot_proxy.characterization import oc_table


def test_a_verified_but_not_null_like_population_changes_no_table(tmp_path):
    products = products_dir(tmp_path)
    plain, plain_dir = characterize(tmp_path, products=products, out="plain")
    project = project_copy(tmp_path)
    record = project / "eras" / "transmitter_off.json"
    raw = json.loads(record.read_text(encoding="utf-8"))
    raw["bands"]["29"] = [{"from": "2023-06", "through": None, "evidence": "test: a confirmed record",
                           "external_record": "confirmed", "independently_verified": True}]
    record.write_text(json.dumps(raw), encoding="utf-8")
    gated, gated_dir = characterize(tmp_path, products=products, project_dir=project, out="gated")
    assert plain["errors"] == [] and gated["errors"] == []
    for name in ("tables/nulls.csv", "tables/eras.csv", "tables/anchors.csv"):
        assert (gated_dir / name).read_bytes() == (plain_dir / name).read_bytes(), name
    # every surface cell is unchanged; only the stated reason there is no P_fa names the rejected population
    plain_rows = oc_table.read_rows(plain_dir / "oc_table.csv")
    gated_rows = oc_table.read_rows(gated_dir / "oc_table.csv")
    assert [{k: v for k, v in r.items() if k != "pfa_status"} for r in gated_rows] == \
        [{k: v for k, v in r.items() if k != "pfa_status"} for r in plain_rows]
    assert {r["pfa_status"] for r in gated_rows if r["band_id"] == "29"} == {
        "unavailable: " + next(r for r in oc_table.read_rows(gated_dir / "oc_summary.csv")
                               if r["band_id"] == "29")["null_rejection_reason"]}
    # the residual chain reads transmitter-on frames outside any dated epoch, as the releases did; its shelf floor
    # describes the dated off frames and calibrates nothing
    chain_plain = oc_table.read_rows(plain_dir / "tables" / "chain.csv")
    chain_gated = oc_table.read_rows(gated_dir / "tables" / "chain.csv")
    assert [r["chain_gain"] for r in chain_gated] == [r["chain_gain"] for r in chain_plain]
    ledger = json.loads((gated_dir / "ledger" / "channels" / "ch29_fid614.json").read_text())
    era = ledger["sections"]["era"]
    assert era["off_independently_verified"] is True and era["off_population_null_like"] is False
    assert era["off_signal_free"] is False
    summary = {r["band_id"]: r for r in oc_table.read_rows(gated_dir / "oc_summary.csv")}
    assert summary["29"]["eta_pfa_status"].startswith("unavailable: off-state, not signal-free: ")
    assert summary["29"]["floor_verified"] == "False"
    assert summary["34"]["eta_pfa_status"] == oc_table.read_rows(plain_dir / "oc_summary.csv")[-1]["eta_pfa_status"]
