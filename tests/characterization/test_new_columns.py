"""The columns M3 adds to the surface: the incoherent residual, the floor share, the ideal-model threshold and
the booked chain components."""
from __future__ import annotations

import math

import pytest
from scipy import stats

from characterization_fixtures import characterize
from pilot_proxy.characterization import false_alarm, oc_table
from pilot_proxy.config.project import default_project


@pytest.fixture(scope="module")
def handoff(tmp_path_factory):
    summary, characterization = characterize(tmp_path_factory.mktemp("new_columns"), replay=None)
    assert summary["errors"] == []
    return (oc_table.read_rows(characterization / "oc_table.csv"),
            oc_table.read_rows(characterization / "oc_summary.csv"))


def _ulps(a: float, b: float) -> float:
    return abs(a - b) / math.ulp(max(abs(a), abs(b)))


def test_the_incoherent_residual_times_the_gain_is_the_residual(handoff):
    table, summary = handoff
    gain = {r["band_id"]: float(r["chain_gain"]) for r in summary}
    checked = 0
    for r in table:
        if r["candidate_set"] != "fine_surface" or not r["r_sys"] or not r["r_sys_incoherent"]:
            continue
        r_sys, incoherent = float(r["r_sys"]), float(r["r_sys_incoherent"])
        assert _ulps(incoherent * gain[r["band_id"]], r_sys) <= 64
        assert r_sys != incoherent or gain[r["band_id"]] == 1.0          # reported beside r_sys, never in its place
        checked += 1
    assert checked


def test_the_floor_share_is_a_fraction(handoff):
    table, _ = handoff
    shares = [float(r["floor_share"]) for r in table if r["floor_share"]]
    assert shares and all(0.0 <= s <= 1.0 for s in shares)


def test_the_ideal_model_threshold_is_the_null_laws_quantile_over_its_mean(handoff):
    _, summary = handoff
    project = default_project()
    dof = project.detector_config.coarse_null_dof(project.instrument)
    alpha = float(project.register.value("detection.false_alarm_target"))
    expected = float(stats.f.isf(alpha, *dof) / stats.f.mean(*dof))
    assert false_alarm.ideal_model_threshold() == expected
    q_rows = [r for r in summary if r["candidate_set"] != "fine_surface"]
    fine_rows = [r for r in summary if r["candidate_set"] == "fine_surface"]
    assert q_rows and fine_rows
    # a Q quantity on Q rows, the fine stage's OS-CFAR design value on fine rows (H2)
    assert {float(r["eta_pfa_ideal_model"]) for r in q_rows} == {expected}
    assert {r["pfa_design_model"] for r in fine_rows} == {repr(alpha)}
    assert {r["eta_pfa_ideal_model"] for r in fine_rows} == {""} and {r["pfa_design_model"] for r in q_rows} == {""}


def test_the_default_model_books_one_chain_component(handoff):
    _, summary = handoff
    for r in summary:
        assert r["variance_split"] == "off"
        pairs = r["chain_components"].split(";")
        assert len(pairs) == 1 and pairs[0].split(":")[0] == "1.0"
        assert float(pairs[0].split(":")[1]) == float(r["chain_gain"])
        assert r["intraday_share"] == "" and r["fast_share"] == ""
