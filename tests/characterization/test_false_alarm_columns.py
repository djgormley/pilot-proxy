"""The false-alarm columns (review H2, M1, M2, L1, L10): eta_Pfa in Q only, filled only when available, the
per-row P_fa measured on the verified population, alpha threaded through, and the validator's rules."""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from characterization_fixtures import characterize
from pilot_proxy.characterization import false_alarm as fa
from pilot_proxy.characterization import nulls, oc_table, run


def _law(n, seed):
    rng = np.random.default_rng(seed)
    return fa.LAW.rvs(size=n, random_state=rng) / fa.LAW.mean()


def _population(mask, *, verified=True, null_like=True):
    return nulls.OffPopulation(mask=mask, off_through=None, off_from="2022-09", record_frames=int(mask.sum()),
                               off_frames=int(mask.sum()), note="", independently_verified=verified,
                               external_record="confirmed" if verified else "", null_like=null_like, null_check="c")


def _days(n, per_day=60):
    return np.repeat(np.arange(n // per_day + 1), per_day)[:n] * 86400.0 + 3600.0


# ------------------------------------------------------------------ L1: alpha reaches the estimator
def test_alpha_is_threaded_through_the_estimator():
    q = _law(24_000, 5)
    day = np.floor(_days(q.size) / 86400.0).astype(np.int64)
    block = fa.eta_pfa_block(q, day, "a", seed=1, alpha=0.01)
    assert block["eta_pfa"] == float(np.sort(q)[fa.higher_index(q.size, 0.01)])
    assert block["support_rule"] == "n_eff >= 1000"
    assert block["eta_pfa"] < fa.eta_pfa_block(q, day, "a", seed=1)["eta_pfa"]
    # the default is the register's alpha, value for value
    assert fa.eta_pfa_block(q, day, "a", seed=1) == fa.eta_pfa_block(q, day, "a", seed=1, alpha=fa.ALPHA)
    era = np.ones(q.size, dtype=bool)
    limit = fa.false_alarm_limit(20, role="screened", off_population=_population(era), q=q, frame_time=_days(q.size),
                                 current_era=era, alpha=0.01)
    assert limit["pfa_target"] == 0.01 and limit["eta_pfa_status"] == fa.AVAILABLE
    assert limit["eta_pfa"] == fa.eta_pfa_block(q, day, "a", seed=fa.band_seed(20), alpha=0.01)["eta_pfa"]


def test_no_declared_alpha_is_undefined_without_patching():
    out = fa.false_alarm_limit(14, role="screened", alpha=None)
    assert out["eta_pfa_status"] == fa.UNDEFINED and out["pfa_target"] == "not declared"


# ------------------------------------------------------------------ M2: eta_pfa only when available
def test_an_unsupported_point_is_a_diagnostic_not_eta_pfa():
    q = _law(2000, 9)
    era = np.ones(q.size, dtype=bool)
    out = fa.false_alarm_limit(32, role="screened", off_population=_population(era), q=q, frame_time=_days(q.size),
                               current_era=era)
    assert out["eta_pfa_status"].startswith("unsupported: n_eff = ")
    assert math.isnan(out["eta_pfa"]) and math.isnan(out["pfa_upper_95"])
    assert math.isfinite(out["eta_pfa_diagnostic"]) and out["eta_pfa_diagnostic_status"] == out["eta_pfa_status"]
    assert out["null_frames"] == out["eta_pfa_diagnostic_frames"] == q.size
    assert math.isfinite(out["pfa_effective_samples"])


def test_an_available_point_fills_eta_pfa_and_no_diagnostic():
    q = _law(24_000, 3)
    era = np.ones(q.size, dtype=bool)
    out = fa.false_alarm_limit(20, role="screened", off_population=_population(era), q=q, frame_time=_days(q.size),
                               current_era=era)
    assert out["eta_pfa_status"] == fa.AVAILABLE and math.isfinite(out["eta_pfa"]) and math.isfinite(out["pfa_upper_95"])
    assert math.isnan(out["eta_pfa_diagnostic"]) and out["eta_pfa_diagnostic_status"] == ""
    assert out["null_frames"] == q.size and out["eta_pfa_diagnostic_frames"] == 0


def test_a_not_signal_free_band_counts_its_diagnostic_frames_apart():
    q = _law(3000, 4)
    era = np.ones(q.size, dtype=bool)
    out = fa.false_alarm_limit(20, role="screened", off_population=_population(era, null_like=False), q=q,
                               frame_time=_days(q.size), current_era=era)
    assert out["eta_pfa_status"] == "unavailable: off-state, not signal-free: c"
    assert out["null_frames"] == 0 and out["eta_pfa_diagnostic_frames"] == q.size


# ------------------------------------------------------------------ M1 and H2: the row columns
def test_the_row_pfa_is_the_exceedance_on_the_verified_population():
    population = np.array([0.9, 1.0, 1.0, 1.1, 1.2, np.nan])
    assert fa.exceedance(population, 1.0) == 2 / 5            # strict: equality keeps
    assert fa.exceedance(population, None) == 0.0             # no threshold masks nothing
    frontier = [{"evaluable": True, "eta_c": 1.05, "frames": 10, "kept": 8, "masked_fraction": 0.2, "r_sys": 1.0,
                 "r_sys_incoherent": 1.0, "floor_share": 0.0},
                {"evaluable": True, "eta_c": 1.15, "frames": 10, "kept": 9, "masked_fraction": 0.1, "r_sys": 1.0,
                 "r_sys_incoherent": 1.0, "floor_share": 0.0}]
    rows = run._coarse_rows({"band_id": "20"}, frontier, 2.0, fa.MEASURED, population)
    assert [r["pfa"] for r in rows] == [2 / 5, 1 / 5] and {r["pfa_status"] for r in rows} == {"measured"}
    unmeasured = run._coarse_rows({"band_id": "20"}, frontier, 2.0, "unavailable: x")
    assert [r["pfa"] for r in unmeasured] == [None, None]


def test_the_row_status_is_per_statistic():
    band = {"eta_pfa_status": "unavailable: no verified signal-free population (x)"}
    assert fa.row_pfa_status(band, "Q", False) == band["eta_pfa_status"]
    assert fa.row_pfa_status({"eta_pfa_status": "unsupported: n_eff = 12"}, "Q", True) == "measured"
    assert fa.row_pfa_status(band, "Z_rho", True) == fa.row_pfa_status(band, "Z_rho", False) == fa.NOT_COMPUTED_FINE
    fine = fa.fine_row_columns({"pfa_target": 0.001, "eta_pfa": 31.7, "eta_pfa_status": "available",
                                "null_source": "s", "null_rejection_reason": "r", "eta_pfa_diagnostic": 1.0})
    assert fine["eta_pfa_status"] == oc_table.NOT_COMPUTED_FINE == "not computed: statistic Z_rho"
    assert math.isnan(fine["eta_pfa"]) and math.isnan(fine["eta_pfa_diagnostic"])
    assert (fine["pfa_target"], fine["null_source"], fine["null_rejection_reason"]) == (0.001, "s", "r")


@pytest.fixture(scope="module")
def handoff(tmp_path_factory):
    summary, characterization = characterize(tmp_path_factory.mktemp("false_alarm_columns"), replay=None)
    assert summary["errors"] == []
    return characterization


def test_the_run_writes_q_values_on_q_rows_only(handoff):
    table = oc_table.read_rows(handoff / "oc_table.csv")
    summary = oc_table.read_rows(handoff / "oc_summary.csv")
    assert {r["pfa_status"] for r in table if r["statistic"] == "Z_rho"} == {oc_table.NOT_COMPUTED_FINE}
    assert all(r["pfa_status"].startswith("unavailable: ") for r in table if r["statistic"] == "Q")
    fine = [r for r in summary if r["candidate_set"] == "fine_surface"]
    q_rows = [r for r in summary if r["candidate_set"] != "fine_surface"]
    assert fine and q_rows
    assert {r["eta_pfa_status"] for r in fine} == {oc_table.NOT_COMPUTED_FINE}
    assert all(r["eta_pfa_status"].startswith("unavailable: ") for r in q_rows)
    assert oc_table.check_summary_rules(handoff / "oc_summary.csv") == []
    assert oc_table.check_oc_rules(handoff / "oc_table.csv") == []


# ------------------------------------------------------------------ L10: the validator
def _rewrite(path, edit):
    rows = oc_table.read_rows(path)
    edit(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _copy(handoff, tmp_path, name):
    target = tmp_path / name
    target.write_bytes((handoff / name).read_bytes())
    return target


def test_the_validator_refuses_q_values_on_fine_rows_and_a_primary_unsupported_value(handoff, tmp_path):
    path = _copy(handoff, tmp_path, "oc_summary.csv")

    def edit(rows):
        fine = next(r for r in rows if r["candidate_set"] == "fine_surface")
        fine["eta_pfa"], fine["eta_pfa_status"] = "31.726779831683572", "unsupported: n_eff = 1124"
        q_row = next(r for r in rows if r["candidate_set"] != "fine_surface")
        q_row["eta_pfa"], q_row["eta_pfa_status"] = "31.726779831683572", "unsupported: n_eff = 1124"
        q_row["pfa_design_model"] = "0.001"
        other = [r for r in rows if r["candidate_set"] != "fine_surface"][-1]
        other["eta_pfa_status"] = "measured"
    _rewrite(path, edit)
    problems = oc_table.check_summary_rules(path)
    assert any("a fine row's eta_pfa_status" in p for p in problems)
    assert any("a fine row carries Q values ['eta_pfa']" in p for p in problems)
    assert any("eta_pfa filled with status 'unsupported: n_eff = 1124'" in p for p in problems)
    assert any("a Q row carries the fine design value" in p for p in problems)
    assert any("eta_pfa_status 'measured'" in p for p in problems)


def test_the_validator_requires_a_pfa_for_measured_and_none_on_fine_rows(handoff, tmp_path):
    path = _copy(handoff, tmp_path, "oc_table.csv")

    def edit(rows):
        q_row = next(r for r in rows if r["statistic"] == "Q")
        q_row["pfa_status"] = "measured"
        fine = next(r for r in rows if r["statistic"] == "Z_rho")
        fine["pfa_status"], fine["pfa"] = "measured", "0.5"
        wrong = [r for r in rows if r["statistic"] == "Q"][-1]
        wrong["pfa_status"] = "unsupported: n_eff = 12"
    _rewrite(path, edit)
    problems = oc_table.check_oc_rules(path)
    assert any("pfa_status measured without a pfa in [0, 1]" in p for p in problems)
    assert any("a Z_rho row carries a measured pfa" in p for p in problems)
    assert any("pfa_status 'unsupported: n_eff = 12'" in p for p in problems)
