"""Within-dump lower bounds on tau: the least over every variant, the cap at the observed span, the null test."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from pilot_proxy.capture import tau_bounds as T

TF = 16384 * 2.56e-6


@pytest.fixture()
def lane(tmp_path):
    ln = tmp_path / "autocorrelation"; ck = tmp_path / "check_ac"; ln.mkdir(); ck.mkdir()
    with open(ln / "implications.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["channel", "range", "cls", "least_lb95_all_variants_s", "variants_lb95_s"])
        w.writerow([29, "bao_short", "0,32", 0.58, json.dumps({"science_median/rho0.5": 0.9, "science_median/rho0.9": 0.58})])
        w.writerow([31, "bao_long", "3,0", 3.2, json.dumps({"science_median/rho0.5": 3.2, "all14_median/rho0.5": 4.0})])
    with open(ln / "lag_profiles.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["channel", "lag"]); [w.writerow([29, lag]) for lag in range(1, 33)]
    json.dump({"primary_results_stokes_I": {"29 (0,32)": {"tau_lb95_s_exp_block": [0.7, 0.8]},
                                            "31 (3,0)": {"tau_lb95_s_exp_block": [2.5, 2.9]}},
               "least_over_variants": {"tau_lb95_s_without_lag_truncation": {"29 (0,32)": 0.61, "31": 2.4},
                                       "tau_lb95_s_lags_1_2_only": {"29 (0,32)": 0.92, "31": 2.2},
                                       "dchi2_thr_min": {"29": 1.0}}}, open(ck / "summary.json", "w"))
    fits = [("29", "0,32", "primary", 0.7, 0.8, 0.7), ("29", "0,32", "dmax_2", 0.9, 0.9, 0.918),
            ("31", "3,0", "primary", 2.5, 2.9, 2.4), ("31", "3,0", "dmax_2", 2.2, 2.2, 2.2)]
    extra = ("perm_null", "mean_agg", "dumpmean_norm", "drop_pilot", "rho_ep_0", "rho_ep_1")
    with open(ck / "fits_check.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["channel", "cls", "variant", "exp_lb95_s", "block_lb95_s", "lb95_both_s"])
        for r in fits: w.writerow(r)
        for ch, c, full in (("29", "0,32", 0.61), ("31", "3,0", 2.4)):
            for v in extra: w.writerow([ch, c, v, full + 1, full + 1, full + (0.0 if v == "mean_agg" else 1)])
    null = tmp_path / "null_bounds.csv"
    with open(null, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cls", "least_bound_s"])
        for i in range(91): w.writerow(["0,32" if i % 2 else "3,0", 0.01 * (i + 1)])
    b4 = tmp_path / "bounds_b4.json"
    json.dump({"bounds": {"15|3,0": {"range": "bao_long", "variants": {"lane a": 55.0, "check dmax_2": 0.7, "check x": None}}}},
              open(b4, "w"))
    return dict(lane=str(ln), check=str(ck), b3_null=str(null), b4_bounds=str(b4), frame_seconds=TF)


def test_the_bound_is_the_least_over_every_variant_capped_at_the_span(lane):
    lb = T.tau_lower_bounds(**lane)
    o = lb[(29, (0, 32))]
    assert o["lb"] == 0.58 and o["least_variant"] == "lane science_median/rho0.9"
    assert o["span"] == 32 * TF and o["lb_priced"] == 0.58 and not o["capped"]
    o = lb[(31, (3, 0))]
    assert o["lb"] == 2.2 and o["capped"] and o["lb_priced"] == 32 * TF
    assert o["check"] == 2.2 and o["lane"] == 3.2


def test_censored_classes_and_the_fitted_check(lane):
    lb = T.tau_lower_bounds(censor=True, source="fits", calibrate=False, **lane)
    assert lb[(15, (3, 0))]["b4"] and lb[(15, (3, 0))]["lb"] == 0.7
    assert lb[(29, (0, 32))]["variants"]["check lags_1_2_only"] == 0.918


def test_a_bound_passes_only_when_it_beats_the_null(lane):
    lb = T.tau_lower_bounds(calibrate=True, **lane)
    cal = lb[(29, (0, 32))]["calibration"]
    n_at = sum(1 for i in range(91) if 0.01 * (i + 1) >= 0.58)
    assert cal["n_at_or_above"] == n_at and cal["n_cells"] == 91
    assert cal["passes"] == (n_at / 91 <= 0.05)
    assert lb[(31, (3, 0))]["calibration"]["passes"] and lb[(31, (3, 0))]["calibration"]["fcr"] == 0.0


def test_the_table_round_trips(lane, tmp_path):
    table = T.rows(**lane)
    T.write(table, tmp_path / "t.csv")
    # the lane's range grouping (a science rule) is not carried into the detector table
    assert (tmp_path / "t.csv").read_text().splitlines()[0].split(",") == T.COLUMNS and "range" not in T.COLUMNS
    back = T.read(tmp_path / "t.csv")
    for s in T.SETTINGS:
        lb = T.tau_lower_bounds(censor=s[0], source=s[1], calibrate=s[2], **lane)
        assert set(back[s]) == set(lb)
        for k, o in lb.items():
            assert back[s][k]["lb_priced"] == o["lb_priced"] and back[s][k]["variants"] == o["variants"]
            assert back[s][k].get("calibration") == o.get("calibration")
            assert np.isnan(back[s][k]["lane"]) if np.isnan(o["lane"]) else back[s][k]["lane"] == o["lane"]
