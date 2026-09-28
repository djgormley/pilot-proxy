"""The two named record configurations, and a tie the two rules break differently."""
from __future__ import annotations

import math

import pytest

from pilot_proxy.characterization import surface
from pilot_proxy.config.project import default_project
from pilot_proxy.records.chime_atsc_2026 import archive_releases as ar


def test_both_releases_are_named_with_their_choices():
    assert set(ar.RECORDS) == {"archive_author_eras_2026_09_23", "archive_no_split_2026_09_24_r2"}
    booked = ar.record("archive_author_eras_2026_09_23")
    off = ar.record("archive_no_split_2026_09_24_r2")
    assert (booked.variance_split, booked.ties.name) == ("booked_when_tau_usable", "first_minimum")
    assert (off.variance_split, off.ties.name) == ("off", "selector_order")
    base = default_project().integration_model
    assert base.variance_split == "off"
    model = booked.integration_model(base)
    assert model.variance_split == "booked_when_tau_usable"
    assert (model.frame_seconds, model.coherence_cap_seconds) == (base.frame_seconds, base.coherence_cap_seconds)
    with pytest.raises(KeyError, match="unknown archive record"):
        ar.record("archive_v5_2026_09_07")


def _row(rho, eta_q16, f, r):
    return {"rho": rho, "eta_q16": eta_q16, "eta": eta_q16 / 65536, "masked_fraction": f, "kept": 1000,
            "r_sys": r, "exposure_cost_uniform_loss": 1.0 / (1.0 - f)}


def test_an_exact_tie_goes_first_in_order_under_one_rule_and_to_the_selector_order_under_the_other():
    r = 10.933749754233736
    rows = [_row(18, 400_000, 0.9, r - 2e-15), _row(4, 450_000, 0.9, r), _row(7, 300_000, 0.95, r + 1e-15)]
    key = lambda p: (p["masked_fraction"], p["rho"], p["eta_q16"])          # noqa: E731
    first = ar.FIRST_MINIMUM.least(rows, lambda p: p["r_sys"], key)
    selector = surface.SELECTOR_ORDER.least(rows, lambda p: p["r_sys"], key)
    assert (first["rho"], first["eta_q16"]) == (18, 400_000)                 # the last digit decides
    assert (selector["rho"], selector["eta_q16"]) == (4, 450_000)            # the least mask, rank, multiplier


def test_the_frontiers_differ_on_a_rounding_plateau():
    floor = 10.933749754233736
    rows = [_row(1, 70_000, 0.6, floor), _row(1, 70_001, 0.8, math.nextafter(floor, 0.0)),
            _row(1, 70_002, 0.95, math.nextafter(math.nextafter(floor, 0.0), 0.0))]
    first = ar.FIRST_MINIMUM.frontier(rows, 30)
    selector = surface.SELECTOR_ORDER.frontier(rows, 30)
    assert [p.masked_fraction for p in first] == [0.6, 0.8, 0.95]
    assert [p.masked_fraction for p in selector] == [0.6]
