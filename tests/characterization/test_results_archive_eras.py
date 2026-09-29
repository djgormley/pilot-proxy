"""The section 8.1 stable-era procedure on synthetic monthly records and a synthetic product."""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.characterization import blocks, eras
from pilot_proxy.characterization.eras import (
    AMBIGUOUS_ONLY, EVIDENCE_INSTRUMENT, EVIDENCE_SIGN_ON, EVIDENCE_START, EVIDENCE_STATE, EVIDENCE_STATION,
    PROXY_HIGH, PROXY_LOW, EraConfig, MonthRecord, segment,
)
from pilot_proxy.products.reader import Product

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("v5_fixture", ROOT / "tests" / "products" / "test_pilotproxy_v5.py")
v5_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v5_fixture)

M0 = 2020 * 12          # January 2020
NAN = float("nan")


def _rec(month, level, peak=NAN, maps=("a",), frames=100, units=10, days=10, tags=("t",)):
    return MonthRecord(M0 + month, frames, units, days, True, level, "", peak, "detected", frames, tuple(maps), tuple(tags))


def _spans(seg):
    return [(e.first_month - M0, e.last_month - M0, e.state) for e in seg.eras]


# ------------------------------------------------------------- config
def test_config_is_versioned_and_digested():
    cfg = EraConfig()
    assert cfg.digest == hashlib.sha256(cfg.canonical_json().encode()).hexdigest()
    assert cfg.replace(high_db=2.0).digest != cfg.digest
    assert json.loads(cfg.canonical_json())["min_units"] == 5
    with pytest.raises(ValueError):
        EraConfig(low_db=1.5, high_db=1.0)
    with pytest.raises(ValueError):
        EraConfig(one_sided_ambiguous="guess")
    assert eras.state_of(1.0) == PROXY_HIGH and eras.state_of(0.5) == PROXY_LOW
    assert eras.state_of(0.75) == "ambiguous" and eras.state_of(NAN) == "ambiguous"


# -------------------------------------------------------- state rule
def test_ambiguous_months_between_agreeing_neighbours_are_absorbed():
    seg = segment([_rec(0, 2.0), _rec(1, 2.0), _rec(2, 0.7), _rec(3, 0.8), _rec(4, 2.0), _rec(5, 2.0)])
    assert _spans(seg) == [(0, 5, PROXY_HIGH)]
    assert seg.ambiguous_months == 2 and seg.zone_months == () and seg.excursions == ()
    assert seg.eras[0].evidence == EVIDENCE_START and seg.eras[0].populated_months == 6
    assert seg.fallback == "no_off_state"


def test_ambiguous_run_between_disagreeing_states_is_a_transition_zone_with_the_gap_as_uncertainty():
    # months 0-2 high, 3 ambiguous, 4 unpopulated (absent), 5-6 low
    seg = segment([_rec(0, 2.0), _rec(1, 2.0), _rec(2, 2.0), _rec(3, 0.7), _rec(5, 0.1), _rec(6, 0.2)])
    assert _spans(seg) == [(0, 2, PROXY_HIGH), (5, 6, PROXY_LOW)]
    assert seg.zone_months == (M0 + 3,)
    new = seg.eras[1]
    assert new.evidence == EVIDENCE_STATE
    assert new.boundary_uncertainty_months == 2      # months 3 and 4 lie strictly between 2 and 5
    assert new.boundary_gap_months == 1 and new.boundary_ambiguous_months == 1
    assert new.months_spanned == 2 and new.coverage == 1.0
    assert seg.fallback == ""


def test_single_month_excursions_are_flagged_not_transitions():
    seg = segment([_rec(0, 2.0), _rec(1, 2.0), _rec(2, 0.1), _rec(3, 2.0), _rec(4, 2.0)])
    assert _spans(seg) == [(0, 4, PROXY_HIGH)]
    assert [(e.month - M0, e.kind, e.detail) for e in seg.excursions] == [(2, "state", PROXY_LOW)]
    # a trailing single month of the other state is an excursion too: no 2-month persistence
    seg = segment([_rec(0, 2.0), _rec(1, 2.0), _rec(2, 2.0), _rec(3, 0.1)])
    assert _spans(seg) == [(0, 3, PROXY_HIGH)] and len(seg.excursions) == 1
    # two consecutive months of the new state are a transition
    seg = segment([_rec(0, 2.0), _rec(1, 2.0), _rec(2, 0.1), _rec(3, 0.1)])
    assert _spans(seg) == [(0, 1, PROXY_HIGH), (2, 3, PROXY_LOW)]
    assert seg.eras[1].boundary_uncertainty_months == 0


def test_persistence_and_inheritance_iterate_to_a_fixed_point():
    # H A H is one persistent high run; the lone L inside it is an excursion whose removal merges the run
    seg = segment([_rec(0, 2.0), _rec(1, 0.7), _rec(2, 2.0), _rec(3, 0.1), _rec(4, 2.0), _rec(5, 2.0)])
    assert _spans(seg) == [(0, 5, PROXY_HIGH)]
    assert [e.month - M0 for e in seg.excursions] == [3]
    # alternating single months between two persistent states all fall into the zone
    seg = segment([_rec(0, 2.0), _rec(1, 2.0), _rec(2, 0.1), _rec(3, 2.0), _rec(4, 0.7), _rec(5, 0.1), _rec(6, 0.1)])
    assert _spans(seg) == [(0, 1, PROXY_HIGH), (5, 6, PROXY_LOW)]
    assert seg.zone_months == (M0 + 2, M0 + 3, M0 + 4)
    assert sorted(e.month - M0 for e in seg.excursions) == [2, 3]
    assert seg.eras[1].boundary_ambiguous_months == 3 and seg.eras[1].boundary_gap_months == 0


def test_one_sided_ambiguous_runs_inherit_by_default_or_separate_when_asked():
    records = [_rec(0, 0.7), _rec(1, 0.8), _rec(2, 2.0), _rec(3, 2.0), _rec(4, 0.7)]
    seg = segment(records)
    assert _spans(seg) == [(0, 4, PROXY_HIGH)]
    seg = segment(records, EraConfig(one_sided_ambiguous="separate"))
    assert _spans(seg) == [(0, 1, AMBIGUOUS_ONLY), (2, 4, PROXY_HIGH)]     # the trailing single month cannot persist
    assert seg.eras[1].evidence == EVIDENCE_STATE


def test_no_definite_state_is_one_ambiguous_only_era_and_reported_as_fallback():
    seg = segment([_rec(0, 0.7), _rec(1, 0.8), _rec(2, 0.6)])
    assert _spans(seg) == [(0, 2, AMBIGUOUS_ONLY)]
    assert seg.fallback == "no_off_state;no_persistent_state"
    assert segment([]).fallback == "no_populated_months" and segment([]).eras == ()
    # a month below the support gate is not populated and does not enter
    seg = segment([_rec(0, 2.0), _rec(1, 2.0), _rec(2, 0.1, units=2), _rec(3, 0.1, units=2)])
    assert _spans(seg) == [(0, 1, PROXY_HIGH)] and seg.populated == (M0, M0 + 1)


# ------------------------------------------------------ station rule
def test_station_change_needs_a_persistent_shift_against_the_running_location():
    highs = [_rec(i, 2.0, peak=2.0) for i in range(4)] + [_rec(i, 2.0, peak=7.0) for i in range(4, 7)]
    seg = segment(highs)
    assert _spans(seg) == [(0, 3, PROXY_HIGH), (4, 6, PROXY_HIGH)]
    assert seg.eras[1].evidence == EVIDENCE_STATION and seg.eras[1].boundary_uncertainty_months == 0
    assert seg.eras[0].peak_offset_bins == 2.0 and seg.eras[1].peak_offset_bins == 7.0
    # a one-month shift is a station excursion, and a shift below 3 bins is nothing
    seg = segment([_rec(0, 2.0, peak=2.0), _rec(1, 2.0, peak=2.0), _rec(2, 2.0, peak=7.0),
                   _rec(3, 2.0, peak=2.0), _rec(4, 2.0, peak=4.0), _rec(5, 2.0, peak=2.0)])
    assert _spans(seg) == [(0, 5, PROXY_HIGH)]
    assert [(e.month - M0, e.kind, e.detail) for e in seg.excursions] == [(2, "station", "+5.0 bins")]
    # an ambiguous month between the two locations is the zone of the station change
    seg = segment([_rec(0, 2.0, peak=2.0), _rec(1, 2.0, peak=2.0), _rec(2, 0.7, peak=5.0),
                   _rec(3, 2.0, peak=7.0), _rec(4, 2.0, peak=7.0)])
    assert _spans(seg) == [(0, 1, PROXY_HIGH), (3, 4, PROXY_HIGH)]
    assert seg.zone_months == (M0 + 2,) and seg.eras[1].boundary_ambiguous_months == 1


def test_station_change_must_be_confirmed_at_the_new_location():
    # a one-month excursion followed by a genuine change: the excursion month is not promoted to an era
    locs = [-2, -3, 1, -1, 0, 3, 7, 6, 5, 4, 6, 7, 4]
    seg = segment([_rec(i, 2.0, peak=float(v)) for i, v in enumerate(locs)])
    assert _spans(seg) == [(0, 5, PROXY_HIGH), (6, 12, PROXY_HIGH)]
    # month 2 (+3.5 bins for one month) and month 5 (the month before the change) are excursions
    assert [(e.month - M0, e.kind) for e in seg.excursions] == [(2, "station"), (5, "station")]
    assert seg.eras[1].evidence == EVIDENCE_STATION
    # the first tested month disagreeing with two agreeing followers is the excursion, not a one-month era
    seg = segment([_rec(i, 2.0, peak=float(v)) for i, v in enumerate([1, -2, -3, -2.5, -2, -3])])
    assert _spans(seg) == [(0, 5, PROXY_HIGH)]
    assert [(e.month - M0, e.kind, e.detail) for e in seg.excursions] == [(0, "station", "+3.0 bins")]
    assert seg.eras[0].peak_offset_bins == pytest.approx(-2.25)   # the era location counts every tested month, the excursion included


def test_persistence_counts_definite_months_only():
    # one proxy-low month with two trailing ambiguous months does not open a proxy-low era
    seg = segment([_rec(0, 2.0), _rec(1, 2.0), _rec(2, 2.0), _rec(3, 0.3), _rec(4, 0.7), _rec(5, 0.8)])
    assert _spans(seg) == [(0, 5, PROXY_HIGH)]
    assert [(e.month - M0, e.kind) for e in seg.excursions] == [(3, "state")]
    # a leading ambiguous month cannot lend persistence to a single definite month
    seg = segment([_rec(0, 0.7), _rec(1, 2.0), _rec(2, 0.3), _rec(3, 0.3)])
    assert _spans(seg) == [(0, 3, PROXY_LOW)]
    assert [(e.month - M0, e.kind) for e in seg.excursions] == [(1, "state")]


def test_station_rule_ignores_proxy_low_months():
    lows = [_rec(i, 0.1, peak=float(p)) for i, p in enumerate([-20, 5, 17, -3, 9, 25])]
    seg = segment(lows)
    assert _spans(seg) == [(0, 5, PROXY_LOW)] and seg.excursions == ()
    assert math.isnan(seg.eras[0].peak_offset_bins)


# --------------------------------------------------- instrument rule
def test_instrument_change_is_an_entering_map_that_persists():
    seg = segment([_rec(i, 2.0, maps=("a",)) for i in range(3)] + [_rec(i, 2.0, maps=("b",)) for i in range(3, 6)])
    assert _spans(seg) == [(0, 2, PROXY_HIGH), (3, 5, PROXY_HIGH)]
    assert seg.eras[1].evidence == EVIDENCE_INSTRUMENT and seg.instrument_change_months == (M0 + 3,)
    # the map enters inside a mixed month: the change opens there
    seg = segment([_rec(0, 2.0, maps=("a",)), _rec(1, 2.0, maps=("a",)), _rec(2, 2.0, maps=("a", "b")),
                   _rec(3, 2.0, maps=("b",)), _rec(4, 2.0, maps=("b",))])
    assert _spans(seg) == [(0, 1, PROXY_HIGH), (2, 4, PROXY_HIGH)]
    # a map that leaves is not a change; a map present for one month is an excursion
    seg = segment([_rec(0, 2.0, maps=("a", "z")), _rec(1, 2.0, maps=("a",)), _rec(2, 2.0, maps=("a", "c")),
                   _rec(3, 2.0, maps=("a",)), _rec(4, 2.0, maps=("a", "d"))])
    assert _spans(seg) == [(0, 4, PROXY_HIGH)]
    assert [(e.month - M0, e.detail) for e in seg.excursions if e.kind == "instrument"] == [(2, "c"), (4, "d")]


def test_coincident_boundaries_join_their_evidence_and_records_name_transitions():
    records = [_rec(0, 0.1, maps=("a",)), _rec(1, 0.1, maps=("a",)), _rec(2, 2.0, maps=("b",)), _rec(3, 2.0, maps=("b",))]
    seg = segment(records)
    assert seg.eras[1].evidence == f"{EVIDENCE_INSTRUMENT}+{EVIDENCE_STATE}"
    on = blocks.month_label(M0 + 2)
    seg = segment(records, station_record={on: "sign-on"})
    assert seg.eras[1].evidence == f"{EVIDENCE_INSTRUMENT}+{EVIDENCE_SIGN_ON}"
    assert seg.eras[1].record_agreement == f"confirmed by record {on}"
    seg = segment(records, station_record={on: "sign-off", blocks.month_label(M0 + 9): "sign-on"})
    assert seg.eras[1].evidence == f"{EVIDENCE_INSTRUMENT}+{EVIDENCE_STATE}"
    assert "direction disagrees" in seg.eras[1].record_agreement
    assert seg.unmatched_station_records == (blocks.month_label(M0 + 9),)


# ------------------------------------------------------ on a product
HIGH_DB, AMBIGUOUS_DB, LOW_DB = 2.5, 0.75, 0.2
PEAK_OFFSET = 4
MONTHS = 24
UNITS_PER_MONTH = 10
FRAMES_PER_UNIT = 10


def _month_of_unit(unit):
    return unit // UNITS_PER_MONTH


def _derived_arrays(target, p_ref_sum, target_norm, reference_norm, spec):
    """The five float views the residual view re-derives from the exact terms and checks."""
    n = target.size
    t = target.astype(np.float64)
    r = p_ref_sum.astype(np.float64)
    ratio = np.full(n, np.nan)
    np.divide(t * float(reference_norm), r * float(target_norm), out=ratio, where=p_ref_sum > 0)
    coarse = np.full(n, np.nan)
    np.divide(2.0 * t, r, out=coarse, where=p_ref_sum > 0)
    excess = ratio - 1.0
    ratio_db = np.full(n, np.nan)
    ratio_db[ratio > 0.0] = 10.0 * np.log10(ratio[ratio > 0.0])
    excess_db = np.full(n, np.nan)
    excess_db[excess > 0.0] = 10.0 * np.log10(excess[excess > 0.0])
    offset = (spec["pilot_below_data_db"] - 10.0 * np.log10(spec["dtv_bandwidth_hz"] / spec["bin_enbw_hz"])
              - 10.0 * np.log10(spec["pilot_capture_efficiency"]))
    return {
        "coarse_power_ratio": coarse.reshape(-1, 1), "normalized_coarse_power_ratio_db": ratio_db.reshape(-1, 1),
        "normalized_pilot_excess": excess.reshape(-1, 1), "pilot_excess_db": excess_db.reshape(-1, 1),
        "estimated_data_shelf_snr_db": (excess_db + offset).reshape(-1, 1),
    }


def _level_db_of_month(month):
    if month == 5:
        return LOW_DB                       # a single-month excursion inside the high era
    if month < 10:
        return HIGH_DB
    if month == 10:
        return AMBIGUOUS_DB                 # the ambiguous month of the transition zone
    return LOW_DB                           # month 11 is left empty; 12.. are low


@pytest.fixture
def product_path(tmp_path):
    frames = MONTHS * UNITS_PER_MONTH * FRAMES_PER_UNIT
    units = MONTHS * UNITS_PER_MONTH
    path = v5_fixture._write_product(tmp_path / "506.npz", 36, frames=frames, units=units)
    spec = v5_fixture._fixture_spec()
    with np.load(path, allow_pickle=False) as z:
        unit_of_frame = np.asarray(z["frame_unit_index"]).reshape(-1)
        fine = np.array(z["fine_power_u64"], copy=True)
        target_norm = int(np.asarray(z["target_norm_sq"]).reshape(-1)[0])
        reference_norm = int(np.asarray(z["reference_norm_sum_sq"]).reshape(-1)[0])
        p_ref_sum = np.asarray(z["p_ref_sum_u64"]).reshape(-1)
        maps = np.array(z["unit_input_map_sha256"], copy=True)
    # acquisitions: ten per month on ten distinct days, months 22-23 hold two acquisitions only
    time0 = np.empty(units)
    keep = np.ones(units, dtype=bool)
    for u in range(units):
        month = _month_of_unit(u)
        day = u % UNITS_PER_MONTH + 1
        time0[u] = dt.datetime(2020 + month // 12, month % 12 + 1, day, 12, tzinfo=dt.timezone.utc).timestamp()
        if month == 11 or (month >= 22 and day > 2):
            keep[u] = False
    delta = np.full(units, 1.0 / spec["sample_rate_hz"])
    delta[3 * UNITS_PER_MONTH] = np.nan               # one acquisition of month 3 has no sample interval
    # per-frame level: p_target = Q * (target_norm / reference_norm) * p_ref_sum makes 10 log10(F / mu0) = level
    month_of_frame = np.array([_month_of_unit(u) for u in unit_of_frame])
    level = np.array([_level_db_of_month(m) for m in month_of_frame])
    q = 10.0 ** (level / 10.0)
    target = np.rint(q * target_norm / reference_norm * p_ref_sum).astype(np.uint64)
    # dropped acquisitions are invalid frames: the view requires valid == (p_ref_sum != 0)
    valid_frames = keep[unit_of_frame]
    p_ref_sum = np.where(valid_frames, p_ref_sum, 0).astype(np.uint64)
    half = (p_ref_sum // 2).astype(np.uint64)
    rejected = (valid_frames & (target.astype(object) * reference_norm > target_norm * p_ref_sum.astype(object))
                ).astype(np.uint8)
    # fine terms: a flat target with one spike PEAK_OFFSET bins above the nominal bin in the high months
    fine[:, 0, :] = 20
    fine[:, 1, :] = 20
    fine[:, 2, :] = 20
    with Product(path) as p:
        nominal = p.geometry.nominal_fine_bin
    fine[level >= HIGH_DB, 0, (nominal + PEAK_OFFSET) % fine.shape[2]] = 1000
    # instrument: map 'b' from month 8 on (inside the high era); one 'c' acquisition in month 15
    maps[:] = "a" * 64
    maps[np.array([_month_of_unit(u) for u in range(units)]) >= 8] = "b" * 64
    maps[15 * UNITS_PER_MONTH + 4] = "c" * 64
    tags = np.array([f"tag{u % 2}" for u in range(units)])
    v5_fixture._replace(
        path, unit_time0_ctime=time0, unit_delta_time=delta, p_target_u64=target.reshape(-1, 1),
        p_ref_lower_u64=half.reshape(-1, 1), p_ref_upper_u64=half.reshape(-1, 1), p_ref_sum_u64=p_ref_sum.reshape(-1, 1),
        valid=valid_frames.astype(np.uint8).reshape(-1, 1), reject_mask=rejected.reshape(-1, 1),
        fine_power_u64=fine, unit_input_map_sha256=maps, unit_git_version_tag=tags,
        baseband_power_linear=np.full((frames, 1), 4.0),
        **_derived_arrays(target, p_ref_sum, target_norm, reference_norm, spec),
    )
    return path


def test_era_table_on_a_synthetic_product(product_path, tmp_path):
    with Product(product_path) as p:
        table = eras.era_table(p, p.selected, campaign_last_month=M0 + 23)
        assert [(e.first_month - M0, e.last_month - M0, e.state) for e in table.eras] == [
            (0, 7, PROXY_HIGH), (8, 9, PROXY_HIGH), (12, 21, PROXY_LOW)]
        first, second, current = table.eras
        assert first.evidence == EVIDENCE_START and first.boundary_uncertainty_months == 0
        assert second.evidence == EVIDENCE_INSTRUMENT and second.boundary_uncertainty_months == 0
        assert current.evidence == EVIDENCE_STATE
        assert current.boundary_uncertainty_months == 2       # months 10 (ambiguous) and 11 (empty)
        assert current.boundary_gap_months == 1 and current.boundary_ambiguous_months == 1
        assert table.transition_zone_months == (M0 + 10,)
        assert table.current_index == 2 and table.stale_latest and table.stale_reference == "campaign"
        assert table.stale_lag_months == 2 and table.campaign_last_month == M0 + 23
        assert table.fallback == "" and table.indeterminate == ""
        assert table.ambiguous_months == 1 and table.populated_months == 21
        assert table.ambiguous_fraction == pytest.approx(1 / 21)
        # the excursion inside the high era and the one-acquisition map in month 15
        assert [(e.month - M0, e.kind, e.detail) for e in table.excursions] == [
            (5, "state", PROXY_LOW), (15, "instrument", "c" * 8)]
        assert table.instrument_change_months == (M0 + 8,)
        # 214 acquisitions carry selected frames (21 full months and 2 + 2 in the last two); tags alternate
        assert table.software_tags == 2 and table.software_tag_changes == 21 * UNITS_PER_MONTH + 4 - 1
        # frames: 100 per month; month 3's acquisition without a sample interval still counts through unit_time
        assert first.frames == 800 and first.frames_without_time == 10 and first.units == 80
        assert second.frames == 200 and second.units == 20 and second.frames_without_time == 0
        assert current.frames == 1000 and current.units == 100 and current.populated_months == 10
        # 21 populated months of 100 frames (month 11 is empty) plus two acquisitions in each of months 22 and 23
        assert table.frames_without_time == 10 and table.frames_selected == 21 * 100 + 4 * FRAMES_PER_UNIT
        assert first.level_median_db == pytest.approx(HIGH_DB, abs=0.01)
        assert current.level_median_db == pytest.approx(LOW_DB, abs=0.01)
        assert first.peak_offset_bins == second.peak_offset_bins == PEAK_OFFSET and math.isnan(current.peak_offset_bins)
        # the per-frame peak sits at the spike in high months; the monthly record says so
        peaks = eras.frame_peak_offsets(p, p.selected)
        month3 = table.months[3]
        assert month3.peak_offset_bins == PEAK_OFFSET and month3.peak_cohort == "detected"
        assert month3.frames == 90 and month3.units == 9 and month3.days == 9 and month3.populated
        assert np.all(peaks[(p.selected) & (p.level_db > 2.0)] == PEAK_OFFSET)
        assert table.months[10].state == "ambiguous" and table.resolved[M0 + 10] == eras.ZONE
        assert [m.label for m in table.months if not m.populated] == ["2021-11", "2021-12"]
        # masks: the current era's frames are months 12-21; frames with NaN time follow their acquisition's month
        current_mask = table.current_era_mask(p)
        assert current_mask.sum() == 1000
        months = eras.frame_months(p)
        assert np.isnan(p.frame_time[table.era_mask(p, 0)]).sum() == 10
        assert (months[table.era_mask(p, 0)] <= M0 + 7).all() and (months[current_mask] >= M0 + 12).all()
        assert not (table.era_mask(p, 0) & current_mask).any()
        # sensitivity: making 0.75 dB proxy-low (1/1 dB) moves the current-era start to month 10; 0.5/1 is the policy;
        # 0.5/0.5 makes it proxy-high and extends the instrument era instead, leaving the current era in place
        moves = {(o.parameter, o.value): o for o in table.sensitivity}
        assert moves[("thresholds_db", "1/1")].moves and moves[("thresholds_db", "1/1")].first_month == M0 + 10
        assert not moves[("thresholds_db", "0.5/1")].moves and not moves[("thresholds_db", "0.5/0.5")].moves
        assert moves[("thresholds_db", "0.5/0.5")].n_eras == 3
        assert not moves[("min_units", "3")].moves and not moves[("min_units", "10")].moves
        assert moves[("min_units", "10")].span == "2021-01..2021-10"
        # without a campaign month the channel is its own reference and cannot be stale
        own = eras.era_table(p, p.selected)
        assert not own.stale_latest and own.stale_reference == "channel"
        assert own.campaign_last_month == M0 + 21
        assert eras.campaign_last_populated_month([p]) == M0 + 21

        out = eras.write_eras_csv([table], tmp_path / "eras.csv")
        with out.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert tuple(rows[0].keys()) == eras.ERA_COLUMNS and len(rows) == 3
        assert rows[2]["first_month"] == "2021-01" and rows[2]["last_month"] == "2021-10"
        assert [r["is_current"] for r in rows] == ["False", "False", "True"]
        assert rows[2]["stale_latest"] == "True" and rows[2]["coverage"] == "1.0"
        assert rows[2]["peak_offset_bins"] == "" and rows[0]["peak_offset_bins"] == "4.0"
        assert rows[1]["evidence"] == EVIDENCE_INSTRUMENT and rows[1]["first_month"] == "2020-09"
        assert rows[0]["config_digest"] == eras.DEFAULT_CONFIG.digest
        chan = eras.write_channels_csv([table], tmp_path / "eras_channels.csv")
        with chan.open(newline="") as fh:
            crow = list(csv.DictReader(fh))
        assert tuple(crow[0].keys()) == eras.CHANNEL_COLUMNS and len(crow) == 1
        assert crow[0]["current_first_month"] == "2021-01" and crow[0]["transition_zone_months"] == "2020-11"
        assert crow[0]["sensitivity_thresholds_moves"] == "1/1;1/2;2/2" and crow[0]["sensitivity_units_moves"] == ""
        assert crow[0]["state_excursions"] == "2020-06:state:proxy-low"
        assert crow[0]["instrument_change_months"] == "2020-09" and crow[0]["stale_lag_months"] == "2"
        js = json.loads(eras.write_era_json(table, tmp_path / "ch36.json").read_text())
        assert js["eras"][2]["evidence"] == current.evidence and js["config_digest"] == table.config.digest
        assert [m["era"] for m in js["months"]][8:13] == [2, 2, "zone", 3, 3]     # month 11 has no record
        assert js["months"][3]["peak_offset_bins"] == PEAK_OFFSET and js["eras"][2]["peak_offset_bins"] is None


def test_month_records_reject_a_mask_of_the_wrong_shape(product_path):
    with Product(product_path) as p:
        with pytest.raises(ValueError):
            eras.month_records(p, np.ones(3, dtype=bool))
        assert eras.month_records(p, np.zeros(p.n_frames, dtype=bool)) == ()
        assert np.isnan(eras.frame_peak_offsets(p, np.zeros(p.n_frames, dtype=bool))).all()


def test_an_author_dated_era_list_replaces_the_rule_eras_and_keeps_its_month_record(product_path, tmp_path):
    with Product(product_path) as p:
        rule = eras.era_table(p, p.selected, campaign_last_month=M0 + 23)
        spec = [("2020-01", "2020-06", EVIDENCE_START), ("2020-09", "2021-10", eras.EVIDENCE_POWER)]
        table = eras.impose_eras(rule, p, spec)
        assert [(e.first_month - M0, e.last_month - M0) for e in table.eras] == [(0, 5), (8, 21)]
        first, current = table.eras
        assert first.state == PROXY_HIGH and first.frames == 600 and first.units == 60
        assert first.frames_without_time == 10 and first.peak_offset_bins == PEAK_OFFSET
        # the current era runs across the rule's instrument era, its zone month and its low era
        assert current.state == PROXY_LOW and current.evidence == eras.EVIDENCE_POWER
        assert current.frames == 1300 and current.units == 130 and current.populated_months == 13
        assert current.months_spanned == 14 and current.peak_offset_bins == PEAK_OFFSET and current.peak_months == 2
        # months 6 and 7 are populated and in no span: the boundary interval, reported as zone months
        assert current.boundary_uncertainty_months == 2 and current.boundary_ambiguous_months == 2
        assert current.boundary_gap_months == 0 and table.transition_zone_months == (M0 + 6, M0 + 7)
        assert table.current_index == 1 and table.stale_latest and table.stale_lag_months == 2
        assert table.boundary_source == eras.SOURCE_AUTHOR and table.rule_current_era == "2021-01..2021-10"
        assert table.sensitivity == () and table.indeterminate == ""
        # the rule's month record, excursions and instrument changes are kept as recorded
        assert table.months == rule.months and table.excursions == rule.excursions
        assert table.instrument_change_months == rule.instrument_change_months
        mask = table.current_era_mask(p)
        months = eras.frame_months(p)
        assert mask.sum() == 1300 and months[mask].min() == M0 + 8 and months[mask].max() == M0 + 21
        crow = eras.channel_row(table)
        assert crow["boundary_source"] == eras.SOURCE_AUTHOR and crow["rule_current_era"] == "2021-01..2021-10"
        assert crow["current_first_month"] == "2020-09" and crow["sensitivity_thresholds_moves"] == ""
        assert eras.channel_row(rule)["boundary_source"] == eras.PREDECLARED_RULE
        js = json.loads(eras.write_era_json(table, tmp_path / "author.json").read_text())
        assert [m["era"] for m in js["months"]][5:10] == [1, "zone", "zone", 2, 2]
        assert js["boundary_source"] == eras.SOURCE_AUTHOR
        rows = eras.era_rows(table)
        assert [r["boundary_source"] for r in rows] == [eras.SOURCE_AUTHOR] * 2
        # spans must be ordered, disjoint and hold a populated month
        for bad in ([("2020-05", "2020-01", EVIDENCE_START)],
                    [("2020-01", "2020-08", EVIDENCE_START), ("2020-08", "2021-10", eras.EVIDENCE_POWER)],
                    [("2019-01", "2019-06", EVIDENCE_START)], []):
            with pytest.raises(ValueError):
                eras.impose_eras(rule, p, bad)


def _same(a, b):
    """Equality with NaN equal to NaN (era fields and CSV rows carry NaN for absent peaks)."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def _changed(a: dict, b: dict) -> set:
    assert a.keys() == b.keys()
    return {k for k in a if not _same(a[k], b[k])}


@pytest.fixture
def gap_product_path(product_path):
    """The synthetic product with month 11's (invalid) acquisitions moved into month 10, so month 11 holds no frame."""
    with np.load(product_path, allow_pickle=False) as z:
        time0 = np.array(z["unit_time0_ctime"], copy=True)
    for u in range(11 * UNITS_PER_MONTH, 12 * UNITS_PER_MONTH):
        time0[u] = dt.datetime(2020, 11, u % UNITS_PER_MONTH + 15, 12, tzinfo=dt.timezone.utc).timestamp()
    v5_fixture._replace(product_path, unit_time0_ctime=time0)
    return product_path


def test_an_author_dated_first_month_inside_a_frameless_gap_moves_only_the_label(gap_product_path, tmp_path):
    with Product(gap_product_path) as p:
        months = eras.frame_months(p)
        assert not (months == M0 + 11).any() and (months == M0 + 10).sum() == 2 * UNITS_PER_MONTH * FRAMES_PER_UNIT
        rule = eras.era_table(p, p.selected, campaign_last_month=M0 + 23)
        snapped = eras.impose_eras(rule, p, [("2020-01", "2020-10", EVIDENCE_START),
                                             ("2021-01", "2021-10", eras.EVIDENCE_POWER)])
        dated = eras.impose_eras(rule, p, [("2020-01", "2020-10", EVIDENCE_START),
                                           ("2020-12", "2021-10", eras.EVIDENCE_POWER)])
        a, b = snapped.eras[1], dated.eras[1]
        assert (a.first_month, b.first_month) == (M0 + 12, M0 + 11) and a.last_month == b.last_month == M0 + 21
        assert (a.months_spanned, b.months_spanned) == (10, 11) and (a.coverage, b.coverage) == (1.0, 10 / 11)
        # the frames, the month record and the boundary interval are the snapped era's
        assert b.months == a.months and b.frames == a.frames == 1000 and b.units == a.units
        assert b.boundary_uncertainty_months == a.boundary_uncertainty_months == 2
        assert b.boundary_gap_months == a.boundary_gap_months == 1
        assert b.boundary_ambiguous_months == a.boundary_ambiguous_months == 1
        assert _changed(dataclasses.asdict(a), dataclasses.asdict(b)) == {"first_month"}
        assert _changed(dataclasses.asdict(snapped.eras[0]), dataclasses.asdict(dated.eras[0])) == set()
        assert dated.transition_zone_months == snapped.transition_zone_months == (M0 + 10,)
        for index in range(2):
            assert np.array_equal(dated.era_mask(p, index), snapped.era_mask(p, index))
        assert np.array_equal(dated.current_era_mask(p), snapped.current_era_mask(p))
        # the written rows and the era JSON differ only in the dated start and what it spans
        (r0, r1), (d0, d1) = eras.era_rows(snapped), eras.era_rows(dated)
        assert _changed(r0, d0) == set() and _changed(r1, d1) == {"first_month", "months_spanned", "coverage"}
        assert d1["first_month"] == "2020-12" and d1["months_spanned"] == 11
        crow, drow = eras.channel_row(snapped), eras.channel_row(dated)
        assert _changed(crow, drow) == {"current_first_month"} and drow["current_first_month"] == "2020-12"
        js = json.loads(eras.write_era_json(snapped, tmp_path / "snapped.json").read_text())
        jd = json.loads(eras.write_era_json(dated, tmp_path / "dated.json").read_text())
        assert {k for k in js["eras"][1] if js["eras"][1][k] != jd["eras"][1][k]} == {
            "first_month", "months_spanned", "coverage"}
        js["eras"][1] = jd["eras"][1] = None
        assert js == jd


def test_an_author_dated_first_month_over_frames_is_refused(product_path):
    with Product(product_path) as p:
        # month 11's acquisitions are invalid: the month has no record, but its frames are frames of the product
        months = eras.frame_months(p)
        assert (months == M0 + 11).sum() == UNITS_PER_MONTH * FRAMES_PER_UNIT
        assert not (p.selected & (months == M0 + 11)).any()
        rule = eras.era_table(p, p.selected, campaign_last_month=M0 + 23)
        with pytest.raises(ValueError, match="100 frames lie between the dated first month"):
            eras.impose_eras(rule, p, [("2020-01", "2020-10", EVIDENCE_START),
                                       ("2020-12", "2021-10", eras.EVIDENCE_POWER)])
        # the same era started on its first populated month is accepted
        table = eras.impose_eras(rule, p, [("2020-01", "2020-10", EVIDENCE_START),
                                           ("2021-01", "2021-10", eras.EVIDENCE_POWER)])
        assert table.eras[1].first_month == M0 + 12
