"""The operating point: the frontier, the knee, and what the mask buys."""
from __future__ import annotations

import math

import pytest

from pilot_proxy.characterization import surface as operating


def _point(rho, eta_q16, f, kept, r, cost=None):
    return {"rho": rho, "eta_q16": eta_q16, "eta": eta_q16 / 65536, "masked_fraction": f, "kept": kept,
            "r_sys": r, "exposure_cost_uniform_loss": cost if cost is not None else (1.0 / (1.0 - f) if f < 1 else math.inf)}


def test_the_frontier_is_the_lower_envelope_and_drops_unsupported_points():
    points = [_point(1, 65536, 0.0, 1000, 10.0), _point(2, 66000, 0.5, 500, 1.0), _point(3, 67000, 0.5, 500, 2.0),
              _point(4, 68000, 0.8, 200, 0.9), _point(5, 69000, 0.9, 20, 0.1),        # keeps too few: not a point
              _point(6, 70000, 0.95, 50, 0.85)]
    f = operating.frontier(points)
    assert [p.masked_fraction for p in f] == [0.0, 0.5, 0.8, 0.95]
    assert [p.r_sys for p in f] == [10.0, 1.0, 0.9, 0.85]      # the 0.5 point at r = 2.0 is dominated
    assert all(p.kept >= operating.MIN_KEPT for p in f)
    assert operating.frontier([_point(1, 65536, 0.9, 5, 0.1)]) == []


def test_the_knee_is_the_corner_not_the_end():
    """A frontier that falls steeply and then flattens: the point is the corner, not the last point."""
    points = [_point(1, 65536 + i, f, 1000, r) for i, (f, r) in enumerate(
        [(0.0, 100.0), (0.2, 20.0), (0.4, 4.0), (0.5, 1.0), (0.7, 0.95), (0.9, 0.92), (0.99, 0.90)])]
    o = operating.knee(7, points)
    assert o.status == "measured" and o.point is not None
    assert 0.4 <= o.point.masked_fraction <= 0.7           # the corner, not 0.99
    assert o.r_floor == 0.90 and o.keep_everything_r == 100.0
    assert o.suppression == pytest.approx(100.0 / o.point.r_sys)
    # the margin rule chases the flat tail, which is why it is reported and not used
    assert o.sensitivity[0.05][0] >= o.point.masked_fraction
    assert o.point is o.knee


def test_a_channel_with_no_usable_point_says_so():
    o = operating.knee(9, [_point(1, 65536, 0.9, 5, 0.1)])
    assert o.status == "no frontier" and o.point is None and math.isnan(o.suppression)
    assert any("keeps" in n for n in o.notes)
    row = o.as_row()
    assert row["channel"] == 9 and row["operating_rho"] == -1 and math.isnan(row["r_floor"])


def test_rows_round_trip(tmp_path):
    points = [_point(1, 65536 + i, f, 1000, r) for i, (f, r) in enumerate([(0.0, 10.0), (0.4, 1.0), (0.9, 0.9)])]
    o = operating.knee(3, points)
    out = operating.write_knee([o], tmp_path / "op.csv")
    lines = out.read_text().splitlines()
    assert lines[0].startswith("channel,frontier_points,r_floor") and len(lines) == 2
    row = o.as_row()
    assert row["suppression_db"] == pytest.approx(10 * math.log10(o.suppression))
    assert set(row) >= set(operating.OPERATING_COLUMNS)


def _below(x, ulps):
    for _ in range(ulps):
        x = math.nextafter(x, -math.inf)
    return x


def test_rounding_ties_go_to_the_selector_order_not_the_last_digit():
    """The 2026-09-24 release's channel 33: f = 0 is reached at every rank with residuals equal up to rounding,
    and the floor plateau descends only in the last digit. Neither may choose a rank or a frontier point."""
    keep_all = {1: 14.550659032281358, 10: 14.550659032281365, 18: 14.550659032281343, 100: 14.550659032281427}
    points = [_point(rho, 5_000_000 + rho, 0.0, 1000, r) for rho, r in keep_all.items()]
    points.append(_point(1, 4_000_000, 0.0, 1000, 14.550659032281466))       # rank 1 again, a lower multiplier
    points.append(_point(3, 90_000, 0.3, 700, 12.5))
    floor = 10.933749754233736
    points += [_point(2, 70_000 + i, f, int(1000 * (1 - f)) + 40, _below(floor, 2 * i))
               for i, f in enumerate((0.6, 0.8, 0.95))]
    frontier = operating.frontier(points)
    assert [(p.masked_fraction, p.rho, p.eta_q16) for p in frontier] == [(0.0, 1, 4_000_000), (0.3, 3, 90_000), (0.6, 2, 70_000)]
    o = operating.knee(33, points)
    # margin 0.5: f = 0 lies inside, and it is rank 1 at its lowest multiplier, not rank 18 by the last digit
    assert o.sensitivity[0.5][0] == 0.0 and o.sensitivity[0.5][2:] == (1, 4_000_000)
    assert o.sensitivity[0.05][0] == 0.6 and o.sensitivity[0.05][2] == 2
    assert o.knee in frontier and o.as_row()["sensitivity_0p5_rho"] == 1
    # the exact comparison (the old rule) would have taken rank 18 and the two last-digit plateau points
    exact = min((p for p in points if p["masked_fraction"] == 0.0), key=lambda p: p["r_sys"])
    assert exact["rho"] == 18
