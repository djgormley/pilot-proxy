"""``ties``: least-residual points with rounding ties broken by the selector's order."""
from __future__ import annotations

import math

from pilot_proxy.characterization import surface as ties


def _down(x: float, ulps: int) -> float:
    for _ in range(ulps):
        x = math.nextafter(x, -math.inf)
    return x


def test_rounding_is_a_tie_and_a_real_difference_is_not():
    r = 163.4505217626282
    assert ties.tied(r, 163.45052176262828) and ties.tied(r, _down(r, 50))
    assert not ties.tied(r, r * (1 + 1e-6)) and not ties.tied(1.0, math.inf)
    assert ties.tied(0.0, 0.0) and not ties.tied(0.0, 1e-300)
    assert ties.TIE_REL_TOL == 1e-9


def test_least_breaks_ties_by_the_key_not_the_last_digit():
    rows = [("a", 0.97, 5.0), ("b", 0.65, 5.0 + 1e-14), ("c", 0.99, _down(5.0, 3)), ("d", 0.10, 5.1)]
    best = ties.least(rows, lambda r: r[2], lambda r: r[1])
    assert best[0] == "b"                                    # the least mask among the tied, not the exact minimum "c"
    assert ties.least(rows, lambda r: r[2], lambda r: r[1], rel_tol=0.0)[0] == "c"
    assert ties.least([], lambda r: r, lambda r: r) is None
    # an untied least residual wins whatever its key
    assert ties.least([("x", 0.9, 1.0), ("y", 0.1, 2.0)], lambda r: r[2], lambda r: r[1])[0] == "x"
