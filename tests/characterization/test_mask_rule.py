"""The handoff's mask rule, ``statistic > eta``: equality keeps, and the written counts follow it."""
from __future__ import annotations

import numpy as np

from characterization_fixtures import characterize
from pilot_proxy.characterization import oc_table, surface
from pilot_proxy.characterization.surface import ALWAYS_MASKED_Q16


def test_a_frame_is_masked_exactly_when_its_boundary_exceeds_eta():
    rng = np.random.default_rng(5)
    required = rng.integers(1, 1 << 20, size=500).astype(object)
    required[::17] = ALWAYS_MASKED_Q16
    for eta_q16 in (1, 65536, int(required[3]), (1 << 20) - 1, 2 ** 62):
        kept = surface.kept_at(required, eta_q16)
        np.testing.assert_array_equal(kept, np.array([r != ALWAYS_MASKED_Q16 and not r > eta_q16 for r in required]))
        # equality keeps
        at = required == eta_q16
        assert kept[at].all()


def test_the_always_masked_sentinel_is_never_kept():
    assert not surface.kept_at(np.array([ALWAYS_MASKED_Q16], dtype=object), 2 ** 63).any()


def test_the_written_fine_counts_follow_the_rule(tmp_path):
    summary, characterization = characterize(tmp_path, replay=None)
    assert summary["errors"] == []
    rows = [r for r in oc_table.read_rows(characterization / "oc_table.csv") if r["statistic"] == "Z_rho"
            and r["block"] == "calibration"]
    assert rows
    by_family = {}
    for r in rows:
        by_family.setdefault((r["band_id"], r["threshold_family"]), []).append(r)
    for family in by_family.values():
        family.sort(key=lambda r: int(r["eta_q16"]))
        kept = [int(r["kept"]) for r in family]
        # a larger eta masks fewer frames
        assert kept == sorted(kept)
        for r in family:
            assert int(r["kept"]) + int(r["masked"]) == int(r["frames"])
            assert float(r["masked_fraction"]) == int(r["masked"]) / int(r["frames"])
