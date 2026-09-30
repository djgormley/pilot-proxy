"""The admissible minimum coherence gain (capture ruling release r5.2, tests/test_a18.py test_admissible_min_G)."""
from __future__ import annotations

from pilot_proxy.characterization.coherence import admissible_min_G, gmin_models, zero_frequency_gain


def test_admissible_min_G():
    got = {n: admissible_min_G(n) for n in (11, 33, 4)}
    assert [f"{got[n]:.2f}" for n in (11, 33, 4)] == ["8.41", "21.06", "4.06"], got
    assert all(gmin_models(n)["exp"] < gmin_models(n)["block"] for n in got)


def test_zero_frequency_gain_is_capped():
    tf = 16384 * 2.56e-6; cap = 86164.0905
    assert zero_frequency_gain(15.0, cap, tf) == 15.0 / tf
    assert zero_frequency_gain(1e9, cap, tf) == cap / tf
