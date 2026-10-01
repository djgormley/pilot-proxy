"""Capture units: the power form of a class amplitude, and N, T_frame and the cap against the capture ruling's literals."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("yaml")

from pilot_proxy.capture import units  # noqa: E402
from pilot_proxy.config.project import default_project  # noqa: E402

# The capture ruling's literals (results/capture_ruling_a19_2026-09-25_r5_2/source/a18_units.py L12-15, L20)
A18_NFFT = 16384
A18_TSAMP = 2.56e-6
A18_TF = A18_NFFT * A18_TSAMP
A18_CAP = 86164.0905
A18_NB = {(0, 1): 1020, (0, 32): 896, (0, 64): 768, (0, 128): 512, (0, 255): 4, (1, 0): 768, (2, 0): 512, (3, 0): 256}


def _a18_to_power(a, cls, n_frame=A18_NFFT):
    return a * a * n_frame * A18_NB[cls] if a > 0 else a


def _a18_gate_power(gate, cls, n_frame=A18_NFFT):
    return gate * gate * n_frame * A18_NB[cls]


@pytest.mark.parametrize("a", [2.5e-4, 1.0e-6, 0.0, -3.0e-5, 7.123456789e-3])
@pytest.mark.parametrize("cls", sorted(A18_NB))
def test_power_form_equals_the_ruling_bit_for_bit(a, cls):
    for n_frame in (A18_NFFT, 2 * A18_NFFT):
        got = units.to_power(a, A18_NB[cls], n_frame)
        want = _a18_to_power(a, cls, n_frame)
        assert type(got) is type(want) and got.hex() == want.hex()
        g = units.gate_power(abs(a) + 1e-5, A18_NB[cls], n_frame)
        assert g.hex() == _a18_gate_power(abs(a) + 1e-5, cls, n_frame).hex()


def test_a_deficit_is_not_squared():
    assert units.to_power(-2.0, 512, A18_NFFT) == -2.0
    assert units.to_power(0.0, 512, A18_NFFT) == 0.0


def test_frame_length_and_cap_come_from_the_profile():
    p = default_project()
    assert units.frame_samples(p) == A18_NFFT
    assert p.integration_model.frame_seconds.hex() == A18_TF.hex()
    assert p.integration_model.coherence_cap_seconds.hex() == A18_CAP.hex()
    assert (1.0 / p.instrument.sample_rate_hz).hex() == A18_TSAMP.hex()


def _product(n_frames=5):
    keys = np.array([(0, 1, 0, 0), (0, 1, 1, 1), (0, 32, 0, 0), (0, 32, 1, 1), (1, 0, 0, 0), (1, 0, 1, 1)])
    return {"keys": keys, "count": np.array([1020, 1020, 896, 896, 768, 768]),
            "stacks": np.zeros((n_frames, len(keys)), np.complex64)}


def test_baseline_count_and_frames_are_read_from_the_product():
    z = _product(11)
    assert units.class_baseline_count(z, (0, 32)) == 896
    assert units.class_baseline_count(z, (1, 0), pol=1) == 768
    assert units.frames_per_dump(z) == 11
    with pytest.raises(KeyError):
        units.class_baseline_count(z, (3, 0))


def test_first_product_is_in_name_order(tmp_path):
    for name in ("491.npz", "477.npz", "506.npz"):
        (tmp_path / name).write_bytes(b"")
    assert units.first_product(tmp_path).endswith("477.npz")
    with pytest.raises(FileNotFoundError):
        units.first_product(tmp_path / "empty")
