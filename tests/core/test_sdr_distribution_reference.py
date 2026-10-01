"""Independent signed-arithmetic and failure checks for the reference study."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2]/'tools/run_sdr_distribution_reference.py'
SPEC = importlib.util.spec_from_file_location('sdr_distribution_runner', PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_ratio_pools_powers_before_dividing():
    # A mean of row ratios would be 9; ratio of pooled powers is 18/9=2.
    z = np.array([[[3, 0], [1, 2], [0, 2]]], dtype=complex)
    powers, q = RUNNER.powers_and_ratio(z)
    np.testing.assert_array_equal(powers, [[9, 5, 4]])
    np.testing.assert_array_equal(q, [2])


def test_zero_reference_is_explicit_failure():
    with pytest.raises(ValueError, match='Zero reference'):
        RUNNER.powers_and_ratio(np.array([[[1], [0], [0]]], dtype=complex))


def signed_fixture():
    # x=2-3i; conjugated weights 1,-i,-1+i give 2-3i,-3-2i,1+5i.
    x = np.full((1, 128, 128), 0x2D, dtype=np.uint8)
    weights = np.repeat(np.array([0x10, 0x01, 0xFF], dtype=np.uint8)[:, None], 128, axis=1)
    projections = np.repeat((128*np.array([[2, -3], [-3, -2], [1, 5]]))[None, :, None, :], 128, axis=2)
    powers = 128*128**2*np.array([[13, 13, 26]], dtype=np.int64)
    return {'packed_frames': x, 'packed_weights': weights,
            'projections_i32': projections, 'term_power_sums': powers}


def test_signed_nibble_and_complex_orientation():
    assert RUNNER.packed_recount(signed_fixture()) == 384


def test_altered_projection_is_detected():
    fixture = signed_fixture()
    fixture['projections_i32'][0, 2, 3, 1] += 1
    with pytest.raises(ValueError, match='projection mismatch'):
        RUNNER.packed_recount(fixture)
