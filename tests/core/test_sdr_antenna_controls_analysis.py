import importlib.util
from pathlib import Path
import sys

import numpy as np

TOOLS = Path(__file__).resolve().parents[2]/'tools'
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location('antenna_analysis', TOOLS/'analyze_sdr_antenna_controls.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_power_is_pooled_before_ratio_and_zero_reference_is_explicit():
    z = np.array([[[1, 2], [1, 0], [0, 2]],
                  [[0, 0], [0, 0], [0, 0]],
                  [[1, 0], [0, 0], [0, 0]]], dtype=complex)
    powers, ratio = MODULE.powers_ratio(z)
    np.testing.assert_array_equal(powers, [[5, 1, 4], [0, 0, 0], [1, 0, 0]])
    assert ratio[0] == 2
    assert np.isnan(ratio[1])
    assert np.isinf(ratio[2])


def test_nonfinite_observations_remain_in_denominator_accounting():
    report = MODULE.moments([1, 3, np.inf, np.nan])
    assert report['total_count'] == 4
    assert report['finite_count'] == 2
    assert report['undefined_count'] == 1
    assert report['infinite_count'] == 1
    assert report['finite_mean'] == 2
    assert MODULE.moments([np.nan])['finite_median'] is None
