"""Score each incumbent against the same synthetic product population."""
import numpy as np
import pytest

from pilot_proxy.characterization import flaggers as incumbent


@pytest.mark.parametrize("power,message", [(np.zeros(8), "no usable"), (np.ones(8), "degenerate")])
def test_sk_refuses_unidentifiable_null(power, message):
    with pytest.raises(ValueError, match=message):
        incumbent.calibrate_sk_null(power, np.zeros(8))
    assert incumbent.spectral_kurtosis(np.zeros(8), 1) == 1
    assert incumbent.spectral_kurtosis(np.array([1]), 1) == 1
