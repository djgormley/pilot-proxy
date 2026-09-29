"""Review M6: a pre-v5 (legacy) product is read only on request; production callers refuse it."""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.characterization import flaggers
from pilot_proxy.detectors.narrowband_marker import shelf
from pilot_proxy.products.reader import ProductContractError, legacy_reading, open_product


def _legacy(n=40):
    rng = np.random.default_rng(2)
    return {"valid": np.ones((n, 1), dtype=np.uint8),
            "reject_mask": (rng.random((n, 1)) > 0.5).astype(np.uint8),
            "snr_shelf_db": rng.normal(-20.0, 1.0, (n, 1)),
            "physical_channel": np.array([35], dtype=np.int32), "freq_id": np.array([521], dtype=np.int64)}


def test_a_legacy_product_is_refused_by_default():
    with pytest.raises(ProductContractError, match="legacy products are read only on request"):
        open_product(_legacy())


def test_a_legacy_product_opens_when_asked():
    assert open_product(_legacy(), allow_legacy=True).schema == "legacy"
    with legacy_reading():
        assert open_product(_legacy()).schema == "legacy"
    with pytest.raises(ProductContractError):
        open_product(_legacy())                 # the opt-in ends with the block


def test_an_explicit_refusal_wins_inside_the_block():
    with legacy_reading(), pytest.raises(ProductContractError):
        open_product(_legacy(), allow_legacy=False)


def test_production_callers_refuse_a_legacy_product(tmp_path):
    path = tmp_path / "521.npz"
    np.savez(path, **_legacy())
    with pytest.raises(ProductContractError):
        shelf.shelf_statistics(path)
    with pytest.raises(ProductContractError):
        flaggers.shelf_per_frame(dict(np.load(path)))
