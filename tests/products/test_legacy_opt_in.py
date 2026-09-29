"""Review M6: a pre-v5 (legacy) product is read only on request; production callers refuse it.

The legacy mapping here carries no shelf column, so the legacy view itself refuses it once reached: the two
refusals have different messages, which shows which path a call took.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.characterization import flaggers
from pilot_proxy.detectors.narrowband_marker import shelf
from pilot_proxy.products.reader import ProductContractError, legacy_reading, open_product

REFUSED = "legacy products are read only on request"
REACHED = "neither current v5 nor a supported legacy product"


def _legacy(n=40):
    return {"valid": np.ones((n, 1), dtype=np.uint8), "reject_mask": np.zeros((n, 1), dtype=np.uint8),
            "physical_channel": np.array([35], dtype=np.int32), "freq_id": np.array([521], dtype=np.int64)}


def test_a_legacy_product_is_refused_by_default():
    with pytest.raises(ProductContractError, match=REFUSED):
        open_product(_legacy())


def test_a_legacy_product_reaches_the_legacy_view_when_asked():
    with pytest.raises(ProductContractError, match=REACHED):
        open_product(_legacy(), allow_legacy=True)
    with legacy_reading(), pytest.raises(ProductContractError, match=REACHED):
        open_product(_legacy())
    with pytest.raises(ProductContractError, match=REFUSED):
        open_product(_legacy())                 # the opt-in ends with the block


def test_an_explicit_refusal_wins_inside_the_block():
    with legacy_reading(), pytest.raises(ProductContractError, match=REFUSED):
        open_product(_legacy(), allow_legacy=False)


def test_production_callers_refuse_a_legacy_product(tmp_path):
    path = tmp_path / "521.npz"
    np.savez(path, **_legacy())
    with pytest.raises(ProductContractError, match=REFUSED):
        shelf.shelf_statistics(path)
    with pytest.raises(ProductContractError, match=REFUSED):
        flaggers.shelf_per_frame(dict(np.load(path)))
