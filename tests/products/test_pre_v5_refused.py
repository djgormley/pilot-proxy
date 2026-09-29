"""Review M6 and T13 review H1: a product that does not declare the v5 schema is refused, by the reader and by
every production caller."""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.characterization import flaggers
from pilot_proxy.detectors.narrowband_marker import shelf
from pilot_proxy.products.reader import ProductContractError, coarse_reject_mask, open_product

REFUSED = "pre-v5 products are not read"


def _pre_v5(n=40):
    return {"valid": np.ones((n, 1), dtype=np.uint8), "reject_mask": np.zeros((n, 1), dtype=np.uint8),
            "physical_channel": np.array([35], dtype=np.int32), "freq_id": np.array([521], dtype=np.int64)}


def test_a_pre_v5_product_is_refused():
    with pytest.raises(ProductContractError, match=REFUSED):
        open_product(_pre_v5())
    with pytest.raises(ProductContractError, match=REFUSED):
        coarse_reject_mask(_pre_v5())


def test_production_callers_refuse_a_pre_v5_product(tmp_path):
    path = tmp_path / "521.npz"
    np.savez(path, **_pre_v5())
    with pytest.raises(ProductContractError, match=REFUSED):
        shelf.shelf_statistics(path)
    with pytest.raises(ProductContractError, match=REFUSED):
        flaggers.shelf_per_frame(dict(np.load(path)))
