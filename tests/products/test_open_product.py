"""The product view: one coordinate system for current and legacy products, read through ``open_product``."""
from __future__ import annotations

import numpy as np
import pytest

from test_pilotproxy_v5 import _write_product
from pilot_proxy.products.npzio import load_npz
from pilot_proxy.products.reader import (
    PRODUCT_SCHEMA_TOKEN, Product, ProductContractError, ProductView, is_current_product, open_product,
)


def test_a_current_product_opens_with_its_frames_and_decisions(tmp_path):
    path = _write_product(tmp_path / "552.npz", 33)
    values = load_npz(path)
    view = open_product(values)
    assert isinstance(view, ProductView) and view.is_current and view.schema == PRODUCT_SCHEMA_TOKEN
    assert is_current_product(values)
    n = view.valid.size
    for column in (view.rejected, view.shelf_db, view.statistic, view.frame_unit_index):
        assert column.shape == (n,)
    # at eta = 1 the exact integer decision is the stored survey flag
    np.testing.assert_array_equal(view.rejected_at_multiplier(1.0), view.rejected)
    # a larger multiplier never rejects more
    assert not (view.rejected_at_multiplier(1.5) & ~view.rejected_at_multiplier(1.0)).any()
    with Product(path) as product:
        assert product.geometry.physical_channel == view.physical_channel == 33
        np.testing.assert_array_equal(product.rejected, view.rejected)


def test_the_view_refuses_what_the_contract_does_not_admit(tmp_path):
    values = dict(load_npz(_write_product(tmp_path / "552.npz", 33)))
    with pytest.raises(TypeError):
        open_product(values).rejected_at_multiplier(True)
    with pytest.raises(ValueError):
        open_product(values).rejected_at_multiplier(0.0)
    del values["p_target_u64"]
    with pytest.raises(ProductContractError):
        open_product(values)
