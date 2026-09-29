"""Legacy (pre-v5) product reading for the moved tests whose fixtures are legacy products.

``open_product`` refuses a product that does not declare the v5 schema unless
legacy reading is asked for. The modules named here build legacy fixtures (they
are moved verbatim, so their bodies do not change) and ask for it; no other test
and no run path does.
"""
from __future__ import annotations

import pytest

from pilot_proxy.products.reader import legacy_reading

LEGACY_FIXTURE_MODULES = frozenset({"test_residual_correlation"})


@pytest.fixture(autouse=True)
def _legacy_products_for_moved_fixtures(request):
    if request.module.__name__.rsplit(".", 1)[-1] in LEGACY_FIXTURE_MODULES:
        with legacy_reading():
            yield
    else:
        yield
