"""The detector interface: the narrowband-marker adapter satisfies the protocol and fills the frame table."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from test_pilotproxy_v5 import _replace, _write_product
from pilot_proxy.detectors.interface import DetectorAdapter, FrameTable, NullLaw
from pilot_proxy.detectors.narrowband_marker import NarrowbandMarkerAdapter
from pilot_proxy.detectors.narrowband_marker.frames import marker_hz
from pilot_proxy.config.project import default_project
from pilot_proxy.products.reader import Product


def test_the_adapter_satisfies_the_protocol():
    adapter = NarrowbandMarkerAdapter()
    assert isinstance(adapter, DetectorAdapter) and adapter.kind == "narrowband_marker"
    law = adapter.null_law()
    project = default_project()
    assert isinstance(law, NullLaw) and law.dof == tuple(project.detector_config.coarse_null_dof(project.instrument))
    assert law.law.mean() == pytest.approx(law.dof[1] / (law.dof[1] - 2))
    assert "shelf" in adapter.residual_estimator()


def test_the_frame_table_columns(tmp_path):
    path = _write_product(tmp_path / "552.npz", 33)
    with np.load(path, allow_pickle=False) as z:
        n = int(np.asarray(z["valid"]).shape[0])
    _replace(path, baseband_power_linear=np.full((n, 1), 4.0))
    with Product(path) as product:
        table = NarrowbandMarkerAdapter().frames(product, "33")
        assert isinstance(table, FrameTable) and table.frames == n and table.band_id == "33"
        assert {f.name for f in dataclasses.fields(FrameTable)} == {
            "band_id", "time", "unit", "statistic", "residual_db", "health", "flagged", "unit_name", "rank_scores",
            "handle"}
        np.testing.assert_array_equal(table.statistic, product.statistic)
        np.testing.assert_array_equal(table.health, product.selected)
        assert table.handle is product and table.unit_name == "Q"
        assert NarrowbandMarkerAdapter().candidate_families(table) == ("Q",)
        with pytest.raises(ValueError, match="does not match the frame axis"):
            FrameTable(band_id="33", time=table.time[:-1], unit=table.unit, statistic=table.statistic,
                       residual_db=table.residual_db, health=table.health, flagged=table.flagged)


def test_the_marker_is_the_band_edge_plus_the_template_offset():
    project = default_project()
    band = project.frequency_plan.band("33")
    assert marker_hz(band, project.interference) == project.marker_hz(band)
