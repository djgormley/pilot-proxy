"""Selection on a calibration block and replay on an evaluation block (synthetic v5 fixture)."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.characterization import stability
from pilot_proxy.characterization import surface as selection
from pilot_proxy.characterization.residual_chain import frame_residuals
from pilot_proxy.characterization.surface import ALWAYS_MASKED_Q16
from pilot_proxy.products.reader import Product

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("v5_fixture_sel", ROOT / "tests" / "products" / "test_pilotproxy_v5.py")
v5_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v5_fixture)

FLOOR = selection.Floor(db=-55.0, evidence="stated", population="fixture")


@pytest.fixture
def product(tmp_path):
    path = v5_fixture._write_product(tmp_path / "552.npz", 33)
    with np.load(path, allow_pickle=False) as z:
        frames = int(np.asarray(z["valid"]).shape[0])
    v5_fixture._replace(path, baseband_power_linear=np.full((frames, 1), 4.0))
    with Product(path) as p:
        yield p


def _bulk():
    bulk = np.zeros(256, dtype=bool)
    bulk[:4] = True
    return bulk


def test_kept_at_follows_the_q16_rule():
    req = np.array([1, 65536, 70000, ALWAYS_MASKED_Q16], dtype=object)
    np.testing.assert_array_equal(selection.kept_at(req, 65536), [True, True, False, False])
    np.testing.assert_array_equal(selection.kept_at(req, 2**63), [True, True, True, False])


def test_systematic_residuals_use_shelf_where_finite_and_the_floor_elsewhere(product):
    rows = np.arange(product.n_frames)
    res = frame_residuals(product, rows, FLOOR)
    finite = np.isfinite(product.shelf_db)
    assert np.allclose(res[finite], 10 ** (product.shelf_db[finite] / 10))
    assert np.allclose(res[~finite], FLOOR.linear)
    assert np.allclose(frame_residuals(product, rows, FLOOR, gain=566.0), 566.0 * res)
    with pytest.raises(ValueError, match="finite floor"):
        frame_residuals(product, rows, selection.Floor(math.nan, "refused", "none"))
    with pytest.raises(ValueError, match="chain gain"):
        frame_residuals(product, rows, FLOOR, gain=0.0)


def test_an_undefined_floor_bounds_nothing_and_frames_without_a_shelf_still_need_one(product):
    import numpy as np
    from pilot_proxy.characterization import surface as sel_mod
    rows = np.flatnonzero(np.isfinite(product.shelf_db))[:5]
    r = frame_residuals(product, rows, sel_mod.Floor(float("nan"), "refused", "none"), 2.0)
    assert np.allclose(r, 2.0 * 10.0 ** (product.shelf_db[rows] / 10.0))
    bounded = frame_residuals(product, rows, sel_mod.Floor(0.0, "stated", "x"), 1.0)     # 0 dB = 1.0 linear
    assert (bounded >= 1.0).all()
    without = np.flatnonzero(~np.isfinite(product.shelf_db))[:3]
    if without.size:
        with pytest.raises(ValueError, match="finite floor"):
            frame_residuals(product, without, sel_mod.Floor(float("nan"), "refused", "none"))


def test_drift_diagnostic_names_the_refusing_candidate_and_the_drift_a_point_would_see(product):
    """The screen refuses on the first candidate to cross the pooled floor; the diagnostic says so and
    measures the early/late ratios where a real operating point lives."""
    import numpy as np
    from pilot_proxy.detectors.narrowband_marker.scores import build_score_bundle as build_residual_score_bundle

    era = product.selected & np.isfinite(product.frame_time)
    bulk = np.zeros(256, dtype=bool)
    bulk[::2] = True
    for k in range(-2, 3):
        bulk[(128 + k) % 256] = False
    bundle = build_residual_score_bundle(product.path, era, anchor_bin=128, designated_half_width=2, bulk_mask=bulk)
    residuals = frame_residuals(product, bundle.source_row_index, FLOOR, 1.0)
    d = stability.drift_diagnostic(bundle, residuals, product.frame_time[bundle.source_row_index])
    assert d["drift_status"] in ("measured", "no evaluable candidate")
    assert d["drift_early_frames"] > 0 and d["drift_late_frames"] > 0
    if "drift_refused_rho" in d:
        # the refusing candidate keeps at least the pooled floor and less than the per-half floor in one half
        assert d["drift_refused_early_kept"] + d["drift_refused_late_kept"] >= 30
        assert min(d["drift_refused_early_kept"], d["drift_refused_late_kept"]) < stability.PROVISIONAL_MIN_HALF_RETAINED
        assert 0.0 < d["drift_refused_kept_fraction"] <= 1.0
    if d["drift_status"] == "measured":
        assert d["drift_candidates_at_0"] >= d.get("drift_candidates_at_0p2", 0)
        assert d["drift_max_cost_ratio_at_0"] >= 1.0 and d["drift_max_systematic_ratio_at_0"] >= 1.0
    # an empty or untimed block is reported, not raised
    assert stability.drift_diagnostic(bundle, residuals, np.full(bundle.frame_count, np.nan))["drift_status"] == "no timed frames"
