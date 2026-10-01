"""The capture's feed geometry comes from the instrument file, and its ten baseline classes from one list."""
from __future__ import annotations

import pytest

pytest.importorskip("yaml")

from pilot_proxy.capture import class_excess, control_level, oc_table, units  # noqa: E402
from pilot_proxy.capture.diagnostics import bearing  # noqa: E402
from pilot_proxy.config.instrument import FeedLayout, load_instrument  # noqa: E402


def test_chime_feed_layout_is_read_from_the_instrument_file():
    fl = load_instrument("chime").feed_layout
    assert fl == FeedLayout(ew_spacing_m=22.0, ns_spacing_m=0.3048, inputs_per_cylinder=512, positions_per_cylinder=256)
    assert type(fl.ew_spacing_m) is float and fl.ew_spacing_m.hex() == (22.0).hex()
    assert type(fl.ns_spacing_m) is float and fl.ns_spacing_m.hex() == (0.3048).hex()
    assert type(fl.inputs_per_cylinder) is int and type(fl.positions_per_cylinder) is int


@pytest.mark.parametrize("name", ["gbo", "hco", "kko"])
def test_outriggers_record_no_feed_layout(name):
    assert load_instrument(name).feed_layout is None


def test_bearing_layout_is_the_old_index_arithmetic():
    import numpy as np

    cyl, pos, pol = bearing.layout(2048)
    i = np.arange(2048)
    assert np.array_equal(cyl, (i // 512).astype(float)) and np.array_equal(pos, (i % 256).astype(float))
    assert np.array_equal(pol, (i // 256) % 2)


def test_feed_layout_is_refused_when_not_positive(tmp_path):
    import shutil

    from pilot_proxy.config import instrument

    shutil.copy(instrument._INSTRUMENT_DIR + "/chime.yaml", tmp_path / "chime.yaml")
    (tmp_path / "x.yaml").write_text("name: x\nextends: chime\nfeed_layout: {ew_spacing_m: -1.0}\n")
    with pytest.raises(ValueError, match="feed_layout.ew_spacing_m"):
        load_instrument("x", directory=str(tmp_path))
    (tmp_path / "y.yaml").write_text("name: y\nextends: chime\nfeed_layout: {ew_spacing_m: 22.0, pitch_m: 1.0}\n")
    with pytest.raises(ValueError, match="feed_layout must be null or a mapping"):
        load_instrument("y", directory=str(tmp_path))


def test_the_three_capture_tables_read_one_ten_class_list_in_the_old_order():
    old = [(0, 1), (0, 8), (0, 32), (0, 64), (0, 128), (0, 255), (1, 0), (1, 32), (2, 0), (3, 0)]
    assert list(units.TEN_CLASSES) == old
    assert class_excess.CLASSES == old and control_level.CLASSES == old
    assert oc_table.CLASSES is units.TEN_CLASSES
