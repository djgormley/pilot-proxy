"""The incumbent flaggers against the pilot proxy on one era."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.characterization import flaggers
from pilot_proxy.products.reader import Product

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("v5_fixture_flag", ROOT / "tests" / "products" / "test_pilotproxy_v5.py")
v5_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v5_fixture)


@pytest.fixture
def product(tmp_path):
    path = v5_fixture._write_product(tmp_path / "614.npz", 29)
    with np.load(path, allow_pickle=False) as z:
        frames = int(np.asarray(z["valid"]).shape[0])
    rng = np.random.default_rng(11)
    power = 4.0 + 0.1 * rng.standard_normal((frames, 1))
    power[::7] += 3.0                                   # a few loud frames for the incumbents to find
    v5_fixture._replace(path, baseband_power_linear=power)
    with Product(path) as p:
        yield p


def test_every_flagger_is_scored_on_the_same_frames(product):
    era = product.selected
    r = flaggers.compare(product, era, floor_db=-46.0, floor_evidence="stated", era_label="e", min_frames=2)
    assert r.channel == 29 and r.scored_frames > 0 and r.scored_frames <= int(era.sum())
    names = [row.name for row in r.rows]
    assert names[0] == flaggers.KEEP_EVERYTHING and flaggers.SURVEY_FLAG in names
    keep = r.by_name(flaggers.KEEP_EVERYTHING)
    assert keep.masked_fraction == 0.0 and keep.suppression_db == 0.0 and keep.kept == r.scored_frames
    for row in r.rows:
        assert 0.0 <= row.masked_fraction <= 1.0
        if row.status == "measured":
            assert row.kept == round((1.0 - row.masked_fraction) * r.scored_frames)
            # suppression is the keep-everything mean over this row's, in dB
            assert row.suppression_db == pytest.approx(10 * math.log10(keep.retained_shelf_linear / row.retained_shelf_linear))
    assert 0.0 <= r.duty_cycle <= 1.0
    assert "reported point not scored: no selection" in "; ".join(r.notes)


def test_the_floor_is_the_analysis_floor_not_a_recomputed_percentile(product):
    """A frame without a shelf estimate is booked at the floor it is given, and no frame falls below it."""
    era = product.selected
    high = flaggers.compare(product, era, floor_db=-10.0, floor_evidence="stated", era_label="e", min_frames=2)
    low = flaggers.compare(product, era, floor_db=-60.0, floor_evidence="stated", era_label="e", min_frames=2)
    assert high.by_name(flaggers.KEEP_EVERYTHING).retained_shelf_db > low.by_name(flaggers.KEEP_EVERYTHING).retained_shelf_db
    assert high.by_name(flaggers.KEEP_EVERYTHING).retained_shelf_db >= -10.0 - 1e-9


def test_a_refused_floor_drops_the_frames_that_need_one(product):
    era = product.selected
    r = flaggers.compare(product, era, floor_db=float("nan"), floor_evidence="refused", era_label="e", min_frames=2)
    if any("floor is refused" in n for n in r.notes):
        assert r.scored_frames <= int((era & np.isfinite(product.shelf_db)).sum())
    assert all(math.isfinite(row.retained_shelf_db) or row.status == "undefined" for row in r.rows)


def test_an_era_with_no_long_acquisition_scores_nothing(product):
    r = flaggers.compare(product, product.selected, floor_db=-46.0, floor_evidence="stated", era_label="e",
                         min_frames=10_000)
    assert r.scored_frames == 0 and r.rows == () and any("no acquisition" in n for n in r.notes)


def test_rows_and_the_channel_row_round_trip(product, tmp_path):
    r = flaggers.compare(product, product.selected, floor_db=-46.0, floor_evidence="stated", era_label="e", min_frames=2)
    row = flaggers.channel_row(r)
    assert row["channel"] == 29 and row["scored_frames"] == r.scored_frames
    assert {"keep_masked_fraction", "mad_suppression_db", "sk_kept", "flag_retained_shelf_db"} <= set(row)
    out = flaggers.write_flagger_rows([r], tmp_path / "flaggers.csv")
    lines = out.read_text().splitlines()
    assert lines[0] == ",".join(flaggers.FLAGGER_COLUMNS) and len(lines) == len(r.rows) + 1
