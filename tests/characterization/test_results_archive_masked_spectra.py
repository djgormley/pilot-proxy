"""The before-and-after spectra of a mask, on the block it was not chosen on."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.characterization import masked_spectra
from pilot_proxy.products.reader import NFFT, Product

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("v5_fixture_ms", ROOT / "tests" / "products" / "test_pilotproxy_v5.py")
v5_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v5_fixture)


@pytest.fixture
def product(tmp_path):
    """A product whose loud frames carry an extra line, so a mask has something to remove."""
    path = v5_fixture._write_product(tmp_path / "614.npz", 29)
    with np.load(path, allow_pickle=False) as z:
        frames = int(np.asarray(z["valid"]).shape[0])
    v5_fixture._replace(path, baseband_power_linear=np.full((frames, 1), 4.0))
    with Product(path) as p:
        g = p.geometry
    spectrum = np.ones((frames, NFFT))
    loud = np.zeros(frames, dtype=bool)
    loud[::2] = True
    bin_at_pilot = int(round(float(g.psd_bin_of_rf_offset(0.0)))) % NFFT
    spectrum[loud, bin_at_pilot] = 101.0                     # the loud frames carry 100x at the pilot
    codes = np.round(1000.0 * np.log10(spectrum)).astype(np.int16)
    v5_fixture._replace(path, psd_frame_db_i16=codes,
                        psd_db_reference=np.ones((frames, 1), dtype=np.float64),
                        psd_db_step_per_code=np.asarray(0.01, dtype=np.float64),
                        psd_db_invalid_code=np.asarray(-32768, dtype=np.int16))
    with Product(path) as p:
        yield p, loud, bin_at_pilot


def test_the_removed_contribution_is_on_a_common_denominator(product):
    p, loud, pilot_bin = product
    block = p.selected
    kept = block & ~loud                                     # the mask removes exactly the loud frames
    m = masked_spectra.measure(p, block, kept, basis="test", rho=1, eta_q16=65536, eta=1.0)
    assert m.frames == int(block.sum()) and m.kept == int(kept.sum())
    assert m.masked_fraction == pytest.approx(1.0 - m.kept / m.frames)
    # P_all - P_keep_contrib is what the kept frames do not contribute, on the same denominator
    assert np.all(m.p_all + 1e-9 >= m.p_keep_contrib)
    assert m.removed_window > 0 and m.removed_window <= m.all_window
    # the kept frames are the quiet ones, so P_keep_given is flat and the pilot is suppressed
    assert m.pilot_suppression_db > 10.0
    # and the two denominators differ: the contribution is not the conditional mean
    assert not np.allclose(m.p_keep_contrib, m.p_keep_given)


def test_a_mask_that_keeps_everything_removes_nothing(product):
    p, _, _ = product
    m = masked_spectra.measure(p, p.selected, p.selected, basis="keep all", rho=1, eta_q16=65536, eta=1.0)
    assert m.masked_fraction == pytest.approx(0.0)
    assert m.removed_window == pytest.approx(0.0, abs=1e-9)
    assert m.pilot_suppression_db == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(m.p_keep_contrib, m.p_keep_given)


def test_a_mask_that_keeps_nothing_is_reported_not_divided_by_zero(product):
    p, _, _ = product
    m = masked_spectra.measure(p, p.selected, np.zeros(p.n_frames, dtype=bool), basis="keep none", rho=1,
                               eta_q16=65536, eta=1.0)
    assert m.kept == 0 and m.masked_fraction == pytest.approx(1.0)
    assert math.isnan(m.pilot_suppression_db) and any("keeps no frame" in n for n in m.notes)
    assert m.removed_window == pytest.approx(m.all_window, rel=1e-9)


def test_an_empty_block_is_reported(product):
    p, _, _ = product
    m = masked_spectra.measure(p, np.zeros(p.n_frames, dtype=bool), np.zeros(p.n_frames, dtype=bool),
                               basis="none", rho=1, eta_q16=65536, eta=1.0)
    assert m.frames == 0 and any("empty" in n for n in m.notes)


def test_rows_and_arrays_round_trip(product, tmp_path):
    p, loud, _ = product
    m = masked_spectra.measure(p, p.selected, p.selected & ~loud, basis="operating point", rho=2, eta_q16=70000,
                               eta=70000 / 65536)
    out = masked_spectra.write_spectra_rows([m], tmp_path / "s.csv")
    assert out.read_text().splitlines()[0] == ",".join(masked_spectra.SPECTRA_COLUMNS)
    npz = masked_spectra.write_spectra_npz([m], tmp_path / "s.npz")
    with np.load(npz) as z:
        assert "ch29_operating_point_p_all" in z and z["ch29_operating_point_rf_offset_hz"].size == m.p_all.size
