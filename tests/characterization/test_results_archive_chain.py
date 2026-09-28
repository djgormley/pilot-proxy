"""The residual chain wrapper on a synthetic v5 product, and its booking: all surviving
power at one coherence time on every band, with no variance-split credit."""
from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pilot_proxy.characterization import residual_chain as chain
from pilot_proxy.characterization import residual_chain as residual
from pilot_proxy.characterization import surface as selection
from pilot_proxy.products import reader

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("v5_fixture_chain", ROOT / "tests" / "products" / "test_pilotproxy_v5.py")
v5_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v5_fixture)

# the variance split, no longer written, and n_coh_intraday, which named it and only duplicated chain_gain
SPLIT_FIELDS = ("intraday_share", "fast_share", "ground_filter_db", "n_coh_intraday")

# The frozen chain of the author-dated release (results/archive_author_eras_2026-09-23, manifest cc36de4a,
# tables/chain.csv) on the nine bands where tau_c is usable: quality, tau_c (min), the split's intra-day and fast
# shares, n_coh(tau_c), and the gain booked with the split (phi_intra n_coh + phi_fast).
FROZEN_USABLE = {
    16: ("bounded_above", 5.0, 0.38569794786262424, 0.015300449876634233, 7152.557373046875, 2758.7420012037387),
    17: ("measured", 51.208643234579355, 0.009634863462056723, 0.0002655989965244995, 73254.55174624351, 705.7978696482216),
    18: ("measured", 6.291525778833721, 0.42930053231835447, 0.006438476141880014, 9000.099819422323, 3863.754081872471),
    24: ("measured", 140.3841853830668, 0.00302162182225789, 0.0011948551773051446, 200821.18804416677, 606.806879021187),
    29: ("bounded_above", 5.0, 0.37065692138402695, 0.020804100796839302, 7152.557373046875, 2651.165700016975),
    30: ("measured", 155.20742644599352, 0.006257452114048353, 0.0024789782075861234, 222026.0044755843, 1389.3195700576618),
    31: ("measured", 49.50841221887812, 0.00558899415616351, 0.00014746582468449276, 70822.35176879614, 395.82585762738273),
    33: ("bounded_above", 5.0, 0.06979796020492068, 0.005973668686457452, 7152.557373046875, 499.23988855602425),
    35: ("measured", 43.487216786366176, 0.008279711729687078, 0.0001254299858942642, 62208.96261172224, 515.0724028579275),
}


def _stats(channel: int, intraday: float = 0.3, fast: float = 0.01) -> residual.ShelfStatistics:
    """Shelf statistics carrying a real variance split (the split the chain must not book)."""
    return residual.ShelfStatistics(channel=channel, freq_id=1000 - channel, nu_mhz=400.0 + 6.0 * channel, n_valid=1000,
                                    n_kept=900, on_shelf_db=-20.0, floor_db=-40.0, floor_percentile=90.0,
                                    dc_fraction=0.9 - intraday - fast, interday_fraction=0.1, intraday_fraction=intraday,
                                    fast_fraction=fast, n_off_frames=10)


def _corr(channel: int, quality: str, tau_seconds: float) -> residual.CorrelationTime:
    tau = tau_seconds if quality != "refused" else math.nan
    return residual.CorrelationTime(channel=channel, tau_c=tau, tau_lo=tau, tau_hi=tau, plateau_fraction=0.0, n_days=10,
                                    n_pairs=100, trim_spread=1.0, surviving_spread=1.0, quality=quality)


def _chain(monkeypatch, stats, corr) -> chain.ChainResult:
    monkeypatch.setattr(residual, "shelf_statistics", lambda *a, **k: stats)
    monkeypatch.setattr(residual, "correlation_time", lambda *a, **k: corr)
    return chain.chain("unused.npz")


def test_chain_assembles_rfisher_results_and_books_the_gain(tmp_path):
    path = v5_fixture._write_product(tmp_path / "552.npz", 33)
    result = chain.chain(path)
    assert result.channel == 33 and result.population.startswith("transmitter-on frames of the whole archive")
    assert result.n_valid > 0 and result.tau_quality in ("measured", "bounded_above", "refused")
    assert result.components and all(share >= 0 and n_coh >= 1 for share, n_coh in result.components)
    assert result.gain == sum(s * n for s, n in result.components)
    # one component on every outcome: all surviving power at the booked tau (the cap when refused), no split credit
    tau = result.tau_c_seconds if result.tau_quality != "refused" else residual.MAX_TAU_C_SECONDS
    assert result.components == ((1.0, residual.coherent_frames(tau)),)
    row = result.as_row()
    assert not set(SPLIT_FIELDS) & set(row)
    assert row["chain_gain"] == result.gain and row["tau_outcome"] in ("measured", "bound", "refused (cap)")
    assert math.isfinite(row["chain_gain"]) and row["chain_gain"] >= 1.0
    assert "n_coh_intraday" not in {f.name for f in dataclasses.fields(chain.ChainResult)}
    # an off epoch changes the recorded population
    off = chain.chain(path, off_through="2020-01")
    assert "through 2020-01" in off.population
    assert chain.FRAME_SECONDS == reader.FRAME_SECONDS
    assert np.isfinite(off.n_valid)


def test_masked_valid_and_invalidation_follow_the_product_contract():
    import numpy as np
    from pilot_proxy.characterization import residual_chain as chain
    column = np.ones((6, 1), dtype=np.uint8)
    out = chain.masked_valid(column, np.array([1, 0, 1, 1, 0, 1], dtype=bool))
    assert out.shape == (6, 1) and out.dtype == np.uint8 and out.reshape(-1).tolist() == [1, 0, 1, 1, 0, 1]
    arrays = {"valid": column, "p_ref_sum_u64": np.full((6, 1), 7, dtype=np.uint64), "p_ref_lower_u64": np.full((6, 1), 3, dtype=np.uint64),
              "p_ref_upper_u64": np.full((6, 1), 4, dtype=np.uint64), "reject_mask": np.ones((6, 1), dtype=np.uint8),
              "coarse_power_ratio": np.ones((6, 1)), "estimated_data_shelf_snr_db": np.zeros((6, 1)), "other": np.arange(6)}
    inv = chain.invalidate_frames(arrays, np.array([1, 0, 1, 1, 0, 1], dtype=bool))
    assert inv["valid"].reshape(-1).tolist() == [1, 0, 1, 1, 0, 1]
    assert inv["p_ref_sum_u64"].reshape(-1).tolist() == [7, 0, 7, 7, 0, 7] and inv["p_ref_lower_u64"].reshape(-1)[1] == 0
    assert inv["reject_mask"].reshape(-1).tolist() == [1, 0, 1, 1, 0, 1]
    assert np.isnan(inv["coarse_power_ratio"].reshape(-1)[[1, 4]]).all() and inv["coarse_power_ratio"].reshape(-1)[0] == 1.0
    assert inv["other"].tolist() == list(range(6)) and arrays["p_ref_sum_u64"].reshape(-1).tolist() == [7] * 6   # inputs untouched


@pytest.mark.parametrize("quality", ("measured", "bounded_above", "refused"))
def test_every_band_books_one_component_and_no_split(monkeypatch, quality):
    for channel in range(14, 37):
        stats, corr = _stats(channel), _corr(channel, quality, 1800.0)
        result = _chain(monkeypatch, stats, corr)
        n_coh = residual.coherent_frames(corr.tau_for_budget)
        assert result.components == ((1.0, n_coh),) and result.gain == n_coh
        assert chain.booked_components(corr) == result.components
        row = result.as_row()
        assert row["chain_gain"] == n_coh and not set(SPLIT_FIELDS) & set(row)
        split = residual.surviving_components(stats, corr)
        if corr.is_usable:
            # the split booking took a ground-filter credit; the chain no longer does
            assert len(split) == 2 and sum(f * n for f, n in split) < result.gain
        else:
            assert result.components == split          # a refusal already booked no split: unchanged


@pytest.mark.parametrize("channel", sorted(FROZEN_USABLE))
def test_r_sys_is_the_old_no_split_residual_on_the_nine_usable_bands(monkeypatch, channel):
    quality, tau_minutes, intraday, fast, n_coh_old, gain_old = FROZEN_USABLE[channel]
    result = _chain(monkeypatch, _stats(channel, intraday, fast), _corr(channel, quality, tau_minutes * 60.0))
    # the frozen gain is the split booking, and n_coh(tau_c) is the no-split gain
    assert intraday * n_coh_old + fast == pytest.approx(gain_old, rel=1e-12)
    assert len(result.components) == 1 and result.components[0][0] == 1.0
    assert result.gain == pytest.approx(n_coh_old, rel=1e-12) and result.gain > gain_old
    # r_sys is the kept-frame mean of the floor-bounded shelf times G, so it is the old r_sys_no_split
    product = SimpleNamespace(shelf_db=np.array([-30.0, -25.0, np.nan, -45.0, -20.0]))
    floor, rows = selection.Floor(db=-40.0, evidence="stated", population="test"), np.arange(5)
    level = chain.frame_residuals(product, rows, floor, gain=1.0).mean()
    r_sys = chain.frame_residuals(product, rows, floor, gain=result.gain).mean()
    assert r_sys == pytest.approx(level * n_coh_old, rel=1e-12)                    # old r_sys_no_split
    assert r_sys / (level * gain_old) == pytest.approx(n_coh_old / gain_old, rel=1e-12)   # against the old r_sys
