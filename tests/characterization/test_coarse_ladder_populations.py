"""The all-valid-frames ladder a capture dump is placed on, against the capture scripts' own reading."""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.characterization import coarse_ladder


def _product(path, seed=5, n=400):
    rng = np.random.default_rng(seed)
    valid = rng.random((n, 1)) > 0.1
    q = rng.gamma(20.0, 0.05, (n, 1)); q[3, 0] = np.nan
    np.savez(path, valid=valid, target_norm_sq=np.array([0.75]), reference_norm_sum_sq=np.array([3.0]),
             coarse_power_ratio=q)
    return path


def _ladder_place_reading(path, qs=(0.1, 0.5, 0.9)):
    # ladder_place.py (053db723) L96-100 and ruling_a18.py L244-250, as written there
    z = np.load(path, allow_pickle=True); v = z["valid"][:, 0].astype(bool)
    mu0 = 2.0 * float(z["target_norm_sq"][0]) / float(z["reference_norm_sum_sq"][0])
    Q = z["coarse_power_ratio"][:, 0].astype(float) / mu0; Q = Q[v & np.isfinite(Q)]
    return {q: float(np.quantile(Q, q, method="higher")) for q in qs}, int(Q.size), float(np.median(Q))


def test_all_valid_frames_thresholds_are_the_capture_readings(tmp_path):
    p = _product(tmp_path / "552.npz")
    got = coarse_ladder.product_thresholds(p)
    want, n, med = _ladder_place_reading(p)
    assert {q: got[q] for q in coarse_ladder.QUANTILES} == want
    assert got["n"] == n and got["median"] == med
    assert coarse_ladder.POPULATIONS == ("current_era_calibration_block", "all_valid_frames")


def test_a_rescan_reads_its_own_null_power_ratio(tmp_path):
    rng = np.random.default_rng(9)
    q = rng.gamma(20.0, 0.05, (300, 1)); valid = np.ones((300, 1), bool)
    np.savez(tmp_path / "chime_detector_outputs.npz", valid=valid, coarse_power_ratio=q * 0.5,
             null_power_ratio=np.array([0.5]))
    got = coarse_ladder.run_thresholds(tmp_path / "chime_detector_outputs.npz")
    Q = (q[:, 0] * 0.5) / 0.5
    assert got[0.5] == float(np.quantile(Q, 0.5, method="higher"))


def test_dumps_are_placed_by_their_median_and_kept_fraction(tmp_path):
    rng = np.random.default_rng(11)
    F = rng.gamma(20.0, 0.05, (33, 3)); valid = np.ones((33, 3), bool); valid[0, 1] = False
    run = tmp_path / "chime_detector_outputs.npz"
    np.savez(run, physical_channel=np.array([17, 31, 35]), coarse_power_ratio=F, null_power_ratio=np.array([1.0, 0.9, 1.1]),
             valid=valid)
    eta = {17: {0.1: 0.8, 0.5: 1.0, 0.9: 1.2, "n": 10, "median": 1.0}}
    placed = coarse_ladder.place_dumps(run, eta)
    for j, c in enumerate((17, 31, 35)):
        Q = F[:, j][valid[:, j]] / [1.0, 0.9, 1.1][j]
        assert placed[c]["q_median"] == float(np.median(Q))
    Q = F[:, 0] / 1.0
    assert placed[17]["kept_fraction"] == {q: float(np.mean(Q <= eta[17][q])) for q in (0.1, 0.5, 0.9)}
    assert "kept_fraction" not in placed[31]


def test_a_rescan_is_read_on_its_band_column(tmp_path):
    rng = np.random.default_rng(13)
    F = rng.gamma(20.0, 0.05, (200, 2)); valid = np.ones((200, 2), bool)
    run = tmp_path / "chime_detector_outputs.npz"
    np.savez(run, physical_channel=np.array([37, 33]), coarse_power_ratio=F, null_power_ratio=np.array([1.0, 0.5]),
             valid=valid)
    got = coarse_ladder.run_thresholds(run, band=33)
    assert got[0.5] == float(np.quantile(F[:, 1] / 0.5, 0.5, method="higher")) and got["n"] == 200
    assert coarse_ladder.run_thresholds(run)[0.5] == float(np.quantile(F[:, 0], 0.5, method="higher"))
    with pytest.raises(ValueError, match="band 31"):
        coarse_ladder.run_thresholds(run, band=31)
