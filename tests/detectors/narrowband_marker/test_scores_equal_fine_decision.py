"""The characterization's exact rank scores are the deployed fine decision's boundaries, rank by rank."""
from __future__ import annotations

import numpy as np
import pytest

from test_pilotproxy_v5 import _write_product
from pilot_proxy.detectors.narrowband_marker.scores import required_eta_q16_by_rank
from pilot_proxy.fine_decision import fine_required_multiplier_q16
from pilot_proxy.products.npzio import load_npz

FINE_BINS = 256


def _bulk(designated):
    bulk = np.zeros(FINE_BINS, dtype=bool)
    bulk[::3] = True
    bulk[list(designated)] = False
    return bulk


@pytest.mark.parametrize("seed", range(6))
def test_random_frames_agree_at_every_rank(seed):
    rng = np.random.default_rng(seed)
    powers = rng.integers(0, 50, size=(3, FINE_BINS)).astype(np.uint64)
    if seed % 2:
        powers[0, 10] = 10_000                     # a strong designated bin
    anchor, width = 10, 2
    designated = [(anchor + k) % FINE_BINS for k in range(-width, width + 1)]
    bulk = _bulk(designated)
    boundaries = required_eta_q16_by_rank(powers, anchor_bin=anchor, designated_half_width=width, bulk_mask=bulk)
    for rho, boundary in enumerate(boundaries, start=1):
        deployed = fine_required_multiplier_q16(powers, anchor_bin=anchor, designated_half_width=width,
                                                bulk_mask=bulk, cfar_rank=rho - 1)
        assert deployed.valid and deployed.multiplier_q16 == boundary, rho


def test_the_fixture_products_agree(tmp_path):
    values = load_npz(_write_product(tmp_path / "552.npz", 33))
    fine = values["fine_power_u64"]
    anchor, width = 128, 2
    designated = [(anchor + k) % FINE_BINS for k in range(-width, width + 1)]
    bulk = _bulk(designated)
    for frame in range(0, fine.shape[0], 7):
        powers = np.ascontiguousarray(fine[frame])
        boundaries = required_eta_q16_by_rank(powers, anchor_bin=anchor, designated_half_width=width, bulk_mask=bulk)
        for rho in (1, len(boundaries) // 2, len(boundaries)):
            deployed = fine_required_multiplier_q16(powers, anchor_bin=anchor, designated_half_width=width,
                                                    bulk_mask=bulk, cfar_rank=rho - 1)
            assert deployed.multiplier_q16 == boundaries[rho - 1]
