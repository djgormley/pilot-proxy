"""Synthetic capture products for the capture tests: a few bands of reduced dump files and a detector run.

Every file has the reduced-dump layout (``meta``, ``keys``, ``count``,
``stacks``, ``autos``); the detector run has the kernel outputs the capture
modules read. The values are seeded noise plus a coherent part per band, so
the estimators have something to measure; nothing here is a real capture.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASSES = [(0, 1), (0, 2), (0, 4), (0, 8), (0, 32), (0, 64), (0, 128), (0, 255), (1, 0), (1, 1), (1, 32), (2, 0),
           (3, 0)]
COUNTS = {(0, 1): 1020, (0, 2): 1016, (0, 4): 1008, (0, 8): 992, (0, 32): 896, (0, 64): 768, (0, 128): 512,
          (0, 255): 4, (1, 0): 768, (1, 1): 762, (1, 32): 672, (2, 0): 512, (3, 0): 256}
W_MHZ = 0.390625
# band -> (marker freq_id, the bins written); 37 is the control band, read at 491
BANDS = {14: (844, [839, 840, 841, 842, 843, 844]), 15: (829, [824, 825, 826, 827, 828, 829]),
         16: (813, [808, 809, 810, 811, 812, 813]), 37: (491, [486, 487, 488, 489, 490, 491])}
N_INPUTS = 64
NFFT = 16384


def keys():
    return np.array([(ew, ns, p, p) for (ew, ns) in CLASSES for p in (0, 1)], dtype=np.int64)


def write_dump(directory, *, seed, n_frames=11, t0=1.7e9, strength=None, dead=(3, 17), same_pols=False):
    """One dump: a product per bin of BANDS. ``strength`` maps band -> coherent amplitude (default small);
    ``same_pols`` writes the YY stacks equal to the XX stacks."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    strength = strength or {14: 2e-4, 15: 5e-5, 16: 1e-3, 37: 0.0}
    k = keys(); count = np.array([COUNTS[(int(r[0]), int(r[1]))] for r in k])
    for band, (marker, fids) in BANDS.items():
        for fid in fids:
            autos = (5.0 + 0.1 * rng.standard_normal((n_frames, N_INPUTS))).astype(np.float32)
            autos[:, list(dead)] = 0.0
            P = autos[:, autos.mean(axis=0) > 0.5].mean(axis=1)
            coh = strength.get(band, 0.0) * (1.0 + (fid == marker)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
            noise = (rng.standard_normal((n_frames, len(k))) + 1j * rng.standard_normal((n_frames, len(k)))) \
                / np.sqrt(2 * NFFT * count)[None, :]
            drift = 1.0 + 0.05 * rng.standard_normal((n_frames, 1))
            stacks = ((coh * drift + noise) * (NFFT * P)[:, None]).astype(np.complex64)
            if same_pols:
                stacks[:, 1::2] = stacks[:, 0::2]
            meta = dict(freq_id=int(fid), channel=int(band), freq_mhz=800.0 - fid * W_MHZ, n_frames=int(n_frames),
                        time0_ctime=float(t0), j0=int(1000 * seed % 7), nfft=NFFT, is_pilot=bool(fid == marker),
                        pilot_fine_bin_naive=int(rng.integers(-3000, 3000)), seconds=n_frames * NFFT * 2.56e-6)
            np.savez(directory / f"{fid}.npz", meta=np.array(json.dumps(meta)), keys=k, count=count, stacks=stacks,
                     autos=autos)
    return directory


def write_detector_run(directory, *, seed, n_frames=11, bands=(14, 15, 16), null=1.0):
    """chime_detector_outputs.npz with the keys the capture modules read."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    nb = len(bands)
    valid = np.ones((n_frames, nb), bool); valid[0, 0] = False
    mask = rng.random((n_frames, nb)) < 0.3
    excess_db = rng.normal(1.0, 2.0, (n_frames, nb)); excess_db[2, 1] = np.nan
    np.savez(directory / "chime_detector_outputs.npz", physical_channel=np.array(bands), mask=mask, valid=valid,
             pilot_excess_db=excess_db, normalized_pilot_excess=excess_db / 3.0,
             coarse_power_ratio=rng.normal(1.0, 0.05, (n_frames, nb)) * null,
             null_power_ratio=np.full(nb, float(null)))
    return directory
