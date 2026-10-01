"""Noise normalization on the control band: the per-frame variance of a class stack times N n_b is about one, and
n_b is the stack's count (capture ruling release r5.2, tests/test_a18.py test_noise_normalization, values verbatim)."""
from __future__ import annotations

import glob
import json

import numpy as np

from capture_data import dump_dir
from pilot_proxy.capture import units

N = 16384
# a18_units.NB, the ruling's baseline counts, which the products' count must equal
NB = {(0, 1): 1020, (0, 32): 896, (0, 64): 768, (0, 128): 512, (0, 255): 4, (1, 0): 768, (2, 0): 512, (3, 0): 256}


def test_noise_normalization():
    """Per-frame variance of the class stack times N n_b is about one on the control band, per class and product."""
    ev = "20260917040230"; res = {}
    for f in sorted(glob.glob(f"{dump_dir(ev)}/*.npz")):
        z = np.load(f); m = json.loads(str(z["meta"]))
        if m["channel"] != 37: continue
        k = z["keys"]; P = z["autos"][:, z["autos"].mean(axis=0) > 0.5].mean(axis=1)
        for c in NB:
            for p in (0, 1):
                n_b = units.class_baseline_count(z, c, p)
                assert n_b == NB[c], (c, n_b)
                w = units.class_column(k, c, p)
                v = z["stacks"][:, w] / (N * P); n = v.size
                ve = float(np.mean(np.abs(v - v.mean()) ** 2) * n / (n - 1))
                res.setdefault((c, p), []).append(ve * N * n_b)
    med = {k: float(np.median(v)) for k, v in res.items()}
    allm = float(np.median(list(med.values())))
    assert len(med) == 16
    assert all(0.85 <= x <= 1.2 for x in med.values()), med
    assert abs(allm - 1) < 0.1, allm
