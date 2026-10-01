"""Frame-merge invariance: merging adjacent frame pairs (N to 2N, T_frame to 2 T_frame) leaves the power form of R
unmoved (capture ruling release r5.2, tests/test_a18.py test_frame_merge, assertion values verbatim)."""
from __future__ import annotations

import glob
import json

import numpy as np

from capture_data import dump_dir
from pilot_proxy.capture import units
from pilot_proxy.characterization.coherence import zero_frequency_gain

N = 16384
TF = N * 2.56e-6
CAP = 86164.0905


def _amp(v):
    n = v.size; s = v.sum(); C = (abs(s) ** 2 - (abs(v) ** 2).sum()) / (n * (n - 1)); return np.sqrt(max(C, 0.0))


def _merge_db(v, n_b, tau=100.0):
    """(booked, power) dB change of R = x(A) G when frame pairs merge, G = tau / T_frame."""
    n2 = (v.size // 2) * 2; v2 = 0.5 * (v[:n2:2] + v[1:n2:2])
    a1, a2 = _amp(v[:n2]), _amp(v2)
    if a1 <= 0 or a2 <= 0:
        return None
    G1, G2 = zero_frequency_gain(tau, CAP, TF), zero_frequency_gain(tau, CAP, 2 * TF)
    booked = 10 * np.log10((a2 * G2) / (a1 * G1))
    power = 10 * np.log10((units.to_power(a2, n_b, 2 * N) * G2) / (units.to_power(a1, n_b, N) * G1))
    return booked, power


def test_frame_merge_on_a_synthetic_stationary_stack():
    rng = np.random.default_rng(20260930)
    bk, pw = [], []
    for k in range(64):
        n_b = 512
        noise = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) / np.sqrt(2 * N * n_b)
        v = 3e-4 * np.exp(1j * rng.uniform(0, 2 * np.pi)) + noise           # a stationary coherent residual
        r = _merge_db(v, n_b)
        if r: bk.append(r[0]); pw.append(r[1])
    assert f"{np.median(pw):.2f}" in ("0.00", "-0.00"), np.median(pw)
    assert abs(np.median(bk) + 3.01) < 0.01, np.median(bk)


def test_frame_merge():
    """Merging adjacent frame pairs (T_frame -> 2 T_frame) must leave R unmoved: 0.00 dB in the power form."""
    ev = "20260917040230"; bk, pw = [], []
    for f in sorted(glob.glob(f"{dump_dir(ev)}/*.npz")):
        z = np.load(f); m = json.loads(str(z["meta"]))
        if m["channel"] not in (35, 31, 17): continue
        k = z["keys"]; P = z["autos"][:, z["autos"].mean(axis=0) > 0.5].mean(axis=1)
        for c in [(0, 32), (0, 128), (1, 0)]:
            w = np.where((k[:, 0] == c[0]) & (k[:, 1] == c[1]) & (k[:, 2] == 0) & (k[:, 3] == 0))[0]
            v = (z["stacks"][:, w[0]] / (N * P)).astype(complex)
            r = _merge_db(v, units.class_baseline_count(z, c))
            if r is None: continue
            bk.append(r[0]); pw.append(r[1])
    mb, mp = float(np.median(bk)), float(np.median(pw))
    assert len(pw) == 93, len(pw)
    assert f"{mp:.2f}" in ("0.00", "-0.00"), mp
    assert abs(mb + 3.01) < 0.01, mb


