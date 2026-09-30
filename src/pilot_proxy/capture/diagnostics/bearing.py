"""Bearing of each marker line from the per-input line phasors of one dump, after dividing out a gain solution.

usage: pilot-proxy capture bearing <products dir> <gain .h5, .npz or 'none'> <unix time of the dump> [--selftest]

Model: plane wave from bearing beta (deg east of north) and elevation eps over the feed layout
cylinder = id // 512 (EW 22.0 m per step), position = id % 256 (NS 0.3048 m per step). The EW and NS sign
conventions of the layout are unknown, so every fit is done under the four sign choices; the one that puts the
known stations (17 and 35 at 272 deg, 22 at 12 deg, 21 at 251 deg) nearest their census bearings is reported as
the convention, and the unknown lines (33, 14, 36) are read under it. The station bearings are the campaign's
census values, a record of the 2026 capture.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


C = 299792458.0; EW_M = 22.0; NS_M = 0.3048
KNOWN = {17: 272.2, 35: 272.1, 22: 11.6, 21: 251.3}
UNKNOWN = [33, 14, 36]
BEAR = np.deg2rad(np.arange(0, 360, 0.25)); ELEV = np.deg2rad([0.0])   # ground transmitters; elevation is degenerate with the EW alias anyway

def layout(n=2048):
    i = np.arange(n); return (i // 512).astype(float), (i % 256).astype(float), (i // 256) % 2

def fit_plane_wave(phasor, freq_hz, sx, sy):
    """phasor [2048] complex, already gain-corrected; returns (beta_deg, eps_deg, fraction) for the best fit."""
    cyl, pos, pol = layout(phasor.size)
    x = sx * cyl * EW_M; y = sy * pos * NS_M
    lam = C / freq_hz; w = (np.abs(phasor) > 0).astype(float); u = phasor / np.maximum(np.abs(phasor), 1e-30)   # equal weight per live input
    best = (np.nan, np.nan, -1.0)
    for e in ELEV:
        ce = np.cos(e)
        for b in BEAR:
            geo = 2 * np.pi / lam * (x * np.sin(b) * ce + y * np.cos(b) * ce)
            frac = 0.0
            for p in (0, 1):                      # fit each polarisation with its own reference phase, then add
                m = pol == p
                if not m.any(): continue
                frac += abs(np.sum(w[m] * u[m] * np.exp(-1j * geo[m]))) / max(np.sum(w[m]), 1e-30)
            frac /= 2
            if frac > best[2]: best = (np.rad2deg(b), np.rad2deg(e), frac)
    return best

def line_phasor(z):
    """Per-input phasor of the line at the peak bin, coherent across frames after removing each frame's common phase."""
    cut = z["pilot_cut"][:, :, 64]                # [frames, inputs]
    common = cut.sum(axis=1); common /= np.maximum(np.abs(common), 1e-30)
    return (cut * np.conj(common)[:, None]).mean(axis=0)

def gains_for(gainfile, freq_id, t_unix):
    """Per-input gain at the update nearest t_unix, from the extracted slices (gain_pilots_*.npz) or the full .h5."""
    if gainfile.endswith(".npz"):
        z = np.load(gainfile); ut = z["update_time"]; u = int(np.argmin(np.abs(ut - t_unix)))
        j = int(np.where(z["freq_id"] == freq_id)[0][0]); g = np.zeros(2048, np.complex128); g[z["chan_id"]] = z["gain"][u, j, :]
        return g, float(ut[u])
    import h5py
    with h5py.File(gainfile, "r") as f:
        ut = f["index_map/update_time"][:]; u = int(np.argmin(np.abs(ut - t_unix)))
        chan = f["index_map/input"]["chan_id"][:] if f["index_map/input"].dtype.names else np.arange(2048)
        g = np.zeros(2048, np.complex128); g[chan] = f["gain"][u, freq_id, :]
        return g, float(ut[u])

def selftest():
    rng = np.random.default_rng(1); cyl, pos, pol = layout()
    inst = np.exp(1j * rng.uniform(0, 2 * np.pi, 2048))   # instrumental phases
    freq = 584.3056e6; lam = C / freq
    for beta, eps in ((272.2, 0.0), (166.0, 1.5), (11.6, 0.5)):
        b, e = np.deg2rad(beta), np.deg2rad(eps)
        geo = 2 * np.pi / lam * ((+1) * cyl * EW_M * np.sin(b) * np.cos(e) + (+1) * pos * NS_M * np.cos(b) * np.cos(e))
        ph = inst * np.exp(1j * geo) * (1 + 0.3 * rng.standard_normal(2048)) + 0.5 * (rng.standard_normal(2048) + 1j * rng.standard_normal(2048))
        cal = ph / inst
        print(f"  injected {beta:6.1f} deg, {eps:4.1f} deg ->", "  ".join(f"({sx:+d},{sy:+d}): {fit_plane_wave(cal, freq, sx, sy)[0]:6.1f}/{fit_plane_wave(cal, freq, sx, sy)[1]:4.1f} f={fit_plane_wave(cal, freq, sx, sy)[2]:.2f}" for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1))))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        print("self-test with random instrumental phases, 30% amplitude scatter and noise at half the line amplitude:"); selftest(); return 0
    if len(argv) < 3 or argv[0] in ("-h", "--help"):
        print(__doc__.split("\n\n")[0].split("\n", 1)[1]); return 0 if argv and argv[0] in ("-h", "--help") else 2
    prod_dir, gainfile, t_unix = argv[0], argv[1], float(argv[2])
    rows = []
    for f in sorted(glob.glob(os.path.join(prod_dir, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f); m = json.loads(str(z["meta"]))
        if not m["is_pilot"] or "pilot_cut" not in z.files or float(np.median(z["peak_ratio"])) < 5: continue
        ph = line_phasor(z)
        if gainfile != "none":
            g, tu = gains_for(gainfile, m["freq_id"], t_unix); ok = np.abs(g) > 0; ph = np.where(ok, ph / np.where(ok, g, 1), 0)
        else:
            tu = np.nan
        freq = m["freq_mhz"] * 1e6 - (int(np.median(z["peak_bin"])) if False else 0)
        res = {(sx, sy): fit_plane_wave(ph, freq, sx, sy) for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1))}
        rows.append((m["channel"], m["freq_id"], float(np.median(z["peak_ratio"])), tu, res))
    print("ch  fid   line   gain@      (+,+)            (-,+)            (+,-)            (-,-)      census")
    for ch, fid, ratio, tu, res in rows:
        cells = "  ".join(f"{res[k][0]:5.1f}/{res[k][1]:4.1f} {res[k][2]:.2f}" for k in ((1, 1), (-1, 1), (1, -1), (-1, -1)))
        print(f"{ch:2d}  {fid:3d}  x{ratio:5.0f}  {tu:10.0f}  {cells}   {KNOWN.get(ch, '')}")
    # choose the convention by the known stations
    def err(k): 
        e = [min(abs(res[k][0] - KNOWN[ch]) % 360, 360 - abs(res[k][0] - KNOWN[ch]) % 360) for ch, _, _, _, res in rows if ch in KNOWN]
        return np.mean(e) if e else np.nan
    scores = {k: err(k) for k in ((1, 1), (-1, 1), (1, -1), (-1, -1))}
    best = min(scores, key=lambda k: scores[k]) if rows else None
    print("mean bearing error of the known stations per convention:", {k: round(v, 1) for k, v in scores.items()}, "-> convention", best)
    for ch, fid, ratio, tu, res in rows:
        if ch in UNKNOWN and best: print(f"  ch{ch}: bearing {res[best][0]:.1f} deg, elevation {res[best][1]:.1f} deg, fit fraction {res[best][2]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
