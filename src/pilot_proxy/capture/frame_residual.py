"""Frame-level residual of one dump (the capture's frame analysis, FRAME_ANALYSIS_PREDECLARATION.md).

Per band, baseline class, polarisation and frame set (all frames; the
detector's kept and rejected frames; the archive's policy ladder cal_q0.1,
cal_q0.5, cal_q0.9 and keep_all, ranked by normalized marker excess): the
noise-bias-free coherent amplitude of the class stack over the set in units
of the per-frame live-input power, minus a sky reference (10th percentile over
the bins within 40 MHz), floored at zero; the in-band median and maximum over
the band's bins, the value at the marker bin, and the marker bin's
structure-function diagnostics (phi_fast, the within-dump gain G_dump and the
lag of the first zero).

A ``U_i`` row per band gives the per-input retained power at the marker bin
against the control band's bins: its median (``excess_pilot_bin``) and 90th
percentile (``excess_inband_median``). The fraction of inputs above a
tolerance is a science reading and is not computed here (``excess_inband_max``
is empty on these rows); the per-input values are written beside the table
(``<out stem>_inputs.csv``: channel, n_frames, input, U, with U the float32
value written as repr of its exact float64) so that any threshold can be
applied later.

The column names are frozen on disk (``excess_pilot_bin``, ``G_dump_pilot``
and the rest name the marker bin).

usage: pilot-proxy capture frame-residual --products DIR --detector-run DIR --out CSV [--project DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from .markers import control_bands, marker_freq_ids, resolve_project

CLASSES = [(0, 1), (0, 2), (0, 4), (0, 8), (1, 0), (1, 1), (2, 0), (3, 0)]
COLUMNS = ["channel", "ew", "ns", "pol", "frames", "n_frames", "excess_pilot_bin", "excess_inband_median",
           "excess_inband_max", "phi_fast_pilot", "G_dump_pilot", "lag_zero_pilot", "kernel_excess_db_median"]
INPUT_COLUMNS = ["channel", "n_frames", "input", "U"]


def coherent_power(V):                       # V [K] complex; noise-bias-free |mean|^2 * K^2 -> per-frame-product units
    K = V.size
    if K < 2: return np.nan
    return (abs(V.sum())**2 - (abs(V)**2).sum()) / (K * (K - 1))


def structure(V):
    K = V.size
    if K < 4: return np.nan, np.nan, np.nan
    m = V.sum(); alpha = (V * np.conj(m)).real / max(abs(m), 1e-30)
    var = alpha.var()
    if var <= 0: return np.nan, np.nan, np.nan
    rho = []
    for l in range(1, K):
        D = np.mean((alpha[:-l] - alpha[l:])**2); rho.append(1 - D / (2 * var))
    rho = np.array(rho); phi_fast = 1 - rho[0]
    zero = np.argmax(rho <= 0) if (rho <= 0).any() else len(rho)
    G = 1 + 2 * rho[:zero].sum()
    return phi_fast, G, zero + 1


def measure(prod_dir, run_dir, *, markers, control, nfft):
    """(rows, input_rows): the table rows in COLUMNS order and the per-input U values of the U_i rows."""
    # kernel mask per channel per frame
    det = np.load(os.path.join(run_dir, "chime_detector_outputs.npz"), allow_pickle=True)
    kch = det["physical_channel"]; kmask = det["mask"].astype(bool); kvalid = det["valid"].astype(bool); kexc = det["pilot_excess_db"]
    mask_of = {int(c): kmask[:, j] & kvalid[:, j] for j, c in enumerate(kch)}
    exc_of = {int(c): kexc[:, j] for j, c in enumerate(kch)}
    # the archive's policy ladder: keep the least-affected fraction of valid frames by normalized marker excess
    # (cal_q0.1, cal_q0.5, cal_q0.9) and keep_all; ranking uses normalized_pilot_excess (NaN = no excess = kept first)
    nexc = det["normalized_pilot_excess"] if "normalized_pilot_excess" in det.files else det["pilot_excess_db"]
    POLICIES = {"cal_q0.1": 0.1, "cal_q0.5": 0.5, "cal_q0.9": 0.9, "keep_all": 1.0}

    def policy_sets(ch, n):
        j = int(np.where(kch == ch)[0][0]); x = np.array(nexc[:n, j], dtype=float); v = kvalid[:n, j]
        x = np.where(np.isnan(x), -np.inf, x)
        order = np.argsort(x, kind="stable"); out = {}
        for name, q in POLICIES.items():
            k = int(round(q * v.sum())); sel = np.zeros(n, bool); sel[order[:k]] = True; sel &= v
            out[name] = sel
        return out
    # load products
    files = sorted(glob.glob(os.path.join(prod_dir, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4]))
    Z = {}
    for f in files:
        z = np.load(f); m = json.loads(str(z["meta"]))
        keys = z["keys"]; idx = {}
        for (ew, ns) in CLASSES:
            for pol in (0, 1):
                i = np.where((keys[:, 0] == ew) & (keys[:, 1] == ns) & (keys[:, 2] == pol) & (keys[:, 3] == pol))[0]
                if i.size: idx[(ew, ns, pol)] = int(i[0])
        a = z["autos"]; live = a.mean(axis=0) > 0.5
        Z[m["freq_id"]] = dict(ch=m["channel"], f=m["freq_mhz"], V=z["stacks"], idx=idx, P=a[:, live].mean(axis=1), autos=a, live=live, n=m["n_frames"])
    fids = sorted(Z)

    # amplitude ratio per bin, class, frame set
    def amp(fid, cls, sel):
        d = Z[fid]; i = d["idx"].get(cls)
        sel = sel[:d["n"]]                       # a file can be a few frames short (node wrote less)
        if i is None or sel.sum() < 2: return np.nan
        C = coherent_power(d["V"][sel, i]); P = d["P"][sel].mean()
        return np.sqrt(max(C, 0.0)) / (nfft * P)
    allsel = {fid: np.ones(Z[fid]["n"], bool) for fid in fids}
    A_all = {(fid, cls): amp(fid, cls, allsel[fid]) for fid in fids for cls in [(e, n, p) for (e, n) in CLASSES for p in (0, 1)]}

    def ref_level(fid, cls):
        f0 = Z[fid]["f"]; vals = [A_all[(g, cls)] for g in fids if abs(Z[g]["f"] - f0) <= 40 and np.isfinite(A_all[(g, cls)])]
        return np.percentile(vals, 10) if len(vals) >= 5 else np.nan
    rows = []; inputs = []
    chans = sorted(set(d["ch"] for d in Z.values()))
    for ch in chans:
        bins = [fid for fid in fids if Z[fid]["ch"] == ch]; pil = markers.get(ch)
        n = Z[bins[0]]["n"]
        kept = ~mask_of[ch][:n] if ch in mask_of else None
        sets = {"all": np.ones(n, bool)}
        if kept is not None:
            sets["kept"] = kept; sets["rejected"] = ~kept
            sets.update(policy_sets(ch, n))
        for cls in [(e, nn, p) for (e, nn) in CLASSES for p in (0, 1)]:
            for name, sel in sets.items():
                if sel.sum() < 5 and name != "all":
                    continue
                ex = {}
                for fid in bins:
                    r = ref_level(fid, cls); a = amp(fid, cls, sel)
                    ex[fid] = max(a - r, 0.0) if np.isfinite(a) and np.isfinite(r) else np.nan
                vals = np.array([v for v in ex.values() if np.isfinite(v)])
                if not vals.size: continue
                pf, G, lag0 = structure(Z[pil]["V"][sel[:Z[pil]["n"]], Z[pil]["idx"][cls]]) if (pil in Z and cls in Z[pil]["idx"]) else (np.nan, np.nan, np.nan)
                rows.append(dict(channel=ch, ew=cls[0], ns=cls[1], pol=cls[2], frames=name, n_frames=int(sel.sum()),
                                 excess_pilot_bin=ex.get(pil, np.nan), excess_inband_median=float(np.median(vals)),
                                 excess_inband_max=float(vals.max()),
                                 phi_fast_pilot=pf, G_dump_pilot=G, lag_zero_pilot=lag0,
                                 kernel_excess_db_median=float(np.nanmedian(exc_of[ch][:n][sel])) if ch in exc_of else np.nan))
        # per-input retained power at the marker bin vs the control band's bins, kept frames if available else all
        if pil in Z and any(Z[g]["ch"] in control for g in fids):
            sel = sets.get("kept", sets["all"]) if (kept is not None and kept.sum() >= 5) else sets["all"]
            refbins = [g for g in fids if Z[g]["ch"] in control]
            ref = np.mean([Z[g]["autos"][sel[:Z[g]["n"]]].mean(axis=0) for g in refbins], axis=0)
            live = Z[pil]["live"] & (ref > 0.5)
            U = (Z[pil]["autos"][sel[:Z[pil]["n"]]].mean(axis=0)[live] - ref[live]) / ref[live]
            rows.append(dict(channel=ch, ew=-1, ns=-1, pol=-1, frames="U_i", n_frames=int(sel.sum()),
                             excess_pilot_bin=float(np.median(U)), excess_inband_median=float(np.percentile(U, 90)),
                             excess_inband_max=np.nan,
                             phi_fast_pilot=np.nan, G_dump_pilot=np.nan, lag_zero_pilot=np.nan, kernel_excess_db_median=np.nan))
            for i, u in zip(np.flatnonzero(live), U):
                inputs.append(dict(channel=ch, n_frames=int(sel.sum()), input=int(i), U=repr(float(u))))
    return rows, inputs


def inputs_path(out):
    """The per-input sidecar written beside ``out``."""
    stem = out[:-4] if out.endswith(".csv") else out
    return stem + "_inputs.csv"


def write(rows, inputs, out):
    cols = list(rows[0].keys())
    assert cols == COLUMNS, cols
    with open(out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows: fh.write(",".join("" if (isinstance(r[c], float) and np.isnan(r[c])) else str(r[c]) for c in cols) + "\n")
    with open(inputs_path(out), "w") as fh:
        fh.write(",".join(INPUT_COLUMNS) + "\n")
        for r in inputs: fh.write(",".join(str(r[c]) for c in INPUT_COLUMNS) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilot-proxy capture frame-residual", description=__doc__.split("\n\n")[0])
    ap.add_argument("--products", required=True, help="the dump's reduced products (<freq_id>.npz)")
    ap.add_argument("--detector-run", required=True, help="the detector run directory (chime_detector_outputs.npz)")
    ap.add_argument("--out", required=True, help="output CSV; the per-input sidecar is written beside it")
    ap.add_argument("--project", default=None, help="project profile directory (default: the shipped project)")
    args = ap.parse_args(argv)
    project = resolve_project(args.project)
    rows, inputs = measure(args.products, args.detector_run, markers=marker_freq_ids(project),
                           control=control_bands(project), nfft=int(project.detector_config.nfft))
    write(rows, inputs, args.out)
    print(f"{len(rows)} rows -> {args.out}")
    # compact view: NS-1 same-pol class, pol 0, all frames and the two upper rungs
    print("ch  frames    n   excess@marker   inband med   phi_fast  G_dump  lag0  kernel dB")
    for r in rows:
        if (r["ew"], r["ns"], r["pol"]) == (0, 1, 0) and r["frames"] in ("all", "cal_q0.5", "cal_q0.9"):
            med = r["excess_inband_median"]
            print(f"{r['channel']:2d}  {r['frames']:9s} {r['n_frames']:2d}  {r['excess_pilot_bin']:.2e}  {med:.2e}   {r['phi_fast_pilot']:.2f}   {r['G_dump_pilot']:.2f}   {r['lag_zero_pilot']}   {r['kernel_excess_db_median']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
