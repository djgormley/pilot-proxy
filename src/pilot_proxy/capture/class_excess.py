"""Per-epoch, per-class, per-product in-band excess for every band, and the control band's bins per class
(amendment 8).

Classes: the ten of the control level. For each epoch and each coarse bin
the noise-bias-free coherent amplitude over all frames, the 10th-percentile
sky reference within 40 MHz (per class and product), the excess; per band the
in-band median over its bins; for the control band also the per-bin values
(the control sample).

Products (``--products``): ``xx,yy`` reads the two same-polarisation stacks
separately (pol 0 and 1); ``stokes_i`` reads the complex mean of the XX and
YY class stacks per frame, (v_XX + v_YY)/2, through the same
noise-bias-free estimator, so its debiasing uses its own per-frame noise
(pol ``I``).

Output: class_excess_epochs.csv (channel, epoch, ew, ns, pol, excess, n_bins)
and class_floor_bins.csv (epoch, ew, ns, pol, freq_id, excess for the control
band's bins); the file names are frozen. Every product of an epoch must carry
the same classes and baseline counts (``keys``, ``count``), which the OC table
reads from the first product; a product that differs is refused.

usage: pilot-proxy capture class-excess --epochs FILE --datasets DIR --products xx,yy|stokes_i --out-dir DIR [--project DIR]
       (FILE: whitespace-separated label=<dir ending in _<event>>, as epochs_14.txt)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

from .markers import control_bands, resolve_project
from .units import TEN_CLASSES

CLASSES = list(TEN_CLASSES)
PRODUCT_SETS = {"xx,yy": (0, 1), "stokes_i": ("I",)}


def read_epochs(path):
    """The events of an epochs file (label=<dir>_<event> tokens), in order."""
    with open(path) as fh:
        return [l.split("=")[1].split("_")[-1] for l in fh.read().split()]


def measure(EPOCHS, PRODUCTS, *, datasets, nfft, control):
    rows, floor = [], []
    for ev in EPOCHS:
        d = os.path.join(datasets, f"pilot_reduce_{ev}"); out = {}; first = None
        for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
            z = np.load(f); m = json.loads(str(z["meta"])); k = z["keys"]
            if first is None: first = (k, z["count"])
            elif not (np.array_equal(k, first[0]) and np.array_equal(z["count"], first[1])):
                raise ValueError(f"{f}: classes or baseline counts differ from the epoch's first product")
            P = z["autos"][:, z["autos"].mean(axis=0) > 0.5].mean(axis=1)
            a = {}
            for c in CLASSES:
                for p in PRODUCTS:
                    ws = [np.where((k[:, 0] == c[0]) & (k[:, 1] == c[1]) & (k[:, 2] == q) & (k[:, 3] == q))[0] for q in ((0, 1) if p == "I" else (p,))]
                    if any(w.size == 0 for w in ws): a[(c, p)] = np.nan; continue
                    v = (z["stacks"][:, ws[0][0]] if p != "I" else (z["stacks"][:, ws[0][0]] + z["stacks"][:, ws[1][0]]) / 2) / (nfft * P); n = v.size; s = v.sum()
                    C = (abs(s) ** 2 - (abs(v) ** 2).sum()) / (n * (n - 1)) if n > 1 else np.nan
                    a[(c, p)] = float(np.sqrt(max(C, 0.0))) if np.isfinite(C) else np.nan
            out[int(m["freq_id"])] = dict(ch=int(m["channel"]), f=float(m["freq_mhz"]), a=a)
        fids = sorted(out); fr = np.array([out[f]["f"] for f in fids])
        for c in CLASSES:
            for p in PRODUCTS:
                aa = np.array([out[f]["a"][(c, p)] for f in fids])
                x = {}
                for i, f in enumerate(fids):
                    near = np.abs(fr - fr[i]) <= 40.0
                    x[f] = out[f]["a"][(c, p)] - float(np.nanpercentile(aa[near], 10))
                for ch in sorted({out[f]["ch"] for f in fids}):
                    v = [x[f] for f in fids if out[f]["ch"] == ch and np.isfinite(x[f])]
                    rows.append(dict(channel=ch, epoch=ev, ew=c[0], ns=c[1], pol=p, excess=f"{np.median(v):.6e}" if v else "", n_bins=len(v)))
                for f in fids:
                    if out[f]["ch"] in control and np.isfinite(x[f]): floor.append(dict(epoch=ev, ew=c[0], ns=c[1], pol=p, freq_id=f, excess=f"{x[f]:.6e}"))
        print(ev, "done", flush=True)
    return rows, floor


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilot-proxy capture class-excess", description=__doc__.split("\n\n")[0])
    ap.add_argument("--epochs", required=True, help="epochs file (label=<dir>_<event> tokens)")
    ap.add_argument("--datasets", required=True, help="root holding pilot_reduce_<event>/")
    ap.add_argument("--products", required=True, choices=sorted(PRODUCT_SETS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)
    project = resolve_project(args.project)
    rows, floor = measure(read_epochs(args.epochs), PRODUCT_SETS[args.products], datasets=args.datasets,
                          nfft=int(project.detector_config.nfft), control=control_bands(project))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "class_excess_epochs.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out_dir, "class_floor_bins.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(floor[0])); w.writeheader(); w.writerows(floor)
    print("wrote", args.out_dir, "class_excess_epochs.csv", len(rows), "rows; class_floor_bins.csv", len(floor), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
