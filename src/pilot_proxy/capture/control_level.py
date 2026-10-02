"""Baseline dependence of the residual and the control band's level (2026-09-18, after the ruling audit, finding S1).

For each science dump, each coarse bin and each baseline class in CLASSES
(both polarisations, read separately): the noise-bias-free coherent amplitude
ratio over all frames (item 2 of the predeclaration), the sky reference (10th
percentile over the bins within 40 MHz, per class and polarisation) and the
excess. Per band the in-band median excess. The control band (the plan's band
with no licensed emitter, 608 to 614 MHz) gives, from its per-bin excess, the
level of the estimator per class (mean and scatter over its bins and the
dumps): the control level against which a band's excess is read. This is not
the residual floor of the archive chain.

Output: baseline_floor.csv (the frozen file name; per band and class: A, the
control level's mean and scatter as floor_mean and floor_sd, A over the level)
and a printed table.

usage: pilot-proxy capture control-level --datasets DIR --out-dir DIR [--events E,E,...] [--project DIR]
       (the events default to the project's record of the capture's science dumps)
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


def science_events(project):
    """The science dumps the project's capture record names."""
    return list(project.record_module("capture_campaign").SCIENCE_EVENTS)


def load(ev, *, datasets, nfft):
    out = {}
    d = os.path.join(datasets, f"pilot_reduce_{ev}")
    for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f); m = json.loads(str(z["meta"])); k = z["keys"]
        P = z["autos"][:, z["autos"].mean(axis=0) > 0.5].mean(axis=1)
        a = {}
        for c in CLASSES:
            for p in (0, 1):
                w = np.where((k[:, 0] == c[0]) & (k[:, 1] == c[1]) & (k[:, 2] == p) & (k[:, 3] == p))[0]
                if w.size == 0: a[(c, p)] = np.nan; continue
                v = z["stacks"][:, w[0]] / (nfft * P); n = v.size; s = v.sum()
                C = (abs(s) ** 2 - (abs(v) ** 2).sum()) / (n * (n - 1))
                a[(c, p)] = float(np.sqrt(max(C, 0.0)))
        out[int(m["freq_id"])] = dict(ch=int(m["channel"]), f=float(m["freq_mhz"]), a=a)
    fids = sorted(out); fr = np.array([out[f]["f"] for f in fids])
    for key in [(c, p) for c in CLASSES for p in (0, 1)]:
        aa = np.array([out[f]["a"][key] for f in fids])
        for i, f in enumerate(fids):
            near = np.abs(fr - fr[i]) <= 40.0
            out[f].setdefault("x", {})[key] = out[f]["a"][key] - float(np.nanpercentile(aa[near], 10))
    return out


def rows_for(eps, control, layout=None):
    """Rows of baseline_floor.csv; ``layout`` is the instrument's feed layout (the default project's when None)."""
    if layout is None:
        layout = resolve_project(None).instrument.feed_layout
    if layout is None:
        raise ValueError("the instrument records no feed layout")
    ew_m, ns_m = layout.ew_spacing_m, layout.ns_spacing_m
    chans = sorted({v["ch"] for e in eps.values() for v in e.values()})
    rows = []
    for c in CLASSES:
        for p in (0, 1):
            key = (c, p)
            # control level: the control band's bins, all dumps
            ctrl = [e[f]["x"][key] for e in eps.values() for f in e if e[f]["ch"] in control]
            fl_mean = float(np.nanmean(ctrl)); fl_sd = float(np.nanstd(ctrl))
            for ch in chans:
                per_dump = []
                for e in eps.values():
                    v = [e[f]["x"][key] for f in e if e[f]["ch"] == ch]
                    if v: per_dump.append(float(np.nanmedian(v)))
                A = float(np.nanmedian(per_dump)) if per_dump else np.nan
                rows.append(dict(channel=ch, ew=c[0], ns=c[1], baseline_m=round(c[0] * ew_m + c[1] * ns_m, 2) if c[0] == 0 or c[1] == 0 else round(np.hypot(c[0] * ew_m, c[1] * ns_m), 2), pol=p, A=A, floor_mean=fl_mean, floor_sd=fl_sd, A_over_floor=(A / fl_mean if fl_mean > 0 else np.nan), A_minus_floor_over_sd=((A - fl_mean) / fl_sd if fl_sd > 0 else np.nan)))
    return chans, rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilot-proxy capture control-level", description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", required=True, help="root holding pilot_reduce_<event>/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--events", default=None, help="comma-separated events (default: the record's science dumps)")
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)
    project = resolve_project(args.project)
    if project.instrument.feed_layout is None:
        ap.error("the instrument records no feed layout")
    control = control_bands(project); nfft = int(project.detector_config.nfft)
    DUMPS = args.events.split(",") if args.events else science_events(project)
    eps = {ev: load(ev, datasets=args.datasets, nfft=nfft) for ev in DUMPS}
    chans, rows = rows_for(eps, control, project.instrument.feed_layout)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "baseline_floor.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"A over the control band's level (band {', '.join(map(str, control))}; larger polarisation), per class; "
          "'*' = more than 3 control scatters above the level")
    hdr = "ch  " + "  ".join(f"{('ns%d' % c[1] if c[0] == 0 else 'ew%d' % c[0] + ('ns%d' % c[1] if c[1] else '')):>10s}" for c in CLASSES)
    print(hdr)
    for ch in chans:
        cells = []
        for c in CLASSES:
            best = None
            for p in (0, 1):
                r = next(x for x in rows if x["channel"] == ch and x["ew"] == c[0] and x["ns"] == c[1] and x["pol"] == p)
                if best is None or (r["A"] == r["A"] and r["A"] > best["A"]): best = r
            cells.append(f"{best['A_over_floor']:8.1f}{'*' if best['A_minus_floor_over_sd'] > 3 else ' '} ")
        print(f"{ch:2d}  " + "  ".join(cells))
    ref = control[0]
    print(f"control level (band {ref}) per class, larger pol: " + ", ".join(f"{c}: {max(next(x for x in rows if x['channel']==ref and x['ew']==c[0] and x['ns']==c[1] and x['pol']==p)['floor_mean'] for p in (0,1)):.1e}" for c in CLASSES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
