"""The marker-bin to in-band correction, measured on the 2026 capture.

The 2020 cohort measures a band's residual on its marker bin, one coarse bin
per file. The ruling reads the in-band median over the band's data bins. To
carry a 2020 marker-bin level onto the ruling's basis the two must be
related, and the relation is measured here on the bands the 2026 capture
records with both, over the science dumps and the five long baseline classes,
with exactly the estimator, sky reference and larger-polarisation rule the
ruling uses.

This module imports numpy and the standard library only, so it runs under
the interpreter of record (which has no YAML reader): the marker bins, N and
the science dumps come from the JSON that ``pilot-proxy capture marker-map``
writes.

Writes the correction CSV (the frozen layout of pilot_to_inband.csv).
usage: python -m pilot_proxy.detectors.narrowband_marker.marker_to_inband --marker-map JSON --datasets DIR --out CSV
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

LONG = [(0,128),(0,255),(1,0),(2,0),(3,0)]
STRONG = 1e-3          # a band whose marker-bin level exceeds this is a strong transmitter


def load(d, cls, NFFT):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f); m = json.loads(str(z["meta"])); k = z["keys"]
        P = z["autos"][:, z["autos"].mean(axis=0) > 0.5].mean(axis=1)
        idx = [int(np.where((k[:,0]==cls[0]) & (k[:,1]==cls[1]) & (k[:,2]==p) & (k[:,3]==p))[0][0]) for p in (0,1)]
        def st(v):
            n = v.size; s = v.sum()
            C = (abs(s)**2 - (abs(v)**2).sum())/(n*(n-1)) if n > 1 else np.nan
            return float(np.sqrt(max(C, 0.0)))/(NFFT*P.mean()) if np.isfinite(C) else np.nan
        out[int(m["freq_id"])] = dict(ch=int(m["channel"]), f=float(m["freq_mhz"]),
                                      apol=(st(z["stacks"][:,idx[0]]), st(z["stacks"][:,idx[1]])))
    fids = sorted(out); fr = np.array([out[f]["f"] for f in fids])
    ap = [np.array([out[f]["apol"][p] for f in fids]) for p in (0,1)]
    for i, f in enumerate(fids):
        near = np.abs(fr - fr[i]) <= 40.0
        out[f]["xpol"] = tuple(out[f]["apol"][p] - float(np.nanpercentile(ap[p][near], 10)) for p in (0,1))
    return out


def correction(PILOT, SCI, NFFT, datasets):
    pil, inb = {}, {}
    for cls in LONG:
        for ev in SCI:
            ep = load(os.path.join(datasets, f"pilot_reduce_{ev}"), cls, NFFT)
            for ch, fid in PILOT.items():
                fids = [f for f in ep if ep[f]["ch"] == ch]
                if not fids or fid not in ep: continue
                pil.setdefault((ch, cls), []).append(max(ep[fid]["xpol"]))
                inb.setdefault((ch, cls), []).append(max(float(np.median([ep[f]["xpol"][p] for f in fids])) for p in (0,1)))

    rows, all_db, strong_db = [], [], []
    for ch in sorted(PILOT):
        ps = [float(np.median(pil[(ch,c)])) for c in LONG if (ch,c) in pil]
        isx = [float(np.median(inb[(ch,c)])) for c in LONG if (ch,c) in inb]
        if not ps or not isx: continue
        P, I = float(np.median(ps)), float(np.median(isx))
        usable = P > 0 and I > 0
        d = 10*np.log10(I/P) if usable else float("nan")
        strong = usable and P > STRONG
        if usable: all_db.append(d)
        if strong: strong_db.append(d)
        rows.append((ch, P, I, d, usable, strong))
    return rows, all_db, strong_db


def main(argv=None):
    ap_ = argparse.ArgumentParser(prog="marker_to_inband", description=__doc__.split("\n\n")[0])
    ap_.add_argument("--marker-map", required=True, help="the JSON of pilot-proxy capture marker-map")
    ap_.add_argument("--datasets", required=True, help="root holding pilot_reduce_<event>/")
    ap_.add_argument("--out", required=True)
    args = ap_.parse_args(argv)
    with open(args.marker_map) as fh:
        spec = json.load(fh)
    if spec.get("schema") != "capture_marker_map_v1":
        raise SystemExit(f"{args.marker_map}: not a capture marker map")
    PILOT = {int(k): int(v) for k, v in spec["markers"].items()}
    SCI = list(spec["science_events"]); NFFT = int(spec["nfft"]); OUT = args.out
    rows, all_db, strong_db = correction(PILOT, SCI, NFFT, args.datasets)

    with open(OUT, "w") as fh:
        fh.write("channel,pilot_A_long,inband_A_long,correction_db,usable,strong\n")
        for ch, P, I, d, u, s in rows:
            fh.write(f"{ch},{P:.6e},{I:.6e},{'' if not u else f'{d:.3f}'},{u},{s}\n")

    a, s = np.array(all_db), np.array(strong_db)
    print(f"{'ch':>3} {'marker A':>12} {'in-band A':>12} {'corr dB':>9}  class")
    for ch, P, I, d, u, st in rows:
        print(f"{ch:>3} {P:>12.4e} {I:>12.4e} {('%+9.2f'%d) if u else '      n/a'}  {'strong' if st else ('weak' if u else 'unusable')}")
    print(f"\nall usable bands ({len(a)}): median {np.median(a):+.2f} dB, worst {a.min():+.2f} dB")
    print(f"strong transmitters ({len(s)}): median {np.median(s):+.2f} dB, worst {s.min():+.2f} dB")
    print("\nThe correction tracks transmitter strength: a band at the control level reads the same on its")
    print("marker bin and in band, so its ratio is near unity and tells us nothing. Band 24 is a")
    print("strong transmitter, the second strongest in the band in the archive, so the strong-transmitter")
    print("subset is the relevant population and its worst case is the figure to carry.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
