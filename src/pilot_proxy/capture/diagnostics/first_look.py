"""First look at the reduced products of one dump: one row per frequency file.

The CSV columns are frozen on disk. In the printed lines the control band's bin at the nominal marker position is
named as such (for CHIME's band 37, control bin 491, the nominal ATSC pilot position).

usage: pilot-proxy capture first-look <products dir> [out csv]"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from ..markers import control_bands, resolve_project


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.split("\n\n")[-1]); return 0 if argv else 2
    control = control_bands(resolve_project(None))
    d = argv[0]
    out = argv[1] if len(argv) > 1 else os.path.join(d, "first_look.csv")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        a = z["autos"]; k = z["keys"]; s = z["stacks"]
        dead = int((a.mean(axis=0) < 0.5).sum())
        def stack(key):
            i = np.where((k == key).all(axis=1))[0]
            return np.abs(s[:, i[0]]).mean() / (a.mean() * m["nfft"]) if i.size else np.nan
        r = dict(freq_id=m["freq_id"], channel=m["channel"], freq_mhz=m["freq_mhz"], is_pilot=int(m["is_pilot"]),
                 n_frames=m["n_frames"], j0=m["j0"], mean_power=round(float(a.mean()), 3), dead_inputs=dead,
                 ns1_pp=round(float(stack((0, 1, 0, 0))), 5), ns1_qq=round(float(stack((0, 1, 1, 1))), 5),
                 ew1_pp=round(float(stack((1, 0, 0, 0))), 5), ew2_pp=round(float(stack((2, 0, 0, 0))), 5),
                 seconds=m["seconds"])
        if m["is_pilot"]:
            pr = z["peak_ratio"]; pb = z["peak_bin"]
            r.update(line_ratio_min=round(float(pr.min()), 1), line_ratio_med=round(float(np.median(pr)), 1),
                     line_ratio_max=round(float(pr.max()), 1), line_bin_med=int(np.median(pb)),
                     line_bin_spread=int(pb.max() - pb.min()), line_offset_khz=round((int(np.median(pb)) - m["pilot_fine_bin_naive"]) * 0.390625e3 / m["nfft"], 2))
            # per-input line amplitude at the peak bin, relative to that input's rms: how coherent the line is across the array
            cut = z["pilot_cut"]                     # [frame, input, 129], peak at index 64
            amp = np.abs(cut[:, :, 64])              # [frame, input]
            rms = np.sqrt(a * m["nfft"])            # expected |F| for noise per input
            snr = amp / np.maximum(rms, 1e-6)
            r.update(line_snr_med=round(float(np.median(snr)), 2), line_snr_p90=round(float(np.percentile(snr, 90)), 2),
                     inputs_with_line=int((np.median(snr, axis=0) > 3).sum()))
        rows.append(r)
    cols = list(rows[0].keys()) if rows else []
    for r in rows:
        for c_ in cols:
            r.setdefault(c_, "")
    with open(out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c_]) for c_ in cols) + "\n")
    print(f"{len(rows)} files -> {out}")
    for r in rows:
        if r["is_pilot"]:
            where = f"control bin {r['freq_id']} (nominal ATSC pilot position)" if r["channel"] in control else f"fid {r['freq_id']}"
            print(f"ch{r['channel']:02d} {where} frames {r['n_frames']} line ratio {r['line_ratio_min']}..{r['line_ratio_max']} "
                  f"offset {r['line_offset_khz']} kHz bin spread {r['line_bin_spread']} snr med {r['line_snr_med']} inputs>3 {r['inputs_with_line']} dead {r['dead_inputs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
