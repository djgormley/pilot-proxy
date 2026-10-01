"""Per band, per coarse bin: mean power, frame-to-frame variability, shortest-baseline coherence, marker line.

The CSV columns are frozen on disk (``is_pilot``, ``pilot_line_ratio`` and the rest name the marker bin). In the
printed summary the control band's bin at the nominal marker position is named as such (for CHIME's band 37, control
bin 491, the nominal ATSC pilot position): it is not a pilot.

usage: pilot-proxy capture band-shape <products dir> [out csv]"""
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
    project = resolve_project(None)
    NFFT = int(project.detector_config.nfft); W = project.instrument.channel_width_hz / 1e6
    control = control_bands(project)
    d = argv[0]; out = argv[1] if len(argv) > 1 else os.path.join(d, "band_shape.csv")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f); m = json.loads(str(z["meta"]))
        a = z["autos"]; k = z["keys"]; s = z["stacks"]; live = a.mean(axis=0) > 0.5
        pw = a[:, live].mean(axis=1)                      # mean live-input power per frame
        def coh(key):
            i = np.where((k == key).all(axis=1))[0]
            if not i.size: return np.nan
            v = s[:, i[0]] / (pw * m["nfft"])            # per-frame coherence at that baseline class
            return v
        ns1 = coh((0, 1, 0, 0)); ns1q = coh((0, 1, 1, 1)); ew1 = coh((1, 0, 0, 0)); ew2 = coh((2, 0, 0, 0)); ew3 = coh((3, 0, 0, 0))
        psd = z["block_psd"].sum(axis=1)                  # [frame, 16384]
        med = np.median(psd, axis=1)
        r = dict(channel=m["channel"], freq_id=m["freq_id"], freq_mhz=round(m["freq_mhz"], 4), is_pilot=int(m["is_pilot"]),
                 n_frames=m["n_frames"], live_inputs=int(live.sum()), power_mean=round(float(pw.mean()), 3),
                 power_frame_std_pct=round(float(100 * pw.std() / pw.mean()), 2),
                 ns1_coh_mean=round(float(np.abs(ns1.mean())), 5), ns1_coh_frame_std=round(float(np.abs(ns1 - ns1.mean()).std()), 5),
                 ns1q_coh_mean=round(float(np.abs(ns1q.mean())), 5),
                 ew1_coh_mean=round(float(np.abs(ew1.mean())), 5), ew2_coh_mean=round(float(np.abs(ew2.mean())), 5), ew3_coh_mean=round(float(np.abs(ew3.mean())), 5),
                 psd_peak_over_median=round(float((psd.max(axis=1) / med).mean()), 1))
        if m["is_pilot"]:
            kp = (-m["pilot_fine_bin_naive"]) % NFFT      # inverted sense
            idx = (kp + np.arange(-600, 601)) % NFFT
            j = idx[np.argmax(psd.mean(axis=0)[idx])]
            r.update(pilot_line_ratio=round(float((psd[:, j] / med).mean()), 1),
                     pilot_offset_khz=round(float(-((((j - kp + NFFT // 2) % NFFT) - NFFT // 2)) * W * 1e3 / NFFT), 2),
                     pilot_line_frame_std_pct=round(float(100 * psd[:, j].std() / psd[:, j].mean()), 2))
        rows.append(r)
    cols = []
    for r in rows:
        for c in r:
            if c not in cols: cols.append(c)
    with open(out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows: fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"{len(rows)} rows -> {out}")
    # per-channel summary
    chans = sorted(set(r["channel"] for r in rows))
    print("ch  bins  power(min..max)  frame_std%  ns1_coh(min..max)  ew1_coh(max)  pilot: line x, offset kHz, line frame std %")
    for c in chans:
        rr = [r for r in rows if r["channel"] == c]; pil = [r for r in rr if r["is_pilot"]]
        p = [r["power_mean"] for r in rr]; n1 = [r["ns1_coh_mean"] for r in rr]; e1 = [r["ew1_coh_mean"] for r in rr]; fs = [r["power_frame_std_pct"] for r in rr]
        ps = f"x{pil[0]['pilot_line_ratio']}, {pil[0]['pilot_offset_khz']:+.2f}, {pil[0]['pilot_line_frame_std_pct']}" if pil else "-"
        if pil and c in control: ps = f"control bin {pil[0]['freq_id']} (nominal ATSC pilot position): " + ps
        print(f"{c:2d}  {len(rr):2d}   {min(p):6.2f}..{max(p):6.2f}   {np.median(fs):5.2f}   {min(n1):.4f}..{max(n1):.4f}   {max(e1):.4f}   {ps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
