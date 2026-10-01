"""Where the marker line sits, per marker file, under both spectral senses.

The CHIME fine FFT along time is inverted (bin +k is sky below the coarse centre). The control band's bin at the
nominal marker position is named as such (for CHIME's band 37, control bin 491, the nominal ATSC pilot position).

usage: pilot-proxy capture line-check <products dir>"""
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
    NFFT = int(project.detector_config.nfft); W = project.instrument.channel_width_hz / 1e6; DK = W * 1e3 / NFFT   # kHz per fine bin
    control = control_bands(project)
    for f in sorted(glob.glob(os.path.join(argv[0], "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f); m = json.loads(str(z["meta"]))
        if not m["is_pilot"]: continue
        psd = z["block_psd"].sum(axis=1).mean(axis=0)          # mean over frames, sum over blocks -> [16384]
        med = np.median(psd); kp = m["pilot_fine_bin_naive"]
        def best(c, half=600):
            idx = (c + np.arange(-half, half + 1)) % NFFT; j = idx[np.argmax(psd[idx])]
            d = ((j - c + NFFT // 2) % NFFT) - NFFT // 2
            return j, round(float(psd[j] / med), 1), round(d * DK, 2)
        jn, rn, dn = best(kp % NFFT)          # naive sense: bin +k <-> sky above centre
        ji, ri, di = best((-kp) % NFFT)       # inverted sense: bin +k <-> sky below centre
        top = np.argsort(psd)[-3:][::-1]
        def sky_inv(b): return m["freq_mhz"] - (((b + NFFT // 2) % NFFT) - NFFT // 2) * DK / 1e3
        tops = ", ".join(f"{sky_inv(b):.4f} MHz x{psd[b]/med:.0f}" for b in top)
        where = f"control bin {m['freq_id']} (nominal ATSC pilot position)" if m["channel"] in control else f"fid {m['freq_id']} pilot nominal"
        print(f"ch{m['channel']:02d} {where} {m['freq_mhz'] + kp*DK/1e3:.6f} MHz | naive: x{rn} at {dn:+.2f} kHz | inverted: x{ri} at {di:+.2f} kHz (sky {sky_inv(ji):.4f}) | top lines (inverted sky): {tops}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
