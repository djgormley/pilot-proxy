"""Place a dump on each band's archive policy ladder.

The archive keeps a frame under policy cal_q when Q = F / mu0 <= eta_q, with
eta_q the q-quantile of Q over the calibration frames (method 'higher'). Here
eta_q is taken over all valid archive frames of the band (the per-band
products of record, population ``all_valid_frames``), and each dump's frames
are placed against it.

usage: pilot-proxy capture ladder-place --archive-products DIR <detector run dir> [<detector run dir> ...] [--project DIR]
"""
from __future__ import annotations

import argparse
import os
import sys

from pilot_proxy.characterization.coarse_ladder import place_dumps, product_thresholds

from .markers import resolve_project

QS = (0.1, 0.5, 0.9)


def ladder_thresholds(archive_products, project, quantiles=QS):
    """{band: eta ladder} over the screened bands whose archive product exists."""
    eta = {}
    for band in project.frequency_plan.bands("screened"):
        ch, fid = int(band.label), int(project.target_freq_id(band))
        p = os.path.join(archive_products, f"{fid}.npz")
        if not os.path.exists(p): continue
        eta[ch] = product_thresholds(p, quantiles)
    return eta


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilot-proxy capture ladder-place", description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive-products", required=True, help="the archive per-band products (<freq_id>.npz)")
    ap.add_argument("--project", default=None)
    ap.add_argument("runs", nargs="+", metavar="detector_run")
    args = ap.parse_args(argv)
    eta = ladder_thresholds(args.archive_products, resolve_project(args.project))
    print("archive eta (Q = F/mu0) per channel: ch  n_frames  eta0.1  eta0.5  eta0.9")
    for ch in sorted(eta): print(f"  {ch:2d}  {eta[ch]['n']:6d}   {eta[ch][0.1]:.4f}  {eta[ch][0.5]:.4f}  {eta[ch][0.9]:.4f}")
    for run in args.runs:
        placed = place_dumps(os.path.join(run, "chime_detector_outputs.npz"), eta)
        print(f"\n{os.path.basename(run)}: per channel, median dump Q, and the fraction of dump frames kept under each policy")
        print("  ch   Q_med    archive Q_med   kept@0.1  kept@0.5  kept@0.9   verdict")
        for c in sorted(placed):
            if c not in eta: continue
            frac = placed[c]["kept_fraction"]
            verdict = "kept by cal_q0.1" if frac[0.1] >= 0.5 else "kept by cal_q0.5" if frac[0.5] >= 0.5 else "kept by cal_q0.9" if frac[0.9] >= 0.5 else "rejected by every calibrated policy"
            print(f"  {c:2d}   {placed[c]['q_median']:6.3f}     {eta[c]['median']:6.3f}       {frac[0.1]:.2f}      {frac[0.5]:.2f}      {frac[0.9]:.2f}    {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
