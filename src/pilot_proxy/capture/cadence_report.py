"""Cadence campaign report: tau_c per band, the structure function by lag class, the phase coherence, and the
table of record's dispositions, rendered from their CSVs (the report computes nothing on the table of record).

The rendered text is frozen: it names the estimator scripts of record (cadence_tau.py, cadence_lags.py), whose
code is ``pilot-proxy capture cadence tau`` and ``cadence lags``.

usage: pilot-proxy capture cadence-report <out md> <cadence_tau csv> <lag_coherence csv> <table_of_record csv> [<amendment-3 cadence_tau csv>]"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 4 or argv[0] in ("-h", "--help"):
        print(__doc__.split("\n\n")[-1], file=sys.stderr)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    out, tau_csv, lag_csv, tor_csv = argv[0:4]; alt_csv = argv[4] if len(argv) > 4 else None
    tau = list(csv.DictReader(open(tau_csv))); struct = list(csv.DictReader(open(tau_csv[:-4] + "_structure.csv")))
    lag = list(csv.DictReader(open(lag_csv))); tor = list(csv.DictReader(open(tor_csv)))
    chans = sorted({int(r["channel"]) for r in tau})
    classes = sorted({int(float(r["lag_class"])) for r in struct})
    L = []
    L.append("# Cadence campaign: coherence time of the DTV residual level\n")
    L.append("Estimator: FRAME_ANALYSIS_PREDECLARATION.md amendments 3 and 5 (cadence_tau.py). D is the noise-corrected structure function of the per-epoch level on the 0.3 m same-polarisation baseline, in A^2 units; the plateau is the mean D at lags above 7200 s; tau_c is the lag at which D reaches (1 - 1/e) of the plateau. Status constant means the plateau is not positive at two standard errors and G takes the sidereal cap; bound means D stays below the target through the longest populated cadence class (the class marked 3600 s holds the pairs between D3 and the cadence dumps, at lags of 7178 to 7193 s), and only the lower end of the range is used; refused means the trim probes disagree by more than a factor two and G takes the cap, as the archive does. The ruling reads the in-band row, the same quantity as A. Amendment 5 (per-polarisation level on the polarisation that sets A; the archive's trim probes) is the primary form.\n")
    for bins in ("pilot", "inband"):
        L.append(f"\n## {'Pilot bin' if bins == 'pilot' else 'In-band median'}: tau_c and G\n")
        L.append("| ch | pol | status | tau_c (s) | tau_c high (s) | G | plateau (A^2) | plateau s.e. | pairs | classes | epochs used | probes 75 / 90 / 95 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in tau:
            if r["bins"] != bins: continue
            L.append(f"| {r['channel']} | {r.get('pol','')} | {r['status']} | {r['tau_c_s']} | {r['tau_c_high_s']} | {r['G']} | {r['plateau']} | {r['plateau_se']} | {r['n_pairs']} | {r['n_classes']} | {r.get('n_epochs','')} | {r.get('probes','')} |")
        L.append(f"\n### D(lag) / plateau by lag class ({bins}); parentheses give the subtracted noise term over the plateau, same units\n")
        L.append("| ch | " + " | ".join(f"{c} s" for c in classes) + " | plateau |")
        L.append("|---|" + "---|" * (len(classes) + 1))
        by = defaultdict(dict)
        for r in struct:
            if r["bins"] == bins: by[int(r["channel"])][int(float(r["lag_class"]))] = r
        for ch in chans:
            row = by.get(ch, {})
            cells = []
            for c in classes:
                r = row.get(c)
                if r is None or not r["plateau"]: cells.append("")
                else:
                    pl = float(r["plateau"]); cells.append(f"{float(r['D'])/pl:+.2f} ({float(r['D_noise'])/pl:.2f})" if pl > 0 else f"{float(r['D']):.1e}")
            pl = next((r["plateau"] for r in row.values()), "")
            L.append(f"| {ch} | " + " | ".join(cells) + f" | {pl} |")
    if alt_csv:
        alt = list(csv.DictReader(open(alt_csv)))
        L.append("\n## The amendment-3 form beside it (both polarisations averaged, no trim; not the ruling's input)\n")
        L.append("| ch | status | tau_c (s) | G | plateau (A^2) |")
        L.append("|---|---|---|---|---|")
        for r in alt:
            if r["bins"] == "inband": L.append(f"| {r['channel']} | {r['status']} | {r['tau_c_s']} | {r['G']} | {r['plateau']} |")
    L.append("\n## Phase coherence of the reference-subtracted excess (cadence_lags.py, class ns1_x, in-band), mean rho by lag class\n")
    lc = defaultdict(lambda: defaultdict(list))
    for r in lag:
        if r["cls"] == "ns1_x" and r["bins"] == "inband": lc[int(r["channel"])][int(float(r["lag_class"]))].append(float(r["rho"]))
    lclasses = sorted({c for d in lc.values() for c in d})
    L.append("| ch | " + " | ".join(f"{c} s" for c in lclasses) + " |")
    L.append("|---|" + "---|" * len(lclasses))
    for ch in chans:
        L.append(f"| {ch} | " + " | ".join(f"{np.mean(lc[ch][c]):+.2f}" if lc[ch].get(c) else "" for c in lclasses) + " |")
    L.append("\n## Table of record at the measured gain (one line per channel; the CSV has one row per freq_id)\n")
    L.append("| ch | disposition | policy or reason | tau_c (s) | status | G measured | R none G_meas | R deployed G_meas | R deployed G=1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for r in tor:
        ch = int(r["channel"])
        if ch in seen or r["role"] == "pilot": continue
        seen.add(ch)
        L.append(f"| {ch} | {r['disposition']} | {r['policy_or_reason']} | {r['tau_c_s']} | {r['tau_c_status']} | {r['G_measured']} | {r['R_none_Gmeasured']} | {r['R_deployed_Gmeasured']} | {r['R_deployed_G1']} |")
    cnt = defaultdict(int)
    for r in tor: cnt[r["disposition"]] += 1
    L.append("\nRows by disposition: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) + f" (of {len(tor)}).")
    open(out, "w").write("\n".join(L) + "\n"); print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
