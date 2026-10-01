"""Cadence campaign report, the detector part: tau_c per band, the structure function by lag class and the phase
coherence, rendered from their CSVs.

The rendered text is frozen: it names the estimator scripts of record (cadence_tau.py, cadence_lags.py), whose
code is ``pilot-proxy capture cadence tau`` and ``cadence lags``. The report of record goes on with the table of
record's dispositions; that section is the science side's (``rfisher records cadence-report``), which appends it
to this file. This module writes everything before it, byte for byte as the report of record has it.

usage: pilot-proxy capture cadence-report <out md> <cadence_tau csv> <lag_coherence csv> [<amendment-3 cadence_tau csv>]"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


SCIENCE_COMMAND = "rfisher records cadence-report --detector-report MD --table-of-record CSV --out MD"


def _is_table_of_record(path):
    with open(path) as fh:
        return "disposition" in (csv.DictReader(fh).fieldnames or [])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] in ("-h", "--help"):
        print(__doc__.split("\n\n")[-1], file=sys.stderr)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    if len(argv) > 4 or (len(argv) == 4 and _is_table_of_record(argv[3])):
        print("the table of record is not an input of this report: its section is written by the science side, "
              f"`{SCIENCE_COMMAND}`, from the file this command writes", file=sys.stderr)
        return 2
    out, tau_csv, lag_csv = argv[0:3]; alt_csv = argv[3] if len(argv) > 3 else None
    tau = _rows(tau_csv); struct = _rows(tau_csv[:-4] + "_structure.csv")
    lag = _rows(lag_csv)
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
        alt = _rows(alt_csv)
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
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
