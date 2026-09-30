"""Coherence of the residual level across dumps: the cadence coherence time (``tau``) and the phasor coherence
between dumps (``lags``).

``tau`` (predeclaration, amendments 3 and 5). Per epoch d, band and bin set
(marker bin; in-band median): the noise-bias-free amplitude ratio over all
frames on the class (both polarisations averaged, one polarisation, or Stokes
I), minus the epoch's own sky reference (10th percentile over the bins within
40 MHz), unfloored: x_d. Its noise variance sigma_d^2 = var_k(alpha_k) / n_d
from the per-frame projections. Over every pair of epochs closer than a
sidereal day, D_ab = 0.5 [(x_a - x_b)^2 - sigma_a^2 - sigma_b^2], grouped to
the nearest lag class in log lag. Plateau = mean D over pairs with lag > 7200
s. tau_c = lag at which the class-mean D first reaches (1 - 1/e) of the
plateau, linearly interpolated, with at least three populated classes below
7200 s. Status: constant (plateau not positive at 2 s.e.; G = cap), measured
(G = tau_c / T_frame), bound (D below target through the longest populated
cadence class; G range reported, lower end used), or no cadence lags.

Levels (``--level``): ``polavg``, the level on the complex mean of the XX and
YY class stacks (the pol column reads ``avg``); ``stokesI``, the same level
(the complex mean (v_XX + v_YY)/2 is Stokes I, with its own per-frame noise)
with the pol column labelled ``I``; ``polmax``, one polarisation per band (from
``--pol-table``, else the one with the larger in-band median level), or the
one ``--pol`` forces. ``--trim archive`` is the archive's primary
90th-percentile trim of the epoch levels, with probes 75, 90 and 95 and the
spread gate 2.

writes <out csv> (one row per band and bin set) and <out csv stem>_structure.csv (class-mean D per lag).

``lags``. For each band, class (same polarisation, both pols averaged) and bin
set (marker bin; in-band median), the dump's mean phasor V_d = mean over its
frames of the stack mean. For every pair of dumps (a, b) at lag dt the
normalised coherence is rho_ab = Re(V_a conj V_b) / sqrt(|V_a|^2 |V_b|^2),
noise-bias free in the cross term because the two dumps share no samples; the
per-dump noise floor is estimated from the frame-to-frame scatter and
reported. rho near 1 at a lag means the residual's coherent part has not
decorrelated over that lag (it counts toward G); rho near 0 means it has. The
``_x`` classes subtract a reference phasor, the mean over the bins of the
``--reference-bands`` (bands with no emitter in the capture). Lags are grouped
to the nearest predeclared class and averaged over pairs. ``--classes three``
reads ns1, ns8 and ew1 (rows ns1, ns1_x, ns8_x, ew1_x); ``--classes ten`` the
ten classes of the class excess (rows <class>_x, then ns1).

usage: pilot-proxy capture cadence tau [--level polavg|stokesI|polmax] [--pol-table <table_of_record csv>]
           [--trim none|archive] [--class EW,NS] [--pol P] [--project DIR] <out csv> <label>=<products dir> [...]
       pilot-proxy capture cadence lags --classes three|ten --reference-bands B[,B...] [--project DIR]
           <out csv> <label>=<products dir> [...]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

from .markers import marker_freq_ids, resolve_project

PLATEAU_START = 7200.0; CROSS = 1.0 - 1.0 / np.e; MIN_CLASSES = 3
LAG_CLASSES = [15, 30, 45, 60, 90, 120, 180, 240, 360, 480, 600, 720, 960, 1200, 3600, 18000, 36000, 86400]
MAX_TRIM_SPREAD = 2.0
LEVELS = ("polavg", "stokesI", "polmax")
LAG_CLASS_SETS = {
    "three": {"ns1": (0, 1), "ns8": (0, 8), "ew1": (1, 0)},
    "ten": {"ns1": (0, 1), "ns8": (0, 8), "ns32": (0, 32), "ns64": (0, 64), "ns128": (0, 128), "ns255": (0, 255),
            "ew1": (1, 0), "ew1ns32": (1, 32), "ew2": (2, 0), "ew3": (3, 0)},
}


def _profile_values(project):
    return dict(nfft=int(project.detector_config.nfft), sample_seconds=1.0 / project.instrument.sample_rate_hz,
                frame_seconds=project.integration_model.frame_seconds,
                sidereal=project.integration_model.coherence_cap_seconds,
                cap=project.integration_model.coherence_cap_seconds, markers=marker_freq_ids(project))


# ---------------------------------------------------------------------------------------------------- tau
def load_epoch(d, CLS, *, nfft, sample_seconds):
    """Per bin: level a (noise-bias-free), noise variance, frame count, channel, frequency, start time."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f); m = json.loads(str(z["meta"])); k = z["keys"]
        P = z["autos"][:, z["autos"].mean(axis=0) > 0.5].mean(axis=1)                    # [frames] live-input power
        idx = [int(np.where((k[:, 0] == CLS[0]) & (k[:, 1] == CLS[1]) & (k[:, 2] == p) & (k[:, 3] == p))[0][0]) for p in (0, 1)]

        def stats(v):
            n = v.size; s = v.sum()
            C = (abs(s) ** 2 - (abs(v) ** 2).sum()) / (n * (n - 1)) if n > 1 else np.nan
            a = float(np.sqrt(max(C, 0.0))) if np.isfinite(C) else np.nan
            alpha = (v * np.conj(s)).real / abs(s) if abs(s) > 0 else np.zeros(n)
            var = float(alpha.var(ddof=1) / n) if n > 1 else np.nan
            return a, var, n
        a, var, n = stats(z["stacks"][:, idx].mean(axis=1) / (nfft * P))                  # amendment 3: both polarisations averaged
        a0, var0, _ = stats(z["stacks"][:, idx[0]] / (nfft * P)); a1, var1, _ = stats(z["stacks"][:, idx[1]] / (nfft * P))
        out[int(m["freq_id"])] = dict(ch=int(m["channel"]), a=a, var=var, apol=(a0, a1), varpol=(var0, var1), n=n, f=float(m["freq_mhz"]), t0=m["time0_ctime"] + m["j0"] * sample_seconds)
    fids = sorted(out); fr = np.array([out[f]["f"] for f in fids]); aa = np.array([out[f]["a"] for f in fids])
    ap = [np.array([out[f]["apol"][p] for f in fids]) for p in (0, 1)]
    for i, f in enumerate(fids):                                                            # sky reference per bin (per level kind)
        near = np.abs(fr - fr[i]) <= 40.0; ref = float(np.nanpercentile(aa[near], 10))
        out[f]["x"] = out[f]["a"] - ref; out[f]["ref"] = ref
        out[f]["xpol"] = tuple(out[f]["apol"][p] - float(np.nanpercentile(ap[p][near], 10)) for p in (0, 1))
    return out


def level(ep, ch, binset, pol=None, *, markers):
    fids = [f for f in ep if ep[f]["ch"] == ch]
    if binset == "pilot": fids = [f for f in fids if f == markers.get(ch)]
    if not fids: return np.nan, np.nan
    if pol is None: xs = np.array([ep[f]["x"] for f in fids]); vs = np.array([ep[f]["var"] for f in fids])
    else: xs = np.array([ep[f]["xpol"][pol] for f in fids]); vs = np.array([ep[f]["varpol"][pol] for f in fids])
    if binset == "pilot": return float(xs[0]), float(vs[0])
    j = int(np.argsort(xs)[len(xs) // 2]); return float(np.median(xs)), float(vs[j])      # median bin's noise


def measure(X, labels, t, ch, binset, trim, record_struct, *, srows, sidereal, cap):
    """Structure function on the epoch levels X (label -> (x, var)); trim = upper percentile of x to drop, or None."""
    use = [l for l in labels if np.isfinite(X[l][0]) and np.isfinite(X[l][1])]
    if trim is not None and len(use) > 2:
        cut = np.percentile([X[l][0] for l in use], trim); use = [l for l in use if X[l][0] <= cut]
    pairs = []                                                                              # (lag, D, noise term)
    for i in range(len(use)):
        for j in range(i + 1, len(use)):
            a, b = use[i], use[j]; dt = abs(t[b] - t[a])
            (xa, va), (xb, vb) = X[a], X[b]
            if dt >= sidereal: continue
            pairs.append((dt, 0.5 * ((xa - xb) ** 2 - va - vb), 0.5 * (va + vb)))
    if not pairs: return dict(status="no epochs", tau=np.nan, tau_hi=np.nan, plateau=np.nan, plateau_se=np.nan, n_pairs=0, n_classes=0, n_epochs=len(use))
    lag = np.array([p[0] for p in pairs]); D = np.array([p[1] for p in pairs]); Nz = np.array([p[2] for p in pairs])
    cls = np.array([min(LAG_CLASSES, key=lambda L: abs(np.log(max(l, 1) / L))) for l in lag])
    plat_m = lag > PLATEAU_START
    plateau = float(D[plat_m].mean()) if plat_m.any() else np.nan
    plateau_se = float(D[plat_m].std(ddof=1) / np.sqrt(plat_m.sum())) if plat_m.sum() > 1 else (float(Nz[plat_m].mean()) if plat_m.any() else np.nan)
    centers, values = [], []
    for L in LAG_CLASSES:
        m = cls == L
        if m.any() and L < PLATEAU_START:
            centers.append(L); values.append(float(D[m].mean()))
            if record_struct: srows.append(dict(channel=ch, bins=binset, lag_class=L, D=float(D[m].mean()), D_noise=float(Nz[m].mean()), n_pairs=int(m.sum()), plateau=plateau))
    centers = np.array(centers, float); values = np.array(values, float)
    tau = tau_hi = np.nan; status = ""
    if not np.isfinite(plateau): status = "no plateau lags"
    elif plateau <= 2 * plateau_se: status = "constant"; tau = tau_hi = cap
    elif centers.size < MIN_CLASSES: status = "no cadence lags"
    else:
        target = CROSS * plateau
        if values.max() < target: status = "bound"; tau = float(centers.max()); tau_hi = PLATEAU_START
        else:
            k = int(np.argmax(values >= target))
            if k == 0: tau = float(centers[0])
            else:
                x0, x1, y0, y1 = centers[k - 1], centers[k], values[k - 1], values[k]
                tau = float(x1 if y1 == y0 else x0 + (target - y0) * (x1 - x0) / (y1 - y0))
            tau_hi = tau; status = "measured"
    return dict(status=status, tau=tau, tau_hi=tau_hi, plateau=plateau, plateau_se=plateau_se, n_pairs=len(pairs), n_classes=int(centers.size), n_epochs=len(use))


def tau_main(argv):
    # amendment 5 options: --level polavg|polmax, --pol-table <table_of_record csv> (per-channel polarisation with the larger A),
    # --trim none|archive (archive: primary 90th-percentile trim of the epoch levels, probes 75/90/95, spread gate 2);
    # the Stokes I copy's --level stokesI
    opts = {"--level": "polavg", "--pol-table": "", "--trim": "none", "--class": "0,1", "--pol": "", "--project": ""}
    args = []
    it = iter(argv)
    for a in it:
        if a in ("-h", "--help"): print(__doc__); return 0
        if a in opts: opts[a] = next(it)
        else: args.append(a)
    if opts["--level"] not in LEVELS or opts["--trim"] not in ("none", "archive") or len(args) < 2:
        print(__doc__.split("\n\n")[-1], file=sys.stderr); return 2
    pv = _profile_values(resolve_project(opts["--project"] or None))
    markers, cap, T_FRAME = pv["markers"], pv["cap"], pv["frame_seconds"]
    out_csv = args[0]
    POL_OF = {}
    if opts["--level"] == "polmax" and opts["--pol-table"]:
        for r in csv.DictReader(open(opts["--pol-table"])):
            if r["role"] == "pilot" and r["A_keep_all_ns1_pol0"] and r["A_keep_all_ns1_pol1"]:
                POL_OF[int(r["channel"])] = 1 if float(r["A_keep_all_ns1_pol1"]) > float(r["A_keep_all_ns1_pol0"]) else 0
            elif r["role"] == "pilot" and r["A_keep_all_ns1_pol1"]: POL_OF[int(r["channel"])] = 1
            elif r["role"] == "pilot": POL_OF[int(r["channel"])] = 0
    CLS = tuple(int(x) for x in opts["--class"].split(","))   # (EW cylinder step, NS feed step) of the baseline class
    TRIM_PROBES = (75.0, 90.0, 95.0) if opts["--trim"] == "archive" else (None,)
    PRIMARY_TRIM = 90.0 if opts["--trim"] == "archive" else None
    epochs = {}
    for arg in args[1:]:
        lbl, d = arg.split("=", 1); epochs[lbl] = load_epoch(d, CLS, nfft=pv["nfft"], sample_seconds=pv["sample_seconds"])
    labels = list(epochs); t = {l: float(np.median([v["t0"] for v in epochs[l].values()])) for l in labels}
    chans = sorted(set(v["ch"] for e in epochs.values() for v in e.values()))
    rows, srows = [], []
    for ch in chans:
        pol = POL_OF.get(ch) if opts["--level"] == "polmax" else None
        if opts["--pol"] != "":                       # amendment 11: force the polarisation the ruling prices on this class
            pol = int(opts["--pol"])
        elif opts["--level"] == "polmax" and ch not in POL_OF:
            med = [np.nanmedian([level(epochs[l], ch, "inband", p, markers=markers)[0] for l in labels]) for p in (0, 1)]
            pol = int(np.nanargmax(med)) if np.isfinite(med).any() else 0
        for binset in ("pilot", "inband"):
            X = {l: level(epochs[l], ch, binset, pol, markers=markers) for l in labels}
            probes = {tp: measure(X, labels, t, ch, binset, tp, record_struct=(tp == PRIMARY_TRIM), srows=srows, sidereal=pv["sidereal"], cap=cap) for tp in TRIM_PROBES}
            main = probes[PRIMARY_TRIM]
            status = main["status"]; tau = main["tau"]; tau_hi = main["tau_hi"]
            if opts["--trim"] == "archive":
                taus = [p["tau"] for p in probes.values() if np.isfinite(p["tau"])]
                stats_ = {p["status"] for p in probes.values()}
                if len(taus) == len(probes) and max(taus) / max(min(taus), 1e-9) > MAX_TRIM_SPREAD and not stats_ <= {"bound"}:
                    status = "refused: trim spread %.1f" % (max(taus) / min(taus)); tau = tau_hi = cap
                elif len(taus) < len(probes) and "constant" in stats_ and "measured" in stats_:
                    status = "refused: probes disagree (" + ", ".join(sorted(stats_)) + ")"; tau = tau_hi = cap
            G = min(tau, cap) / T_FRAME if np.isfinite(tau) else np.nan
            rows.append(dict(channel=ch, bins=binset, pol=(("I" if opts["--level"] == "stokesI" else "avg") if pol is None else pol), status=status, tau_c_s=f"{tau:.1f}" if np.isfinite(tau) else "", tau_c_high_s=f"{tau_hi:.1f}" if np.isfinite(tau_hi) else "",
                             G=f"{G:.1f}" if np.isfinite(G) else "", plateau=f"{main['plateau']:.3e}" if np.isfinite(main['plateau']) else "", plateau_se=f"{main['plateau_se']:.3e}" if np.isfinite(main['plateau_se']) else "",
                             n_pairs=main["n_pairs"], n_classes=main["n_classes"], n_epochs=main["n_epochs"],
                             probes=" | ".join(f"{('none' if tp is None else int(tp))}: {p['status']} {p['tau']:.0f}" if np.isfinite(p['tau']) else f"{('none' if tp is None else int(tp))}: {p['status']}" for tp, p in probes.items())))
            continue
    with open(out_csv, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out_csv[:-4] + "_structure.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=["channel", "bins", "lag_class", "D", "D_noise", "n_pairs", "plateau"]); w.writeheader(); w.writerows(srows)
    print(f"{len(labels)} epochs (class {CLS}, {opts['--level']}, trim {opts['--trim']}): " + ", ".join(f"{l} @ {t[l]:.0f}" for l in labels))
    print("ch  bins   pol  status                        tau_c      tau_hi      G          plateau     s.e.     pairs classes epochs   probes")
    for r in rows: print(f"{r['channel']:2d}  {r['bins']:6s} {str(r['pol']):>3s}  {r['status']:28s}  {r['tau_c_s']:>9s}  {r['tau_c_high_s']:>9s}  {r['G']:>10s}  {r['plateau']:>9s}  {r['plateau_se']:>9s}  {r['n_pairs']:4d}  {r['n_classes']:2d}  {r['n_epochs']:2d}   {r['probes']}")
    return 0


# ---------------------------------------------------------------------------------------------------- lags
def load_dump(d, CLASSES, *, nfft, sample_seconds, reference_bands):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=lambda p: int(os.path.basename(p)[:-4])):
        z = np.load(f); m = json.loads(str(z["meta"])); k = z["keys"]
        P = z["autos"][:, z["autos"].mean(axis=0) > 0.5].mean(axis=1)
        V = {}
        for name, (ew, ns) in CLASSES.items():
            idx = [int(np.where((k[:, 0] == ew) & (k[:, 1] == ns) & (k[:, 2] == p) & (k[:, 3] == p))[0][0]) for p in (0, 1)]
            V[name] = z["stacks"][:, idx].mean(axis=1) / (nfft * P)      # [frames], both pols averaged, in A units
        out[m["freq_id"]] = dict(ch=m["channel"], V=V, f=m["freq_mhz"], t0=m["time0_ctime"] + m["j0"] * sample_seconds, n=m["n_frames"])
    # reference phasor per class: mean over the reference bands' bins (no emitter), subtracted from every bin
    for name in CLASSES:
        ref = [v["V"][name].mean() for v in out.values() if v["ch"] in reference_bands]
        r = np.mean(ref) if ref else 0.0
        for v in out.values(): v["V"][name + "_x"] = v["V"][name] - r
    return out


def lags_main(argv):
    ap = argparse.ArgumentParser(prog="pilot-proxy capture cadence lags", description="Phasor coherence between dumps.")
    ap.add_argument("--classes", choices=sorted(LAG_CLASS_SETS), required=True)
    ap.add_argument("--reference-bands", required=True, help="comma-separated bands with no emitter in the capture")
    ap.add_argument("--project", default=None)
    ap.add_argument("out_csv")
    ap.add_argument("dumps", nargs="+", metavar="label=dir")
    a_ = ap.parse_args(argv)
    pv = _profile_values(resolve_project(a_.project))
    markers = pv["markers"]
    CLASSES = LAG_CLASS_SETS[a_.classes]
    reference_bands = tuple(int(b) for b in a_.reference_bands.split(","))
    order = ("ns1", "ns1_x", "ns8_x", "ew1_x") if a_.classes == "three" else [c + "_x" for c in CLASSES] + ["ns1"]
    dumps = {}
    for arg in a_.dumps:
        lbl, d = arg.split("=", 1); dumps[lbl] = load_dump(d, CLASSES, nfft=pv["nfft"], sample_seconds=pv["sample_seconds"], reference_bands=reference_bands)
    labels = list(dumps); t = {l: np.median([v["t0"] for v in dumps[l].values()]) for l in labels}
    chans = sorted(set(v["ch"] for d in dumps.values() for v in d.values()))
    rows = []
    for ch in chans:
        for cls in order:
            for binset in ("pilot", "inband"):
                for a_i in range(len(labels)):
                    for b_i in range(a_i + 1, len(labels)):
                        a, b = labels[a_i], labels[b_i]; dt = abs(t[b] - t[a])
                        fids = [f for f in dumps[a] if dumps[a][f]["ch"] == ch and f in dumps[b]]
                        if binset == "pilot": fids = [f for f in fids if f == markers.get(ch)]
                        if not fids: continue
                        rhos = []; floors = []
                        for f in fids:
                            Va, Vb = dumps[a][f]["V"][cls], dumps[b][f]["V"][cls]
                            ma, mb = Va.mean(), Vb.mean()
                            if abs(ma) == 0 or abs(mb) == 0: continue
                            rho = (ma * np.conj(mb)).real / (abs(ma) * abs(mb))
                            # noise floor: scatter of the per-frame phasors around the mean, relative to the mean
                            fl = np.sqrt((abs(Va - ma) ** 2).mean() / Va.size + (abs(Vb - mb) ** 2).mean() / Vb.size) / np.sqrt(abs(ma) * abs(mb))
                            rhos.append(rho); floors.append(fl)
                        if not rhos: continue
                        lag_class = min(LAG_CLASSES, key=lambda L: abs(np.log(max(dt, 1) / L)))
                        rows.append(dict(channel=ch, cls=cls, bins=binset, a=a, b=b, lag_s=round(dt, 1), lag_class=lag_class, rho=float(np.median(rhos)), rho_noise=float(np.median(floors)), n_bins=len(rhos)))
    with open(a_.out_csv, "w") as fh:
        cols = list(rows[0].keys()) if rows else ["channel"]; fh.write(",".join(cols) + "\n")
        for r in rows: fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"{len(rows)} pairs -> {a_.out_csv}")
    if rows:
        for cls in order:
            print(f"\nin-band coherence between dumps, class {cls} (x = reference-subtracted excess): ch | lag class s: rho (noise)")
            for ch in chans:
                rr = [r for r in rows if r["channel"] == ch and r["bins"] == "inband" and r["cls"] == cls]
                by = {}
                for r in rr: by.setdefault(r["lag_class"], []).append(r)
                print(f"  {ch:2d} | " + "  ".join(f"{L}s: {np.mean([x['rho'] for x in v]):+.2f} ({np.mean([x['rho_noise'] for x in v]):.2f})" for L, v in sorted(by.items())))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("tau", "lags"):
        print(__doc__.split("\n\n")[-1], file=sys.stderr)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return tau_main(argv[1:]) if argv[0] == "tau" else lags_main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
