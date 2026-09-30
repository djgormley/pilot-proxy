"""The capture operating-characteristic handoff (``capture_operating_characteristic_v1``).

What the science side reads from the capture, and nothing else from it: the
detector's readings of each band on the matched capture, in four files.

- ``capture_oc_table.csv``: one row per band x baseline class x product x
  bank x policy x event. An event row is one dump's class excess (the in-band
  median over the band's bins of the noise-bias-free coherent amplitude less
  the sky reference), its frame count, its placement on the band's archive
  ladder (the dump's median Q and whether the policy keeps it) and the
  admissible minimum coherence gain at its frame count. The ``kept`` row of
  a policy aggregates the dumps the policy keeps: the median excess A, the
  control band's level (mean, sample standard deviation and count over its
  bins in the science dumps), the net amplitude A_net = A - mean, the
  amplitude test (A > mean + k sd, k = ``detection_gate_sigma``) and the
  power form A_net^2 N n_b with its gate. Every row of a band x class x
  product repeats the cadence coherence time of that product's level file
  (status, tau, the least of its trim probes, the zero-frequency gain) and
  the phasor coherence gamma between the science dumps.
- ``capture_oc_summary.csv``: one row per band of the plan: its bins and
  their roles, the archive ladder per bank, the frozen archive gain bound of
  the channel ruling (``board_final.csv``) and the frame-residual readings of
  the science dumps (the detector columns of the table of record).
- ``capture_tau_bounds.csv``: the within-dump lower bounds on tau per band x
  class x setting (:mod:`pilot_proxy.capture.tau_bounds`).
- ``manifest.json``: schema, file digests, input digests, the integration
  model, the register values used, the null policy (no band has an eta_Pfa
  on the capture) and the bands.

Products: ``xx`` and ``yy`` (the same-polarisation class stacks, pol 0 and
1), ``stokes_i`` (their complex mean) and ``polmax`` (the per-band level
file's own polarisation; cadence readings only, one row per band and class).
Banks: ``nominal`` (the archive's detector bank) and ``measured_marker`` (a
band rescanned with a bank built on its measured marker; the ladder and the
dump placements are the rescan's).

Floats are written as Python ``repr`` (round-trip exact); NaN and None as an
empty cell; strings the capture files carry (tau statuses, probe text) are
copied verbatim. No tolerance, credit, gain model, range rule or disposition
is computed here.

usage: pilot-proxy capture oc-table --capture-dir DIR --stokes-dir DIR --lags CSV --detector-runs DIR
           --datasets DIR --archive-products DIR --tau-bounds CSV --board CSV --out DIR
           [--measured-marker-rescan DIR] [--project DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

from pilot_proxy.characterization.coarse_ladder import place_dumps, product_thresholds, run_thresholds
from pilot_proxy.characterization.coherence import admissible_min_G
from pilot_proxy.characterization.oc_table import canonical_json, check_table, read_rows, write_rows

from . import units
from .markers import marker_freq_ids, resolve_project

SCHEMA_NAME = "capture_operating_characteristic"
SCHEMA_VERSION = 1
SCHEMA_ID = f"{SCHEMA_NAME}_v{SCHEMA_VERSION}"
SCHEMA_FILE = Path(__file__).resolve().parent / "schemas" / f"{SCHEMA_ID}.schema.json"
TABLE, SUMMARY, TAU_BOUNDS, MANIFEST = ("capture_oc_table.csv", "capture_oc_summary.csv", "capture_tau_bounds.csv",
                                        "manifest.json")
CLASSES = ((0, 1), (0, 8), (0, 32), (0, 64), (0, 128), (0, 255), (1, 0), (1, 32), (2, 0), (3, 0))
PRODUCTS = ("xx", "yy", "stokes_i", "polmax")
POL_OF_PRODUCT = {"xx": "0", "yy": "1", "stokes_i": "I"}
BANKS = ("nominal", "measured_marker")
POLICIES = ("cal_q0.1", "cal_q0.5", "cal_q0.9", "keep_all")
QS = {"cal_q0.1": 0.1, "cal_q0.5": 0.5, "cal_q0.9": 0.9}
KEPT = "kept"
BIN_ROLES = ("marker", "data", "control_target")
ETA_PFA_SCREENED = "unavailable: no verified signal-free population of the band on the capture"
ETA_PFA_CONTROL = "not defined: control band"
PROBE_VALUES = re.compile(r"(?:measured|bound|constant)\s+([\d.]+)")

TABLE_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("band_id", "str", "", "Band label from the frequency plan (the join key, with the edges)."),
    ("low_mhz", "float", "MHz", "Lower band edge from the plan."),
    ("high_mhz", "float", "MHz", "Upper band edge from the plan."),
    ("role", "str", "", "screened or control (the plan's role)."),
    ("class_ew", "int", "cylinders", "East-west step of the redundant baseline class."),
    ("class_ns", "int", "feeds", "North-south step of the redundant baseline class."),
    ("class_name", "str", "", "ns<k>, ew<k> or ew<k>ns<l>."),
    ("product", "str", "", "xx, yy, stokes_i (the class stacks) or polmax (cadence readings of the per-band level "
                           "file only)."),
    ("bank", "str", "", "nominal (the archive's detector bank) or measured_marker (the band's rescan)."),
    ("policy", "str", "", "The archive ladder rung: cal_q0.1, cal_q0.5, cal_q0.9 (Q = F/mu0 at or below the rung's "
                          "eta) or keep_all."),
    ("event", "str", "", "A dump's event id, or 'kept' for the aggregate over the dumps the policy keeps."),
    ("science", "bool", "", "Event rows: the dump is a science dump (the ladder is read on those only)."),
    ("excess", "float", "amplitude", "Event rows: the dump's class excess, the in-band median over the band's bins "
                                     "(empty where no bin has a finite reading)."),
    ("n_bins", "int", "bins", "Event rows: bins with a finite excess."),
    ("frames", "int", "frames", "Event rows: frames of the dump (the products' stacks axis 0)."),
    ("q_median", "float", "", "Event rows: median Q = F/mu0 of the dump's valid frames at the band's marker bin "
                              "(science dumps with a detector reading on the bank)."),
    ("kept", "bool", "", "Event rows of science dumps: the policy keeps the dump (q_median <= eta; keep_all keeps "
                         "every science dump)."),
    ("eta", "float", "", "The rung's threshold on Q: the q-quantile (method higher) of Q over every valid, finite "
                         "frame of the band's archive product (population all_valid_frames), or of the rescan on "
                         "the measured_marker bank. Empty on keep_all."),
    ("G_admissible_min", "float", "", "Event rows: the least coherence gain any tau allows for a reading over the "
                                      "dump's frames (the lower of the exponential and block models)."),
    ("ladder_evaluable", "bool", "", "The band has a ladder on the bank and a detector reading of at least one "
                                     "science dump."),
    ("dumps_kept", "str", "", "Kept rows: the science dumps the policy keeps, space-joined event ids."),
    ("n_dumps", "int", "dumps", "Kept rows: how many."),
    ("A", "float", "amplitude", "Kept rows: median excess over the kept dumps with a reading."),
    ("control_mean", "float", "amplitude", "The control band's level on the class and product: the mean of its "
                                           "per-bin excess over the science dumps."),
    ("control_sd", "float", "amplitude", "Its sample standard deviation (ddof 1)."),
    ("control_n", "int", "bin-dumps", "Its count."),
    ("A_net", "float", "amplitude", "Kept rows: A - control_mean."),
    ("measured", "bool", "", "Kept rows: A > control_mean + k control_sd (the amplitude test), k = "
                             "detection_gate_sigma."),
    ("gate", "float", "amplitude", "k control_sd, the detection gate on A_net."),
    ("n_b", "int", "baselines", "Redundant baseline products averaged into the class stack (the products' count)."),
    ("n_frame", "int", "samples", "N, the samples per frame."),
    ("x_power", "float", "", "Kept rows: the per-frame covariance ratio A_net^2 N n_b (A_net itself where it is "
                             "not positive)."),
    ("gate_power", "float", "", "The gate in the same form, gate^2 N n_b."),
    ("tau_status", "str", "", "The product's cadence coherence time on the class, in-band: measured, bound, "
                              "constant, refused: <reason> or another status of the estimator (verbatim)."),
    ("tau_c_s", "float", "s", "Its tau_c (as the level file writes it, one decimal)."),
    ("tau_c_high_s", "float", "s", "Its upper end."),
    ("tau_G", "float", "", "The level file's G (one decimal)."),
    ("tau_probes", "str", "", "The trim probes, verbatim ('75: measured 144 | 90: ...')."),
    ("tau_pol", "str", "", "The level file's pol column (0, 1, avg or I)."),
    ("tau_least_s", "float", "s", "The least of the probes' values and tau_c (the probes alone where tau_c is "
                                  "empty)."),
    ("G_zero", "float", "", "min(tau_c, cap) / T_frame (empty where tau_c is)."),
    ("gamma", "float", "", "Phasor coherence of the class between the science dumps: the median over the dump "
                           "pairs of the in-band reference-subtracted rho."),
    ("gamma_n", "int", "pairs", "Pairs in that median."),
)

_BANK_RUNG = [(f"eta_{p}_{b}", "float", "", f"The {p} rung's threshold on the {b} bank.")
              for b in BANKS for p in QS]
SUMMARY_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("band_id", "str", "", "Band label (join key)."),
    ("low_mhz", "float", "MHz", "Lower band edge."),
    ("high_mhz", "float", "MHz", "Upper band edge."),
    ("role", "str", "", "screened or control."),
    ("target_freq_id", "int", "", "The coarse channel the band is read at: its marker channel, or the control "
                                  "band's declared target (the nominal marker position)."),
    ("bins", "str", "", "Every coarse channel of the band as freq_id:bin_role, ';'-joined (roles marker, data, "
                        "control_target)."),
    ("in_capture", "bool", "", "The capture products hold the band."),
    ("n_science_dumps", "int", "dumps", "Science dumps of the capture."),
    *_BANK_RUNG,
    ("eta_frames_nominal", "int", "frames", "Frames the nominal ladder is read over."),
    ("eta_frames_measured_marker", "int", "frames", "Frames the rescan ladder is read over."),
    ("ladder_evaluable_nominal", "bool", "", "Ladder and a science-dump reading on the nominal bank."),
    ("ladder_evaluable_measured_marker", "bool", "", "The same on the rescan bank."),
    *[(f"kept_dumps_{p}_{b}", "str", "", f"Science dumps {p} keeps on the {b} bank, space-joined labels.")
      for b in BANKS for p in POLICIES],
    ("board_G_nocredit", "float", "", "The channel ruling's archive gain bound (board_final.csv G_nocredit, "
                                      "frozen 2026-09-14)."),
    ("board_tau_quality", "str", "", "Its tau quality (verbatim)."),
    ("board_tau_booked_s", "str", "", "Its booked tau (verbatim)."),
    ("frame_n_dumps", "int", "dumps", "Science dumps with a frame-residual reading."),
    ("frame_A_ns1", "float", "amplitude", "Frame residual, keep_all: median over the dumps of the in-band median "
                                          "excess on the 0.3 m class, the larger polarisation per dump."),
    ("frame_A_ns1_pol0", "float", "amplitude", "The same on pol 0."),
    ("frame_A_ns1_pol1", "float", "amplitude", "The same on pol 1."),
    ("frame_A_class_median", "float", "amplitude", "Median over the dumps of the median over every class and "
                                                   "polarisation."),
    ("frame_G_dump", "float", "", "Median over the dumps of the marker bin's within-dump gain on the polarisation "
                                  "that sets A."),
    ("frame_A_ew", "float", "amplitude", "Median over the dumps of the median over the three east-west classes of "
                                         "the larger-polarisation reading."),
    ("frame_A_lowest_epoch", "float", "amplitude", "The least 0.3 m in-band excess over every dump on the "
                                                   "polarisation that sets A."),
    ("frame_lowest_epoch", "str", "", "Its event."),
    ("frame_A_ew_lowest_epoch", "float", "amplitude", "The least east-west reading over every dump."),
    ("frame_ew_lowest_epoch", "str", "", "Its event."),
    ("eta_pfa_status", "str", "", "unavailable: <reason> for a screened band, not defined: control band."),
)


def column_schema(columns) -> list[dict]:
    return [{"name": n, "type": t, "unit": u, "description": d} for n, t, u, d in columns]


def class_name(c) -> str:
    ew, ns = c
    return f"ns{ns}" if ew == 0 else (f"ew{ew}" if ns == 0 else f"ew{ew}ns{ns}")


def level_file(c) -> str:
    """The cadence level file of a class ('cadence_tau.csv' for the 0.3 m class, 'cadence_tau_<name>.csv')."""
    return "cadence_tau.csv" if c == (0, 1) else f"cadence_tau_{class_name(c)}.csv"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Inputs:
    """Every input read, with its sha256, for the manifest."""

    def __init__(self):
        self.files: dict[str, str] = {}

    def open(self, path):
        path = os.path.abspath(path)
        if path not in self.files:
            self.files[path] = sha256_file(path)
        return open(path)

    def load(self, path, **kw):
        path = os.path.abspath(path)
        if path not in self.files:
            self.files[path] = sha256_file(path)
        return np.load(path, **kw)

    def note(self, path):
        path = os.path.abspath(path)
        if path not in self.files:
            self.files[path] = sha256_file(path)
        return path


def _f(s):
    return float(s) if s not in ("", None) else None


def read_excess(directory, inputs):
    """{(band, epoch, class, pol): (excess or None, n_bins)} and the epochs in file order."""
    out, epochs = {}, []
    with inputs.open(os.path.join(directory, "class_excess_epochs.csv")) as fh:
        for r in csv.DictReader(fh):
            out[(int(r["channel"]), r["epoch"], (int(r["ew"]), int(r["ns"])), r["pol"])] = (_f(r["excess"]), int(r["n_bins"]))
            if r["epoch"] not in epochs:
                epochs.append(r["epoch"])
    return out, epochs


def read_control(directory, science, inputs):
    """{(class, pol): (mean, sample sd, n)} over the control band's bins of the science dumps."""
    fl = {}
    with inputs.open(os.path.join(directory, "class_floor_bins.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["epoch"] in science:
                fl.setdefault(((int(r["ew"]), int(r["ns"])), r["pol"]), []).append(float(r["excess"]))
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)), len(v)) for k, v in fl.items()}


def read_levels(path, inputs):
    """{band: in-band row} of one cadence level file, or {} when it does not exist."""
    if not os.path.exists(path):
        return {}
    with inputs.open(path) as fh:
        return {int(r["channel"]): r for r in csv.DictReader(fh) if r["bins"] == "inband"}


def tau_reading(row, cap, frame_seconds) -> dict:
    """The cadence columns of one level-file row, and the least of its trim probes (amendment 10 item 7)."""
    if row is None:
        return {}
    vals = [float(m) for m in PROBE_VALUES.findall(row.get("probes", "") or "")]
    tl = min(vals) if vals else np.nan
    tr = float(row["tau_c_s"]) if row["tau_c_s"] else np.nan
    least = min(tl, tr) if (np.isfinite(tl) and np.isfinite(tr)) else tl
    return {"tau_status": row["status"], "tau_c_s": _f(row["tau_c_s"]), "tau_c_high_s": _f(row["tau_c_high_s"]),
            "tau_G": _f(row["G"]), "tau_probes": row.get("probes", ""), "tau_pol": row["pol"],
            "tau_least_s": float(least) if np.isfinite(least) else None,
            "G_zero": (min(tr, cap) / frame_seconds) if np.isfinite(tr) else None}


def read_gamma(path, inputs):
    """{(band, class name): (median rho, n)} over the in-band reference-subtracted rows."""
    v = {}
    with inputs.open(path) as fh:
        for r in csv.DictReader(fh):
            if r["bins"] == "inband" and r["cls"].endswith("_x"):
                v.setdefault((int(r["channel"]), r["cls"][:-2]), []).append(float(r["rho"]))
    return {k: (float(np.median(x)), len(x)) for k, x in v.items()}


def _frames_and_counts(datasets, events, record, inputs):
    frames, n_b = {}, {}
    for ev in events:
        p = units.first_product(record.dataset_dir(datasets, ev))
        z = inputs.load(p)
        frames[ev] = units.frames_per_dump(z)
        if not n_b:
            for c in CLASSES:
                b0, b1 = units.class_baseline_count(z, c, 0), units.class_baseline_count(z, c, 1)
                if b0 != b1:
                    raise ValueError(f"{p}: class {c} counts differ between polarisations")
                n_b[c] = b0
    return frames, n_b


def _ladders(args, project, markers, record, science, inputs):
    """eta[bank][band] and q_median[bank][(band, event)]."""
    eta = {b: {} for b in BANKS}; qmed = {b: {} for b in BANKS}
    for ch, fid in markers.items():
        p = os.path.join(args.archive_products, f"{fid}.npz")
        if not os.path.exists(p):
            continue
        inputs.note(p)
        full = product_thresholds(p, tuple(QS.values()))
        eta["nominal"][ch] = full
    for ev in science:
        path = os.path.join(args.detector_runs, record.DETECTOR_RUNS["nominal"].format(event=ev), "chime_detector_outputs.npz")
        for ch, r in place_dumps(inputs.note(path)).items():
            qmed["nominal"][(ch, ev)] = r["q_median"]
    rescanned = [int(b) for b in record.MEASURED_MARKER_BANDS]
    if args.measured_marker_rescan:
        rp = inputs.note(os.path.join(args.measured_marker_rescan, "chime_detector_outputs.npz"))
        for ch in rescanned:
            eta["measured_marker"][ch] = run_thresholds(rp, tuple(QS.values()))
        for ev in science:
            path = os.path.join(args.detector_runs, record.DETECTOR_RUNS["measured_marker"].format(event=ev),
                                "chime_detector_outputs.npz")
            for ch, r in place_dumps(inputs.note(path)).items():
                if ch in rescanned:
                    qmed["measured_marker"][(ch, ev)] = r["q_median"]
    return eta, qmed, (rescanned if args.measured_marker_rescan else [])


def kept_dumps(ch, policy, science, eta, qmed):
    """The science dumps a policy keeps on one band (the ruling's reading): every one under keep_all; under a rung,
    those whose q_median is at or below its eta; none without a ladder."""
    if policy == "keep_all": return list(science)
    if ch not in eta: return []
    return [ev for ev in science if (ch, ev) in qmed and qmed[(ch, ev)] <= eta[ch][QS[policy]]]


def table_rows(*, project, excess, epochs, control, levels, gamma, frames, n_b, eta, qmed, rescanned, science, k_sigma):
    nfft = int(project.detector_config.nfft)
    cap = project.integration_model.coherence_cap_seconds; tf = project.integration_model.frame_seconds
    gmin = {n: admissible_min_G(n) for n in sorted(set(frames.values()))}
    rows = []
    present = {(ch, c, p) for (ch, _e, c, p) in excess}
    for band in project.frequency_plan.bands():
        ch = int(band.label)
        base = {"band_id": band.label, "low_mhz": float(band.low_mhz), "high_mhz": float(band.high_mhz), "role": band.role}
        for c in CLASSES:
            cbase = dict(base, class_ew=c[0], class_ns=c[1], class_name=class_name(c))
            g = gamma.get((ch, class_name(c)))
            gcols = {"gamma": g[0], "gamma_n": g[1]} if g else {}
            for product in PRODUCTS:
                tau = tau_reading(levels[product].get(c, {}).get(ch), cap, tf)
                if product == "polmax":
                    if tau:
                        rows.append(dict(cbase, product=product, **tau, **gcols))
                    continue
                pol = POL_OF_PRODUCT[product]
                if (ch, c, pol) not in present:
                    continue
                ctl = control.get(product, {}).get((c, pol))
                ccols = {"control_mean": ctl[0], "control_sd": ctl[1], "control_n": ctl[2]} if ctl else {}
                for bank in BANKS:
                    if bank == "measured_marker" and ch not in rescanned:
                        continue
                    e_, q_ = eta[bank], qmed[bank]
                    evaluable = ch in e_ and any((ch, ev) in q_ for ev in science)
                    for policy in POLICIES:
                        rung = e_[ch][QS[policy]] if (policy != "keep_all" and ch in e_) else None
                        keep = kept_dumps(ch, policy, science, e_, q_)
                        common = dict(cbase, product=product, bank=bank, policy=policy, eta=rung,
                                      ladder_evaluable=evaluable, n_b=n_b[c], n_frame=nfft, **ccols, **tau, **gcols)
                        if ctl:
                            common["gate"] = k_sigma * ctl[1]
                            common["gate_power"] = units.gate_power(k_sigma * ctl[1], n_b[c], nfft)
                        for ev in epochs:
                            x, nb_ = excess.get((ch, ev, c, pol), (None, None))
                            if nb_ is None:
                                continue
                            sci = ev in science
                            rows.append(dict(common, event=ev, science=sci, excess=x, n_bins=nb_, frames=frames[ev],
                                             q_median=q_.get((ch, ev)) if sci else None,
                                             kept=(ev in keep) if sci else None,
                                             G_admissible_min=gmin[frames[ev]]))
                        vals = [excess[(ch, ev, c, pol)][0] for ev in keep
                                if (ch, ev, c, pol) in excess and excess[(ch, ev, c, pol)][0] is not None]
                        agg = dict(common, event=KEPT, dumps_kept=" ".join(keep), n_dumps=len(keep))
                        if vals and ctl:
                            A = float(np.median(vals)); fm, fs, _ = ctl
                            agg.update(A=A, A_net=A - fm, measured=bool(A > fm + k_sigma * fs),
                                       x_power=units.to_power(A - fm, n_b[c], nfft))
                        elif vals:
                            agg.update(A=float(np.median(vals)))
                        rows.append(agg)
    return rows


# ---------------------------------------------------------------------------------------------------- summary
def _frame_readings(capture_dir, science, all_events, inputs):
    """The frame-residual detector readings of the table of record (table_of_record.py f2372bf6, the detector half)."""
    EW_CLASSES = ((1, 0), (2, 0), (3, 0))
    lowb, lowb_ew = {}, {}
    for ev in all_events:
        fr = os.path.join(capture_dir, f"frame_residual_{ev}.csv")
        if not os.path.exists(fr):
            continue
        percls = {}
        with inputs.open(fr) as fh:
            for r in csv.DictReader(fh):
                if r["frames"] == "all" and r["ew"] == "0" and r["ns"] == "1" and r["pol"] in ("0", "1") and r["excess_inband_median"]:
                    v = float(r["excess_inband_median"])
                    if np.isfinite(v): lowb.setdefault(int(r["channel"]), {}).setdefault(r["pol"], []).append((v, ev))
                if r["frames"] == "all" and (int(r["ew"]), int(r["ns"])) in EW_CLASSES and r["pol"] in ("0", "1") and r["excess_inband_median"]:
                    v = float(r["excess_inband_median"])
                    if np.isfinite(v): percls.setdefault(int(r["channel"]), {}).setdefault((int(r["ew"]), int(r["ns"])), []).append(v)
        for ch, d in percls.items():
            cls_vals = [max(vv) for vv in d.values()]                       # larger polarisation per class
            if cls_vals: lowb_ew.setdefault(ch, []).append((float(np.median(cls_vals)), ev))
    dumps = {}
    for ev in science:
        fr = os.path.join(capture_dir, f"frame_residual_{ev}.csv")
        if not os.path.exists(fr):
            continue
        with inputs.open(fr) as fh:
            allrows = [r for r in csv.DictReader(fh) if r["frames"] == "all" and r["ew"] != "-1"]
        ex = {}; expol = {}; exew = {}; percls = {}
        for r in allrows:
            if (int(r["ew"]), int(r["ns"])) in EW_CLASSES and r["pol"] in ("0", "1") and r["excess_inband_median"]:
                v = float(r["excess_inband_median"])
                if np.isfinite(v): percls.setdefault(int(r["channel"]), {}).setdefault((int(r["ew"]), int(r["ns"])), []).append(v)
        for ch, d in percls.items():
            exew[ch] = float(np.median([max(vv) for vv in d.values()]))
        for r in [r for r in allrows if r["ew"] == "0" and r["ns"] == "1" and r["pol"] in ("0", "1")]:
            ch = int(r["channel"]); vals = (float(r["excess_inband_median"] or "nan"), float(r["excess_pilot_bin"] or "nan"), float(r["G_dump_pilot"] or "nan"), float(r["phi_fast_pilot"] or "nan"))
            expol.setdefault(ch, {})[r["pol"]] = vals
        for ch, pv in expol.items():
            cand = [v for v in pv.values() if np.isfinite(v[0])]
            if cand: ex[ch] = max(cand, key=lambda v: v[0])
            else: ex[ch] = next(iter(pv.values()))
        typ = {}
        for r in allrows:
            v = float(r["excess_inband_median"] or "nan")
            if np.isfinite(v): typ.setdefault(int(r["channel"]), []).append(v)
        typ = {c: float(np.median(v)) for c, v in typ.items()}
        dumps[ev] = dict(ex=ex, typ=typ, expol=expol, exew=exew)
    out = {}
    for ch in sorted(set(c for dmp in dumps.values() for c in dmp["ex"])):
        kept = list(dumps)
        vals = [dumps[e]["ex"][ch][0] for e in kept if ch in dumps[e]["ex"] and np.isfinite(dumps[e]["ex"][ch][0])]
        Gs = [dumps[e]["ex"][ch][2] for e in kept if ch in dumps[e]["ex"] and np.isfinite(dumps[e]["ex"][ch][2])]
        tv = [dumps[e]["typ"][ch] for e in kept if ch in dumps[e]["typ"]]
        Apol = {}
        for pp in ("0", "1"):
            vv = [dumps[e]["expol"][ch][pp][0] for e in kept if ch in dumps[e]["expol"] and pp in dumps[e]["expol"][ch] and np.isfinite(dumps[e]["expol"][ch][pp][0])]
            Apol[pp] = float(np.median(vv)) if vv else np.nan
        bv = [dumps[e]["exew"][ch] for e in kept if ch in dumps[e]["exew"] and np.isfinite(dumps[e]["exew"][ch])]
        polstar = "1" if (np.isfinite(Apol["1"]) and (not np.isfinite(Apol["0"]) or Apol["1"] >= Apol["0"])) else "0"
        lowvals = lowb.get(ch, {}).get(polstar, [])
        A_low, ep_low = (min(lowvals) if lowvals else (np.nan, ""))
        lb = lowb_ew.get(ch, [])
        A_low_b, ep_low_b = (min(lb) if lb else (np.nan, ""))
        out[ch] = {"frame_n_dumps": len(vals), "frame_A_ns1": float(np.median(vals)) if vals else None,
                   "frame_A_ns1_pol0": Apol["0"], "frame_A_ns1_pol1": Apol["1"],
                   "frame_A_class_median": float(np.median(tv)) if tv else None,
                   "frame_G_dump": float(np.median(Gs)) if Gs else None,
                   "frame_A_ew": float(np.median(bv)) if bv else None,
                   "frame_A_lowest_epoch": A_low, "frame_lowest_epoch": ep_low,
                   "frame_A_ew_lowest_epoch": A_low_b, "frame_ew_lowest_epoch": ep_low_b}
    return out


def _bins(project, band):
    inst = project.instrument
    target = int(project.target_freq_id(band))
    out = []
    for fid in range(inst.n_channels):
        if band.contains_hz(inst.hz_of_freq_id(fid)):
            role = ("control_target" if band.role == "control" else "marker") if fid == target else "data"
            out.append((fid, role))
    return sorted(out)


def summary_rows(*, project, capture_dir, science, record, eta, qmed, rescanned, present_bands, board, all_events,
                 inputs):
    frame = _frame_readings(capture_dir, science, all_events, inputs)
    labels = dict(record.SCIENCE_EVENTS)
    rows = []
    for band in project.frequency_plan.bands():
        ch = int(band.label)
        r = {"band_id": band.label, "low_mhz": float(band.low_mhz), "high_mhz": float(band.high_mhz), "role": band.role,
             "target_freq_id": int(project.target_freq_id(band)),
             "bins": ";".join(f"{fid}:{role}" for fid, role in _bins(project, band)),
             "in_capture": ch in present_bands, "n_science_dumps": len(science),
             "eta_pfa_status": ETA_PFA_CONTROL if band.role == "control" else ETA_PFA_SCREENED}
        for bank in BANKS:
            if bank == "measured_marker" and ch not in rescanned:
                continue
            e_, q_ = eta[bank], qmed[bank]
            if ch in e_:
                for p, q in QS.items():
                    r[f"eta_{p}_{bank}"] = e_[ch][q]
                r[f"eta_frames_{bank}"] = e_[ch]["n"]
            r[f"ladder_evaluable_{bank}"] = ch in e_ and any((ch, ev) in q_ for ev in science)
            for p in POLICIES:
                r[f"kept_dumps_{p}_{bank}"] = " ".join(labels[e] for e in kept_dumps(ch, p, science, e_, q_))
        if ch in board:
            r.update(board[ch])
        r.update(frame.get(ch, {}))
        rows.append(r)
    return rows


def read_board(path, inputs):
    out = {}
    with inputs.open(path) as fh:
        for r in csv.DictReader(fh):
            out[int(r["channel"])] = {"board_G_nocredit": float(r["G_nocredit"]), "board_tau_quality": r["tau_quality"],
                                      "board_tau_booked_s": r["tau_booked_s"]}
    return out


# ---------------------------------------------------------------------------------------------------- files
def _clean(rows):
    """Plain Python scalars (numpy floats and bools would not write in repr form)."""
    out = []
    for r in rows:
        o = {}
        for k, v in r.items():
            if isinstance(v, (bool, np.bool_)): v = bool(v)
            elif isinstance(v, (np.integer,)): v = int(v)
            elif isinstance(v, (float, np.floating)):
                v = float(v)
                if not math.isfinite(v): v = None
            o[k] = v
        out.append(o)
    return out


def schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://github.com/WVURAIL/pilot-proxy/docs/schemas/{SCHEMA_ID}.schema.json",
        "title": "Capture operating characteristic handoff (detector to science)",
        "description": __doc__.strip().split("\n\n")[0],
        "type": "object",
        "required": ["schema", "schema_version", "files", "inputs", "integration", "measurement_allowance_db",
                     "detection_gate_sigma", "null_policy", "r_reference", "residual_shape", "bands", "classes",
                     "events", "products", "banks"],
        "properties": {
            "schema": {"const": SCHEMA_NAME},
            "schema_version": {"const": SCHEMA_VERSION},
            "files": {"type": "object", "additionalProperties": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                      "required": [TABLE, SUMMARY, TAU_BOUNDS]},
            "inputs": {"type": "object", "additionalProperties": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
            "integration": {"type": "object", "required": ["frame_seconds", "coherence_cap_seconds", "n_frame"]},
            "measurement_allowance_db": {"type": "number"},
            "detection_gate_sigma": {"type": "number"},
            "null_policy": {"type": "object", "required": ["alpha", "eta_pfa_status"]},
            "r_reference": {"type": "object"},
            "residual_shape": {"type": "string"},
            "bands": {"type": "array", "items": {"type": "object",
                                                  "required": ["band_id", "low_mhz", "high_mhz", "role", "target_freq_id"]}},
            "classes": {"type": "object"},
            "events": {"type": "object"},
            "products": {"type": "object"},
            "banks": {"type": "object"},
            "producer": {"type": "object", "description": "Informational; a consumer never requires it."},
        },
        "x-capture-oc-table-columns": column_schema(TABLE_COLUMNS),
        "x-capture-oc-summary-columns": column_schema(SUMMARY_COLUMNS),
        "x-capture-tau-bounds-columns": [{"name": n} for n in _tau_bound_columns()],
        "x-row-rules": {
            "order": ["band (plan order)", "class", "product", "bank", "policy", "event (file order, then kept)"],
            "classes": [list(c) for c in CLASSES], "products": list(PRODUCTS), "banks": list(BANKS),
            "policies": list(POLICIES), "bin_roles": list(BIN_ROLES),
            "polmax": "cadence readings only: one row per band and class, empty bank, policy and event",
            "kept": "event 'kept' aggregates the policy's kept science dumps; A, A_net, measured and x_power are "
                    "filled only when a kept dump has a reading and the control level exists",
        },
    }


def _tau_bound_columns():
    from .tau_bounds import COLUMNS
    return COLUMNS


def schema_text() -> str:
    return json.dumps(schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_handoff(out_dir, *, table, summary, tau_bounds_path, manifest):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    write_rows(_clean(table), out / TABLE, TABLE_COLUMNS)
    write_rows(_clean(summary), out / SUMMARY, SUMMARY_COLUMNS)
    if Path(tau_bounds_path).resolve() != (out / TAU_BOUNDS).resolve():
        shutil.copyfile(tau_bounds_path, out / TAU_BOUNDS)
    body = dict(manifest); body["schema"] = SCHEMA_NAME; body["schema_version"] = SCHEMA_VERSION
    body["files"] = {name: sha256_file(out / name) for name in (TABLE, SUMMARY, TAU_BOUNDS)}
    (out / MANIFEST).write_text(canonical_json(body), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------------------------------- checks
def check_rows(path) -> list[str]:
    """Row rules of capture_oc_table.csv: vocabularies, the kept aggregate, and the arithmetic of the power form."""
    problems = []
    for number, r in enumerate(read_rows(path), start=2):
        where = f"{TABLE}:{number}"
        if r["product"] not in PRODUCTS: problems.append(f"{where}: product {r['product']!r}")
        if r["product"] == "polmax":
            if r["bank"] or r["policy"] or r["event"]: problems.append(f"{where}: a polmax row carries a bank, policy or event")
            continue
        if r["bank"] not in BANKS: problems.append(f"{where}: bank {r['bank']!r}")
        if r["policy"] not in POLICIES: problems.append(f"{where}: policy {r['policy']!r}")
        if r["policy"] == "keep_all" and r["eta"]: problems.append(f"{where}: eta filled on keep_all")
        if r["ladder_evaluable"] == "True" and r["policy"] != "keep_all" and not r["eta"]:
            problems.append(f"{where}: a rung with a ladder lacks its eta")
        if r["event"] == KEPT:
            if r["excess"] or r["frames"] or r["kept"]: problems.append(f"{where}: a kept row carries event columns")
            n = len(r["dumps_kept"].split()) if r["dumps_kept"] else 0
            if int(r["n_dumps"]) != n: problems.append(f"{where}: n_dumps != dumps listed")
            if r["A_net"] and r["control_mean"] and float(r["A_net"]) != float(r["A"]) - float(r["control_mean"]):
                problems.append(f"{where}: A_net != A - control_mean")
            if r["x_power"] and r["A_net"]:
                a = float(r["A_net"])
                want = a * a * int(r["n_frame"]) * int(r["n_b"]) if a > 0 else a
                if float(r["x_power"]) != want: problems.append(f"{where}: x_power is not the power form of A_net")
        else:
            if r["dumps_kept"] or r["n_dumps"] or r["A"]: problems.append(f"{where}: an event row carries kept columns")
            if (r["science"] == "True") != (r["kept"] != ""): problems.append(f"{where}: kept filled off a science dump")
    return problems


def check_manifest(manifest: Mapping, directory=None) -> list[str]:
    spec = schema()
    problems = [f"manifest: missing {k}" for k in spec["required"] if k not in manifest]
    if manifest.get("schema") != SCHEMA_NAME: problems.append("manifest: schema")
    if manifest.get("schema_version") != SCHEMA_VERSION: problems.append("manifest: schema_version")
    for key in spec["properties"]["files"]["required"]:
        if key not in manifest.get("files", {}): problems.append(f"manifest: files lacks {key}")
    statuses = manifest.get("null_policy", {}).get("eta_pfa_status", {})
    for b in manifest.get("bands", []):
        want = ETA_PFA_CONTROL if b.get("role") == "control" else ETA_PFA_SCREENED
        if statuses.get(b.get("band_id")) != want: problems.append(f"manifest: eta_pfa_status of band {b.get('band_id')}")
    if directory is not None:
        for name, digest in manifest.get("files", {}).items():
            p = Path(directory) / name
            if not p.is_file() or sha256_file(p) != digest: problems.append(f"manifest: {name} digest differs")
    return problems


def validate_directory(directory) -> list[str]:
    """Every problem with one capture handoff directory ([] when it validates)."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    problems = check_manifest(manifest, directory)
    problems += check_table(directory / TABLE, TABLE_COLUMNS)
    problems += check_table(directory / SUMMARY, SUMMARY_COLUMNS)
    with open(directory / TAU_BOUNDS, newline="") as fh:
        header = next(csv.reader(fh), None)
    if header != list(_tau_bound_columns()): problems.append(f"{TAU_BOUNDS}: header differs from the schema")
    if not problems:
        problems += check_rows(directory / TABLE)
    return problems


def read_table(directory) -> list[dict]:
    return read_rows(Path(directory) / TABLE)


# ---------------------------------------------------------------------------------------------------- build
def build(args, project=None):
    project = project or resolve_project(args.project)
    record = project.record_module("capture_campaign")
    science = list(record.SCIENCE_EVENTS)
    markers = marker_freq_ids(project)
    k_sigma = float(project.register.value("capture.detection_gate_sigma"))
    inputs = Inputs()
    ex_pol, epochs = read_excess(args.capture_dir, inputs)
    ex_i, epochs_i = read_excess(args.stokes_dir, inputs)
    if epochs != epochs_i:
        raise ValueError("the two class-excess files cover different epochs")
    excess = dict(ex_pol); excess.update(ex_i)
    control = {"xx": {}, "yy": {}, "stokes_i": {}}
    for (c, pol), v in read_control(args.capture_dir, science, inputs).items():
        control[{"0": "xx", "1": "yy"}[pol]][(c, pol)] = v
    for (c, pol), v in read_control(args.stokes_dir, science, inputs).items():
        control["stokes_i"][(c, pol)] = v
    levels = {"xx": {}, "yy": {}, "stokes_i": {}, "polmax": {}}
    for c in CLASSES:
        name = level_file(c)
        levels["polmax"][c] = read_levels(os.path.join(args.capture_dir, name), inputs)
        levels["xx"][c] = read_levels(os.path.join(args.capture_dir, name[:-4] + "_pol0.csv"), inputs)
        levels["yy"][c] = read_levels(os.path.join(args.capture_dir, name[:-4] + "_pol1.csv"), inputs)
        levels["stokes_i"][c] = read_levels(os.path.join(args.stokes_dir, name), inputs)
    gamma = read_gamma(args.lags, inputs)
    frames, n_b = _frames_and_counts(args.datasets, epochs, record, inputs)
    eta, qmed, rescanned = _ladders(args, project, markers, record, science, inputs)
    table = table_rows(project=project, excess=excess, epochs=epochs, control=control, levels=levels, gamma=gamma,
                       frames=frames, n_b=n_b, eta=eta, qmed=qmed, rescanned=rescanned, science=science, k_sigma=k_sigma)
    board = read_board(args.board, inputs)
    present = {ch for (ch, _e, _c, _p) in excess}
    summary = summary_rows(project=project, capture_dir=args.capture_dir, science=science, record=record, eta=eta,
                           qmed=qmed, rescanned=rescanned, present_bands=present, board=board, all_events=epochs,
                           inputs=inputs)
    inputs.note(args.tau_bounds)
    manifest = manifest_body(project, record, science, epochs, frames, n_b, k_sigma, rescanned, inputs.files)
    return table, summary, manifest


def manifest_body(project, record, science, epochs, frames, n_b, k_sigma, rescanned, input_files) -> dict:
    im = project.integration_model
    bands = [{"band_id": b.label, "low_mhz": float(b.low_mhz), "high_mhz": float(b.high_mhz), "role": b.role,
              "target_freq_id": int(project.target_freq_id(b))} for b in project.frequency_plan.bands()]
    labels = dict(record.SCIENCE_EVENTS); labels.update(record.CADENCE_EVENTS)
    return {
        "inputs": dict(sorted(input_files.items())),
        "integration": {"frame_seconds": im.frame_seconds, "coherence_cap_seconds": im.coherence_cap_seconds,
                        "n_frame": int(project.detector_config.nfft)},
        "measurement_allowance_db": float(project.register.value("measurement_allowance_db")),
        "detection_gate_sigma": k_sigma,
        "null_policy": {"alpha": float(project.register.value("detection.false_alarm_target")),
                        "rule": "eta_Pfa is read only on a band's own verified, signal-free (off) population; the "
                                "capture holds none, so no band has an eta_Pfa here",
                        "eta_pfa_status": {b["band_id"]: (ETA_PFA_CONTROL if b["role"] == "control" else ETA_PFA_SCREENED)
                                           for b in bands}},
        "r_reference": {"statistic": "A, the noise-bias-free coherent amplitude of a class stack over a dump's frames, "
                                     "in units of the per-frame live-input power times N",
                        "power_form": "A_net^2 N n_b, the per-frame covariance ratio", "n_frame": int(project.detector_config.nfft),
                        "frame_seconds": im.frame_seconds},
        "residual_shape": "class excess: the in-band median over a band's bins of A less the 10th-percentile sky "
                          "reference over the bins within 40 MHz; the control level is the control band's per-bin "
                          "excess over the science dumps",
        "bands": bands,
        "classes": {class_name(c): {"ew": c[0], "ns": c[1], "n_b": n_b[c]} for c in CLASSES},
        "events": {ev: {"label": labels.get(ev, ""), "science": ev in science, "frames": frames[ev]} for ev in epochs},
        "products": {"xx": "pol 0 class stacks", "yy": "pol 1 class stacks",
                     "stokes_i": "the complex mean of the XX and YY class stacks, (v_XX + v_YY)/2",
                     "polmax": "the per-band level file's own polarisation (cadence readings only)"},
        "banks": {"nominal": "the archive's detector bank; ladder over the band's archive product",
                  "measured_marker": {"bands": [str(b) for b in rescanned],
                                      "note": "a rescan with a bank built on the band's measured marker; ladder over "
                                              "the rescan's frames"}},
        "producer": {"module": __name__},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilot-proxy capture oc-table", description=__doc__.split("\n\n")[0])
    ap.add_argument("--capture-dir", required=True, help="per-polarisation capture files: class_excess_epochs.csv, "
                    "class_floor_bins.csv, cadence_tau*.csv (with _pol0/_pol1), frame_residual_<event>.csv")
    ap.add_argument("--stokes-dir", required=True, help="the Stokes I class excess, control bins and level files")
    ap.add_argument("--lags", required=True, help="the ten-class lag coherence between the science dumps")
    ap.add_argument("--detector-runs", required=True, help="directory holding the detector runs of each dump")
    ap.add_argument("--datasets", required=True, help="root holding pilot_reduce_<event>/ (frames and counts)")
    ap.add_argument("--archive-products", required=True, help="the archive per-band products (<freq_id>.npz)")
    ap.add_argument("--measured-marker-rescan", default=None, help="products of the measured-marker rescan")
    ap.add_argument("--tau-bounds", required=True, help="capture_tau_bounds.csv of pilot-proxy capture tau-bounds")
    ap.add_argument("--board", required=True, help="the channel ruling's board_final.csv (archive gain bound)")
    ap.add_argument("--out", required=True, help="output directory; the files are written under <out>/capture/")
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)
    table, summary, manifest = build(args)
    out = write_handoff(Path(args.out) / "capture", table=table, summary=summary, tau_bounds_path=args.tau_bounds,
                        manifest=manifest)
    problems = validate_directory(out)
    print(f"{len(table)} table rows, {len(summary)} bands -> {out}")
    for p in problems[:20]:
        print("PROBLEM", p)
    return 1 if problems else 0


__all__ = ["BANKS", "CLASSES", "KEPT", "POLICIES", "PRODUCTS", "SCHEMA_FILE", "SCHEMA_ID", "SCHEMA_NAME",
           "SCHEMA_VERSION", "SUMMARY_COLUMNS", "TABLE_COLUMNS", "build", "check_manifest", "check_rows", "class_name",
           "kept_dumps", "level_file", "read_table", "schema", "schema_text", "tau_reading", "validate_directory",
           "write_handoff"]

if __name__ == "__main__":
    sys.exit(main())
