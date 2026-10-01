"""Within-dump lower bounds on the coherence time tau (amendment 19 item 35, detection theory).

The per-class 95 % lower bounds of the autocorrelation lane
(``implications.csv``, ``least_lb95_all_variants_s``, the least over the
lane's variants) and of its independent check (``summary.json``: the primary
exp/block bounds, the bound without lag truncation and the bound on lags 1 and
2 only). "The least tau consistent with the data is the bound", so each class
takes the least of every variant of both. Stokes I, the ruling's product. A
bound longer than the longest lag the lane observes (32 frames, 1.34 s,
``lag_profiles.csv``) is an extrapolation of the exponential or block model
(the lane's own caveat); a residual that stays coherent over the whole dump
and then decorrelates is also consistent with the data, so the reported bound
is the bound capped at that span (``lb_priced``, the name the release code gives it).
The uncapped bound is kept beside it.

Options the releases use: ``censor`` adds the bounds the lane measured for
the censored classes (``bounds_b4.json``); ``source="fits"`` reads the check's
variants as fitted (``fits_check.csv``); ``calibrate`` adds the control band's
null test of each capped bound (a bound beats the null when at most 5 % of the
91 null cells certify at or above it).

The lane groups its classes into ranges; that grouping belongs to the science
side's rule and is not carried here.

The lane's own files are inputs (read only); this module does not rerun the
lane. ``pilot-proxy capture tau-bounds`` writes ``capture_tau_bounds.csv``.

usage: pilot-proxy capture tau-bounds --lane DIR --check DIR --b3-null CSV --b4-bounds JSON --out CSV [--project DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

B3_FCR_MAX = 0.05          # B-3/PLAN.md section 3: a bound beats the null when at most 5 % of the 91 null cells certify at or above it
CHECK_FULL_LAGS = ("primary", "perm_null", "mean_agg", "dumpmean_norm", "drop_pilot", "rho_ep_0", "rho_ep_1")   # the check's fits without lag truncation
# the settings of the releases of record: (censor, source, calibrate); r5, and r5.1 / r5.2
SETTINGS = ((False, "summary", False), (True, "fits", True))
COLUMNS = ["censor", "source", "calibrate", "channel", "ew", "ns", "lane", "check", "variants", "lb",
           "least_variant", "span", "lb_priced", "capped", "b4", "calibration_fcr", "calibration_n_at_or_above",
           "calibration_n_cells", "calibration_fcr_same_class", "calibration_null_max_s",
           "calibration_null_max_same_class_s", "calibration_passes"]


def _cls(s):
    a, b = s.strip().strip("()").split(","); return (int(a), int(b))


def check_fits(fits_check):
    """{(channel, class): {variant: row}} from check_ac/fits_check.csv (read only)."""
    out = {}
    for r in csv.DictReader(open(fits_check)):
        out.setdefault((int(r["channel"]), _cls(r["cls"])), {})[r["variant"]] = r
    return out


def null_bounds(b3_null):
    """B-3's control-band null: the least bound of each of the 91 cells (13 bins x 7 classes), by the procedure that bounds a class."""
    rows = list(csv.DictReader(open(b3_null))); assert len(rows) == 91, len(rows)
    return [(_cls(r["cls"]), float(r["least_bound_s"])) for r in rows]


def calibration(b, c, *, b3_null):
    """B-3's test of a capped bound b on class c: the empirical false-certification rate FCR(b), the share of the null cells whose least
    bound is at or above b, over all 91 cells (the declared test) and over the cells of the same class (reported)."""
    nb = null_bounds(b3_null); n_all = sum(1 for _, x in nb if x >= b); same = [x for cc, x in nb if cc == c]
    fcr = n_all / len(nb)
    return dict(fcr=fcr, n_at_or_above=n_all, n_cells=len(nb), fcr_same_class=(sum(1 for x in same if x >= b) / len(same)) if same else None,
                null_max_s=max(x for _, x in nb), null_max_same_class_s=max(same) if same else None, passes=fcr <= B3_FCR_MAX)


def tau_lower_bounds(censor=False, source="summary", calibrate=False, *, lane, check, b3_null=None, b4_bounds=None,
                     frame_seconds):
    """{(channel, class): dict(lb, lane, check, variants)}: the least 95 % lower bound over every variant of the lane and the check.
    With censor=True the bounds the lane run measured for the censored classes (b4_bounds) are added; each is the
    least over the lane's and the check's variants, like the others.
    source="fits" reads the check's variants from fits_check.csv as fitted (the lags-1-and-2 fit is 0.918 s on 26 and 0.439 s
    on 27, which summary.json rounds to 0.92 and 0.44; the other variants agree with summary.json, asserted); calibrate=True adds
    the control band's calibration of each capped bound (calibration(); a bound that does not beat the null has calibration
    passes False)."""
    assert source in ("summary", "fits")
    out = {}
    for r in csv.DictReader(open(os.path.join(lane, "implications.csv"))):
        k = (int(r["channel"]), _cls(r["cls"])); v = {f"lane {a}": float(b) for a, b in json.loads(r["variants_lb95_s"]).items()}
        lane_ = float(r["least_lb95_all_variants_s"])
        assert abs(lane_ - min(v.values())) <= 1e-9 * max(1.0, lane_), (k, lane_, v)   # the lane's least is the least of its variants
        assert k not in out
        out[k] = dict(lane=lane_, variants=v)
    s = json.load(open(os.path.join(check, "summary.json")))
    prim = {}
    for key, x in s["primary_results_stokes_I"].items():
        ch, c = key.split(" ", 1); prim[(int(ch), _cls(c))] = x
    chk = {k: {"check primary exp": x["tau_lb95_s_exp_block"][0], "check primary block": x["tau_lb95_s_exp_block"][1]} for k, x in prim.items()}
    for var, d in s["least_over_variants"].items():
        if not var.startswith("tau_lb95_s"): continue
        for key, val in d.items():
            p = key.split(" ", 1); ch = int(p[0])
            if len(p) == 2: k = (ch, _cls(p[1]))
            else:   # a channel read on one class only: the class of its primary result
                ks = [kk for kk in prim if kk[0] == ch]; assert len(ks) == 1, (key, ks); k = ks[0]
            chk.setdefault(k, {})[f"check {var[len('tau_lb95_s_'):]}"] = float(val)
    if source == "fits":
        F = check_fits(os.path.join(check, "fits_check.csv")); new = {}
        for k, v in chk.items():
            f = F[k]; full = min(float(f[x]["lb95_both_s"]) for x in CHECK_FULL_LAGS)
            nv = {"check primary exp": float(f["primary"]["exp_lb95_s"]), "check primary block": float(f["primary"]["block_lb95_s"]),
                  "check without_lag_truncation": full, "check lags_1_2_only": float(f["dmax_2"]["lb95_both_s"])}
            assert set(nv) == set(v), (k, set(v))
            for kk, x in nv.items():   # summary.json prints the same fits to two decimals (or three significant figures)
                assert abs(x - v[kk]) <= 0.006 * max(1.0, v[kk] / 10), (k, kk, x, v[kk])
            new[k] = nv
        chk = new
    for k, v in chk.items():
        assert k in out, k                      # the check reads only classes the lane bounds
        out[k]["variants"].update(v); out[k]["check"] = min(v.values())
    if censor:
        for key, x in json.load(open(b4_bounds))["bounds"].items():
            ch, c = key.split("|"); k = (int(ch), _cls(c)); assert k not in out, k
            v = {kk: (float(vv) if vv is not None else np.nan) for kk, vv in x["variants"].items()}; v = {kk: vv for kk, vv in v.items() if np.isfinite(vv)}
            if not v: continue                   # no finite bound on any variant: the class keeps the admissible minimum
            out[k] = dict(lane=min((vv for kk, vv in v.items() if kk.startswith("lane")), default=np.nan), variants=v,
                          check=min((vv for kk, vv in v.items() if kk.startswith("check")), default=np.nan), b4=True)
    span = lb_span(lane, frame_seconds)
    for k, o in out.items():
        o["lb"] = min(o["variants"].values()); o["least_variant"] = min(o["variants"], key=o["variants"].get)
        o["span"] = span; o["lb_priced"] = min(o["lb"], span); o["capped"] = o["lb"] > span
        if calibrate: o["calibration"] = calibration(o["lb_priced"], k[1], b3_null=b3_null)
    return out


def lb_span(lane, frame_seconds):
    """The longest within-dump lag the lane observes, in seconds (32 frames of the 33-frame dumps, 1.34 s)."""
    m = max(int(r["lag"]) for r in csv.DictReader(open(os.path.join(lane, "lag_profiles.csv"))))
    return m * frame_seconds


def _cell(x):
    if x is None: return ""
    if isinstance(x, (bool, np.bool_)): return "True" if x else "False"
    if isinstance(x, (float, np.floating)): return repr(float(x)) if math.isfinite(x) else ""
    return str(x)


def rows(settings=SETTINGS, **inputs):
    out = []
    for censor, source, calibrate in settings:
        lb = tau_lower_bounds(censor=censor, source=source, calibrate=calibrate, **inputs)
        for (ch, c), o in sorted(lb.items()):
            cal = o.get("calibration", {})
            out.append({"censor": censor, "source": source, "calibrate": calibrate, "channel": ch, "ew": c[0], "ns": c[1],
                        "lane": o.get("lane"), "check": o.get("check"),
                        "variants": json.dumps(o["variants"]), "lb": o["lb"], "least_variant": o["least_variant"],
                        "span": o["span"], "lb_priced": o["lb_priced"], "capped": o["capped"], "b4": bool(o.get("b4", False)),
                        **{f"calibration_{k}": cal.get(k) for k in ("fcr", "n_at_or_above", "n_cells", "fcr_same_class",
                                                                    "null_max_s", "null_max_same_class_s", "passes")}})
    return out


def write(table, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(COLUMNS)
        for r in table: w.writerow([_cell(r[c]) for c in COLUMNS])


def read(path):
    """{(censor, source, calibrate): {(channel, (ew, ns)): row}} with the floats and flags parsed."""
    out = {}
    for r in csv.DictReader(open(path)):
        key = (r["censor"] == "True", r["source"], r["calibrate"] == "True")
        f = lambda x: float(x) if x != "" else np.nan  # noqa: E731
        row = dict(r, lane=f(r["lane"]), check=f(r["check"]), lb=f(r["lb"]), span=f(r["span"]), lb_priced=f(r["lb_priced"]),
                   capped=r["capped"] == "True", b4=r["b4"] == "True", variants=json.loads(r["variants"]))
        if r["calibration_passes"]:
            row["calibration"] = dict(fcr=float(r["calibration_fcr"]), n_at_or_above=int(r["calibration_n_at_or_above"]),
                                      n_cells=int(r["calibration_n_cells"]),
                                      fcr_same_class=f(r["calibration_fcr_same_class"]) if r["calibration_fcr_same_class"] else None,
                                      null_max_s=float(r["calibration_null_max_s"]),
                                      null_max_same_class_s=f(r["calibration_null_max_same_class_s"]) if r["calibration_null_max_same_class_s"] else None,
                                      passes=r["calibration_passes"] == "True")
        out.setdefault(key, {})[(int(r["channel"]), (int(r["ew"]), int(r["ns"])))] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilot-proxy capture tau-bounds", description=__doc__.split("\n\n")[0])
    ap.add_argument("--lane", required=True, help="the autocorrelation lane (implications.csv, lag_profiles.csv)")
    ap.add_argument("--check", required=True, help="its independent check (summary.json, fits_check.csv)")
    ap.add_argument("--b3-null", required=True, help="the control band's null bounds (null_bounds.csv)")
    ap.add_argument("--b4-bounds", required=True, help="the censored classes' bounds (bounds_b4.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)
    from .markers import resolve_project
    tf = resolve_project(args.project).integration_model.frame_seconds
    table = rows(lane=args.lane, check=args.check, b3_null=args.b3_null, b4_bounds=args.b4_bounds, frame_seconds=tf)
    write(table, args.out)
    print(f"{len(table)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
