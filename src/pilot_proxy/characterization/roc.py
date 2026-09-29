"""Coarse-versus-fine ROC and Youden J, recomputed from the survey products.

A diagnostic, never a threshold rule: Youden's J weighs a missed detection
and a false alarm equally, and the masking costs are not equal
(``docs/DESIGN_DECISIONS.md``). The only threshold rule of the method is the
science side's selection inside the deployable interval.

The candidate-selection record (docs/DESIGN_DECISIONS.md, 2026-07) chose the
fine designated-set CFAR as the runtime decision from measured ROCs: every
calibrated coarse operating point either spends about half of the dated
off-state time (the positive-excess point holds P_d ~ 1 but rejects ~48.5% of
the null by construction) or collapses on weak channels. This module
recomputes that comparison on the released per-pilot products with the
populations stated.

Populations (:class:`Populations`; read from the project's records package,
``<records>.roc_populations``, which for the repository's profile is
``pilot_proxy.records.chime_atsc_2026.roc_populations``):
  null   : one band's frames in its dated transmitter-off era; the labels are
           dated epochs read from this archive, so the truth is weak
           (``label_source = "dated epochs (archive-inferred)"``).
  signal : the on-epoch frames of the signal bands from a declared month.

Per-frame statistics:
  coarse : F / mu0 (the stored exact statistic over the exact null mean).
  fine   : designated-window maximum of the stored 256-bin fine statistic
           over anchor +/- 2 bins, normalized by the frame's bulk median;
           each band's anchor is the argmax of its mean signal-epoch fine
           spectrum, and the null frames are scored in the same window
           (null bins are exchangeable, Section 5.5 of the dissertation).

Output: the ROC table at the record's null-quantile P_fa points, each
statistic's Youden J (max over threshold of P_d - P_fa), and the coarse
positive-excess point's measured null exceedance.

    pilot-proxy characterize roc --products DIR [--csv FILE] [--project DIR]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pilot_proxy.archived_product_keys import measurement
from pilot_proxy.product_contract import fine_power_ratio_of, null_power_ratio_of

PFA_POINTS = (0.10, 0.05, 0.015)
HALF = 2                          # designated window: anchor +/- 2 fine bins
LABEL_SOURCE = "dated epochs (archive-inferred)"
ROC_COLUMNS = ("statistic", "null_population", "signal_population", "label_source", "pfa_point", "threshold",
               "pd", "n_null", "n_signal", "youden_j")


@dataclass(frozen=True)
class Populations:
    """The labelled populations of one ROC comparison."""

    null_band: int                    # the band whose dated off era is the null
    null_off_through: str             # the last YYYY-MM of that off era
    signal: dict                      # signal band -> freq_id
    signal_from: str                  # the first YYYY-MM of the signal epochs


def load_npz(path) -> dict[str, np.ndarray]:
    """Load a product archive without pickle, all arrays materialised."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True)
                for name in archive.files}


def paths(channels=None, per_pilot: Path | None = None) -> dict[int, str]:
    """physical channel -> product path, optionally filtered to `channels`."""
    found: dict[int, str] = {}
    for p in sorted(glob.glob(str(Path(per_pilot) / "*.npz"))):
        with np.load(p, allow_pickle=False) as archive:
            ch = int(np.ravel(archive["physical_channel"])[0])
        found[ch] = p
    if not found:
        raise SystemExit(f"no per-pilot products (*.npz) under {per_pilot}; "
                         "pass --products")
    if channels is not None:
        missing = [c for c in channels if c not in found]
        if missing:
            raise SystemExit(f"no product for channel(s) {missing} "
                             f"under {per_pilot}")
        found = {c: found[c] for c in channels}
    return found


def _ts(ym: str, end: bool = False) -> float:
    y, m = int(ym[:4]), int(ym[5:7])
    if end:                       # first instant after the month
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp()


def _frames(d):
    """(frame times, valid mask, coarse F/mu0, fine [nframes, 256])."""
    t = d["unit_time0_ctime"][d["frame_unit_index"].ravel()]
    valid = d["valid"].ravel().astype(bool)
    mu0 = null_power_ratio_of(d)
    coarse = measurement(d, "coarse_power_ratio").ravel() / mu0
    fine = fine_power_ratio_of(d)
    return t, valid, coarse, fine


def _fine_stat(fine, window):
    """Designated-window max over the frame's bulk median."""
    med = np.nanmedian(fine, axis=1)
    win = np.nanmax(fine[:, window], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = win / med
    return s


def roc_curve(null_s, sig_s):
    """P_d at the P_fa grid plus Youden J over the swept threshold."""
    null_s = null_s[np.isfinite(null_s)]
    sig_s = sig_s[np.isfinite(sig_s)]
    thresholds = np.unique(null_s)
    pfa = np.array([(null_s > th).mean() for th in thresholds])
    pd_ = np.array([(sig_s > th).mean() for th in thresholds])
    j = pd_ - pfa
    best = int(np.argmax(j))
    at = {}
    for target in PFA_POINTS:
        k = int(np.argmin(np.abs(pfa - target)))
        at[target] = (float(pfa[k]), float(pd_[k]), float(thresholds[k]))
    return at, (float(j[best]), float(thresholds[best]),
                float(pfa[best]), float(pd_[best]))


def youden_j(null_s, sig_s):
    """Youden's J and where it lies: ``(J, threshold, P_fa, P_d)`` (a diagnostic)."""
    return roc_curve(null_s, sig_s)[1]


def run_roc(per_pilot: Path, populations: Populations, *, write=print, csv_path: Path | None = None) -> int:
    """The ROC comparison on a product directory; the text goes to ``write``, the rows to ``csv_path``."""
    paths_by_band = paths(per_pilot=per_pilot)
    rows = []

    d_null = load_npz(paths_by_band[populations.null_band])
    t, valid, coarse, fine = _frames(d_null)
    off = valid & (t < _ts(populations.null_off_through, end=True))
    null_coarse = coarse[off]
    write(f"null: ch{populations.null_band} off-era frames n={int(off.sum())} "
          f"(through {populations.null_off_through})")
    pe = float((null_coarse > 1.0).mean())
    write(f"coarse positive-excess point on the null: "
          f"P_fa = {pe:.3f} (the deployed rule's verified-quiet spend)")
    null_label = f"band {populations.null_band} through {populations.null_off_through}"

    for ch, fid in populations.signal.items():
        d = load_npz(paths_by_band[ch])
        ts, va, co, fi = _frames(d)
        on = va & (ts >= _ts(populations.signal_from))
        # anchor: argmax of the mean signal-epoch fine spectrum
        anchor = int(np.nanargmax(np.nanmean(fi[on], axis=0)))
        window = (np.arange(anchor - HALF, anchor + HALF + 1)) % fi.shape[1]
        sig_fine = _fine_stat(fi[on], window)
        null_fine = _fine_stat(fine[off], window)
        at_c, j_c = roc_curve(null_coarse, co[on])
        at_f, j_f = roc_curve(null_fine, sig_fine)
        write(f"\nch{ch}: signal n={int(on.sum())} (from {populations.signal_from}), "
              f"anchor bin {anchor}")
        write("  Pfa(null q)   coarse Pd   fine Pd")
        for target in PFA_POINTS:
            write(f"  {target:11.3f}   {at_c[target][1]:9.3f}   "
                  f"{at_f[target][1]:7.3f}")
        write(f"  Youden J: coarse {j_c[0]:.3f} "
              f"(Pfa {j_c[2]:.3f}, Pd {j_c[3]:.3f}) | "
              f"fine {j_f[0]:.3f} (Pfa {j_f[2]:.3f}, Pd {j_f[3]:.3f})")
        signal_label = f"band {ch} from {populations.signal_from}"
        for statistic, at, j, n_null, n_signal in (
                ("Q", at_c, j_c, int(np.isfinite(null_coarse).sum()), int(np.isfinite(co[on]).sum())),
                ("fine designated max / bulk median", at_f, j_f, int(np.isfinite(null_fine).sum()),
                 int(np.isfinite(sig_fine).sum()))):
            for target in PFA_POINTS:
                rows.append({"statistic": statistic, "null_population": null_label,
                             "signal_population": signal_label, "label_source": LABEL_SOURCE,
                             "pfa_point": at[target][0], "threshold": at[target][2], "pd": at[target][1],
                             "n_null": n_null, "n_signal": n_signal, "youden_j": j[0]})
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(ROC_COLUMNS), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    return 0


def main(argv=None) -> int:
    from pilot_proxy.config.project import default_project_dir, load_project

    ap = argparse.ArgumentParser(prog="pilot-proxy characterize roc", description=__doc__)
    ap.add_argument("--products", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None, help="also write the ROC rows (roc.csv) here")
    ap.add_argument("--project", type=Path, default=None,
                    help="project profile directory whose records name the populations (default: the repository's)")
    args = ap.parse_args(argv)
    populations = load_project(args.project or default_project_dir()).record_module("roc_populations").POPULATIONS
    return run_roc(args.products, populations, csv_path=args.csv)


__all__ = ["HALF", "LABEL_SOURCE", "PFA_POINTS", "Populations", "ROC_COLUMNS", "main", "paths", "roc_curve",
           "run_roc", "youden_j"]
