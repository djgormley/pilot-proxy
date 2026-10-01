# coding=utf-8
"""Transmitter-census case study: carrier-offset dispersion by service class.

Consumes two declared inputs and produces the paper's two case-study figures
plus their supporting tables. Nothing here touches the archive; both inputs
already exist: the scan's per-pilot high-resolution spectra or a
pre-detected line CSV, and the FCC/ISED 500-mile census.

Input schemas (CSV with a header row; extra columns are ignored)
-----------------------------------------------------------------
census:  rf_channel:int, callsign:str, service_class:str, and either
         detectability_db:float (a precomputed received-strength score, e.g.
         a propagation-model field strength; blanks rank last, tie-broken by
         distance_km when present) or erp_kw:float + distance_km:float
         (score = ERP / distance^2) [, bearing_deg, lat, lon, ...]
         service_class is normalized case-insensitively (punctuation and
         parentheticals collapse to underscores):
           full_service / full-power / primary        -> primary
           translator (lptv) / repeater / relay /
           lptv / low-power (lptv) / class a          -> non_primary
           anything else                              -> other
lines:   rf_channel:int, offset_hz:float (about the channel's nominal
         pilot), snr_db:float [, epoch, ...]
         Alternatively, --lines-from-run extracts the line list directly
         from a scan work dir's per-pilot products: each carries the
         high-resolution time-averaged spectrum of its coarse channel
         (integrated_spectrum_before/after_mask; natural FFT order, bin 0
         at the coarse-channel center, `sense` flipping the frequency
         axis). The spectrum's bin spacing is fs/nfft, derived from the
         product as bin_enbw_hz * detector_window_samples / nfft --
         `bin_enbw_hz` itself is the DETECTOR's bin width (fs/K) rather than the
         spectrum's.

Association rule (pluggable, recorded per line in association.csv)
------------------------------------------------------------------
Measured lines cannot be matched to transmitters by frequency alone --
co-channel transmitters share the nominal pilot; their *offsets* are the
unknown being studied. Per RF channel, lines are ranked by SNR and census
entries by a detectability score (ERP / distance^2); ranks are paired
positionally. Confidence tiers:
  dominant     strongest line paired with the top-scoring entry when that
               entry is primary (the usual case: one full-power station
               dominates the channel)
  ranked       remaining positional pairs
  unassociated line count exceeds census count (a finding, reported)
The alternative "dominant/secondary" strategy labels only the strongest
line per channel (primary where the census has one) and pools the rest as
non_primary without per-transmitter claims.

Outputs (into --output-dir)
---------------------------
association.csv                per-line class label, tier, paired callsign
summary.json                   per-class dispersion, per-channel stats,
                               Spearman rho + bootstrap CI, run parameters
fig_offset_by_class.png/.pdf   class-split offset distribution (Figure A)
fig_spread_vs_composition.png/.pdf
                               per-channel spread vs non-primary fraction
                               among associated lines (Figure B)
threshold_stability.csv        per-channel MAD vs SNR threshold sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PRIMARY_ALIASES = {"primary", "full_service", "full-service", "full_power",
                   "full-power", "fs", "dt", "dtv"}
NON_PRIMARY_ALIASES = {"non_primary", "translator", "translator_lptv",
                       "tx_translator", "repeater", "rebroadcaster", "relay",
                       "lptv", "low_power", "low_power_lptv", "ld", "dc",
                       "class_a"}


def normalize_class(raw: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
    if key in PRIMARY_ALIASES:
        return "primary"
    if key in NON_PRIMARY_ALIASES:
        return "non_primary"
    return "other"


@dataclass
class Transmitter:
    rf_channel: int
    callsign: str
    service_class: str          # normalized
    raw_class: str
    detectability: float        # higher ranks first; -inf when unscored
    distance_km: float          # tiebreak among unscored; inf when absent


@dataclass
class Line:
    rf_channel: int
    offset_hz: float
    snr_db: float


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_census(path: Path) -> list[Transmitter]:
    rows = _read_rows(path)
    if not rows:
        raise SystemExit(f"census: {path} is empty")
    scored_mode = "detectability_db" in rows[0]
    out = []
    for r in rows:
        dist = float(r["distance_km"]) if str(r.get("distance_km", "")).strip() else float("inf")
        if dist <= 0:
            raise SystemExit(f"census: non-positive distance for {r.get('callsign')}")
        if scored_mode:
            raw = str(r.get("detectability_db", "")).strip()
            score = float(raw) if raw else float("-inf")
        else:
            score = float(r["erp_kw"]) / (dist * dist)
        out.append(Transmitter(
            rf_channel=int(r["rf_channel"]),
            callsign=str(r.get("callsign", "")).strip(),
            service_class=normalize_class(r["service_class"]),
            raw_class=str(r["service_class"]).strip(),
            detectability=score,
            distance_km=dist,
        ))
    return out


def _bin_offsets_hz(nfft: int, sense: int, center_hz: float,
                    pilot_hz: float, bin_spacing_hz: float) -> np.ndarray:
    """Offset from the nominal pilot for every spectrum bin (natural FFT
    order: bin 0 = DC = coarse-channel center, upper half negative)."""
    k = np.arange(int(nfft))
    f_bb = np.where(k < nfft // 2, k, k - nfft).astype(np.float64) * bin_spacing_hz
    return center_hz + sense * f_bb - pilot_hz


_SPECTRUM_KEYS = {
    "before": "integrated_spectrum_before_mask",
    "after": "integrated_spectrum_after_mask",
}


def _prominence_db(snr: np.ndarray, i: int) -> float:
    """Topographic prominence of peak i in a 1-D dB array: height above the
    higher of the two key saddles (the minimum between the peak and the
    nearest higher sample on each side; the window edge if none)."""
    peak = snr[i]
    saddles = []
    for step in (-1, 1):
        j, lo = i + step, peak
        while 0 <= j < snr.size and snr[j] <= peak:
            lo = min(lo, snr[j])
            j += step
        saddles.append(lo)
    return float(peak - max(saddles))


def extract_lines_from_run(work_dir: Path, *, spectrum: str = "before",
                           search_window_hz: float = 30_000.0,
                           min_snr_db: float = 6.0,
                           min_separation_hz: float = 100.0,
                           min_prominence_db: float = 6.0) -> list[Line]:
    """Detected carrier lines from a scan work dir's per-pilot spectra.

    Within +/- search_window_hz of each channel's nominal pilot, local maxima
    at least min_snr_db above the window's median floor AND at least
    min_prominence_db above their key saddle become lines, greedily thinned
    (strongest first) to min_separation_hz. The prominence criterion rejects
    window-leakage sidelobes: a strong carrier's accumulated skirt carries
    ripples a few dB above their local valleys, while a real isolated carrier
    rises its full SNR from the floor. This is the case study's line source;
    it replaces the retired pilotcal offset survey."""
    key = _SPECTRUM_KEYS.get(spectrum)
    if key is None:
        raise SystemExit(f"unknown --spectrum {spectrum!r} (before/after)")
    need = {key, "nfft", "sense", "chime_frequency_hz", "pilot_frequency_hz",
            "physical_channel", "bin_enbw_hz", "detector_window_samples"}
    lines: list[Line] = []
    paths = sorted(Path(work_dir).glob("*.npz"))
    if not paths:
        raise SystemExit(f"--lines-from-run: no per-pilot products in {work_dir}")
    for path in paths:
        with np.load(str(path)) as z:
            if not need.issubset(set(z.files)):
                continue
            spec = np.asarray(z[key], np.float64).reshape(-1)
            nfft = int(np.asarray(z["nfft"]).reshape(-1)[0])
            sense = int(np.asarray(z["sense"]).reshape(-1)[0])
            center = float(np.asarray(z["chime_frequency_hz"]).reshape(-1)[0])
            pilot = float(np.asarray(z["pilot_frequency_hz"]).reshape(-1)[0])
            ch = int(np.asarray(z["physical_channel"]).reshape(-1)[0])
            enbw = float(np.asarray(z["bin_enbw_hz"]).reshape(-1)[0])
            kwin = int(np.asarray(z["detector_window_samples"]).reshape(-1)[0])
        if spec.size != nfft:
            continue
        # spectrum bin spacing = fs / nfft; the product self-describes fs as
        # bin_enbw_hz * K (the detector's rectangular-window bin is fs/K)
        spacing = enbw * kwin / nfft
        offs = _bin_offsets_hz(nfft, sense, center, pilot, spacing)
        sel = np.abs(offs) <= float(search_window_hz)
        # DC guard: the coarse-channel center bin carries the known DC spur
        # (dominant on the translator-oscillator channels); never call it a
        # carrier line
        k = np.arange(nfft)
        f_bb = np.where(k < nfft // 2, k, k - nfft).astype(np.float64) * spacing
        sel &= np.abs(f_bb) > 1.5 * spacing
        w, wo = spec[sel], offs[sel]
        order = np.argsort(wo)
        w, wo = w[order], wo[order]
        pos = w[w > 0]
        if w.size < 8 or pos.size == 0:
            continue
        floor = float(np.median(pos))
        snr = 10.0 * np.log10(np.maximum(w, 1e-300) / floor)
        interior = np.arange(1, w.size - 1)
        peaks = interior[(snr[interior] >= min_snr_db)
                         & (snr[interior] > snr[interior - 1])
                         & (snr[interior] >= snr[interior + 1])]
        kept: list[int] = []
        for i in sorted(peaks, key=lambda i: -snr[i]):
            if _prominence_db(snr, i) < min_prominence_db:
                continue
            if all(abs(wo[i] - wo[j]) >= min_separation_hz for j in kept):
                kept.append(i)
        for i in sorted(kept, key=lambda i: wo[i]):
            lines.append(Line(rf_channel=ch, offset_hz=float(wo[i]),
                              snr_db=float(snr[i])))
    if not lines:
        raise SystemExit(
            f"--lines-from-run: no carrier lines extracted from {work_dir} "
            f"(window {search_window_hz} Hz, floor+{min_snr_db} dB)")
    return lines


def load_lines(path: Path) -> list[Line]:
    return [Line(rf_channel=int(r["rf_channel"]),
                 offset_hz=float(r["offset_hz"]),
                 snr_db=float(r["snr_db"]))
            for r in _read_rows(path)]


# ------------------------------------------------------------- association

def associate(lines: list[Line], census: list[Transmitter], *,
              strategy: str = "ranked") -> list[dict]:
    """Per-line association records. See module docstring for the rule."""
    if strategy not in ("ranked", "dominant_secondary"):
        raise SystemExit(f"unknown association strategy: {strategy!r}")
    records: list[dict] = []
    channels = sorted({l.rf_channel for l in lines})
    for ch in channels:
        ch_lines = sorted((l for l in lines if l.rf_channel == ch),
                          key=lambda l: -l.snr_db)
        ch_census = sorted((t for t in census if t.rf_channel == ch),
                           key=lambda t: (-t.detectability, t.distance_km))
        for rank, line in enumerate(ch_lines):
            rec = {"rf_channel": ch, "offset_hz": line.offset_hz,
                   "snr_db": line.snr_db, "line_rank": rank,
                   "callsign": "", "tier": "unassociated",
                   "service_class": "unassociated"}
            if strategy == "ranked" and rank < len(ch_census):
                t = ch_census[rank]
                rec.update(callsign=t.callsign, service_class=t.service_class,
                           tier=("dominant" if rank == 0 and
                                 t.service_class == "primary" else "ranked"))
            elif strategy == "dominant_secondary" and ch_census:
                if rank == 0:
                    t = ch_census[0]
                    rec.update(callsign=t.callsign,
                               service_class=t.service_class, tier="dominant")
                else:
                    rec.update(service_class="non_primary", tier="pooled")
            records.append(rec)
    return records


# --------------------------------------------------------------- statistics

def mad_hz(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def rms_hz(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(v * v))) if v.size else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, average ranks for ties (no scipy)."""
    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(a.size, dtype=np.float64)
        ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
        # average ties
        for val in np.unique(a):
            sel = a == val
            if sel.sum() > 1:
                ranks[sel] = ranks[sel].mean()
        return ranks
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def bootstrap_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = spearman(x[idx], y[idx])
        if np.isfinite(r):
            stats.append(r)
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def per_channel_stats(records: list[dict]) -> list[dict]:
    out = []
    for ch in sorted({r["rf_channel"] for r in records}):
        rs = [r for r in records if r["rf_channel"] == ch]
        offsets = np.asarray([r["offset_hz"] for r in rs], np.float64)
        assoc = [r for r in rs if r["service_class"] != "unassociated"]
        n_np = sum(1 for r in assoc if r["service_class"] == "non_primary")
        out.append({
            "rf_channel": ch,
            "n_lines": len(rs),
            "n_unassociated": sum(1 for r in rs
                                  if r["service_class"] == "unassociated"),
            "mad_hz": mad_hz(offsets),
            "rms_hz": rms_hz(offsets),
            "span_hz": float(offsets.max() - offsets.min()) if offsets.size else float("nan"),
            "non_primary_fraction": (n_np / len(assoc)) if assoc else float("nan"),
        })
    return out


def threshold_sweep(records: list[dict], thresholds: np.ndarray) -> list[dict]:
    rows = []
    for t in thresholds:
        for ch in sorted({r["rf_channel"] for r in records}):
            offs = np.asarray([r["offset_hz"] for r in records
                               if r["rf_channel"] == ch and r["snr_db"] >= t],
                              np.float64)
            rows.append({"snr_threshold_db": float(t), "rf_channel": ch,
                         "n_lines": int(offs.size), "mad_hz": mad_hz(offs)})
    return rows


# ------------------------------------------------------------------ figures

def _figures(records, chan_stats, rho, ci, out_dir: Path,
             qualifying=None) -> None:
    from pilot_proxy.plot_style import setup_matplotlib
    try:
        plt = setup_matplotlib()
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for the census figures. Install the "
            "optional plot dependency, for example: "
            "python -m pip install matplotlib"
        ) from exc

    # Figure A: class-split offset distribution (per detected line)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    groups = [("primary", "C0"), ("non_primary", "C1"), ("unassociated", "C7")]
    rng = np.random.default_rng(1)
    for i, (cls, color) in enumerate(groups):
        offs = np.asarray([r["offset_hz"] for r in records
                           if r["service_class"] == cls], np.float64)
        if offs.size == 0:
            continue
        x = i + rng.uniform(-0.14, 0.14, size=offs.size)
        ax.plot(x, offs, ".", ms=4, alpha=0.6, color=color)
        q1, q2, q3 = np.percentile(offs, [25, 50, 75])
        ax.hlines([q1, q2, q3], i - 0.22, i + 0.22, color=color,
                  lw=[1.0, 1.8, 1.0])
        ax.text(i, ax.get_ylim()[0], "", ha="center")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0].replace("_", " ") for g in groups])
    ax.set_ylabel("pilot offset from nominal (Hz)")
    ax.set_title("Carrier offset by service class (per detected line)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_offset_by_class.{ext}", dpi=200)
    plt.close(fig)

    # Figure B: per-channel spread vs composition
    xs = np.asarray([c["non_primary_fraction"] for c in chan_stats], np.float64)
    ys = np.asarray([c["mad_hz"] for c in chan_stats], np.float64)
    finite = np.isfinite(xs) & np.isfinite(ys)
    qual = (np.asarray(qualifying, bool) if qualifying is not None
            else finite)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(xs[qual], ys[qual], "o", ms=5)
    excl = finite & ~qual
    if excl.any():   # pre-registered exclusions are shown rather than hidden
        ax.plot(xs[excl], ys[excl], "o", ms=4, mfc="none", color="0.6")
    for c, q, f in zip(chan_stats, qual, finite):
        if f:
            ax.annotate(str(c["rf_channel"]),
                        (c["non_primary_fraction"], c["mad_hz"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=7,
                        color=("0.2" if q else "0.6"))
    label = (f"Spearman $\\rho$ = {rho:.2f} "
             f"[{ci[0]:.2f}, {ci[1]:.2f}], N = {int(qual.sum())}"
             + (f" ({int(excl.sum())} grayed: n<min)" if excl.any() else ""))
    ax.set_xlabel("non-primary fraction among associated lines")
    ax.set_ylabel("per-channel offset MAD (Hz)")
    ax.set_title(label)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_spread_vs_composition.{ext}", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze-transmitter-census",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--census", type=Path, required=True,
                   help="Census CSV (schema in the module docstring).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--lines", type=Path, default=None,
                     help="Detected-line CSV: rf_channel, offset_hz, snr_db.")
    src.add_argument("--lines-from-run", type=Path, default=None,
                     help="Scan work dir (_per_pilot/): extract the line "
                          "list from the products' integrated high-res "
                          "spectra instead of reading a CSV.")
    p.add_argument("--spectrum", choices=["before", "after"], default="before")
    p.add_argument("--search-window-hz", type=float, default=30_000.0)
    p.add_argument("--min-snr-db", type=float, default=6.0)
    p.add_argument("--min-separation-hz", type=float, default=100.0)
    p.add_argument("--min-prominence-db", type=float, default=6.0)
    p.add_argument("--min-channel-lines", type=int, default=3,
                   help="Pre-registered Figure-B inclusion rule: channels "
                        "enter the spread-vs-composition panel only with at "
                        "least this many extracted lines (MAD is not a "
                        "dispersion below n=3). Excluded channels are shown "
                        "grayed; summary.json carries the Spearman statistic "
                        "both ways.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--association", choices=["ranked", "dominant_secondary"],
                   default="ranked")
    p.add_argument("--snr-threshold-db", type=float, default=None,
                   help="Drop lines below this SNR before analysis "
                        "(default: keep all; sweep is reported regardless).")
    p.add_argument("--sweep-start-db", type=float, default=0.0)
    p.add_argument("--sweep-stop-db", type=float, default=20.0)
    p.add_argument("--sweep-step-db", type=float, default=2.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    census = load_census(args.census)
    if args.lines_from_run is not None:
        lines = extract_lines_from_run(
            args.lines_from_run, spectrum=args.spectrum,
            search_window_hz=args.search_window_hz,
            min_snr_db=args.min_snr_db,
            min_separation_hz=args.min_separation_hz,
            min_prominence_db=args.min_prominence_db)
        line_source = f"extracted:{args.lines_from_run}"
    else:
        lines = load_lines(args.lines)
        line_source = f"csv:{args.lines}"
    if args.snr_threshold_db is not None:
        lines = [l for l in lines if l.snr_db >= args.snr_threshold_db]
    if not lines:
        raise SystemExit("no lines above threshold; nothing to analyze")

    records = associate(lines, census, strategy=args.association)
    chan_stats = per_channel_stats(records)

    xs = np.asarray([c["non_primary_fraction"] for c in chan_stats], np.float64)
    ys = np.asarray([c["mad_hz"] for c in chan_stats], np.float64)
    ns = np.asarray([c["n_lines"] for c in chan_stats])
    finite = np.isfinite(xs) & np.isfinite(ys)
    ok = finite & (ns >= int(args.min_channel_lines))   # pre-registered rule
    rho = spearman(xs[ok], ys[ok])
    ci = bootstrap_ci(xs[ok], ys[ok], n_boot=args.bootstrap, seed=args.seed)
    rho_all = spearman(xs[finite], ys[finite])

    by_class = {}
    for cls in ("primary", "non_primary", "unassociated"):
        offs = np.asarray([r["offset_hz"] for r in records
                           if r["service_class"] == cls], np.float64)
        by_class[cls] = {"n": int(offs.size), "mad_hz": mad_hz(offs),
                         "rms_hz": rms_hz(offs),
                         "span_hz": (float(offs.max() - offs.min())
                                     if offs.size else float("nan"))}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.lines_from_run is not None:
        with open(args.output_dir / "extracted_lines.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["rf_channel", "offset_hz", "snr_db"])
            w.writeheader()
            w.writerows({"rf_channel": l.rf_channel, "offset_hz": l.offset_hz,
                         "snr_db": l.snr_db} for l in lines)
    with open(args.output_dir / "association.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    sweep = threshold_sweep(records, np.arange(args.sweep_start_db,
                                               args.sweep_stop_db + 1e-9,
                                               args.sweep_step_db))
    with open(args.output_dir / "threshold_stability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        w.writeheader(); w.writerows(sweep)
    summary = {"schema_version": "transmitter_census_v1",
               "association_strategy": args.association,
               "line_source": line_source,
               "min_channel_lines": int(args.min_channel_lines),
               "spearman_rho_all_channels": rho_all,
               "n_channels_qualifying": int(ok.sum()),
               "n_channels_excluded": int(finite.sum() - ok.sum()),
               "snr_threshold_db": args.snr_threshold_db,
               "by_class": by_class, "per_channel": chan_stats,
               "spearman_rho": rho,
               "spearman_ci95": list(ci),
               "n_channels": int(ok.sum())}
    with open(args.output_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    _figures(records, chan_stats, rho, ci, args.output_dir,
             qualifying=ok.tolist())

    print(f"lines: {len(records)}  channels: {len(chan_stats)}  "
          f"primary MAD {by_class['primary']['mad_hz']:.1f} Hz vs "
          f"non-primary MAD {by_class['non_primary']['mad_hz']:.1f} Hz  "
          f"| rho={rho:.2f} CI95=[{ci[0]:.2f},{ci[1]:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
