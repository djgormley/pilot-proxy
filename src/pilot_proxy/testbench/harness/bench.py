"""``pilot-proxy bench``: bench figures and tables from frozen releases.

    pilot-proxy bench estimator-transfer --release DIR --out FILE.pdf
        [--calibration pilot_below_db,bin_enbw_hz,dtv_bandwidth_hz[,efficiency]]
        [--title TEXT] [--y-min DB] [--no-tex]
    pilot-proxy bench transfer-points --sweep ROOT --out plot_points.csv
        [--conditioning conditioning.json] [--bootstrap-samples N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pilot_proxy import figure_style as style

from .estimator_transfer import Calibration, figure_estimator_transfer, load_release
from .evaluations import Conditioning, digital_sweep_layout, load_evaluations, transfer_points, write_points


def _calibration(text: str | None) -> Calibration | None:
    if not text:
        return None
    parts = [float(p) for p in text.split(",")]
    if len(parts) not in (3, 4):
        raise SystemExit("--calibration takes pilot_below_db,bin_enbw_hz,dtv_bandwidth_hz[,efficiency]")
    return Calibration(*parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pilot-proxy bench", description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="command", required=True)
    et = sub.add_parser("estimator-transfer", help="digital or over-the-air estimator-transfer figure")
    et.add_argument("--release", type=Path, required=True, help="release directory (data/plot_points.csv inside)")
    et.add_argument("--out", type=Path, required=True, help="PDF to write")
    et.add_argument("--calibration", default=None, help="pilot calibration when the release has no run/run_state.json")
    et.add_argument("--title", default=None)
    et.add_argument("--y-min", type=float, default=None, dest="y_min")
    et.add_argument("--no-tex", action="store_true", help="preview without LaTeX text rendering")
    tp = sub.add_parser("transfer-points", help="pool a raw evaluate-snr sweep into the release's plot_points.csv")
    tp.add_argument("--sweep", type=Path, required=True, help="root of the 40 digital shard directories")
    tp.add_argument("--out", type=Path, required=True, help="CSV to write")
    tp.add_argument("--conditioning", type=Path, default=None, help="conditioning.json with the waveform coefficients")
    tp.add_argument("--bootstrap-samples", type=int, default=10_000, dest="bootstrap_samples")
    args = ap.parse_args(argv)

    if args.command == "transfer-points":
        shards = load_evaluations(digital_sweep_layout(args.sweep))
        conditioning = Conditioning.from_json(args.conditioning) if args.conditioning else None
        print(write_points(transfer_points(shards, conditioning=conditioning,
                                           bootstrap_samples=args.bootstrap_samples), args.out))
        return 0

    style.configure(require_tex=not args.no_tex)
    if args.command == "estimator-transfer":
        release = load_release(args.release, calibration=_calibration(args.calibration))
        out = figure_estimator_transfer(release, out=args.out, title=args.title, y_min_db=args.y_min)
        print(out)
    return 0


__all__ = ["main"]
