"""A synthetic two-band capture (band 33, rescanned with a measured-marker bank, and control band 37) and its
capture handoff, for the schema tests and the golden handoff ``tests/golden/handoff/capture``.

The capture files (class excess, control bins, level files, lags, frame
residuals) are written directly with fixed values, so the handoff depends on
no estimator; the small products (the dumps' first files, the archive
products, the detector runs) are seeded. Regenerate the golden with
``python tests/capture/capture_handoff_fixtures.py --write-golden``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "handoff" / "capture"
SCIENCE = ["20260916162300", "20260917040230", "20260917090230", "20260917140230"]
CADENCE = ["20260917160208", "20260917160223"]
EPOCHS = SCIENCE + CADENCE
FRAMES = {SCIENCE[0]: 11, SCIENCE[1]: 33, SCIENCE[2]: 33, SCIENCE[3]: 33, CADENCE[0]: 4, CADENCE[1]: 4}
CLASSES = [(0, 1), (0, 8), (0, 32), (0, 64), (0, 128), (0, 255), (1, 0), (1, 32), (2, 0), (3, 0)]
COUNTS = {(0, 1): 1020, (0, 8): 992, (0, 32): 896, (0, 64): 768, (0, 128): 512, (0, 255): 4, (1, 0): 768,
          (1, 32): 672, (2, 0): 512, (3, 0): 256}
BANDS = {33: (552, [548, 549, 550, 551, 552]), 37: (491, [487, 488, 489, 490, 491])}
LEVEL_CLASSES = [(0, 1), (0, 32), (0, 64), (0, 128), (0, 255), (1, 0), (2, 0), (3, 0)]


def _name(c):
    ew, ns = c
    return f"ns{ns}" if ew == 0 else (f"ew{ew}" if ns == 0 else f"ew{ew}ns{ns}")


def _level(c):
    return "cadence_tau.csv" if c == (0, 1) else f"cadence_tau_{_name(c)}.csv"


def _excess(band, ev, c, pol):
    """A fixed, decodable value: band 33 carries an excess growing with the class; the control band sits near 0."""
    i = EPOCHS.index(ev); j = CLASSES.index(c); p = {"0": 0, "1": 1, "I": 2}[pol]
    base = 2.5e-4 * (1 + j) if band == 33 else 1.0e-5
    return base * (1.0 + 0.125 * i) * (1.0 + 0.0625 * p) + (1e-6 * (i - j) if band == 37 else 0.0)


def _write_capture_files(d, pols):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "class_excess_epochs.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["channel", "epoch", "ew", "ns", "pol", "excess", "n_bins"])
        for ev in EPOCHS:
            for c in CLASSES:
                for pol in pols:
                    for band in sorted(BANDS):
                        empty = band == 33 and c == (0, 255) and ev == CADENCE[1]
                        w.writerow([band, ev, c[0], c[1], pol, "" if empty else f"{_excess(band, ev, c, pol):.6e}",
                                    0 if empty else len(BANDS[band][1])])
    with open(d / "class_floor_bins.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "ew", "ns", "pol", "freq_id", "excess"])
        for ev in EPOCHS:
            for c in CLASSES:
                for pol in pols:
                    for k, fid in enumerate(BANDS[37][1]):
                        w.writerow([ev, c[0], c[1], pol, fid, f"{_excess(37, ev, c, pol) * (1 + 0.5 * (k - 2)):.6e}"])


def _write_levels(d, suffix, pol_label):
    statuses = ["measured", "bound", "constant", "refused: trim spread 3.1", "measured", "no cadence lags",
                "measured", "bound"]
    for j, c in enumerate(LEVEL_CLASSES):
        path = d / (_level(c)[:-4] + suffix + ".csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["channel", "bins", "pol", "status", "tau_c_s", "tau_c_high_s", "G",
                                               "plateau", "plateau_se", "n_pairs", "n_classes", "n_epochs", "probes"])
            w.writeheader()
            for band in sorted(BANDS):
                st = statuses[(j + band) % len(statuses)]
                tau = {"measured": 15.0 + 30 * j, "bound": 1200.0, "constant": 86164.1}.get(st.split(":")[0], None)
                if st.startswith("refused"): tau = 86164.1
                for bins in ("pilot", "inband"):
                    probes = (f"75: {st.split(':')[0]} {int(tau or 0) + 5} | 90: {st.split(':')[0]} {int(tau or 0)} | "
                              f"95: measured {int(tau or 0) + 9}" if tau else "75: no cadence lags | 90: no cadence lags")
                    w.writerow({"channel": band, "bins": bins, "pol": pol_label, "status": st,
                                "tau_c_s": f"{tau:.1f}" if tau else "", "tau_c_high_s": f"{(tau or 0) * 1.5:.1f}" if tau else "",
                                "G": f"{min(tau, 86164.0905) / (16384 * 2.56e-6):.1f}" if tau else "", "plateau": "1.234e-07",
                                "plateau_se": "2.000e-08", "n_pairs": 12, "n_classes": 5, "n_epochs": 6, "probes": probes})


def _write_frame_residuals(d):
    cols = ["channel", "ew", "ns", "pol", "frames", "n_frames", "excess_pilot_bin", "excess_inband_median",
            "excess_inband_max", "phi_fast_pilot", "G_dump_pilot", "lag_zero_pilot", "kernel_excess_db_median"]
    for i, ev in enumerate(EPOCHS):
        with open(d / f"frame_residual_{ev}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(cols)
            for band in sorted(BANDS):
                for c in [(0, 1), (1, 0), (2, 0), (3, 0)]:
                    for pol in (0, 1):
                        v = (3e-3 if band == 33 else 2e-5) * (1 + 0.25 * i) * (1 + 0.5 * pol) * (1 + c[0])
                        w.writerow([band, c[0], c[1], pol, "all", FRAMES[ev], repr(v * 1.5), repr(v), repr(v * 2),
                                    "0.25", repr(1.5 + 0.125 * i + pol), 3, "1.0"])


def _write_lags(path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["channel", "cls", "bins", "a", "b", "lag_s", "lag_class", "rho", "rho_noise",
                                        "n_bins"])
        labels = ["pilot", "D1", "D2", "D3"]
        for band in sorted(BANDS):
            for j, c in enumerate(CLASSES):
                for a in range(4):
                    for b in range(a + 1, 4):
                        rho = (0.9 if band == 33 else 0.1) - 0.05 * j + 0.01 * (a + b)
                        w.writerow([band, _name(c) + "_x", "inband", labels[a], labels[b], 18000.0, 18000, repr(rho),
                                    "0.01", 5])


def _write_products(root):
    rng = np.random.default_rng(20260930)
    datasets = root / "datasets"
    keys = np.array([(ew, ns, p, p) for (ew, ns) in CLASSES for p in (0, 1)])
    count = np.array([COUNTS[(int(k[0]), int(k[1]))] for k in keys])
    for ev in EPOCHS:
        d = datasets / f"pilot_reduce_{ev}"; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "487.npz", keys=keys, count=count, stacks=np.zeros((FRAMES[ev], len(keys)), np.complex64))
    arch = root / "per_band"; arch.mkdir()
    for band, (fid, _) in BANDS.items():
        if band == 37:
            continue          # the control band has no archive product
        n = 500
        np.savez(arch / f"{fid}.npz", valid=(rng.random((n, 1)) > 0.05), target_norm_sq=np.array([0.5]),
                 reference_norm_sum_sq=np.array([2.0]), coarse_power_ratio=rng.gamma(40.0, 0.025, (n, 1)) * 0.5)
    runs = root / "runs"
    for ev in SCIENCE:
        for tag, shift in (("k230", 1.0), ("k230_ch33measured", 0.9)):
            d = runs / f"kernel_{ev}_{tag}"; d.mkdir(parents=True)
            n = FRAMES[ev]; scale = 1.0 + 0.1 * SCIENCE.index(ev)
            F = rng.gamma(40.0, 0.025, (n, 2)) * np.array([0.5 * scale * shift, 1.0])
            np.savez(d / "chime_detector_outputs.npz", physical_channel=np.array([33, 37]), coarse_power_ratio=F,
                     null_power_ratio=np.array([0.5, 1.0]), valid=np.ones((n, 2), bool))
    rescan = root / "rescan"; rescan.mkdir()
    np.savez(rescan / "chime_detector_outputs.npz", coarse_power_ratio=rng.gamma(40.0, 0.025, (400, 1)) * 0.45,
             null_power_ratio=np.array([0.45]), valid=np.ones((400, 1), bool), physical_channel=np.array([33]))
    return datasets, arch, runs, rescan


def _write_board(path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["channel", "tau_quality", "tau_booked_s", "G_nocredit"])
        w.writerow([33, "measured", "8423.0", "200822.4"])


def _write_tau_bounds(path):
    from pilot_proxy.capture import tau_bounds
    rows = []
    for censor, source, calibrate in tau_bounds.SETTINGS:
        for c, lb in (((0, 32), 0.579), ((3, 0), 2.2)):
            span = 32 * 16384 * 2.56e-6
            rows.append({"censor": censor, "source": source, "calibrate": calibrate, "channel": 33, "ew": c[0], "ns": c[1],
                         "lane": lb, "check": lb * 1.25,
                         "variants": json.dumps({"lane a": lb, "check b": lb * 1.25}), "lb": lb, "least_variant": "lane a",
                         "span": span, "lb_priced": min(lb, span), "capped": lb > span, "b4": False,
                         **{f"calibration_{k}": v for k, v in (dict(fcr=0.0, n_at_or_above=0, n_cells=91, fcr_same_class=0.0,
                                                                    null_max_s=0.121, null_max_same_class_s=0.1,
                                                                    passes=True) if calibrate else {}).items()}})
    for r in rows:
        for k in tau_bounds.COLUMNS:
            r.setdefault(k, None)
    tau_bounds.write(rows, path)


def build_inputs(root: Path) -> SimpleNamespace:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    cap = root / "capture"; si = root / "stokes_i"
    _write_capture_files(cap, ("0", "1")); _write_capture_files(si, ("I",))
    _write_levels(cap, "", "0"); _write_levels(cap, "_pol0", "0"); _write_levels(cap, "_pol1", "1")
    _write_levels(si, "", "I")
    _write_frame_residuals(cap)
    _write_lags(root / "lags.csv")
    datasets, arch, runs, rescan = _write_products(root)
    _write_board(root / "board.csv")
    _write_tau_bounds(root / "tau_bounds.csv")
    return SimpleNamespace(capture_dir=str(cap), stokes_dir=str(si), lags=str(root / "lags.csv"),
                           detector_runs=str(runs), datasets=str(datasets), archive_products=str(arch),
                           measured_marker_rescan=str(rescan), tau_bounds=str(root / "tau_bounds.csv"),
                           board=str(root / "board.csv"), project=None)


def characterize(tmp: Path) -> Path:
    from pilot_proxy.capture import oc_table
    args = build_inputs(Path(tmp) / "inputs")
    table, summary, manifest = oc_table.build(args)
    return oc_table.write_handoff(Path(tmp) / "out" / "capture", table=table, summary=summary,
                                  tau_bounds_path=args.tau_bounds, manifest=manifest)


def write_golden(handoff: Path, dest: Path) -> Path:
    """The handoff with the manifest's input paths made relative to the fixture root and the npz digests (which
    depend on the numpy that wrote them) replaced by 'fixture'; the producer is dropped."""
    from pilot_proxy.capture import oc_table
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    for name in (oc_table.TABLE, oc_table.SUMMARY, oc_table.TAU_BOUNDS):
        (dest / name).write_bytes((Path(handoff) / name).read_bytes())
    manifest = json.loads((Path(handoff) / oc_table.MANIFEST).read_text(encoding="utf-8"))
    manifest.pop("producer", None)
    base = os.path.commonpath(list(manifest["inputs"]))
    manifest["inputs"] = {os.path.relpath(p, base): ("fixture" if p.endswith(".npz") else d)
                          for p, d in sorted(manifest["inputs"].items())}
    from pilot_proxy.characterization.oc_table import canonical_json
    (dest / oc_table.MANIFEST).write_text(canonical_json(manifest), encoding="utf-8")
    return dest


if __name__ == "__main__":
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write-golden", action="store_true", help="rewrite tests/golden/handoff/capture")
    if not parser.parse_args().write_golden:
        parser.error("nothing to do: pass --write-golden")
    with tempfile.TemporaryDirectory() as scratch:
        write_golden(characterize(Path(scratch)), GOLDEN)
    print(f"wrote {GOLDEN}")
