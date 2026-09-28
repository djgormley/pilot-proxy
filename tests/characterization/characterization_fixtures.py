"""Synthetic archives for the characterization driver, and the golden handoff built from one.

The populated product is two years of monthly acquisitions with four
repeating interference levels (so the mask/residual surface is not flat) and,
optionally, per-frame spectra with a three-bin carrier at the nominal marker.
Its eras come from the section 8.1 rule with the support gates relaxed to one
frame, one acquisition and one day, which the small product needs.

Run as a script to rewrite ``tests/golden/handoff``:

    PYTHONPATH=src python tests/characterization/characterization_fixtures.py --write-golden
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "tests" / "products") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests" / "products"))

from test_pilotproxy_v5 import _replace, _write_product  # noqa: E402

from pilot_proxy.characterization import eras, oc_table, run  # noqa: E402
from pilot_proxy.config.project import default_project, default_project_dir  # noqa: E402
from pilot_proxy.products.npzio import load_npz  # noqa: E402
from pilot_proxy.products.reader import NFFT, Product  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "handoff"
BANDS = (29, 34)                      # both keep the rule's eras (no author-dated list)
RELAXED_ERAS = eras.EraConfig(min_frames=1, min_units=1, min_days=1, min_peak_cohort=1,
                              units_sensitivity=(1,), threshold_sensitivity_db=((0.5, 1.0),))
GENERATED = "2026-09-27T00:00:00+00:00"
GOLDEN_CANDIDATES = 3


def populated_product(path: Path, channel: int, *, frames: int = 384, units: int = 24, spectra: bool = True) -> Path:
    """A v5 product whose frames fill one proxy-low era of ``units`` monthly acquisitions."""
    path = _write_product(path, channel, frames=frames, units=units)
    times = np.array([dt.datetime(2023 + m // 12, 1 + m % 12, 2, tzinfo=dt.timezone.utc).timestamp()
                      for m in range(units)])
    values = load_npz(path)
    with Product(path) as product:
        geometry = product.geometry
    fine = values["fine_power_u64"]
    fine[:, :, :] = 20
    fine[:, 0, geometry.nominal_fine_bin] = np.tile([40, 80, 160, 500], frames // 4)
    changes = dict(unit_time0_ctime=times, unit_input_map_sha256=np.full(units, "a" * 64), fine_power_u64=fine,
                   baseband_power_linear=np.tile([4.0, 4.1, 5.0, 10.0], frames // 4)[:, None])
    reference = values["p_ref_sum_u64"]
    norm = float(values["target_norm_sq"][0]) / float(values["reference_norm_sum_sq"][0])
    target = np.rint(reference * norm * np.tile([0.99, 1.01, 1.08, 20.0], frames // 4)[:, None]).astype(np.uint64)
    ratio = target / (reference * norm)
    excess = ratio - 1.0
    excess_db = np.full(excess.shape, np.nan)
    excess_db[excess > 0] = 10 * np.log10(excess[excess > 0])
    shelf_offset = (float(values["pilot_below_data_db"])
                    - 10 * np.log10(float(values["dtv_bandwidth_hz"]) / float(values["bin_enbw_hz"]))
                    - 10 * np.log10(float(values["pilot_capture_efficiency"])))
    changes.update(p_target_u64=target, coarse_power_ratio=2.0 * target / reference,
                   normalized_coarse_power_ratio_db=10 * np.log10(ratio), normalized_pilot_excess=excess,
                   pilot_excess_db=excess_db, estimated_data_shelf_snr_db=excess_db + shelf_offset,
                   reject_mask=(ratio > 1).astype(np.uint8))
    if spectra:
        codes = np.zeros((frames, NFFT), dtype=np.int16)
        marker = int(round(geometry.nominal_psd_bin)) % NFFT
        codes[:, [(marker - 1) % NFFT, marker, (marker + 1) % NFFT]] = [1000, 2000, 1000]
        changes.update(psd_frame_db_i16=codes, psd_db_reference=np.ones((frames, 1)),
                       psd_db_step_per_code=np.float64(0.01), psd_db_invalid_code=np.int16(-32768))
    _replace(path, **changes)
    return path


def products_dir(tmp: Path, bands=BANDS) -> Path:
    project = default_project()
    directory = tmp / "products"
    directory.mkdir(parents=True, exist_ok=True)
    for band in bands:
        populated_product(directory / f"{project.target_freq_id(str(band))}.npz", band)
    return directory


def project_copy(tmp: Path) -> Path:
    """A writable copy of the repository's project profile."""
    target = tmp / "project"
    shutil.copytree(default_project_dir(), target)
    return target


def characterize(tmp: Path, *, products: Path | None = None, project_dir: Path | None = None, replay=(29,),
                 record_name: str | None = None, out: str = "out") -> tuple[dict, Path]:
    """Run the driver in-process on the fixture products; returns the summary and ``<out>/characterization``."""
    products = products or products_dir(tmp)
    replay_file = None
    if replay is not None:
        replay_file = tmp / "replay_points.csv"
        replay_file.write_text("band_id\n" + "".join(f"{b}\n" for b in replay), encoding="utf-8")
    result = run.characterize_archive(products, tmp / out, project_dir=project_dir or default_project_dir(),
                                      workers=1, replicates=8, seed=91, era_config=RELAXED_ERAS,
                                      replay_points=replay_file, record_name=record_name, generated=GENERATED)
    return result, tmp / out / "characterization"


def write_golden(characterization: Path, dest: Path) -> Path:
    """The first candidates of each band's fine calibration surface, their families' summary rows, and a manifest
    without the producer (informational, and it names the commit) or the local replay path."""
    dest.mkdir(parents=True, exist_ok=True)
    rows = oc_table.read_rows(characterization / "oc_table.csv")
    chosen, seen = [], {}
    for row in rows:
        key = row["band_id"]
        if (row["candidate_set"], row["block"]) != ("fine_surface", "calibration"):
            continue
        if seen.get(key, 0) < GOLDEN_CANDIDATES:
            chosen.append(row)
            seen[key] = seen.get(key, 0) + 1
    families = {(r["band_id"], r["candidate_set"], r["population"], r["threshold_family"]) for r in chosen}
    summary = [r for r in oc_table.read_rows(characterization / "oc_summary.csv")
               if (r["band_id"], r["candidate_set"], r["population"], r["threshold_family"]) in families]
    _write_text_rows(chosen, dest / "oc_table.csv", oc_table.OC_COLUMNS)
    _write_text_rows(summary, dest / "oc_summary.csv", oc_table.SUMMARY_COLUMNS)
    manifest = json.loads((characterization / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("producer", None)
    manifest.pop("files", None)
    manifest.pop("tables", None)
    if isinstance(manifest.get("inputs", {}).get("replay_points"), dict):
        manifest["inputs"]["replay_points"]["path"] = Path(manifest["inputs"]["replay_points"]["path"]).name
    oc_table.write_manifest(dest, manifest, ("oc_table.csv", "oc_summary.csv"))
    return dest


def _write_text_rows(rows, path: Path, columns) -> None:
    """Rows read back from a written table are text already; they are written as read."""
    import csv

    names = oc_table.column_names(columns)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(names), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in names})


if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write-golden", action="store_true", help="rewrite tests/golden/handoff")
    args = parser.parse_args()
    if not args.write_golden:
        parser.error("nothing to do: pass --write-golden")
    with tempfile.TemporaryDirectory() as scratch:
        summary, characterization = characterize(Path(scratch))
        if summary["errors"]:
            raise SystemExit(summary["errors"])
        write_golden(characterization, GOLDEN)
    print(f"wrote {GOLDEN}")
