"""Monthly marker-region spectrograms of every band, from the per-pilot products.

For every frame the histograms can use (health selected, finite frame time,
finite positive Q, any era or none; the ``histogram_eligible`` mask of the
histogram frame export), it takes the stored per-frame PSD, feeds summed,
which the products code at 0.01 dB relative to the frame's own median positive
fine bin across the coarse channel, and keeps the 501 fine bins centred on
the nominal marker. The median over the frames of each UTC month gives one
column; a month with no such frame stays empty. The fine axis is converted to
sky frequency with the settled inverted sense,
``sky = coarse centre - m * f_s/nfft`` for signed fine bin ``m``, so a
positive fine bin is a lower sky frequency, and the rows are stored in
ascending sky frequency. The saved eras of the characterization release, with
the cause the dating run recorded for every boundary, are copied beside it as
exact gzipped bytes.

Every input is checked before a byte is read: each product against the
sha256 its band's ledger records, each frame export against the digest in the
histogram release manifest, each ledger and era file against the
characterization release manifest. The two release manifests are arguments,
and their sha256 values are recorded in the output manifest.

    pilot-proxy characterize spectrogram --products DIR --frames DIR
        --frames-release-manifest FILE --release DIR --eras-release-manifest FILE --out DIR

The dissertation's renderer (``figure_src/archive_spectrogram.py``) reads the
output; this module is its reduction.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

from pilot_proxy.config.project import default_project

_PROJECT = default_project()
CHANNELS = tuple(int(label) for label in _PROJECT.frequency_plan.labels("screened"))
NFFT = _PROJECT.detector_config.nfft
FINE_HZ = _PROJECT.detector_config.psd_bin_hz(_PROJECT.instrument)
HALF = 250
CAUSES = ("archive start", "spectral-state transition", "power transition")
ARRAYS = ("channel", "month_index", "psd_db", "frames", "sky_offset_hz",
          "nominal_fine_bin", "nominal_bin_fraction")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def npy_bytes(array) -> bytes:
    import numpy as np
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_npz(path: Path, arrays: dict) -> None:
    """A zip of .npy members with fixed timestamps, so equal arrays give equal bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ARRAYS:
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, npy_bytes(arrays[name]), compresslevel=9)
    path.write_bytes(buffer.getvalue())


def read_manifest(path: Path | str) -> bytes:
    """The raw bytes of a release manifest (gunzipped when the file is ``.gz``)."""
    raw = Path(path).read_bytes()
    return gzip.decompress(raw) if str(path).endswith(".gz") else raw


def reduce_products(*, products: Path, frames: Path, frames_release_manifest: Path, release: Path,
                    eras_release_manifest: Path, out: Path) -> None:
    """The monthly reduction of every screened band into ``out`` (a new directory)."""
    import numpy as np
    PRODUCTS, FRAMES, DATA = Path(products), Path(frames), Path(out)
    ERAS = Path(release) / "archive" / "channels"
    if DATA.exists():
        raise ValueError(f"{DATA} exists; the reduction writes a new directory")
    corrected_raw = read_manifest(eras_release_manifest)
    RELEASE_SHA256 = sha(corrected_raw)
    corrected_release = json.loads(corrected_raw)["files"]
    histogram_release = json.loads(read_manifest(frames_release_manifest))["files"]
    ledgers = {}
    for path in sorted((Path(release) / "archive" / "ledger" / "channels").glob("ch*_fid*.json")):
        raw = path.read_bytes()
        relative = f"archive/ledger/channels/{path.name}"
        if sha(raw) != corrected_release[relative]["sha256"] or len(raw) != corrected_release[relative]["bytes"]:
            raise ValueError(f"ledger differs from the characterization release: {path.name}")
        ledger = json.loads(raw)
        ledgers[int(ledger["channel"])] = ledger
    if tuple(sorted(ledgers)) != CHANNELS:
        raise ValueError("the release's ledgers do not cover the screened bands")
    reduced, provenance = {}, {}
    for ch in CHANNELS:
        ledger = ledgers[ch]
        product = PRODUCTS / ledger["product"]
        product_raw_sha = hashlib.sha256()
        with product.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                product_raw_sha.update(block)
        if product_raw_sha.hexdigest() != ledger["product_sha256"]:
            raise ValueError(f"product differs from its ledger: channel {ch}")
        export_path = FRAMES / f"ch{ch}.npz"
        export_raw = export_path.read_bytes()
        expected = histogram_release[f"frames/ch{ch}.npz"]
        if sha(export_raw) != expected["sha256"] or len(export_raw) != expected["bytes"]:
            raise ValueError(f"frame export differs from the histogram release: channel {ch}")
        with np.load(io.BytesIO(export_raw), allow_pickle=False) as export:
            eligible = np.asarray(export["histogram_eligible"], dtype=bool)
            month = np.asarray(export["frame_month"], dtype=np.int64)
            export_units = np.asarray(export["frame_unit_index"], dtype=np.int64)
        with np.load(product, allow_pickle=False) as z:
            if (int(np.asarray(z["physical_channel"]).ravel()[0]) != ch
                    or int(np.asarray(z["freq_id"]).ravel()[0]) != ledger["freq_id"]
                    or int(np.asarray(z["nfft"]).ravel()[0]) != NFFT):
                raise ValueError(f"product identity or transform length changed: channel {ch}")
            if not np.array_equal(np.asarray(z["frame_unit_index"], dtype=np.int64), export_units):
                raise ValueError(f"frame order differs between product and export: channel {ch}")
            step = float(np.asarray(z["psd_db_step_per_code"]))
            invalid = int(np.asarray(z["psd_db_invalid_code"]))
            center = float(np.asarray(z["chime_frequency_hz"]).ravel()[0])
            pilot = float(np.asarray(z["pilot_frequency_hz"]).ravel()[0])
            codes = np.asarray(z["psd_frame_db_i16"])
        if step != 0.01 or invalid != -32768 or codes.shape != (eligible.size, NFFT):
            raise ValueError(f"PSD coding changed: channel {ch}")
        # Inverted sense: sky = center - m * FINE_HZ, so the pilot sits at
        # m = (center - pilot) / FINE_HZ; FFT order stores bin m at m mod NFFT.
        exact = (center - pilot) / FINE_HZ
        nominal = int(round(exact))
        offsets = np.arange(-HALF, HALF + 1)
        window = codes[:, (nominal + offsets) % NFFT][eligible]
        del codes
        sky = (exact - (nominal + offsets)) * FINE_HZ           # descending with m
        order = np.argsort(sky)
        window, sky = window[:, order], sky[order]
        reduced[ch] = (month[eligible], window, sky, nominal, exact - nominal)
        provenance[ch] = {
            "product": ledger["product"], "product_sha256": ledger["product_sha256"],
            "frame_export": f"frames/ch{ch}.npz", "frame_export_sha256": expected["sha256"],
            "frames_used": int(eligible.sum()),
            "invalid_codes_in_window": int(np.count_nonzero(window == invalid)),
            "chime_frequency_hz": center, "pilot_frequency_hz": pilot,
        }
        print(f"channel {ch}: {eligible.sum()} frames", flush=True)
    first = min(int(m.min()) for m, *_ in reduced.values())
    last = max(int(m.max()) for m, *_ in reduced.values())
    months = np.arange(first, last + 1, dtype=np.int32)
    psd = np.full((len(CHANNELS), months.size, 2 * HALF + 1), np.nan, dtype=np.float32)
    frames = np.zeros((len(CHANNELS), months.size), dtype=np.int32)
    sky_axis = np.zeros((len(CHANNELS), 2 * HALF + 1))
    nominal_bins = np.zeros(len(CHANNELS), dtype=np.int32)
    fractions = np.zeros(len(CHANNELS))
    for i, ch in enumerate(CHANNELS):
        month, window, sky, nominal, fraction = reduced.pop(ch)
        sky_axis[i], nominal_bins[i], fractions[i] = sky, nominal, fraction
        for k, value in enumerate(months):
            rows = window[month == value]
            frames[i, k] = rows.shape[0]
            if rows.shape[0]:
                level = rows.astype(np.float64)
                level[rows == -32768] = np.nan
                psd[i, k] = np.nanmedian(level, axis=0) * 0.01
    arrays = {"channel": np.asarray(CHANNELS, dtype=np.int16), "month_index": months,
              "psd_db": psd, "frames": frames, "sky_offset_hz": sky_axis,
              "nominal_fine_bin": nominal_bins, "nominal_bin_fraction": fractions}
    DATA.mkdir(parents=True, exist_ok=True)
    write_npz(DATA / "spectrogram.npz", arrays)
    stored = {"spectrogram.npz": {
        "stored_sha256": sha((DATA / "spectrogram.npz").read_bytes()),
        "stored_bytes": (DATA / "spectrogram.npz").stat().st_size,
        "array_sha256": {name: sha(npy_bytes(arrays[name])) for name in ARRAYS}}}
    for ch in CHANNELS:
        relative = f"archive/channels/ch{ch}/eras.json"
        raw = (ERAS / f"ch{ch}/eras.json").read_bytes()
        if sha(raw) != corrected_release[relative]["sha256"] or len(raw) != corrected_release[relative]["bytes"]:
            raise ValueError(f"eras differ from the corrected release: channel {ch}")
        target = DATA / f"eras/ch{ch}_eras.json.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
        stored[f"eras/ch{ch}_eras.json.gz"] = {
            "encoding": "gzip; exact original bytes; mtime=0",
            "source_relative_path": relative, "source_sha256": sha(raw), "source_bytes": len(raw),
            "stored_sha256": sha(target.read_bytes()), "stored_bytes": target.stat().st_size}
    manifest = {
        "schema": "dissertation-archive-spectrogram-v1",
        "scope": "Descriptive monthly view of the archive pilot region for the Appendix C plates; no frame is selected, no era is inferred and no number feeds a decision.",
        "reduction": {
            "script": "pilot_proxy/characterization/spectrogram.py",
            "script_sha256": sha(Path(__file__).read_bytes()),
            "products_dir": str(PRODUCTS),
            "frame_export_dir": str(FRAMES),
            "frame_selection": "histogram_eligible of the histogram release export: health selected, finite frame time, finite positive Q, any era or none",
            "month_key": "frame_month of the export, year * 12 + month - 1, UTC",
            "statistic": "per-bin median over the month's frames of psd_frame_db_i16 * 0.01; invalid code -32768 excluded",
            "psd_reference": "each frame's own median positive fine bin across the whole coarse channel, feeds summed (pilot_proxy.archive.detector._append_psd_frame)",
            "fine_bin_hz": FINE_HZ, "half_width_bins": HALF,
            "sky_frequency": "sky = chime_frequency_hz - m * fine_bin_hz for signed fine bin m (inverted sense of 2026-09-21); rows stored in ascending sky offset from pilot_frequency_hz",
        },
        "eras_source": {"release_manifest_sha256": RELEASE_SHA256,
                        "tree": str(ERAS),
                        "cause_field": "eras[].evidence: what cut the start of each saved era"},
        "channels": {str(ch): provenance[ch] for ch in CHANNELS},
        "stored": stored,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {DATA}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pilot-proxy characterize spectrogram", description=__doc__.split("\n\n")[0])
    parser.add_argument("--products", type=Path, required=True, help="directory of the per-pilot products")
    parser.add_argument("--frames", type=Path, required=True, help="the histogram frame export (frames/)")
    parser.add_argument("--frames-release-manifest", type=Path, required=True,
                        help="the histogram release's manifest (its files carry the frame exports' digests)")
    parser.add_argument("--release", type=Path, required=True,
                        help="the characterization release (archive/ledger/channels and archive/channels)")
    parser.add_argument("--eras-release-manifest", type=Path, required=True,
                        help="that release's manifest (json or json.gz), which authenticates its ledgers and eras")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    reduce_products(products=args.products, frames=args.frames, frames_release_manifest=args.frames_release_manifest,
                    release=args.release, eras_release_manifest=args.eras_release_manifest, out=args.out)
    return 0


__all__ = ["ARRAYS", "CAUSES", "CHANNELS", "HALF", "main", "read_manifest", "reduce_products", "write_npz"]
