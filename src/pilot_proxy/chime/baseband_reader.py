"""
Reader: CHIME / outrigger baseband HDF5 (offset-binary 4+4-bit).

The concrete reader the bundled example runs on. It owns the on-disk format
knowledge via `_baseband_format` (see that module for the exact dataset and
attribute layout) and yields the [nfft, n_feeds] complex64 frames an analyzer
consumes. CHIME and its outriggers (KKO/GBO/HCO) share this product, so this one reader
serves them all. A different file format needs a different reader, not a change
here -- see docs/ADDING_A_READER.md (this file is a worked example).
"""
from __future__ import annotations

from typing import Iterable, Iterator, Mapping

from pilot_proxy.archive.interfaces import (
    READY,
    STREAM_COMPLEX_BASEBAND,
    PluginInfo,
    Reader,
    RunContext,
)
from pilot_proxy.archive.names import baseband_filename

from . import baseband_format as fmt
from .unreadable import unreadable_file

# A truncated baseband object can still be valid HDF5, but archive objects this
# small cannot contain even a useful fraction of the native payload. The floor
# belongs to this format, not to the generic CADC source: other readers may
# legitimately describe much smaller products.
MINIMUM_ARCHIVE_BYTES = 1 << 20


class ChimeBasebandReader(Reader):
    # Schema 2 renames the file-size-derived ``n_frames`` column: an HDF5
    # object's total archive size includes container overhead (and may include
    # compression), so it is useful planning metadata but not an exact count.
    survey_schema = 2
    minimum_archive_bytes = MINIMUM_ARCHIVE_BYTES

    info = PluginInfo(
        name="chime-baseband",
        kind="reader",
        summary="CHIME/outrigger baseband HDF5 (offset-binary 4+4-bit, "
                "dataset 'baseband', attr 'freq' in MHz).",
        status=READY,
        instruments=("chime", "kko", "gbo", "hco"),
        requires=("h5py",),
        stream_kind=STREAM_COMPLEX_BASEBAND,
        notes="Yields [nfft, n_feeds] complex64 frames. Shared by CHIME and all "
              "CHIME-compatible outriggers.",
    )

    def probe(self, path: str) -> Mapping[str, object]:
        with unreadable_file():
            f_center_hz = fmt.channel_center_hz(path)
        return {"f_center_hz": f_center_hz,
                "f_center_mhz": f_center_hz / fmt.HZ_PER_MHZ,
                "fs_hz": fmt.FS}

    def iter_arrays(self, path: str, ctx: RunContext) -> Iterator:
        nfft = int(getattr(ctx.instrument, "nfft", fmt.NFFT) or fmt.NFFT)
        if nfft < 1:
            raise ValueError("reader nfft must be positive")
        with unreadable_file():
            yield from fmt.iter_frames(path, nfft=nfft)

    # -- archive file shape ------------------------------------------------
    # Baseband: one HDF5 per freq_id per event. `selection` is the survey's
    # resolved freq_id list (see CadcDatatrailSource.survey), which is exactly
    # how this product is selected.
    def survey_files(self, event, common_path, selection,
                     ctx: RunContext) -> Iterable[tuple]:
        for ch in (selection or ()):
            yield baseband_filename(event, ch), {"freq_id": int(ch)}

    def annotate_row(self, row: dict, instrument) -> None:
        """Geometry annotation plus an explicitly approximate frame count.

        The packed payload is one byte per feed per sample, but ``size_bytes``
        is the size of the entire HDF5 object. Report a whole-frame planning
        estimate rather than presenting a fractional value as an exact count.
        """
        if instrument is None or row.get("freq_id") is None:
            return
        ch = int(row["freq_id"])
        bytes_per_frame = instrument.nfft * instrument.n_feeds
        row["freq_mhz"] = round(instrument.freq_of_freq_id(ch), 4)
        row["n_frames_estimate"] = int(
            row.get("size_bytes", 0) // bytes_per_frame)
