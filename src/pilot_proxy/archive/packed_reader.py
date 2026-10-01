# coding=utf-8
"""Read CHIME/outrigger baseband HDF5 as native packed bytes.

The complex reader unpacks each chunk to complex64. The local-reference
power-ratio detector instead needs the original 4-bit nibbles so they can be
re-packed *losslessly* into the kernel's two's-complement int4 layout (via
``pack_chime_block_for_detector``) with no float requantization. This reader is
the same layout but yields raw ``uint8 [nfft, n_feeds]`` blocks.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterator, Mapping

import h5py
import numpy as np

from pilot_proxy.archive.interfaces import READY, PluginInfo, Reader, RunContext
from pilot_proxy.chime import baseband_format as fmt
from pilot_proxy.chime.unreadable import unreadable_file

from .stream_kinds import STREAM_PACKED_COMPLEX_INT4_BASEBAND


def _dtype_signature(dtype: np.dtype) -> object:
    if dtype.names:
        return {
            "fields": [
                [name, _dtype_signature(dtype.fields[name][0]), dtype.fields[name][1]]
                for name in dtype.names
            ],
            "itemsize": dtype.itemsize,
        }
    if dtype.subdtype:
        base, shape = dtype.subdtype
        return {"base": _dtype_signature(base), "shape": list(shape)}
    return dtype.str


def _dataset_content_sha256(dataset: h5py.Dataset) -> str:
    """Hash an HDF5 dataset's typed values in a stable order."""
    values = np.asarray(dataset[...])
    if values.dtype.hasobject:
        raise ValueError("index_map/input must not contain object values")
    header = json.dumps(
        {"dtype": _dtype_signature(values.dtype), "shape": list(values.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def _iter_packed_chunks(path: str, nfft: int) -> Iterator:
    """Yield raw uint8 [nfft, n_feeds] blocks (final partial frame dropped).

    A file too short for one complete transform at this nfft raises
    UnreadableUnitError (via unreadable_file). Raised while the reader
    streams, that type is the engine's one quarantinable iteration error:
    the unit lands in the quarantine ledger and a rerun resumes without it,
    whereas letting the analyzer see zero frames turns the same bytes into
    an unquarantinable run-level ValueError. The check lives here, on the
    same nfft the framing uses, so the reject and framing criteria cannot
    diverge the way a probe-time gate at the format default could.
    """
    with unreadable_file(), h5py.File(path, "r") as h:
        bb = h["baseband"]
        n_chunks = int(bb.shape[0]) // int(nfft)
        if n_chunks == 0:
            raise ValueError(
                "CHIME packed baseband is shorter than one transform: "
                f"{int(bb.shape[0])} time samples < nfft {int(nfft)}; "
                "it cannot yield a complete frame"
            )
        for c in range(n_chunks):
            yield bb[c * nfft:(c + 1) * nfft, :]


class ChimeBasebandPackedReader(Reader):
    info = PluginInfo(
        name="chime-baseband-packed",
        kind="reader",
        summary="CHIME/outrigger baseband HDF5 yielded as raw offset-binary 4+4-bit "
                "uint8 (no unpack); for the F-stat detector's lossless re-pack.",
        status=READY,
        instruments=("chime", "kko", "gbo", "hco"),
        requires=("h5py", "pilot-proxy"),
        stream_kind=STREAM_PACKED_COMPLEX_INT4_BASEBAND,
        notes="Same on-disk format as 'chime-baseband'; yields uint8 [nfft, n_feeds] "
              "instead of complex64 so the kernel packing keeps the native int4 grid.",
    )

    def probe(self, path: str) -> Mapping[str, object]:
        # Classify open/schema failures as an unreadable unit (quarantine)
        # instead of crashing the scan, matching the 'chime-baseband' reader.
        with unreadable_file():
            return self._probe(path)

    @staticmethod
    def _probe(path: str) -> Mapping[str, object]:
        f_center_hz = fmt.channel_center_hz(path)
        meta: dict = {"f_center_hz": f_center_hz,
                      "f_center_mhz": f_center_hz / 1e6,
                      "fs_hz": fmt.FS,
                      "nfft": fmt.NFFT}
        # Absolute-time axis + provenance from the file's root attrs, surfaced so
        # the detector can stamp a per-unit time and derive a per-frame time as
        # time0_ctime + frame_in_unit*nfft*delta_time. A real CHIME baseband file
        # carries the full set; missing attrs degrade to NaN / 0 / None / ""
        # rather than failing the probe.
        with h5py.File(path, "r") as h:
            a = h.attrs
            baseband = h["baseband"]
            if baseband.ndim != 2:
                raise ValueError(
                    "CHIME packed baseband must have shape (time, input_stream); "
                    f"got {baseband.shape}."
                )
            meta["num_input_streams"] = int(baseband.shape[1])
            # A file too short to yield one complete transform is a property of
            # the staged bytes, not an analyzer fault, so reject it here where
            # unreadable_file() turns it into UnreadableUnitError and the engine
            # quarantines it WITHOUT interrupting the run. The framing length is
            # ctx.instrument.nfft and the engine calls probe without a ctx, so
            # this gate can only use the format default fmt.NFFT: every shipped
            # instrument frames at that value, and an instrument configured
            # below it would see files quarantined here that could still frame
            # (none exists; the gate is deliberately conservative for the
            # unattended multi-week scan). _iter_packed_chunks enforces the
            # per-run length again at the exact nfft it frames with, so an
            # instrument nfft above the default quarantines at iteration time
            # (a resumable stop) instead of aborting the run from the analyzer.
            if int(baseband.shape[0]) < int(fmt.NFFT):
                raise ValueError(
                    "CHIME packed baseband is shorter than one transform: "
                    f"{int(baseband.shape[0])} time samples < nfft {int(fmt.NFFT)}; "
                    "it cannot yield a complete frame"
                )

            def _f(name: str) -> float:
                return float(a[name]) if name in a else float("nan")

            def _s(name: str) -> str:
                if name not in a:
                    return ""
                v = a[name]
                return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)

            meta["time0_ctime"] = _f("time0_ctime")
            meta["time0_ctime_offset"] = _f("time0_ctime_offset")
            meta["delta_time"] = _f("delta_time")
            meta["time0_fpga_count"] = int(a["time0_fpga_count"]) if "time0_fpga_count" in a else 0
            meta["event_id"] = int(a["event_id"]) if "event_id" in a else None
            meta["archive_version"] = _s("archive_version")
            meta["git_version_tag"] = _s("git_version_tag")
            meta["collection_server"] = _s("collection_server")
            meta["input_map_sha256"] = (
                _dataset_content_sha256(h["index_map/input"])
                if "index_map/input" in h
                else ""
            )
        return meta

    def iter_arrays(self, path: str, ctx: RunContext) -> Iterator:
        nfft = int(getattr(ctx.instrument, "nfft", fmt.NFFT) or fmt.NFFT)
        return _iter_packed_chunks(path, nfft)


__all__ = ["ChimeBasebandPackedReader"]
