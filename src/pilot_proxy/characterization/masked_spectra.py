"""What the operating point does to the spectrum, measured on the held-out half.

The point comes from the calibration block (:mod:`.surface`, the knee); this applies it
to the evaluation block, which the calibration never saw, and measures what it
removed. That is the closing measurement of the offline protocol: a mask chosen
on one half and scored on the other.

Three spectra, and the one subtraction that is legitimate. Over the evaluation
block's ``N`` frames, with ``K`` of them kept:

    P_all           = (1/N) sum over all N frames
    P_keep_contrib  = (1/N) sum over the K kept frames      <- the same denominator
    P_keep_given    = (1/K) sum over the K kept frames      <- what the kept data looks like

``P_all - P_keep_contrib`` is the contribution the mask removed, in linear
power on a common denominator, and it is non-negative by construction. It is
*not* ``P_all - P_keep_given``: those two have different denominators, and
their difference is not a removed contribution but an artefact of the masked
fraction. The plates draw ``P_all`` and ``P_keep_given``; the removed
contribution is tabulated.

Reported per channel: the removed contribution integrated over the designated
window and over the census window, the suppression at the pilot bin in dB, the
masked fraction and residual achieved on the evaluation block, and the same
integrals for the survey flag as the reference every operating point is
measured against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from pilot_proxy.detectors.narrowband_marker import psd
from pilot_proxy.detectors.narrowband_marker.scores import build_score_bundle
from pilot_proxy.products.reader import NFFT, Product

from . import surface

WINDOW_HZ = psd.WINDOW_HZ
DESIGNATED_HZ = psd.span_half_width_hz(128)      # the K = 128 span the decision reads


@dataclass(frozen=True)
class MaskedSpectra:
    """One channel's before-and-after on the evaluation block."""

    channel: int
    freq_id: int
    frames: int                       # N: evaluation frames with a spectrum
    kept: int                         # K
    masked_fraction: float
    rho: int
    eta_q16: int
    eta: float
    basis: str                        # which point this is ('operating point', 'survey flag', ...)
    rf_offset_hz: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    p_all: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    p_keep_contrib: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    p_keep_given: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    removed_window: float = math.nan          # integrated removed contribution, |rf| <= W
    removed_designated: float = math.nan      # the same over the K = 128 span
    all_window: float = math.nan
    all_designated: float = math.nan
    pilot_suppression_db: float = math.nan    # P_all / P_keep_given at the pilot bin
    window_suppression_db: float = math.nan   # the same integrated over W
    notes: tuple[str, ...] = ()

    @property
    def removed_fraction_window(self) -> float:
        return self.removed_window / self.all_window if self.all_window > 0 else math.nan

    def as_row(self) -> dict:
        return {"channel": self.channel, "freq_id": self.freq_id, "basis": self.basis, "frames": self.frames,
                "kept": self.kept, "masked_fraction": self.masked_fraction, "rho": self.rho, "eta_q16": self.eta_q16,
                "eta": self.eta, "removed_window": self.removed_window, "removed_designated": self.removed_designated,
                "all_window": self.all_window, "all_designated": self.all_designated,
                "removed_fraction_window": self.removed_fraction_window,
                "pilot_suppression_db": self.pilot_suppression_db,
                "window_suppression_db": self.window_suppression_db, "notes": "; ".join(self.notes)}


def _accumulate(product: Product, block: np.ndarray, kept: np.ndarray, *, chunk: int = 2048):
    """One pass: the summed spectrum over the block and over its kept frames, plus the counts."""
    g = product.geometry
    rf = psd.centred_offset_hz(g.psd_rf_offset_hz(np.arange(NFFT)))
    order = np.argsort(rf, kind="stable")
    total = np.zeros(NFFT)
    kept_total = np.zeros(NFFT)
    n = k = 0
    for rows_slice, power in product.psd_rows(chunk):
        m = block[rows_slice]
        if not m.any():
            continue
        sub = power[m]                                  # the chunk's rows, already decoded
        finite = np.isfinite(sub).any(axis=1)           # before the fill: a frame with no finite bin is absent
        np.nan_to_num(sub, copy=False, nan=0.0)         # sub is our own copy; zeros drop out of the sum
        total += sub.sum(axis=0)
        n += int(finite.sum())
        km = kept[rows_slice][m]
        if km.any():
            kept_total += sub[km].sum(axis=0)
            k += int((km & finite).sum())
    return rf[order], total[order], kept_total[order], n, k


def measure(product: Product, block: np.ndarray, kept: np.ndarray, *, basis: str, rho: int, eta_q16: int,
            eta: float, chunk: int = 2048) -> MaskedSpectra:
    """The before-and-after spectra of one channel on one block under one decision."""
    block = np.asarray(block, dtype=bool) & product.selected
    kept = np.asarray(kept, dtype=bool) & block
    g = product.geometry
    notes: list[str] = []
    if not block.any():
        return MaskedSpectra(g.physical_channel, g.freq_id, 0, 0, math.nan, rho, eta_q16, eta, basis,
                             notes=("the block is empty",))
    rf, total, kept_total, n, k = _accumulate(product, block, kept, chunk=chunk)
    if n == 0:
        return MaskedSpectra(g.physical_channel, g.freq_id, 0, 0, math.nan, rho, eta_q16, eta, basis,
                             notes=("no frame of the block carries a spectrum",))
    p_all = total / n
    p_keep_contrib = kept_total / n                       # the common denominator: what the kept frames contribute
    p_keep_given = kept_total / k if k else np.zeros_like(total)
    if not k:
        notes.append("the decision keeps no frame of the block: there is no retained spectrum")
    removed = np.maximum(p_all - p_keep_contrib, 0.0)
    in_window = np.abs(rf) <= WINDOW_HZ
    in_designated = np.abs(rf) <= DESIGNATED_HZ
    pilot = int(np.argmin(np.abs(rf)))
    with np.errstate(divide="ignore", invalid="ignore"):
        pilot_db = 10.0 * math.log10(p_all[pilot] / p_keep_given[pilot]) if k and p_keep_given[pilot] > 0 else math.nan
        win_all, win_keep = float(p_all[in_window].sum()), float(p_keep_given[in_window].sum())
        window_db = 10.0 * math.log10(win_all / win_keep) if k and win_keep > 0 else math.nan
    return MaskedSpectra(
        channel=g.physical_channel, freq_id=g.freq_id, frames=n, kept=k,
        masked_fraction=1.0 - k / n, rho=int(rho), eta_q16=int(eta_q16), eta=float(eta), basis=basis,
        rf_offset_hz=rf[in_window], p_all=p_all[in_window], p_keep_contrib=p_keep_contrib[in_window],
        p_keep_given=p_keep_given[in_window],
        removed_window=float(removed[in_window].sum()), removed_designated=float(removed[in_designated].sum()),
        all_window=win_all, all_designated=float(p_all[in_designated].sum()),
        pilot_suppression_db=pilot_db, window_suppression_db=window_db, notes=tuple(notes))


def kept_at_point(product: Product, block: np.ndarray, *, anchor_bin: int, bulk_mask: np.ndarray,
                  rho: int, eta_q16: int) -> np.ndarray:
    """The frames of ``block`` a ``(rho, eta)`` decision keeps, from the exact integer fine powers."""
    bundle = build_score_bundle(
        product.path, np.asarray(block, dtype=bool), anchor_bin=int(anchor_bin),
        designated_half_width=surface.DESIGNATED_HALF_WIDTH, bulk_mask=np.asarray(bulk_mask, dtype=bool))
    required = np.asarray(bundle.requirements_by_rho()[int(rho)], dtype=object)
    kept = np.zeros(product.n_frames, dtype=bool)
    kept[bundle.source_row_index] = surface.kept_at(required, int(eta_q16))
    return kept


SPECTRA_COLUMNS = ("channel", "freq_id", "basis", "frames", "kept", "masked_fraction", "rho", "eta_q16", "eta",
                   "removed_window", "removed_designated", "all_window", "all_designated",
                   "removed_fraction_window", "pilot_suppression_db", "window_suppression_db", "notes")


def write_spectra_rows(results: Sequence[MaskedSpectra], path: Path | str) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SPECTRA_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for r in results:
            writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.as_row().items()})
    return path


def write_spectra_npz(results: Sequence[MaskedSpectra], path: Path | str) -> Path:
    """The drawn arrays: the figure and the plates read this rather than the products."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for r in results:
        if r.rf_offset_hz.size:
            tag = f"ch{r.channel:02d}_{r.basis.replace(' ', '_')}"
            arrays[f"{tag}_rf_offset_hz"] = r.rf_offset_hz
            arrays[f"{tag}_p_all"] = r.p_all
            arrays[f"{tag}_p_keep_contrib"] = r.p_keep_contrib
            arrays[f"{tag}_p_keep_given"] = r.p_keep_given
    np.savez_compressed(path, **arrays)
    return path
