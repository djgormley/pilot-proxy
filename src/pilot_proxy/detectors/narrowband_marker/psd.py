"""The per-frame 23.84 Hz spectra: era-mean spectrum, dominant lobe, per-frame
peak offsets, and the K = 64/128/256 containment analysis behind K = 128.

Every v5 product retains one feed-summed power spectrum per frame
(``psd_frame_db_i16``: 16384 bins of ``f_s / 16384 = 23.84 Hz``, int16 codes
of 0.01 dB about the per-frame ``psd_db_reference``, ``-32768`` invalid).
Decoded through :meth:`Product.psd_rows` (linear power, NaN at the invalid
code) and read in row chunks, these spectra cover all three candidate spans
in the RF coordinate of ``docs/archive-results-design.md`` section 2 (offset
from the nominal pilot, positive towards higher RF), which is where the
containment analysis of chapter 4 has to be done: the fine axis measures the
carrier modulo ``f_s / K`` about the coarse grid, so the three spans are
nested only about the nominal pilot.

Definitions implemented, per channel and per boolean frame mask (the era):

era-mean spectrum
    The NaN-aware linear mean over the masked frames of every bin, kept for
    the full axis (the appendix plates) and reported within the census window
    ``|offset| <= W = 15 kHz`` in dB relative to the median positive bin of
    the window. ``top_frame_share`` (the largest single frame's share of the
    summed window power) says how far the mean is from being one frame.

dominant lobe (ch08 §156-171)
    The argmax of the era-mean within ``W`` excluding the instrumental
    channel-centre line ``|offset - centre_line| <= 60 Hz``, in Hz from
    nominal and dB over the window median, refined to sub-bin precision by a
    three-point parabola through ``log10`` of the peak bin and its two
    neighbours. "A lobe read from an averaged spectrum is not the same
    estimand as the on-minus-off anchor" (ch08), so it is reported beside
    the anchor of ``archive.anchors`` rather than in place of it. Two further
    argmaxes are reported: the strongest feature within ``+-5 kHz`` of nominal
    (the earlier ``ppcal`` reading's search radius) and the *in-span lobe*,
    the strongest feature within the campaign span ``|offset| <= f_s / 2K``
    at ``K = 128``; the in-span lobe counts as recovered only when it stands
    at least ``FEATURE_MIN_DB`` over the local baseline.

per-frame peak offsets (tab:calibration:anchors)
    For every coarse-detected frame in the mask (``reject_mask = 1``), the RF
    offset of that frame's raw-spectrum peak within ``W`` excluding the
    centre line; the median, 90th and 99th percentiles of ``|offset|`` and of
    the signed offset, and the fraction of those frames whose peak lies
    inside each span ``|offset| <= f_s / 2K``, ``K in {64, 128, 256}``.

pilot-associated energy and ``E_K(c)`` (ch04 eq:param:kstar)
    "the fraction of the pilot-associated energy of the channel's current-era
    spectrum that falls inside the unambiguous span ``+-f_s/2K`` around the
    nominal target". Pilot-associated energy is the excess of the era-mean
    over a robust local baseline: the sliding median of the era-mean over
    ``+-1.5 kHz`` whose source bins exclude the centre line (``+-60 Hz``),
    the K = 64 span about nominal (``+-f_s/128 = +-3.05 kHz``) and the same
    half-width about the dominant lobe, with bins whose sliding window is
    empty filled by linear interpolation in ``log10`` between the nearest
    defined bins (the baseline under the pilot region is therefore the
    annulus interpolation of the surrounding medians, never the pilot).
    ``excess = max(mean - baseline, 0)``, zero at centre-line bins;
    ``E_K = sum(excess * overlap_K) / sum(excess * overlap_W)`` with each
    bin weighted by the fraction of its 23.84 Hz width inside the interval,
    so the two edge bins of every hard window enter fractionally. The spans
    are nested about nominal, so ``E_K`` is non-increasing in ``K``.

axis
    Offsets are read on the receiver's circular axis centred on the nominal
    pilot: the per-frame spectrum is an FFT of the coarse channel, as is the
    detector's K-tap transform, so a feature beyond the coarse-channel edge
    nearest the pilot appears (to both) at the aliased offset on the far side
    of the pilot, one sample rate away. ``edge_distance_hz`` is the pilot's
    distance to that edge; ``window_aliased_hz`` is the width of ``W`` beyond
    it (content read there is aliased from the channel's other edge; zero on
    channels whose pilot sits more than ``W`` from either edge) and
    ``ref_aliased_K`` marks a reference passband that crosses it. Channels 21
    and 32 are the archive's cases.

reference-region contamination
    The excess over baseline inside the two reference passbands (centres
    ``+-2 f_s / K``, each ``+-f_s / 2K`` wide, on the circular axis, so a
    passband that crosses the coarse-channel edge is read where the detector
    reads it) as a fraction of the in-span excess at the same ``K``. The centre line's bins are *included* here (the
    detector's references integrate it), where ``E_K`` excludes them; the
    contaminant is named when the centre line falls in a passband
    (``centre_line``) or a feature in it exceeds ``FEATURE_MIN_DB`` over the
    baseline (``feature@<offset>Hz(<dB>dB)``).

straddle loss and margin (ch04 eq:dirichlet, ch05 §94)
    ``-10 log10 |D_K(delta)|^2`` with ``D_K(x) = sin(pi x) / (K sin(pi x / K))``
    and ``delta = |offset| K / f_s`` the in-span lobe's offset in coarse bins
    (``|D_K(1/2)|^2 = -3.92 dB`` at half a bin); the margin to the span edge
    ``f_s / 2K - |offset|`` in Hz and as a fraction of the half-span,
    negative when the lobe lies outside that span.

``K*`` (ch04 eq:param:kstar)
    ``max{K = 2^j : E_K(c) >= E_min for every supported channel c}`` at
    ``E_min = 0.9`` (policy) and 0.8, 0.95 (sensitivity); "a channel that
    fails ``E_min`` at every candidate is a sentinel channel that does not
    drag the choice"; the binding channel is the eligible channel with the
    smallest ``E_K`` at the first ``K`` that fails. Eligible channels are
    those whose disposition is not ``unsupported``.

disposition (design section 6)
    ``supported`` when an in-span lobe is recovered, it is the dominant
    excess within ``W`` outside the centre line, ``E_128 >= E_min`` and the
    K = 128 reference contamination is below 5% (the self-leakage bound);
    ``supported with sentinel`` when a lobe is recovered but a stronger
    out-of-span feature exists, ``E_128 < E_min``, contamination is at or
    above 5%, or (told by the caller) the fine anchor aliases an out-of-span
    feature or disagrees with the in-span lobe by more than the designated
    half-width (``anchor_lobe_offset_bins``); ``unsupported`` when no in-span
    lobe is recoverable.

Output columns of ``containment.csv`` (one row per channel; floats
``repr()``, NaN blank), in order:

``channel, freq_id, frames, frames_with_spectrum, detected_frames,
top_frame_share, centre_line_rf_offset_hz, centre_line_db,
window_median_power, edge_distance_hz, window_aliased_hz,
dominant_offset_hz, dominant_refined_offset_hz,
dominant_db, dominant_excess_db, near_offset_hz, near_db,
in_span_offset_hz, in_span_refined_offset_hz, in_span_db,
in_span_excess_db, in_span_recovered, out_of_span_offset_hz,
out_of_span_excess_db, peak_abs_median_hz, peak_abs_p90_hz,
peak_abs_p99_hz, peak_signed_median_hz, peak_signed_p90_hz,
peak_signed_p99_hz, frames_in_span_64, frames_in_span_128,
frames_in_span_256, excess_total, e_64, e_128, e_256,
ref_contamination_64, ref_contamination_128, ref_contamination_256,
ref_contaminant_64, ref_contaminant_128, ref_contaminant_256,
ref_aliased_64, ref_aliased_128, ref_aliased_256, straddle_loss_db_64, straddle_loss_db_128, straddle_loss_db_256,
margin_hz_64, margin_hz_128, margin_hz_256, margin_fraction_64,
margin_fraction_128, margin_fraction_256, disposition, reasons``

``frames_in_span_K`` are fractions of ``detected_frames``; ``excess_total`` is
the summed excess (linear power) within ``W``; the reference passbands are
read on the full circular axis (the K = 64 passbands reach 15.26 kHz, past
``W``); offsets are Hz from nominal;
``*_db`` are dB over the window median except ``*_excess_db`` (over the local
baseline). ``kstar.csv`` has one row per ``E_min``: ``e_min, k_star,
failing_k, binding_channel, binding_e, sentinels, eligible``. The window
arrays for the plates go to a JSON (per channel: ``rf_offset_hz, mean_db,
baseline_db, excess, count`` and the marker offsets), the full-axis means to
an ``.npz``.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pilot_proxy.products.reader import (
    FINE_BIN_HZ,
    NFFT,
    PSD_BIN_HZ,
    SAMPLE_RATE_HZ,
    Geometry,
    Product,
)

SPANS = (64, 128, 256)
TARGET_K = 128                                   # the campaign's tap length: the in-span lobe lives here
REFERENCE_OFFSET_BINS = 2                        # references at +-2 bins of the K-tap grid
WINDOW_HZ = 15_000.0                             # W: the census window (a declared choice)
CENTRE_LINE_HALF_WIDTH_HZ = 60.0                 # the instrumental channel-centre line
NEAR_NOMINAL_HZ = 5_000.0                        # the earlier reading's pilot search radius
BASELINE_HALF_WIDTH_HZ = 1_500.0                 # sliding-median half-width
BASELINE_EXCLUSION_HZ = SAMPLE_RATE_HZ / (2 * 64)  # 3051.76 Hz: K = 64 half-span, about nominal and the dominant lobe
FEATURE_MIN_DB = 3.0                             # a feature is a feature at this excess over baseline
CONTAMINATION_LIMIT = 0.05                       # the self-leakage bound of ch04 (4.5% at delta k = 2)
E_MIN = 0.9
E_MIN_SENSITIVITY = (0.8, 0.9, 0.95)
DEFAULT_CHUNK = 2048

SUPPORTED = "supported"
SUPPORTED_SENTINEL = "supported with sentinel"
UNSUPPORTED = "unsupported"


# ------------------------------------------------------------------ geometry
def span_half_width_hz(k: int) -> float:
    """``f_s / 2K``: the unambiguous half-span of a K-tap window."""
    return SAMPLE_RATE_HZ / (2.0 * k)


def reference_centres_hz(k: int) -> tuple[float, float]:
    """RF offsets of the two reference passbands, ``+-2 f_s / K`` about nominal."""
    d = REFERENCE_OFFSET_BINS * SAMPLE_RATE_HZ / k
    return (-d, d)


def dirichlet_power(delta, k: int) -> np.ndarray:
    """``|D_K(delta)|^2`` with ``D_K(x) = sin(pi x) / (K sin(pi x / K))`` (ch04 eq:dirichlet)."""
    x = np.asarray(delta, dtype=float)
    den = k * np.sin(np.pi * x / k)
    safe = den != 0.0
    ratio = np.where(safe, np.sin(np.pi * x) / np.where(safe, den, 1.0), 1.0)
    return ratio ** 2


def straddle_loss_db(offset_hz, k: int) -> np.ndarray:
    """``-10 log10 |D_K(delta)|^2`` at ``delta = |offset| K / f_s`` coarse bins."""
    delta = np.abs(np.asarray(offset_hz, dtype=float)) * k / SAMPLE_RATE_HZ
    return -10.0 * np.log10(dirichlet_power(delta, k))


def overlap_weights(rf_hz, lo: float, hi: float, bin_hz: float = PSD_BIN_HZ) -> np.ndarray:
    """Fraction of each bin (centred at ``rf_hz``, width ``bin_hz``) inside ``[lo, hi]``."""
    rf = np.asarray(rf_hz, dtype=float)
    left = np.maximum(rf - bin_hz / 2.0, lo)
    right = np.minimum(rf + bin_hz / 2.0, hi)
    return np.clip((right - left) / bin_hz, 0.0, 1.0)


def centred_offset_hz(rf_offset_hz, sample_rate_hz: float = SAMPLE_RATE_HZ) -> np.ndarray:
    """Wrap RF offsets onto the receiver's circular axis centred on the nominal pilot, ``[-f_s/2, f_s/2)``."""
    x = np.asarray(rf_offset_hz, dtype=float)
    return (x + sample_rate_hz / 2.0) % sample_rate_hz - sample_rate_hz / 2.0


def edge_distance_hz(geometry: Geometry, sample_rate_hz: float = SAMPLE_RATE_HZ) -> float:
    """Distance from the nominal pilot to the nearer coarse-channel edge, in Hz."""
    receiver = float(np.sign(geometry.sense) or 1) * (geometry.pilot_hz - geometry.centre_hz)
    return float(sample_rate_hz / 2.0 - abs(receiver))


def aliased_width_hz(lo: float, hi: float, geometry: Geometry, sample_rate_hz: float = SAMPLE_RATE_HZ) -> float:
    """How much of the RF-offset interval ``[lo, hi]`` lies beyond the coarse-channel edges (read aliased)."""
    receiver = float(np.sign(geometry.sense) or 1) * (geometry.pilot_hz - geometry.centre_hz)
    sense = float(np.sign(geometry.sense) or 1)
    a, b = sorted((receiver + sense * lo, receiver + sense * hi))     # receiver frequencies of the interval
    half = sample_rate_hz / 2.0
    return float(max(0.0, -half - a) + max(0.0, b - half))


def anchor_lobe_offset_bins(anchor_rf_offset_hz: float, lobe_rf_offset_hz: float, fine_bin_hz: float = FINE_BIN_HZ) -> float:
    """Fine anchor minus PSD in-span lobe, in fine bins (NaN when either is undefined)."""
    a, b = float(anchor_rf_offset_hz), float(lobe_rf_offset_hz)
    return (a - b) / fine_bin_hz if math.isfinite(a) and math.isfinite(b) else float("nan")


def _db(ratio) -> np.ndarray:
    r = np.asarray(ratio, dtype=float)
    out = np.full(r.shape, np.nan)
    ok = np.isfinite(r) & (r > 0)
    out[ok] = 10.0 * np.log10(r[ok])
    return out


# --------------------------------------------------------------- accumulation
@dataclass(frozen=True)
class EraSpectrum:
    """The era-mean spectrum on the full axis (ascending RF offset) and the per-frame peaks."""

    rf_offset_hz: np.ndarray        # (NFFT,) ascending
    mean: np.ndarray                # linear mean per bin; NaN where no masked frame had a finite code
    count: np.ndarray               # finite frames per bin
    frames: int                     # frames in the mask
    frames_with_spectrum: int       # masked frames with at least one finite bin
    top_frame_share: float          # largest single frame's share of the summed window power
    peak_offsets_hz: np.ndarray     # per detected frame: RF offset of its raw peak in W (centre line excluded); NaN where none
    detected_frames: int


def accumulate_spectra(product: Product, mask, detected=None, *, window_hz: float = WINDOW_HZ,
                       centre_line_hz: float = CENTRE_LINE_HALF_WIDTH_HZ, chunk: int = DEFAULT_CHUNK) -> EraSpectrum:
    """One pass over the per-frame spectra: era mean over ``mask``, raw peaks of the ``detected`` frames.

    ``detected`` defaults to the product's stored survey flag (``reject_mask``,
    the coarse decision at ``eta = 1``) and is always intersected with
    ``mask``. Never more than one decoded chunk is alive.
    """
    g = product.geometry
    n = product.n_frames
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n,):
        raise ValueError(f"mask must have shape ({n},); got {mask.shape}")
    det = mask & (product.rejected if detected is None else np.asarray(detected, dtype=bool))
    rf_fft = centred_offset_hz(g.psd_rf_offset_hz(np.arange(NFFT)))      # the circular axis, pilot at the centre
    order = np.argsort(rf_fft, kind="stable")
    in_window = np.abs(rf_fft) <= window_hz
    peak_cols = np.flatnonzero(in_window & (np.abs(rf_fft - g.centre_line_rf_offset_hz) > centre_line_hz))
    window_cols = np.flatnonzero(in_window)
    acc = np.zeros(NFFT)
    count = np.zeros(NFFT, dtype=np.int64)
    peaks = np.full(int(det.sum()), np.nan)
    filled = 0
    frames_with = 0
    top = 0.0
    total = 0.0
    for rows_slice, power in product.psd_rows(chunk):
        m = mask[rows_slice]
        if not m.any():
            continue
        rows = power[m]
        finite = np.isfinite(rows)
        count += finite.sum(axis=0)
        frames_with += int(finite.any(axis=1).sum())
        np.nan_to_num(rows, copy=False, nan=0.0)          # rows is our own copy; zeros drop out of the sum
        acc += rows.sum(axis=0)
        window_sum = rows[:, window_cols].sum(axis=1)
        total += float(window_sum.sum())
        top = max(top, float(window_sum.max()))
        d = det[rows_slice][m]
        if d.any():
            sub = rows[d][:, peak_cols]
            any_finite = finite[d][:, peak_cols].any(axis=1)
            offsets = rf_fft[peak_cols[np.argmax(sub, axis=1)]]
            offsets[~any_finite] = np.nan
            peaks[filled:filled + offsets.size] = offsets
            filled += offsets.size
    mean = np.full(NFFT, np.nan)
    has = count > 0
    mean[has] = acc[has] / count[has]
    return EraSpectrum(rf_fft[order], mean[order], count[order], int(mask.sum()), frames_with,
                       top / total if total > 0 else float("nan"), peaks, int(det.sum()))


# -------------------------------------------------------------------- analysis
@dataclass(frozen=True)
class Lobe:
    """A spectral peak of the era-mean: bin-centre and refined position, level, excess."""

    index: int                  # into the ascending-RF arrays
    offset_hz: float            # bin-centre RF offset from nominal
    refined_offset_hz: float    # three-point log-parabolic refinement
    db: float                   # dB over the window median at the peak bin
    refined_db: float           # the parabola's vertex, dB over the window median
    excess_db: float            # dB over the local baseline at the peak bin


@dataclass(frozen=True)
class WindowSpectrum:
    """The era-mean within W, its baseline and excess, on the ascending-RF axis."""

    rf_offset_hz: np.ndarray
    mean: np.ndarray            # linear
    mean_db: np.ndarray         # dB over the window median
    baseline: np.ndarray        # linear robust local background
    excess: np.ndarray          # max(mean - baseline, 0), zero at centre-line bins
    count: np.ndarray
    window_median_power: float
    window_weights: np.ndarray  # fractional overlap with [-W, W]
    centre_line: np.ndarray     # bool: |offset - centre_line| <= 60 Hz


@dataclass(frozen=True)
class PeakOffsets:
    """The per-frame raw-peak offset distribution of the detected frames."""

    frames: int                            # detected frames with a peak
    abs_percentiles_hz: tuple[float, ...]  # 50, 90, 99 of |offset|
    signed_percentiles_hz: tuple[float, ...]
    in_span_fraction: dict[int, float]     # K -> fraction of frames with |offset| <= f_s/2K


@dataclass(frozen=True)
class ContainmentRow:
    """One channel's row of ``containment.csv``; the module docstring defines every column."""

    channel: int
    freq_id: int
    frames: int
    frames_with_spectrum: int
    detected_frames: int
    top_frame_share: float
    centre_line_rf_offset_hz: float
    centre_line_db: float
    window_median_power: float
    edge_distance_hz: float
    window_aliased_hz: float
    dominant_offset_hz: float
    dominant_refined_offset_hz: float
    dominant_db: float
    dominant_excess_db: float
    near_offset_hz: float
    near_db: float
    in_span_offset_hz: float
    in_span_refined_offset_hz: float
    in_span_db: float
    in_span_excess_db: float
    in_span_recovered: bool
    out_of_span_offset_hz: float
    out_of_span_excess_db: float
    peak_abs_median_hz: float
    peak_abs_p90_hz: float
    peak_abs_p99_hz: float
    peak_signed_median_hz: float
    peak_signed_p90_hz: float
    peak_signed_p99_hz: float
    frames_in_span_64: float
    frames_in_span_128: float
    frames_in_span_256: float
    excess_total: float
    e_64: float
    e_128: float
    e_256: float
    ref_contamination_64: float
    ref_contamination_128: float
    ref_contamination_256: float
    ref_contaminant_64: str
    ref_contaminant_128: str
    ref_contaminant_256: str
    ref_aliased_64: bool
    ref_aliased_128: bool
    ref_aliased_256: bool
    straddle_loss_db_64: float
    straddle_loss_db_128: float
    straddle_loss_db_256: float
    margin_hz_64: float
    margin_hz_128: float
    margin_hz_256: float
    margin_fraction_64: float
    margin_fraction_128: float
    margin_fraction_256: float
    disposition: str
    reasons: str

    def e_k(self, k: int) -> float:
        return float(getattr(self, f"e_{k}"))


COLUMNS = tuple(f.name for f in fields(ContainmentRow))


@dataclass(frozen=True)
class ChannelPsd:
    """Everything the psd module knows about one channel under one frame mask."""

    row: ContainmentRow
    spectrum: EraSpectrum
    window: WindowSpectrum
    dominant: Lobe | None
    near: Lobe | None
    in_span: Lobe | None
    peaks: PeakOffsets


def _argmax_lobe(mean_db: np.ndarray, allowed: np.ndarray) -> int | None:
    scores = np.where(allowed & np.isfinite(mean_db), mean_db, -np.inf)
    if not np.isfinite(scores).any():
        return None
    return int(np.argmax(scores))


def _refine(rf: np.ndarray, mean: np.ndarray, i: int) -> tuple[float, float]:
    """Three-point parabola through log10 of bins i-1, i, i+1: (refined offset Hz, refined log10 value)."""
    if i <= 0 or i >= mean.size - 1:
        return float(rf[i]), float(np.log10(mean[i]))
    y0, y1, y2 = mean[i - 1], mean[i], mean[i + 1]
    if not (np.isfinite(y0) and np.isfinite(y2) and y0 > 0 and y2 > 0 and y1 > 0):
        return float(rf[i]), float(np.log10(y1))
    l0, l1, l2 = np.log10([y0, y1, y2])
    curvature = l0 - 2.0 * l1 + l2
    if curvature >= 0.0:
        return float(rf[i]), float(l1)
    p = 0.5 * (l0 - l2) / curvature
    p = float(np.clip(p, -0.5, 0.5))
    step = 0.5 * (rf[i + 1] - rf[i - 1])
    return float(rf[i] + p * step), float(l1 - 0.25 * (l0 - l2) * p)


def _lobe(window: WindowSpectrum, i: int | None) -> Lobe | None:
    if i is None:
        return None
    refined_hz, refined_log = _refine(window.rf_offset_hz, window.mean, i)
    baseline = window.baseline[i]
    excess_db = float(_db(window.mean[i] / baseline)) if np.isfinite(baseline) and baseline > 0 else float("nan")
    return Lobe(index=i, offset_hz=float(window.rf_offset_hz[i]), refined_offset_hz=refined_hz,
                db=float(window.mean_db[i]), refined_db=float(10.0 * refined_log - 10.0 * np.log10(window.window_median_power)),
                excess_db=excess_db)


def robust_baseline(rf: np.ndarray, mean: np.ndarray, excluded: np.ndarray, targets: np.ndarray, *,
                    half_width_hz: float = BASELINE_HALF_WIDTH_HZ) -> np.ndarray:
    """Sliding median of ``mean`` over ``+-half_width_hz`` about each target bin.

    ``rf`` is ascending; source bins are the finite, non-excluded bins inside
    the sliding window (drawn from the whole axis, so a target near the edge
    of W still sees a full window). Targets whose window is empty are filled
    by linear interpolation in ``log10`` between the nearest defined targets
    (nearest-value extension beyond the last defined one). Returns the
    baseline at the ``targets`` (NaN only if no target has any source).
    """
    usable = np.isfinite(mean) & (mean > 0) & ~excluded
    out = np.full(targets.size, np.nan)
    for j, t in enumerate(targets):
        lo = int(np.searchsorted(rf, rf[t] - half_width_hz, side="left"))
        hi = int(np.searchsorted(rf, rf[t] + half_width_hz, side="right"))
        src = mean[lo:hi][usable[lo:hi]]
        if src.size:
            out[j] = np.median(src)
    defined = np.isfinite(out)
    if defined.any() and not defined.all():
        out[~defined] = 10.0 ** np.interp(rf[targets][~defined], rf[targets][defined], np.log10(out[defined]))
    return out


def window_spectrum(spectrum: EraSpectrum, geometry: Geometry, *, window_hz: float = WINDOW_HZ,
                    centre_line_hz: float = CENTRE_LINE_HALF_WIDTH_HZ, baseline_half_width_hz: float = BASELINE_HALF_WIDTH_HZ,
                    baseline_exclusion_hz: float = BASELINE_EXCLUSION_HZ) -> tuple[WindowSpectrum, int | None]:
    """The window W of the era-mean with its dB scale, dominant-lobe index, baseline and excess."""
    rf, mean = spectrum.rf_offset_hz, spectrum.mean
    in_w = np.abs(rf) <= window_hz
    targets = np.flatnonzero(in_w)
    centre = np.abs(rf - geometry.centre_line_rf_offset_hz) <= centre_line_hz
    positive = in_w & np.isfinite(mean) & (mean > 0)
    median = float(np.median(mean[positive])) if positive.any() else float("nan")
    mean_db_full = _db(mean / median) if np.isfinite(median) else np.full(rf.shape, np.nan)
    dominant_full = _argmax_lobe(mean_db_full, in_w & ~centre)
    excluded = centre | (np.abs(rf) <= baseline_exclusion_hz)
    if dominant_full is not None:
        excluded |= np.abs(rf - rf[dominant_full]) <= baseline_exclusion_hz
    baseline = robust_baseline(rf, mean, excluded, targets, half_width_hz=baseline_half_width_hz)
    if np.isnan(baseline).all() and np.isfinite(median):
        baseline = np.full(targets.size, median)
    w_mean = mean[targets]
    excess = np.where(np.isfinite(w_mean) & np.isfinite(baseline), w_mean - baseline, 0.0)
    excess = np.where(centre[targets], 0.0, np.maximum(excess, 0.0))
    window = WindowSpectrum(rf_offset_hz=rf[targets], mean=w_mean, mean_db=mean_db_full[targets], baseline=baseline,
                            excess=excess, count=spectrum.count[targets], window_median_power=median,
                            window_weights=overlap_weights(rf[targets], -window_hz, window_hz), centre_line=centre[targets])
    dominant = None if dominant_full is None else int(np.searchsorted(targets, dominant_full))
    return window, dominant


def energy_fraction(window: WindowSpectrum, k: int) -> float:
    """``E_K``: excess inside ``|offset| <= f_s/2K`` over excess inside W, edge bins fractional."""
    total = float((window.excess * window.window_weights).sum())
    if total <= 0:
        return float("nan")
    h = span_half_width_hz(k)
    return float((window.excess * overlap_weights(window.rf_offset_hz, -h, h)).sum() / total)


def reference_contamination(window: WindowSpectrum, geometry: Geometry, k: int, *,
                            feature_min_db: float = FEATURE_MIN_DB) -> tuple[float, str]:
    """Excess in the two reference passbands over the in-span excess at K, and the named contaminants."""
    h = span_half_width_hz(k)
    rf = window.rf_offset_hz
    raw_excess = np.where(np.isfinite(window.mean) & np.isfinite(window.baseline), window.mean - window.baseline, 0.0)
    raw_excess = np.maximum(raw_excess, 0.0)                      # centre-line bins included: the references see them
    in_span = float((window.excess * overlap_weights(rf, -h, h)).sum())
    contamination = 0.0
    names = []
    with np.errstate(divide="ignore", invalid="ignore"):
        excess_db = _db(window.mean / window.baseline)
    for centre in reference_centres_hz(k):
        weights = overlap_weights(rf, centre - h, centre + h)
        contamination += float((raw_excess * weights).sum())
        if abs(geometry.centre_line_rf_offset_hz - centre) <= h:
            names.append("centre_line")
        inside = (weights > 0) & ~window.centre_line & np.isfinite(excess_db)
        if inside.any():
            j = int(np.argmax(np.where(inside, excess_db, -np.inf)))
            if excess_db[j] >= feature_min_db:
                names.append(f"feature@{rf[j]:.0f}Hz({excess_db[j]:.1f}dB)")
    fraction = contamination / in_span if in_span > 0 else float("nan")
    return fraction, ";".join(names)


def peak_offsets(offsets_hz: np.ndarray, spans: Sequence[int] = SPANS,
                 percentiles: Sequence[float] = (50, 90, 99)) -> PeakOffsets:
    """Percentiles of the per-frame peak offsets and the fraction inside each span."""
    x = np.asarray(offsets_hz, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        nan = tuple(float("nan") for _ in percentiles)
        return PeakOffsets(0, nan, nan, {k: float("nan") for k in spans})
    abs_p = tuple(float(v) for v in np.percentile(np.abs(x), list(percentiles)))
    signed_p = tuple(float(v) for v in np.percentile(x, list(percentiles)))
    inside = {k: float((np.abs(x) <= span_half_width_hz(k)).mean()) for k in spans}
    return PeakOffsets(int(x.size), abs_p, signed_p, inside)


def peak_counts(offsets_hz: np.ndarray, rf_offset_hz: np.ndarray) -> np.ndarray:
    """Detected frames whose raw peak fell on each window bin (the peaks are bin centres, so this is lossless)."""
    x = np.asarray(offsets_hz, dtype=float)
    x = x[np.isfinite(x)]
    rf = np.asarray(rf_offset_hz, dtype=float)
    if x.size == 0 or rf.size == 0:
        return np.zeros(rf.size, dtype=np.int64)
    edges = np.concatenate(([rf[0] - PSD_BIN_HZ / 2.0], (rf[1:] + rf[:-1]) / 2.0, [rf[-1] + PSD_BIN_HZ / 2.0]))
    counts, _ = np.histogram(x, bins=edges)
    return counts.astype(np.int64)


def disposition(in_span: Lobe | None, out_of_span_offset_hz: float, e_128: float, contamination_128: float, *,
                e_min: float = E_MIN, contamination_limit: float = CONTAMINATION_LIMIT,
                feature_min_db: float = FEATURE_MIN_DB, anchor_aliases: bool = False,
                target_k: int = TARGET_K, anchor_note: str = "") -> tuple[str, str]:
    """Design section 6: supported / supported with sentinel / unsupported, with the reasons."""
    if in_span is None or not (np.isfinite(in_span.excess_db) and in_span.excess_db >= feature_min_db):
        return UNSUPPORTED, (f"no in-span lobe at or above {feature_min_db:g} dB over baseline within "
                             f"+-{span_half_width_hz(target_k):.0f} Hz of nominal")
    reasons = []
    if np.isfinite(out_of_span_offset_hz):
        reasons.append(f"stronger out-of-span feature at {out_of_span_offset_hz:.0f} Hz")
    if not (np.isfinite(e_128) and e_128 >= e_min):
        reasons.append(f"E_128 = {e_128:.3f} < E_min = {e_min:g}")
    if not (np.isfinite(contamination_128) and contamination_128 < contamination_limit):
        reasons.append(f"K = 128 reference contamination {contamination_128:.3f} >= {contamination_limit:g}")
    if anchor_aliases:
        reasons.append(anchor_note or "fine anchor aliases an out-of-span feature")
    if reasons:
        return SUPPORTED_SENTINEL, "; ".join(reasons)
    return SUPPORTED, ""


def analyse(spectrum: EraSpectrum, geometry: Geometry, *, e_min: float = E_MIN, anchor_aliases: bool = False,
            window_hz: float = WINDOW_HZ, near_hz: float = NEAR_NOMINAL_HZ, target_k: int = TARGET_K,
            feature_min_db: float = FEATURE_MIN_DB, contamination_limit: float = CONTAMINATION_LIMIT,
            anchor_note: str = "") -> ChannelPsd:
    """Every containment quantity from an accumulated era spectrum (pure array work)."""
    window, dominant_i = window_spectrum(spectrum, geometry, window_hz=window_hz)
    rf = window.rf_offset_hz
    allowed = ~window.centre_line
    dominant = _lobe(window, dominant_i)
    near = _lobe(window, _argmax_lobe(window.mean_db, allowed & (np.abs(rf) <= near_hz)))
    h_target = span_half_width_hz(target_k)
    in_span = _lobe(window, _argmax_lobe(window.mean_db, allowed & (np.abs(rf) <= h_target)))
    recovered = in_span is not None and np.isfinite(in_span.excess_db) and in_span.excess_db >= feature_min_db
    # a stronger excess outside the campaign span (the co-channel-carrier case)
    outside = allowed & (np.abs(rf) > h_target) & (np.abs(rf) <= window_hz)
    out_offset = out_excess_db = float("nan")
    if in_span is not None and outside.any():
        j = int(np.argmax(np.where(outside, window.excess, -np.inf)))
        if window.excess[j] > window.excess[in_span.index]:
            out_offset = float(rf[j])
            out_excess_db = float(_db(window.mean[j] / window.baseline[j]))
    peaks = peak_offsets(spectrum.peak_offsets_hz)
    e_k = {k: energy_fraction(window, k) for k in SPANS}
    refs = {k: reference_contamination(window, geometry, k, feature_min_db=feature_min_db) for k in SPANS}
    lobe_offset = in_span.refined_offset_hz if recovered else float("nan")
    straddle = {k: float(straddle_loss_db(lobe_offset, k)) if recovered else float("nan") for k in SPANS}
    margin = {k: span_half_width_hz(k) - abs(lobe_offset) if recovered else float("nan") for k in SPANS}
    verdict, reasons = disposition(in_span, out_offset, e_k[128], refs[128][0], e_min=e_min,
                                   contamination_limit=contamination_limit, feature_min_db=feature_min_db,
                                   anchor_aliases=anchor_aliases, target_k=target_k, anchor_note=anchor_note)
    edge = edge_distance_hz(geometry)
    aliased = {k: any(aliased_width_hz(c - span_half_width_hz(k), c + span_half_width_hz(k), geometry) > 0
                      for c in reference_centres_hz(k)) for k in SPANS}
    centre_idx = np.flatnonzero(window.centre_line)
    centre_db = float(np.nanmax(window.mean_db[centre_idx])) if centre_idx.size and np.isfinite(window.mean_db[centre_idx]).any() else float("nan")
    nan = float("nan")
    row = ContainmentRow(
        channel=geometry.physical_channel, freq_id=geometry.freq_id, frames=spectrum.frames,
        frames_with_spectrum=spectrum.frames_with_spectrum, detected_frames=spectrum.detected_frames,
        top_frame_share=float(spectrum.top_frame_share), centre_line_rf_offset_hz=float(geometry.centre_line_rf_offset_hz),
        centre_line_db=centre_db, window_median_power=window.window_median_power,
        edge_distance_hz=edge, window_aliased_hz=aliased_width_hz(-window_hz, window_hz, geometry),
        dominant_offset_hz=dominant.offset_hz if dominant else nan,
        dominant_refined_offset_hz=dominant.refined_offset_hz if dominant else nan,
        dominant_db=dominant.db if dominant else nan, dominant_excess_db=dominant.excess_db if dominant else nan,
        near_offset_hz=near.offset_hz if near else nan, near_db=near.db if near else nan,
        in_span_offset_hz=in_span.offset_hz if in_span else nan,
        in_span_refined_offset_hz=in_span.refined_offset_hz if in_span else nan,
        in_span_db=in_span.db if in_span else nan, in_span_excess_db=in_span.excess_db if in_span else nan,
        in_span_recovered=bool(recovered), out_of_span_offset_hz=out_offset, out_of_span_excess_db=out_excess_db,
        peak_abs_median_hz=peaks.abs_percentiles_hz[0], peak_abs_p90_hz=peaks.abs_percentiles_hz[1],
        peak_abs_p99_hz=peaks.abs_percentiles_hz[2], peak_signed_median_hz=peaks.signed_percentiles_hz[0],
        peak_signed_p90_hz=peaks.signed_percentiles_hz[1], peak_signed_p99_hz=peaks.signed_percentiles_hz[2],
        frames_in_span_64=peaks.in_span_fraction[64], frames_in_span_128=peaks.in_span_fraction[128],
        frames_in_span_256=peaks.in_span_fraction[256],
        excess_total=float((window.excess * window.window_weights).sum()),
        e_64=e_k[64], e_128=e_k[128], e_256=e_k[256],
        ref_contamination_64=refs[64][0], ref_contamination_128=refs[128][0], ref_contamination_256=refs[256][0],
        ref_contaminant_64=refs[64][1], ref_contaminant_128=refs[128][1], ref_contaminant_256=refs[256][1],
        ref_aliased_64=aliased[64], ref_aliased_128=aliased[128], ref_aliased_256=aliased[256],
        straddle_loss_db_64=straddle[64], straddle_loss_db_128=straddle[128], straddle_loss_db_256=straddle[256],
        margin_hz_64=margin[64], margin_hz_128=margin[128], margin_hz_256=margin[256],
        margin_fraction_64=margin[64] / span_half_width_hz(64), margin_fraction_128=margin[128] / span_half_width_hz(128),
        margin_fraction_256=margin[256] / span_half_width_hz(256),
        disposition=verdict, reasons=reasons,
    )
    return ChannelPsd(row, spectrum, window, dominant, near, in_span, peaks)


def containment(product: Product, mask, detected=None, *, e_min: float = E_MIN, anchor_aliases: bool = False,
                chunk: int = DEFAULT_CHUNK, anchor_note: str = "") -> ChannelPsd:
    """The containment analysis of one product under a frame mask (the era frames)."""
    spectrum = accumulate_spectra(product, mask, detected, chunk=chunk)
    return analyse(spectrum, product.geometry, e_min=e_min, anchor_aliases=anchor_aliases, anchor_note=anchor_note)


# ---------------------------------------------------------------------- K*
@dataclass(frozen=True)
class KStar:
    """The eq:param:kstar rule at one ``E_min`` over a set of channel rows."""

    e_min: float
    k_star: int | None              # None when no candidate passes on the eligible channels
    failing_k: int | None           # the first K at which an eligible channel falls below e_min
    binding_channel: int | None     # the eligible channel with the smallest E at failing_k
    binding_e: float
    sentinels: tuple[int, ...]      # eligible channels below e_min at every K (do not drag the choice)
    eligible: tuple[int, ...]       # channels whose disposition is not 'unsupported' and whose E_K is finite

    def as_dict(self) -> dict:
        return {"e_min": self.e_min, "k_star": self.k_star, "failing_k": self.failing_k,
                "binding_channel": self.binding_channel, "binding_e": self.binding_e,
                "sentinels": ";".join(str(c) for c in self.sentinels), "eligible": ";".join(str(c) for c in self.eligible)}


def k_star(rows: Iterable[ContainmentRow], e_min: float = E_MIN, spans: Sequence[int] = SPANS) -> KStar:
    """``max{K : E_K(c) >= e_min for every eligible, non-sentinel channel}``, naming the binding channel."""
    rows = list(rows)
    eligible = [r for r in rows if r.disposition != UNSUPPORTED and all(math.isfinite(r.e_k(k)) for k in spans)]
    sentinels = tuple(r.channel for r in eligible if all(r.e_k(k) < e_min for k in spans))
    counted = [r for r in eligible if r.channel not in sentinels]
    if not counted:
        # nothing to count (no rows, all unsupported, or every eligible channel a sentinel): the rule is undefined
        return KStar(e_min, None, None, None, float("nan"), sentinels, tuple(r.channel for r in eligible))
    best = None
    for k in sorted(spans):
        failing = [r for r in counted if r.e_k(k) < e_min]
        if failing:
            worst = min(failing, key=lambda r: r.e_k(k))
            return KStar(e_min, best, k, worst.channel, worst.e_k(k), sentinels, tuple(r.channel for r in eligible))
        best = k
    return KStar(e_min, best, None, None, float("nan"), sentinels, tuple(r.channel for r in eligible))


def k_star_table(rows: Iterable[ContainmentRow], e_mins: Sequence[float] = E_MIN_SENSITIVITY) -> list[KStar]:
    rows = list(rows)
    return [k_star(rows, e) for e in e_mins]


# ------------------------------------------------------------------- writers
def _cell(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "" if not math.isfinite(value) else repr(value)
    if value is None:
        return ""
    return value


def write_containment_csv(rows: Sequence[ContainmentRow], path: Path | str) -> Path:
    """One row per channel, the module docstring's columns, ``repr()`` floats and blank NaNs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r.channel):
            writer.writerow({k: _cell(v) for k, v in asdict(row).items()})
    return path


KSTAR_COLUMNS = ("e_min", "k_star", "failing_k", "binding_channel", "binding_e", "sentinels", "eligible")


def write_kstar_csv(table: Sequence[KStar], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(KSTAR_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for entry in table:
            writer.writerow({k: _cell(v) for k, v in entry.as_dict().items()})
    return path


def read_containment_csv(path: Path | str) -> list[ContainmentRow]:
    """Rows back from ``containment.csv`` (blank -> NaN, bools and ints restored)."""
    types = {f.name: f.type for f in fields(ContainmentRow)}
    out = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            values = {}
            for name, text in raw.items():
                kind = types[name]
                if kind == "int":
                    values[name] = int(text)
                elif kind == "bool":
                    values[name] = text == "True"
                elif kind == "float":
                    values[name] = float(text) if text != "" else float("nan")
                else:
                    values[name] = text
            out.append(ContainmentRow(**values))
    return out


def _rounded(values, digits: int = 4) -> list:
    return [None if not math.isfinite(v) else round(float(v), digits) for v in np.asarray(values, dtype=float)]


def spectra_json_payload(channels: Sequence[ChannelPsd], *, provenance: str = "") -> dict:
    """Per channel: the window arrays and the marker offsets the plates draw; NaN as null."""
    def lobe(l: Lobe | None):
        return None if l is None else {"offset_hz": l.offset_hz, "refined_offset_hz": l.refined_offset_hz,
                                       "db": l.db, "excess_db": l.excess_db}
    out = {
        "schema": "rfisher_results.archive.psd window spectra v1",
        "provenance": provenance,
        "parameters": {"window_hz": WINDOW_HZ, "centre_line_half_width_hz": CENTRE_LINE_HALF_WIDTH_HZ,
                       "near_nominal_hz": NEAR_NOMINAL_HZ, "baseline_half_width_hz": BASELINE_HALF_WIDTH_HZ,
                       "baseline_exclusion_hz": BASELINE_EXCLUSION_HZ, "feature_min_db": FEATURE_MIN_DB,
                       "contamination_limit": CONTAMINATION_LIMIT, "e_min": E_MIN, "e_min_sensitivity": list(E_MIN_SENSITIVITY),
                       "spans": {str(k): {"half_width_hz": span_half_width_hz(k), "reference_centres_hz": list(reference_centres_hz(k))}
                                 for k in SPANS},
                       "db_reference": "median positive bin of the window", "array_rounding_digits": 4},
        "channels": {},
    }
    for c in sorted(channels, key=lambda c: c.row.channel):
        w = c.window
        baseline_db = _db(w.baseline / w.window_median_power)
        out["channels"][str(c.row.channel)] = {
            "freq_id": c.row.freq_id, "frames": c.row.frames, "detected_frames": c.row.detected_frames,
            "centre_line_rf_offset_hz": c.row.centre_line_rf_offset_hz,
            "window_median_power": c.row.window_median_power,
            "rf_offset_hz": _rounded(w.rf_offset_hz), "mean_db": _rounded(w.mean_db), "baseline_db": _rounded(baseline_db),
            "excess": _rounded(w.excess / w.window_median_power, 6), "count": [int(v) for v in w.count],
            "dominant": lobe(c.dominant), "near": lobe(c.near), "in_span": lobe(c.in_span),
            # the per-frame peaks are bin centres of the window: a count per bin is lossless
            "peak_counts": [int(v) for v in peak_counts(c.spectrum.peak_offsets_hz, w.rf_offset_hz)],
            "disposition": c.row.disposition, "reasons": c.row.reasons,
        }
    return out


def write_spectra_json(channels: Sequence[ChannelPsd], path: Path | str, *, provenance: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spectra_json_payload(channels, provenance=provenance), indent=None, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    return path


def write_spectra_npz(channels: Sequence[ChannelPsd], path: Path | str) -> Path:
    """The full 16384-bin era means (ascending RF offset) for the appendix plates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for c in channels:
        ch = c.row.channel
        arrays[f"rf_offset_hz_{ch}"] = c.spectrum.rf_offset_hz
        arrays[f"mean_{ch}"] = c.spectrum.mean
        arrays[f"count_{ch}"] = c.spectrum.count
    np.savez_compressed(path, channels=np.asarray(sorted(c.row.channel for c in channels)), **arrays)
    return path
