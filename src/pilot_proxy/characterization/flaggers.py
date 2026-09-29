"""Baseline flaggers against the detector, on each band's current era.

Chapter 9's flagger table asks what the flaggers a survey already runs would
leave behind on the same frames the pilot proxy is scored on. The flaggers
themselves are below (:func:`mad_flag`, :func:`sk_flag`) --- a
median-absolute-deviation cut within an acquisition, and a spectral-kurtosis
cut with the Nita and Gary null --- and :func:`compare` scores them on one
band's current era, for two reasons the v5 products force.

*The field names.* :func:`shelf_per_frame` reads the product through the
residual view; the v5 products carry the shelf as
``estimated_data_shelf_snr_db``, which :class:`~pilot_proxy.products.reader.Product`
exposes as ``shelf_db``.

*The era, and one floor.* The incumbent comparison was written on the whole
archive. Chapter 9 evaluates every channel on its current era, and the floor a
frame without a resolved excess is booked at is the analysis's own
(:mod:`.nulls`), not a percentile recomputed here: a flagger comparison that
invents its own floor is not comparable with the residual chain that the rest
of the chapter reports.

Scored population. The era's frames, restricted to acquisitions of at least
``min_frames`` (a block statistic needs a block), and to frames the health
gate admits. Every flagger sees exactly those frames, so the masked fractions
and the surviving shelf are comparable across rows.

Rows. Keep everything; the MAD cut; the spectral-kurtosis cut; the pilot proxy
at the survey flag (``F > mu_0``, the bootstrap rule the kernel runs); and the
pilot proxy at the point the selection reports --- on this run a declared
diagnostic, because no channel has a selected point. The last is the row
chapter 9 calls the one that matters, and it is computed from the exact
integer fine powers through the same bundle the selector uses, not from a
stored decision.

Why these two.

The detector has to be worth deploying over what CHIME already runs, so the
incumbents need to be measured rather than argued about. Two families cover
the ground:

MAD outlier rejection
    Integrated power per block, cut at ``median + k * MAD``, which is the
    style of the flagging described in CHIME's overview paper.
spectral kurtosis
    The Nita & Gary statistic over ``M`` accumulated power estimates, standard
    in single-dish and VLBI RFI excision.

Both estimate their own reference from the data. That is the property that
decides this comparison: a transmitter with a duty cycle above ~50% moves the
median and the kurtosis together with the data, so the shelf is absorbed into
the baseline and read as sky. The pilot matched filter has no baseline to
corrupt (the tone is at a known offset from the band edge whether or not it
was there a second ago), so it is indifferent to duty cycle. The measurements
below are that argument in numbers.

Blocks never cross an acquisition boundary. The trawl archive is ~8500
baseband snapshots with a median length of 3 frames, spaced about three hours
apart over 7.6 years; a block built across that gap measures the archive's
sampling pattern rather than the sky. The first version of this comparison did cross
those boundaries and returned spectral kurtosis values of 11-100 against an
expectation of 1. The consequence is a real limit on what this dataset can
settle: the longest snapshot is 33 frames, so CHIME's few-second flagging
cadence cannot be reproduced here at all. That needs the contiguous scan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from pilot_proxy.detectors.narrowband_marker.scores import build_score_bundle
from pilot_proxy.products.reader import Product, open_product

from . import surface

MAD_TO_SIGMA = 1.4826
DEFAULT_MAD_K = 1.8                       # the overview paper's MAD factor (sec 3.2.3)



# ----------------------------------------------------------------------
# The common measurement every flagger is scored against
# ----------------------------------------------------------------------

def shelf_per_frame(d) -> tuple[np.ndarray, float]:
    """(shelf power / system noise per frame, sensitivity floor in dB).

    Frames with no pilot detection have no shelf measurement. They are not
    known to be clean (they are frames where the shelf sits at or below the
    floor), so they are assigned the floor rather than zero. That choice is
    conservative in the direction that matters: it raises the residual left
    behind by *every* flagger equally, including the pilot proxy's own, so it
    can only understate the detector's advantage.
    """
    # Read through the residual view rather than the raw keys: the view checks
    # the v5 contract and derives the shelf from the exact power terms, so
    # every consumer reads the same validated values.
    view = open_product(d)
    shelf_db = np.asarray(view.shelf_db, dtype=float).reshape(-1)
    valid = np.asarray(view.valid, dtype=bool).reshape(-1)
    rejected = np.asarray(view.rejected, dtype=bool).reshape(-1)
    seen = valid & rejected & np.isfinite(shelf_db)
    clean = valid & ~rejected & np.isfinite(shelf_db)
    if clean.sum():
        floor_db = float(np.percentile(shelf_db[clean], 90.0))
    elif seen.sum():
        floor_db = float(np.nanmin(shelf_db[seen]))
    else:
        raise ValueError("no finite shelf measurements in this product")
    lin = np.full(shelf_db.shape, 10.0 ** (floor_db / 10.0))
    lin[seen] = 10.0 ** (shelf_db[seen] / 10.0)
    return lin, floor_db


def acquisition_blocks(unit: np.ndarray, min_frames: int = 2):
    """Contiguous frame runs within one acquisition, as (start, stop) pairs."""
    if len(unit) == 0:
        return []
    edges = np.flatnonzero(np.diff(unit)) + 1
    return [(int(a), int(b))
            for a, b in zip(np.r_[0, edges], np.r_[edges, len(unit)])
            if b - a >= min_frames]


# ----------------------------------------------------------------------
# The flaggers
# ----------------------------------------------------------------------

def mad_flag(power, unit, k: float = DEFAULT_MAD_K, min_frames: int = 4):
    """Per-frame flag: power above ``median + k*MAD`` of its own acquisition."""
    out = np.zeros(len(power), bool)
    for a, b in acquisition_blocks(unit, min_frames):
        p = power[a:b]
        med = float(np.median(p))
        mad = float(np.median(np.abs(p - med))) * MAD_TO_SIGMA
        if mad > 0:
            out[a:b] = p > med + k * mad
    return out


def spectral_kurtosis(p: np.ndarray, n_accum: float) -> float:
    """Nita & Gary generalised SK for one block of accumulated powers."""
    M = p.size
    if M < 2:
        return 1.0
    s1 = float(p.sum())
    if s1 <= 0:
        return 1.0
    s2 = float((p.astype(float) ** 2).sum())
    return ((M * n_accum + 1) / (M - 1)) * (M * s2 / s1 ** 2 - 1)


def sk_sigma(M: int, n_accum: float) -> float:
    """Standard deviation of SK under the Gaussian null, mean 1.

    The exact Nita & Gary (2010) null variance for the generalised estimator,
    ``2 M^2 N (N+1) / ((M-1)(MN+2)(MN+3))`` with ``N = n_accum`` raw spectra
    behind each of the ``M`` power estimates. For large ``N`` this tends to
    the same ``2/(M-1)`` as the large-accumulation shortcut that treats
    ``M*N`` as a single accumulation count, but at the small ``N`` this
    archive forces the shortcut understates the scatter (M=8, N=1: 0.43
    against the exact 0.58), which would flag at a tighter threshold than the
    nominal nsigma.
    """
    N = float(n_accum)
    return float(np.sqrt((2.0 * M * M * N * (N + 1.0))
                         / ((M - 1) * (M * N + 2.0) * (M * N + 3.0))))


def calibrate_sk_null(power, unit, min_frames: int = 8) -> float:
    """Effective accumulation length implied by the data's own scatter.

    The analytic ``n_accum`` for this product is not recoverable from it: the
    power accumulates over 2048 input streams whose correlation depends on
    whether the source is the sky or a distant transmitter, and the product
    does not record which normalization was applied. Rather than assume one,
    the null is calibrated so the *median* block sits at SK = 1. That makes SK
    a fair detector by construction: it is handed the best possible
    normalization, tuned on this very data, which the pilot proxy is not.
    """
    blocks = acquisition_blocks(unit, min_frames)
    if not blocks:
        raise ValueError("no acquisition long enough to calibrate SK")
    # SK is linear in n_accum once the +1 and +2/+3 corrections are dropped,
    # so solve for the value that puts the median raw ratio at 1.
    ratios = []
    for a, b in blocks:
        p = power[a:b].astype(float)
        M = p.size
        s1 = float(p.sum())
        if s1 <= 0:
            continue
        s2 = float((p ** 2).sum())
        ratios.append((M / (M - 1)) * (M * s2 / s1 ** 2 - 1))
    if not ratios:
        raise ValueError("no usable acquisitions for SK calibration")
    med = float(np.median(ratios))
    if med <= 0:
        raise ValueError("degenerate SK null: non-positive median ratio")
    return 1.0 / med


def sk_flag(power, unit, n_accum: float | None = None, nsigma: float = 3.0,
            min_frames: int = 8):
    """Per-frame flag: acquisitions whose SK departs from the null.

    Returns ``(flag, sk_values, n_accum)``. ``n_accum=None`` calibrates the
    null on this channel; see :func:`calibrate_sk_null`.
    """
    if n_accum is None:
        n_accum = calibrate_sk_null(power, unit, min_frames)
    out = np.zeros(len(power), bool)
    sks = []
    for a, b in acquisition_blocks(unit, min_frames):
        p = power[a:b].astype(float)
        sk = spectral_kurtosis(p, n_accum)
        sks.append(sk)
        if abs(sk - 1.0) > nsigma * sk_sigma(p.size, n_accum):
            out[a:b] = True
    return out, np.array(sks), float(n_accum)


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

@dataclass
class FlaggerResult:
    name: str
    f: float
    r: float
    n_kept: int
    shelf_kept_db: float
    reduction_db: float


def score_flagger(name, flag, lin, base) -> FlaggerResult:
    """Masked fraction and surviving shelf for one flagger, over ``base``."""
    keep = base & ~flag
    f = 1.0 - keep.sum() / base.sum()
    r = float(lin[keep].mean()) if keep.sum() else float("inf")
    r_all = float(lin[base].mean())
    db = 10 * np.log10(r) if np.isfinite(r) and r > 0 else float("-inf")
    return FlaggerResult(name=name, f=float(f), r=r, n_kept=int(keep.sum()),
                         shelf_kept_db=float(db),
                         reduction_db=float(10 * np.log10(r_all / r))
                         if np.isfinite(r) and r > 0 else float("inf"))


def duty_cycle(d) -> float:
    """Fraction of valid frames carrying a positive pilot excess.

    Above 0.5 a self-calibrating flagger's own reference is contaminated, so
    the shelf is measured as sky. This single number predicts most of what the
    incumbent comparison finds.
    """
    valid = d["valid"][:, 0].astype(bool)
    return float(d["reject_mask"][valid, 0].astype(bool).mean())



MIN_BLOCK_FRAMES = 8              # a block statistic needs a block; the incumbents' own minimum
SK_NSIGMA = 3.0
SURVEY_FLAG = "pilot proxy, survey flag"
DIAGNOSTIC = "pilot proxy, reported point"
KEEP_EVERYTHING = "keep everything"


@dataclass(frozen=True)
class FlaggerRow:
    """One flagger on one channel's scored population."""

    channel: int
    name: str
    masked_fraction: float
    kept: int
    retained_shelf_linear: float      # mean shelf-or-floor over the kept frames
    retained_shelf_db: float
    suppression_db: float             # keep-everything mean over this row's mean, in dB
    status: str = "measured"          # 'measured' | 'undefined' (nothing kept)

    def as_row(self) -> dict:
        return {"channel": self.channel, "flagger": self.name, "masked_fraction": self.masked_fraction,
                "kept": self.kept, "retained_shelf_linear": self.retained_shelf_linear,
                "retained_shelf_db": self.retained_shelf_db, "suppression_db": self.suppression_db,
                "status": self.status}


@dataclass(frozen=True)
class ChannelFlaggers:
    channel: int
    freq_id: int
    era_label: str
    scored_frames: int
    era_frames: int
    blocks: int
    block_median_frames: float
    floor_db: float
    floor_evidence: str
    duty_cycle: float                 # fraction of the scored frames the survey flag rejects
    rows: tuple[FlaggerRow, ...]
    diagnostic_basis: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def by_name(self, name: str) -> FlaggerRow | None:
        return next((r for r in self.rows if r.name == name), None)


def _score(name: str, channel: int, flag: np.ndarray, linear: np.ndarray, base: np.ndarray,
           all_mean: float) -> FlaggerRow:
    keep = base & ~flag
    n = int(base.sum())
    kept = int(keep.sum())
    if not kept:
        return FlaggerRow(channel, name, 1.0, 0, math.nan, math.nan, math.nan, "undefined")
    mean = float(linear[keep].mean())
    return FlaggerRow(channel, name, 1.0 - kept / n, kept, mean, 10.0 * math.log10(mean) if mean > 0 else -math.inf,
                      10.0 * math.log10(all_mean / mean) if mean > 0 and all_mean > 0 else math.inf)


def compare(product: Product, era: np.ndarray, *, floor_db: float, floor_evidence: str, era_label: str,
            diagnostic: dict | None = None, anchor_bin: int | None = None, bulk_mask: np.ndarray | None = None,
            mad_k: float = DEFAULT_MAD_K, sk_nsigma: float = SK_NSIGMA,
            min_frames: int = MIN_BLOCK_FRAMES) -> ChannelFlaggers:
    """Every flagger on one channel's era, scored on the frames all of them can see."""
    era = np.asarray(era, dtype=bool) & product.selected
    unit = product.frame_unit_index
    notes: list[str] = []
    inblock = np.zeros(era.shape, dtype=bool)
    for a, b in acquisition_blocks(np.where(era, unit, -1), min_frames):
        inblock[a:b] = True
    base = era & inblock
    sizes = np.array([b - a for a, b in acquisition_blocks(np.where(era, unit, -1), min_frames)], dtype=float)
    if not base.any():
        notes.append(f"no acquisition of the era reaches {min_frames} frames: nothing is scored")
        return ChannelFlaggers(product.geometry.physical_channel, product.geometry.freq_id, era_label,
                               0, int(era.sum()), 0, 0.0, floor_db, floor_evidence, math.nan, (), "", tuple(notes))

    floor_linear = 10.0 ** (floor_db / 10.0) if math.isfinite(floor_db) else math.nan
    shelf = product.shelf_db
    finite = np.isfinite(shelf)
    if not math.isfinite(floor_linear) and not finite[base].all():
        notes.append("floor is refused and some scored frames carry no shelf estimate: those frames are dropped")
        base = base & finite
    linear = np.full(shelf.shape, floor_linear, dtype=float)
    linear[finite] = np.maximum(10.0 ** (shelf[finite] / 10.0), floor_linear if math.isfinite(floor_linear) else -math.inf)
    all_mean = float(linear[base].mean())
    power = np.asarray(product.archive["baseband_power_linear"])[:, 0].astype(float)
    channel = product.geometry.physical_channel

    sk, _, _ = sk_flag(power, unit, nsigma=sk_nsigma, min_frames=min_frames)
    rows = [_score(KEEP_EVERYTHING, channel, np.zeros(era.shape, dtype=bool), linear, base, all_mean),
            _score(f"MAD {mad_k:g}x within acquisition", channel, mad_flag(power, unit, k=mad_k), linear, base, all_mean),
            _score(f"SK {sk_nsigma:g}sigma within acquisition", channel, sk, linear, base, all_mean),
            _score(SURVEY_FLAG, channel, product.rejected, linear, base, all_mean)]

    basis = ""
    if diagnostic and anchor_bin is not None and bulk_mask is not None:
        rho, eta_q16 = diagnostic.get("rho"), diagnostic.get("eta_q16")
        if rho is not None and eta_q16 is not None:
            try:
                bundle = build_score_bundle(
                    product.path, base, anchor_bin=int(anchor_bin),
                    designated_half_width=surface.DESIGNATED_HALF_WIDTH, bulk_mask=np.asarray(bulk_mask, dtype=bool))
                required = np.asarray(bundle.requirements_by_rho()[int(rho)], dtype=object)
                flag = np.ones(era.shape, dtype=bool)
                flag[bundle.source_row_index] = ~surface.kept_at(required, int(eta_q16))
                rows.append(_score(DIAGNOSTIC, channel, flag, linear, base, all_mean))
                basis = diagnostic.get("basis", "")
            except Exception as exc:            # the bundle refuses loudly on some blocks; record, do not stop
                notes.append(f"reported point not scored: {type(exc).__name__}: {exc}")
        else:
            notes.append("reported point not scored: the selection carries no rank or multiplier")
    else:
        notes.append("reported point not scored: no selection to read it from")

    return ChannelFlaggers(
        channel=channel, freq_id=product.geometry.freq_id, era_label=era_label, scored_frames=int(base.sum()),
        era_frames=int(era.sum()), blocks=int(sizes.size), block_median_frames=float(np.median(sizes)) if sizes.size else 0.0,
        floor_db=floor_db, floor_evidence=floor_evidence,
        duty_cycle=float(product.rejected[base].mean()), rows=tuple(rows), diagnostic_basis=basis, notes=tuple(notes))


def channel_row(result: ChannelFlaggers) -> dict:
    """One flat row per channel: the summary the ledger carries."""
    row = {"channel": result.channel, "freq_id": result.freq_id, "era": result.era_label,
           "scored_frames": result.scored_frames, "era_frames": result.era_frames, "blocks": result.blocks,
           "block_median_frames": result.block_median_frames, "floor_db": result.floor_db,
           "floor_evidence": result.floor_evidence, "duty_cycle": result.duty_cycle,
           "diagnostic_basis": result.diagnostic_basis, "notes": "; ".join(result.notes)}
    for r in result.rows:
        tag = {KEEP_EVERYTHING: "keep", SURVEY_FLAG: "flag", DIAGNOSTIC: "point"}.get(r.name)
        if tag is None:
            tag = "mad" if r.name.startswith("MAD") else "sk"
        row[f"{tag}_masked_fraction"] = r.masked_fraction
        row[f"{tag}_kept"] = r.kept
        row[f"{tag}_retained_shelf_db"] = r.retained_shelf_db
        row[f"{tag}_suppression_db"] = r.suppression_db
        row[f"{tag}_status"] = r.status
    return row


FLAGGER_COLUMNS = ("channel", "flagger", "masked_fraction", "kept", "retained_shelf_linear", "retained_shelf_db",
                   "suppression_db", "status")


def write_flagger_rows(results: Sequence[ChannelFlaggers], path: Path | str) -> Path:
    """One row per (channel, flagger): the long form the chapter's summary is taken over."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FLAGGER_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for result in results:
            for r in result.rows:
                writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.as_row().items()})
    return path


__all__ = ["DEFAULT_MAD_K", "DIAGNOSTIC", "KEEP_EVERYTHING", "MAD_TO_SIGMA", "MIN_BLOCK_FRAMES", "SK_NSIGMA",
           "SURVEY_FLAG", "ChannelFlaggers", "FlaggerResult", "FlaggerRow", "FLAGGER_COLUMNS",
           "acquisition_blocks", "calibrate_sk_null", "channel_row", "compare", "duty_cycle", "mad_flag",
           "score_flagger", "shelf_per_frame", "sk_flag", "sk_sigma", "spectral_kurtosis", "write_flagger_rows"]
