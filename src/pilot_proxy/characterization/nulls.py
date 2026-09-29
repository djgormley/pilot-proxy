"""Null calibration of one band's current era: centres, widths, tails,
exchangeability and the floor (chapter 8 tab:calibration:nulls; chapter 5
sec:estimator:statistics); and the band's dated transmitter-off population,
with the two tests that make it a verified signal-free null.

Two statistics, two nulls.

- Coarse: ``Q = F / mu_0`` per frame. Under the i.i.d. model
  ``F ~ F(524288, 1048576)``, mean 1.000002, standard deviation 0.239%. The
  null population is, in order of strength (ch08 §253-263): a verified
  transmitter-off era; an independently identified quiet subset; a
  reference-bin surrogate; a model-only null; or unavailable. Without an off
  era this module reads the null off the bulk of the era's mixture and says
  so: centre = median of the bulk, scale = left-side deviation about that
  median, the mixture assumption declared beside the number. The ideal law's
  degrees of freedom ``(2P, 4P)`` come from the project's detector
  configuration.
- Fine: ``T[f]`` on the bulk bins ``B`` (independent, non-designated, guard
  and census excluded). Every frame's bulk bins are null by construction, so
  the fine null is read from all era frames. Under the i.i.d. model
  ``T ~ F(4096, 8192)``, mean 1.00024, standard deviation 2.71%.

Width factors are the measured width over the i.i.d. width, in two forms: raw
(standard deviation) and robust core (left-side scale about the median from
three quantile probes, the median of the three). Tail fraction: values beyond
three core widths above the centre. The i.i.d. widths are evaluated from the
F distributions, not typed in.

Probe convention. The register value ``floor.null_scale_probes`` is
``((32.0, 1.0), (5.0, 1.96), (0.3, 2.9677))`` and
``residual_chain.null_scale`` applies the percentiles to the *kept*
frames about ``mu_0`` (the same convention as the residual chain's null
scale): the kept sample is the lower half of a symmetric null,
so its 32nd percentile is the null's 16th, one sigma below the centre
(:func:`kept_half_null`; simulated recovery of an ideal null's width 0.99 to
1.00). The bulk description of a block (:func:`describe_null`, centre =
median, left-side scale about the median) needs the full-null percentiles
the same deviates sit at (``15.87 / 2.5 / 0.15``, ``CORE_PROBES``); applying
the register's percentiles to a full null would return 0.84 of the width.

Which stated floor. The kept half is a null half only when the block's bulk
sits at ``mu_0``: on channel 33's calibration block (2023-12..2025-11) the
bulk median is 1.0715 (+0.30 dB, 30 i.i.d. widths above ``mu_0``), 97.3% of
frames are rejected and the 2.7% "kept" are the Gaussian lower tail of the
carrier-on distribution (the tail predicts 2.70%); their scale (width factor
8.6) is the carrier's, not the receiver's, and it would print a floor 2.6 dB
more optimistic than the bulk's own left-side scale. So ``floor_basis`` is
the kept half about ``mu_0`` (the register's convention) when the bulk
centre is within ``OFF_CENTRE_TOLERANCE`` of ``mu_0`` and the three probes
agree (``kept.spread <= KEPT_SPREAD_LIMIT``; channel 14's probes disagree by
8x because its kept sample mixes a narrow null with the bulk's tail), and
otherwise the bulk's left-side scale about its median (the chapter 8
mixture-read convention), labelled ``stated (bulk, not H0)``, provided the bulk centre lies within
``BULK_CENTRE_LIMIT`` of ``mu_0`` (a bulk at 1.3 or 150 is the carrier, not
a mixture with a null, and the floor is refused: channels 15, 17, 22, 24,
28, 30, 31, 36). Both values are always reported.

Off population. A band's dated transmitter-off interval
(``projects/<p>/eras/transmitter_off.json``) gives a *transmitter-off
population* (:func:`transmitter_off`). It is a *verified signal-free*
population only when it passes both named tests: ``independent_verification``
(the off state is established by an external record, the interval's
``independently_verified`` flag) and ``null_likeness``
(:func:`off_population_check`, two-sided: centre at ``mu_0``, the left-side
core width within the declared limit, and the upper side through the
central-68 width ratio ``r68`` within the same limit; see
:func:`off_null_like`). Only a verified signal-free population calibrates the
off-null or its measured percentile floor (:attr:`OffPopulation.signal_free`);
otherwise the fallback remains explicitly mixture-conditioned and the reason
is recorded (:attr:`OffPopulation.rejection_reason`). Archive-inferred dates
are not independent evidence. Even an empirical floor percentile is an
assigned screening allowance, not a physical lower bound or a
coverage-calibrated upper limit.

Exchangeability (ch08 §275-283) is the detector adapter's
(:mod:`pilot_proxy.detectors.narrowband_marker.exchangeability`): the marker
bins against the rank ``rho`` of the bulk on quiet frames.

Floor (ch08 §206-239): where the channel has a verified off era, the 90th
percentile of the finite shelf estimates over that era's frames, ``measured``
when at least 30 frames support it (register ``floor.minimum_null_frames``);
otherwise the sigma-implied substitute ``10 log10(sigma_core) + offset``,
labelled ``stated``. Never the minimum detected shelf.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats

from pilot_proxy.config.project import default_project
from pilot_proxy.detectors.narrowband_marker.exchangeability import Exchangeability, exchangeability_rate
from pilot_proxy.products.reader import Product, fine_power_ratio

from . import blocks, eras

_PROJECT = default_project()
_REGISTER = _PROJECT.register
COARSE_DOF = _PROJECT.detector_config.coarse_null_dof(_PROJECT.instrument)
FINE_DOF = (4096, 8192)
# one-sided lower percentiles and the Gaussian deviates they sit at
CORE_PROBES = ((15.87, 1.0), (2.5, 1.96), (0.15, 2.9677))
# the register's pairs as residual_chain.null_scale applies them (one-sided at p)
AS_CODED_PROBES = _REGISTER.value("floor.null_scale_probes")
TAIL_WIDTHS = 3.0
FLOOR_PERCENTILE = float(_REGISTER.value("floor.upper_percentile"))
FLOOR_MIN_FRAMES = int(_REGISTER.value("floor.minimum_null_frames"))
MIN_NULL_FRAMES = int(_REGISTER.value("floor.minimum_null_frames"))
# an off population is null-like when its coarse centre is within this of mu_0 (2% = 0.086 dB, about eight
# i.i.d. widths) and its robust-core width factor is at most this; provisional policy values, recorded per row
OFF_CENTRE_TOLERANCE = 0.02
OFF_WIDTH_LIMIT = 5.0
# the upper-tail clause of an off population's null-likeness (author ruling T13, 2026-09-28): the central-68
# half-width over the ideal law's, r68 (the predeclared null's width measure, output/ch37-control-491-2026-09-24/
# null/PLAN.md test W), must not exceed the same OFF_WIDTH_LIMIT. The core width probes the left side only, so a
# population with a clean lower side and a heavy upper side, where eta_Pfa lives, passed; r68 reads both sides.
CENTRAL_68 = (float(stats.norm.cdf(-1.0)), float(stats.norm.cdf(1.0)))
KEPT_SPREAD_LIMIT = 3.0      # the kept-half probes must agree to this factor for the kept half to state the floor
BULK_CENTRE_LIMIT = 0.1      # beyond this the block's bulk is the carrier, not a mixture with a null: no stated floor


def iid_width(dof: tuple[int, int]) -> tuple[float, float]:
    """(mean, standard deviation) of the F distribution with these degrees of freedom."""
    mean, var = stats.f.stats(dof[0], dof[1], moments="mv")
    return float(mean), float(np.sqrt(var))


def core_scale(values, centre: float, probes=CORE_PROBES) -> tuple[float, float]:
    """Left-side robust scale about ``centre``: median over probes of (centre - q_p) / z; and the probe spread."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < MIN_NULL_FRAMES:
        return math.nan, math.nan
    ests = [(centre - float(np.percentile(x, p))) / z for p, z in probes]
    ests = [e for e in ests if e > 0.0]
    if not ests:
        return math.nan, math.nan
    return float(np.median(ests)), float(max(ests) / min(ests))


@dataclass(frozen=True)
class NullWidths:
    """One statistic's null description."""

    frames: int                    # frames (coarse) or frame x bulk-bin samples (fine)
    centre: float                  # median
    centre_db: float               # 10 log10(centre)
    iid_mean: float
    iid_sigma: float
    raw_sigma: float
    raw_width_factor: float        # raw_sigma / iid_sigma
    core_sigma: float
    core_width_factor: float
    core_spread: float             # probe spread under the corrected probes
    as_coded_sigma: float          # residual_chain.null_scale convention, for comparison
    as_coded_spread: float
    tail_fraction: float           # fraction beyond centre + 3 core widths
    tail_fraction_iid: float       # the same for the i.i.d. model

    def as_dict(self, prefix: str) -> dict:
        return {f"{prefix}_{k}": v for k, v in self.__dict__.items()}


def describe_null(values, dof: tuple[int, int]) -> NullWidths:
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    mean, sigma = iid_width(dof)
    if x.size < MIN_NULL_FRAMES:
        nan = math.nan
        return NullWidths(int(x.size), nan, nan, mean, sigma, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    centre = float(np.median(x))
    raw = float(np.std(x))
    core, spread = core_scale(x, centre)
    coded, coded_spread = core_scale(x, centre, probes=AS_CODED_PROBES)
    tail = float(np.mean(x > centre + TAIL_WIDTHS * core)) if math.isfinite(core) else math.nan
    median_iid = float(stats.f.ppf(0.5, *dof))
    tail_iid = float(stats.f.sf(median_iid + TAIL_WIDTHS * sigma, *dof))
    return NullWidths(int(x.size), centre, 10.0 * math.log10(centre), mean, sigma, raw, raw / sigma, core, core / sigma,
                      spread, coded, coded_spread, tail, tail_iid)


@dataclass(frozen=True)
class KeptHalfNull:
    """The register's kept-frame null scale: frames with ``Q <= 1``, left-side scale about ``mu_0``."""

    frames: int
    core_sigma: float
    width_factor: float          # core_sigma / iid_sigma
    spread: float                # largest over smallest probe estimate

    def as_dict(self, prefix: str = "kept") -> dict:
        return {f"{prefix}_{k}": v for k, v in self.__dict__.items()}


def kept_half_null(q, dof: tuple[int, int] = COARSE_DOF, probes=AS_CODED_PROBES) -> KeptHalfNull:
    """Scale of the null from the kept half about ``mu_0`` (``Q = F / mu_0 <= 1``), the register's convention."""
    x = np.asarray(q, dtype=float).ravel()
    x = x[np.isfinite(x) & (x <= 1.0)]
    _, sigma = iid_width(dof)
    if x.size < MIN_NULL_FRAMES:
        return KeptHalfNull(int(x.size), math.nan, math.nan, math.nan)
    ests = [(1.0 - float(np.percentile(x, p))) / z for p, z in probes]
    ests = [e for e in ests if e > 0.0]
    if not ests:
        return KeptHalfNull(int(x.size), math.nan, math.nan, math.nan)
    core = float(np.median(ests))
    return KeptHalfNull(int(x.size), core, core / sigma, float(max(ests) / min(ests)))


def null_like(coarse: NullWidths, *, centre_tolerance: float = OFF_CENTRE_TOLERANCE,
              width_limit: float = OFF_WIDTH_LIMIT) -> tuple[bool, str]:
    """Whether a coarse population reads as a null: centre near mu_0 and a modest core width."""
    if not (math.isfinite(coarse.centre) and math.isfinite(coarse.core_width_factor)):
        return False, "too few frames to describe"
    reasons = []
    if abs(coarse.centre - 1.0) > centre_tolerance:
        reasons.append(f"centre {coarse.centre:.4f} is more than {centre_tolerance:g} from mu_0")
    if coarse.core_width_factor > width_limit:
        reasons.append(f"core width factor {coarse.core_width_factor:.1f} exceeds {width_limit:g}")
    if reasons:
        return False, "; ".join(reasons)
    return True, f"centre {coarse.centre:.4f}, core width factor {coarse.core_width_factor:.2f}"


def central68_ratio(values, dof: tuple[int, int] = COARSE_DOF) -> float:
    """r68: the central-68 half-width ``(q(0.8413) - q(0.1587)) / 2`` over the ideal law's (NaN below the minimum)."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < MIN_NULL_FRAMES:
        return math.nan
    low, high = (float(v) for v in np.quantile(x, CENTRAL_68))
    law_low, law_high = (float(v) for v in stats.f.ppf(CENTRAL_68, *dof))
    return (high - low) / (law_high - law_low)


def off_null_like(coarse: NullWidths, r68: float, *, centre_tolerance: float = OFF_CENTRE_TOLERANCE,
                  width_limit: float = OFF_WIDTH_LIMIT) -> tuple[bool, str]:
    """Whether an off population reads as a null on both sides: :func:`null_like`, and ``r68 <= width_limit``."""
    ok, reason = null_like(coarse, centre_tolerance=centre_tolerance, width_limit=width_limit)
    if not (math.isfinite(coarse.centre) and math.isfinite(coarse.core_width_factor)):
        return False, reason
    upper = math.isfinite(r68) and r68 <= width_limit
    if ok and upper:
        return True, f"{reason}, r68 {r68:.2f}"
    reasons = [] if ok else [reason]
    if not upper:
        reasons.append(f"central-68 width r68 {r68:.1f} exceeds {width_limit:g} (upper tail not null-like)")
    return False, "; ".join(reasons)


def _off_check(product: Product, off_mask) -> tuple[bool, NullWidths | None, str, float]:
    if off_mask is None:
        return False, None, "no recorded off epoch", math.nan
    frames = np.asarray(off_mask, dtype=bool) & product.selected
    if frames.sum() < MIN_NULL_FRAMES:
        return False, None, f"off population has {int(frames.sum())} frames < {MIN_NULL_FRAMES}", math.nan
    values = product.statistic[frames]
    widths = describe_null(values, COARSE_DOF)
    r68 = central68_ratio(values, COARSE_DOF)
    ok, reason = off_null_like(widths, r68)
    return ok, widths, reason, r68


def off_population_check(product: Product, off_mask) -> tuple[bool, NullWidths | None, str]:
    """Describe a recorded off population and say whether it is null-like (None when there is no population).

    Two-sided (:func:`off_null_like`): the centre and the left-side core width
    as the release tested them, and the central-68 width ratio ``r68``, which
    reads the upper side, within the same ``OFF_WIDTH_LIMIT``.
    """
    ok, widths, reason, _ = _off_check(product, off_mask)
    return ok, widths, reason


@dataclass(frozen=True)
class FloorEstimate:
    db: float
    evidence: str                  # 'measured' | 'stated' | 'refused'
    population: str
    frames: int
    percentile: float = FLOOR_PERCENTILE
    stated_kept_half_db: float = math.nan   # the sigma-implied value from the kept half about mu_0 (the register's convention)
    stated_bulk_db: float = math.nan        # the same from the bulk's left-side core width about its median
    basis: str = ""                         # 'off era p90' | 'kept half about mu_0' | 'bulk left side (not H0)' | 'none'

    def as_dict(self) -> dict:
        return {"floor_db": self.db, "floor_evidence": self.evidence, "floor_population": self.population,
                "floor_frames": self.frames, "floor_percentile": self.percentile, "floor_basis": self.basis,
                "floor_stated_kept_half_db": self.stated_kept_half_db, "floor_stated_bulk_db": self.stated_bulk_db}


def _sigma_implied_db(sigma: float, offset: float) -> float:
    return 10.0 * math.log10(sigma) + offset if math.isfinite(sigma) and sigma > 0 else math.nan


def floor_estimate(product: Product, off_era: np.ndarray | None, coarse: NullWidths,
                   kept: KeptHalfNull | None = None) -> FloorEstimate:
    """Off-era 90th-percentile shelf where a verified off era exists, else the sigma-implied substitute.

    The substitute is ``10 log10(sigma) + offset``: ``sigma`` from the kept
    half about ``mu_0`` when the bulk centre lies within ``OFF_CENTRE_TOLERANCE``
    of ``mu_0`` and the probes agree (``spread <= KEPT_SPREAD_LIMIT``), else
    from the bulk's left-side core width about its median (the mixture read).
    """
    offset = float(product.view.shelf_offset_db)
    kept_db = _sigma_implied_db(kept.core_sigma, offset) if kept is not None else math.nan
    bulk_db = _sigma_implied_db(coarse.core_sigma, offset)
    bulk_at_mu0 = math.isfinite(coarse.centre) and abs(coarse.centre - 1.0) <= OFF_CENTRE_TOLERANCE
    probes_agree = kept is not None and math.isfinite(kept.spread) and kept.spread <= KEPT_SPREAD_LIMIT
    bulk_is_mixture = math.isfinite(coarse.centre) and abs(coarse.centre - 1.0) <= BULK_CENTRE_LIMIT
    if math.isfinite(kept_db) and bulk_at_mu0 and probes_agree:
        stated, basis, population = kept_db, "kept half about mu_0", f"kept half about mu_0, {kept.frames} frames"
    elif math.isfinite(bulk_db) and not bulk_is_mixture:
        stated, basis = math.nan, "none"
        population = f"bulk centre {coarse.centre:.3g} is beyond {BULK_CENTRE_LIMIT:g} of mu_0: the block carries no null population"
    elif math.isfinite(bulk_db):
        why = []
        if not bulk_at_mu0:
            why.append(f"bulk centre {coarse.centre:.4f} is not at mu_0")
        if kept is not None and not probes_agree:
            why.append(f"kept-half probes disagree (spread {kept.spread:.1f})" if math.isfinite(kept.spread) else "too few kept frames")
        stated, basis = bulk_db, "bulk left side (not H0)"
        population = "bulk left-side scale about its median (" + "; ".join(why) + ")"
    else:
        stated, basis, population = math.nan, "none", "no measurable null width"
    if off_era is not None and np.asarray(off_era, dtype=bool).any() and off_population_check(product, off_era)[0]:
        mask = np.asarray(off_era, dtype=bool) & product.selected
        shelf = product.shelf_db[mask]
        shelf = shelf[np.isfinite(shelf)]
        if shelf.size >= FLOOR_MIN_FRAMES:
            return FloorEstimate(float(np.percentile(shelf, FLOOR_PERCENTILE)), "measured",
                                 f"verified off era: {shelf.size} frames with a shelf estimate of {int(mask.sum())}",
                                 int(shelf.size), stated_kept_half_db=kept_db, stated_bulk_db=bulk_db, basis="off era p90")
        return FloorEstimate(stated, "stated", f"off era has only {shelf.size} frames with a shelf estimate; substitute: {population}",
                             int(shelf.size), stated_kept_half_db=kept_db, stated_bulk_db=bulk_db, basis=basis)
    if math.isfinite(stated):
        return FloorEstimate(stated, "stated", f"no verified off era; sigma-implied substitute: {population}",
                             kept.frames if (kept is not None and basis.startswith("kept")) else int(coarse.frames),
                             stated_kept_half_db=kept_db, stated_bulk_db=bulk_db, basis=basis)
    return FloorEstimate(math.nan, "refused", f"no off era; {population}", 0, stated_kept_half_db=kept_db, stated_bulk_db=bulk_db,
                         basis="none")


@dataclass(frozen=True)
class NullCalibration:
    channel: int
    freq_id: int
    era_label: str
    era_frames: int
    null_source: str                     # 'verified transmitter-off era' | 'bulk of the mixture (declared)' | 'recorded off epoch, not null-like ...'
    mixture_declared: bool
    coarse: NullWidths
    fine: NullWidths
    fine_designated_median: float        # median T over the designated window, all era frames (for the plates)
    floor: FloorEstimate
    exchangeability: Exchangeability | None
    bulk_size: int
    quiet_frames: int
    detected_frames: int
    kept: KeptHalfNull | None = None     # the register's kept-half null scale on the block
    off_null_like: bool | None = None    # None: no recorded off population
    off_check: str = ""
    off_coarse: NullWidths | None = None # the off population's own description (when one exists)
    fine_bulk_size: int = 0              # bulk bins the fine null was read on (the nominal window may be excluded)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_row(self) -> dict:
        row = {"channel": self.channel, "freq_id": self.freq_id, "era": self.era_label, "era_frames": self.era_frames,
               "null_source": self.null_source, "mixture_declared": self.mixture_declared, "bulk_size": self.bulk_size,
               "quiet_frames": self.quiet_frames, "detected_frames": self.detected_frames,
               "fine_designated_median": self.fine_designated_median,
               "off_null_like": self.off_null_like, "off_check": self.off_check,
               "off_centre": self.off_coarse.centre if self.off_coarse is not None else math.nan,
               "off_core_width_factor": self.off_coarse.core_width_factor if self.off_coarse is not None else math.nan,
               "off_frames": self.off_coarse.frames if self.off_coarse is not None else 0,
               "off_centre_tolerance": OFF_CENTRE_TOLERANCE, "off_width_limit": OFF_WIDTH_LIMIT,
               "kept_spread_limit": KEPT_SPREAD_LIMIT, "fine_bulk_size": self.fine_bulk_size}
        row.update((self.kept or KeptHalfNull(0, math.nan, math.nan, math.nan)).as_dict("kept"))
        row.update(self.coarse.as_dict("coarse"))
        row.update(self.fine.as_dict("fine"))
        row.update(self.floor.as_dict())
        if self.exchangeability is not None:
            row.update(self.exchangeability.as_dict())
        row["notes"] = "; ".join(self.notes)
        return row


def calibrate_null(product: Product, era: np.ndarray, *, anchor_bin: int, bulk_mask: np.ndarray, era_label: str,
                   off_era: np.ndarray | None = None, rho: int | None = None, quiet_block: np.ndarray | None = None,
                   fine_t: np.ndarray | None = None, exclude_fine_bins: Sequence[int] | None = None) -> NullCalibration:
    """Null calibration on the era's frames.

    ``off_era`` marks frames of a verified transmitter-off era (the coarse null
    population and the floor come from it); without it the coarse null is
    read from the bulk of the era's mixture. ``rho`` and ``quiet_block`` (the
    evaluation block's frames, restricted here to coarse-quiet ones) drive
    the exchangeability check. ``fine_t`` may be supplied as the (N, 256)
    fine statistic of every frame to avoid reading the terms twice.
    """
    era = np.asarray(era, dtype=bool) & product.selected
    notes = []
    q = product.statistic
    kept = kept_half_null(q[era])
    off_ok, off_widths, off_reason = off_population_check(product, off_era)
    off_null = None if off_widths is None else off_ok
    if off_widths is not None and off_ok:
        null_frames = np.asarray(off_era, dtype=bool) & product.selected
        source, mixture = "verified transmitter-off era", False
    else:
        if off_widths is not None:
            notes.append(f"off candidate rejected as null ({off_reason}); its positive shelf is not a sensitivity calibration")
        null_frames = era
        source, mixture = "bulk of the mixture (declared)", True
        notes.append("coarse null read from the bulk of the block's mixture: centre = median, scale = left side")
    coarse = describe_null(q[null_frames], COARSE_DOF)
    if fine_t is None:
        fine_t = fine_power_ratio(product.fine_terms_all())
    bulk = np.asarray(bulk_mask, dtype=bool)
    fine_bulk = bulk.copy()
    if exclude_fine_bins is not None:
        excluded = [int(b) % fine_t.shape[1] for b in exclude_fine_bins]
        fine_bulk[excluded] = False
        notes.append(f"fine null read on {int(fine_bulk.sum())} of {int(bulk.sum())} bulk bins: the nominal window is excluded "
                     f"because the anchor is suspect (the bulk may carry the pilot)")
    fine = describe_null(fine_t[era][:, fine_bulk], FINE_DOF)
    designated = [(int(anchor_bin) + k) % fine_t.shape[1] for k in range(-2, 3)]
    fine_designated_median = float(np.median(fine_t[era][:, designated])) if era.any() else math.nan
    floor = floor_estimate(product, off_era, coarse, kept)
    exch = None
    quiet = era & ~product.rejected if quiet_block is None else np.asarray(quiet_block, dtype=bool) & product.selected & ~product.rejected
    if rho is not None and quiet.sum() >= MIN_NULL_FRAMES:
        exch = exchangeability_rate(fine_t[quiet], bulk, int(rho), designated)
    elif rho is not None:
        notes.append(f"exchangeability skipped: {int(quiet.sum())} quiet frames < {MIN_NULL_FRAMES}")
    return NullCalibration(
        channel=product.geometry.physical_channel, freq_id=product.geometry.freq_id, era_label=era_label,
        era_frames=int(era.sum()), null_source=source, mixture_declared=mixture, coarse=coarse, fine=fine,
        fine_designated_median=fine_designated_median, floor=floor, exchangeability=exch, bulk_size=int(bulk.sum()),
        quiet_frames=int((era & ~product.rejected).sum()), detected_frames=int((era & product.rejected).sum()),
        kept=kept, off_null_like=off_null, off_check=off_reason if off_widths is not None else "", off_coarse=off_widths,
        fine_bulk_size=int(fine_bulk.sum()), notes=tuple(notes))


def write_null_rows(results: Sequence[NullCalibration], path) -> None:
    import csv
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.as_row() for r in results]
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 10_000)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if (isinstance(v, float) and not math.isfinite(v)) or v is None
                                 else repr(v) if isinstance(v, float) else v) for k, v in row.items()})


# ------------------------------------------------------------ transmitter-off populations
NO_OFF_EPOCH = ("no transmitter-off epoch; the current-era bulk is a mixture "
                "(null source: bulk of the mixture (declared))")


def station_records(off_from: str | None, off_through: str | None) -> dict[str, str]:
    """Archive-inferred change dates; these are not independent station records."""
    out: dict[str, str] = {}
    if off_from:
        out[off_from] = "sign-off"
    if off_through:
        y, m = (int(x) for x in off_through.split("-"))
        idx = y * 12 + m            # the month after the off epoch
        out[blocks.month_label(idx)] = "sign-on"
    return out


@dataclass(frozen=True)
class OffPopulation:
    """A band's dated transmitter-off population, and whether it is a verified signal-free null.

    ``mask`` is the frame mask of the dated off interval restricted to the
    frames of the era table's proxy-low eras, so a month the section 8.1
    procedure classifies proxy-high or ambiguous never enters the null
    population even when the interval covers it (channel 20's record says
    2022-09 while the procedure reads the transmitter on through 2022-12).
    ``mask`` is None when no interval exists or nothing survives.

    ``independently_verified`` is the interval's flag (an external record
    confirms the date); ``null_like`` is :func:`off_population_check` on the
    mask (None when there is no population), two-sided. The population is
    signal-free only when both hold.
    """

    mask: np.ndarray | None
    off_through: str | None
    off_from: str | None
    record_frames: int
    off_frames: int
    note: str
    independently_verified: bool = False
    external_record: str = ""
    null_like: bool | None = None
    null_widths: NullWidths | None = None
    null_check: str = ""
    null_r68: float = math.nan          # the population's central-68 width ratio (the upper-tail clause)

    @property
    def signal_free(self) -> bool:
        """Both tests pass: independently verified and null-like."""
        return bool(self.mask is not None and self.independently_verified and self.null_like)

    @property
    def dated(self) -> str:
        if self.off_from:
            return f"from {self.off_from}"
        if self.off_through:
            return f"through {self.off_through}"
        return ""

    @property
    def rejection_reason(self) -> str:
        """Why this population is not a verified signal-free null ('' when it is one)."""
        if self.off_from is None and self.off_through is None:
            return NO_OFF_EPOCH
        if self.signal_free:
            return ""
        if self.independently_verified:
            if self.null_like is None:
                return f"off-state, not signal-free: {self.null_check}"
            return f"off-state, not signal-free: {self.null_check}"
        record = f"; external record {self.external_record}" if self.external_record else ""
        return (f"transmitter-off epoch dated from this archive ({self.dated}){record}; "
                "fails independent_verification")


def transmitter_off(product: Product, months: np.ndarray, intervals, table: eras.EraTable | None = None) -> OffPopulation:
    """Frames of the band's dated transmitter-off interval, restricted to proxy-low eras when a table is given.

    ``intervals`` are the band's :class:`~pilot_proxy.config.eras.OffInterval`
    entries. As in the release, one interval is read: an open-ended off epoch
    (``from`` set, ``through`` null) first, else one that runs from the start
    of the record (``from`` null).
    """
    intervals = tuple(intervals or ())
    after = next((i for i in intervals if i.from_month is not None and i.through_month is None), None)
    before = next((i for i in intervals if i.from_month is None and i.through_month is not None), None)
    chosen = after or before
    off_from = after.from_month if after is not None else None
    off_through = before.through_month if before is not None else None
    verified = bool(chosen.independently_verified) if chosen is not None else False
    external = getattr(chosen, "external_record", "") if chosen is not None else ""
    record = None
    if off_from:
        y, m = (int(x) for x in off_from.split("-"))
        record = product.selected & (months >= y * 12 + m - 1)
    elif off_through:
        y, m = (int(x) for x in off_through.split("-"))
        record = product.selected & (months <= y * 12 + m - 1)
    base = dict(off_through=off_through, off_from=off_from, independently_verified=verified, external_record=external)
    if record is None:
        return OffPopulation(None, record_frames=0, off_frames=0, note="", **base)
    if table is None:
        mask = record if record.any() else None
        ok, widths, reason, r68 = _off_check(product, mask)
        return OffPopulation(mask, record_frames=int(record.sum()), off_frames=int(record.sum()), note="",
                             null_like=(ok if widths is not None else None), null_widths=widths, null_check=reason,
                             null_r68=r68, **base)
    low = np.zeros(record.shape, dtype=bool)
    for index, era in enumerate(table.eras):
        if era.state == eras.PROXY_LOW:
            low |= table.era_mask(product, index)
    mask = record & low
    notes = []
    dropped = int(record.sum() - mask.sum())
    if dropped:
        notes.append(f"off population: {dropped} of {int(record.sum())} frames of the recorded off epoch fall outside the "
                     f"procedure's proxy-low eras and are excluded")
    if table.unmatched_station_records:
        notes.append("station record not matched by a transition: " + ", ".join(table.unmatched_station_records))
    mask = mask if mask.any() else None
    ok, widths, reason, r68 = _off_check(product, mask)
    return OffPopulation(mask, record_frames=int(record.sum()), off_frames=int(mask.sum()) if mask is not None else 0,
                         note="; ".join(notes), null_like=(ok if widths is not None else None), null_widths=widths,
                         null_check=reason, null_r68=r68, **base)


__all__ = ["NullWidths", "FloorEstimate", "NullCalibration", "OffPopulation", "describe_null", "core_scale",
           "floor_estimate", "calibrate_null", "write_null_rows", "iid_width", "station_records",
           "transmitter_off", "CORE_PROBES", "AS_CODED_PROBES", "COARSE_DOF", "FINE_DOF", "NO_OFF_EPOCH",
           "CENTRAL_68", "central68_ratio", "off_null_like", "Exchangeability", "exchangeability_rate"]
