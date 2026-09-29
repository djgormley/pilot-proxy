"""The emitter's in-band power plateau (the shelf) as a narrowband-marker product reports it.

A per-pilot product reports, per frame, the marker's normalized excess
converted to the emitter's in-band power relative to system noise in the band
(the *shelf*, dB): the product calls it ``estimated_data_shelf_snr_db`` and
derives it from exact integer powers after correcting the quantized-weight null
point, as

    shelf_db = excess_db + marker_to_band_db - 10 log10(B_band / bin_enbw)
               - 10 log10(capture efficiency)

It is finite only where the excess is positive.

:func:`shelf_statistics` measures what the product says about one band's
shelf: the transmitter-on level, the sensitivity floor over a dated off epoch
(or the kept frames), and a three-level nested variance decomposition of the
linear shelf keyed on sidereal day and acquisition unit (DC, inter-day,
intra-day, fast). The residual chain uses the level and, for its correlation
time, the decomposition (:mod:`pilot_proxy.characterization.residual_chain`);
whether the split is *booked* as a ground-filter credit is the integration
model's choice (``variance_split``), not this module's.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pilot_proxy.config.project import default_project
from pilot_proxy.products.npzio import load_npz
from pilot_proxy.products.reader import open_product

_REGISTER = default_project().register
SIDEREAL_DAY = float(_REGISTER.value("correlation.sidereal_day_seconds"))
DEFAULT_FLOOR_PERCENTILE = float(_REGISTER.value("floor.upper_percentile"))
DEFAULT_TRIM_PERCENTILE = float(
    _REGISTER.value("correlation.primary_trim_percentile"))
MIN_SHELF_SPLIT_FRAMES = int(
    _REGISTER.value("floor.minimum_shelf_split_frames"))
MIN_ACQUISITION_FRAMES = int(
    _REGISTER.value("correlation.minimum_frames_per_acquisition"))


@dataclass
class ShelfStatistics:
    """What a pilot-proxy survey product says about one channel's shelf.

    All powers are linear ratios to system noise in the DTV bandwidth; the
    ``_db`` fields are 10 log10 of the corresponding ratio.
    """
    channel: int
    freq_id: int
    nu_mhz: float
    n_valid: int
    n_kept: int
    on_shelf_db: float            # median shelf, transmitter on
    floor_db: float               # sensitivity floor -> bound on kept frames
    floor_percentile: float
    dc_fraction: float            # time-constant share of shelf power
    interday_fraction: float      # varies between sidereal days
    intraday_fraction: float      # varies between acquisitions within a day
    fast_fraction: float          # varies within an acquisition
    n_off_frames: int = 0
    n_units: int = 0
    n_days: int = 0
    trim_percentile: float | None = None

    @property
    def masked_fraction(self) -> float:
        return 1.0 - self.n_kept / self.n_valid if self.n_valid else 1.0

    @property
    def slow_fraction(self) -> float:
        """All variation slower than an acquisition."""
        return self.interday_fraction + self.intraday_fraction

    @property
    def surviving_fraction(self) -> float:
        """Share of shelf power the ground filter cannot remove.

        A residual constant *within* a sidereal day is m = 0 no matter how it
        drifts from one day to the next, because the common-mode filter is
        applied per day. So DC and inter-day both go; only variation on
        timescales shorter than a day lands at m != 0.
        """
        return self.intraday_fraction + self.fast_fraction

    @property
    def ground_filter_db(self) -> float:
        """Suppression from removing the m = 0 component."""
        var = self.surviving_fraction
        if var <= 0.0:
            return np.inf
        return float(10.0 * np.log10(1.0 / var))

    def summary(self) -> str:
        return (
            f"ch{self.channel:>3d} ({self.nu_mhz:7.2f} MHz, freq_id {self.freq_id})  "
            f"masked {100 * self.masked_fraction:5.2f}%\n"
            f"    shelf on-air      {self.on_shelf_db:7.2f} dB\n"
            f"    kept-frame bound  {self.floor_db:7.2f} dB "
            f"(p{self.floor_percentile:g} of {self.n_off_frames} null frames)\n"
            f"    power split       DC       {100 * self.dc_fraction:7.4f}%\n"
            f"                      inter-day{100 * self.interday_fraction:7.4f}%  "
            f"(m=0, removed)\n"
            f"                      intra-day{100 * self.intraday_fraction:7.4f}%  "
            f"(survives)\n"
            f"                      fast     {100 * self.fast_fraction:7.4f}%  "
            f"(averages as noise)\n"
            f"    ground filter     {self.ground_filter_db:7.2f} dB "
            f"({self.n_units} acquisitions over {self.n_days} sidereal days)"
        )


def _month_of(ctime: np.ndarray) -> np.ndarray:
    return np.array([dt.datetime.fromtimestamp(x, dt.timezone.utc).strftime("%Y-%m")
                     for x in ctime])


def _epoch_time_masks(month: np.ndarray, off_through: str | None,
                      off_from: str | None):
    """(on, off) time masks from a transmitter epoch specification.

    ``off_through`` is the last ``YYYY-MM`` of an off epoch that *precedes*
    the on epoch (a sign-on channel); ``off_from`` is the first ``YYYY-MM``
    of an off epoch that *follows* it (a sign-off channel). At most one may
    be given; with neither there is no epoch information and both masks are
    ``None``.
    """
    if off_through is not None and off_from is not None:
        raise ValueError("give at most one of off_through / off_from; a "
                         "channel with off epochs on both sides needs the "
                         "on epoch split into two products")
    if off_through is not None:
        return month > off_through, month <= off_through
    if off_from is not None:
        return month < off_from, month >= off_from
    return None, None


def _on_epoch(d, off_through: str | None, trim_percentile: float | None,
              off_from: str | None = None):
    """Transmitter-on frames, optionally with the burst tail trimmed.

    Gating on ``reject_mask`` is what makes this a transmitter-on selection
    without needing an off epoch: a rejected frame carries a positive pilot
    excess, so the shelf is demonstrably there. ``off_through`` (sign-on
    channels) or ``off_from`` (sign-off channels) narrows it further when a
    clean off epoch is known.

    Trimming matters for intermittent transmitters. On a channel whose shelf is
    a quiet baseline punctuated by strong bursts, the linear-power moments are
    entirely the bursts (on ch34 the top 1% of frames carry 99.5% of the
    variance), and a variance decomposition of that is meaningless. Trimming
    restricts to the regime the residual budget is actually about, since the
    bursts are exactly what the detector flags and removes.
    """
    view = open_product(d)
    valid = view.valid
    rejected = view.rejected
    shelf = view.shelf_db
    unit = view.frame_unit_index
    t0 = view.unit_time0_ctime

    on = valid & rejected & np.isfinite(shelf)
    if off_through is not None or off_from is not None:
        on_t, _ = _epoch_time_masks(_month_of(t0)[unit], off_through, off_from)
        on &= on_t
    if on.sum() and trim_percentile is not None:
        on &= shelf <= np.percentile(shelf[on], trim_percentile)
    return on, shelf, unit, t0


def _nested_split(lin, unit, t0):
    """(dc, interday, intraday, fast, n_units, n_days, v_fast_abs, groups).

    ``groups`` is (unit_mean, n_frames, t_start, sidereal_day) per acquisition,
    which the structure-function estimator reuses.
    """
    sday = np.floor(t0[unit] / SIDEREAL_DAY).astype(np.int64)
    key = sday * (int(unit.max()) + 1) + unit
    order = np.argsort(key, kind="stable")
    lin_s, key_s, day_s, unit_s = lin[order], key[order], sday[order], unit[order]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    parts = list(zip(np.split(lin_s, bounds), np.split(day_s, bounds),
                     np.split(unit_s, bounds)))
    groups = [(g, dd[0], uu[0]) for g, dd, uu in parts
              if g.size >= MIN_ACQUISITION_FRAMES]
    if not groups:
        return (np.nan,) * 4 + (0, 0, np.nan, None)

    within = np.concatenate([g - g.mean() for g, _, _ in groups])
    unit_means = np.array([g.mean() for g, _, _ in groups])
    n_frames = np.array([g.size for g, _, _ in groups])
    unit_days = np.array([dd for _, dd, _ in groups])
    unit_t = np.array([t0[uu] for _, _, uu in groups])

    day_means, intra = [], []
    for dd in np.unique(unit_days):
        m = unit_means[unit_days == dd]
        day_means.append(m.mean())
        if m.size >= 2:
            intra.append(m - m.mean())
    day_means = np.array(day_means)
    intra = np.concatenate(intra) if intra else np.zeros(1)

    grand = float(day_means.mean())
    v_day, v_intra, v_fast = day_means.var(), intra.var(), within.var()
    total = grand ** 2 + v_day + v_intra + v_fast
    if total <= 0:
        return (np.nan,) * 4 + (0, 0, np.nan, None)
    return (float(grand ** 2 / total), float(v_day / total),
            float(v_intra / total), float(v_fast / total),
            len(groups), int(np.unique(unit_days).size), float(v_fast),
            (unit_means, n_frames, unit_t, unit_days, grand))


def shelf_statistics(npz_path: str | Path, off_through: str | None = None,
                     floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
                     trim_percentile: float | None = DEFAULT_TRIM_PERCENTILE,
                     off_from: str | None = None) -> ShelfStatistics:
    """Measure the residual chain's data-driven terms from a survey product.

    ``off_through`` is the last ``YYYY-MM`` of a transmitter-off epoch for this
    channel; frames at or before it define the sensitivity floor. For a
    sign-off channel, whose off epoch *follows* the on epoch, pass
    ``off_from`` (the first off month) instead; at most one of the two may be
    given. Without either, the floor is read from the frames the detector
    *kept*; on a v5 product a kept frame's exact pilot excess is not
    positive, so it carries no shelf and this floor is NaN with
    ``n_off_frames`` 0 (only float rounding, with a power term above 2**53,
    could leave a kept frame a finite shelf).

    The power split is a three-level nested variance decomposition of the
    linear shelf over the transmitter-on epoch, keyed on sidereal day and
    acquisition unit:

        DC        = (grand mean)^2
        inter-day = variance of the per-sidereal-day means
        intra-day = variance of unit means about their own day's mean
        fast      = variance of frames about their own unit's mean

    The sidereal-day boundary is the one that matters. A residual constant
    within a day is m = 0 however much it drifts day to day, so DC and
    inter-day are both removed by the common-mode filter; only intra-day and
    faster variation reaches m != 0. Splitting at the acquisition instead --
    which lumps day-to-day drift in with intra-day variation, understates the
    ground filter and double-counts that power in the coherence term.
    """
    d = load_npz(npz_path)
    view = open_product(d)
    valid = view.valid
    rejected = view.rejected
    shelf = view.shelf_db
    unit = view.frame_unit_index
    t0 = view.unit_time0_ctime
    month = _month_of(t0)[unit]

    if off_through is not None or off_from is not None:
        _, off_t = _epoch_time_masks(month, off_through, off_from)
        off = valid & off_t
    else:
        off = valid & ~rejected

    finite_off = off & np.isfinite(shelf)
    if finite_off.sum() == 0:
        floor_db = float("nan")
    else:
        floor_db = float(np.percentile(shelf[finite_off], floor_percentile))

    on, _, _, _ = _on_epoch(d, off_through, trim_percentile, off_from=off_from)
    on_db = float(np.median(shelf[on])) if on.sum() else float("nan")

    dc = interday = intraday = fast = float("nan")
    n_units = n_days = 0
    if on.sum() >= MIN_SHELF_SPLIT_FRAMES:
        dc, interday, intraday, fast, n_units, n_days, _, _ = _nested_split(
            10.0 ** (shelf[on] / 10.0), unit[on], t0)

    return ShelfStatistics(
        channel=view.physical_channel,
        freq_id=view.freq_id,
        nu_mhz=view.chime_frequency_hz / 1e6,
        n_valid=int(valid.sum()),
        n_kept=int((valid & ~rejected).sum()),
        on_shelf_db=on_db,
        floor_db=floor_db,
        floor_percentile=float(floor_percentile),
        trim_percentile=trim_percentile,
        dc_fraction=dc, interday_fraction=interday,
        intraday_fraction=intraday, fast_fraction=fast,
        n_off_frames=int(finite_off.sum()),
        n_units=n_units, n_days=n_days,
    )


__all__ = ["DEFAULT_FLOOR_PERCENTILE", "DEFAULT_TRIM_PERCENTILE", "MIN_ACQUISITION_FRAMES",
           "MIN_SHELF_SPLIT_FRAMES", "SIDEREAL_DAY", "ShelfStatistics", "shelf_statistics"]
