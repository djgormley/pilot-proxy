"""Stable-era identification: the section 8.1 procedure as versioned configuration.

The dissertation (chapter 8, ``sec:calibration:eras``) fixes a predeclared,
calibration-free rule that reads each channel's era record from the survey
products; this module is that rule, with every policy value a field of
:class:`EraConfig` (versioned, and identified by the sha256 digest of its
canonical JSON, recorded in every output row). The definitions implemented
are quoted from the text:

- *Inputs.* "Per valid frame: UTC; the exact normalized coarse ratio
  ``F/mu_0`` (``mu_0`` from weight norms alone); the fine peak location
  within the *nominal* acquisition neighbourhood ``f_pred +- 30`` bins (never
  the era-calibrated anchor); ... and the recorded receiver software tag and
  input map."
- *Aggregation.* "UTC calendar month. A month is populated only with at
  least 30 valid frames drawn from at least 5 distinct acquisitions on at
  least 3 distinct days; policy values, with sensitivity values (3, 5, 10)
  acquisitions recorded." (``blocks.month_support`` applies the same gate.)
  The monthly statistic is the median over the month's frames of the
  per-frame ``10 log10(F/mu_0)`` (``Product.level_db``).
- *State.* "A populated month is *proxy-high* if its median
  ``10 log10(F/mu_0)`` is at least 1 dB, *proxy-low* if at most 0.5 dB, and
  *ambiguous* between (sensitivity values 0.5, 1, 2 dB) ... A *station
  change* is a shift of the monthly peak location by at least 3 fine bins
  (36 Hz) against the running era. An *instrument change* is a change in the
  recorded software tag or input map."
- *Persistence.* "A new state or anchor must hold for at least 2 consecutive
  populated months; a single-month excursion is flagged, not a transition.
  Ambiguous months inherit the state of their populated neighbours if those
  agree and open a transition otherwise."
- *Placement and uncertainty.* "A transition is placed between the last
  month of the old state and the first of the new; when unpopulated months
  lie between them it is placed at that gap and the gap's span is the
  boundary uncertainty."
- *Fallback.* "A channel with no proxy-low month has its current era bounded
  only by station and instrument changes."
- *Naming.* "A boundary is a *transmitter sign-on* or *sign-off* only where a
  station record confirms it; otherwise it is a *spectral-state transition*."
- The latest era is the "latest archive-observed stable era", written
  "current era", and is *stale-latest* when it ends before the snapshot.

Where the text leaves the procedure open this module takes the reading below
(each a named, documented parameter or rule; ``docs/archive-results-design.md``
section 6 records the instrument choice):

- **Monthly fine-peak location.** Per frame, the fine statistic
  ``T = 2 S_0 / (S_1 + S_2)`` on the exact terms, restricted to the window
  ``nominal_fine_bin +- window_half_width`` (mod 256); the per-frame peak is
  the argmax (ties to the lowest offset; a frame whose window is all zero has
  no peak). The month's location is the median of the per-frame peaks over
  the month's coarse-detected frames (the stored ``F > mu_0`` flag), in
  unwrapped bins about the nominal; a month with fewer than
  ``min_peak_cohort`` detected frames uses all its frames instead and is
  flagged (``peak_cohort = 'all_selected'``).
- **Inheritance and persistence to a fixed point.** Ambiguous months take the
  state of the nearest definite month on each side when those agree; between
  disagreeing states they form a *transition zone*. Runs of equal definite
  state over consecutive populated months shorter than ``persistence_months``
  are excursions: flagged, relabelled ambiguous, and re-resolved; the two
  passes iterate until nothing changes (each pass only removes definite
  labels, so it terminates). A leading or trailing ambiguous run has one
  populated neighbour and inherits it (``one_sided_ambiguous = 'inherit'``,
  the vacuous-agreement reading); ``'separate'`` makes such a run its own
  ``ambiguous-only`` era when it is long enough to persist.
- **Transition zone months belong to no era.** They lie inside the boundary
  interval: the boundary uncertainty is the number of calendar months
  strictly between the last month of the old state and the first month of the
  new, reported with its unpopulated (gap) and populated-ambiguous parts. The
  months are always populated months, including for an author-dated era
  whose first month is dated before its first populated month.
- **Station changes** are evaluated only on definite proxy-high months
  (``station_check_states``): a proxy-low month has no pilot and its peak
  location is noise. The running era location is the median of the accepted
  months' locations; a month shifted by at least ``station_shift_bins`` opens a
  station change only when the next tested month is also shifted against the
  same running location, otherwise it is a station excursion. The zone of a
  station change is the untested populated months between the last tested
  month of the old location and the first of the new.
- **Instrument change** is a change of ``unit_input_map_sha256`` (design
  section 6); a month's instrument state is the set of map values its
  acquisitions carry. A map that *enters* (present in a month, absent from
  the previous populated month) and stays present for ``persistence_months``
  consecutive populated months opens an instrument change at the month it
  enters; a map that enters and leaves within one month is an instrument
  excursion; a map that merely leaves is not a change. The software tag is
  recorded as a count only (``software_tags``, ``software_tag_changes``).
- **Eras** are the maximal runs of populated months not in any zone, cut at
  every boundary; an era's evidence is the kinds of boundary that opened it
  (``'+'``-joined when kinds coincide), ``'archive start'`` for the first.
  An era's frames are every masked frame whose UTC month lies in
  ``[first_month, last_month]``; a frame without a recorded sample interval
  (NaN time) follows its acquisition's ``unit_time`` month. An author-dated
  era (:func:`impose_eras`) may start before its first populated month only
  when no frame of the product lies between the two, so its frames are those
  of its populated months.
- **Stale-latest** compares the current era's last populated month with
  ``campaign_last_month`` (the last populated month over the whole campaign,
  from :func:`campaign_last_populated_month`): the era is stale when it ends
  more than ``stale_grace_months`` (policy 1: the snapshot month is partial) before it,
  and the lag is reported (``stale_lag_months``). When the caller cannot
  supply the campaign month the channel's own last month is used and
  ``stale_reference`` says so.
- **Indeterminate** (reported, never resolved by hand): the ambiguous months
  are a majority of the populated months (``indeterminate_fraction``), no
  definite state persists, or the current era is ambiguous-only.
- **Sensitivity** (no PELT): the segmentation is repeated under each
  ``units_sensitivity`` gate and each ``threshold_sensitivity_db`` pair, and
  whether the current era's first or last month moves is recorded.

Output, ``tables/eras.csv``, one row per era, columns in order:
``channel, freq_id, era, n_eras, first_month, last_month, state, evidence,
record_agreement, boundary_uncertainty_months, boundary_gap_months,
boundary_ambiguous_months, units, frames, frames_without_time,
populated_months, months_spanned, coverage, level_median_db,
peak_offset_bins, is_current, stale_latest, fallback, config_version,
config_digest``. ``state`` is ``proxy-high | proxy-low | ambiguous-only``;
``coverage = populated_months / months_spanned``; ``peak_offset_bins`` is the
median monthly location of the era's definite proxy-high months in fine bins
from the nominal bin (fine-array direction; a feature of the record, not the
anchor of chapter 8's next section). Months are ``YYYY-MM``.

``tables/eras_channels.csv``, one row per channel: ``channel, freq_id,
n_eras, current_era, current_first_month, current_last_month, current_state,
current_evidence, current_boundary_uncertainty_months, current_units,
current_frames, stale_latest, stale_lag_months, stale_reference,
campaign_last_month, fallback, indeterminate, populated_months, ambiguous_months, ambiguous_fraction,
transition_zone_months, state_excursions, station_excursions,
instrument_change_months, instrument_excursions, software_tags,
software_tag_changes, frames_selected, frames_without_time,
peak_cohort_fallback_months, sensitivity_units, sensitivity_units_moves,
sensitivity_thresholds, sensitivity_thresholds_moves, config_version,
config_digest``. List columns are ``;``-joined; the sensitivity columns give
``value:first..last`` per setting and the settings under which the current
era moves. The per-channel JSON carries the same plus every month's record.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pilot_proxy.products.reader import Product, fine_power_ratio

from . import blocks

PROXY_HIGH = "proxy-high"
PROXY_LOW = "proxy-low"
AMBIGUOUS = "ambiguous"
AMBIGUOUS_ONLY = "ambiguous-only"
ZONE = "transition-zone"

EVIDENCE_START = "archive start"
EVIDENCE_STATE = "spectral-state transition"
EVIDENCE_STATION = "station change"
EVIDENCE_INSTRUMENT = "instrument change"
EVIDENCE_SIGN_ON = "transmitter sign-on"
EVIDENCE_SIGN_OFF = "transmitter sign-off"
EVIDENCE_POWER = "power transition"

PREDECLARED_RULE = "section 8.1 rule"
SOURCE_AUTHOR = "author-dated"

ERA_COLUMNS = (
    "channel", "freq_id", "era", "n_eras", "first_month", "last_month", "state", "evidence",
    "record_agreement", "boundary_uncertainty_months", "boundary_gap_months", "boundary_ambiguous_months",
    "units", "frames", "frames_without_time", "populated_months", "months_spanned", "coverage",
    "level_median_db", "peak_offset_bins", "peak_drift_bins_per_month", "peak_range_bins", "peak_months",
    "is_current", "stale_latest", "fallback", "boundary_source",
    "config_version", "config_digest",
)
CHANNEL_COLUMNS = (
    "channel", "freq_id", "n_eras", "current_era", "current_first_month", "current_last_month",
    "current_state", "current_evidence", "current_boundary_uncertainty_months", "current_units",
    "current_frames", "stale_latest", "stale_lag_months", "stale_reference", "campaign_last_month", "fallback",
    "indeterminate",
    "populated_months", "ambiguous_months", "ambiguous_fraction", "transition_zone_months",
    "state_excursions", "station_excursions", "unmatched_station_records", "instrument_change_months",
    "instrument_excursions", "unconfirmed_instrument_change_last_month",
    "current_peak_drift_bins_per_month", "current_peak_range_bins", "software_tags", "software_tag_changes", "frames_selected", "frames_without_time",
    "peak_cohort_fallback_months", "sensitivity_units", "sensitivity_units_moves",
    "sensitivity_thresholds", "sensitivity_thresholds_moves", "boundary_source", "rule_current_era",
    "config_version", "config_digest",
)


# ------------------------------------------------------------------ config
@dataclass(frozen=True)
class EraConfig:
    """Every policy value of the section 8.1 procedure, versioned and digested."""

    version: str = "section-8.1-eras-v2"
    min_frames: int = blocks.MONTH_MIN_FRAMES
    min_units: int = blocks.MONTH_MIN_UNITS
    min_days: int = blocks.MONTH_MIN_DAYS
    high_db: float = 1.0                       # proxy-high: monthly median level >= high_db
    low_db: float = 0.5                        # proxy-low: monthly median level <= low_db
    station_shift_bins: float = 3.0            # station change: shift of the monthly location, fine bins
    window_half_width: int = 30                # nominal acquisition neighbourhood, fine bins
    min_peak_cohort: int = 30                  # detected frames needed for the monthly location cohort
    persistence_months: int = 2                # populated months a new state, anchor or map must hold
    one_sided_ambiguous: str = "inherit"       # 'inherit' | 'separate' for leading/trailing ambiguous runs
    station_check_states: tuple[str, ...] = (PROXY_HIGH,)
    instrument_field: str = "unit_input_map_sha256"
    indeterminate_fraction: float = 0.5        # ambiguous majority -> rule indeterminate
    stale_grace_months: int = 1                # stale-latest when the era ends more than this before the snapshot
                                               # (1: the snapshot month itself is partial and is not held against a channel)
    units_sensitivity: tuple[int, ...] = blocks.MONTH_MIN_UNITS_SENSITIVITY
    threshold_sensitivity_db: tuple[tuple[float, float], ...] = (
        (0.5, 0.5), (0.5, 1.0), (0.5, 2.0), (1.0, 1.0), (1.0, 2.0), (2.0, 2.0))

    def __post_init__(self):
        if not self.low_db <= self.high_db:
            raise ValueError("low_db must not exceed high_db")
        if self.persistence_months < 1:
            raise ValueError("persistence_months must be at least 1")
        if not 0 <= self.window_half_width < 128:
            raise ValueError("window_half_width must lie in [0, 128)")
        if self.one_sided_ambiguous not in ("inherit", "separate"):
            raise ValueError("one_sided_ambiguous must be 'inherit' or 'separate'")
        for low, high in self.threshold_sensitivity_db:
            if not low <= high:
                raise ValueError("every sensitivity pair must satisfy low <= high")

    def canonical_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def replace(self, **changes) -> "EraConfig":
        return dataclasses.replace(self, **changes)


DEFAULT_CONFIG = EraConfig()


def state_of(level_db: float, config: EraConfig = DEFAULT_CONFIG) -> str:
    """The section 8.1 state of a monthly median level (NaN is ambiguous)."""
    if not math.isfinite(level_db):
        return AMBIGUOUS
    if level_db >= config.high_db:
        return PROXY_HIGH
    if level_db <= config.low_db:
        return PROXY_LOW
    return AMBIGUOUS


# ---------------------------------------------------------- month records
@dataclass(frozen=True)
class MonthRecord:
    """The calibration-free monthly features of one UTC month with masked frames.

    ``populated`` and ``state`` are evaluated under the policy configuration;
    the sensitivity runs re-derive them from ``frames``, ``units``, ``days``
    and ``level_db``.
    """

    month: int
    frames: int
    units: int
    days: int
    populated: bool
    level_db: float
    state: str
    peak_offset_bins: float           # median per-frame peak, fine bins from the nominal bin; NaN if none
    peak_cohort: str                  # 'detected' | 'all_selected' | 'none'
    peak_frames: int
    input_maps: tuple[str, ...]
    software_tags: tuple[str, ...]

    @property
    def label(self) -> str:
        return blocks.month_label(self.month)


def is_populated(record: MonthRecord, config: EraConfig) -> bool:
    return (record.frames >= config.min_frames and record.units >= config.min_units
            and record.days >= config.min_days)


def frame_peak_offsets(product: Product, mask, *, half_width: int = DEFAULT_CONFIG.window_half_width) -> np.ndarray:
    """Per-frame fine-peak offset from the nominal bin within ``+- half_width`` bins.

    The fine statistic is formed from the exact terms for the masked frames
    only, on the window columns only; the whole ``fine_power_u64`` member is
    read once. NaN outside the mask and where the window is all zero.
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.full(product.n_frames, np.nan)
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return out
    offsets = np.arange(-int(half_width), int(half_width) + 1)
    terms = product.fine_terms_all()
    window = (product.geometry.nominal_fine_bin + offsets) % terms.shape[2]
    sub = terms[np.ix_(rows, np.arange(terms.shape[1]), window)]
    del terms
    ratio = fine_power_ratio(sub)
    peak = ratio.argmax(axis=1)
    has_peak = ratio.max(axis=1) > 0.0
    out[rows[has_peak]] = offsets[peak[has_peak]]
    return out


def frame_months(product: Product) -> np.ndarray:
    """UTC month of each frame; a frame without a recorded time follows its acquisition's month."""
    months = blocks.month_index(product.frame_time)
    missing = months < 0
    if missing.any():
        months[missing] = blocks.month_index(product.unit_time[missing])
    return months


def month_records(product: Product, mask, config: EraConfig = DEFAULT_CONFIG) -> tuple[MonthRecord, ...]:
    """One record per UTC month holding at least one masked, timed frame, in calendar order."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (product.n_frames,):
        raise ValueError("mask must be a boolean (N,) frame mask")
    t = product.frame_time
    timed = mask & np.isfinite(t)
    months = blocks.month_index(t)
    days = blocks.day_index(t)
    units = product.frame_unit_index
    level = product.level_db
    detected = product.rejected
    peak = frame_peak_offsets(product, mask, half_width=config.window_half_width)
    maps = getattr(product, config.instrument_field)
    tags = product.unit_git_version_tag
    out = []
    for m in np.unique(months[timed]):
        here = timed & (months == m)
        here_units = np.unique(units[here])
        levels = level[here]
        levels = levels[np.isfinite(levels)]
        level_db = float(np.median(levels)) if levels.size else float("nan")
        cohort = here & detected
        cohort_name = "detected"
        if int(cohort.sum()) < config.min_peak_cohort:
            cohort, cohort_name = here, "all_selected"
        peaks = peak[cohort]
        peaks = peaks[np.isfinite(peaks)]
        if peaks.size == 0:
            cohort_name = "none"
        record = MonthRecord(
            month=int(m), frames=int(here.sum()), units=int(here_units.size), days=int(np.unique(days[here]).size),
            populated=False, level_db=level_db, state=AMBIGUOUS,
            peak_offset_bins=float(np.median(peaks)) if peaks.size else float("nan"),
            peak_cohort=cohort_name, peak_frames=int(peaks.size),
            input_maps=tuple(sorted({str(v) for v in maps[here_units]})),
            software_tags=tuple(sorted({str(v) for v in tags[here_units]})),
        )
        populated = is_populated(record, config)
        out.append(dataclasses.replace(record, populated=populated,
                                       state=state_of(level_db, config) if populated else "unpopulated"))
    return tuple(out)


# ------------------------------------------------------------ segmentation
@dataclass(frozen=True)
class Boundary:
    """One transition: between the last month of the old segment and the first of the new."""

    kind: str
    old_last: int
    new_first: int
    zone: tuple[int, ...]             # populated months strictly between, excluded from eras
    old_state: str = ""
    new_state: str = ""


@dataclass(frozen=True)
class Excursion:
    """A single-month (or sub-persistence) departure that is flagged, not a transition."""

    month: int
    kind: str                         # 'state' | 'station' | 'instrument'
    detail: str

    @property
    def label(self) -> str:
        return f"{blocks.month_label(self.month)}:{self.kind}:{self.detail}"


@dataclass(frozen=True)
class Era:
    """One stable era: a contiguous run of populated months with one state and one instrument.

    ``first_month`` is the first populated month, except for an author-dated era
    whose first month is dated inside a data gap (:func:`impose_eras`); no frame
    lies between the two, and ``months`` holds the populated months only.
    """

    index: int
    first_month: int
    last_month: int
    state: str
    evidence: str
    boundary_uncertainty_months: int
    boundary_gap_months: int
    boundary_ambiguous_months: int
    months: tuple[int, ...]           # the populated months
    peak_offset_bins: float
    record_agreement: str = ""
    units: int = 0
    frames: int = 0
    frames_without_time: int = 0
    level_median_db: float = float("nan")
    peak_drift_bins_per_month: float = float("nan")   # least-squares slope of the tested months' peak location
    peak_range_bins: float = float("nan")             # max - min of the tested months' peak location
    peak_months: int = 0                              # tested (definite proxy-high, located) months in the era

    @property
    def populated_months(self) -> int:
        return len(self.months)

    @property
    def months_spanned(self) -> int:
        return self.last_month - self.first_month + 1

    @property
    def coverage(self) -> float:
        return self.populated_months / self.months_spanned if self.months_spanned > 0 else float("nan")

    @property
    def first_label(self) -> str:
        return blocks.month_label(self.first_month)

    @property
    def last_label(self) -> str:
        return blocks.month_label(self.last_month)


@dataclass(frozen=True)
class Segmentation:
    """The procedure's result on monthly records alone (no frame counts yet)."""

    eras: tuple[Era, ...]
    boundaries: tuple[Boundary, ...]
    populated: tuple[int, ...]                 # populated months in order
    resolved: dict                             # month -> resolved state (or ZONE)
    raw_states: dict                           # month -> raw state
    excursions: tuple[Excursion, ...]
    zone_months: tuple[int, ...]
    instrument_change_months: tuple[int, ...]
    fallback: str
    unmatched_station_records: tuple[str, ...]

    @property
    def ambiguous_months(self) -> int:
        return sum(1 for m in self.populated if self.raw_states[m] == AMBIGUOUS)


def _inherit(labels: list[str], config: EraConfig) -> list[str]:
    """Resolve ambiguous labels from the nearest definite labels on each side."""
    n = len(labels)
    definite = [i for i, s in enumerate(labels) if s in (PROXY_HIGH, PROXY_LOW)]
    if not definite:
        return [AMBIGUOUS_ONLY] * n
    resolved = list(labels)
    i = 0
    while i < n:
        if labels[i] != AMBIGUOUS:
            i += 1
            continue
        j = i
        while j < n and labels[j] == AMBIGUOUS:
            j += 1
        before = labels[i - 1] if i > 0 else None
        after = labels[j] if j < n else None
        if before is None or after is None:
            neighbour = after if before is None else before
            separate = config.one_sided_ambiguous == "separate" and (j - i) >= config.persistence_months
            fill = AMBIGUOUS_ONLY if separate else neighbour
        elif before == after:
            fill = before
        else:
            fill = ZONE
        for k in range(i, j):
            resolved[k] = fill
        i = j
    return resolved


def _resolve_states(raw: list[str], config: EraConfig) -> tuple[list[str], list[int]]:
    """Inheritance and persistence iterated to a fixed point.

    Returns the resolved label per populated month and the positions whose
    definite raw state was demoted to an excursion.
    """
    labels = [s if s in (PROXY_HIGH, PROXY_LOW) else AMBIGUOUS for s in raw]
    while True:
        resolved = _inherit(labels, config)
        demoted = []
        i = 0
        n = len(resolved)
        while i < n:
            if resolved[i] not in (PROXY_HIGH, PROXY_LOW):
                i += 1
                continue
            j = i
            while j < n and resolved[j] == resolved[i]:
                j += 1
            # persistence counts definite months only: an inherited ambiguous month does not hold a state
            definite = [k for k in range(i, j) if labels[k] != AMBIGUOUS]
            if len(definite) < config.persistence_months:
                demoted.extend(definite)
            i = j
        if not demoted:
            break
        for k in demoted:
            labels[k] = AMBIGUOUS
    excursions = [k for k in range(len(raw)) if raw[k] in (PROXY_HIGH, PROXY_LOW) and labels[k] == AMBIGUOUS]
    return resolved, excursions


def _segments(resolved: list[str]) -> list[tuple[str, list[int]]]:
    """Maximal runs of equal resolved state over consecutive positions; zone months break runs."""
    out: list[tuple[str, list[int]]] = []
    for i, s in enumerate(resolved):
        if s == ZONE:
            continue
        if out and out[-1][0] == s and out[-1][1][-1] == i - 1:
            out[-1][1].append(i)
        else:
            out.append((s, [i]))
    return out


def _station_boundaries(pop: Sequence[MonthRecord], raw: list[str], positions: list[int],
                        config: EraConfig) -> tuple[list[Boundary], list[Excursion]]:
    tested = [i for i in positions if raw[i] == PROXY_HIGH and math.isfinite(pop[i].peak_offset_bins)]
    boundaries: list[Boundary] = []
    excursions: list[Excursion] = []
    running: list[float] = []
    last_tested = None
    skip = -1
    for k, i in enumerate(tested):
        loc = pop[i].peak_offset_bins
        if not running:
            # seed: the first tested month starts the running location unless the following
            # persistence_months tested months agree with each other and disagree with it,
            # in which case the first month is the excursion and they seed the location
            follow = tested[k + 1: k + 1 + config.persistence_months]
            if len(follow) == config.persistence_months:
                locs = [pop[q].peak_offset_bins for q in follow]
                agree = all(abs(v - locs[0]) < config.station_shift_bins for v in locs[1:])
                if agree and all(abs(v - loc) >= config.station_shift_bins for v in locs):
                    excursions.append(Excursion(pop[i].month, "station", f"{loc - locs[0]:+.1f} bins"))
                    last_tested = i
                    continue
            running.append(loc)
            last_tested = i
            continue
        if k == skip:
            # the month that confirmed a change is already part of the new running location
            running.append(loc)
            last_tested = i
            continue
        reference = float(np.median(running))
        shift = loc - reference
        if abs(shift) >= config.station_shift_bins:
            nxt = tested[k + 1] if k + 1 < len(tested) else None
            # a change holds when the next tested month sits at the new location, not merely away from the old one
            holds = (nxt is not None and abs(pop[nxt].peak_offset_bins - reference) >= config.station_shift_bins
                     and abs(pop[nxt].peak_offset_bins - loc) < config.station_shift_bins)
            if holds:
                zone = tuple(pop[p].month for p in positions if last_tested < p < i)
                boundaries.append(Boundary(EVIDENCE_STATION, pop[last_tested].month, pop[i].month, zone,
                                           f"{reference:+.1f} bins", f"{loc:+.1f} bins"))
                running = [loc]
                skip = k + 1
            else:
                excursions.append(Excursion(pop[i].month, "station", f"{shift:+.1f} bins"))
        else:
            running.append(loc)
        last_tested = i
    return boundaries, excursions


def _instrument_boundaries(pop: Sequence[MonthRecord], config: EraConfig) -> tuple[list[Boundary], list[Excursion]]:
    boundaries: list[Boundary] = []
    excursions: list[Excursion] = []
    previous: set[str] | None = None
    for k, record in enumerate(pop):
        maps = set(record.input_maps)
        if previous is not None:
            opened = False
            for value in sorted(maps - previous):
                window = pop[k: k + config.persistence_months]
                holds = len(window) == config.persistence_months and all(value in r.input_maps for r in window)
                if holds and not opened:
                    boundaries.append(Boundary(EVIDENCE_INSTRUMENT, pop[k - 1].month, record.month, (),
                                               ";".join(sorted(previous)), ";".join(sorted(maps))))
                    opened = True
                elif not holds:
                    excursions.append(Excursion(record.month, "instrument", value[:8]))
        previous = maps
    return boundaries, excursions


def _month_of_label(label: str) -> int:
    year, month = label.split("-")
    return int(year) * 12 + int(month) - 1


def segment(records: Sequence[MonthRecord], config: EraConfig = DEFAULT_CONFIG,
            station_record: Mapping[str, str] | None = None) -> Segmentation:
    """Run the procedure on monthly records; pure, so the sensitivity runs reuse it.

    ``station_record`` maps ``YYYY-MM`` (the first month of the recorded new
    state) to ``'sign-on'`` or ``'sign-off'``; a spectral-state transition
    whose boundary interval contains the record month in the matching
    direction is named a transmitter sign-on/sign-off.
    """
    pop = [r for r in records if is_populated(r, config)]
    if not pop:
        return Segmentation((), (), (), {}, {}, (), (), (), "no_populated_months",
                            tuple(sorted(station_record or ())))
    raw = [state_of(r.level_db, config) for r in pop]
    resolved, demoted = _resolve_states(raw, config)
    excursions = [Excursion(pop[k].month, "state", raw[k]) for k in demoted]
    boundaries: list[Boundary] = []
    segments = _segments(resolved)
    for (state_a, pos_a), (state_b, pos_b) in zip(segments, segments[1:]):
        zone = tuple(pop[p].month for p in range(pos_a[-1] + 1, pos_b[0]))
        boundaries.append(Boundary(EVIDENCE_STATE, pop[pos_a[-1]].month, pop[pos_b[0]].month, zone, state_a, state_b))
    for state, positions in segments:
        if state in config.station_check_states:
            found, flagged = _station_boundaries(pop, raw, positions, config)
            boundaries.extend(found)
            excursions.extend(flagged)
    found, flagged = _instrument_boundaries(pop, config)
    boundaries.extend(found)
    excursions.extend(flagged)
    boundaries.sort(key=lambda b: (b.new_first, b.kind))

    zone_months = sorted({m for b in boundaries for m in b.zone})
    cuts = {b.new_first for b in boundaries}
    eras_months: list[list[int]] = []
    previous_in_zone = True
    for record in pop:
        if record.month in zone_months:
            previous_in_zone = True
            continue
        if previous_in_zone or record.month in cuts or not eras_months:
            eras_months.append([record.month])
        else:
            eras_months[-1].append(record.month)
        previous_in_zone = False

    position = {r.month: i for i, r in enumerate(pop)}
    record_months = {_month_of_label(k): v for k, v in (station_record or {}).items()}
    matched: set[int] = set()
    eras: list[Era] = []
    for index, months in enumerate(eras_months):
        first, last = months[0], months[-1]
        prev_last = eras[-1].last_month if eras else None
        state = resolved[position[first]]
        opened = [b for b in boundaries if prev_last is not None and prev_last < b.new_first <= first]
        kinds = sorted({b.kind for b in opened})
        agreement = ""
        if EVIDENCE_STATE in kinds and record_months:
            expected = {(PROXY_LOW, PROXY_HIGH): "sign-on", (PROXY_HIGH, PROXY_LOW): "sign-off"}.get(
                (eras[-1].state, state))
            for m, kind in sorted(record_months.items()):
                if prev_last < m <= first:
                    matched.add(m)
                    if kind == expected:
                        kinds[kinds.index(EVIDENCE_STATE)] = EVIDENCE_SIGN_ON if kind == "sign-on" else EVIDENCE_SIGN_OFF
                        agreement = f"confirmed by record {blocks.month_label(m)}"
                    else:
                        agreement = f"record {blocks.month_label(m)} says {kind}; direction disagrees"
        evidence = "+".join(kinds) if kinds else EVIDENCE_START
        uncertainty = 0 if prev_last is None else first - prev_last - 1
        ambiguous_between = 0 if prev_last is None else sum(1 for m in zone_months if prev_last < m < first)
        located = [(m, pop[position[m]].peak_offset_bins) for m in months
                   if raw[position[m]] == PROXY_HIGH and math.isfinite(pop[position[m]].peak_offset_bins)]
        peaks = [v for _, v in located]
        drift = float("nan")
        if len(located) >= 3:
            xs = np.array([m for m, _ in located], dtype=float)
            ys = np.array(peaks, dtype=float)
            drift = float(np.polyfit(xs - xs.mean(), ys, 1)[0])
        eras.append(Era(index, first, last, state, evidence, uncertainty, uncertainty - ambiguous_between,
                        ambiguous_between, tuple(months), float(np.median(peaks)) if peaks else float("nan"),
                        agreement, peak_drift_bins_per_month=drift,
                        peak_range_bins=float(max(peaks) - min(peaks)) if peaks else float("nan"), peak_months=len(peaks)))
    fallback = []
    if PROXY_LOW not in raw:
        fallback.append("no_off_state")
    if not any(s in (PROXY_HIGH, PROXY_LOW) for s in resolved):
        fallback.append("no_persistent_state")
    unmatched = tuple(blocks.month_label(m) for m in sorted(record_months) if m not in matched)
    return Segmentation(tuple(eras), tuple(boundaries), tuple(r.month for r in pop),
                        {r.month: s for r, s in zip(pop, resolved)}, {r.month: s for r, s in zip(pop, raw)},
                        tuple(excursions), tuple(zone_months),
                        tuple(b.new_first for b in boundaries if b.kind == EVIDENCE_INSTRUMENT),
                        ";".join(fallback), unmatched)


# --------------------------------------------------------------- the table
@dataclass(frozen=True)
class SensitivityOutcome:
    """The current era under one alternative setting, and whether it moved."""

    parameter: str                    # 'min_units' | 'thresholds_db'
    value: str
    first_month: int                  # -1 when no era
    last_month: int
    n_eras: int
    moves: bool

    @property
    def span(self) -> str:
        if self.first_month < 0:
            return "none"
        return f"{blocks.month_label(self.first_month)}..{blocks.month_label(self.last_month)}"


@dataclass(frozen=True)
class EraTable:
    """The era record of one channel under one configuration and one frame mask."""

    channel: int
    freq_id: int
    config: EraConfig
    eras: tuple[Era, ...]
    months: tuple[MonthRecord, ...]
    resolved: dict
    current_index: int
    stale_latest: bool
    stale_lag_months: int             # campaign last month minus the current era's last month; -1 without an era
    stale_reference: str              # 'campaign' | 'channel'
    campaign_last_month: int
    fallback: str
    indeterminate: str
    ambiguous_months: int
    ambiguous_fraction: float
    transition_zone_months: tuple[int, ...]
    excursions: tuple[Excursion, ...]
    instrument_change_months: tuple[int, ...]
    software_tags: int
    software_tag_changes: int
    frames_selected: int
    frames_without_time: int
    peak_cohort_fallback_months: tuple[int, ...]
    sensitivity: tuple[SensitivityOutcome, ...]
    unmatched_station_records: tuple[str, ...]
    boundary_source: str = PREDECLARED_RULE      # who placed the eras: the rule, or an author-dated list (impose_eras)
    rule_current_era: str = ""              # the rule's own current era, kept beside an imposed one

    @property
    def current_era(self) -> Era | None:
        return self.eras[self.current_index] if 0 <= self.current_index < len(self.eras) else None

    @property
    def populated_months(self) -> int:
        return sum(1 for m in self.months if m.populated)

    def era_mask(self, product: Product, index: int, mask=None) -> np.ndarray:
        """Frames of one era: masked frames whose month lies within the era's months."""
        base = product.selected if mask is None else np.asarray(mask, dtype=bool)
        era = self.eras[index]
        months = frame_months(product)
        return base & (months >= era.first_month) & (months <= era.last_month)

    def current_era_mask(self, product: Product, mask=None) -> np.ndarray:
        """Frames of the current era (all False when the channel has no era)."""
        if self.current_era is None:
            return np.zeros(product.n_frames, dtype=bool)
        return self.era_mask(product, self.current_index, mask)


def _software_tag_changes(product: Product, mask) -> tuple[int, int]:
    units = np.unique(product.frame_unit_index[mask])
    if units.size == 0:
        return 0, 0
    order = units[np.argsort(product.unit_time0[units], kind="stable")]
    tags = product.unit_git_version_tag[order]
    return int(np.unique(tags).size), int((tags[1:] != tags[:-1]).sum())


def era_table(product: Product, mask=None, config: EraConfig = DEFAULT_CONFIG, *,
              campaign_last_month: int | None = None,
              station_record: Mapping[str, str] | None = None) -> EraTable:
    """The section 8.1 procedure on one product over the given frame mask.

    ``mask`` defaults to ``product.selected`` (valid frames passing the health
    gate). ``campaign_last_month`` is the last populated month over the whole
    campaign (:func:`campaign_last_populated_month`); without it the channel's
    own last month is used and ``stale_latest`` cannot be true.
    """
    mask = product.selected if mask is None else np.asarray(mask, dtype=bool)
    records = month_records(product, mask, config)
    seg = segment(records, config, station_record)
    months = frame_months(product)
    timed = np.isfinite(product.frame_time)
    level = product.level_db
    eras = []
    for era in seg.eras:
        here = mask & (months >= era.first_month) & (months <= era.last_month)
        levels = level[here]
        levels = levels[np.isfinite(levels)]
        eras.append(dataclasses.replace(
            era, units=int(np.unique(product.frame_unit_index[here]).size), frames=int(here.sum()),
            frames_without_time=int((here & ~timed).sum()),
            level_median_db=float(np.median(levels)) if levels.size else float("nan")))
    current = len(eras) - 1
    own_last = seg.populated[-1] if seg.populated else -1
    if campaign_last_month is None:
        reference, campaign_last = "channel", own_last
    else:
        reference, campaign_last = "campaign", int(campaign_last_month)
    lag = campaign_last - eras[-1].last_month if eras else -1
    stale = bool(eras) and lag > config.stale_grace_months

    outcomes = []
    policy = (eras[-1].first_month, eras[-1].last_month) if eras else (-1, -1)

    def outcome(parameter, value, alt):
        alt_seg = segment(records, alt, station_record)
        span = (alt_seg.eras[-1].first_month, alt_seg.eras[-1].last_month) if alt_seg.eras else (-1, -1)
        return SensitivityOutcome(parameter, value, span[0], span[1], len(alt_seg.eras), span != policy)

    for units in config.units_sensitivity:
        outcomes.append(outcome("min_units", str(int(units)), config.replace(min_units=int(units))))
    for low, high in config.threshold_sensitivity_db:
        outcomes.append(outcome("thresholds_db", f"{low:g}/{high:g}", config.replace(low_db=low, high_db=high)))

    n_pop = len(seg.populated)
    fraction = seg.ambiguous_months / n_pop if n_pop else float("nan")
    reasons = []
    if n_pop and fraction > config.indeterminate_fraction:
        reasons.append("ambiguous majority")
    if "no_persistent_state" in seg.fallback:
        reasons.append("no persistent definite state")
    if eras and eras[-1].state == AMBIGUOUS_ONLY:
        reasons.append("current era ambiguous-only")
    tags, tag_changes = _software_tag_changes(product, mask)
    return EraTable(
        channel=product.geometry.physical_channel, freq_id=product.geometry.freq_id, config=config,
        eras=tuple(eras), months=records, resolved=dict(seg.resolved), current_index=current,
        stale_latest=stale, stale_lag_months=lag, stale_reference=reference, campaign_last_month=campaign_last,
        fallback=seg.fallback, indeterminate="; ".join(reasons),
        ambiguous_months=seg.ambiguous_months, ambiguous_fraction=fraction,
        transition_zone_months=seg.zone_months, excursions=seg.excursions,
        instrument_change_months=seg.instrument_change_months, software_tags=tags, software_tag_changes=tag_changes,
        frames_selected=int(mask.sum()), frames_without_time=int((mask & ~timed).sum()),
        peak_cohort_fallback_months=tuple(r.month for r in records if r.populated and r.peak_cohort != "detected"),
        sensitivity=tuple(outcomes), unmatched_station_records=seg.unmatched_station_records,
    )


def _span_label(era: Era | None) -> str:
    return f"{era.first_label}..{era.last_label}" if era else ""


def impose_eras(table: EraTable, product: Product, spec: Sequence[Sequence[str]], mask=None, *,
                source: str = SOURCE_AUTHOR) -> EraTable:
    """Replace the rule's eras with an externally dated list, keeping the rule's month record.

    ``spec`` is the channel's whole era list in calendar order, each entry
    ``(first_month, last_month, evidence)`` with months as ``YYYY-MM``. An era is
    the populated months inside its span, bounded by the first and last of them;
    its first month may instead be the span's own first month when that month
    is dated before the first populated month and no frame of the product
    (selected or not, timed or not) lies between the two, so the era's frames
    never depend on where inside the gap the date falls; frames there refuse the
    list. The boundary uncertainty is always the months strictly between the
    previous era's last populated month and this era's first populated month.
    A populated month inside no span belongs to no era and is reported as a
    transition-zone month, so the placement convention is the rule's. The state
    of an era is the section 8.1 state of its frame-median level. Everything the
    rule recorded per month (states, excursions, instrument changes, peak
    locations) is kept as it is; the rule's sensitivity outcomes are dropped,
    because its thresholds no longer place the eras, and its own current era is
    kept in ``rule_current_era``.
    """
    mask = product.selected if mask is None else np.asarray(mask, dtype=bool)
    config = table.config
    populated = [r for r in table.months if r.populated]
    raw = {r.month: state_of(r.level_db, config) for r in populated}
    record = {r.month: r for r in populated}
    spans = [(_month_of_label(a), _month_of_label(b), str(evidence)) for a, b, evidence in spec]
    if not spans:
        raise ValueError("an era list needs at least one era")
    for (a, b, _), (c, _, _) in zip(spans, spans[1:] + [(None, None, None)]):
        if b < a or (c is not None and c <= b):
            raise ValueError("era spans must be ordered and must not overlap")
    months_all = frame_months(product)
    timed = np.isfinite(product.frame_time)
    level = product.level_db
    eras: list[Era] = []
    for index, (a, b, evidence) in enumerate(spans):
        months = [m for m in sorted(record) if a <= m <= b]
        if not months:
            raise ValueError(f"era {blocks.month_label(a)}..{blocks.month_label(b)} holds no populated month")
        first_populated, last = months[0], months[-1]
        first = first_populated
        if a < first_populated:
            # a dated start inside a data gap: allowed only over months that hold no frame of the product,
            # so the era's frames never depend on where inside the gap the date falls
            lead = (months_all >= a) & (months_all < first_populated)
            if lead.any():
                raise ValueError(f"era {blocks.month_label(a)}..{blocks.month_label(b)}: {int(lead.sum())} frames lie "
                                 "between the dated first month and the first populated month")
            first = a
        prev_last = eras[-1].last_month if eras else None
        uncertainty = 0 if prev_last is None else first_populated - prev_last - 1
        between = 0 if prev_last is None else sum(1 for m in record if prev_last < m < first_populated)
        located = [(m, record[m].peak_offset_bins) for m in months
                   if raw[m] == PROXY_HIGH and math.isfinite(record[m].peak_offset_bins)]
        peaks = [v for _, v in located]
        drift = float("nan")
        if len(located) >= 3:
            xs = np.array([m for m, _ in located], dtype=float)
            drift = float(np.polyfit(xs - xs.mean(), np.array(peaks, dtype=float), 1)[0])
        here = mask & (months_all >= first) & (months_all <= last)
        levels = level[here]
        levels = levels[np.isfinite(levels)]
        median = float(np.median(levels)) if levels.size else float("nan")
        eras.append(Era(index, first, last, state_of(median, config), evidence, uncertainty, uncertainty - between,
                        between, tuple(months), float(np.median(peaks)) if peaks else float("nan"), "",
                        units=int(np.unique(product.frame_unit_index[here]).size), frames=int(here.sum()),
                        frames_without_time=int((here & ~timed).sum()), level_median_db=median,
                        peak_drift_bins_per_month=drift,
                        peak_range_bins=float(max(peaks) - min(peaks)) if peaks else float("nan"),
                        peak_months=len(peaks)))
    covered = {m for e in eras for m in e.months}
    zone = tuple(m for m in sorted(record) if m not in covered and eras[0].first_month < m < eras[-1].last_month)
    lag = table.campaign_last_month - eras[-1].last_month
    return dataclasses.replace(
        table, eras=tuple(eras), current_index=len(eras) - 1,
        stale_latest=lag > config.stale_grace_months, stale_lag_months=lag,
        transition_zone_months=zone, indeterminate="", sensitivity=(),
        boundary_source=source, rule_current_era=_span_label(table.current_era))


def campaign_last_populated_month(products: Sequence[Product], config: EraConfig = DEFAULT_CONFIG,
                                  masks: Sequence[np.ndarray] | None = None) -> int:
    """The last populated month over every product (the campaign snapshot); -1 if none."""
    last = -1
    for i, product in enumerate(products):
        mask = product.selected if masks is None else masks[i]
        months = blocks.month_support(product.frame_time, product.frame_unit_index, mask,
                                      min_frames=config.min_frames, min_units=config.min_units,
                                      min_days=config.min_days)
        if months:
            last = max(last, months[-1].month)
    return last


# ------------------------------------------------------------------ writers
def _labels(months) -> str:
    return ";".join(blocks.month_label(m) for m in months)


def era_rows(table: EraTable) -> list[dict]:
    rows = []
    for era in table.eras:
        rows.append({
            "channel": table.channel, "freq_id": table.freq_id, "era": era.index + 1, "n_eras": len(table.eras),
            "first_month": era.first_label, "last_month": era.last_label, "state": era.state,
            "evidence": era.evidence, "record_agreement": era.record_agreement,
            "boundary_uncertainty_months": era.boundary_uncertainty_months,
            "boundary_gap_months": era.boundary_gap_months, "boundary_ambiguous_months": era.boundary_ambiguous_months,
            "units": era.units, "frames": era.frames, "frames_without_time": era.frames_without_time,
            "populated_months": era.populated_months, "months_spanned": era.months_spanned,
            "coverage": float(era.coverage), "level_median_db": float(era.level_median_db),
            "peak_offset_bins": float(era.peak_offset_bins), "peak_drift_bins_per_month": float(era.peak_drift_bins_per_month),
            "peak_range_bins": float(era.peak_range_bins), "peak_months": era.peak_months,
            "is_current": era.index == table.current_index,
            "stale_latest": table.stale_latest, "fallback": table.fallback, "boundary_source": table.boundary_source,
            "config_version": table.config.version, "config_digest": table.config.digest,
        })
    return rows


def channel_row(table: EraTable) -> dict:
    current = table.current_era
    by_parameter = {}
    for o in table.sensitivity:
        by_parameter.setdefault(o.parameter, []).append(o)
    units = by_parameter.get("min_units", [])
    thresholds = by_parameter.get("thresholds_db", [])
    return {
        "channel": table.channel, "freq_id": table.freq_id, "n_eras": len(table.eras),
        "current_era": table.current_index + 1 if current else 0,
        "current_first_month": current.first_label if current else "",
        "current_last_month": current.last_label if current else "",
        "current_state": current.state if current else "", "current_evidence": current.evidence if current else "",
        "current_boundary_uncertainty_months": current.boundary_uncertainty_months if current else "",
        "current_units": current.units if current else 0, "current_frames": current.frames if current else 0,
        "stale_latest": table.stale_latest, "stale_lag_months": table.stale_lag_months,
        "stale_reference": table.stale_reference,
        "campaign_last_month": blocks.month_label(table.campaign_last_month) if table.campaign_last_month >= 0 else "",
        "fallback": table.fallback, "indeterminate": table.indeterminate,
        "populated_months": table.populated_months, "ambiguous_months": table.ambiguous_months,
        "ambiguous_fraction": float(table.ambiguous_fraction),
        "transition_zone_months": _labels(table.transition_zone_months),
        "state_excursions": ";".join(e.label for e in table.excursions if e.kind == "state"),
        "station_excursions": ";".join(e.label for e in table.excursions if e.kind == "station"),
        "unmatched_station_records": ";".join(table.unmatched_station_records),
        "instrument_change_months": _labels(table.instrument_change_months),
        "instrument_excursions": ";".join(e.label for e in table.excursions if e.kind == "instrument"),
        "unconfirmed_instrument_change_last_month": bool(table.months) and any(
            e.kind == "instrument" and e.month == table.months[-1].month for e in table.excursions),
        "current_peak_drift_bins_per_month": float(current.peak_drift_bins_per_month) if current else float("nan"),
        "current_peak_range_bins": float(current.peak_range_bins) if current else float("nan"),
        "software_tags": table.software_tags, "software_tag_changes": table.software_tag_changes,
        "frames_selected": table.frames_selected, "frames_without_time": table.frames_without_time,
        "peak_cohort_fallback_months": _labels(table.peak_cohort_fallback_months),
        "sensitivity_units": ";".join(f"{o.value}:{o.span}" for o in units),
        "sensitivity_units_moves": ";".join(o.value for o in units if o.moves),
        "sensitivity_thresholds": ";".join(f"{o.value}:{o.span}" for o in thresholds),
        "sensitivity_thresholds_moves": ";".join(o.value for o in thresholds if o.moves),
        "boundary_source": table.boundary_source, "rule_current_era": table.rule_current_era or _span_label(current),
        "config_version": table.config.version, "config_digest": table.config.digest,
    }


def _write_csv(rows: Sequence[dict], path: Path | str, columns: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {}
            for name in columns:
                value = row.get(name, "")
                if isinstance(value, float):
                    out[name] = "" if not math.isfinite(value) else repr(value)
                else:
                    out[name] = value
            writer.writerow(out)
    return path


def write_eras_csv(tables: Sequence[EraTable], path: Path | str) -> Path:
    """``tables/eras.csv``: one row per era, channels in the given order."""
    rows = [row for table in tables for row in era_rows(table)]
    return _write_csv(rows, path, ERA_COLUMNS)


def write_channels_csv(tables: Sequence[EraTable], path: Path | str) -> Path:
    """``tables/eras_channels.csv``: one row per channel with the current era and the sensitivity."""
    return _write_csv([channel_row(t) for t in tables], path, CHANNEL_COLUMNS)


def _finite_or_none(value: float):
    return float(value) if math.isfinite(value) else None


def era_table_dict(table: EraTable) -> dict:
    """Everything in the table, JSON-ready (months as labels, NaN as null)."""
    era_months = {m: e.index + 1 for e in table.eras for m in e.months}
    return {
        "channel": table.channel, "freq_id": table.freq_id,
        "config": dataclasses.asdict(table.config), "config_digest": table.config.digest,
        "current_era": table.current_index + 1 if table.current_era else 0,
        "stale_latest": table.stale_latest, "stale_lag_months": table.stale_lag_months,
        "stale_reference": table.stale_reference,
        "campaign_last_month": blocks.month_label(table.campaign_last_month) if table.campaign_last_month >= 0 else None,
        "fallback": table.fallback, "indeterminate": table.indeterminate,
        "populated_months": table.populated_months, "ambiguous_months": table.ambiguous_months,
        "ambiguous_fraction": _finite_or_none(table.ambiguous_fraction),
        "transition_zone_months": [blocks.month_label(m) for m in table.transition_zone_months],
        "excursions": [{"month": blocks.month_label(e.month), "kind": e.kind, "detail": e.detail} for e in table.excursions],
        "instrument_change_months": [blocks.month_label(m) for m in table.instrument_change_months],
        "software_tags": table.software_tags, "software_tag_changes": table.software_tag_changes,
        "frames_selected": table.frames_selected, "frames_without_time": table.frames_without_time,
        "peak_cohort_fallback_months": [blocks.month_label(m) for m in table.peak_cohort_fallback_months],
        "unmatched_station_records": list(table.unmatched_station_records),
        "boundary_source": table.boundary_source,
        "rule_current_era": table.rule_current_era or _span_label(table.current_era),
        "eras": [{
            "era": e.index + 1, "first_month": e.first_label, "last_month": e.last_label, "state": e.state,
            "evidence": e.evidence, "record_agreement": e.record_agreement,
            "boundary_uncertainty_months": e.boundary_uncertainty_months,
            "boundary_gap_months": e.boundary_gap_months, "boundary_ambiguous_months": e.boundary_ambiguous_months,
            "units": e.units, "frames": e.frames, "frames_without_time": e.frames_without_time,
            "populated_months": e.populated_months, "months_spanned": e.months_spanned,
            "coverage": _finite_or_none(e.coverage), "level_median_db": _finite_or_none(e.level_median_db),
            "peak_offset_bins": _finite_or_none(e.peak_offset_bins),
            "months": [blocks.month_label(m) for m in e.months],
        } for e in table.eras],
        "sensitivity": [{"parameter": o.parameter, "value": o.value, "current_era": o.span,
                         "n_eras": o.n_eras, "moves": o.moves} for o in table.sensitivity],
        "months": [{
            "month": m.label, "frames": m.frames, "units": m.units, "days": m.days, "populated": m.populated,
            "level_db": _finite_or_none(m.level_db), "state": m.state,
            "resolved": table.resolved.get(m.month, "unpopulated"),
            "era": era_months.get(m.month, "zone" if m.month in table.transition_zone_months else None),
            "peak_offset_bins": _finite_or_none(m.peak_offset_bins), "peak_cohort": m.peak_cohort,
            "peak_frames": m.peak_frames, "input_maps": list(m.input_maps), "software_tags": len(m.software_tags),
        } for m in table.months],
    }


def write_era_json(table: EraTable, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(era_table_dict(table), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path
