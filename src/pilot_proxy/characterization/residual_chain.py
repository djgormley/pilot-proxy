"""The residual chain of one band: correlation time, coherence gain, and the
per-frame residual that turns a kept-frame shelf into ``r_sys``.

eq:tolerance:chain: ``p_kept = 10^(S_kept / 10)``,
``r_proxy = p_kept G``, ``G = sum_k phi_k n_coh,k``,
``n_coh = min(tau_c, T_cap) / T_frame``, with ``tau_c`` the measured value or
its upper bound where it is usable and the cap where it is refused. Under the
unity-transfer closure ``r_var = r_sys = r_proxy``.

The integration model (``projects/<p>/integration_model.yaml``,
:class:`~pilot_proxy.config.integration_model.IntegrationModel`) declares how
the science analysis integrates a frame residual: the frame length, the
coherence cap (one sidereal day), and ``variance_split``:

- ``off`` (the default): all surviving shelf power is booked at one
  coherence time on every band, ``((1.0, n_coh(tau)),)``. No
  post-processing credit is taken.
- ``booked_when_tau_usable``: where the correlation time is usable, the
  day / acquisition / frame variance split of
  :func:`~pilot_proxy.detectors.narrowband_marker.shelf.shelf_statistics` is
  booked: intra-day power at ``n_coh(tau)`` and sub-acquisition power at 1;
  a refused correlation time books all power at the cap
  (:func:`surviving_components`). This is a ground-filter credit; it is kept
  as an option because it reproduces the author-dated-eras release.

The published components (``share:n_coh`` pairs) let a science side with a
different integration model recompute G. ``chain_gain_status`` says how G was
obtained: ``measured`` (tau_c measured), ``bounded`` (tau_c bounded above) or
``cap`` (tau_c refused: G is the cap, and r_sys an upper bound).

Correlation time. The estimator is a noise-corrected same-sidereal-day
structure function of the acquisition-mean shelf power, read at the (1 - 1/e)
crossing, with a day-block bootstrap for the interval
(:func:`correlation_time`).

Population. The shelf and correlation functions select the transmitter-on
population by calendar month strings (``off_through`` / ``off_from``) from the
product's ``valid`` frames. The chain is evaluated on the band's current era
(the previous on era where the current era is off), so :func:`chain_on_frames`
restricts the population to a frame mask by writing a temporary copy of the
product whose ``valid`` flag is cleared outside the mask and running the
unchanged functions on it; the population is recorded on the result. The
archive-wide form :func:`chain` remains for comparison. The chain's own floor
term (the off-epoch percentile inside the restricted population, usually
absent) is not the selector's floor; that is the null section's.

Per-frame residual (:func:`frame_residuals`, the ``shelf-or-floor linear``
convention): ``max(10^(shelf_db / 10), floor)`` on frames with a shelf
estimate, the floor otherwise, times G; ``r_sys`` is its kept-frame mean.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from pilot_proxy.config.integration_model import IntegrationModel
from pilot_proxy.config.project import default_project
from pilot_proxy.detectors.narrowband_marker.shelf import (
    DEFAULT_TRIM_PERCENTILE,
    ShelfStatistics,
    _nested_split,
    _on_epoch,
    shelf_statistics,
)
from pilot_proxy.products.npzio import load_npz
from pilot_proxy.products.reader import Product

_PROJECT = default_project()
_REGISTER = _PROJECT.register
DEFAULT_MODEL = _PROJECT.integration_model
FRAME_SECONDS = DEFAULT_MODEL.frame_seconds
# The cap on the booked correlation time: one sidereal day. Anything
# correlated for longer is m = 0 within each day and is removed by the
# common-mode filter, so it never reaches the coherence term.
MAX_TAU_C_SECONDS = DEFAULT_MODEL.coherence_cap_seconds
VARIANCE_SPLIT_BOOKED = "booked_when_tau_usable"

# These reproduce the released screen. Their evidence state and sensitivity
# values live in selection_policy; none is an unnamed quality threshold.
STRUCTURE_LAG_EDGES = np.array(
    _REGISTER.value("correlation.lag_edges_seconds"), dtype=float)
PLATEAU_LAG_SECONDS = float(
    _REGISTER.value("correlation.plateau_start_seconds"))
TRIM_PROBES = tuple(float(value) for value in
                    _REGISTER.value("correlation.trim_probes"))
MIN_STRUCTURE_PAIRS_PER_BIN = int(
    _REGISTER.value("correlation.minimum_pairs_per_lag_bin"))
MIN_POPULATED_LAG_BINS = int(
    _REGISTER.value("correlation.minimum_populated_lag_bins"))
STRUCTURE_CROSSING_FRACTION = float(
    _REGISTER.value("correlation.crossing_fraction"))
MIN_CORRELATION_FRAMES = int(
    _REGISTER.value("correlation.minimum_selected_frames"))
MIN_CORRELATION_DAYS = int(
    _REGISTER.value("correlation.minimum_sidereal_days"))
MIN_CORRELATION_PAIRS = int(
    _REGISTER.value("correlation.minimum_same_day_pairs"))
MAX_TRIM_SPREAD = float(
    _REGISTER.value("correlation.maximum_trim_spread"))
CORRELATION_BOOTSTRAP_REPLICATES = int(
    _REGISTER.value("correlation.bootstrap_replicates"))
CORRELATION_BOOTSTRAP_SEED = int(
    _REGISTER.value("correlation.bootstrap_seed"))
CORRELATION_BOOTSTRAP_PERCENTILES = tuple(
    float(value) for value in
    _REGISTER.value("correlation.bootstrap_interval_percentiles"))
MIN_BOOTSTRAP_SUCCESSES = int(
    _REGISTER.value("correlation.minimum_bootstrap_successes"))
MIN_BOOTSTRAP_SUCCESS_FRACTION = float(
    _REGISTER.value("correlation.minimum_bootstrap_success_fraction"))


@dataclass
class CorrelationTime:
    """Intra-day correlation time of the shelf, or a refusal with a reason.

    ``quality`` is one of:

    * ``'measured'``: every gate passed; ``tau_c`` is a measurement with
      a bootstrap interval.
    * ``'bounded_above'``: the shelf is stationary but already decorrelated
      at the shortest lag the acquisition cadence resolves. ``tau_c`` is that
      lag, and it is an *upper* bound, so the budget built from it is a bound
      in the favorable direction rather than a measurement.
    * ``'refused'``: the shelf is episodic and admits no stationary
      timescale at all; ``tau_c`` is nan.

    Only the last is a refusal, and it is not a failure of the channel; it is
    the estimator declining to put a number on a process that does not admit
    one. See :func:`correlation_time`.
    """
    channel: int
    tau_c: float                 # s; nan when refused
    tau_lo: float                # 16th percentile, day-block bootstrap
    tau_hi: float                # 84th percentile
    plateau_fraction: float      # intra-day variance, share of DC power
    n_days: int
    n_pairs: int
    trim_spread: float           # max/min tau_c across TRIM_PROBES
    surviving_spread: float      # max/min (intra + fast) across TRIM_PROBES
    quality: str                 # 'measured' | 'refused'
    reason: str = ""

    @property
    def is_measured(self) -> bool:
        return self.quality == "measured"

    @property
    def is_usable(self) -> bool:
        """Measured or bounded above; either way the budget can use it."""
        return self.quality in ("measured", "bounded_above")

    @property
    def tau_for_budget(self) -> float:
        """The measured value or upper bound; the cap only when refused."""
        return self.tau_c if self.is_usable else MAX_TAU_C_SECONDS

    def summary(self) -> str:
        if self.quality == "refused":
            return (f"ch{self.channel:>3d}  tau_c REFUSED: {self.reason}\n"
                    f"    falling back to the {MAX_TAU_C_SECONDS / 3600:.1f} h cap "
                    f"(bound rather than measurement)")
        if self.quality == "bounded_above":
            return (
                f"ch{self.channel:>3d}  tau_c <= {self.tau_c / 60:.0f} min "
                f"(upper bound): {self.reason}\n"
                f"    plateau {100 * self.plateau_fraction:.4f}% of DC power, "
                f"{self.n_pairs} same-day pairs over {self.n_days} sidereal days\n"
                f"    stability across trim: tau x{self.trim_spread:.2f}, "
                f"surviving x{self.surviving_spread:.2f}")
        return (
            f"ch{self.channel:>3d}  tau_c = {self.tau_c / 60:.0f} min "
            f"[{self.tau_lo / 60:.0f}-{self.tau_hi / 60:.0f} at 68%]\n"
            f"    plateau {100 * self.plateau_fraction:.4f}% of DC power, "
            f"{self.n_pairs} same-day pairs over {self.n_days} sidereal days\n"
            f"    stability across trim: tau x{self.trim_spread:.2f}, "
            f"surviving x{self.surviving_spread:.2f}")


def _same_day_structure(groups, v_fast_abs, edges=STRUCTURE_LAG_EDGES,
                        days_subset=None):
    """Noise-corrected structure function D(dt) over same-sidereal-day pairs.

    D(dt) = <[x(t+dt) - x(t)]^2> / 2 rises from 0 to the intra-day variance as
    dt passes the correlation time. Each unit mean carries estimation noise
    V_fast / n_frames, which inflates every squared difference by a known
    amount; subtracting it is what keeps sparsely-sampled acquisitions from
    faking a short correlation time.

    Pairs are restricted to a single sidereal day so inter-day drift (which
    is m = 0 and already priced as ground-filter suppression) cannot leak in.
    """
    unit_means, n_frames, unit_t, unit_days, _ = groups
    lags, sq = [], []
    for dd in (np.unique(unit_days) if days_subset is None else days_subset):
        idx = np.flatnonzero(unit_days == dd)
        if idx.size < 2:
            continue
        tt, xx, nn = unit_t[idx], unit_means[idx], n_frames[idx]
        i, j = np.triu_indices(idx.size, 1)
        lags.append(np.abs(tt[j] - tt[i]))
        sq.append((xx[j] - xx[i]) ** 2 - v_fast_abs * (1.0 / nn[j] + 1.0 / nn[i]))
    if not lags:
        return np.array([]), np.array([]), np.array([]), np.nan, 0
    lags = np.concatenate(lags)
    D = 0.5 * np.concatenate(sq)
    plateau = float(D[lags > PLATEAU_LAG_SECONDS].mean()) \
        if (lags > PLATEAU_LAG_SECONDS).any() else np.nan

    centers, values, counts = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lags >= a) & (lags < b)
        if m.sum() >= MIN_STRUCTURE_PAIRS_PER_BIN:
            centers.append(0.5 * (a + b))
            values.append(float(D[m].mean()))
            counts.append(int(m.sum()))
    return (np.array(centers), np.array(values), np.array(counts),
            plateau, int(lags.size))


def _tau_from_structure(centers, values, plateau):
    """Lag at which D reaches (1 - 1/e) of its plateau, linearly interpolated."""
    if (not np.isfinite(plateau) or plateau <= 0
            or centers.size < MIN_POPULATED_LAG_BINS):
        return np.nan
    target = STRUCTURE_CROSSING_FRACTION * plateau
    if values.max() < target:
        return np.nan
    k = int(np.argmax(values >= target))
    if k == 0:
        return float(centers[0])
    x0, x1, y0, y1 = centers[k - 1], centers[k], values[k - 1], values[k]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def _measure_at_trim(d, off_through, trim, off_from=None):
    on, shelf, unit, t0 = _on_epoch(d, off_through, trim, off_from=off_from)
    if on.sum() < MIN_CORRELATION_FRAMES:
        return None
    split = _nested_split(10.0 ** (shelf[on] / 10.0), unit[on], t0)
    if split[-1] is None:
        return None
    _, _, intraday, fast, _, n_days, v_fast_abs, groups = split
    centers, values, _, plateau, n_pairs = _same_day_structure(groups, v_fast_abs)
    grand = groups[4]
    return dict(tau=_tau_from_structure(centers, values, plateau),
                surviving=intraday + fast, n_days=n_days, n_pairs=n_pairs,
                plateau_frac=(plateau / grand ** 2 if grand else np.nan),
                groups=groups, v_fast_abs=v_fast_abs)


def correlation_time(npz_path: str | Path, off_through: str | None = None,
                     trim_percentile: float = DEFAULT_TRIM_PERCENTILE,
                     trim_probes=TRIM_PROBES,
                     max_trim_spread: float = MAX_TRIM_SPREAD,
                     min_days: int = MIN_CORRELATION_DAYS,
                     min_pairs: int = MIN_CORRELATION_PAIRS,
                     n_boot: int = CORRELATION_BOOTSTRAP_REPLICATES,
                     seed: int = CORRELATION_BOOTSTRAP_SEED,
                     off_from: str | None = None) -> CorrelationTime:
    """Measure the intra-day correlation time, or refuse and say why.

    The estimator is a noise-corrected same-sidereal-day structure function of
    the acquisition-mean shelf power, read at the (1 - 1/e) crossing, with a
    day-block bootstrap for the interval. Resampling has to be by whole day
    with each day's time ordering intact; shuffling acquisitions within a day
    destroys exactly the structure being measured and drives tau_c to the
    shortest lag bin.

    Four gates must pass, and the last two are the ones that matter:

    * enough same-day pairs and enough days;
    * a positive plateau the structure function actually reaches, at a lag the
      acquisition cadence can resolve; a crossing in the first bin means
      tau_c is *below* the shortest measurable lag, which is a bound rather than a
      measurement;
    * **tau_c stable across trim level**; and
    * **the surviving power fraction stable across trim level**.

    The stability gates are what separate a stationary shelf from an episodic
    one. On ch35 the answer moves by a factor 1.2 as the trim runs 75-95%; on
    ch34 and ch36, whose transmitters burst rather than sit on, it moves by 10x
    and 15x, because there is no stationary variance to decompose and the
    answer is entirely an artefact of where the tail was cut. Those channels
    get a refusal and the conservative cap rather than a number.
    """
    d = load_npz(npz_path)
    channel = int(d["physical_channel"][0])

    probes = {}
    for tp in trim_probes:
        r = _measure_at_trim(d, off_through, tp, off_from=off_from)
        if r is not None and np.isfinite(r["tau"]) and r["surviving"] > 0:
            probes[tp] = r

    def refuse(reason, ts=np.nan, ss=np.nan, nd=0, npair=0, pf=np.nan):
        return CorrelationTime(channel, np.nan, np.nan, np.nan, pf, nd, npair,
                               ts, ss, "refused", reason)

    if len(probes) < len(trim_probes):
        return refuse(f"only {len(probes)}/{len(trim_probes)} trim probes "
                      f"yielded a finite estimate")

    taus = np.array([r["tau"] for r in probes.values()])
    survs = np.array([r["surviving"] for r in probes.values()])
    trim_spread = float(taus.max() / taus.min())
    surv_spread = float(survs.max() / survs.min())

    main = _measure_at_trim(d, off_through, trim_percentile, off_from=off_from)
    if main is None or not np.isfinite(main["tau"]):
        return refuse("no finite estimate at the requested trim",
                      trim_spread, surv_spread)
    if main["n_days"] < min_days:
        return refuse(f"only {main['n_days']} sidereal days with a same-day "
                      f"pair (need {min_days})", trim_spread, surv_spread,
                      main["n_days"], main["n_pairs"], main["plateau_frac"])
    if main["n_pairs"] < min_pairs:
        return refuse(f"only {main['n_pairs']} same-day pairs (need {min_pairs})",
                      trim_spread, surv_spread, main["n_days"], main["n_pairs"],
                      main["plateau_frac"])
    # Stationarity is checked BEFORE resolution, because the two failure modes
    # point opposite ways. An episodic shelf gives a meaningless answer at any
    # lag. A stationary shelf whose crossing lands in the first bin has a
    # *short* correlation time (an upper bound in the favorable direction,
    # worth ~24 dB against the cap) and must not be thrown away as if it were
    # the same kind of failure.
    if trim_spread > max_trim_spread:
        return refuse(f"tau_c moves x{trim_spread:.1f} across trim probes "
                      f"(max {max_trim_spread:g}); the shelf is episodic, "
                      f"not stationary", trim_spread, surv_spread,
                      main["n_days"], main["n_pairs"], main["plateau_frac"])
    if surv_spread > max_trim_spread:
        return refuse(f"surviving power fraction moves x{surv_spread:.1f} "
                      f"across trim probes (max {max_trim_spread:g}); the "
                      f"variance split is set by where the tail was cut",
                      trim_spread, surv_spread, main["n_days"], main["n_pairs"],
                      main["plateau_frac"])
    if main["tau"] <= STRUCTURE_LAG_EDGES[1]:
        # Stationary, and already decorrelated at the shortest lag the
        # acquisition cadence resolves: tau_c <= that lag.
        return CorrelationTime(
            channel=channel, tau_c=float(STRUCTURE_LAG_EDGES[1]),
            tau_lo=np.nan, tau_hi=float(STRUCTURE_LAG_EDGES[1]),
            plateau_fraction=float(main["plateau_frac"]),
            n_days=int(main["n_days"]), n_pairs=int(main["n_pairs"]),
            trim_spread=trim_spread, surviving_spread=surv_spread,
            quality="bounded_above",
            reason=f"decorrelated by {STRUCTURE_LAG_EDGES[1] / 60:.0f} min, the "
                   f"shortest resolvable lag; tau_c is an upper bound")

    groups, v_fast_abs = main["groups"], main["v_fast_abs"]
    days = np.unique(groups[3])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(days, size=days.size, replace=True)
        c, v, _, pl, _ = _same_day_structure(groups, v_fast_abs, days_subset=pick)
        t = _tau_from_structure(c, v, pl)
        if np.isfinite(t):
            boots.append(t)
    minimum_successes = max(
        MIN_BOOTSTRAP_SUCCESSES,
        int(np.ceil(MIN_BOOTSTRAP_SUCCESS_FRACTION * n_boot)))
    if len(boots) < minimum_successes:
        return refuse("bootstrap did not converge", trim_spread, surv_spread,
                      main["n_days"], main["n_pairs"], main["plateau_frac"])
    boots = np.array(boots)

    return CorrelationTime(
        channel=channel, tau_c=float(main["tau"]),
        tau_lo=float(np.percentile(
            boots, CORRELATION_BOOTSTRAP_PERCENTILES[0])),
        tau_hi=float(np.percentile(
            boots, CORRELATION_BOOTSTRAP_PERCENTILES[1])),
        plateau_fraction=float(main["plateau_frac"]),
        n_days=int(main["n_days"]), n_pairs=int(main["n_pairs"]),
        trim_spread=trim_spread, surviving_spread=surv_spread,
        quality="measured")



# Quantile probes for the null scale, and the standard-normal deviates they
# correspond to *in the full null*; the kept sample is its lower half, so the
# p-th percentile of the kept frames is the (p/2)-th percentile of the null.
NULL_SCALE_PROBES = tuple(
    (float(percentile), float(deviate))
    for percentile, deviate in _REGISTER.value("floor.null_scale_probes"))


def null_scale(f_kept: np.ndarray, mu0: float) -> tuple[float, float]:
    """Robust scale of the decision statistic under the null, left side only.

    Returns ``(sigma, spread)``, where ``spread`` is the ratio of the largest
    to the smallest of the individual quantile estimates. A spread near one
    means the left tail is Gaussian and well sampled; a large spread means the
    channel is masked so heavily that the kept frames cannot characterise the
    null, and the scale should be treated as indicative rather than measured.
    """
    if np.asarray(f_kept).size == 0:
        return float("nan"), float("nan")
    ests = [(mu0 - float(np.percentile(f_kept, p))) / z
            for p, z in NULL_SCALE_PROBES]
    ests = [e for e in ests if e > 0.0]
    if not ests:
        return float("nan"), float("nan")
    return float(np.median(ests)), float(max(ests) / min(ests))


def coherent_frames(tau_c_seconds: float,
                    frame_seconds: float = FRAME_SECONDS,
                    cap_seconds: float = MAX_TAU_C_SECONDS) -> float:
    """Rectangular coherent-block variance factor used by the screen.

    ``tau_c`` at the frame scale gives 1; an hour gives about 8.6e4. This is
    exact for the declared block model. It must not be relabeled as an
    integrated-autocorrelation estimate for an arbitrary process.
    """
    if frame_seconds <= 0:
        raise ValueError("frame_seconds must be positive")
    tau = min(float(tau_c_seconds), float(cap_seconds))
    return max(1.0, tau / float(frame_seconds))


def surviving_components(stats: ShelfStatistics, corr: CorrelationTime,
                         tau_intraday: float | None = None, *,
                         model: IntegrationModel = DEFAULT_MODEL) -> tuple:
    """The ``(power_fraction, n_coh)`` components the budget's own discipline
    assigns to the surviving shelf power.

    A usable correlation time books the measured variance split: intra-day
    power at ``n_coh(tau)``, sub-acquisition power at 1. A refusal makes the
    split itself unmeasurable --- the non-stationarity that defeats the
    timescale estimator defeats the variance decomposition too, since both
    are moments of the same process --- so the fallback takes no
    ground-filter credit: all shelf power surviving, at the sidereal-day cap.
    ``tau_intraday`` overrides the timescale only; it cannot restore credit
    a refusal removed. Passing a sub-cap ``tau_intraday`` on a refused
    channel is therefore a *what-if* -- all power at the assumed timescale
    -- and any result built from one must be labelled as such. This is the
    single home of that rule. The frame length and the cap are the
    integration model's.
    """
    if corr.is_usable:
        tau = corr.tau_for_budget if tau_intraday is None else tau_intraday
        return ((float(stats.intraday_fraction),
                 coherent_frames(tau, model.frame_seconds, model.coherence_cap_seconds)),
                (float(stats.fast_fraction), 1.0))
    tau = model.coherence_cap_seconds if tau_intraday is None else tau_intraday
    return ((1.0, coherent_frames(tau, model.frame_seconds, model.coherence_cap_seconds)),)


def booked_components(corr: CorrelationTime, model: IntegrationModel = DEFAULT_MODEL) -> tuple[tuple[float, float], ...]:
    """All surviving shelf power at one coherence time: ``((1.0, n_coh(tau)),)``.

    ``tau`` is the measured correlation time or its upper bound where it is
    usable, the model's cap where it is refused. The variance split is not
    booked, so no ground-filter credit is taken. On a refused band this is
    what :func:`surviving_components` books.
    """
    tau = corr.tau_c if corr.is_usable else model.coherence_cap_seconds
    return ((1.0, float(coherent_frames(tau, model.frame_seconds, model.coherence_cap_seconds))),)


def chain_components(stats: ShelfStatistics, corr: CorrelationTime,
                     model: IntegrationModel = DEFAULT_MODEL) -> tuple[tuple[float, float], ...]:
    """The ``(share, n_coh)`` pairs the integration model books."""
    if model.variance_split == VARIANCE_SPLIT_BOOKED:
        return tuple((float(share), float(n_coh)) for share, n_coh in surviving_components(stats, corr, model=model))
    return booked_components(corr, model)


GAIN_STATUS = {"measured": "measured", "bounded_above": "bounded", "refused": "cap"}


@dataclass(frozen=True)
class ChainResult:
    channel: int
    freq_id: int
    population: str                    # how the transmitter-on population was chosen
    n_valid: int
    n_kept: int                        # kept at the survey flag (F <= mu_0)
    n_off_frames: int                  # the null population behind the floor
    on_shelf_db: float                 # on-air shelf level
    floor_db: float                    # kept-frame / off-era floor (NaN when none)
    floor_percentile: float
    tau_c_seconds: float
    tau_c_low: float
    tau_c_high: float
    tau_quality: str                   # 'measured' | 'bounded_above' | 'refused'
    tau_reason: str
    components: tuple[tuple[float, float], ...]   # (share, n_coh) pairs as booked
    gain: float                        # G = sum share * n_coh
    variance_split: str = "off"        # the integration model's rule
    intraday_share: float = math.nan   # phi_intra (reported where the split is booked)
    fast_share: float = math.nan       # phi_fast
    ground_filter_db: float = math.nan # removed share in dB (description only when tau_c refused)
    frame_seconds: float = FRAME_SECONDS             # the model's frame length
    coherence_cap_seconds: float = MAX_TAU_C_SECONDS  # the model's cap

    @property
    def n_coh_intraday(self) -> float:
        """``min(tau_c, T_cap) / T_frame`` at the booked tau; written only where the split is booked."""
        tau = self.tau_c_seconds if math.isfinite(self.tau_c_seconds) else self.coherence_cap_seconds
        return float(coherent_frames(tau, self.frame_seconds, self.coherence_cap_seconds))

    @property
    def tau_c_minutes(self) -> float:
        return self.tau_c_seconds / 60.0 if math.isfinite(self.tau_c_seconds) else math.nan

    @property
    def tau_outcome(self) -> str:
        """The chapter 8 vocabulary: measured interval, one-sided bound, or refused (cap)."""
        return {"measured": "measured", "bounded_above": "bound", "refused": "refused (cap)"}.get(self.tau_quality, self.tau_quality)

    @property
    def gain_status(self) -> str:
        """How G was obtained: ``measured``, ``bounded`` (tau_c an upper bound) or ``cap``."""
        return GAIN_STATUS.get(self.tau_quality, self.tau_quality)

    @property
    def split_booked(self) -> bool:
        return self.variance_split == VARIANCE_SPLIT_BOOKED

    def components_text(self) -> str:
        """The booked components as ``share:n_coh`` pairs (floats as repr), ``;``-joined."""
        return ";".join(f"{float(s)!r}:{float(n)!r}" for s, n in self.components)

    def as_row(self) -> dict:
        row = {
            "channel": self.channel, "freq_id": self.freq_id, "chain_population": self.population,
            "n_valid": self.n_valid, "n_kept_flag": self.n_kept, "n_off_frames": self.n_off_frames,
            "on_shelf_db": self.on_shelf_db, "chain_floor_db": self.floor_db,
        }
        if self.split_booked:
            row.update({"intraday_share": self.intraday_share, "fast_share": self.fast_share,
                        "ground_filter_db": self.ground_filter_db})
        row.update({
            "tau_c_minutes": self.tau_c_minutes, "tau_c_low_minutes": self.tau_c_low / 60.0 if math.isfinite(self.tau_c_low) else math.nan,
            "tau_c_high_minutes": self.tau_c_high / 60.0 if math.isfinite(self.tau_c_high) else math.nan,
            "tau_quality": self.tau_quality, "tau_outcome": self.tau_outcome, "tau_reason": self.tau_reason,
        })
        if self.split_booked:
            row["n_coh_intraday"] = self.n_coh_intraday
        row["chain_gain"] = self.gain
        return row


def chain(product_path: Path | str, model: IntegrationModel = DEFAULT_MODEL, *, off_through: str | None = None,
          off_from: str | None = None) -> ChainResult:
    """Shelf statistics, correlation time and the booked coherence gain of one band, under ``model``."""
    path = str(product_path)
    stats = shelf_statistics(path, off_through=off_through, off_from=off_from)
    corr = correlation_time(path, off_through=off_through, off_from=off_from)
    components = chain_components(stats, corr, model)
    gain = float(sum(share * n_coh for share, n_coh in components))
    tau = float(corr.tau_c) if corr.tau_c is not None else math.nan
    if off_through and off_from:
        population = f"transmitter-on frames outside the off epochs through {off_through} and from {off_from}"
    elif off_through:
        population = f"transmitter-on frames after the off epoch through {off_through}"
    elif off_from:
        population = f"transmitter-on frames before the off epoch from {off_from}"
    else:
        population = "transmitter-on frames of the whole archive (no declared off epoch)"
    split = {}
    if model.variance_split == VARIANCE_SPLIT_BOOKED:
        split = dict(intraday_share=float(stats.intraday_fraction), fast_share=float(stats.fast_fraction),
                     ground_filter_db=float(stats.ground_filter_db))
    return ChainResult(
        channel=int(stats.channel), freq_id=int(stats.freq_id), population=population,
        n_valid=int(stats.n_valid), n_kept=int(stats.n_kept), n_off_frames=int(stats.n_off_frames),
        on_shelf_db=float(stats.on_shelf_db), floor_db=float(stats.floor_db), floor_percentile=float(stats.floor_percentile),
        tau_c_seconds=tau, tau_c_low=float(corr.tau_lo) if corr.tau_lo is not None else math.nan,
        tau_c_high=float(corr.tau_hi) if corr.tau_hi is not None else math.nan,
        tau_quality=str(corr.quality), tau_reason=str(corr.reason or ""),
        components=components, gain=gain, variance_split=model.variance_split, **split,
        frame_seconds=float(model.frame_seconds), coherence_cap_seconds=float(model.coherence_cap_seconds),
    )


LARGE_UNUSED_KEYS = ("psd_frame_db_i16",)


def masked_valid(valid: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """The product's ``valid`` array with frames outside ``frames`` cleared, in the array's own shape and dtype."""
    valid = np.asarray(valid)
    flat = valid.reshape(-1).astype(bool) & np.asarray(frames, dtype=bool).reshape(-1)
    return flat.reshape(valid.shape).astype(valid.dtype)


ZEROED_WHEN_INVALID = ("p_ref_sum_u64", "p_ref_lower_u64", "p_ref_upper_u64", "reject_mask")
NAN_WHEN_INVALID = ("coarse_power_ratio", "normalized_coarse_power_ratio_db", "normalized_pilot_excess", "pilot_excess_db",
                    "estimated_data_shelf_snr_db")


def invalidate_frames(arrays: dict, frames) -> dict:
    """Make the frames outside ``frames`` invalid under the v5 product contract.

    The contract ties the flags together (``valid`` iff ``p_ref_sum != 0``;
    ``reject_mask`` equals the exact decision on valid frames; the derived
    per-frame ratios are NaN where the reference sum is zero), so an
    era-restricted copy clears the reference terms and the flag together and
    blanks the derived fields on the excluded frames.
    """
    keep = np.asarray(frames, dtype=bool).reshape(-1)
    out = dict(arrays)
    out["valid"] = masked_valid(arrays["valid"], keep)
    drop = ~keep
    for key in ZEROED_WHEN_INVALID:
        if key in out:
            a = np.array(out[key], copy=True)
            a.reshape(-1)[drop] = 0
            out[key] = a
    for key in NAN_WHEN_INVALID:
        if key in out:
            a = np.array(out[key], dtype=np.float64, copy=True)
            a.reshape(-1)[drop] = np.nan
            out[key] = a
    return out


def chain_on_frames(product: Product, frames, model: IntegrationModel = DEFAULT_MODEL, *, population: str,
                    off_through: str | None = None, off_from: str | None = None) -> ChainResult:
    """The chain on a frame mask: ``valid`` is cleared outside ``frames`` in a temporary copy of the product."""
    frames = np.asarray(frames, dtype=bool)
    if frames.shape != (product.n_frames,):
        raise ValueError(f"frames must have shape ({product.n_frames},); got {frames.shape}")
    arrays = {}
    with np.load(product.path, allow_pickle=False) as z:
        for key in z.files:
            if key in LARGE_UNUSED_KEYS:
                continue
            arrays[key] = z[key]
    arrays = invalidate_frames(arrays, frames)
    valid = np.asarray(arrays["valid"]).reshape(-1).astype(bool)
    fd, tmp = tempfile.mkstemp(prefix=f"chain_{product.path.stem}_", suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        result = chain(tmp, model, off_through=off_through, off_from=off_from)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return replace(result, population=f"{population}: {int(valid.sum())} valid frames")


def frame_residuals(product: Product, rows: np.ndarray, floor, gain: float = 1.0) -> np.ndarray:
    """Floor-bounded shelf residual per frame row, times the chain gain.

    This is a conditional screening allowance, not a physical lower bound
    or a calibrated confidence limit. A frame with a shelf estimate carries ``max(10^(shelf/10), floor)``; a
    frame without one carries the floor. The floor is the level the mask can
    be held to, so no frame's residual falls below it (a flagged frame whose
    excess is not resolved above the floor is booked at the floor, not below
    an unflagged one). ``gain`` is the chain gain ``G`` of :func:`chain`
    (1.0 gives the frame-stage residual); with it the kept-frame mean is
    ``r_proxy``. ``floor`` is a :class:`~pilot_proxy.characterization.surface.Floor`
    (anything with a ``linear`` level).
    """
    shelf = product.shelf_db[rows]
    finite = np.isfinite(shelf)
    out = np.full(rows.shape, floor.linear, dtype=float)
    out[finite] = np.fmax(10.0 ** (shelf[finite] / 10.0), floor.linear)     # fmax: an undefined floor does not bound
    if not np.isfinite(out).all():
        raise ValueError("systematic residuals need a finite floor for frames without a shelf estimate")
    if not (math.isfinite(gain) and gain > 0.0):
        raise ValueError("the chain gain must be a positive finite number")
    return out * float(gain)


def floor_dominated(product: Product, rows: np.ndarray, floor) -> np.ndarray:
    """Whether the floor sets each frame's residual in :func:`frame_residuals`.

    True for a frame without a shelf estimate (it carries the floor) and for a
    frame whose shelf level is at or below the floor; an undefined floor
    dominates nothing.
    """
    shelf = product.shelf_db[rows]
    finite = np.isfinite(shelf)
    out = ~finite
    out[finite] = float(floor.linear) >= 10.0 ** (shelf[finite] / 10.0)
    return out


__all__ = ["ChainResult", "CorrelationTime", "DEFAULT_MODEL", "FRAME_SECONDS", "GAIN_STATUS", "MAX_TAU_C_SECONDS",
           "booked_components", "chain", "chain_components", "chain_on_frames", "coherent_frames",
           "correlation_time", "floor_dominated", "frame_residuals", "invalidate_frames", "masked_valid", "null_scale",
           "surviving_components"]
