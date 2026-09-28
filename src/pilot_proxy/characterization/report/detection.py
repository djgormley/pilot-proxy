"""The LimeSDR/GNU Radio detection campaign: paired implementation loss and the
detection crossings it is measured on.

Chapter 6 (Section ``sec:detection:synthetic``) asks the bench campaign for
three results at the production geometry ($K=128$, $L=128$, $M=2048$, the fine
designated-set statistic): a four-panel synthetic-verification figure, the
paired implementation-loss table ``tab:detection:loss``, and the
mask-versus-residual frontier ``fig:detection:frontier``. The products on this
machine carry one of the three. This module renders that one and says exactly
which measurement the other two are missing; nothing here is scaled, fitted or
extrapolated to a geometry that was not run.

Inputs
------
``estimator_transfer_2026-08-25`` (40 ``pilot-proxy evaluate-snr`` shards,
pooled in the release's own order by :func:`evaluations.digital_sweep_layout`)
and ``ota_transfer_2026-08-24_ch35`` (one shard, the over-the-air channel-35
broadcast capture with the noise added digitally). Both are read with
:mod:`pilot_proxy.testbench.harness.evaluations`, which owns the column fallbacks. Each trial
row carries the same samples through three arithmetic paths, which is what
makes the comparison paired:

``gpu``        the production CUDA path, exact ``uint64`` power terms;
``cpu_packed`` the fixed-point CPU oracle, packed int4 samples *and* packed
               int4 weights;
``cpu_float``  the full-precision control: unquantized detector rows with ideal
               complex DFT weights (``detector_output.cpu_float_reference``).

The statistic is the run's normalized power ratio $Q$ (``evaluations``'
``trial_records``): $F = 2P_{\\rm target}/(P_{\\rm ref,lower}+P_{\\rm ref,upper})$
divided by its weight-norm null level, so $Q=1$ is the null and $Q-1$ is the
pilot excess. The shelf-SNR axis is the run's ``requested_data_shelf_snr_db``,
which its own ``measured_truth_data_shelf_snr_db`` reproduces to better than
$0.015$ dB.

The table (``tab:detection:loss``)
----------------------------------
One row per (capture, crossing criterion). Sign convention: **packed minus
full-precision**, so a positive shift means the packed crossing sits at higher
shelf SNR and is the less favourable one.

``Crossing``  the criterion. Two are *fixed-threshold*: a threshold $\\tau_p$ is
    set once, per path, at the nominal null false-alarm rate
    :data:`NOMINAL_PFA` on that capture's null population (every trial at or
    below :data:`NULL_CEILING_DB`, where the injected shelf is more than 50 dB
    under the noise), and $P_d(s)$ is the fraction of trials at requested shelf
    SNR $s$ with $Q>\\tau_p$. The third is the detector's own *positive-excess*
    rule, $Q>1$ (``normalized_positive_excess_decision``), which has no
    threshold. No number in this table is an operating point: the campaign
    selects nothing (Section ``sec:detection:synthetic``).
``Trials``   paired trials behind the row, and the null trials the threshold
    was set on.
``Shift``    the paired horizontal shift at the stated $P_d$ level:
    $\\Delta = -[P_d^{\\rm packed}-P_d^{\\rm float}](s^\\star)\\,/\\,({\\rm d}P_d^{\\rm float}/{\\rm d}s)$
    at $s^\\star$, the full-precision crossing, both curves read on the same
    grid interval. Because the two paths are evaluated on the *same* trials,
    the difference of the two $P_d$ curves is what carries the shift; taking it
    this way rather than differencing two separately located crossings removes
    the grid-interval jump that otherwise dominates the interval, and the two
    agree exactly on the full sample.
``Interval`` paired bootstrap, :data:`BOOTSTRAP_SAMPLES` replicates at seed
    :data:`BOOTSTRAP_SEED`, resampling whole trials with replacement inside
    each requested SNR point (so every replicate keeps the campaign's own trial
    allocation) and carrying both paths of a resampled trial together. The
    fixed-threshold rows hold $\\tau_p$ at its full-sample value, which is what
    "fixed threshold" means; the positive-excess row has no threshold to hold.
``Margin``   :data:`MARGIN_DB`, the bound Appendix A's evidence matrix already
    claims from the retained 512-row study. It is not declared in Chapter 6,
    and it is reproduced here rather than invented.
``Verdict``  ``within`` when the interval's upper limit is at or below the
    margin, ``not established`` when it is above it, and ``unresolved`` when the
    capture cannot decide the margin at all --- when one trial changing its
    decision at the crossing's grid point would move the crossing by more than
    the margin. The loss is bounded above by the interval's upper limit, never
    by the point estimate.

All three criteria are rank-based at a matched null rate, which fixes what a
crossing shift can mean: a packed path that only *rescaled* the excess would
move its own threshold with it and shift nothing. What these rows measure is a
change in separation between the null and the signal, not a level offset
between the two paths; the level offset is panel (b)'s per-trial residual.

Two component rows close the table: the frozen transform's own shift of
:data:`TRANSFORM_ONLY_DB` (Section ``sec:impl:fxfft``), listed as a component
and not a bound --- and, since these evaluations form the designated set by
coarse target/reference power summation rather than by ``fxfft256``, a component that this
bench ladder does not exercise; and the same-seed CPU/GPU equality count, the
number of trials whose packed CPU statistic equals the GPU statistic exactly
(``cpu_gpu_abs_diff``).

The figure (``fig_detection_crossings``)
----------------------------------------
(a) $P_d$ against requested shelf SNR at the fixed null $P_{\\rm fa}$, packed
    and full-precision, with the positive-excess rule dashed beside them, on
    the synthetic sweep; the two fixed-threshold crossings are marked. This is
    stub panel (b) of ``fig:detection:mscaling`` at the geometry that was run.
(b) The paired fixed-minus-float statistic residual $10\\log_{10}(Q_{\\rm
    packed}/Q_{\\rm float})$ per trial --- median and 5--95\\% spread against
    shelf SNR, with the transform-only component marked. It is the per-trial
    form of the same comparison and shows where representation matters.
(c) The crossing shifts of the table with their intervals, against zero and the
    margin band. This is stub panel (d).

What these products do not carry
--------------------------------
Named precisely, because the chapter's stub asks for it and the honest answer
is a list of missing measurements rather than an approximation:

* **The spatial axis.** Every trial in both captures was evaluated at
  ``num_input_streams`` $=4$ and ``detector_rows_per_frame`` $=512$ --- the
  superseded 512-row coarse geometry of Appendix B, not $M=2048$. There is no
  $M$-sweep, no independent-capture stack at any other $M$, and no replicated
  capture, so stub panel (a) --- null width and deflection against
  $1/\\sqrt{\\beta M}$ and $\\sqrt{M}$, with the fully correlated bracket --- has
  no input at all. Nothing here is scaled by $\\sqrt{M}$ to stand in for it.
* **The second statistic.** These runs compute one statistic per trial, the
  coarse target/reference power ratio at ``bin_enbw_hz`` $=3051.76$ Hz. There is no
  coarse channel-power counterpart on the same trials, so the fine-versus-coarse
  comparison and stub panel (c)'s predicted-versus-observed gain against
  $5\\log_{10}L\\approx10.5$ dB cannot be formed: the missing measurement is the
  fine statistic, not the bound.
* **The pilot offsets.** ``frequency_offset_hz`` is $0$ on all 10\\,860 trials of
  both captures. The half-fine-bin straddle, $\\pm1$ kHz and $\\pm1.4$ kHz rows
  the stub asks for are dashed, not interpolated.
* **The middle rungs of the ladder.** ``cpu_float`` differs from ``cpu_packed``
  in *both* weight quantization and input quantization at once. The
  quantized-weight float rung and the unquantized-input branch of
  Figure ``fig:detection:paired`` are not in these products, so the table
  measures the whole representation step and cannot decompose it.
* **The frontier.** ``fig:detection:frontier`` needs labelled duty-cycle trials
  with a per-frame injection truth, a rank $\\rho$, a multiplier $\\eta$ and the
  resulting masked fraction. These evaluations carry none of those columns: no
  transmitter duty cycle was run, no frame is labelled on/off, and no mask was
  applied. That figure has no input here and is not attempted.

Numbers: ``ch06.detection.*`` --- the shifts and interval limits keyed by
capture and criterion, the crossings themselves, the thresholds and their null
populations, the trial counts, the margin, the transform-only component and the
CPU/GPU equality count.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pilot_proxy.testbench.harness import evaluations as ev
from .core import DASH, Fragment, Run, booktabs, fmt, fmt_int, tex

NAME = "detection"
LABEL = "tab:detection:loss"
FIGURE = "fig_detection_crossings"

SWEEP_ENV = "PILOT_PROXY_DETECTION_SWEEP"
OTA_ENV = "PILOT_PROXY_DETECTION_OTA"

# The null population: every trial whose injected shelf sits at least this far
# below the noise. At -50 dB the shelf moves the statistic's mean by 1e-5 of the
# null, far under the trial scatter, so these points are H0 for this purpose.
NULL_CEILING_DB = -50.0
NOMINAL_PFA = 0.05          # the finest null rate the smaller capture's null supports
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20_260_907
MARGIN_DB = 0.1             # Appendix A's declared bound for the 512-row study
TRANSFORM_ONLY_DB = -0.0026  # Section sec:impl:fxfft, a component and not a bound
SEARCHED_BINS = 1           # one designated target bin per trial: no |D| correction applies

# (key, printed label, which curve, level)
CRITERIA = (
    ("pd50", r"Fixed threshold, $P_{\rm d}=0.5$", "fixed", 0.5),
    ("pd90", r"Fixed threshold, $P_{\rm d}=0.9$", "fixed", 0.9),
    ("pos90", r"Positive excess, $P_{\rm d}=0.9$", "positive", 0.9),
)
# the campaign offsets the stub asks for and these products do not carry
UNMEASURED_OFFSETS = (
    ("half_bin", r"Pilot offset $+\tfrac{1}{2}$ padded fine bin (5.96 Hz)"),
    ("khz1", r"Pilot offset $\pm 1$ kHz"),
    ("khz14", r"Pilot offset $\pm 1.4$ kHz"),
)


# ------------------------------------------------------------------ locations
def sweep_dir(path: Path | str | None = None) -> Path:
    """Where the 2026-08-25 synthetic sweep is on this machine."""
    if path is None and not os.environ.get(SWEEP_ENV):
        raise ValueError(f"the detection table needs the sweep: pass --sweep or set ${SWEEP_ENV}")
    return Path(path) if path is not None else Path(os.environ[SWEEP_ENV]).expanduser()


def ota_dir(path: Path | str | None = None) -> Path:
    """Where the 2026-08-24 over-the-air channel-35 evaluation is on this machine."""
    if path is None and not os.environ.get(OTA_ENV):
        raise ValueError(f"the detection table needs the over-the-air capture: pass --ota or set ${OTA_ENV}")
    return Path(path) if path is not None else Path(os.environ[OTA_ENV]).expanduser()


# ------------------------------------------------------------------- captures
@dataclass(frozen=True)
class Capture:
    """One campaign capture: its paired per-trial statistics and its geometry."""

    key: str
    label: str
    inputs: tuple[Path, ...]
    snr_db: np.ndarray                       # requested shelf SNR, one entry per paired trial
    statistic: dict                          # prefix -> normalized ratio Q, aligned with snr_db
    offsets_hz: tuple[float, ...]
    input_streams: tuple[int, ...]
    detector_rows: tuple[int, ...]
    cpu_gpu_equal: int
    cpu_gpu_trials: int

    @property
    def trials(self) -> int:
        return int(self.snr_db.size)

    @property
    def points(self) -> np.ndarray:
        return np.array(sorted(set(self.snr_db.tolist())), dtype=float)

    @property
    def null_mask(self) -> np.ndarray:
        return self.snr_db <= NULL_CEILING_DB

    @property
    def null_trials(self) -> int:
        return int(self.null_mask.sum())


def paired_statistics(rows: Sequence[dict]) -> tuple[np.ndarray, dict]:
    """Requested shelf SNR and the normalized ratio of every path, on the trials that carry all three.

    ``evaluations.trial_records`` owns the per-path column fallbacks and returns
    the very row objects it was given, so the three paths are re-aligned here by
    row identity: a trial enters only when every path has a finite statistic for
    it, which is what makes the comparison paired.
    """
    per_row: dict[int, dict] = {}
    for prefix in ev.PREFIXES:
        for row, ratio, _weight in ev.trial_records(rows, prefix):
            per_row.setdefault(id(row), {})[prefix] = float(ratio)
    keep = [row for row in rows if len(per_row.get(id(row), ())) == len(ev.PREFIXES)]
    snr = np.array([float(row[ev.REQUESTED_SNR]) for row in keep], dtype=float)
    statistic = {p: np.array([per_row[id(row)][p] for row in keep], dtype=float) for p in ev.PREFIXES}
    return snr, statistic


def _column(rows: Sequence[dict], name: str) -> tuple:
    values = sorted({row[name] for row in rows if name in row and math.isfinite(float(row[name]))})
    return tuple(values)


def load_capture(key: str, label: str, paths: Sequence[Path]) -> Capture | None:
    """One capture from its ``evaluate-snr`` shards, or ``None`` when they are not on this machine."""
    shards = [Path(p) for p in paths if (Path(p) / ev.TRIALS_NAME).is_file()]
    if not shards:
        return None
    rows: list[dict] = []
    for shard in shards:
        rows.extend(ev.read_numeric_csv(shard / ev.TRIALS_NAME, required=(ev.REQUESTED_SNR,)))
    if not rows:
        return None
    snr, statistic = paired_statistics(rows)
    if snr.size == 0:
        return None
    diffs = [float(row["cpu_gpu_abs_diff"]) for row in rows if math.isfinite(float(row.get("cpu_gpu_abs_diff", math.nan)))]
    return Capture(
        key=key, label=label, inputs=tuple(shard / ev.TRIALS_NAME for shard in shards),
        snr_db=snr, statistic=statistic,
        offsets_hz=_column(rows, ev.FREQUENCY_OFFSET),
        input_streams=tuple(int(v) for v in _column(rows, "num_input_streams")),
        detector_rows=tuple(int(v) for v in _column(rows, "detector_rows_per_frame")),
        cpu_gpu_equal=sum(1 for d in diffs if d == 0.0), cpu_gpu_trials=len(diffs))


def load_captures(sweep: Path | str | None = None, ota: Path | str | None = None) -> list[Capture]:
    """The campaign's captures, in table order; a missing one is simply absent."""
    sweep_root = sweep_dir(sweep)
    out = []
    synthetic = load_capture("synthetic", "Synthetic GNU Radio sweep", ev.digital_sweep_layout(sweep_root))
    if synthetic is not None:
        out.append(synthetic)
    radio = load_capture("ota", "Over-the-air channel-35 capture", [ota_dir(ota)])
    if radio is not None:
        out.append(radio)
    return out


# -------------------------------------------------------------------- curves
def thresholds(capture: Capture, *, pfa: float = NOMINAL_PFA) -> dict:
    """Per path, the statistic exceeded by a fraction ``pfa`` of that capture's null trials."""
    null = capture.null_mask
    if not null.any():
        return {p: math.nan for p in capture.statistic}
    return {p: float(np.quantile(capture.statistic[p][null], 1.0 - pfa)) for p in capture.statistic}


def detection_curves(capture: Capture, *, pfa: float = NOMINAL_PFA) -> tuple[np.ndarray, dict, dict, dict]:
    """``(points, fixed-threshold Pd, positive-excess Pd, thresholds)`` on the capture's own SNR grid."""
    points = capture.points
    tau = thresholds(capture, pfa=pfa)
    fixed, positive = {}, {}
    for prefix, values in capture.statistic.items():
        fixed[prefix] = np.array([float((values[capture.snr_db == s] > tau[prefix]).mean()) for s in points])
        positive[prefix] = np.array([float((values[capture.snr_db == s] > 1.0).mean()) for s in points])
    return points, fixed, positive, tau


def crossing(points: np.ndarray, pd: np.ndarray, level: float) -> tuple[float, float, int]:
    """``(SNR, slope, index)`` where ``pd`` last rises through ``level``, linear in the grid interval.

    Scanning from the top makes a non-monotone low-SNR tail (finite trials at a
    Pd of a few per cent) irrelevant to where the curve actually crosses.
    """
    s, p = np.asarray(points, float), np.asarray(pd, float)
    below = np.nonzero(p < level)[0]
    if below.size == 0 or below[-1] + 1 >= s.size:
        return math.nan, math.nan, -1
    i = int(below[-1])
    slope = (p[i + 1] - p[i]) / (s[i + 1] - s[i])
    if not (math.isfinite(slope) and slope > 0.0):
        return math.nan, math.nan, -1
    return float(s[i] + (level - p[i]) / slope), float(slope), i


def crossing_shift(points: np.ndarray, reference: np.ndarray, test: np.ndarray, level: float) -> float:
    """The paired horizontal shift of ``test`` from ``reference`` at ``level``, in dB of shelf SNR.

    The two curves come from the same trials, so their *difference* at the
    reference crossing carries the shift; dividing it by the reference slope
    turns it into a horizontal displacement without ever locating the test
    curve's own crossing, which on a 3 dB grid would jump between intervals.
    """
    star, slope, i = crossing(points, reference, level)
    if i < 0:
        return math.nan
    s = np.asarray(points, float)
    difference = np.asarray(test, float) - np.asarray(reference, float)
    at_star = difference[i] + (difference[i + 1] - difference[i]) * (star - s[i]) / (s[i + 1] - s[i])
    return float(-at_star / slope)


def _shift_of(points, fixed, positive, curve: str, level: float) -> float:
    table = fixed if curve == "fixed" else positive
    return crossing_shift(points, table["cpu_float"], table["cpu_packed"], level)


def bootstrap_shifts(capture: Capture, *, pfa: float = NOMINAL_PFA, samples: int = BOOTSTRAP_SAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """``{criterion: (low, high, zero_fraction)}`` from the paired trial bootstrap.

    Trials are resampled with replacement inside each requested SNR point, so
    every replicate keeps the campaign's own allocation, and a resampled trial
    carries all three arithmetic paths together. The fixed-threshold criteria
    hold the thresholds at their full-sample values --- that is what a fixed
    threshold is --- so the interval is the sampling spread of the crossings and
    not of the null calibration.
    """
    points = capture.points
    tau = thresholds(capture, pfa=pfa)
    groups = [np.nonzero(capture.snr_db == s)[0] for s in points]
    if any(g.size < 2 for g in groups) or int(samples) < 1:
        return {key: (math.nan, math.nan, math.nan) for key, _, _, _ in CRITERIA}
    total = int(samples)
    # the decisions are fixed before resampling: a replicate only chooses which trials to count
    decisions = {("fixed", p): capture.statistic[p] > tau[p] for p in capture.statistic}
    decisions.update({("positive", p): capture.statistic[p] > 1.0 for p in capture.statistic})
    paths = ("cpu_float", "cpu_packed")
    curves = {(curve, path): np.empty((total, points.size))
              for curve in ("fixed", "positive") for path in paths}
    rng = np.random.default_rng(int(seed))
    for column, group in enumerate(groups):
        picks = rng.integers(0, group.size, size=(total, group.size))
        taken = group[picks]
        for curve in ("fixed", "positive"):
            for path in paths:
                curves[(curve, path)][:, column] = decisions[(curve, path)][taken].mean(axis=1)
    out = {}
    for key, _label, curve, level in CRITERIA:
        draws = np.array([crossing_shift(points, curves[(curve, "cpu_float")][b],
                                         curves[(curve, "cpu_packed")][b], level) for b in range(total)])
        draws = draws[np.isfinite(draws)]
        if draws.size < 2:
            out[key] = (math.nan, math.nan, math.nan)
            continue
        low, high = (float(v) for v in np.quantile(draws, [0.025, 0.975]))
        out[key] = (low, high, float((draws == 0.0).mean()))
    return out


@dataclass(frozen=True)
class LossRow:
    """One measured crossing: the shift, its interval and the crossings it came from."""

    capture: str
    capture_label: str
    criterion: str
    criterion_label: str
    shift_db: float
    low_db: float
    high_db: float
    zero_fraction: float
    float_crossing_db: float
    packed_crossing_db: float
    slope_per_db: float
    resolution_db: float                     # the shift one extra detected trial would produce

    @property
    def verdict(self) -> str:
        """Judged against :data:`MARGIN_DB`, but only where the capture can resolve it.

        One trial changing its decision moves this crossing by
        ``resolution_db``; a capture whose quantum is coarser than the margin
        cannot decide the margin, however its interval happens to fall.
        """
        if not (math.isfinite(self.high_db) and math.isfinite(self.resolution_db)):
            return DASH
        if self.resolution_db > MARGIN_DB:
            return "unresolved"
        return "within" if self.high_db <= MARGIN_DB else "not established"


def loss_rows(capture: Capture, *, pfa: float = NOMINAL_PFA, samples: int = BOOTSTRAP_SAMPLES,
              seed: int = BOOTSTRAP_SEED) -> list[LossRow]:
    """The three crossing rows of one capture."""
    points, fixed, positive, _tau = detection_curves(capture, pfa=pfa)
    intervals = bootstrap_shifts(capture, pfa=pfa, samples=samples, seed=seed)
    rows = []
    for key, label, curve, level in CRITERIA:
        table = fixed if curve == "fixed" else positive
        star, slope, index = crossing(points, table["cpu_float"], level)
        packed, _s, _i = crossing(points, table["cpu_packed"], level)
        low, high, zero = intervals[key]
        at_point = int((capture.snr_db == points[index]).sum()) if index >= 0 else 0
        resolution = (1.0 / at_point) / slope if at_point and math.isfinite(slope) and slope > 0 else math.nan
        rows.append(LossRow(capture.key, capture.label, key, label,
                            _shift_of(points, fixed, positive, curve, level), low, high, zero,
                            star, packed, slope, resolution))
    return rows


# --------------------------------------------------------------------- table
def _interval(low: float, high: float) -> str:
    if not (math.isfinite(low) and math.isfinite(high)):
        return DASH
    return f"$[{fmt(low, 3, plus=True)},\\,{fmt(high, 3, plus=True)}]$"


def _maths(value, digits: int = 3, *, plus: bool = False) -> str:
    text = fmt(value, digits, plus=plus)
    return DASH if text == DASH else f"${text}$"


def build(run: Run, *, captures: Sequence[Capture] | None = None, samples: int = BOOTSTRAP_SAMPLES) -> Fragment:
    """``tab:detection:loss``: the paired crossing shifts, their intervals and the two components."""
    caps = list(captures) if captures is not None else load_captures()
    frag = Fragment(NAME, LABEL, "")
    frag.inputs = [p for c in caps for p in c.inputs]
    header = ["Crossing", "Trials", r"$\Delta$ (dB)", r"Paired-bootstrap 95\% interval (dB)",
              "Margin (dB)", "Verdict"]
    rows: list[list[str]] = []
    midrules: list[int] = []
    total_equal = total_trials = 0
    measured: list[LossRow] = []

    for capture in caps:
        points, fixed, positive, tau = detection_curves(capture)
        if rows:
            midrules.append(len(rows))
        geometry = f"coarse M={','.join(map(str, capture.input_streams))}, rows={','.join(map(str, capture.detector_rows))}"
        rows.append([r"\emph{" + tex(capture.label) + "}, " + tex(geometry) + "; offset " +
                     (r"$0$ Hz" if capture.offsets_hz == (0.0,) else tex(str(capture.offsets_hz))),
                     "", "", "", "", ""])
        for row in loss_rows(capture, samples=samples):
            measured.append(row)
            rows.append([r"\quad " + row.criterion_label, fmt_int(capture.trials),
                         _maths(row.shift_db, plus=True), _interval(row.low_db, row.high_db),
                         _maths(MARGIN_DB, 2), row.verdict])
            base = f"ch06.detection.{{}}.{capture.key}_{row.criterion}"
            source = {"capture": capture.label, "crossing": row.criterion}
            frag.add(base.format("shift_db"), row.shift_db, precision=3, row=source, column="shift")
            frag.add(base.format("ci_low_db"), row.low_db, precision=3, row=source, column="interval")
            frag.add(base.format("ci_high_db"), row.high_db, precision=3, row=source, column="interval",
                     status="bounded")
            frag.add(base.format("crossing_float_db"), row.float_crossing_db, precision=2, row=source,
                     column="crossing")
            frag.add(base.format("crossing_packed_db"), row.packed_crossing_db, precision=2, row=source,
                     column="crossing")
            frag.add(base.format("resolution_db"), row.resolution_db, precision=3, status="derived",
                     row=source, column="Verdict")
        for prefix in ("cpu_packed", "cpu_float"):
            frag.add(f"ch06.detection.threshold.{capture.key}_{prefix}", tau.get(prefix, math.nan), precision=4,
                     row={"capture": capture.label}, column="threshold")
        frag.add(f"ch06.detection.trials.{capture.key}", capture.trials, kind="int",
                 row={"capture": capture.label}, column="Trials")
        frag.add(f"ch06.detection.null_trials.{capture.key}", capture.null_trials, kind="int",
                 row={"capture": capture.label}, column="Trials")
        frag.add(f"ch06.detection.points.{capture.key}", int(points.size), kind="int",
                 row={"capture": capture.label}, column="Trials")
        frag.add(f"ch06.detection.detector_rows.{capture.key}",
                 capture.detector_rows[0] if capture.detector_rows else math.nan, kind="int",
                 row={"capture": capture.label}, column="geometry")
        frag.add(f"ch06.detection.input_streams.{capture.key}",
                 capture.input_streams[0] if capture.input_streams else math.nan, kind="int",
                 row={"capture": capture.label}, column="geometry")
        total_equal += capture.cpu_gpu_equal
        total_trials += capture.cpu_gpu_trials

    if caps:
        midrules.append(len(rows))
    rows.append([r"\emph{Pilot offsets the campaign asks for and these products do not carry}",
                 "", "", "", "", ""])
    for key, label in UNMEASURED_OFFSETS:
        rows.append([r"\quad " + label, DASH, DASH, DASH, _maths(MARGIN_DB, 2), DASH])
        frag.add(f"ch06.detection.shift_db.offset_{key}", None, kind="text", status="pending",
                 renderings=(DASH,), row={"offset": label}, column="shift")

    midrules.append(len(rows))
    rows.append([r"\emph{Components}", "", "", "", "", ""])
    rows.append([r"\quad Frozen transform only (Sec.~\ref{sec:impl:fxfft})", DASH,
                 _maths(TRANSFORM_ONLY_DB, 4, plus=True), DASH, DASH, "component"])
    equality = f"${fmt_int(total_equal)}/{fmt_int(total_trials)}$" if total_trials else DASH
    rows.append([r"\quad Same-seed CPU/GPU statistic equality", equality,
                 _maths(0.0, 3) if total_trials else DASH, DASH, DASH,
                 "exact" if total_trials and total_equal == total_trials else DASH])
    frag.add("ch06.detection.transform_only_db", TRANSFORM_ONLY_DB, precision=4, status="derived",
             row={"component": "frozen transform"}, column="shift")
    frag.add("ch06.detection.cpu_gpu_equal", total_equal, kind="int", row={"component": "CPU/GPU"},
             column="Trials")
    frag.add("ch06.detection.cpu_gpu_trials", total_trials, kind="int", row={"component": "CPU/GPU"},
             column="Trials")
    frag.add("ch06.detection.margin_db", MARGIN_DB, precision=2, status="derived", column="Margin")
    frag.add("ch06.detection.nominal_pfa", NOMINAL_PFA, precision=2, status="derived", column="threshold")
    frag.add("ch06.detection.null_ceiling_db", NULL_CEILING_DB, precision=1, status="derived",
             column="threshold")
    frag.add("ch06.detection.searched_bins", SEARCHED_BINS, kind="int", status="derived", column="threshold")
    frag.add("ch06.detection.bootstrap_samples", BOOTSTRAP_SAMPLES, kind="int", status="derived",
             column="interval")

    frag.tex = booktabs(header, rows, "lrrccl", midrules=midrules)
    frag.notes.extend(_notes(caps, measured))
    return frag


def _notes(caps: Sequence[Capture], measured: Sequence[LossRow] = ()) -> list[str]:
    """What the table leaves out, and why, in the words the chapter will need."""
    notes = [
        "sign convention: packed minus full-precision, so a positive shift is the less favourable packed "
        "crossing; the loss is bounded above by the interval's upper limit, never by the point estimate",
        "every criterion is rank-based at a matched null rate, so a packed path that only rescaled the excess "
        "would shift nothing: these rows measure a change in separation, not a level offset between the paths",
        f"the thresholds are set once per path at a nominal null false-alarm rate of {NOMINAL_PFA:g} on the "
        f"trials at or below {NULL_CEILING_DB:g} dB and are then held fixed; no threshold in this table is an "
        "operating point, because the bench campaign selects nothing",
        f"one designated target bin is searched per trial, so no |D| correction applies to the false-alarm rate "
        f"({SEARCHED_BINS} searched bin)",
        f"the {MARGIN_DB:g} dB margin is Appendix A's declared bound for the retained 512-row study, reproduced "
        "here; chapter 6 declares no margin of its own",
        f"the frozen transform's {TRANSFORM_ONLY_DB:+.4f} dB is a component from Sec. sec:impl:fxfft, not a "
        "bound, and this ladder does not exercise it: these evaluations sum coarse target/reference powers without fxfft256",
    ]
    if not caps:
        notes.append("no campaign products were found on this machine: every measured cell is dashed")
        return notes
    rows = sorted({r for c in caps for r in c.detector_rows})
    streams = sorted({m for c in caps for m in c.input_streams})
    notes.append(
        f"geometry: every trial of every capture ran at num_input_streams {streams} and "
        f"detector_rows_per_frame {rows} -- the superseded 512-row coarse geometry, not the production "
        "M = 2048; there is no M-sweep, no replicated-capture bracket and no 2048-input stack in these "
        "products, so the spatial-scaling panel of fig:detection:mscaling has no input and is not drawn")
    notes.append(
        "these runs compute one coarse ratio per trial; no fine designated-set counterpart exists on the same trials, so the fine-versus-coarse comparison and the "
        "predicted-versus-observed gain against the 10.5 dB aligned-tone benchmark cannot be formed -- the "
        "missing measurement is the fine statistic")
    offsets = sorted({o for c in caps for o in c.offsets_hz})
    notes.append(
        f"frequency_offset_hz is {offsets} on every trial of every capture, so the half-fine-bin, +/-1 kHz and "
        "+/-1.4 kHz rows are dashed rather than interpolated")
    notes.append(
        "the full-precision control differs from the packed path in weight quantization and input quantization "
        "at once; the quantized-weight float rung and the unquantized-input branch of fig:detection:paired are "
        "not in these products, so the table measures the whole representation step and cannot decompose it")
    notes.append(
        "the mask-versus-residual frontier of fig:detection:frontier has no input here: no transmitter duty "
        "cycle was run, no frame carries an on/off label or a per-frame injection truth, and no rank or "
        "multiplier was applied, so it is not attempted")
    for capture in caps:
        coarse = [r for r in measured if r.capture == capture.key and math.isfinite(r.resolution_db)
                  and r.resolution_db > MARGIN_DB]
        if coarse:
            notes.append(
                f"{capture.label}: {capture.trials} paired trials over {capture.points.size} SNR points; at "
                f"{', '.join(r.criterion for r in coarse)} one trial changing its decision moves the crossing by "
                f"more than the {MARGIN_DB:g} dB margin, so those rows are marked unresolved rather than judged, "
                "and their bootstrap distributions carry a large atom at exactly zero shift")
    return notes


# -------------------------------------------------------------------- figure
def render(run: Run, out_dir: Path | str, *, captures: Sequence[Capture] | None = None,
           samples: int = BOOTSTRAP_SAMPLES) -> list[Path]:
    """The detection-crossing figure; an empty list when the campaign products are absent."""
    import matplotlib.pyplot as plt

    from pilot_proxy import figure_style as style

    caps = list(captures) if captures is not None else load_captures()
    if not caps:
        return []
    primary = caps[0]
    points, fixed, positive, tau = detection_curves(primary)
    rows = [row for capture in caps for row in loss_rows(capture, samples=samples)]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(style.TEXT_WIDTH, 4.75))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.9), hspace=0.62, wspace=0.34)
    ax_pd = fig.add_subplot(grid[0, 0])
    ax_res = fig.add_subplot(grid[0, 1])
    ax_shift = fig.add_subplot(grid[1, :])

    colours = {"cpu_packed": style.MEASURED, "cpu_float": style.MODEL}
    names = {"cpu_packed": "packed int4 (production)", "cpu_float": "full precision"}
    window = (points >= -48.0) & (points <= -21.0)
    # the two paths land on top of each other -- which is the result -- so the control is
    # drawn as a wide pale line that the production path is laid over rather than hidden by
    widths = {"cpu_float": (3.4, 0.5, 0.0), "cpu_packed": (1.3, 1.0, 3.0)}
    for prefix in ("cpu_float", "cpu_packed"):
        lw, alpha, ms = widths[prefix]
        ax_pd.plot(points[window], fixed[prefix][window], color=colours[prefix], marker="o", ms=ms,
                   lw=lw, alpha=alpha, label=names[prefix], solid_capstyle="round")
        ax_pd.plot(points[window], positive[prefix][window], color=colours[prefix],
                   ls=(0, (2.4, 1.8)) if prefix == "cpu_packed" else "-", lw=lw, alpha=alpha)
    for level in (0.5, 0.9):
        star, _slope, _i = crossing(points, fixed["cpu_float"], level)
        if math.isfinite(star):
            ax_pd.plot([star], [level], marker="|", ms=9, color=style.INK, mew=1.1, zorder=5)
            ax_pd.annotate(f"${level:.1f}$", xy=(star, level), xytext=(3, -9), textcoords="offset points",
                           fontsize=6.6, color=style.MUTED)
    ax_pd.axhline(NOMINAL_PFA, color=style.PENDING, lw=0.8, ls=(0, (1.2, 1.6)))
    ax_pd.annotate(rf"null $P_{{\rm fa}} = {NOMINAL_PFA:g}$", xy=(-21.6, NOMINAL_PFA), xytext=(0, -4),
                   textcoords="offset points", fontsize=6.6, color=style.MUTED, va="top", ha="right")
    ax_pd.set_xlabel(r"Injected data-shelf SNR [dB]")
    ax_pd.set_ylabel(r"$P_{\rm d}$")
    ax_pd.set_ylim(-0.14, 1.08)
    ax_pd.set_title(r"Detection crossings, $M = 4$ (512 rows)", pad=4, fontsize=8.6)
    style.clean_axes(ax_pd)
    keys = [plt.Line2D([], [], color=colours[p], lw=widths[p][0], alpha=widths[p][1], label=names[p])
            for p in ("cpu_float", "cpu_packed")]
    keys += [plt.Line2D([], [], color=style.MUTED, lw=1.1, label="fixed threshold"),
             plt.Line2D([], [], color=style.MUTED, lw=1.1, ls=(0, (2.4, 1.8)), label="positive excess")]
    ax_pd.legend(handles=keys, loc="upper left", fontsize=6.6, handlelength=1.7, borderaxespad=0.2,
                 labelspacing=0.3, borderpad=0.25)
    style.panel_label(ax_pd, "a", x=-0.17)

    residual = 10.0 * np.log10(primary.statistic["cpu_packed"] / primary.statistic["cpu_float"])
    median = np.array([float(np.median(residual[primary.snr_db == s])) for s in points])
    lo = np.array([float(np.quantile(residual[primary.snr_db == s], 0.05)) for s in points])
    hi = np.array([float(np.quantile(residual[primary.snr_db == s], 0.95)) for s in points])
    ax_res.fill_between(points, lo, hi, color=style.LIGHT_BLUE, lw=0, label=r"5--95\% of trials")
    ax_res.plot(points, median, color=style.MEASURED, label="median")
    ax_res.axhline(0.0, color=style.MUTED, lw=0.7)
    ax_res.axhline(TRANSFORM_ONLY_DB, color=style.FAILURE, lw=0.9, ls=(0, (2.4, 1.8)))
    ax_res.annotate(rf"transform only, ${TRANSFORM_ONLY_DB:+.4f}$ dB", xy=(0.03, 0.04),
                    xycoords="axes fraction", ha="left", va="bottom", fontsize=6.4, color=style.FAILURE)
    ax_res.set_xlabel(r"Injected data-shelf SNR [dB]")
    ax_res.set_ylabel(r"$10\log_{10}(Q_{\rm pk}/Q_{\rm fl})$ [dB]")
    ax_res.set_title("Paired statistic residual, same trials", pad=4, fontsize=8.6)
    style.clean_axes(ax_res)
    ax_res.legend(loc="upper left", fontsize=6.8, handlelength=1.6, borderaxespad=0.2)
    style.panel_label(ax_res, "b", x=-0.21)

    ax_shift.axvspan(-MARGIN_DB, MARGIN_DB, color=style.LIGHT_GREEN, lw=0, zorder=0)
    ax_shift.axvline(0.0, color=style.MUTED, lw=0.8, zorder=1)
    labels, ticks = [], []
    for index, row in enumerate(reversed(rows)):
        y = float(index)
        colour = style.MEASURED if row.capture == primary.key else style.CONDITIONAL
        if math.isfinite(row.low_db) and math.isfinite(row.high_db):
            ax_shift.plot([row.low_db, row.high_db], [y, y], color=colour, lw=1.5, solid_capstyle="butt",
                          zorder=3)
        ax_shift.plot([row.shift_db], [y], marker="o", ms=4.4, color=colour, mec="white", mew=0.6, zorder=4)
        ticks.append(y)
        labels.append(row.criterion_label)
    ax_shift.plot([TRANSFORM_ONLY_DB], [len(rows)], marker="D", ms=3.6, color=style.FAILURE, zorder=4)
    ticks.append(float(len(rows)))
    labels.append(rf"Frozen transform only, ${TRANSFORM_ONLY_DB:+.4f}$ dB \emph{{(component)}}")
    ax_shift.set_yticks(ticks)
    ax_shift.set_yticklabels(labels, fontsize=7.4)
    ax_shift.set_ylim(-1.05, len(rows) + 2.0)
    if len(caps) > 1:
        ax_shift.axhline(len(rows) - len(CRITERIA) * len(caps) + len(CRITERIA) - 0.5, color=style.GRID, lw=0.7,
                         zorder=1)
    ax_shift.set_xlim(-1.34, 0.96)
    ax_shift.set_xlabel(r"Crossing shift, packed minus full precision [dB]")
    ax_shift.set_title(r"Paired implementation loss, with 95\% paired-bootstrap intervals", pad=4, fontsize=8.6)
    style.clean_axes(ax_shift, grid="x")
    handles = []
    for capture in caps:
        colour = style.MEASURED if capture.key == primary.key else style.CONDITIONAL
        handles.append(plt.Line2D([], [], color=colour, marker="o", ms=4.0, lw=1.5,
                                  label=f"{capture.label} ({capture.trials} trials)"))
    ax_shift.legend(handles=handles, loc="upper right", fontsize=6.8, handlelength=1.6, borderaxespad=0.3,
                    labelspacing=0.35)
    ax_shift.annotate(rf"margin $\pm{MARGIN_DB:g}$ dB", xy=(-MARGIN_DB, -0.78), xytext=(-4, 0),
                      textcoords="offset points", fontsize=6.4, color=style.CONDITIONAL, va="center",
                      ha="right")
    style.panel_label(ax_shift, "c", x=-0.315)

    pdf = style.save(fig, out / f"{FIGURE}.pdf", title="Bench detection crossings and paired implementation loss")
    png = out / f"{FIGURE}.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return [Path(pdf), png]
