"""Readers for pilot-proxy's detector evaluations, and the pooling that turns
them into estimator-transfer points.

``pilot-proxy evaluate-snr`` writes one directory per run:

- ``dtv_snr_eval.json``: the report (schema
  ``pilotproxy_snr_validation_report_v1``) with the waveform audit, the
  pilot calibration, the detector geometry and a per-point summary;
- ``dtv_snr_eval.csv``: one row per noise trial, the exact ``uint64``
  detector power terms beside the float estimates of every implementation;
- ``dtv_snr_summary.csv``: one row per requested input point, with the
  powers pooled over the point's trials before any ratio is formed.

The synthetic GNU Radio sweep of 2026-08-25 is forty such directories; the
channel-35 over-the-air capture evaluation is one. This module loads them and
pools trials into transfer points exactly the way the frozen
``estimator_transfer_YYYYMMDD`` release was built (pooled powers, a seeded
95% trial bootstrap in linear excess, the shelf-SNR conversion through the
run's own pilot calibration), so ``data/plot_points.csv`` of that release can
be re-derived from the raw shards and the two checked against each other.
Interpreting the points is what ``estimator_transfer`` does; this module only
reads and pools.
"""
from __future__ import annotations

import csv
import json
import math
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .estimator_transfer import Calibration

REPORT_SCHEMA = "pilotproxy_snr_validation_report_v1"
REPORT_NAME = "dtv_snr_eval.json"
TRIALS_NAME = "dtv_snr_eval.csv"
SUMMARY_NAME = "dtv_snr_summary.csv"

REQUESTED_SNR = "requested_data_shelf_snr_db"
RECEIVED_SNR = "received_input_data_shelf_snr_db"
CONTROL_EXPECTED = "gpu_control_expected_data_shelf_snr_db"
FREQUENCY_OFFSET = "frequency_offset_hz"
SOURCE_INDEX = "_trial_source_index"

PREFIXES = ("gpu", "cpu_float", "cpu_packed")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_825
DB = 10.0

Row = dict[str, float]


# ----------------------------------------------------------------- reading
def read_numeric_csv(path: Path | str, *, required: Iterable[str] = ()) -> list[Row]:
    """Every cell as a float (non-numeric cells become NaN); the offset defaults to 0."""
    path = Path(path)
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header")
        missing = set(required).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for raw in reader:
            row: Row = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = math.nan
            row.setdefault(FREQUENCY_OFFSET, 0.0)
            rows.append(row)
    return rows


@dataclass(frozen=True)
class Evaluation:
    """One ``evaluate-snr`` run: report, per-trial rows, per-point summary."""

    path: Path
    report: dict
    trials: tuple[Row, ...]
    summary: tuple[Row, ...]

    @property
    def calibration(self) -> Calibration:
        """The pilot calibration the run estimated shelf SNR with."""
        cal = self.report.get("calibration", {})
        geometry = self.report.get("detector_geometry", {})
        return Calibration(
            pilot_below_data_db=float(cal["pilot_below_data_db_used"]),
            bin_enbw_hz=float(geometry.get("bin_enbw_hz", cal.get("bin_enbw_hz_assumed"))),
            dtv_bandwidth_hz=float(geometry.get("dtv_bandwidth_hz", 6.0e6)),
            pilot_capture_efficiency=float(
                geometry.get("pilot_capture_efficiency", cal.get("pilot_capture_efficiency_assumed", 1.0))),
        )

    @property
    def requested_snr_db(self) -> tuple[float, ...]:
        return tuple(sorted({float(row[REQUESTED_SNR]) for row in self.summary}))

    @property
    def is_radio(self) -> bool:
        """Whether the run measured a received input (the SDR path) rather than a requested one."""
        return any(CONTROL_EXPECTED in row for row in self.summary)


def load_evaluation(path: Path | str) -> Evaluation:
    """Read one ``evaluate-snr`` directory, refusing an unknown report schema."""
    path = Path(path)
    report = json.loads((path / REPORT_NAME).read_text(encoding="utf-8"))
    schema = report.get("schema_version")
    if schema != REPORT_SCHEMA:
        raise ValueError(f"{path}: report schema {schema!r}, expected {REPORT_SCHEMA!r}")
    trials = read_numeric_csv(path / TRIALS_NAME, required=(REQUESTED_SNR,))
    summary = read_numeric_csv(path / SUMMARY_NAME, required=(REQUESTED_SNR,))
    if not trials or not summary:
        raise ValueError(f"{path}: an evaluation needs trial and summary rows")
    return Evaluation(path=path, report=report, trials=tuple(trials), summary=tuple(summary))


def load_evaluations(paths: Sequence[Path | str]) -> list[Evaluation]:
    """Load shards in the given order; the order is part of the pooling identity
    (the bootstrap resamples trials in their concatenated order)."""
    return [load_evaluation(p) for p in paths]


def digital_sweep_layout(root: Path | str) -> list[Path]:
    """The forty shard directories of the 2026-08-25 synthetic sweep, in the order
    the release pooled them: -60..-45 dB, the -42..-30 dB base and its two
    extension parts per point, the upper and high bases, then +6..+60 dB."""
    root = Path(root)
    paths = [root / "extreme_low" / f"snr_m{abs(v)}" for v in range(-60, -44, 3)]
    paths.append(root / "lower")
    for v in range(-45, -29, 3):
        for part in (1, 2):
            paths.append(root / "lower_additional" / f"snr_m{abs(v)}" / f"part_{part}")
    paths.extend((root / "upper", root / "high"))
    paths.extend(root / "high" / f"snr_p{v}" for v in range(6, 61, 3))
    return paths


# ----------------------------------------------------------------- pooling
def _finite(row: Row, *names: str) -> float:
    for name in names:
        value = float(row.get(name, math.nan))
        if math.isfinite(value):
            return value
    return math.nan


def _raw_ratio(row: Row, prefix: str) -> float:
    if prefix == "gpu":
        return _finite(row, "coarse_power_ratio")
    if prefix == "cpu_float":
        return _finite(row, "cpu_float_coarse_power_ratio")
    return _finite(row, "cpu_packed_coarse_power_ratio", "cpu_coarse_power_ratio")


def _direct_ratio(row: Row, prefix: str) -> float:
    """The normalized coarse power ratio Q of one trial, from whichever column carries it."""
    if prefix == "gpu":
        ratio = _finite(row, "normalized_coarse_power_ratio")
        if math.isfinite(ratio):
            return ratio
        excess = _finite(row, "normalized_pilot_excess")
        if math.isfinite(excess):
            return excess + 1.0
        level = _finite(row, "normalized_coarse_power_ratio_db")
        if math.isfinite(level):
            return float(DB ** (level / DB))
        return math.nan
    ratio = _finite(row, f"{prefix}_normalized_coarse_power_ratio")
    if math.isfinite(ratio):
        return ratio
    excess = _finite(row, f"{prefix}_normalized_pilot_excess")
    return excess + 1.0 if math.isfinite(excess) else math.nan


def _weight(row: Row, prefix: str) -> float:
    """The trial's reference power: the weight of its ratio in a pooled ratio."""
    if prefix == "gpu":
        direct = _finite(row, "p_ref_sum_u64", "p_ref_sum")
        lower, upper = _finite(row, "p_ref_lower_u64", "p_ref_lower"), _finite(row, "p_ref_upper_u64", "p_ref_upper")
    elif prefix == "cpu_float":
        direct = _finite(row, "cpu_float_p_ref_sum")
        lower, upper = _finite(row, "cpu_float_p_ref_lower"), _finite(row, "cpu_float_p_ref_upper")
    else:
        direct = _finite(row, "cpu_packed_p_ref_sum", "p_ref_sum_u64")
        lower, upper = _finite(row, "cpu_packed_p_ref_lower", "p_ref_lower_u64"), _finite(row, "cpu_packed_p_ref_upper", "p_ref_upper_u64")
    if math.isfinite(direct):
        return direct
    return lower + upper if math.isfinite(lower) and math.isfinite(upper) else math.nan


def _null_ratio(rows: Sequence[Row], prefix: str) -> float:
    """Median raw/normalized ratio: the weight-norm null level, for rows that carry only the raw ratio."""
    candidates = []
    for row in rows:
        raw = _raw_ratio(row, prefix)
        ratio = _direct_ratio(row, prefix)
        if prefix == "cpu_float" and not math.isfinite(ratio):
            excess_db = _finite(row, "cpu_float_pilot_excess_db")
            if math.isfinite(excess_db):
                ratio = 1.0 + DB ** (excess_db / DB)
        if prefix == "cpu_packed" and not math.isfinite(ratio):
            ratio = _direct_ratio(row, "gpu")
        if math.isfinite(raw) and math.isfinite(ratio) and ratio > 0.0:
            candidates.append(raw / ratio)
    return float(np.median(np.asarray(candidates, dtype=np.float64))) if candidates else math.nan


def trial_records(rows: Sequence[Row], prefix: str) -> list[tuple[Row, float, float]]:
    """(row, normalized ratio, weight) for every trial with both, in row order."""
    null = _null_ratio(rows, prefix)
    records = []
    for row in rows:
        ratio = _direct_ratio(row, prefix)
        if not math.isfinite(ratio) and math.isfinite(null) and null > 0.0:
            raw = _raw_ratio(row, prefix)
            if math.isfinite(raw):
                ratio = raw / null
        weight = _weight(row, prefix)
        if math.isfinite(ratio) and math.isfinite(weight) and weight > 0.0:
            records.append((row, ratio, weight))
    return records


def pooled_ratio(rows: Sequence[Row], prefix: str) -> float:
    """Powers summed over trials before the ratio: sum(Q_i w_i) / sum(w_i)."""
    pairs = [(ratio, weight) for _, ratio, weight in trial_records(rows, prefix)]
    if not pairs:
        return math.nan
    return float(sum(r * w for r, w in pairs) / sum(w for _, w in pairs))


def summary_ratio(rows: Sequence[Row], prefix: str) -> float:
    """The summary's own pooled ratio; overlapping shards are combined by pooled reference power."""
    excess_names = [f"{prefix}_pooled_normalized_pilot_excess"]
    ratio_names = [f"{prefix}_pooled_normalized_coarse_power_ratio"]
    weight_names = [f"{prefix}_pooled_p_ref_sum"]
    if prefix == "gpu":
        excess_names.append("pooled_normalized_pilot_excess")
        ratio_names.append("pooled_normalized_coarse_power_ratio")
        weight_names.append("pooled_p_ref_sum")
    pairs = []
    for row in rows:
        excess = _finite(row, *excess_names)
        ratio = excess + 1.0 if math.isfinite(excess) else _finite(row, *ratio_names)
        if math.isfinite(ratio):
            pairs.append((ratio, _finite(row, *weight_names)))
    if not pairs:
        return math.nan
    if len(pairs) == 1:
        return float(pairs[0][0])
    if any(not math.isfinite(w) or w <= 0.0 for _, w in pairs):
        raise ValueError(f"overlapping {prefix} summary shards need pooled reference powers")
    return float(sum(r * w for r, w in pairs) / sum(w for _, w in pairs))


def point_seed(prefix: str, offset_hz: float, snr_db: float) -> int:
    """The per-point bootstrap seed the releases used: fixed base plus a CRC of the point's name."""
    token = f"{prefix}:{offset_hz:.9f}:{snr_db:.9f}".encode("ascii")
    return BOOTSTRAP_SEED + int(zlib.crc32(token))


def bootstrap_excess_interval(rows: Sequence[Row], prefix: str, *, samples: int = BOOTSTRAP_SAMPLES,
                              seed: int) -> tuple[float, float]:
    """95% interval of the pooled excess Q - 1 over resampled trials (or pass clusters, when
    the rows carry ``pass_index``), in linear coordinates."""
    records = trial_records(rows, prefix)
    if len(records) < 2 or int(samples) < 1:
        return math.nan, math.nan
    passes = [_finite(row, "pass_index") for row, _, _ in records]
    if all(math.isfinite(p) for p in passes):
        grouped: dict[tuple[float, int], list[float]] = {}
        for (row, ratio, weight), p in zip(records, passes):
            source = _finite(row, SOURCE_INDEX)
            key = (source if math.isfinite(source) else 0.0, int(p))
            acc = grouped.setdefault(key, [0.0, 0.0])
            acc[0] += ratio * weight
            acc[1] += weight
        units = np.asarray(list(grouped.values()), dtype=np.float64)
    else:
        units = np.asarray([(ratio * weight, weight) for _, ratio, weight in records], dtype=np.float64)
    if units.shape[0] < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, units.shape[0], size=(int(samples), units.shape[0]))
    sampled = units[indices]
    pooled = np.sum(sampled[:, :, 0], axis=1) / np.sum(sampled[:, :, 1], axis=1)
    low, high = np.quantile(pooled - 1.0, [0.025, 0.975])
    return float(low), float(high)


def excess_to_shelf_db(excess: float, calibration: Calibration) -> float:
    """Linear one-bin pilot excess to data-shelf SNR, NaN for a non-positive excess."""
    if not math.isfinite(excess) or excess <= 0.0:
        return math.nan
    return float(calibration.pilot_to_shelf(DB * math.log10(excess)))


def ideal_transfer_db(snr_db) -> np.ndarray:
    """The ideal local-reference estimator: x - 10 log10(1 + 10^(x/10))."""
    x = np.asarray(snr_db, dtype=np.float64)
    return x - DB * np.log10(1.0 + DB ** (x / DB))


@dataclass(frozen=True)
class Conditioning:
    """The waveform-conditioned expected transfer, y = C + 10 log10((delta + a 10^(x/10)) / (1 + b 10^(x/10)))."""

    C_db: float
    delta: float
    a: float
    b: float

    @classmethod
    def from_json(cls, path: Path | str) -> "Conditioning":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        c = doc["coefficients"]
        return cls(C_db=float(c["C_db"]), delta=float(c["delta"]), a=float(c["a"]), b=float(c["b"]))

    def expected_db(self, snr_db) -> np.ndarray:
        x = np.asarray(snr_db, dtype=np.float64)
        linear = DB ** (x / DB)
        excess = (self.delta + self.a * linear) / (1.0 + self.b * linear)
        out = np.full_like(excess, np.nan)
        keep = excess > 0.0
        out[keep] = self.C_db + DB * np.log10(excess[keep])
        return out


def _rows_at(rows: Sequence[Row], *, offset_hz: float, snr_db: float) -> list[Row]:
    return [row for row in rows
            if float(row.get(FREQUENCY_OFFSET, 0.0)) == float(offset_hz)
            and float(row[REQUESTED_SNR]) == float(snr_db)]


def _concatenate(evaluations: Sequence[Evaluation]) -> tuple[list[Row], list[Row]]:
    trials: list[Row] = []
    summary: list[Row] = []
    for index, ev in enumerate(evaluations):
        for row in ev.trials:
            trials.append({**row, SOURCE_INDEX: float(index)})
        summary.extend(dict(row) for row in ev.summary)
    return trials, summary


def curve_points(evaluations: Sequence[Evaluation], *, prefix: str, calibration: Calibration,
                 offset_hz: float = 0.0, bootstrap_samples: int = BOOTSTRAP_SAMPLES,
                 radio: bool = False) -> dict[float, dict[str, float]]:
    """Per requested input point: the pooled excess, its shelf-SNR estimate and its interval."""
    trials, summary = _concatenate(evaluations)
    requested = sorted({float(row[REQUESTED_SNR]) for row in summary
                        if float(row.get(FREQUENCY_OFFSET, 0.0)) == float(offset_hz)
                        and math.isfinite(float(row[REQUESTED_SNR]))})
    points: dict[float, dict[str, float]] = {}
    for snr in requested:
        summaries = _rows_at(summary, offset_hz=offset_hz, snr_db=snr)
        point_trials = _rows_at(trials, offset_hz=offset_hz, snr_db=snr)
        ratio = summary_ratio(summaries, prefix)
        if not math.isfinite(ratio):
            ratio = pooled_ratio(point_trials, prefix)
        if not math.isfinite(ratio):
            continue
        x = _finite(summaries[0], RECEIVED_SNR, REQUESTED_SNR) if radio else _finite(summaries[0], REQUESTED_SNR)
        if not math.isfinite(x):
            continue
        low, high = bootstrap_excess_interval(point_trials, prefix, samples=bootstrap_samples,
                                              seed=point_seed(prefix, offset_hz, snr))
        points[snr] = {
            "x": x,
            "excess": ratio - 1.0,
            "y": excess_to_shelf_db(ratio - 1.0, calibration),
            "low_excess": low,
            "high_excess": high,
            "low_db": excess_to_shelf_db(low, calibration),
            "high_db": excess_to_shelf_db(high, calibration),
        }
    return points


DIGITAL_COLUMNS = (
    "requested_data_shelf_snr_db", "ideal_local_reference_db", "waveform_conditioned_expected_db",
    "gpu_fixed_db", "gpu_ci95_low_db", "gpu_ci95_high_db",
    "cpu_float_db", "cpu_float_ci95_low_db", "cpu_float_ci95_high_db", "cpu_packed_db",
    "trials", "positive_excess_trials", "positive_excess_fraction",
)


def transfer_points(evaluations: Sequence[Evaluation], *, calibration: Calibration | None = None,
                    conditioning: Conditioning | None = None, offset_hz: float = 0.0,
                    bootstrap_samples: int = BOOTSTRAP_SAMPLES) -> list[dict[str, float]]:
    """The digital release's ``plot_points.csv`` rows from the raw shards.

    ``calibration`` defaults to the first shard's own; every shard must agree
    with it. ``conditioning`` fills the waveform-conditioned column, otherwise
    NaN (the coefficients are a separate derivation, recorded in the release's
    ``run/conditioning.json``).
    """
    if not evaluations:
        raise ValueError("no evaluations to pool")
    if calibration is None:
        calibration = evaluations[0].calibration
    for ev in evaluations:
        if ev.calibration != calibration:
            raise ValueError(f"{ev.path}: pilot calibration differs from the first shard's")
    curves = {prefix: curve_points(evaluations, prefix=prefix, calibration=calibration, offset_hz=offset_hz,
                                   bootstrap_samples=bootstrap_samples) for prefix in PREFIXES}
    trials, _ = _concatenate(evaluations)
    at_offset = [row for row in trials if float(row.get(FREQUENCY_OFFSET, 0.0)) == float(offset_hz)]
    allocation = Counter(float(row[REQUESTED_SNR]) for row in at_offset)
    positive: Counter = Counter()
    for row in at_offset:
        ratio = _direct_ratio(row, "gpu")
        if math.isfinite(ratio) and ratio > 1.0:
            positive[float(row[REQUESTED_SNR])] += 1
    rows = []
    for snr in sorted(allocation):
        gpu, cpu_float, cpu_packed = (curves[p].get(snr) for p in PREFIXES)
        if gpu is None or cpu_float is None or cpu_packed is None:
            raise ValueError(f"{snr:g} dB: not every implementation has a pooled ratio")
        rows.append({
            "requested_data_shelf_snr_db": snr,
            "ideal_local_reference_db": float(ideal_transfer_db([snr])[0]),
            "waveform_conditioned_expected_db": (
                float(conditioning.expected_db([snr])[0]) if conditioning is not None else math.nan),
            "gpu_fixed_db": gpu["y"], "gpu_ci95_low_db": gpu["low_db"], "gpu_ci95_high_db": gpu["high_db"],
            "cpu_float_db": cpu_float["y"], "cpu_float_ci95_low_db": cpu_float["low_db"],
            "cpu_float_ci95_high_db": cpu_float["high_db"],
            "cpu_packed_db": cpu_packed["y"],
            "trials": allocation[snr],
            "positive_excess_trials": positive.get(snr, 0),
            "positive_excess_fraction": positive.get(snr, 0) / allocation[snr],
        })
    return rows


def write_points(rows: Sequence[dict[str, float]], path: Path | str,
                 columns: Sequence[str] = DIGITAL_COLUMNS) -> Path:
    """Write pooled points the way the releases do: shortest round-trip floats, blanks for NaN."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {}
            for name in columns:
                value = row.get(name, math.nan)
                if isinstance(value, float):
                    out[name] = "" if not math.isfinite(value) else repr(value)
                else:
                    out[name] = value
            writer.writerow(out)
    return path
