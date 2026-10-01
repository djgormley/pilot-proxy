"""Estimator-transfer figures from a frozen release.

A release directory (``estimator_transfer_YYYYMMDD`` or
``sdr_ota_transfer_YYYYMMDD``) carries ``data/plot_points.csv`` with the
pooled estimator outputs per input point and their bootstrap intervals, and
``data/analysis.json`` describing the sweep. This module renders the two
dissertation figures from those tables alone, in the dissertation style, so
the figure and the release it came from can be re-derived from each other.

The secondary axes express the same quantities as one-bin pilot excess; the
conversion needs the detector's pilot calibration, which the over-the-air
release records in ``run/run_state.json`` and the digital release does not.
Pass ``calibration`` explicitly for the latter. The two releases were
calibrated differently: the digital release used the waveform audit's pilot
level, 11.918446870168612 dB below the data, and the settled sweep 11.264 dB
(each is recorded with its release).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from pilot_proxy import figure_style as style

DB = 10.0

DIGITAL_TITLE = "Synthetic GNU Radio input: CPU/GPU estimator transfer"
OTA_TITLE = "LimeSDR over-the-air estimator transfer"
IDEAL_LABEL = "Ideal local-reference benchmark"
CONDITIONED_LABEL = "Waveform-conditioned expected transfer"
CONTROL_LABEL = "Control-conditioned expected transfer"


@dataclass(frozen=True)
class Calibration:
    """Pilot calibration for the pilot-excess axes."""

    pilot_below_data_db: float
    bin_enbw_hz: float
    dtv_bandwidth_hz: float
    pilot_capture_efficiency: float = 1.0

    @property
    def offset_db(self) -> float:
        spreading = DB * math.log10(self.dtv_bandwidth_hz / self.bin_enbw_hz)
        efficiency = DB * math.log10(self.pilot_capture_efficiency)
        return -self.pilot_below_data_db + spreading + efficiency

    def shelf_to_pilot(self, values):
        return np.asarray(values, dtype=float) + self.offset_db

    def pilot_to_shelf(self, values):
        return np.asarray(values, dtype=float) - self.offset_db


@dataclass(frozen=True)
class Release:
    """One estimator-transfer release, loaded."""

    path: Path
    kind: str                      # "digital" or "ota"
    rows: tuple[dict[str, float], ...]
    analysis: dict
    calibration: Calibration

    @property
    def x_column(self) -> str:
        return "received_input_data_shelf_snr_db" if self.kind == "ota" else "requested_data_shelf_snr_db"

    def column(self, name: str) -> np.ndarray:
        return np.asarray([row.get(name, math.nan) for row in self.rows], dtype=float)


def _float(value: str) -> float:
    value = (value or "").strip()
    return float(value) if value else math.nan


def load_release(path: Path | str, *, calibration: Calibration | None = None) -> Release:
    path = Path(path)
    with (path / "data" / "plot_points.csv").open(newline="", encoding="utf-8") as fh:
        rows = tuple({k: _float(v) for k, v in row.items()} for row in csv.DictReader(fh))
    analysis = json.loads((path / "data" / "analysis.json").read_text(encoding="utf-8"))
    kind = "ota" if "received_input_data_shelf_snr_db" in rows[0] else "digital"
    if calibration is None:
        state = path / "run" / "run_state.json"
        if not state.exists():
            raise ValueError(
                f"{path.name}: no run/run_state.json to read the pilot calibration from; "
                "pass calibration= explicitly"
            )
        cal = json.loads(state.read_text(encoding="utf-8"))["detector_output_calibration"]
        calibration = Calibration(
            pilot_below_data_db=float(cal["pilot_below_data_db"]),
            bin_enbw_hz=float(cal["bin_enbw_hz"]),
            dtv_bandwidth_hz=float(cal["dtv_bandwidth_hz"]),
            pilot_capture_efficiency=float(cal.get("pilot_capture_efficiency", 1.0)),
        )
    return Release(path=path, kind=kind, rows=rows, analysis=analysis, calibration=calibration)


def _interval_label(release: Release) -> str:
    interval = str(release.analysis.get("plot", {}).get("interval", "95% bootstrap"))
    return interval.replace("%", r"\%") + " CI"


def _series(ax, x, y, low, high, *, color, label, linestyle, marker, y_floor, ci_label):
    """One estimator series: line through finite points, CI bars, and floor
    markers where the pooled excess was not positive (no finite estimate)."""
    finite = np.isfinite(y)
    if finite.any():
        ax.plot(x[finite], y[finite], color=color, linestyle=linestyle, marker=marker,
                markersize=3.6, linewidth=1.35, label=label, zorder=3)
        has_ci = finite & np.isfinite(low) & np.isfinite(high)
        if has_ci.any():
            ax.errorbar(x[has_ci], y[has_ci], yerr=[y[has_ci] - low[has_ci], high[has_ci] - y[has_ci]],
                        fmt="none", ecolor=color, elinewidth=0.8, capsize=2.0, capthick=0.8,
                        label=ci_label, zorder=2)
            ci_label = None
    censored = ~finite
    if censored.any():
        ax.scatter(x[censored], np.full(censored.sum(), y_floor), marker="v", s=16,
                   facecolors="none", edgecolors=color, linewidths=0.8, zorder=3,
                   label=None if finite.any() else label)
    return ci_label


def _parity_text(release: Release) -> str | None:
    gpu, packed = release.column("gpu_fixed_db"), release.column("cpu_packed_db")
    both = np.isfinite(gpu) & np.isfinite(packed)
    if not both.any():
        return None
    delta = float(np.max(np.abs(gpu[both] - packed[both])))
    if delta == 0.0:
        return "Packed CPU and GPU agree exactly"
    return rf"Packed CPU/GPU max $|\Delta|={delta:.3g}$ dB"


def _y_upper(x_max: float, plotted: Sequence[float]) -> float:
    baseline = max(min(float(x_max) + 1.0, 1.0), -2.0)
    finite = [float(v) for v in plotted if math.isfinite(float(v))]
    return max(baseline, max(finite, default=-math.inf) + 1.0)


def _render(release: Release, *, out: Path, title: str | None, y_min_db: float | None) -> Path:
    x = release.column(release.x_column)
    if not np.isfinite(x).any():
        raise ValueError("no finite input SNR values in plot_points.csv")
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_floor = x_min - 4.0 if y_min_db is None else float(y_min_db)
    ota = release.kind == "ota"
    cal = release.calibration

    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH, 4.3))
    dense = np.linspace(x_min, x_max, 600)
    # ideal local-reference benchmark: SNR - 10 log10(1 + 10^(SNR/10))
    ax.plot(dense, dense - DB * np.log10(1.0 + 10.0 ** (dense / DB)), linestyle=":",
            color=style.PENDING, linewidth=1.0, alpha=0.9, label=IDEAL_LABEL, zorder=1)
    expected = release.column("control_conditioned_expected_db" if ota else "waveform_conditioned_expected_db")
    fin = np.isfinite(expected)
    if fin.any():
        order = np.argsort(x[fin])
        ax.plot(x[fin][order], expected[fin][order], linestyle="--", color=style.MODEL,
                linewidth=1.3, label=CONTROL_LABEL if ota else CONDITIONED_LABEL, zorder=2)

    ci_label = _interval_label(release)
    if not ota:
        ci_label = _series(ax, x, release.column("cpu_float_db"), release.column("cpu_float_ci95_low_db"),
                           release.column("cpu_float_ci95_high_db"), color=style.CONDITIONAL,
                           label="CPU float, 0 Hz", linestyle=":", marker="o", y_floor=y_floor, ci_label=ci_label)
    ci_label = _series(ax, x, release.column("gpu_fixed_db"), release.column("gpu_ci95_low_db"),
                       release.column("gpu_ci95_high_db"), color=style.MEASURED,
                       label="GPU fixed-point, 0 Hz", linestyle="-", marker="s", y_floor=y_floor, ci_label=ci_label)
    if not ota:
        packed = release.column("cpu_packed_db"); fp = np.isfinite(packed)
        if fp.any():
            ax.scatter(x[fp], packed[fp], marker="x", s=18, color=style.INK, linewidths=0.8,
                       label="Packed CPU reference, 0 Hz", zorder=4)
        parity = _parity_text(release)
        if parity:
            ax.text(0.98, 0.03, parity, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.4, color=style.MUTED)

    ax.set_title(title or (OTA_TITLE if ota else DIGITAL_TITLE), pad=22)
    ax.set_xlabel((r"Received input data-shelf SNR, $\mathrm{SNR}_{\mathrm{shelf}}\;[\mathrm{dB}]$" if ota
                   else r"Known data-shelf SNR, $\mathrm{SNR}_{\mathrm{shelf}}\;[\mathrm{dB}]$"))
    ax.set_ylabel(r"Estimated data-shelf SNR, $\widehat{\mathrm{SNR}}_{\mathrm{shelf}}\;[\mathrm{dB}]$")
    ax.set_xlim(x_min - 0.6, x_max + 0.6)
    plotted = [v for line in ax.get_lines() for v in np.asarray(line.get_ydata(), dtype=float).ravel()]
    ax.set_ylim(y_floor, _y_upper(x_max, plotted))
    style.clean_axes(ax)
    top = ax.secondary_xaxis("top", functions=(cal.shelf_to_pilot, cal.pilot_to_shelf))
    top.set_xlabel((r"Received input pilot-bin excess, $\rho_{\mathrm{pilot}}\;[\mathrm{dB}]$" if ota
                    else r"Known pilot-bin excess, $\rho_{\mathrm{pilot}}\;[\mathrm{dB}]$"))
    right = ax.secondary_yaxis("right", functions=(cal.shelf_to_pilot, cal.pilot_to_shelf))
    right.set_ylabel(r"Measured pilot-bin excess, $\hat{\rho}_{\mathrm{pilot}}\;[\mathrm{dB}]$")
    for sec in (top, right):
        sec.spines[:].set_color(style.MUTED); sec.tick_params(colors=style.MUTED, labelsize=7.6)
    ax.legend(loc="upper left" if ota else "lower right", fontsize=7.2,
              bbox_to_anchor=None if ota else (0.98, 0.10))
    fig.tight_layout()
    with style.stable_pdf_subset_tags():
        return style.save(fig, Path(out), title=title or (OTA_TITLE if ota else DIGITAL_TITLE))


# amssymb's amsfonts replaces \widehat with an msbm glyph, which the
# dissertation's font contract rejects. Nothing here needs AMS symbols, so the
# figure renders with Latin Modern's own extension font for the wide hat.
_PREAMBLE = r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}"


def figure_estimator_transfer(release: Release, *, out: Path, title: str | None = None,
                              y_min_db: float | None = None) -> Path:
    """Render the transfer figure for a release; returns the PDF path."""
    with matplotlib.rc_context({"text.latex.preamble": _PREAMBLE}):
        return _render(release, out=out, title=title, y_min_db=y_min_db)
