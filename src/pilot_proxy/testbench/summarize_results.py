#!/usr/bin/env python3
# coding=utf-8
"""Summarize generic PilotProxy JSON result files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.plot_style import setup_matplotlib

DEFAULT_INPUT_JSON = Path("generated/dtv_snr_eval/dtv_snr_eval.json")
DEFAULT_OUTPUT_DIR = Path("generated/summary")
DEFAULT_HISTOGRAM_BINS = 40
DEFAULT_PLOT_DPI = 300  # publication-grade raster; match chime FIGURE_DPI
FIGURE_WIDTH_IN = 7.0
FIGURE_HEIGHT_IN = 4.5
HISTOGRAM_MODE_AUTO = "auto"
HISTOGRAM_MODE_ALWAYS = "always"
HISTOGRAM_MODE_NEVER = "never"
HISTOGRAM_MODES = (HISTOGRAM_MODE_AUTO, HISTOGRAM_MODE_ALWAYS, HISTOGRAM_MODE_NEVER)

FSTAT_COLUMN = "coarse_power_ratio"
PNR_BIN_COLUMN = "pilot_excess_db"
SNR_SHELF_COLUMN = "estimated_data_shelf_snr_db"
REQUESTED_SNR_COLUMN = "requested_data_shelf_snr_db"
SUMMARY_CSV_NAME = "summary.csv"
SUMMARY_JSON_NAME = "summary.json"
FSTAT_HISTOGRAM_NAME = "fstat_histogram.png"
SNR_SHELF_HISTOGRAM_NAME = "snr_shelf_histogram.png"


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def extract_result_rows(payload: Any) -> list[dict[str, Any]]:
    """Extract result rows from validation or detection JSON payloads."""
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        raise ValueError("result JSON must contain an object or list of objects.")
    results = payload.get("results")
    if isinstance(results, list):
        return [dict(row) for row in results if isinstance(row, dict)]
    detector_output = payload.get("detector_output")
    if isinstance(detector_output, dict) and isinstance(
        detector_output.get("results"),
        list,
    ):
        return [
            dict(row)
            for row in detector_output["results"]
            if isinstance(row, dict)
        ]
    raise ValueError("could not find a results array in the input JSON.")


def _finite_array(rows: list[dict[str, Any]], column: str) -> np.ndarray:
    values = np.asarray([_as_float(row.get(column)) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def _unique_finite_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    values = _finite_array(rows, column)
    return sorted({float(value) for value in values})


def _column_summary(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    values = _finite_array(rows, column)
    if values.size == 0:
        return {
            "column": column,
            "finite_count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "column": column,
        "finite_count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact numeric summaries for core detector quantities."""
    return [
        _column_summary(rows, FSTAT_COLUMN),
        _column_summary(rows, PNR_BIN_COLUMN),
        _column_summary(rows, SNR_SHELF_COLUMN),
    ]


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["column", "finite_count", "mean", "std", "min", "max"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_histogram(
    *,
    rows: list[dict[str, Any]],
    column: str,
    output_png: Path,
    title: str,
    xlabel: str,
    bins: int,
) -> bool:
    values = _finite_array(rows, column)
    if values.size == 0:
        return False
    try:
        plt = setup_matplotlib(force_agg=True)
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for histogram output. Install the optional "
            "plot dependency, for example: python -m pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    ax.hist(values, bins=int(bins), edgecolor="black", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)
    return True


def _sweep_histogram_skip_reason(rows: list[dict[str, Any]]) -> str | None:
    requested_snr_values = _unique_finite_values(rows, REQUESTED_SNR_COLUMN)
    if len(requested_snr_values) > 1:
        return (
            "input contains multiple requested_data_shelf_snr_db values; "
            "histograms over a sweep mix distinct SNR populations."
        )
    return None


def _remove_discontinued_histogram(path: Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return


def summarize_result_json(
    *,
    input_json: Path,
    output_dir: Path,
    bins: int,
    histograms: str = HISTOGRAM_MODE_AUTO,
) -> dict[str, Any]:
    """Write generic summary CSV/JSON and optional histograms for a result file."""
    if histograms not in HISTOGRAM_MODES:
        raise ValueError(
            "histograms must be one of " + ", ".join(HISTOGRAM_MODES)
        )
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    rows = extract_result_rows(payload)
    if not rows:
        raise ValueError("input JSON contained no result rows.")
    summary_rows = summarize_rows(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / SUMMARY_CSV_NAME
    summary_json = output_dir / SUMMARY_JSON_NAME
    fstat_histogram = output_dir / FSTAT_HISTOGRAM_NAME
    snr_histogram = output_dir / SNR_SHELF_HISTOGRAM_NAME

    _write_summary_csv(summary_csv, summary_rows)
    sweep_skip_reason = _sweep_histogram_skip_reason(rows)
    histogram_skip_reason = None
    should_write_histograms = histograms == HISTOGRAM_MODE_ALWAYS
    if histograms == HISTOGRAM_MODE_AUTO:
        should_write_histograms = sweep_skip_reason is None
        histogram_skip_reason = sweep_skip_reason
    elif histograms == HISTOGRAM_MODE_NEVER:
        should_write_histograms = False
        histogram_skip_reason = "histogram output disabled by --histograms never."

    wrote_fstat = False
    wrote_snr = False
    if should_write_histograms:
        wrote_fstat = _plot_histogram(
            rows=rows,
            column=FSTAT_COLUMN,
            output_png=fstat_histogram,
            title="Raw F-statistic histogram",
            xlabel="F",
            bins=int(bins),
        )
        wrote_snr = _plot_histogram(
            rows=rows,
            column=SNR_SHELF_COLUMN,
            output_png=snr_histogram,
            title="Estimated DTV shelf SNR histogram",
            xlabel="Estimated DTV shelf SNR [dB]",
            bins=int(bins),
        )
    else:
        _remove_discontinued_histogram(fstat_histogram)
        _remove_discontinued_histogram(snr_histogram)

    summary = {
        "schema_version": "pilotproxy_result_summary_v1",
        "input_json": str(input_json),
        "num_rows": int(len(rows)),
        "summary_csv": str(summary_csv),
        "histogram_mode": str(histograms),
        "histograms_skipped_reason": histogram_skip_reason,
        "fstat_histogram_png": str(fstat_histogram) if wrote_fstat else None,
        "snr_shelf_histogram_png": str(snr_histogram) if wrote_snr else None,
        "columns": summary_rows,
    }
    write_json_strict(summary_json, summary, indent=2)
    summary["summary_json"] = str(summary_json)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize generic PilotProxy JSON result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    parser.add_argument(
        "--histograms",
        choices=HISTOGRAM_MODES,
        default=HISTOGRAM_MODE_AUTO,
        help=(
            "Histogram policy. In auto mode, histograms are skipped for SNR "
            "sweeps because they mix distinct requested-SNR populations."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bins <= 0:
        raise SystemExit("--bins must be positive.")
    summary = summarize_result_json(
        input_json=args.input,
        output_dir=args.output_dir,
        bins=int(args.bins),
        histograms=str(args.histograms),
    )
    print(f"Wrote {summary['summary_csv']}")
    print(f"Wrote {summary['summary_json']}")
    if summary["histograms_skipped_reason"] is not None:
        print(f"Skipped histograms: {summary['histograms_skipped_reason']}")
    if summary["fstat_histogram_png"] is not None:
        print(f"Wrote {summary['fstat_histogram_png']}")
    if summary["snr_shelf_histogram_png"] is not None:
        print(f"Wrote {summary['snr_shelf_histogram_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
