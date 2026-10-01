#!/usr/bin/env python3
# coding=utf-8
"""Audit a generated clean ATSC complex64 waveform before detector evaluation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from pilot_proxy.dtv_units import PILOT_BELOW_DATA_DB
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.testbench.quantize import (
    ATSC_CHANNEL_WIDTH_HZ,
    ATSC_PILOT_OFFSET_HZ,
    GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
)

DB_POWER_FACTOR = 10.0
HALF_SCALE = 2.0
DEFAULT_AUDIT_MAX_SAMPLES = 262_144
MIN_AUDIT_SAMPLES = 1024
MAX_PERIODOGRAM_SEGMENT_SAMPLES = 65_536
DEFAULT_PILOT_SEARCH_HALF_WIDTH_HZ = 100_000.0
DEFAULT_PILOT_WINDOW_HALF_WIDTH_HZ = 10_000.0
DEFAULT_PILOT_EXCLUSION_HZ = 150_000.0
DEFAULT_EDGE_EXCLUSION_HZ = 250_000.0
DEFAULT_OCCUPIED_POWER_FRACTION = 0.99
SHELF_FLATNESS_LOW_PERCENTILE = 5
SHELF_FLATNESS_HIGH_PERCENTILE = 95
MIN_SHELF_MASK_BINS = 16
DEFAULT_MAX_PILOT_FREQUENCY_ERROR_HZ = 1_000.0
DEFAULT_PILOT_BELOW_DATA_TOLERANCE_DB = 2.0
DEFAULT_MIN_OCCUPIED_BANDWIDTH_FRACTION = 0.80
DEFAULT_MAX_OCCUPIED_BANDWIDTH_FRACTION = 1.03
DEFAULT_MAX_SHELF_FLATNESS_DB = 12.0
DEFAULT_MAX_EDGE_ROLLOFF_DB = -3.0


def _positive_to_db(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        return float("-inf")
    return float(DB_POWER_FACTOR * math.log10(value))


def _read_iq(path: Path, max_samples: int) -> np.ndarray:
    iq = np.fromfile(path, dtype=np.complex64, count=int(max_samples))
    if iq.size < MIN_AUDIT_SAMPLES:
        raise SystemExit(
            f"Need at least {MIN_AUDIT_SAMPLES} complex64 samples, got {iq.size}."
        )
    n_fft = 1 << int(math.floor(math.log2(iq.size)))
    return np.ascontiguousarray(iq[:n_fft])


def _periodogram(iq: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(MAX_PERIODOGRAM_SEGMENT_SAMPLES, iq.size)
    nperseg = 1 << int(math.floor(math.log2(nperseg)))
    nseg = max(1, iq.size // nperseg)
    trimmed = np.asarray(iq[: nseg * nperseg], dtype=np.complex128)
    segments = trimmed.reshape(nseg, nperseg)
    window = np.hanning(nperseg).astype(np.float64)
    spec = np.fft.fftshift(np.fft.fft(segments * window[None, :], axis=1), axes=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / sample_rate_hz))
    psd = np.mean(np.abs(spec) ** 2, axis=0)
    psd /= float(sample_rate_hz) * float(np.sum(window**2))
    return freqs, psd


def _occupied_bandwidth_hz(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    mask: np.ndarray,
    fraction: float,
) -> float:
    selected_freqs = freqs[mask]
    selected_power = psd[mask]
    order = np.argsort(selected_freqs)
    selected_freqs = selected_freqs[order]
    selected_power = selected_power[order]
    cumulative = np.cumsum(selected_power)
    if cumulative.size == 0 or cumulative[-1] <= 0.0:
        return float("nan")
    lower_q = (1.0 - float(fraction)) / HALF_SCALE
    upper_q = 1.0 - lower_q
    lo = np.interp(lower_q * cumulative[-1], cumulative, selected_freqs)
    hi = np.interp(upper_q * cumulative[-1], cumulative, selected_freqs)
    return float(hi - lo)


def _quality_check(
    *,
    name: str,
    passed: bool,
    value: float,
    margin: float,
    units: str,
    description: str,
    target: float | None = None,
    tolerance: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": name,
        "passed": bool(passed),
        "value": float(value),
        "margin": float(margin),
        "units": units,
        "description": description,
    }
    if target is not None:
        out["target"] = float(target)
    if tolerance is not None:
        out["tolerance"] = float(tolerance)
    if minimum is not None:
        out["minimum"] = float(minimum)
    if maximum is not None:
        out["maximum"] = float(maximum)
    return out


def _quality_report(
    audit: dict[str, Any],
    *,
    expected_pilot_below_data_db: float,
    max_pilot_frequency_error_hz: float,
    pilot_below_data_tolerance_db: float,
    min_occupied_bandwidth_hz: float,
    max_occupied_bandwidth_hz: float,
    max_shelf_flatness_db: float,
    max_edge_rolloff_db: float,
) -> dict[str, Any]:
    pilot_frequency_error = abs(float(audit["pilot_frequency_error_hz"]))
    pilot_frequency_margin = float(max_pilot_frequency_error_hz) - pilot_frequency_error

    pilot_level_error = abs(
        float(audit["measured_pilot_below_data_db"])
        - float(expected_pilot_below_data_db)
    )
    pilot_level_margin = float(pilot_below_data_tolerance_db) - pilot_level_error

    occupied = float(audit["occupied_bandwidth_hz"])
    occupied_margin = min(
        occupied - float(min_occupied_bandwidth_hz),
        float(max_occupied_bandwidth_hz) - occupied,
    )

    shelf_flatness = float(audit["shelf_flatness_db"])
    shelf_flatness_margin = float(max_shelf_flatness_db) - shelf_flatness

    edge_rolloff = float(audit["edge_rolloff_check_db"])
    edge_rolloff_margin = float(max_edge_rolloff_db) - edge_rolloff

    checks = [
        _quality_check(
            name="pilot_frequency_error",
            passed=pilot_frequency_margin >= 0.0,
            value=pilot_frequency_error,
            target=0.0,
            tolerance=float(max_pilot_frequency_error_hz),
            margin=pilot_frequency_margin,
            units="Hz",
            description="Absolute pilot frequency error from the expected ATSC pilot offset.",
        ),
        _quality_check(
            name="pilot_level",
            passed=pilot_level_margin >= 0.0,
            value=float(audit["measured_pilot_below_data_db"]),
            target=float(expected_pilot_below_data_db),
            tolerance=float(pilot_below_data_tolerance_db),
            margin=pilot_level_margin,
            units="dB",
            description="Measured ATSC pilot power below the estimated data shelf.",
        ),
        _quality_check(
            name="occupied_bandwidth",
            passed=occupied_margin >= 0.0,
            value=occupied,
            minimum=float(min_occupied_bandwidth_hz),
            maximum=float(max_occupied_bandwidth_hz),
            margin=occupied_margin,
            units="Hz",
            description="99% occupied bandwidth inside the nominal 6 MHz channel.",
        ),
        _quality_check(
            name="shelf_flatness",
            passed=shelf_flatness_margin >= 0.0,
            value=shelf_flatness,
            maximum=float(max_shelf_flatness_db),
            margin=shelf_flatness_margin,
            units="dB",
            description="95th-to-5th percentile PSD spread over the interior data shelf.",
        ),
        _quality_check(
            name="edge_rolloff",
            passed=edge_rolloff_margin >= 0.0,
            value=edge_rolloff,
            maximum=float(max_edge_rolloff_db),
            margin=edge_rolloff_margin,
            units="dB",
            description="Mean channel-edge PSD relative to the interior data shelf; more negative is better.",
        ),
    ]
    passed = sum(1 for check in checks if bool(check["passed"]))
    return {
        "schema_version": "pilotproxy_atsc_waveform_quality_v1",
        "quality_passed": bool(passed == len(checks)),
        "quality_score": float(passed / len(checks)),
        "num_quality_checks_passed": int(passed),
        "num_quality_checks": int(len(checks)),
        "quality_checks": checks,
    }


def audit_atsc_iq(
    *,
    input_iq: Path,
    sample_rate_hz: float = GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
    channel_width_hz: float = ATSC_CHANNEL_WIDTH_HZ,
    pilot_offset_hz: float = ATSC_PILOT_OFFSET_HZ,
    max_samples: int = DEFAULT_AUDIT_MAX_SAMPLES,
    pilot_search_half_width_hz: float = DEFAULT_PILOT_SEARCH_HALF_WIDTH_HZ,
    pilot_window_half_width_hz: float = DEFAULT_PILOT_WINDOW_HALF_WIDTH_HZ,
    pilot_exclusion_hz: float = DEFAULT_PILOT_EXCLUSION_HZ,
    edge_exclusion_hz: float = DEFAULT_EDGE_EXCLUSION_HZ,
    occupied_power_fraction: float = DEFAULT_OCCUPIED_POWER_FRACTION,
    expected_pilot_below_data_db: float = PILOT_BELOW_DATA_DB,
    max_pilot_frequency_error_hz: float = DEFAULT_MAX_PILOT_FREQUENCY_ERROR_HZ,
    pilot_below_data_tolerance_db: float = DEFAULT_PILOT_BELOW_DATA_TOLERANCE_DB,
    min_occupied_bandwidth_hz: float | None = None,
    max_occupied_bandwidth_hz: float | None = None,
    max_shelf_flatness_db: float = DEFAULT_MAX_SHELF_FLATNESS_DB,
    max_edge_rolloff_db: float = DEFAULT_MAX_EDGE_ROLLOFF_DB,
) -> dict[str, Any]:
    """Return spectral audit metrics for a clean ATSC IQ file."""
    iq = _read_iq(input_iq, max_samples)
    freqs, psd = _periodogram(iq, sample_rate_hz)
    df = float(abs(freqs[1] - freqs[0]))

    band_lower_hz = -float(channel_width_hz) / HALF_SCALE
    band_upper_hz = float(channel_width_hz) / HALF_SCALE
    expected_pilot_hz = band_lower_hz + float(pilot_offset_hz)
    band_mask = (freqs >= band_lower_hz) & (freqs <= band_upper_hz)

    pilot_search = (
        (freqs >= expected_pilot_hz - float(pilot_search_half_width_hz))
        & (freqs <= expected_pilot_hz + float(pilot_search_half_width_hz))
    )
    if not np.any(pilot_search):
        raise SystemExit("Pilot search window does not overlap the FFT grid.")
    pilot_candidates = np.flatnonzero(pilot_search)
    pilot_idx = int(pilot_candidates[int(np.argmax(psd[pilot_search]))])
    measured_pilot_frequency_hz = float(freqs[pilot_idx])

    pilot_window = (
        (freqs >= measured_pilot_frequency_hz - float(pilot_window_half_width_hz))
        & (freqs <= measured_pilot_frequency_hz + float(pilot_window_half_width_hz))
    )
    shelf_mask = (
        band_mask
        & (freqs >= band_lower_hz + float(edge_exclusion_hz))
        & (freqs <= band_upper_hz - float(edge_exclusion_hz))
        & (np.abs(freqs - measured_pilot_frequency_hz) >= float(pilot_exclusion_hz))
    )
    if np.count_nonzero(shelf_mask) < MIN_SHELF_MASK_BINS:
        raise SystemExit("Shelf mask has too few FFT bins for an audit.")

    shelf_psd = psd[shelf_mask]
    shelf_psd_median = float(np.median(shelf_psd))
    shelf_psd_mean = float(np.mean(shelf_psd))
    shelf_flatness_db = float(
        DB_POWER_FACTOR
        * math.log10(
            float(np.percentile(shelf_psd, SHELF_FLATNESS_HIGH_PERCENTILE))
            / float(np.percentile(shelf_psd, SHELF_FLATNESS_LOW_PERCENTILE))
        )
    )
    pilot_window_power = float(np.sum(psd[pilot_window]) * df)
    pilot_baseline_power = float(shelf_psd_median * np.count_nonzero(pilot_window) * df)
    measured_pilot_power = max(0.0, pilot_window_power - pilot_baseline_power)
    estimated_data_shelf_power = float(shelf_psd_median * float(channel_width_hz))
    pilot_to_data_power_db = _positive_to_db(
        measured_pilot_power / estimated_data_shelf_power
    )
    measured_pilot_below_data_db = -float(pilot_to_data_power_db)

    # Direct integration (additive v1 fields, 2026-07): no shelf model.
    # Total in-allocation power decomposes as pilot line + everything else;
    # the shelf continues underneath the pilot window, so the direct data
    # power adds the baseline back (equivalently: band total minus the
    # baseline-corrected pilot power). This is the estimator-independent
    # integrated pilot-to-data ratio the shelf extrapolation approximates.
    band_power_integrated = float(np.sum(psd[band_mask]) * df)
    data_power_direct = max(0.0, band_power_integrated - measured_pilot_power)
    measured_pilot_below_data_direct_db = -float(
        _positive_to_db(measured_pilot_power / data_power_direct)
        if data_power_direct > 0.0
        else float("nan")
    )
    # Mean-shelf variant of the extrapolated estimator, bounding the
    # median-vs-mean convention sensitivity alongside the direct integral.
    measured_pilot_below_data_mean_shelf_db = -float(
        _positive_to_db(
            measured_pilot_power / (shelf_psd_mean * float(channel_width_hz))
        )
    )

    lower_edge = band_mask & (freqs < band_lower_hz + float(edge_exclusion_hz))
    upper_edge = band_mask & (freqs > band_upper_hz - float(edge_exclusion_hz))
    edge_psd = psd[lower_edge | upper_edge]
    edge_rolloff_check_db = float(
        DB_POWER_FACTOR * math.log10(float(np.mean(edge_psd)) / shelf_psd_mean)
    )
    occupied_bandwidth_hz = _occupied_bandwidth_hz(
        freqs,
        psd,
        mask=band_mask,
        fraction=float(occupied_power_fraction),
    )

    audit = {
        "schema_version": "pilotproxy_atsc_waveform_audit_v1",
        "input_iq": str(input_iq),
        "num_samples_used": int(iq.size),
        "sample_rate_hz": float(sample_rate_hz),
        "symbol_rate_hz": float(sample_rate_hz),
        "channel_width_hz": float(channel_width_hz),
        "pilot_offset_hz": float(pilot_offset_hz),
        "expected_pilot_frequency_hz": float(expected_pilot_hz),
        "measured_pilot_frequency_hz": measured_pilot_frequency_hz,
        "pilot_frequency_error_hz": float(measured_pilot_frequency_hz - expected_pilot_hz),
        "measured_pilot_to_data_power_db": float(pilot_to_data_power_db),
        "measured_pilot_below_data_db": float(measured_pilot_below_data_db),
        "occupied_bandwidth_hz": float(occupied_bandwidth_hz),
        "occupied_power_fraction": float(occupied_power_fraction),
        "shelf_flatness_db": float(shelf_flatness_db),
        "edge_rolloff_check_db": float(edge_rolloff_check_db),
        "fft_bin_width_hz": float(df),
        "pilot_window_half_width_hz": float(pilot_window_half_width_hz),
        "pilot_exclusion_hz": float(pilot_exclusion_hz),
        "edge_exclusion_hz": float(edge_exclusion_hz),
        "shelf_psd_median": float(shelf_psd_median),
        "shelf_psd_mean": float(shelf_psd_mean),
        "band_power_integrated": float(band_power_integrated),
        "data_power_direct_integration": float(data_power_direct),
        "measured_pilot_below_data_direct_db": float(
            measured_pilot_below_data_direct_db),
        "measured_pilot_below_data_mean_shelf_db": float(
            measured_pilot_below_data_mean_shelf_db),
    }
    min_bw = (
        float(min_occupied_bandwidth_hz)
        if min_occupied_bandwidth_hz is not None
        else float(channel_width_hz) * DEFAULT_MIN_OCCUPIED_BANDWIDTH_FRACTION
    )
    max_bw = (
        float(max_occupied_bandwidth_hz)
        if max_occupied_bandwidth_hz is not None
        else float(channel_width_hz) * DEFAULT_MAX_OCCUPIED_BANDWIDTH_FRACTION
    )
    audit["quality"] = _quality_report(
        audit,
        expected_pilot_below_data_db=float(expected_pilot_below_data_db),
        max_pilot_frequency_error_hz=float(max_pilot_frequency_error_hz),
        pilot_below_data_tolerance_db=float(pilot_below_data_tolerance_db),
        min_occupied_bandwidth_hz=float(min_bw),
        max_occupied_bandwidth_hz=float(max_bw),
        max_shelf_flatness_db=float(max_shelf_flatness_db),
        max_edge_rolloff_db=float(max_edge_rolloff_db),
    )
    audit["quality_passed"] = bool(audit["quality"]["quality_passed"])
    audit["quality_score"] = float(audit["quality"]["quality_score"])
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a generated clean ATSC complex64 waveform.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-iq", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("generated/atsc/atsc_waveform_audit.json"),
    )
    parser.add_argument("--sample-rate-hz", type=float, default=GNU_RADIO_ATSC_SYMBOL_RATE_HZ)
    parser.add_argument("--channel-width-hz", type=float, default=ATSC_CHANNEL_WIDTH_HZ)
    parser.add_argument("--pilot-offset-hz", type=float, default=ATSC_PILOT_OFFSET_HZ)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_AUDIT_MAX_SAMPLES)
    parser.add_argument(
        "--pilot-search-half-width-hz",
        type=float,
        default=DEFAULT_PILOT_SEARCH_HALF_WIDTH_HZ,
    )
    parser.add_argument(
        "--pilot-window-half-width-hz",
        type=float,
        default=DEFAULT_PILOT_WINDOW_HALF_WIDTH_HZ,
    )
    parser.add_argument(
        "--pilot-exclusion-hz",
        type=float,
        default=DEFAULT_PILOT_EXCLUSION_HZ,
    )
    parser.add_argument(
        "--edge-exclusion-hz",
        type=float,
        default=DEFAULT_EDGE_EXCLUSION_HZ,
    )
    parser.add_argument(
        "--occupied-power-fraction",
        type=float,
        default=DEFAULT_OCCUPIED_POWER_FRACTION,
    )
    parser.add_argument(
        "--expected-pilot-below-data-db",
        type=float,
        default=PILOT_BELOW_DATA_DB,
    )
    parser.add_argument(
        "--max-pilot-frequency-error-hz",
        type=float,
        default=DEFAULT_MAX_PILOT_FREQUENCY_ERROR_HZ,
    )
    parser.add_argument(
        "--pilot-below-data-tolerance-db",
        type=float,
        default=DEFAULT_PILOT_BELOW_DATA_TOLERANCE_DB,
    )
    parser.add_argument("--min-occupied-bandwidth-hz", type=float, default=None)
    parser.add_argument("--max-occupied-bandwidth-hz", type=float, default=None)
    parser.add_argument(
        "--max-shelf-flatness-db",
        type=float,
        default=DEFAULT_MAX_SHELF_FLATNESS_DB,
    )
    parser.add_argument(
        "--max-edge-rolloff-db",
        type=float,
        default=DEFAULT_MAX_EDGE_ROLLOFF_DB,
    )
    parser.add_argument(
        "--fail-on-quality",
        action="store_true",
        help="Exit non-zero if any waveform quality check fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_atsc_iq(
        input_iq=args.input_iq,
        sample_rate_hz=float(args.sample_rate_hz),
        channel_width_hz=float(args.channel_width_hz),
        pilot_offset_hz=float(args.pilot_offset_hz),
        max_samples=int(args.max_samples),
        pilot_search_half_width_hz=float(args.pilot_search_half_width_hz),
        pilot_window_half_width_hz=float(args.pilot_window_half_width_hz),
        pilot_exclusion_hz=float(args.pilot_exclusion_hz),
        edge_exclusion_hz=float(args.edge_exclusion_hz),
        occupied_power_fraction=float(args.occupied_power_fraction),
        expected_pilot_below_data_db=float(args.expected_pilot_below_data_db),
        max_pilot_frequency_error_hz=float(args.max_pilot_frequency_error_hz),
        pilot_below_data_tolerance_db=float(args.pilot_below_data_tolerance_db),
        min_occupied_bandwidth_hz=args.min_occupied_bandwidth_hz,
        max_occupied_bandwidth_hz=args.max_occupied_bandwidth_hz,
        max_shelf_flatness_db=float(args.max_shelf_flatness_db),
        max_edge_rolloff_db=float(args.max_edge_rolloff_db),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_strict(args.output_json, audit, indent=2)
    print(f"Wrote {args.output_json}")
    print(f"sample_rate_hz={audit['sample_rate_hz']:.9g}")
    print(f"symbol_rate_hz={audit['symbol_rate_hz']:.9g}")
    print(f"pilot_offset_hz={audit['pilot_offset_hz']:.9g}")
    print(f"measured_pilot_frequency_hz={audit['measured_pilot_frequency_hz']:.9g}")
    print(f"measured_pilot_to_data_power_db={audit['measured_pilot_to_data_power_db']:.3f}")
    print(f"measured_pilot_below_data_db={audit['measured_pilot_below_data_db']:.3f}")
    print("measured_pilot_below_data_direct_db="
          f"{audit['measured_pilot_below_data_direct_db']:.3f}")
    print(f"occupied_bandwidth_hz={audit['occupied_bandwidth_hz']:.9g}")
    print(f"shelf_flatness_db={audit['shelf_flatness_db']:.3f}")
    print(f"edge_rolloff_check_db={audit['edge_rolloff_check_db']:.3f}")
    print(
        "quality_passed="
        f"{audit['quality_passed']} "
        f"({int(audit['quality']['num_quality_checks_passed'])}/"
        f"{int(audit['quality']['num_quality_checks'])})"
    )
    for check in audit["quality"]["quality_checks"]:
        status = "PASS" if bool(check["passed"]) else "FAIL"
        print(
            f"quality_check.{check['name']}={status} "
            f"value={float(check['value']):.6g}{check['units']} "
            f"margin={float(check['margin']):.6g}{check['units']}"
        )
    if args.fail_on_quality and not bool(audit["quality_passed"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
