#!/usr/bin/env python3
"""Produce a CPU-only digital adapter receipt from synthetic and existing IQ.

No device discovery, capture, transmission or GPU API is used.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import scipy
from scipy.signal import freqz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pilot_proxy.integration.sdr_upgrade_adapter import (  # noqa: E402
    DigitalAdapterConfig, INPUT_RATE_HZ, OUTPUT_RATE_HZ, K, L,
    adapt_record, digital_resample, resampler_coefficients,
)

ACCEPTED_CAPTURE_SHA256 = "556d1370f0cfb0781a21c7031a6c9df419da0ce5e9e94dd7285406257752a214"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def tone_power_comparison(power, term):
    """Report a conservative finite comparison when another term is zero."""
    other = float(max(power[i] for i in range(3) if i != term))
    return {
        "largest_other_term_power": other,
        "denominator_floor_code_squared": 1.0,
        "desired_to_other_power_lower_bound_with_unit_floor": float(power[term] / max(1., other)),
        "power_comparison_definition": "desired power / max(1, largest other-term power), with a one-squared-projection-code denominator floor; a finite lower bound when other power is zero, not the exact ratio",
    }


def independent_projection_check(result):
    """Independently decode signed nibbles and sum real integer products."""
    x = result["packed_frames"].astype(np.uint8)
    w = result["packed_weights"].astype(np.uint8)
    xr = (x >> 4).astype(np.int64)
    xi = (x & 15).astype(np.int64)
    wr = (w >> 4).astype(np.int64)
    wi = (w & 15).astype(np.int64)
    xr[xr >= 8] -= 16
    xi[xi >= 8] -= 16
    wr[wr >= 8] -= 16
    wi[wi >= 8] -= 16
    for frame in range(len(x)):
        for term in range(3):
            for row in range(L):
                re = sum(int(xr[frame, row, k]) * int(wr[term, k]) + int(xi[frame, row, k]) * int(wi[term, k]) for k in range(K))
                im = sum(int(xi[frame, row, k]) * int(wr[term, k]) - int(xr[frame, row, k]) * int(wi[term, k]) for k in range(K))
                if (re, im) != tuple(result["projections_i32"][frame, term, row]):
                    raise ValueError(f"Independent projection mismatch: {frame}, {term}, {row}")
    return {"all_equal": True, "complex_projections": int(len(x) * 3 * L), "method": "scalar signed-nibble decode and Python integer complex products; no adapter projection or unpack helper"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True,
                        help="Existing sdr_reference_transport_2026-09-09 directory; never a device")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be new or empty; existing receipts are not overwritten.")
    capture_root = args.capture_root.resolve()
    source_manifest_path = capture_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    accepted = capture_root / "attempt-001-noise/accepted.cfile"
    receipt_path = capture_root / "attempt-001-noise/receipt.json"
    protocol_path = capture_root / "protocol.json"
    input_checks = []
    for path in (accepted, receipt_path, protocol_path):
        expected = source_manifest["files"][str(path.relative_to(capture_root))]
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"Input hash mismatch: {path}")
        input_checks.append({"path": str(path), "sha256": actual})
    if sha(accepted) != ACCEPTED_CAPTURE_SHA256:
        raise ValueError("This validation freezes the existing accepted ambient record only.")
    receipt = json.loads(receipt_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if (not receipt["success"] or not receipt["accepted_interval_confirmed"]
            or receipt["worker_report"]["tx_attempted"]
            or protocol["settings"]["sample_rate_hz"] != INPUT_RATE_HZ):
        raise ValueError("Capture receipt does not establish the expected engineering input.")
    output.mkdir(parents=True, exist_ok=True)
    checks = []

    h = resampler_coefficients()
    np.save(output / "resampler_coefficients_float64.npy", h, allow_pickle=False)
    frequencies, response = freqz(h, worN=262144, fs=INPUT_RATE_HZ * 25)
    pass_error = float(np.max(abs(abs(response[frequencies <= 135000]) - 1)))
    stop_max = float(np.max(abs(response[frequencies >= OUTPUT_RATE_HZ / 2])))
    checks.append({"name": "digital_filter_response", "pass": pass_error < 5e-5 and stop_max < 4e-5,
                   "max_passband_amplitude_error": pass_error,
                   "max_stopband_amplitude": stop_max,
                   "max_stopband_db": float(20 * np.log10(stop_max)),
                   "frequency_grid_step_hz": float(frequencies[1] - frequencies[0]),
                   "scope": "sampled FIR transfer at 50 MHz, not analog receiver transfer"})
    with (output / "filter_response.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz", "amplitude"])
        for frequency, amplitude in zip(frequencies[:5244], abs(response[:5244])):
            writer.writerow([frequency, amplitude])

    synthetic = []
    for target in (-4, 4):
        config = DigitalAdapterConfig(INPUT_RATE_HZ, 100000., target, 5.)
        for term, frequency in enumerate(config.term_frequencies_hz):
            iq = np.exp(2j * np.pi * (config.mixer_hz + frequency) * np.arange(100000) / INPUT_RATE_HZ + .3j)
            result = adapt_record(iq, config)
            power = result["term_power_sums"][0]
            comparison = tone_power_comparison(power, term)
            dominance = comparison["desired_to_other_power_lower_bound_with_unit_floor"]
            check = independent_projection_check(result)
            synthetic.append({"target_bin": target, "injected_term": term,
                              "input_digital_frequency_hz": config.mixer_hz + frequency,
                              "output_digital_frequency_hz": frequency,
                              "term_power_sums": power.tolist(),
                              **comparison,
                              "independent_projections": check,
                              "pass": bool(np.argmax(power) == term and dominance > 1000)})
    checks.append({"name": "signed_target_and_both_reference_placements", "pass": all(row["pass"] for row in synthetic)})
    for offset in (-225000., 225000.):
        cfg = DigitalAdapterConfig(INPUT_RATE_HZ, 100000., 0, 5.)
        iq = np.exp(2j * np.pi * (cfg.mixer_hz + offset) * np.arange(100000) / INPUT_RATE_HZ)
        y, _ = digital_resample(iq, cfg)
        power = float(np.mean(abs(y) ** 2))
        checks.append({"name": f"alias_rejection_{int(offset)}_hz", "pass": power < 1e-8,
                       "output_power_for_unit_input_power": power})

    # Scale is fixed here as an engineering configuration, with clipping counted.
    # It is not estimated or optimized on these evaluation frames.
    config = DigitalAdapterConfig(INPUT_RATE_HZ, 100000., 0, 500.)
    iq = np.fromfile(accepted, dtype="<c8")
    result = adapt_record(iq, config)
    independent = independent_projection_check(result)
    checks.append({"name": "accepted_capture_all_integer_projections", "pass": independent["all_equal"], **independent})
    checks.append({"name": "upgrade_frame_and_row_timing", "pass": bool(
        result["metadata"]["frame_size_samples"] == 16384
        and result["packed_frames"].shape[1:] == (128, 128)
        and np.allclose(np.diff(result["frame_start_seconds"]), .04194304, atol=1e-15, rtol=0)
        and np.allclose(np.diff(result["row_start_seconds"], axis=1), .00032768, atol=1e-15, rtol=0))})
    if not all(c["pass"] for c in checks):
        raise ValueError("One or more digital adapter checks failed.")
    np.save(output / "packed_weights_int8.npy", result["packed_weights"], allow_pickle=False)
    np.save(output / "natural_weights_complex128.npy", result["natural_weights"], allow_pickle=False)
    with (output / "ambient_frame_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "first_sample_center_seconds", "target_power_sum", "lower_reference_power_sum", "upper_reference_power_sum", "coarse_ratio"])
        for index, (time, powers, ratio) in enumerate(zip(result["frame_start_seconds"], result["term_power_sums"], result["coarse_ratio"])):
            writer.writerow([index, time, *powers, ratio])
    write_json(output / "adapter_metadata.json", {"config": result["config"], **result["metadata"]})
    finite = result["coarse_ratio"][np.isfinite(result["coarse_ratio"])]
    summary = {
        "schema": "sdr-upgrade-adapter-validation-v1", "all_digital_checks_pass": True,
        "hardware_attempted": False, "gpu_used": False,
        "scope": "candidate digital adapter and replay of existing antenna-attached ambient record; no thermal-noise, steady-signal, physical RF mapping or CHIME array validation",
        "source_capture_manifest": {"path": str(source_manifest_path), "sha256": sha(source_manifest_path)},
        "inputs": input_checks, "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "thread_limits": {name: os.environ[name] for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "checks": checks, "synthetic_tones": synthetic,
        "ambient_replay": {
            "nominal_input_rate_hz": INPUT_RATE_HZ,
            "device_reported_input_rate_hz": receipt["worker_report"]["rx_rate_hz"],
            "timing_scope": "nominal digital sample-clock lattice relative to capture start; no absolute UTC or oscillator calibration",
            "input_samples": len(iq), "input_scope": "existing accepted 2-second antenna-attached RX-only ambient record",
            "frame_count": result["metadata"]["frame_count"],
            "first_frame_start_seconds": float(result["frame_start_seconds"][0]),
            "last_frame_end_exclusive_seconds": float(result["frame_start_seconds"][-1] + 16384 / OUTPUT_RATE_HZ),
            "sample_quantization": result["metadata"]["sample_quantization"],
            "ratio_status": result["metadata"]["ratio_status"],
            "finite_ratio_median": float(np.median(finite)),
            "finite_ratio_standard_deviation_ddof1": float(np.std(finite, ddof=1)),
            "independent_projections": independent,
        },
        "remaining_physical_acceptance": [
            "Establish signed digital-to-RF mapping with independently known positive and negative frequency offsets.",
            "Measure analog plus digital transfer, branch covariance, clipping and gain stability with frozen gain and scale.",
            "Collect qualified terminated or independently bounded noise and stable-signal records with complete transport receipts.",
            "Validate resulting null and signal statistics at one input; never multiply or replicate one input into 2048 independent inputs.",
            "Validate true streaming continuity and fine-detector backend handoff before live Pathfinder use.",
        ],
    }
    write_json(output / "summary.json", summary)
    sources = [
        Path(__file__), ROOT / "src/pilot_proxy/integration/sdr_upgrade_adapter.py",
        ROOT / "src/pilot_proxy/detector_reference.py",
        ROOT / "tests/core/test_sdr_upgrade_adapter.py",
        ROOT / "docs/SDR_UPGRADE_ADAPTER.md",
    ]
    source_records = []
    for source in sources:
        destination = output / "source_snapshots" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_records.append({"path": str(source), "sha256": sha(source)})
    write_json(output / "release-manifest.json", {
        "schema": "sdr-upgrade-adapter-release-manifest-v1", "sources": source_records,
        "files": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    })
    print(json.dumps({"all_digital_checks_pass": True, "frames": result["metadata"]["frame_count"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
