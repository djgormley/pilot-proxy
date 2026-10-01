#!/usr/bin/env python3
"""Reproducible CPU engineering checks for the single-input streaming adapter."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import scipy

from pilot_proxy.integration.sdr_upgrade_adapter import (
    DigitalAdapterConfig, FRAME_SAMPLES, INPUT_RATE_HZ, adapt_record,
    array_sha256, projector_weights, resampler_coefficients,
)
from pilot_proxy.integration.sdr_upgrade_streaming import (
    INPUT_BLOCK_SAMPLES, MAX_HISTORY_SAMPLES, StreamingDigitalAdapter,
)

REPO = Path(__file__).resolve().parents[1]
SOURCES = [
    "src/pilot_proxy/integration/sdr_upgrade_adapter.py",
    "src/pilot_proxy/integration/sdr_upgrade_streaming.py",
    "src/pilot_proxy/detector_reference.py",
    "tests/core/test_sdr_upgrade_streaming.py",
    "tools/validate_sdr_upgrade_streaming.py",
    "docs/SDR_UPGRADE_STREAMING.md",
]
PARTITIONS = [[175_001], [1], [127, 128, 129, 8191, 8192, 8193], [83_890], [17, 65537, 3]]
SPECS = [
    {"name": "gaussian_signed_mixer", "seed": 431_097, "kind": "gaussian", "length": 175_001,
     "mixer_hz": -23456.125, "target_bin": -4, "sample_scale": 3.0},
    {"name": "steady_target_plus_gaussian", "seed": 431_098, "kind": "signal_noise", "length": 201_733,
     "mixer_hz": 100_123.75, "target_bin": 4, "sample_scale": 3.0},
    {"name": "steady_lower_reference", "seed": 431_099, "kind": "lower_tone", "length": 175_001,
     "mixer_hz": 100_000., "target_bin": -4, "sample_scale": 5.0},
    {"name": "zero_input", "seed": None, "kind": "zero", "length": 175_001,
     "mixer_hz": 0., "target_bin": 0, "sample_scale": 5.0},
    {"name": "steady_target_zero_reference", "seed": None, "kind": "constant", "length": 175_001,
     "mixer_hz": 0., "target_bin": 0, "sample_scale": 5.0},
    {"name": "saturated_components", "seed": None, "kind": "saturated", "length": 175_001,
     "mixer_hz": 0., "target_bin": 0, "sample_scale": 5.0},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fixture(spec):
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, spec["mixer_hz"], spec["target_bin"], spec["sample_scale"])
    size = spec["length"]
    rng = np.random.default_rng(spec["seed"])
    kind = spec["kind"]
    if kind == "zero":
        x = np.zeros(size, dtype=complex)
    elif kind in ("constant", "saturated"):
        x = np.full(size, 1 + 0j if kind == "constant" else 100 + 100j)
    elif kind == "gaussian":
        x = rng.normal(size=size) + 1j * rng.normal(size=size)
    else:
        term = 1 if kind == "lower_tone" else 0
        frequency = cfg.mixer_hz + cfg.term_frequencies_hz[term]
        x = np.exp(2j * np.pi * frequency * np.arange(size) / INPUT_RATE_HZ + .43j)
        if kind == "signal_noise":
            x += .2 * (rng.normal(size=size) + 1j * rng.normal(size=size))
    return x, cfg


def stream(x, cfg, sizes, origin=9_001_003):
    s = StreamingDigitalAdapter(cfg, first_input_index=origin)
    frames = []
    maximum = {key: 0 for key in ("pending_input_samples", "mixed_history_samples", "pending_frame_samples")}
    offset = count = 0
    while offset < x.size:
        take = min(sizes[count % len(sizes)], x.size - offset)
        frames.extend(s.push(x[offset:offset + take], input_start_index=origin + offset))
        offset += take
        count += 1
        state = s.state
        for key in maximum:
            maximum[key] = max(maximum[key], state[key])
        assert state["pending_input_samples"] < INPUT_BLOCK_SAMPLES
        assert state["mixed_history_samples"] <= MAX_HISTORY_SAMPLES
        assert state["pending_frame_samples"] < FRAME_SAMPLES
    final = s.finish()
    frames.extend(final["frames"])
    return frames, final["receipt"], maximum


def scalar_projections(packed, weights):
    # Independent integer scalar loops; do not call the projector or unpacker.
    def decode(v):
        value = int(v) & 255
        real, imag = value // 16, value % 16
        return real - 16 if real >= 8 else real, imag - 16 if imag >= 8 else imag
    result = np.empty((3, 128, 2), dtype=np.int32)
    for term in range(3):
        w = [decode(v) for v in weights[term]]
        for row in range(128):
            rr = ii = 0
            for v, (wr, wi) in zip(packed[row], w):
                xr, xi = decode(v)
                rr += xr * wr + xi * wi
                ii += xi * wr - xr * wi
            result[term, row] = rr, ii
    return result


def direct_samples(x, cfg, frames):
    # Exact FIR support indices and scalar exp independently reproduce selected
    # full convolutions, including samples on either side of internal blocks.
    h = resampler_coefficients()
    indices = [0, 1, 127, 128, 1535, 1536, 4095, 8191, 16383]
    maximum_error = 0.
    checked = 0
    for frame in frames:
        actual = frame["resampled_frame"].ravel()
        first = frame["first_untrimmed_output_index"]
        for i in indices:
            j = first + i
            low = (j * 128 - 8192 + 24) // 25
            high = j * 128 // 25
            assert 0 <= low <= high < len(x)
            total = 0j
            for n in range(low, high + 1):
                angle = -2 * math.pi * cfg.mixer_hz * n / INPUT_RATE_HZ
                mixed = x[n] * complex(math.cos(angle), math.sin(angle))
                total += 25 * h[j * 128 - n * 25] * mixed
            error = float(abs(actual[i] - total))
            maximum_error = max(maximum_error, error)
            # Scalar trig evaluation can differ with expression/kernel order.
            assert error < 2e-10 * max(1., float(np.max(abs(x))))
            checked += 1
    return {"sample_checks": checked, "max_absolute_error": maximum_error}


def compare_record(spec):
    x, cfg = fixture(spec)
    batch = adapt_record(x, cfg)
    baseline, receipt, maximum = stream(x, cfg, PARTITIONS[0])
    y = np.stack([r["resampled_frame"] for r in baseline])
    component_error = float(max(np.max(abs(y.real - batch["resampled_frames"].real)),
                                np.max(abs(y.imag - batch["resampled_frames"].imag))))
    complex_error = float(np.max(abs(y - batch["resampled_frames"])))
    assert complex_error < 2e-12 * max(1., float(np.max(abs(x))))
    margins = [r["minimum_unclipped_quantizer_half_step_distance"] for r in baseline]
    margin = min(v for v in margins if v is not None) if any(v is not None for v in margins) else None
    # The measured error is far smaller than all active quantizer-boundary
    # distances for these fixtures; saturated-only records are exempt.
    if margin is not None:
        assert margin > component_error * cfg.sample_scale
    comparisons = []
    for partition in PARTITIONS:
        frames, current_receipt, occupancy = stream(x, cfg, partition)
        assert current_receipt == receipt
        for key in maximum:
            maximum[key] = max(maximum[key], occupancy[key])
        for got, expected in zip(frames, baseline):
            for key in ("resampled_frame", "packed_frame", "projections_i32", "term_power_sums", "coarse_ratio"):
                np.testing.assert_array_equal(got[key], expected[key])
        comparisons.append({"partition_input_samples": partition, "frames": len(frames),
                            "streaming_float_and_packed_bitwise_equal": True})
    _, weights = projector_weights(cfg)
    projection_count = 0
    for index, frame in enumerate(baseline):
        np.testing.assert_array_equal(frame["packed_frame"], batch["packed_frames"][index])
        np.testing.assert_array_equal(frame["projections_i32"], batch["projections_i32"][index])
        independent = scalar_projections(frame["packed_frame"], weights)
        np.testing.assert_array_equal(frame["projections_i32"], independent)
        independent_power = np.sum(independent.astype(np.int64) ** 2, axis=(1, 2))
        np.testing.assert_array_equal(frame["term_power_sums"], independent_power)
        np.testing.assert_array_equal(frame["coarse_ratio"], batch["coarse_ratio"][index])
        projection_count += 3 * 128
    return {
        "name": spec["name"], "config": asdict(cfg), "input_samples": len(x),
        "input_complex128_le_sha256": array_sha256(x, "<c16"),
        "frame_count": len(baseline), "receipt": receipt,
        "max_resampled_complex_error_vs_batch": complex_error,
        "max_resampled_component_error_vs_batch": component_error,
        "max_scaled_component_error_vs_batch": component_error * cfg.sample_scale,
        "minimum_active_quantizer_half_step_distance": margin,
        "packed_frames_equal_batch": True,
        "direct_convolution": direct_samples(x, cfg, baseline),
        "independently_verified_complex_integer_projections": projection_count,
        "maximum_retained_buffer_occupancy": maximum,
        "partitions": comparisons,
        "saturated_component_count": sum(r["saturated_component_count"] for r in baseline),
        "ratio_statuses": [r["ratio_status"] for r in baseline],
        "resampled_complex128_le_sha256": array_sha256(y, "<c16"),
        "packed_int8_sha256": array_sha256(np.stack([r["packed_frame"] for r in baseline]), "i1"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing validation release.")
    hashes = {p: sha(REPO / p) for p in SOURCES}
    output.mkdir(parents=True)
    plan = {
        "schema": "sdr-upgrade-streaming-validation-plan-v1",
        "scope": "CPU engineering equivalence only; no hardware, RF calibration or statistical acceptance claim",
        "frame_size_samples": FRAME_SAMPLES, "fixtures": SPECS, "partitions": PARTITIONS,
        "source_sha256": hashes,
        "arithmetic_contract": {
            "chunk_partition_invariance": "bitwise float and packed equality",
            "batch_float_tolerance": "absolute complex error < 2e-12 * max(1, max(abs(input)))",
            "direct_scalar_tolerance": "absolute complex error < 2e-10 * max(1, max(abs(input)))",
            "quantizer_equivalence": "fixture-specific exact packed equality plus measured half-step margin; no universal tie equivalence",
        },
    }
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha(output / "plan.json") + "  plan.json\n")
    for source in SOURCES:
        target = output / "source_snapshots" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / source, target)
    records = [compare_record(spec) for spec in SPECS]
    tests = subprocess.run([sys.executable, "-m", "pytest", "tests/core/test_sdr_upgrade_streaming.py", "-q"],
                           cwd=REPO, capture_output=True, text=True, check=False)
    (output / "unit-tests.txt").write_text(tests.stdout + tests.stderr)
    assert tests.returncode == 0, tests.stdout + tests.stderr
    after = {p: sha(REPO / p) for p in SOURCES}
    assert after == hashes, "Generating/reference source changed during validation."
    summary = {
        "schema": "sdr-upgrade-streaming-validation-v1", "plan_sha256": sha(output / "plan.json"),
        "python_version": platform.python_version(), "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "records": records, "record_count": len(records),
        "frame_count": sum(r["frame_count"] for r in records),
        "chunk_partition_comparisons": len(records) * len(PARTITIONS),
        "independently_verified_complex_integer_projections": sum(r["independently_verified_complex_integer_projections"] for r in records),
        "direct_convolution_samples": sum(r["direct_convolution"]["sample_checks"] for r in records),
        "all_sources_unchanged": True, "all_checks_passed": True,
        "limitations": [
            "Single input, CPU only; no real-time throughput, device transport or physical RF validation.",
            "Gap/reset segments independently restart mixer phase, FIR/output lattice and frame origin.",
            "Single-input frame geometry is 16384=128*128; this does not supply 2048 independent inputs.",
            "Floating arithmetic can straddle exact quantizer half-steps; universal batch packed equality is not asserted.",
            "Memory bound covers retained state and working blocks, not caller-owned input/returned frames.",
        ],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "source-integrity.json", {"before": hashes, "after": after, "unchanged": True})
    (output / "README.md").write_text(
        "# SDR streaming digital validation\n\n"
        "This release checks a CPU streaming companion to the frozen single-record adapter. "
        "It does not operate an SDR or establish a physical noise/signal reference.\n\n"
        f"All {len(records)} deterministic records, {len(records) * len(PARTITIONS)} chunk partitions, "
        f"{summary['frame_count']} complete 16384-sample frames, "
        f"{summary['independently_verified_complex_integer_projections']} independent complex integer projections, "
        f"and {summary['direct_convolution_samples']} selected direct convolutions pass. "
        "The unit-test receipt includes known and unknown gaps, reset/restart, invalid input, "
        "subframe/tail discards, timing, bounded state, zero/infinite ratios and adversarial quantizer ties.\n\n"
        "`summary.json` records the observed batch discrepancy and active half-step margin for each record. "
        "The margin supports exact packed parity for these fixtures; no universal half-tie equivalence is claimed. "
        "Streaming output is bitwise invariant to all tested chunk partitions.\n\n"
        "Every boundary drains complete valid frames and discards an incomplete frame. "
        "New segments restart the phase/lattice; source indices provide a digital time label only. "
        "No filter history, incomplete frame or synthetic zero crosses a gap.\n\n"
        "Reproduce from the repository with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src "
        "python tools/validate_sdr_upgrade_streaming.py --output /path/to/new-release`. "
        "Existing outputs are never overwritten. Dependencies and source hashes are recorded.\n"
    )
    files = {str(p.relative_to(output)): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "release-manifest.json", {"schema": "sha256-file-release-v1", "files": files})
    print(json.dumps({"output": str(output), "records": len(records), "frames": summary["frame_count"],
                      "projections": summary["independently_verified_complex_integer_projections"], "passed": True}))


if __name__ == "__main__":
    main()
