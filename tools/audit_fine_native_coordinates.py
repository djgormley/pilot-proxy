#!/usr/bin/env python3
"""Frozen engineering audit of inverted native coordinates; no sensitivity fit.

The synthetic native analogue is the complex conjugate of an already normalized
time stream. The existing adapter reverses samples within each K window once.
This tests a declared digital convention, not a measurement of the instrument's
analogue phase response. No thresholds, probabilities, or policy are fitted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pilot_proxy.chime.frame_adapter import pack_chime_block_for_detector
from pilot_proxy.chime.hdf5_input import CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4
from pilot_proxy.detector_geometry import apply_spectral_sense_to_detector_matrix
from pilot_proxy.detector_reference import (
    matched_filter_row_projections_cpu_reference_packed,
    quantize_complex_numpy,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.fxfft import fine_power_fx
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.testbench.evaluate_snr import _ideal_float_weights_from_layout
from pilot_proxy.testbench.fine_validation_gpu import (
    DEFAULT_INPUT_SCALE,
    _quantized_inputs,
    arrays_sha256,
)

K, WINDOWS, FINE = 128, 128, 256
RTOL, ATOL = 2e-11, 1e-8
MIRROR = (-np.arange(FINE)) % FINE
AMPLITUDE = 0.03


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def raw_adapt(normal_rows):
    """Conjugate the physical time samples, then invoke per-K reversal once."""
    rows = np.asarray(normal_rows)
    if rows.ndim != 2 or rows.shape[1] != K:
        raise ValueError("normalized rows must be two-dimensional with K=128")
    return apply_spectral_sense_to_detector_matrix(
        np.conjugate(rows), spectral_sense="inverted"
    )


def projection_powers(rows, weights, streams):
    z = (rows.astype(np.complex128) @ weights.conjugate().T).T.reshape(
        3, streams, WINDOWS
    )
    transformed = np.fft.fft(z, n=FINE, axis=-1)
    fine = np.sum(abs(transformed) ** 2, axis=1, dtype=np.float64)
    coarse = np.sum(abs(z) ** 2, axis=(1, 2), dtype=np.float64)
    return z, fine, coarse


def differences(actual, expected):
    delta = abs(actual - expected)
    scale = max(float(np.max(abs(expected))), np.finfo(float).tiny)
    return {
        "maximum_absolute_difference": float(np.max(delta)),
        "maximum_difference_over_global_expected_maximum": float(np.max(delta) / scale),
    }


def assert_float_mapping(normal_rows, adapted_rows, weights, streams):
    zn, fn, cn = projection_powers(normal_rows, weights, streams)
    zr, fr, cr = projection_powers(adapted_rows, weights, streams)
    # w[n] = exp(-i omega n). Substitution t=K-1-n gives
    # z_raw = conj(w[K-1]) * conj(z_normal), independently for each weight.
    expected_z = np.conjugate(weights[:, -1])[:, None, None] * np.conjugate(zn)
    np.testing.assert_allclose(zr, expected_z, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fr, fn[:, MIRROR], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(cr, cn, rtol=RTOL, atol=ATOL)
    return {
        "projection_phase_identity": differences(zr, expected_z),
        "fine_absolute_power_mirror_identity": differences(fr, fn[:, MIRROR]),
        "coarse_absolute_power_identity": differences(cr, cn),
    }, (fn, cn, fr, cr)


def native_pack(normal_rows, *, streams, channel, coarse_channel):
    """Exercise the actual native-offset-binary block adapter before the kernel.

    The generator's symmetric int4 quantizer covers -7..7. This byte conversion
    is lossless; it does not claim native -8 clipping matches that quantizer.
    Separate byte-domain tests cover every signed nibble, including -8.
    """
    raw_packed = quantize_complex_numpy(
        np.conjugate(normal_rows), 4, DEFAULT_INPUT_SCALE
    )
    # Offset binary and two's complement differ by the high bit of each nibble.
    offset = np.bitwise_xor(raw_packed.view(np.uint8), np.uint8(0x88))
    block = offset.reshape(streams, 1, WINDOWS * K)
    result = pack_chime_block_for_detector(
        block,
        frame_size_samples=WINDOWS * K,
        detector_window_samples=K,
        spectral_sense="inverted",
        frames_in_chunk=1,
        sample_encoding=CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
        selected_coarse_channel=coarse_channel,
        physical_channel=channel,
    )
    expected = quantize_complex_numpy(raw_adapt(normal_rows), 4, DEFAULT_INPUT_SCALE)
    np.testing.assert_array_equal(result.packed[0], expected)
    return result.packed[0], {
        "native_offset_adapter_exact": True,
        "baseband_power_linear": result.baseband_power_linear.tolist(),
        "input_layout": result.input_layout,
        "packing_scope": "Digital synthetic signed -7..7 inputs, native offset bytes, one existing per-K reversal and lossless native repack",
    }


def gpu_integer(packed, weights, kernel, cp):
    device_packed = cp.asarray(packed)
    diagnostic = cp.zeros(1, dtype=cp.float32)
    fine = cp.empty((3, FINE), dtype=cp.uint64)
    coarse = cp.empty(3, dtype=cp.uint64)
    rows = cp.empty((3, packed.shape[0], 2), dtype=cp.int32)
    handle = kernel.create_raw(packed.shape[0], device_packed.data.ptr, diagnostic.data.ptr)
    try:
        kernel.compute_fused_fine_u64(
            handle, weights.ctypes.data, fine.data.ptr, coarse.data.ptr, rows.data.ptr
        )
        cp.cuda.get_current_stream().synchronize()
        return cp.asnumpy(rows), cp.asnumpy(fine), cp.asnumpy(coarse)
    finally:
        kernel.destroy(handle)


def recursive_seed_values(value, key=""):
    if isinstance(value, dict):
        for name, item in value.items():
            yield from recursive_seed_values(item, name)
    elif isinstance(value, list):
        for item in value:
            yield from recursive_seed_values(item, key)
    elif "seed" in key.lower() and isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value < 2**64:
            yield value


def allocate_seeds(root, cache, engineering, combinations):
    paths = [
        root / "results/channel29_residual_controls_2026-09-09/preflight/seed-exclusions.npz",
        root / "results/channel29_residual_controls_2026-09-09/preflight/planned-frame-seeds.npz",
    ]
    ledgers = []
    sources = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            ledgers.extend(
                np.sort(data[key].astype(np.uint64))
                for key in data.files
                if "seed" in key.lower() and data[key].dtype.kind in "iu"
            )
        sources[str(path)] = sha(path)
    extra = set(range(20260909002, 20260909008))
    plans = [cache / "plan.json", cache.parent / "preparation/plan.json"]
    plans.extend(sorted(engineering.glob("engineering-preflight-v*/plan.json")))
    plans.extend(sorted(engineering.glob("engineering-preflight-v*/report.json")))
    # Conservative enclosing interval contains all documented preflight draws,
    # including warmups that some report formats do not enumerate individually.
    prior_engineering_interval = (2026090917000000, 2026090917000000 + 10_000_000)
    for path in plans:
        value = json.loads(path.read_text())
        sources[str(path)] = sha(path)
        extra.update(recursive_seed_values(value))
    cache_plan = json.loads((cache / "plan.json").read_text())
    extra.update(int(x) for x in cache_plan["gain_seeds"].values())
    extra.update([2066258509, 2902804430, 2291208105, 424813154])
    extra32 = {s % 2**32 for s in extra}
    extra63 = {s % 2**63 for s in extra}
    selected = []
    for streams, channel in combinations:
        attempt = 0
        while True:
            token = f"fine-native-coordinate-engineering-v1/M{streams}/ch{channel}/{attempt}"
            seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") % 2**63
            candidates = (seed, seed % 2**32)
            collision = seed in extra63 or seed % 2**32 in extra32
            collision |= prior_engineering_interval[0] <= seed <= prior_engineering_interval[1]
            for ledger in ledgers:
                for candidate in candidates:
                    location = int(np.searchsorted(ledger, np.uint64(candidate)))
                    collision |= location < len(ledger) and int(ledger[location]) == candidate
            if not collision:
                break
            attempt += 1
        extra63.add(seed)
        extra32.add(seed % 2**32)
        selected.append({"streams": streams, "channel": channel, "seed": seed, "seed32": seed % 2**32, "allocation_attempt": attempt})
    return selected, {
        "source_sha256": sources,
        "prior_engineering_interval_inclusive": prior_engineering_interval,
        "excluded_focused_gpu_test_seeds": list(range(20260909002, 20260909008)),
        "rule": "Reject collisions in full and effective63 identities and conservative low32 identities; reserve all new identities before drawing",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--engineering-root", type=Path, required=True)
    args = parser.parse_args()
    if not all(p.is_absolute() for p in (args.output, args.cache, args.engineering_root)):
        raise ValueError("all CLI paths must be absolute")
    if args.output.exists():
        raise ValueError("output must be a new directory; a failed audit is never overwritten")
    import cupy as cp

    root = REPO.parent
    bank = DetectorWeightBank(explicit_path=REPO / "weights/chime_dtv_weights_k128.bin")
    channels = list(bank.supported_physical_channels())
    if channels != list(range(14, 37)):
        raise ValueError("this frozen audit requires all 23 profiles, channels14..36")
    combinations = [(8, ch) for ch in channels] + [(2048, ch) for ch in (14, 29, 36)]
    seeds, seed_audit = allocate_seeds(root, args.cache, args.engineering_root, combinations)
    kernel_path = REPO / "cuda/libfstatistic.so"
    kernel = FStatKernel(kernel_path)
    inputs = {str(p): sha(p) for p in (args.cache / "plan.json", args.cache / "manifest.json", kernel_path)}
    for ch in channels:
        index = (ch - 14) * 28
        for suffix in ("json", "npz"):
            path = args.cache / f"cases/{index:04d}_ch{ch}_nominal_p00.{suffix}"
            inputs[str(path)] = sha(path)
    sources = [Path(__file__).resolve(), REPO / "tests/testbench/test_fine_native_coordinates.py"]
    sources.extend(sorted((REPO / "src/pilot_proxy").rglob("*.py")))
    sources.extend(sorted((REPO / "weights").glob("chime_dtv_weights_k128.*")))
    source_hashes = {str(path): sha(path) for path in sources}
    plan = {
        "schema": "fine-native-coordinate-engineering-plan-v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Digital coordinate and arithmetic engineering only; no detection outcomes, physical phase certification, threshold fitting or cache requalification",
        "cases": seeds,
        "waveform_selection": "Own-channel nominal cache, payload00 window0 calibration/engineering fixture, all23 profiles; M8 all23, M2048 ch14/ch29/ch36",
        "amplitude": AMPLITUDE,
        "noise": "Independent NumPy PCG64 full seed; separate float32 normal real/imag arrays, unit complex variance; normalized/raw members deliberately share exactly one realization",
        "coordinate_recipe": "Conjugate normalized time samples, then existing per-K reversal exactly once; inverted anchor = (-normal anchor) %256",
        "expected_ideal_identity": "z_raw = conj(w[K-1])*conj(z_normal); fine_raw[k]=fine_normal[-k mod256]; coarse_raw=coarse_normal",
        "integer_scope": "Direct CPU/GPU raw-adapted equality required. Integer raw/normal mirrored equality is not a gate: rounded weights and fixed FFT phase rounding need not preserve it.",
        "native_quantizer": "Synthetic quantizer symmetric -7..7; native byte adapter also unit-tested for all256 byte values including -8",
        "gates": {"float_rtol": RTOL, "float_atol": ATOL, "tone_peak_maximum_circular_distance_bins": 1, "native_pack_exact": True, "cpu_gpu_pack_rows_fine_coarse_exact": True},
        "seed_exclusions": seed_audit,
        "input_sha256": inputs,
        "source_sha256": source_hashes,
        "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "cupy": cp.__version__, "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "kernel": kernel.version.as_string(), "specs": kernel.specs.as_descriptive_dict()},
    }
    args.output.mkdir(parents=True)
    snapshots = args.output / "source_snapshots"
    for source in sources:
        target = snapshots / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write(args.output / "plan.json", plan)
    plan_hash = sha(args.output / "plan.json")
    records = []
    (args.output / "cases").mkdir()
    for case in seeds:
        started = time.perf_counter()
        ch, streams, seed = case["channel"], case["streams"], case["seed"]
        record = dict(case, plan_sha256=plan_hash)
        try:
            path = args.cache / f"cases/{(ch-14)*28:04d}_ch{ch}_nominal_p00.npz"
            receipt = json.loads(path.with_suffix(".json").read_text())
            if sha(path) != receipt["array_sha256"] or not receipt["passed"]:
                raise ValueError("cache receipt identity or existing engineering gate failed")
            with np.load(path, allow_pickle=False) as data:
                clean, tone = data["atsc_rows"], data["ideal_tone_rows"]
            layout = bank.layout_for_physical_channel(ch)
            packed_weights, valid = bank.get_weights_for_physical_channel(ch)
            if not valid:
                raise ValueError("invalid packed weight profile")
            weights = _ideal_float_weights_from_layout(layout, detector_window_samples=K)
            tone_checks, tone_powers = assert_float_mapping(tone, raw_adapt(tone), weights, 1)
            peak = int(np.argmax(tone_powers[2][0]))
            anchor = receipt["case"]["calibrated_anchor_bin_inverted"]
            distance = min((peak-anchor) % FINE, (anchor-peak) % FINE)
            if distance > 1:
                raise AssertionError("raw tone does not land within declared inverted anchor gate")
            rng = np.random.Generator(np.random.PCG64(seed))
            shape = (streams, WINDOWS, K)
            normal = (rng.standard_normal(shape, dtype=np.float32) + 1j * rng.standard_normal(shape, dtype=np.float32)).astype(np.complex64)
            normal *= np.float32(1 / np.sqrt(2))
            normal += np.float32(AMPLITUDE) * clean[None, :, :]
            normal = normal.reshape(-1, K)
            adapted = raw_adapt(normal)
            checks, powers = assert_float_mapping(normal, adapted, weights, streams)
            packed, pack_info = native_pack(normal, streams=streams, channel=ch, coarse_channel=layout["coarse_channel_index"])
            device_pack, dequantized, clipped = _quantized_inputs(cp.asarray(adapted), DEFAULT_INPUT_SCALE, cp)
            np.testing.assert_array_equal(cp.asnumpy(device_pack), packed)
            del device_pack, dequantized
            integer_cpu = matched_filter_row_projections_cpu_reference_packed(packed, packed_weights, 4)
            fine_cpu = fine_power_fx(integer_cpu, num_streams=streams)
            coarse_cpu = np.sum(integer_cpu.astype(np.int64)**2, axis=(1, 2), dtype=np.int64).astype(np.uint64)
            integer_gpu, fine_gpu, coarse_gpu = gpu_integer(packed, packed_weights, kernel, cp)
            np.testing.assert_array_equal(integer_cpu, integer_gpu)
            np.testing.assert_array_equal(fine_cpu, fine_gpu)
            np.testing.assert_array_equal(coarse_cpu, coarse_gpu)
            # Descriptive comparison only: no invariant is presumed after quantization.
            normal_packed = quantize_complex_numpy(normal, 4, DEFAULT_INPUT_SCALE)
            normal_integer = matched_filter_row_projections_cpu_reference_packed(normal_packed, packed_weights, 4)
            normal_fine = fine_power_fx(normal_integer, num_streams=streams)
            normal_coarse = np.sum(normal_integer.astype(np.int64)**2, axis=(1, 2), dtype=np.int64).astype(np.uint64)
            arrays = {"normal_ideal_fine": powers[0], "normal_ideal_coarse": powers[1], "raw_ideal_fine": powers[2], "raw_ideal_coarse": powers[3], "raw_fixed_fine": fine_cpu, "raw_fixed_coarse": coarse_cpu, "normal_fixed_fine": normal_fine, "normal_fixed_coarse": normal_coarse}
            out = args.output / f"cases/M{streams}_ch{ch}.npz"
            np.savez_compressed(out, **arrays)
            record.update(passed=True, float_mapping=checks, tone_mapping=tone_checks, normal_anchor=receipt["case"]["calibrated_anchor_bin_normal"], inverted_anchor=anchor, raw_tone_peak=peak, tone_distance=distance, native_pack=pack_info, gpu_packing_exact=True, cpu_gpu_integer_rows_exact=True, cpu_gpu_fixed_fine_exact=True, cpu_gpu_coarse_exact=True, clip_count=int(clipped), packed_input_sha256=arrays_sha256({"packed_raw": packed}), native_integer_mirror_comparison={"is_acceptance_gate": False, "fine_bit_equal": bool(np.array_equal(fine_cpu, normal_fine[:, MIRROR])), "coarse_bit_equal": bool(np.array_equal(coarse_cpu, normal_coarse)), "fine_difference": differences(fine_cpu.astype(float), normal_fine[:, MIRROR].astype(float)), "coarse_difference": differences(coarse_cpu.astype(float), normal_coarse.astype(float))}, powers_file=str(out), powers_sha256=sha(out))
            del normal, adapted, packed, normal_packed, integer_cpu, integer_gpu, normal_integer
        except Exception as exc:
            record.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
        record["wall_seconds"] = time.perf_counter() - started
        write(args.output / f"cases/M{streams}_ch{ch}.json", record)
        records.append(record)
        cp.get_default_memory_pool().free_all_blocks()
        print(f"M{streams} ch{ch}: passed={record['passed']} {record['wall_seconds']:.3f}s", flush=True)
    identities = all(sha(path) == value for path, value in {**source_hashes, **inputs}.items())
    report = {"schema": "fine-native-coordinate-engineering-report-v1", "plan_sha256": plan_hash, "passed": identities and all(r["passed"] for r in records), "source_and_input_identities_unchanged": identities, "cases": records, "passed_count": sum(r["passed"] for r in records), "failed_count": sum(not r["passed"] for r in records), "scope": plan["scope"], "completed_utc": datetime.now(timezone.utc).isoformat()}
    write(args.output / "report.json", report)
    write(args.output / "manifest.json", {"schema": "fine-native-coordinate-manifest-v1", "files": {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()}})
    if not report["passed"]:
        raise SystemExit("one or more frozen engineering gates failed; all receipts preserved")


if __name__ == "__main__":
    main()
