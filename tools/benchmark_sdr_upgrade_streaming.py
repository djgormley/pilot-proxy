#!/usr/bin/env python3
"""Frozen, CPU-only service-throughput benchmark; no live/USB claims."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback

import numpy as np
import scipy

from pilot_proxy.integration.sdr_upgrade_adapter import DigitalAdapterConfig
from pilot_proxy.integration.sdr_upgrade_streaming import StreamingDigitalAdapter

ROOT = Path(__file__).resolve().parents[1]
SAMPLES, CHUNK, REPEATS, WARMUP = 4_194_304, 8192, 5, 524_288
RATE = 2_000_000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sources():
    paths = {Path(__file__).resolve()}
    for name, module in sys.modules.copy().items():
        if name.startswith("pilot_proxy") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT):
                paths.add(path)
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def process_snapshot():
    status = Path("/proc/self/status").read_text().splitlines()
    return {
        "allowed_cpus": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "threads": int(
            next(x.split(":")[1] for x in status if x.startswith("Threads:"))
        ),
        "max_rss_kib_process_lifetime": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    model = next(
        (
            s.split(":", 1)[1].strip()
            for s in Path("/proc/cpuinfo").read_text().splitlines()
            if s.startswith("model name")
        ),
        "unavailable",
    )
    plan = {
        "schema": "sdr-streaming-throughput-plan-v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 202609090823,
        "input_samples_per_repeat": SAMPLES,
        "chunk_samples": CHUNK,
        "warmup_samples": WARMUP,
        "warmup_runs": 1,
        "timed_repeats": REPEATS,
        "nominal_input_rate_hz": RATE,
        "geometry": {
            "frame_samples": 16384,
            "K": 128,
            "L": 128,
            "output_rate_hz": 390625,
            "num_inputs": 1,
        },
        "config": asdict(DigitalAdapterConfig(RATE, 100_000.0, 0, 5.0)),
        "input_model": "NumPy PCG64(seed).uniform(-1,1,(N,2)) independent real/imaginary digital noise, plus 0.25*exp(2j*pi*100000*n/2000000), final contiguous complex64. One fixed record reused unchanged; each repeat constructs a fresh adapter at source index zero. No random-data selection.",
        "timing": {
            "primary": "perf_counter_ns wall interval around the entire push loop and finish; includes IQ slicing, service calls, state probes, immediate SHA256 packed/projection checksum sink, receipt-array assignments and normal Python loop overhead",
            "excluded": [
                "input generation and input hashing",
                "adapter constructor/filter/weight setup",
                "imports",
                "writing results to disk",
                "hardware acquisition",
                "USB transfer",
                "queued arrivals",
                "application output transport",
            ],
            "per_chunk": "perf_counter_ns around slicing, push, immediate sink, and public-state read; per-chunk state-array assignments and loop overhead are additionally covered by primary total interval",
            "cpu": "process_time_ns over the same primary interval",
            "warmup": "one fresh adapter processes the fixed record prefix of warmup_samples once; warmup times retained but excluded by design from the five-repeat primary comparison",
            "scheduling": "nice10; pin only this process to the highest-numbered initially allowed logical CPU; all requested numerical-library thread environment variables must equal1",
            "arrival_interval_seconds": CHUNK / RATE,
            "latency_scope": "synchronous instrumented service time only; no wall-clock-paced arrivals, no measured transport backlog or live end-to-end latency",
        },
        "state_scope": "maxima of public occupancies sampled after push calls, with declared internal capacities; not transient peak RSS or queue backlog. Lifetime RSS includes IQ generation and imports.",
        "acceptance": "All five attempted timed runs must complete, preserve the expected output counts/frame sizes/state bounds and identical packed/projection checksums, and each total input_samples/wall_seconds must be >=2000000. This is an observed engineering-rate comparison on one host, not a future real-time guarantee. No retries or selection.",
        "cpu_model": model,
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "initial_process": process_snapshot(),
        "source_sha256": sources(),
    }
    save(out / "plan.json", plan)
    (out / "plan.sha256").write_text(digest(out / "plan.json") + "\n")
    for name in plan["source_sha256"]:
        target = out / "source_snapshots" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)


def make_input(plan):
    rng = np.random.Generator(np.random.PCG64(plan["seed"]))
    n = plan["input_samples_per_repeat"]
    component = rng.uniform(-1, 1, (n, 2))
    tone = 0.25 * np.exp(2j * np.pi * 100_000 * np.arange(n) / RATE)
    return np.ascontiguousarray(
        component[:, 0] + 1j * component[:, 1] + tone, dtype=np.complex64
    )


def one_run(iq, config, label, out):
    adapter = StreamingDigitalAdapter(config, first_input_index=0)
    count = len(iq) // CHUNK
    service = np.zeros(count, dtype=np.int64)
    emitted = np.zeros(count, dtype=np.int64)
    occupancy = np.zeros((count, 3), dtype=np.int64)
    accumulator = hashlib.sha256()
    frames, saturated, invalid, completed_chunks = 0, 0, 0, 0
    errors = []

    def consume(items):
        nonlocal frames, saturated, invalid
        for frame in items:
            if (
                frame["frame_size_samples"] != 16384
                or frame["K"] != 128
                or frame["L"] != 128
            ):
                errors.append("invalid emitted geometry")
            if frame["frame_index_in_segment"] != frames:
                errors.append("nonconsecutive frame index")
            accumulator.update(frame["packed_frame"].tobytes(order="C"))
            accumulator.update(
                np.asarray(frame["projections_i32"], dtype="<i4").tobytes(order="C")
            )
            saturated += frame["saturated_component_count"]
            invalid += frame["ratio_status"] != "finite"
            frames += 1

    error = None
    final = None
    finish_ns = 0
    start_cpu, start_wall = time.process_time_ns(), time.perf_counter_ns()
    try:
        for i in range(count):
            tick = time.perf_counter_ns()
            chunk = iq[i * CHUNK : (i + 1) * CHUNK]
            output = adapter.push(chunk, input_start_index=i * CHUNK)
            emitted[i] = len(output)
            consume(output)
            del output
            state = adapter.state
            tock = time.perf_counter_ns()
            service[i] = tock - tick
            occupancy[i] = [
                state["pending_input_samples"],
                state["mixed_history_samples"],
                state["pending_frame_samples"],
            ]
            completed_chunks += 1
        tick = time.perf_counter_ns()
        final = adapter.finish()
        consume(final["frames"])
        del final["frames"]
        finish_ns = time.perf_counter_ns() - tick
    except Exception:
        error = traceback.format_exc()
    wall_ns = time.perf_counter_ns() - start_wall
    cpu_ns = time.process_time_ns() - start_cpu
    state = adapter.state
    capacities = state["buffer_capacities_samples"]
    limits = np.array(
        [
            capacities["pending_input"],
            capacities["mixed_history"],
            capacities["pending_frame"],
        ]
    )
    bounded = bool(np.all(occupancy[:completed_chunks] <= limits))
    valid_outputs = max(0, ((len(iq) - 1) * 25) // 128 + 1 - 64)
    expected_frames, expected_tail = divmod(valid_outputs, 16384)
    correct_count = (
        final is not None
        and frames == expected_frames
        and final["receipt"]["discarded_incomplete_frame_samples"] == expected_tail
    )
    record = {
        "label": label,
        "attempt_complete": error is None and completed_chunks == count,
        "exception": error,
        "input_samples": len(iq),
        "completed_input_chunks": completed_chunks,
        "completed_input_samples": completed_chunks * CHUNK,
        "wall_ns": wall_ns,
        "process_cpu_ns": cpu_ns,
        "finish_ns": finish_ns,
        "service_time_sum_ns": int(service.sum()),
        "input_samples_per_wall_second": len(iq) / (wall_ns * 1e-9)
        if error is None
        else None,
        "nominal_input_duration_seconds": len(iq) / RATE,
        "emitted_frames": frames,
        "expected_complete_frames": expected_frames,
        "expected_full_support_tail_samples": expected_tail,
        "output_count_pass": bool(correct_count),
        "frame_errors": errors,
        "saturated_components": saturated,
        "nonfinite_ratio_frames": invalid,
        "packed_and_projection_sha256": accumulator.hexdigest(),
        "observed_post_push_occupancy_maxima": dict(
            zip(
                ("pending_input", "mixed_history", "pending_frame"),
                occupancy[:completed_chunks].max(axis=0).tolist(),
            )
        )
        if completed_chunks
        else {},
        "declared_capacities_samples": capacities,
        "observed_state_bounds_pass": bounded,
        "post_finish_state": state,
        "finish_receipt": final,
        "process_after_run": process_snapshot(),
    }
    np.savez_compressed(
        out / f"{label}-calls.npz",
        service_ns=service,
        frames_emitted=emitted,
        post_push_occupancy=occupancy,
    )
    save(out / f"{label}.json", record)
    return record


def run(out):
    plan = json.loads((out / "plan.json").read_text())
    assert digest(out / "plan.json") == (out / "plan.sha256").read_text().strip()
    assert sources() == plan["source_sha256"], "Frozen sources changed"
    assert (
        np.__version__ == plan["numpy_version"]
        and scipy.__version__ == plan["scipy_version"]
    )
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert os.environ.get(name) == "1", name + " must be1 before process import"
    assert os.getpriority(os.PRIO_PROCESS, 0) == 10, "Launch with nice -n10"
    chosen_cpu = max(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {chosen_cpu})
    result_dir = out / "timings"
    result_dir.mkdir(exist_ok=False)
    save(
        out / "execution-environment.json",
        {
            "process": process_snapshot(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "chosen_logical_cpu": chosen_cpu,
            "perf_counter": vars(time.get_clock_info("perf_counter")),
            "process_time": vars(time.get_clock_info("process_time")),
        },
    )
    iq = make_input(plan)
    iq.flags.writeable = False
    save(
        out / "input-record.json",
        {
            "dtype": str(iq.dtype),
            "samples": len(iq),
            "bytes": iq.nbytes,
            "sha256_complex64_le": hashlib.sha256(
                np.asarray(iq, dtype="<c8").tobytes()
            ).hexdigest(),
        },
    )
    config = DigitalAdapterConfig(**plan["config"])
    warmup = one_run(iq[: plan["warmup_samples"]], config, "warmup", result_dir)
    records = []
    # An individual failure is retained; subsequent predeclared attempts still run.
    for repeat in range(plan["timed_repeats"]):
        record = one_run(iq, config, f"repeat-{repeat + 1:02d}", result_dir)
        records.append(record)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "label",
                        "attempt_complete",
                        "wall_ns",
                        "input_samples_per_wall_second",
                        "emitted_frames",
                        "observed_post_push_occupancy_maxima",
                    )
                }
            ),
            flush=True,
        )
    complete = [r for r in records if r["attempt_complete"]]
    rates = [r["input_samples_per_wall_second"] for r in complete]
    services = np.concatenate(
        [
            np.load(result_dir / f"{r['label']}-calls.npz")["service_ns"][
                : r["completed_input_chunks"]
            ]
            for r in records
        ]
    )
    checksum_match = len({r["packed_and_projection_sha256"] for r in records}) == 1
    integrity = sources() == plan["source_sha256"]
    result = {
        "schema": "sdr-streaming-throughput-summary-v1",
        "plan_sha256": digest(out / "plan.json"),
        "timed_repeats_attempted": len(records),
        "timed_repeats_completed": len(complete),
        "input_samples_per_repeat": SAMPLES,
        "nominal_rate_hz": RATE,
        "observed_min_input_samples_per_second": min(rates) if rates else None,
        "observed_median_input_samples_per_second": float(np.median(rates))
        if rates
        else None,
        "observed_max_input_samples_per_second": max(rates) if rates else None,
        "slowest_speed_relative_to_nominal": min(rates) / RATE if rates else None,
        "per_chunk_service_ns_quantiles": {
            str(q): float(np.quantile(services, q)) for q in (0, 0.5, 0.95, 0.99, 1)
        }
        if len(services)
        else {},
        "chunk_service_times_exceeding_nominal_arrival_interval": int(
            np.count_nonzero(services > CHUNK / RATE * 1e9)
        ),
        "total_timed_push_calls": len(services),
        "emitted_frames_per_completed_repeat": [r["emitted_frames"] for r in complete],
        "repeat_output_checksums_identical": checksum_match,
        "all_observed_state_bounds_pass": all(
            r["observed_state_bounds_pass"] for r in records
        ),
        "all_observed_geometry_and_output_counts_pass": all(
            r["output_count_pass"] and not r["frame_errors"] for r in records
        ),
        "frozen_sources_unchanged": integrity,
        "warmup_complete": warmup["attempt_complete"],
        "all_five_meet_observed_rate_criterion": len(complete) == REPEATS
        and min(rates) >= RATE
        and checksum_match
        and integrity
        and all(
            r["observed_state_bounds_pass"]
            and r["output_count_pass"]
            and not r["frame_errors"]
            for r in records
        ),
        "scope": "Preloaded deterministic synthetic IQ with instrumented checksum sink on one pinned CPU, at nice10. No USB, SDR acquisition, paced-arrival queue, real backlog, live deadline, physical RF, all-input array, or future throughput guarantee. Warmup excluded by predeclared rule; all five timing attempts retained.",
    }
    save(out / "summary.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (freeze if args.command == "freeze" else run)(args.output)


if __name__ == "__main__":
    main()
