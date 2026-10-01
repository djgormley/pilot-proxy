#!/usr/bin/env python3
"""Independently recount retained throughput timings; do not rerun timings."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    root = args.release
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    checks, failures = 0, []

    def verify(ok, label, count=1):
        nonlocal checks
        checks += count
        if not bool(ok):
            failures.append(label)

    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    verify(
        digest(root / "plan.json") == (root / "plan.sha256").read_text().strip(),
        "frozen plan hash",
    )
    for name, expected in plan["source_sha256"].items():
        verify(
            digest(root / "source_snapshots" / name) == expected,
            "source snapshot " + name,
        )
    # Reproduce only the deterministic input, not any timing measurement.
    n = plan["input_samples_per_repeat"]
    rng = np.random.Generator(np.random.PCG64(plan["seed"]))
    component = rng.uniform(-1, 1, (n, 2))
    # Byte identity requires the frozen complex-phase expression and operation
    # order; a float-phase cos/sin expansion is not bitwise interchangeable.
    signal = 0.25 * np.exp(2j * np.pi * 100000 * np.arange(n) / 2000000)
    iq = np.asarray(component[:, 0] + 1j * component[:, 1] + signal, dtype="<c8")
    input_hash = json.loads((root / "input-record.json").read_text())[
        "sha256_complex64_le"
    ]
    verify(
        hashlib.sha256(iq.tobytes()).hexdigest() == input_hash,
        "regenerated complex64 IQ bytes",
    )
    del iq, component, signal
    rates, all_calls, checksums = [], [], []
    valid = []
    for label in ["warmup"] + [
        f"repeat-{i + 1:02d}" for i in range(plan["timed_repeats"])
    ]:
        record = json.loads((root / "timings" / f"{label}.json").read_text())
        data = np.load(root / "timings" / f"{label}-calls.npz")
        samples = plan["warmup_samples"] if label == "warmup" else n
        chunks = samples // plan["chunk_samples"]
        service = data["service_ns"]
        verify(record["attempt_complete"], label + ": completed")
        verify(record["exception"] is None, label + ": no exception")
        verify(record["completed_input_chunks"] == chunks, label + ": chunk count")
        verify(
            record["completed_input_samples"] == samples, label + ": processed count"
        )
        verify(
            len(service) == chunks and np.all(service > 0),
            label + ": positive service durations",
            chunks,
        )
        verify(
            int(service.sum()) == record["service_time_sum_ns"], label + ": service sum"
        )
        verify(
            int(service.sum()) + record["finish_ns"] <= record["wall_ns"],
            label + ": total wall includes service and finish",
        )
        rate = samples * 1e9 / record["wall_ns"]
        verify(
            np.isclose(rate, record["input_samples_per_wall_second"], rtol=1e-14),
            label + ": throughput arithmetic",
        )
        output_count = max(0, ((samples - 1) * 25) // 128 + 1 - 64)
        expected_frames = output_count // 16384
        expected_tail = output_count % 16384
        verify(record["emitted_frames"] == expected_frames, label + ": frame count")
        verify(
            int(data["frames_emitted"].sum()) == expected_frames,
            label + ": per-call frame sum",
        )
        verify(
            record["finish_receipt"]["receipt"]["discarded_incomplete_frame_samples"]
            == expected_tail,
            label + ": discarded tail",
        )
        verify(not record["frame_errors"], label + ": geometry/index errors")
        occupancy = data["post_push_occupancy"]
        verify(
            np.all(occupancy >= 0), label + ": nonnegative occupancy", occupancy.size
        )
        verify(
            np.all(occupancy[:, 0] < 8192),
            label + ": retained input bound",
            len(occupancy),
        )
        verify(
            np.all(occupancy[:, 1] <= 455),
            label + ": retained history bound",
            len(occupancy),
        )
        verify(
            np.all(occupancy[:, 2] < 16384),
            label + ": pending frame bound",
            len(occupancy),
        )
        for j, name in enumerate(("pending_input", "mixed_history", "pending_frame")):
            verify(
                int(occupancy[:, j].max())
                == record["observed_post_push_occupancy_maxima"][name],
                label + ": reported max " + name,
            )
        verify(record["post_finish_state"]["closed"], label + ": closed segment")
        verify(
            record["process_after_run"]["threads"] == 1,
            label + ": thread count after run",
        )
        verify(record["process_after_run"]["nice"] == 10, label + ": nice priority")
        verify(
            len(record["process_after_run"]["allowed_cpus"]) == 1,
            label + ": pinned CPU",
        )
        if label != "warmup":
            rates.append(rate)
            all_calls.append(service)
            checksums.append(record["packed_and_projection_sha256"])
            valid.append(
                record["attempt_complete"]
                and not record["frame_errors"]
                and record["output_count_pass"]
                and record["observed_state_bounds_pass"]
            )
    service = np.concatenate(all_calls)
    for name, expected in (
        ("min", min(rates)),
        ("median", float(np.median(rates))),
        ("max", max(rates)),
    ):
        verify(
            np.isclose(
                summary[f"observed_{name}_input_samples_per_second"],
                expected,
                rtol=1e-14,
            ),
            "summary " + name + " rate",
        )
    for q in (0, 0.5, 0.95, 0.99, 1):
        verify(
            np.isclose(
                summary["per_chunk_service_ns_quantiles"][str(q)],
                np.quantile(service, q),
                rtol=1e-14,
            ),
            "service quantile " + str(q),
        )
    over = int(
        np.count_nonzero(
            service > plan["chunk_samples"] * 1e9 / plan["nominal_input_rate_hz"]
        )
    )
    verify(
        over == summary["chunk_service_times_exceeding_nominal_arrival_interval"],
        "over-interval call count",
    )
    identical = len(set(checksums)) == 1
    verify(
        identical == summary["repeat_output_checksums_identical"],
        "repeated output identity",
    )
    verify(
        summary["all_five_meet_observed_rate_criterion"]
        == (all(valid) and identical and min(rates) >= plan["nominal_input_rate_hz"]),
        "primary rate verdict",
    )
    result = {
        "schema": "sdr-throughput-independent-audit-v1",
        "checks": checks,
        "failures": failures,
        "scope": "Independent arithmetic/count/state audit of all retained timing records plus regenerated deterministic IQ hash; measurement instrumentation and hardware timing cannot be independently recovered from elapsed-duration files. Timings were not rerun.",
    }
    (root / "independent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
