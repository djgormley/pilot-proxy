#!/usr/bin/env python3
"""Prepare or explicitly launch five finite, descriptive antenna-control triplets.

Preparation performs no device operation. Run is exclusive, consumes the frozen
schedule once, and stops at the first failed acquisition or unconfirmed cleanup.
No thermal-noise, RF-power or distribution-acceptance claim is made.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys

# Set before dynamically loading the NumPy-using capture preparer.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

REPO = Path(__file__).resolve().parents[1]
CAPTURE_HELPER = REPO / "tools/lime_reference_capture.py"
TEST_SOURCE = REPO / "tests/testbench/test_sdr_antenna_controls.py"
MODES = ("noise", "txzero", "tone")
LABELS = {"noise": "receiver-only ambient", "txzero": "TX-active zero-IQ", "tone": "commanded steady tone"}
SEED = 20260909
TRIPLETS = 5
RECORD_SAMPLES = 4_000_000
RECORD_SECONDS = 2.0
SERIAL = "0x1d423d9108f273"
ANALYSIS_CONFIG = {"input_rate_hz": 2_000_000, "mixer_hz": 100_000.0,
                   "target_bin": 0, "sample_scale": 500.0}
SCALE_REFERENCE = REPO.parent / "results/sdr_adapter_validation_2026-09-09/adapter_metadata.json"
SCALE_SUMMARY = REPO.parent / "results/sdr_adapter_validation_2026-09-09/summary.json"


def numerical_runtime():
    import numpy
    import scipy
    return {"python": sys.version, "executable": sys.executable,
            "numpy": numpy.__version__, "scipy": scipy.__version__}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mode_order():
    """Fixed random.Random(MT19937) shuffles; never redraw after acquisition."""
    rng = random.Random(SEED)
    order = []
    for triplet in range(TRIPLETS):
        modes = list(MODES)
        rng.shuffle(modes)
        for position, mode in enumerate(modes):
            order.append({"attempt_index": len(order), "triplet_index": triplet,
                          "position_in_triplet": position, "mode": mode, "label": LABELS[mode]})
    return order


def capture_helper():
    spec = importlib.util.spec_from_file_location("lime_reference_capture", CAPTURE_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_build(path):
    build = json.loads(Path(path).read_text())
    for source, digest in build["inputs"].items():
        require(sha(source) == digest, "Build source differs: " + source)
    require(sha(build["worker"]) == build["worker_sha256"], "Compiled capture worker differs")
    require(Path(build["library"]).is_file(), "Bound LimeSuite library is missing")
    return build


def verify_attempt(plan, expected):
    mode = expected["mode"]
    require(plan["mode"] == mode and plan["record_samples"] == RECORD_SAMPLES and
            plan["record_seconds"] == RECORD_SECONDS and plan["frequency_hz"] == 500_000_000 and
            plan["sample_rate_hz"] == 2_000_000 and plan["native_rx_gain_db"] == 30,
            "Attempt geometry or mode differs")
    require(plan["requested_native_tx_gain_db"] == (0 if mode == "noise" else 50) and
            plan["expected_native_tx_gain_readback_db"] == (None if mode == "noise" else 50) and
            plan["tone_peak_component"] == (.005 if mode == "tone" else 0),
            "Attempt fixed TX settings differ")


def prepare(output, *, build_manifest, analysis_sources, serial=SERIAL):
    """Freeze all fifteen plans and source identities; never open the radio."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty study directory; preparation cannot overwrite a study.")
    require(isinstance(analysis_sources, (list, tuple)) and analysis_sources,
            "At least the final analyzer source must be supplied before preparation.")
    sources = [Path(p).resolve() for p in analysis_sources]
    require(len(set(sources)) == len(sources) and all(p.is_file() for p in sources),
            "Analysis sources must be distinct existing files.")
    int(serial.removeprefix("0x").removeprefix("0X"), 16)
    build_manifest = Path(build_manifest).resolve()
    build = verify_build(build_manifest)
    scale_record = json.loads(SCALE_REFERENCE.read_text())
    require(scale_record["sample_quantization"]["scale"] == ANALYSIS_CONFIG["sample_scale"],
            "Independent prior scale reference differs")
    bound_paths = sorted({Path(__file__).resolve(), CAPTURE_HELPER, TEST_SOURCE, build_manifest,
                          Path(build["worker"]), Path(build["library"]), SCALE_REFERENCE, SCALE_SUMMARY, *sources,
                          *(Path(p) for p in build["inputs"])}, key=str)
    before = {str(path): sha(path) for path in bound_paths}
    helper = capture_helper()
    output.mkdir(parents=True, exist_ok=True)
    attempts = []
    for item in mode_order():
        relative = f"attempt-{item['attempt_index'] + 1:03d}-triplet-{item['triplet_index'] + 1:02d}-{item['mode']}"
        path = output / relative
        attempt = helper.prepare(path, build_manifest=build_manifest, serial=serial,
                                 mode=item["mode"], record_seconds=RECORD_SECONDS,
                                 tone_component=.005, native_tx_gain_db=50,
                                 expected_tx_gain_readback_db=50)
        verify_attempt(attempt, item)
        attempts.append({**item, "directory": relative, "plan_sha256": sha(path / "plan.json"),
                         "plan_digest_sha256": sha(path / "plan-digest.json"),
                         "tx_payload_sha256": sha(path / "tx.cfile")})
    require({str(path): sha(path) for path in bound_paths} == before, "Sources changed during preparation")
    snapshots = {}
    for path in bound_paths:
        if path.is_relative_to(REPO.parent):
            relative = Path("source_snapshots/workspace") / path.relative_to(REPO.parent)
        else:
            relative = Path("source_snapshots/external") / path.relative_to(path.anchor)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(sha(target) == before[str(path)], "Source changed while snapshotting: " + str(path))
        snapshots[str(path)] = {"relative": relative.as_posix(), "sha256": before[str(path)]}
    require({str(path): sha(path) for path in bound_paths} == before, "Sources changed before final freeze")
    plan = {
        "schema": "sdr-antenna-controls-plan-v1", "frozen_utc": now(),
        "scope": "Descriptive antenna-connected receiver/TX-state controls; no thermal-null, physical-power calibration or statistical distribution acceptance",
        "hardware_attempted_during_preparation": False,
        "randomization": {"seed": SEED, "generator": "Python random.Random MT19937; shuffle noise, txzero, tone independently for each triplet",
                          "triplets": TRIPLETS, "redraws": 0, "python_version": sys.version},
        "attempts": attempts, "attempt_count": 15, "serial": serial,
        "settings": {"frequency_hz": 500_000_000, "input_rate_hz": 2_000_000,
                     "native_rx_gain_db": 30, "requested_native_tx_gain_db": 50,
                     "expected_native_tx_gain_readback_db": 50, "tone_digital_frequency_hz": 100_000,
                     "tone_peak_component": .005, "record_samples": RECORD_SAMPLES,
                     "record_seconds": RECORD_SECONDS, "rx_bandwidth_hz": 1_500_000,
                     "tx_bandwidth_hz": 5_000_000, "rx_path": "LNAW", "active_tx_path": "TX2"},
        "record_role": "All five repeats per mode are descriptive evaluation records; no fitted calibration, thresholds, whitening, gain search, interval selection or rescaling from these data",
        "analysis_config": ANALYSIS_CONFIG,
        "analysis_scale_reference": {"path": str(SCALE_REFERENCE), "sha256": sha(SCALE_REFERENCE),
                                     "summary_path": str(SCALE_SUMMARY), "summary_sha256": sha(SCALE_SUMMARY),
                                     "interpretation": "Scale 500 fixed from the earlier independent ambient adapter validation; not fitted from these controls"},
        "grouping": "Triplets preserve their acquisition grouping; chronological mode order is randomized and frozen before hardware. Adjacent frames are not independent acquisition replicates.",
        "sample_targets": {"accepted_per_attempt": RECORD_SAMPLES, "accepted_if_all_complete": 60_000_000,
                           "accepted_seconds_if_all_complete": 30, "commanded_tx_payload_seconds_if_all_complete": 23},
        "capture_contract": "Use the unchanged finite reference helper; require success, exact accepted interval/count and confirmed cleanup. TX-disabled and TX-active zero-IQ are different states.",
        "initialization": "Each attempt uses documented LMS_Init/internal gain calibration and LPF tuning before the accepted interval; no explicit extra calibration/reset or firmware operation.",
        "failure_policy": "Stop on the first failed capture, missing receipt, incomplete interval or unconfirmed cleanup; preserve failed and unattempted preparations. No retries, shortened intervals, schedule extension or gain search.",
        "resource_policy": "One CPU capture at a time; numerical threads=1; no GPU or fine-campaign source/configuration/process modifications.",
        "per_attempt_timeouts_seconds": {"worker": 20, "parent": 30, "SIGTERM_cleanup": 5},
        "build_manifest": str(build_manifest), "build_manifest_sha256": sha(build_manifest),
        "analysis_sources": [str(path) for path in sources], "source_sha256": before,
        "source_snapshots": snapshots, "numerical_runtime": numerical_runtime(),
    }
    write_new(output / "plan.json", plan)
    write_new(output / "plan-digest.json", {"sha256": sha(output / "plan.json")})
    return plan


def load(output, *, expected_plan_sha256):
    """Authenticate the reviewed plan and every unlaunched prepared attempt."""
    output = Path(output).resolve()
    require(len(expected_plan_sha256) == 64 and sha(output / "plan.json") == expected_plan_sha256,
            "Reviewed plan identity differs")
    require(json.loads((output / "plan-digest.json").read_text())["sha256"] == expected_plan_sha256,
            "Frozen plan digest differs")
    plan = json.loads((output / "plan.json").read_text())
    require(plan["schema"] == "sdr-antenna-controls-plan-v1" and plan["attempt_count"] == 15,
            "Study schema or attempt count differs")
    expected = mode_order()
    require(len(plan["attempts"]) == len(expected) and
            [{key: item[key] for key in expected[0]} for item in plan["attempts"]] == expected,
            "Frozen randomized order differs")
    for source, digest in plan["source_sha256"].items():
        require(sha(source) == digest, "Frozen source differs: " + source)
        snapshot = plan["source_snapshots"][source]
        path = output / snapshot["relative"]
        require(path.resolve().is_relative_to(output) and sha(path) == snapshot["sha256"] == digest,
                "Frozen source snapshot differs: " + source)
    require(numerical_runtime() == plan["numerical_runtime"], "Prepared numerical runtime differs")
    require(sha(plan["build_manifest"]) == plan["build_manifest_sha256"], "Build manifest differs")
    verify_build(plan["build_manifest"])
    helper = capture_helper()
    for item in plan["attempts"]:
        path = output / item["directory"]
        require(path.resolve().is_relative_to(output) and path.parent == output, "Attempt path escapes study")
        require(sha(path / "plan.json") == item["plan_sha256"] and
                sha(path / "plan-digest.json") == item["plan_digest_sha256"] and
                sha(path / "tx.cfile") == item["tx_payload_sha256"], "Prepared attempt identity differs")
        require({p.name for p in path.iterdir()} == {"plan.json", "plan-digest.json", "tx.cfile"},
                "Attempt already launched or contains unexpected artifacts")
        verify_attempt(helper.load(path), item)
    return plan


def run(output, *, expected_plan_sha256, hardware_authorized=False,
        transmit=False, rf_confined_authorized=False):
    """Execute exactly the frozen sequence once, with explicit launch flags."""
    require(hardware_authorized and transmit and rf_confined_authorized,
            "Run requires --hardware-authorized --transmit --rf-confined-authorized.")
    output = Path(output).resolve()
    for name in ("launch.json", "run-events.jsonl", "run-receipt.json"):
        require(not (output / name).exists(), "Study already launched; no resume or retry is supported")
    plan = load(output, expected_plan_sha256=expected_plan_sha256)
    # Exclusive creation also prevents two reviewed invocations racing to launch.
    write_new(output / "launch.json", {"schema": "sdr-antenna-controls-launch-v1", "started_utc": now(),
              "plan_sha256": expected_plan_sha256, "hardware_attempted": True,
              "authorization": "Explicit confined-room hardware and TX flags; existing user authorization",
              "attempt_limit": 15})
    states = [{"attempt_index": item["attempt_index"], "directory": item["directory"],
               "mode": item["mode"], "label": item["label"], "status": "unattempted"}
              for item in plan["attempts"]]
    failure = None
    try:
        helper = capture_helper()
        with (output / "run-events.jsonl").open("x") as events:
            def event(value):
                events.write(json.dumps({"utc": now(), **value}, sort_keys=True, allow_nan=False) + "\n")
                events.flush()
                os.fsync(events.fileno())
            for item, state in zip(plan["attempts"], states):
                state["status"] = "started"
                event({"event": "attempt_started", "attempt_index": item["attempt_index"], "mode": item["mode"]})
                path = output / item["directory"]
                try:
                    result = helper.capture(path, hardware_authorized=True, transmit=True, rf_confined_authorized=True)
                    receipt_path = path / "receipt.json"
                    require(receipt_path.is_file(), "Capture returned without a durable receipt")
                    recorded = json.loads(receipt_path.read_text())
                    require(all(recorded.get(key) is True and result.get(key) is True for key in
                                ("success", "cleanup_confirmed", "accepted_interval_confirmed", "hardware_attempted")),
                            "Capture failed, interval incomplete or cleanup unconfirmed")
                    require(recorded.get("plan_sha256") == item["plan_sha256"] and recorded.get("mode") == item["mode"],
                            "Capture receipt identity differs")
                    require((path / "accepted.cfile").stat().st_size == 8 * RECORD_SAMPLES,
                            "Accepted capture has an incomplete sample count")
                    for source, digest in plan["source_sha256"].items():
                        require(sha(source) == digest, "Frozen source changed during capture: " + source)
                    state.update(status="complete", receipt_sha256=sha(receipt_path), accepted_samples=RECORD_SAMPLES,
                                 cleanup_confirmed=True)
                    event({"event": "attempt_complete", **state})
                except BaseException as error:
                    state["status"] = "failed"
                    state["error"] = repr(error)
                    if (path / "receipt.json").is_file():
                        state["receipt_sha256"] = sha(path / "receipt.json")
                    event({"event": "attempt_failed", **state})
                    raise
    except BaseException as error:
        failure = error
    finally:
        complete = sum(item["status"] == "complete" for item in states)
        receipt = {"schema": "sdr-antenna-controls-run-receipt-v1", "completed_utc": now(),
                   "plan_sha256": expected_plan_sha256, "scope": plan["scope"],
                   "status": "complete" if complete == 15 and failure is None else "stopped_failed",
                   "complete_attempts": complete, "attempted": sum(item["status"] != "unattempted" for item in states),
                   "remaining_unattempted": sum(item["status"] == "unattempted" for item in states),
                   "attempts": states, "error": repr(failure) if failure is not None else None,
                   "automatic_retries": 0, "schedule_extended": False, "gain_search_performed": False}
        write_new(output / "run-receipt.json", receipt)
    if failure is not None:
        raise RuntimeError("Antenna-control study stopped; preserve all evidence, no retries: " + repr(failure)) from failure
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--analysis-source", type=Path, action="append")
    parser.add_argument("--serial", default=SERIAL)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--hardware-authorized", action="store_true")
    parser.add_argument("--transmit", action="store_true")
    parser.add_argument("--rf-confined-authorized", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.build_manifest is None or not args.analysis_source:
            parser.error("prepare requires --build-manifest and at least one --analysis-source")
        result = prepare(args.output, build_manifest=args.build_manifest,
                         analysis_sources=args.analysis_source, serial=args.serial)
        print(json.dumps({"prepared": True, "hardware_attempted": False, "attempts": len(result["attempts"]),
                          "plan_sha256": sha(args.output / "plan.json"),
                          "order": [item["mode"] for item in result["attempts"]]}, indent=2))
    else:
        if args.expected_plan_sha256 is None:
            parser.error("run requires the independently reviewed --expected-plan-sha256")
        result = run(args.output, expected_plan_sha256=args.expected_plan_sha256,
                     hardware_authorized=args.hardware_authorized, transmit=args.transmit,
                     rf_confined_authorized=args.rf_confined_authorized)
        print(json.dumps({"status": result["status"], "complete_attempts": result["complete_attempts"]}, indent=2))


if __name__ == "__main__":
    main()
