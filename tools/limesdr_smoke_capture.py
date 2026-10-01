#!/usr/bin/env python3
"""Prepare or explicitly launch one confined-room, uncalibrated LimeSDR smoke.

Preparation does not open a device. Capture requires both explicit launch flags.
This helper does not supply measured dBm, an OTA calibration or an observing policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(output, *, build_manifest, serial, frequency_hz, native_tx_gain_db, expected_tx_gain_readback_db, rate_hz=2000000., native_rx_gain_db=30):
    if not math.isfinite(frequency_hz) or not 30000000 <= frequency_hz <= 1900000000:
        raise ValueError("frequency outside Mini BAND2 smoke range")
    if rate_hz != 2000000. or type(native_rx_gain_db) is not int or not 0 <= native_rx_gain_db <= 50:
        raise ValueError("smoke uses 2 MHz sample rate and native RX gain 0..50")
    if type(native_tx_gain_db) is not int or not 0 <= native_tx_gain_db <= 50:
        raise ValueError("explicit requested TX gain must be an integer from 0 through 50")
    if type(expected_tx_gain_readback_db) is not int or not 0 <= expected_tx_gain_readback_db <= 50:
        raise ValueError("explicit expected TX readback must be an integer from 0 through 50")
    int(serial.removeprefix("0x").removeprefix("0X"), 16)
    manifest_path = Path(build_manifest).resolve()
    build = json.loads(manifest_path.read_text())
    for path, expected in build["inputs"].items():
        if sha(path) != expected:
            raise ValueError(f"build input changed: {path}")
    worker = Path(build["worker"]).resolve()
    library = Path(build["library"]).resolve()
    if sha(worker) != build["worker_sha256"]:
        raise ValueError("worker binary differs from build receipt")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new empty attempt directory")
    output.mkdir(parents=True, exist_ok=True)
    guard = np.zeros(int(rate_hz*.05), dtype=np.complex64)
    t = np.arange(int(rate_hz*.05))/rate_hz
    # A 50 ms 100 kHz complex tone, tapered over 1 ms at both edges.
    tone = .005*np.exp(2j*np.pi*100000*t)
    ramp = int(rate_hz*.001)
    tone[:ramp] *= np.sin(np.linspace(0, math.pi/2, ramp))**2
    tone[-ramp:] *= np.sin(np.linspace(math.pi/2, 0, ramp))**2
    waveform = np.concatenate([guard, tone.astype(np.complex64), guard])
    tx_path = output/"tx.cfile"
    with tx_path.open("xb") as stream:
        stream.write(waveform.tobytes())
    command = [str(worker), "--serial", serial, "--frequency-hz", str(frequency_hz),
               "--sample-rate-hz", str(rate_hz), "--rx-bandwidth-hz", "1500000", "--tx-bandwidth-hz", "5000000",
               "--native-rx-gain-db", str(native_rx_gain_db), "--native-tx-gain-db", str(native_tx_gain_db),
               "--expected-tx-gain-readback-db", str(expected_tx_gain_readback_db), "--tx-samples", str(waveform.size),
               "--tx-file", str(tx_path), "--rx-file", str(output/"rx.cfile"),
               "--status-file", str(output/"worker-status.json"),
               "--startup-file", str(output/"startup.cfile"), "--chunks-file", str(output/"rx-chunks.jsonl"),
               "--tx-status-file", str(output/"tx-status.jsonl")]
    plan = {"schema": "lime-confined-smoke-plan-v5", "frozen_utc": now(),
            "scope": "uncalibrated single-RF-path tone transport, no measured dBm or scientific calibration",
            "hardware_attempted": False,
            "stream_start_protocol": "Select TX path and start empty TX then RX before settling/qualification; qualify RX before any finite tone submission; no post-ready RF or stream-start calls",
            "tx_host_empty_counter_rule": "Record each pre/submission/post-capture delta; host FIFO empty waits alone do not establish RF loss. Require full finite API acceptance, active TX, zero TX overrun/drop, strict qualified RX and confirmed cleanup",
            "library": str(library), "command": command,
            "serial": serial, "frequency_hz": frequency_hz, "sample_rate_hz": rate_hz,
            "native_tx_gain_db": native_tx_gain_db, "requested_native_tx_gain_db": native_tx_gain_db,
            "expected_native_tx_gain_readback_db": expected_tx_gain_readback_db, "native_rx_gain_db": native_rx_gain_db,
            "rx_lpf_hz": 1500000., "tx_lpf_hz": 5000000., "tx_path": "BAND2", "rx_path": "LNAW",
            "initialization_includes_internal_tx_gain_calibration": True,
            "tone_offset_hz": 100000., "tone_peak_component": .005,
            "tone_seconds": .05, "zero_prefix_seconds": .05, "zero_tail_seconds": .05,
            "submitted_samples": int(waveform.size),
            "rx_startup_rule": {"settling_received_sample_seconds": .25, "required_clean_seconds": .1,
                                "wall_budget_seconds": 2, "received_sample_budget_seconds": 2,
                                "after_readiness": "Any timestamp gap or stream error is fatal",
                                "raw_retention": "All pre-ready samples in startup.cfile; accepted candidate duplicated at start of rx.cfile with recorded offset; chunk receipts preserve every read and counter delta"}, "measured_power_dbm": None,
            "power_calibrated": False, "explicit_rf_calibration_run": False,
            "explicit_rf_calibration_meaning": "No explicit LMS_Calibrate call; initialization gain calibration and LPF tuning run internally",
            "lpf_configuration_includes_internal_filter_tuning": True,
            "inputs": {**build["inputs"], str(worker): sha(worker), str(library): sha(library),
                       str(manifest_path): sha(manifest_path), str(Path(__file__).resolve()): sha(__file__),
                       str(tx_path): sha(tx_path)}}
    write_new(output/"plan.json", plan)
    write_new(output/"plan-digest.json", {"sha256": sha(output/"plan.json")})
    return plan


def load(output):
    output = Path(output).resolve()
    if sha(output/"plan.json") != json.loads((output/"plan-digest.json").read_text())["sha256"]:
        raise ValueError("frozen plan changed")
    plan = json.loads((output/"plan.json").read_text())
    for path, expected in plan["inputs"].items():
        if sha(path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    return plan


def capture(output, *, transmit=False, rf_confined_authorized=False):
    if not transmit or not rf_confined_authorized:
        raise ValueError("launch needs explicit --transmit and --rf-confined-authorized")
    output = Path(output).resolve()
    plan = load(output)
    write_new(output/"launch.json", {"started_utc": now(), "hardware_attempted": True,
              "plan_sha256": sha(output/"plan.json"), "authorization": "explicit confined-room smoke"})
    env = os.environ.copy()
    env["LD_PRELOAD"] = plan["library"]
    process = None
    requested_stop = False
    timeout = False
    forced_kill = False
    previous = {}
    def interrupt(signum, _frame):
        nonlocal requested_stop
        requested_stop = True
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, interrupt)
    receipt = {"schema": "lime-confined-smoke-receipt-v5", "hardware_attempted": True,
               "plan_sha256": sha(output/"plan.json"), "scope": plan["scope"]}
    try:
        with (output/"worker.log").open("x") as log:
            if requested_stop:
                raise InterruptedError("stop requested before child launch")
            process = subprocess.Popen(plan["command"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            started = time.monotonic()
            while process.poll() is None:
                if requested_stop or time.monotonic()-started > 25:
                    timeout = not requested_stop
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        forced_kill = True
                        process.kill()
                        process.wait()
                    break
                time.sleep(.05)
            receipt["returncode"] = process.returncode
    except Exception as error:
        receipt["launch_error"] = repr(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            # Also clean up unexpected parent-side exceptions, not only timeout.
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                forced_kill = True
                process.kill()
                process.wait()
        if process is not None:
            receipt["returncode"] = process.returncode
        for signum, handler in previous.items(): signal.signal(signum, handler)
        receipt.update(completed_utc=now(), timeout=timeout, requested_stop=requested_stop,
                       forced_kill=forced_kill, cleanup_confirmed=False)
        status_path = output/"worker-status.json"
        if status_path.exists() and status_path.stat().st_size:
            try:
                report = json.loads(status_path.read_text())
                receipt["worker_report"] = report
                cleanup = report.get("cleanup", {})
                receipt["cleanup_confirmed"] = not forced_kill and all(
                    cleanup.get(key) == 0 for key in ("antenna_off_return", "gain_zero_return", "disable_tx_return", "antenna_off_readback", "close_return"))
            except (ValueError, TypeError) as error:
                receipt["status_error"] = str(error)
        receipt["artifacts"] = {path.name: sha(path) for path in output.iterdir() if path.is_file() and path.name != "receipt.json"}
        receipt["success"] = bool(receipt.get("returncode") == 0 and not timeout and not requested_stop
                                  and not receipt.get("launch_error") and receipt["cleanup_confirmed"] and receipt.get("worker_report", {}).get("success"))
        write_new(output/"receipt.json", receipt)
    if not receipt["success"]:
        raise RuntimeError("smoke failed; preserve this attempt and inspect receipt/cleanup before another run")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--frequency-hz", type=float)
    parser.add_argument("--native-rx-gain-db", type=int, default=30)
    parser.add_argument("--native-tx-gain-db", type=int)
    parser.add_argument("--expected-tx-gain-readback-db", type=int)
    parser.add_argument("--transmit", action="store_true")
    parser.add_argument("--rf-confined-authorized", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.build_manifest is None or args.serial is None or args.frequency_hz is None or args.expected_tx_gain_readback_db is None or args.native_tx_gain_db is None:
            parser.error("prepare needs build manifest, explicit serial, frequency, requested TX gain and expected TX gain readback")
        result = prepare(args.output, build_manifest=args.build_manifest, serial=args.serial,
                         frequency_hz=args.frequency_hz, native_tx_gain_db=args.native_tx_gain_db, expected_tx_gain_readback_db=args.expected_tx_gain_readback_db,
                         native_rx_gain_db=args.native_rx_gain_db)
    else:
        result = capture(args.output, transmit=args.transmit, rf_confined_authorized=args.rf_confined_authorized)
    print(json.dumps({"scope": result["scope"], "hardware_attempted": result["hardware_attempted"],
                      "success": result.get("success")}, indent=2))


if __name__ == "__main__":
    main()
