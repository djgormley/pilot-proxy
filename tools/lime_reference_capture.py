#!/usr/bin/env python3
"""Prepare or explicitly launch one finite noise/TX-zero/tone reference capture.

Preparation does not open a device. Hardware capture requires explicit launch flags.
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


def prepare(output, *, build_manifest, serial, mode, record_seconds=2., tone_component=.005,
            native_tx_gain_db=50, expected_tx_gain_readback_db=50):
    """Prepare one explicit attempt; this never opens the radio."""
    if mode not in {"noise", "txzero", "tone"}:
        raise ValueError("mode must be noise, txzero or tone")
    if not math.isfinite(record_seconds) or not .02 <= record_seconds <= 4:
        raise ValueError("record duration must be 20 ms through 4 s")
    rate_hz, frequency_hz, native_rx_gain_db = 2000000., 500000000., 30
    record_samples = round(record_seconds*rate_hz)
    if abs(record_samples/rate_hz-record_seconds) > 1e-12:
        raise ValueError("record duration must resolve to an integer sample count")
    if not math.isfinite(tone_component) or not 0 < tone_component <= .01:
        raise ValueError("tone component must be positive and at most 0.01")
    if (type(native_tx_gain_db) is not int or not 0 <= native_tx_gain_db <= 50 or
            type(expected_tx_gain_readback_db) is not int or not 0 <= expected_tx_gain_readback_db <= 50):
        raise ValueError("requested/expected native TX gains must be explicit integers 0..50")
    int(serial.removeprefix("0x").removeprefix("0X"), 16)
    manifest_path = Path(build_manifest).resolve()
    build = json.loads(manifest_path.read_text())
    for path, expected in build["inputs"].items():
        if sha(path) != expected:
            raise ValueError(f"build input changed: {path}")
    worker, library = Path(build["worker"]).resolve(), Path(build["library"]).resolve()
    if sha(worker) != build["worker_sha256"]:
        raise ValueError("worker differs from build receipt")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new empty attempt directory")
    output.mkdir(parents=True, exist_ok=True)
    if mode == "noise":
        waveform = np.empty(0, dtype=np.complex64)
    else:
        waveform = np.zeros(record_samples+600000, dtype=np.complex64)
        if mode == "tone":
            n = record_samples+400000
            tone = tone_component*np.exp(2j*np.pi*100000*np.arange(n)/rate_hz)
            ramp = 2000
            tone[:ramp] *= np.sin(np.linspace(0, math.pi/2, ramp))**2
            tone[-ramp:] *= np.sin(np.linspace(math.pi/2, 0, ramp))**2
            waveform[100000:100000+n] = tone.astype(np.complex64)
    tx_path = output/"tx.cfile"
    with tx_path.open("xb") as stream:
        stream.write(waveform.tobytes())
    requested = 0 if mode == "noise" else native_tx_gain_db
    expected_gain = 0 if mode == "noise" else expected_tx_gain_readback_db
    command = [str(worker), "--serial", serial, "--mode", mode, "--record-samples", str(record_samples),
               "--accepted-file", str(output/"accepted.cfile"), "--schedule-file", str(output/"schedule.json"), "--frequency-hz", str(frequency_hz),
               "--sample-rate-hz", str(rate_hz), "--rx-bandwidth-hz", "1500000", "--tx-bandwidth-hz", "5000000",
               "--native-rx-gain-db", str(native_rx_gain_db), "--native-tx-gain-db", str(requested),
               "--expected-tx-gain-readback-db", str(expected_gain), "--tx-samples", str(waveform.size),
               "--tx-file", str(tx_path), "--rx-file", str(output/"rx.cfile"),
               "--status-file", str(output/"worker-status.json"), "--startup-file", str(output/"startup.cfile"),
               "--chunks-file", str(output/"rx-chunks.jsonl"), "--tx-status-file", str(output/"tx-status.jsonl")]
    plan = {"schema": "lime-reference-plan-v1", "frozen_utc": now(), "mode": mode,
            "scope": "Fixed-window reference capture; no power calibration, IID guarantee or inference that TX-enabled zero IQ equals TX-disabled noise",
            "hardware_attempted": False, "library": str(library), "command": command, "serial": serial,
            "frequency_hz": frequency_hz, "sample_rate_hz": rate_hz, "native_rx_gain_db": native_rx_gain_db,
            "requested_native_tx_gain_db": requested, "expected_native_tx_gain_readback_db": expected_gain if mode != "noise" else None,
            "record_seconds": record_samples/rate_hz, "record_samples": record_samples,
            "tone_frequency_hz": 100000., "tone_peak_component": tone_component if mode == "tone" else 0.,
            "payload_samples": int(waveform.size), "payload_seconds": waveform.size/rate_hz,
            "tone_payload_interval_seconds": [.05, record_samples/rate_hz+.25] if mode == "tone" else None,
            "tone_edge_taper_seconds": .001 if mode == "tone" else None,
            "accepted_interval_rule": "noise: schedule_base + 0.2 s; txzero/tone: tx_start + 0.15 s; exactly record_samples; schedule_base is the latest complete qualified RX timestamp at scheduling",
            "tx_start_rule": "txzero/tone: schedule_base + 0.2 s; noise: TX never set up or started",
            "tone_steady_margin_seconds": .1 if mode == "tone" else None,
            "rx_startup_rule": {"settling_received_sample_seconds": .25, "required_clean_seconds": .1,
                                "wall_budget_seconds": 2, "received_sample_budget_seconds": 2},
            "mode_meaning": {"noise": "TX disabled by API and register/path readbacks; RX only after initialization",
                             "txzero": "TX chain/stream active at the requested gain with all-zero IQ payload",
                             "tone": "TX chain/stream active with additive deterministic complex tone"}[mode],
            "initialization_includes_internal_tx_gain_calibration": True,
            "initialization_scope": "Internal initialization calibration precedes the baseline; TX-disabled claims apply to the qualified accepted noise interval",
            "inputs": {**build["inputs"], str(worker): sha(worker), str(library): sha(library),
                       str(manifest_path): sha(manifest_path), str(Path(__file__).resolve()): sha(__file__), str(tx_path): sha(tx_path)}}
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


def capture(output, *, hardware_authorized=False, transmit=False, rf_confined_authorized=False):
    if not hardware_authorized or not rf_confined_authorized:
        raise ValueError("launch needs explicit --hardware-authorized and --rf-confined-authorized; initialization can calibrate internally")
    output = Path(output).resolve()
    plan = load(output)
    if plan["mode"] != "noise" and not transmit:
        raise ValueError("TX-enabled modes also require explicit --transmit")
    write_new(output/"launch.json", {"started_utc": now(), "hardware_attempted": True,
              "plan_sha256": sha(output/"plan.json"), "authorization": "explicit confined-room reference capture"})
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
    receipt = {"schema": "lime-reference-receipt-v1", "hardware_attempted": True,
               "plan_sha256": sha(output/"plan.json"), "scope": plan["scope"], "mode": plan["mode"]}
    try:
        with (output/"worker.log").open("x") as log:
            if requested_stop:
                raise InterruptedError("stop requested before child launch")
            process = subprocess.Popen(plan["command"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            started = time.monotonic()
            while process.poll() is None:
                if requested_stop or time.monotonic()-started > 30:
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
                    cleanup.get(key) == 0 for key in ("antenna_off_return", "gain_zero_return", "disable_tx_return", "antenna_off_readback", "close_return")) and all(
                    cleanup.get(key) in (-999, 0) for key in ("stop_tx_return", "stop_rx_return", "destroy_tx_return", "destroy_rx_return", "disable_rx_return"))
            except (ValueError, TypeError) as error:
                receipt["status_error"] = str(error)
        report = receipt.get("worker_report", {})
        receipt["accepted_interval_confirmed"] = bool(report.get("accepted_interval_passed") and
            report.get("mode") == plan["mode"] and report.get("accepted_samples") == plan["record_samples"] and
            (output/"accepted.cfile").is_file() and (output/"accepted.cfile").stat().st_size == 8*plan["record_samples"])
        receipt["artifacts"] = {path.name: sha(path) for path in output.iterdir() if path.is_file() and path.name != "receipt.json"}
        receipt["success"] = bool(receipt.get("returncode") == 0 and not timeout and not requested_stop
                                  and not receipt.get("launch_error") and receipt["cleanup_confirmed"] and receipt["accepted_interval_confirmed"] and receipt.get("worker_report", {}).get("success"))
        write_new(output/"receipt.json", receipt)
    if not receipt["success"]:
        raise RuntimeError("reference capture failed; preserve this attempt and inspect receipt/cleanup before another run")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--mode", choices=("noise", "txzero", "tone"))
    parser.add_argument("--record-seconds", type=float, default=2.)
    parser.add_argument("--tone-component", type=float, default=.005)
    parser.add_argument("--native-tx-gain-db", type=int, default=50)
    parser.add_argument("--expected-tx-gain-readback-db", type=int, default=50)
    parser.add_argument("--hardware-authorized", action="store_true")
    parser.add_argument("--transmit", action="store_true")
    parser.add_argument("--rf-confined-authorized", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.build_manifest is None or args.serial is None or args.mode is None:
            parser.error("prepare requires build manifest, explicit serial and mode")
        result = prepare(args.output, build_manifest=args.build_manifest, serial=args.serial, mode=args.mode,
                         record_seconds=args.record_seconds, tone_component=args.tone_component,
                         native_tx_gain_db=args.native_tx_gain_db, expected_tx_gain_readback_db=args.expected_tx_gain_readback_db)
    else:
        result = capture(args.output, hardware_authorized=args.hardware_authorized, transmit=args.transmit,
                         rf_confined_authorized=args.rf_confined_authorized)
    print(json.dumps({"scope": result["scope"], "mode": result["mode"],
                      "hardware_attempted": result["hardware_attempted"], "success": result.get("success")}, indent=2))


if __name__ == "__main__":
    main()
