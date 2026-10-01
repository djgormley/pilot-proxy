#!/usr/bin/env python3
"""Finite RX-only file-backed live handoff; no TX payload or RF inference."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import select
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import scipy

from pilot_proxy.integration.sdr_upgrade_adapter import DigitalAdapterConfig, adapt_record
from pilot_proxy.integration.sdr_upgrade_streaming import StreamingDigitalAdapter

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CONFIG = DigitalAdapterConfig(2000000, 100000., 0, 500.)
BLOCK = 8192
QUEUE_CHUNKS = 32
SOURCES = [Path(__file__).resolve(), ROOT / "tools/lime_reference_capture.py",
           ROOT / "tools/lime_reference_worker.cpp", ROOT / "src/pilot_proxy/detector_reference.py",
           ROOT / "src/pilot_proxy/integration/sdr_upgrade_adapter.py",
           ROOT / "src/pilot_proxy/integration/sdr_upgrade_streaming.py",
           ROOT / "tests/testbench/test_sdr_live_handoff.py", ROOT / "docs/SDR_LIVE_HANDOFF.md"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def load_capture():
    spec = importlib.util.spec_from_file_location("qualified_lime_capture", SOURCES[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame_record(frame):
    return {"segment_id": frame["segment_id"], "frame_index": frame["frame_index_in_segment"],
            "input_origin": frame["segment_first_input_index"],
            "sample_center_time_numerator": frame["sample_center_time_numerator"],
            "sample_center_time_denominator_hz": frame["sample_center_time_denominator_hz"],
            "packed_sha256": hashlib.sha256(frame["packed_frame"].astype("i1").tobytes()).hexdigest(),
            "projections_sha256": hashlib.sha256(frame["projections_i32"].astype("<i4").tobytes()).hexdigest(),
            "float_sha256": hashlib.sha256(frame["resampled_frame"].astype("<c16").tobytes()).hexdigest(),
            "term_power_sums": frame["term_power_sums"].tolist(),
            "ratio_status": frame["ratio_status"],
            "coarse_ratio": frame["coarse_ratio"] if np.isfinite(frame["coarse_ratio"]) else None,
            "saturated_component_count": frame["saturated_component_count"],
            "minimum_half_step_distance": frame["minimum_unclipped_quantizer_half_step_distance"]}


def committed_line(stream):
    """A partial JSON line is unpublished; reread it after its newline arrives."""
    offset = stream.tell()
    raw = stream.readline(65537)
    if len(raw) > 65536:
        raise ValueError("chunk publication exceeds fixed line bound")
    if not raw.endswith(b"\n"):
        stream.seek(offset)
        return None
    return json.loads(raw)


def accepted_span(row, schedule, expected_offset):
    """Validate exact native committed data coordinates before handing off IQ."""
    count = row["accepted_samples_from_chunk"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("invalid accepted chunk count")
    if count == 0:
        return None
    begin, end = schedule["accepted_first_timestamp"], schedule["accepted_stop_timestamp_exclusive"]
    received = row["received_count"]
    lo, hi = max(row["timestamp"], begin), min(row["timestamp"] + received, end)
    valid = (row["phase"] == "qualified" and 0 < received <= 16384
             and hi - lo == count and row["accepted_file_sample_offset"] == expected_offset
             and lo == begin + expected_offset and row["expected_timestamp_known"]
             and row["timestamp"] == row["expected_timestamp"] and not row["timestamp_gap"]
             and row["status_return"] == 0 and row["stream_active"] and row["finite"]
             and row["underrun_delta"] == row["overrun_delta"] == row["dropped_delta"] == 0
             and row["peak_component"] < .98 and row["rms"] < .35)
    if not valid:
        raise ValueError("committed accepted chunk fails native qualification/continuity")
    return lo, count


def published_bytes(path, sample_offset, count):
    """Return None for a short span; after newline publication this is corruption."""
    with Path(path).open("rb") as f:
        f.seek(sample_offset * 8)
        raw = f.read(count * 8)
    return raw if len(raw) == count * 8 else None


def receipt_accepted(receipt, count):
    report = receipt.get("worker_report", {})
    return bool(receipt.get("success") and receipt.get("cleanup_confirmed")
                and receipt.get("accepted_interval_confirmed") and report.get("accepted_samples") == count
                and report.get("mode") == "noise" and not report.get("tx_attempted")
                and not report.get("payload_submission_attempted") and report.get("tx_samples_sent") == 0
                and report.get("tx_disabled_register_readbacks", {}).get("before")
                == report.get("tx_disabled_register_readbacks", {}).get("after")
                == {"TXEN_A": 0, "EN_TXTSP": 0, "EN_G_TRF": 0, "PD_TXPAD_TRF": 1})


def prepare(out, kind, build_manifest, serial):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("study destination must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    seconds, count = (2., 1) if kind == "smoke" else (4., 3)
    capture = load_capture()
    attempts = []
    for i in range(count):
        directory = f"attempt-{i+1:03d}"
        capture.prepare(out / directory, build_manifest=build_manifest, serial=serial,
                        mode="noise", record_seconds=seconds)
        attempts.append({"directory": directory, "plan_sha256": sha(out / directory / "plan.json")})
    identities = {str(p): sha(p) for p in SOURCES}
    for p in SOURCES:
        copy = out / "source_snapshots" / p.relative_to(WORKSPACE)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(p.read_bytes())
    reference = WORKSPACE / "results/sdr_adapter_validation_2026-09-09/adapter_metadata.json"
    plan = {"schema": "sdr-live-handoff-plan-v1", "kind": kind, "attempts": attempts,
            "record_seconds_each": seconds, "record_samples_each": int(seconds*2000000),
            "config": asdict(CONFIG), "source_sha256": identities,
            "sample_scale_origin": {"path": str(reference), "sha256": sha(reference)},
            "queue_capacity_chunks": QUEUE_CHUNKS, "maximum_chunk_samples": BLOCK,
            "queue_payload_bound_bytes": QUEUE_CHUNKS*BLOCK*8,
            "spool": "Native regular files grow while RX runs. JSON newline publishes exact accepted byte spans after native writes. File size can lead validation publication.",
            "timing": "Local monotonic observation/queue/service times only; no USB-call, RF arrival or hardware-clock latency inferred.",
            "acceptance": "All fixed attempts: native transport/cleanup accepted, every accepted sample handed off once, complete frame parity with offline streaming and batch, actual frame production observed before native exit and before complete accepted file existed; queue never overflows. Backlog/service timing descriptive, not a deadline qualification.",
            "failure_policy": "Stop at first failure; preserve partial outputs and unattempted plans; no retries, shortening, gain/scale search or threshold fitting.",
            "scope": "Antenna ambient, TX-disabled accepted interval, file-backed live CPU host handoff only; no thermal-null/RF-calibration/live-Pathfinder qualification.",
            "initialization": "Unchanged worker LMS_Init/internal gain calibration and LPF tuning precede accepted RX; no TX stream or payload in mode=noise.",
            "numpy": np.__version__, "scipy": scipy.__version__,
            "frozen_monotonic_ns": time.monotonic_ns(), "frozen_unix_ns": time.time_ns()}
    save(out / "plan.json", plan)
    save(out / "plan-digest.json", {"sha256": sha(out / "plan.json")})
    return plan


def verify_plan(out):
    plan = json.loads((out / "plan.json").read_text())
    if sha(out / "plan.json") != json.loads((out / "plan-digest.json").read_text())["sha256"]:
        raise ValueError("study plan changed")
    if plan["config"] != asdict(CONFIG) or plan["numpy"] != np.__version__ or plan["scipy"] != scipy.__version__:
        raise ValueError("frozen processing configuration/runtime changed")
    for p, expected in plan["source_sha256"].items():
        if sha(p) != expected:
            raise ValueError(f"frozen source changed: {p}")
    for attempt in plan["attempts"]:
        path = out / attempt["directory"]
        if sha(path / "plan.json") != attempt["plan_sha256"]:
            raise ValueError("frozen capture plan changed")
        load_capture().load(path)
    return plan


class WorkerIdentity:
    """Observe the native child through pidfd; never open or query radio hardware."""
    def __init__(self, helper_pid, executable):
        self.helper_pid, self.executable = helper_pid, Path(executable).resolve()
        self.pid = self.fd = self.first_exit_observed_ns = None
        self.lock = threading.Lock()

    def observe(self):
        with self.lock:
            return self._observe()

    def _observe(self):
        if self.fd is None:
            children = Path(f"/proc/{self.helper_pid}/task/{self.helper_pid}/children")
            if children.exists():
                for value in children.read_text().split():
                    pid = int(value)
                    try:
                        if Path(f"/proc/{pid}/exe").resolve() == self.executable:
                            self.fd, self.pid = os.pidfd_open(pid), pid
                            break
                    except (OSError, FileNotFoundError):
                        pass
        alive = self.fd is not None and not select.select([self.fd], [], [], 0)[0]
        observed = time.monotonic_ns()
        if self.fd is not None and not alive and self.first_exit_observed_ns is None:
            self.first_exit_observed_ns = observed
        return {"native_worker_pid": self.pid, "native_worker_alive_observed": alive,
                "worker_liveness_observed_monotonic_ns": observed}

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class CaptureSupervisor:
    """Always stop/reap owned processes, including setup faults and user signals."""
    def __init__(self, folder, executable):
        self.folder, self.executable = Path(folder), executable
        self.stop = threading.Event()
        self.process = self.identity = self.log = None
        self.previous = {}
        self.closed = self.interrupted = self.forced = False

    def interrupt(self, _signum, _frame):
        self.interrupted = True
        self.stop.set()
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)

    def __enter__(self):
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self.previous[signum] = signal.signal(signum, self.interrupt)
            self.log = (self.folder / "handoff-capture.log").open("x")
            if self.interrupted:
                raise InterruptedError("stopped before helper launch")
            self.process = subprocess.Popen(
                [sys.executable, str(SOURCES[1]), "capture", "--output", str(self.folder),
                 "--hardware-authorized", "--rf-confined-authorized"],
                stdout=self.log, stderr=subprocess.STDOUT, env=os.environ.copy())
            self.identity = WorkerIdentity(self.process.pid, self.executable)
            return self
        except BaseException:
            self.close(abort=True)
            raise

    def close(self, *, abort=False):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process is not None:
                if (abort or self.interrupted) and self.process.poll() is None:
                    self.process.send_signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.forced = True
                    # The qualified helper normally stops/reaps its native child
                    # within five seconds. Exceptional fallback must not orphan it.
                    if self.identity is not None:
                        self.identity.observe()
                        if self.identity.fd is not None:
                            try:
                                signal.pidfd_send_signal(self.identity.fd, signal.SIGTERM)
                                if not select.select([self.identity.fd], [], [], 5)[0]:
                                    signal.pidfd_send_signal(self.identity.fd, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    self.process.kill()
                    self.process.wait()
        finally:
            for signum, handler in self.previous.items():
                signal.signal(signum, handler)
            if self.log is not None:
                self.log.close()
            save(self.folder / "supervisor.json", {"interrupted": self.interrupted,
                 "abort_requested": abort, "forced_kill_cleanup_unconfirmed": self.forced,
                 "helper_returncode": self.process.returncode if self.process is not None else None})

    def __exit__(self, exc_type, _value, _traceback):
        self.close(abort=exc_type is not None)
        if self.identity is not None:
            self.identity.close()


def _live_attempt(folder, plan, supervisor):
    """Capture helper owns native cleanup; consumer can abort it via SIGTERM."""
    folder = Path(folder)
    adapter = StreamingDigitalAdapter(CONFIG)
    fifo = queue.Queue(maxsize=QUEUE_CHUNKS)
    done, stop = threading.Event(), supervisor.stop
    errors, frames, service = [], [], []
    observed = {"committed_samples": 0, "enqueued_samples": 0, "processed_samples": 0,
                "max_queue_chunks_observed": 0, "max_visible_file_backlog_samples": 0,
                "max_committed_backlog_samples": 0, "input_sha256": None,
                "final_publication_visible_monotonic_ns": None,
                "first_complete_file_size_observed_monotonic_ns": None}
    pending_hash = hashlib.sha256()
    # Preserve the qualified worker's unrestricted I/O scheduling. Pin only this
    # consumer and its subsequently created publication reader to one logical CPU.
    process, identity = supervisor.process, supervisor.identity
    original_affinity = sorted(os.sched_getaffinity(0))
    cpu = max(original_affinity)
    os.sched_setaffinity(0, {cpu})
    schedule = None
    started = time.monotonic()

    def read_publications():
        nonlocal schedule
        offset, expected_chunk = 0, 0
        try:
            path = folder / "rx-chunks.jsonl"
            while not path.exists() and process.poll() is None and not stop.is_set():
                identity.observe()
                time.sleep(.001)
            if not path.exists():
                raise RuntimeError("native publication file missing")
            with path.open("rb") as stream, (folder / "publication-observations.jsonl").open("x") as receipt:
                while not stop.is_set():
                    identity.observe()
                    row = committed_line(stream)
                    if row is None:
                        if process.poll() is not None:
                            break
                        time.sleep(.001)
                        continue
                    visible = time.monotonic_ns()
                    if row["chunk_index"] != expected_chunk:
                        raise ValueError("missing, repeated or reordered publication")
                    expected_chunk += 1
                    if schedule is None and (folder / "schedule.json").exists():
                        try:
                            schedule = json.loads((folder / "schedule.json").read_text())
                        except json.JSONDecodeError:
                            pass
                    if row["accepted_samples_from_chunk"]:
                        if schedule is None:
                            raise ValueError("accepted publication before complete schedule")
                        span = accepted_span(row, schedule, offset)
                        begin, count = span
                        raw = published_bytes(folder / "accepted.cfile", offset, count)
                        if raw is None:
                            raise ValueError("committed accepted byte span is incomplete")
                        pending_hash.update(raw)
                        offset += count
                        observed["committed_samples"] = offset
                        if offset == plan["record_samples_each"]:
                            observed["final_publication_visible_monotonic_ns"] = visible
                        iq = np.frombuffer(raw, dtype="<c8")
                        for part in range(0, count, BLOCK):
                            block = iq[part:part+BLOCK].copy()
                            enqueue = time.monotonic_ns()
                            fifo.put_nowait((begin+part, block, visible, enqueue))
                            observed["enqueued_samples"] += block.size
                            observed["max_queue_chunks_observed"] = max(observed["max_queue_chunks_observed"], fifo.qsize())
                        observed["max_committed_backlog_samples"] = max(
                            observed["max_committed_backlog_samples"], offset-observed["processed_samples"])
                    receipt.write(json.dumps({"chunk_index": row["chunk_index"], "visibility_observed_monotonic_ns": visible,
                                              "accepted_samples_from_chunk": row["accepted_samples_from_chunk"],
                                              "accepted_sample_offset_after_chunk": offset})+"\n")
                if offset != plan["record_samples_each"]:
                    raise ValueError("publication population incomplete")
                if stream.read():
                    raise ValueError("trailing unpublished partial JSON")
            observed["input_sha256"] = pending_hash.hexdigest()
        except Exception as error:
            errors.append("publication reader: " + repr(error))
            stop.set()
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        finally:
            done.set()

    thread = threading.Thread(target=read_publications, name="committed-RX-reader")
    thread.start()
    first = True
    try:
        while not done.is_set() or not fifo.empty():
            if supervisor.interrupted:
                raise InterruptedError("supervisor stop requested")
            if errors:
                raise RuntimeError(errors[-1])
            if time.monotonic()-started > 40:
                raise TimeoutError("handoff wall budget exceeded")
            try:
                begin, iq, visible, enqueue = fifo.get(timeout=.02)
            except queue.Empty:
                continue
            if first:
                adapter = StreamingDigitalAdapter(CONFIG, first_input_index=begin)
                first = False
            t0 = time.monotonic_ns()
            emitted = adapter.push(iq, input_start_index=begin)
            t1 = time.monotonic_ns()
            observed["processed_samples"] += iq.size
            visible_samples = (folder / "accepted.cfile").stat().st_size // 8
            size_observed = time.monotonic_ns()
            if visible_samples == plan["record_samples_each"] and observed["first_complete_file_size_observed_monotonic_ns"] is None:
                observed["first_complete_file_size_observed_monotonic_ns"] = size_observed
            backlog = visible_samples-observed["processed_samples"]
            observed["max_visible_file_backlog_samples"] = max(observed["max_visible_file_backlog_samples"], backlog)
            alive = identity.observe()
            service.append({"input_start_index": begin, "samples": int(iq.size),
                            "publication_visibility_observed_monotonic_ns": visible,
                            "enqueue_monotonic_ns": enqueue, "service_start_monotonic_ns": t0,
                            "service_end_monotonic_ns": t1, "visible_file_samples": visible_samples,
                            "file_size_observed_monotonic_ns": size_observed,
                            "processed_samples": observed["processed_samples"],
                            "visible_file_backlog_samples": backlog, "queue_chunks_observed": fifo.qsize(),
                            "emitted_frames": len(emitted), **alive})
            for frame in emitted:
                frames.append({**frame_record(frame), "produced_monotonic_ns": t1,
                               "visible_file_samples_observed_after_production": visible_samples,
                               "file_size_observed_monotonic_ns": size_observed, **alive})
            fifo.task_done()
            service[-1]["consumer_work_completed_monotonic_ns"] = time.monotonic_ns()
            observed["last_input_service_completed_monotonic_ns"] = t1
            observed["last_input_consumer_work_completed_monotonic_ns"] = service[-1]["consumer_work_completed_monotonic_ns"]
        if errors:
            raise RuntimeError(errors[-1])
        boundary = adapter.finish(reason="accepted_interval_end")
        for frame in boundary["frames"]:
            frames.append({**frame_record(frame), "produced_monotonic_ns": time.monotonic_ns(),
                           "visible_file_samples_observed_after_production": (folder / "accepted.cfile").stat().st_size//8,
                           "file_size_observed_monotonic_ns": time.monotonic_ns(),
                           **identity.observe()})
        save(folder / "stream-boundary.json", boundary["receipt"])
        observed["boundary_finalization_completed_monotonic_ns"] = time.monotonic_ns()
        published = observed["final_publication_visible_monotonic_ns"]
        observed["last_input_service_after_final_publication_observed_seconds"] = (
            (observed["last_input_service_completed_monotonic_ns"]-published)/1e9 if published is not None else None)
        observed["boundary_finalization_after_final_publication_observed_seconds"] = (
            (observed["boundary_finalization_completed_monotonic_ns"]-published)/1e9 if published is not None else None)
        observed["boundary_finalization_interval_includes_helper_cleanup_wait"] = True
    except BaseException as error:
        errors.append("consumer: " + repr(error))
        stop.set()
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        if not adapter.state["closed"]:
            boundary = adapter.finish(reason="host_or_transport_failure")
            save(folder / "stream-boundary.json", boundary["receipt"])
    finally:
        stop.set()
        thread.join(timeout=2)
        if thread.is_alive():
            errors.append("publication reader did not stop")
        supervisor.close(abort=bool(errors))
        if supervisor.forced:
            errors.append("forced process cleanup; hardware cleanup unavailable")
        identity.observe()
        observed["native_exit_first_observed_monotonic_ns"] = identity.first_exit_observed_ns
        observed["native_worker_pid"] = identity.pid
        os.sched_setaffinity(0, original_affinity)
    save(folder / "live-frames.json", frames)
    save(folder / "service.json", service)
    receipt = json.loads((folder / "receipt.json").read_text()) if (folder / "receipt.json").exists() else {}
    accepted = receipt_accepted(receipt, plan["record_samples_each"])
    if supervisor.interrupted:
        errors.append("supervisor interrupted; attempt cannot qualify")
    during = sum(f["native_worker_alive_observed"] and f["visible_file_samples_observed_after_production"] < plan["record_samples_each"] for f in frames)
    result = {"schema": "sdr-live-handoff-attempt-v1", "errors": errors, "capture_helper_returncode": process.returncode,
              "native_capture_accepted": accepted, "frames": len(frames), "frames_during_incomplete_capture": during,
              "consumer_cpu": cpu, "consumer_and_reader_cpu_count": 1, "native_worker_inherited_affinity_expected": original_affinity,
              "observer_wall_seconds": time.monotonic()-started, "observations": observed,
              "success": not errors and not supervisor.interrupted and accepted and process.returncode == 0 and during > 0
                         and observed["processed_samples"] == plan["record_samples_each"]}
    save(folder / "handoff.json", result)
    return result


def live_attempt(folder, plan):
    capture_plan = json.loads((Path(folder) / "plan.json").read_text())
    with CaptureSupervisor(folder, capture_plan["command"][0]) as supervisor:
        return _live_attempt(folder, plan, supervisor)


def offline_parity(folder, plan):
    frames = json.loads((folder / "live-frames.json").read_text())
    data = np.fromfile(folder / "accepted.cfile", dtype="<c8")
    schedule = json.loads((folder / "schedule.json").read_text())
    origin = schedule["accepted_first_timestamp"]
    adapter = StreamingDigitalAdapter(CONFIG, first_input_index=origin)
    stream_frames = []
    for begin in range(0, data.size, BLOCK):
        stream_frames.extend(adapter.push(data[begin:begin+BLOCK], input_start_index=origin+begin))
    ending = adapter.finish()
    stream_frames.extend(ending["frames"])
    # Finite records have at most95 outputs; keeping this offline evidence is
    # bounded by the fixed four-second design, independent of live queue state.
    batch = adapt_record(data, CONFIG)
    differences, parity, batch_packed = [], True, True
    count_match = len(frames) == len(stream_frames) == batch["metadata"]["frame_count"]
    for i, frame in enumerate(stream_frames[:min(len(frames), batch["metadata"]["frame_count"])]):
        record = frame_record(frame)
        parity &= all(frames[i][k] == v for k, v in record.items())
        diff = float(np.max(abs(frame["resampled_frame"]-batch["resampled_frames"][i])))
        differences.append(diff)
        batch_packed &= bool(np.array_equal(frame["packed_frame"], batch["packed_frames"][i])
                             and np.array_equal(frame["projections_i32"], batch["projections_i32"][i]))
    handoff = json.loads((folder / "handoff.json").read_text())
    delta = max(differences, default=0.)
    result = {"schema": "sdr-live-handoff-parity-v1", "live_frame_count": len(frames),
              "offline_stream_frame_count": len(stream_frames), "batch_frame_count": batch["metadata"]["frame_count"],
              "accepted_samples": int(data.size), "accepted_sha256": sha(folder / "accepted.cfile"),
              "stream_exact_partition_parity": bool(parity and count_match),
              "batch_exact_packed_projection_parity": bool(batch_packed and count_match),
              "maximum_batch_float_absolute_difference": delta, "batch_float_absolute_tolerance": 2e-12,
              "minimum_quantizer_half_step_distance": min((f["minimum_half_step_distance"] for f in frames if f["minimum_half_step_distance"] is not None), default=None),
              "batch_quantization_scope": "Exact packed/projection equality is checked for these records; no universal floating half-step parity claim.",
              "unused_full_support_tail_samples": batch["metadata"]["unused_full_support_tail_samples"],
              "batch_ratio_status": batch["metadata"]["ratio_status"],
              "saturated_components": batch["metadata"]["sample_quantization"]["saturated_component_count"]}
    result["success"] = bool(parity and batch_packed and delta <= 2e-12 and count_match
                             and data.size == plan["record_samples_each"] and result["accepted_sha256"] == handoff["observations"]["input_sha256"])
    save(folder / "offline-parity.json", result)
    return result


def run(out, hardware_authorized, rf_confined_authorized):
    if not hardware_authorized or not rf_confined_authorized:
        raise ValueError("explicit hardware/confined-room authorization required for unchanged initialization")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError(f"{key}=1 required")
    plan = verify_plan(out)
    save(out / "launch.json", {"plan_sha256": sha(out / "plan.json"), "unix_ns": time.time_ns()})
    states = [{"directory": a["directory"], "state": "unattempted"} for a in plan["attempts"]]
    for state in states:
        folder = out / state["directory"]
        state["state"] = "attempted"
        try:
            result = live_attempt(folder, plan)
            if not result["success"]:
                raise RuntimeError("live handoff failed; native and host receipts retained")
            if not offline_parity(folder, plan)["success"]:
                raise RuntimeError("offline arithmetic parity failed")
            state["state"] = "complete"
        except Exception as error:
            state.update(state="failed", error=repr(error))
            break
    integrity = {"unchanged": False}
    try:
        verify_plan(out)
        integrity["unchanged"] = True
    except Exception as error:
        integrity["error"] = repr(error)
    save(out / "source-integrity.json", integrity)
    result = {"schema": "sdr-live-handoff-study-v1", "kind": plan["kind"], "states": states,
              "success": all(s["state"] == "complete" for s in states) and integrity["unchanged"], "scope": plan["scope"]}
    save(out / "run-receipt.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "run"))
    p.add_argument("output", type=Path)
    p.add_argument("--kind", choices=("smoke", "primary"))
    p.add_argument("--build-manifest", type=Path)
    p.add_argument("--serial")
    p.add_argument("--hardware-authorized", action="store_true")
    p.add_argument("--rf-confined-authorized", action="store_true")
    a = p.parse_args()
    if a.stage == "prepare":
        if a.kind is None or a.build_manifest is None or a.serial is None:
            p.error("prepare requires kind, build-manifest and explicit serial")
        result = prepare(a.output, a.kind, a.build_manifest, a.serial)
    else:
        result = run(a.output.resolve(), a.hardware_authorized, a.rf_confined_authorized)
    print(json.dumps(result, indent=2, allow_nan=False))
    if a.stage == "run" and not result["success"]:
        raise SystemExit(1)
