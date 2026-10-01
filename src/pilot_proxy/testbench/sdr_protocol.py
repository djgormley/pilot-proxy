"""Offline OTA protocol and single-slot simulation receipts; no hardware adapter.

All acceptance here is protocol/fixture acceptance. Stationarity and RF-power
reports are supplied evidence, not measurements made or certified by this module.
The existing sdr_transfer worker must not be used as a single-slot adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

SCHEMA = "pilotproxy-offline-ota-protocol-v1"
NOISE_MODELS = ("independent_complex_gaussian", "orthogonal_unit_power")
POWER_METRICS = ("average_conducted_dbm", "peak_conducted_dbm", "average_eirp_dbm", "peak_eirp_dbm")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_new(path, value):
    import os

    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return float(value)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _reference(record, name):
    if not isinstance(record, dict):
        raise ValueError(f"{name} is required")
    path = Path(record["path"]).resolve()
    if digest(path) != record["sha256"]:
        raise ValueError(f"{name} digest changed")
    return path


def settings_identity(spec):
    """Bind power evidence to the complete proposed waveform and radio settings."""
    return identity({k: spec[k] for k in ("waveform", "radio", "noise_model", "slots")})


def validate_spec(spec):
    """Raise on missing evidence/limits; return the conservative stated power bound.

    No jurisdictional limit is invented. The supplied limit uses its named
    average/peak conducted/EIRP convention. The conversion report must explicitly
    cover every slot, its marker and transition, and total measurement uncertainty.
    """
    identity(spec)  # Reject non-JSON/nonfinite fields before freezing.
    if spec.get("schema") != SCHEMA or not str(spec.get("run_id", "")).strip():
        raise ValueError("a versioned protocol and run_id are required")
    if spec.get("noise_model") not in NOISE_MODELS:
        raise ValueError("declare independent Gaussian or orthogonal-conditioned noise")
    sources = spec.get("implementation_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("implementation source receipts are required")
    for source in sources:
        _reference(source, "implementation source")
    wave = spec["waveform"]
    path = _reference(wave, "waveform")
    start = _integer(wave["start_sample"], "waveform start")
    samples = _integer(wave["samples"], "waveform samples", 1)
    discarded = _integer(wave["source_startup_discard_samples"], "source startup discard")
    if path.stat().st_size % 8 or (start + samples) * 8 > path.stat().st_size:
        raise ValueError("waveform interval is outside the complex64 file")
    values = np.memmap(path, dtype=np.complex64, mode="r", offset=start*8, shape=(samples,))
    if not np.isfinite(values).all() or not np.any(values):
        raise ValueError("waveform interval must be finite and nonzero")
    audit_path = _reference(spec["stationarity_audit"], "stationarity audit")
    audit = json.loads(audit_path.read_text())
    if (audit.get("schema") != "pilotproxy-stationary-input-audit-v1"
            or audit.get("waveform_sha256") != wave["sha256"]
            or audit.get("start_sample") != start or audit.get("samples") != samples
            or audit.get("source_startup_discard_samples") != discarded
            or audit.get("stationarity_passed") is not True
            or not audit.get("method") or not audit.get("thresholds")
            or len(audit.get("segments", [])) < 3):
        raise ValueError("stationarity evidence must cover this exact interval and startup discard")
    # The evidence report supplies the spectral definition and predeclared limits;
    # three disjoint chronological slices are mandatory. This is not a new audit.
    last_stop = start
    for segment in audit["segments"]:
        a = _integer(segment["start_sample"], "stationarity segment start")
        b = _integer(segment["stop_sample"], "stationarity segment stop", 1)
        if a < last_stop or b <= a or b > start + samples or segment.get("passed") is not True:
            raise ValueError("stationarity segments must pass and be disjoint within the waveform")
        last_stop = b
    if audit["segments"][0]["start_sample"] != start or last_stop != start + samples:
        raise ValueError("stationarity evidence must include the beginning and end")
    radio = spec["radio"]
    for key in ("frequency_hz", "sample_rate_hz", "emission_bandwidth_hz", "radio_filter_bandwidth_hz", "tx_rms", "antenna_separation_cm"):
        if _number(radio[key], key, 0) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("tx_gain_setting_db", "rx_gain_setting_db"):
        _number(radio[key], key)
    for key in ("device_serial", "tx_antenna", "rx_antenna", "antenna_polarization", "clock_source"):
        if not isinstance(radio.get(key), str) or not radio[key].strip():
            raise ValueError(f"{key} is required")
    for direction in ("tx", "rx"):
        setting = radio[f"{direction}_gain_setting_db"]
        native = _integer(radio[f"native_{direction}_gain_db"], "native gain")
        if not -12 <= setting <= 61 or setting != int(setting) or native != setting + 12:
            raise ValueError("Lime gain setting/native mapping differs")
    for key in ("settle_samples", "marker_guard_samples", "capture_samples"):
        _integer(spec["capture"][key], key, 1)
    if spec["capture"]["settle_samples"] + spec["capture"]["capture_samples"] > samples:
        raise ValueError("waveform interval is shorter than settle plus measurement")
    guards = spec["guards"]
    for key in ("clip_level", "rms_min", "rms_max", "clip_fraction_max", "marker_correlation_min"):
        _number(guards[key], key, 0)
    if not (0 < guards["clip_level"] <= 1 and 0 <= guards["clip_fraction_max"] <= 1
            and 0 < guards["marker_correlation_min"] <= 1 and guards["rms_min"] < guards["rms_max"]):
        raise ValueError("invalid fixed level/alignment guards")
    slots = spec["slots"]
    if not isinstance(slots, list) or len(slots) < 3 or [s.get("kind") for s in slots[:3]] != ["tx_zero", "signal_only", "noise_only"]:
        raise ValueError("fixed schedule must begin with zero, signal and noise controls")
    seeds = set()
    for index, slot in enumerate(slots):
        if slot.get("slot_index") != index or slot.get("kind") not in ("tx_zero", "signal_only", "noise_only", "mixture", "drift"):
            raise ValueError("invalid chronological slot identity")
        if slot["kind"] in ("mixture", "drift"):
            _number(slot["data_shelf_snr_db"], "declared mixture SNR")
        seed = _integer(slot["seed"], "slot seed")
        if seed in seeds:
            raise ValueError("slot noise seeds must be distinct")
        seeds.add(seed)
    limits = spec.get("rf_limits")
    if not isinstance(limits, dict):
        raise ValueError("allowed RF frequency and power limits are unresolved")
    low = _number(limits["frequency_low_hz"], "allowed frequency low", 0)
    high = _number(limits["frequency_high_hz"], "allowed frequency high", 0)
    center, bandwidth = radio["frequency_hz"], radio["emission_bandwidth_hz"]
    if high <= low or center-bandwidth/2 < low or center+bandwidth/2 > high:
        raise ValueError("characterized emission interval exceeds supplied allowed band")
    metric = limits.get("power_metric")
    if metric not in POWER_METRICS or not limits.get("provided_by"):
        raise ValueError("power convention and constraint source are required")
    maximum = _number(limits["maximum_power_dbm"], "allowed maximum power")
    power_path = _reference(spec.get("power_conversion"), "measured power conversion (gain is not dBm)")
    power = json.loads(power_path.read_text())
    if (power.get("schema") != "pilotproxy-measured-rf-power-v1"
            or power.get("settings_sha256") != settings_identity(spec)
            or power.get("power_metric") != metric
            or power.get("covers_all_slots_markers_transitions") is not True
            or not power.get("measurement_id") or not power.get("method")):
        raise ValueError("power conversion does not bind this full waveform/settings/schedule")
    bound = _number(power["conducted_power_dbm"], "measured conducted power")
    if "eirp" in metric:
        bound += _number(power["antenna_gain_dbi"], "antenna gain")
        bound -= _number(power["cable_loss_db"], "cable loss", 0)
    bound += _number(power["total_uncertainty_db"], "total power uncertainty", 0)
    if bound > maximum:
        raise ValueError("measured power plus total uncertainty exceeds supplied limit")
    return {"power_metric": metric, "upper_power_dbm": bound, "maximum_power_dbm": maximum}


def freeze_protocol(directory, spec):
    """Create an immutable offline protocol only after required evidence is supplied."""
    power = validate_spec(spec)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("a protocol requires a new empty directory")
    directory.mkdir(parents=True, exist_ok=True)
    plan = {"schema": SCHEMA, "frozen_utc": _now(), "spec": deepcopy(spec),
            "power_check": power, "implementation_sha256": digest(__file__),
            "scope": "offline single-slot simulation; no hardware adapter, RF permission or physical calibration certified"}
    _write_new(directory/"plan.json", plan)
    _write_new(directory/"plan-digest.json", {"sha256": digest(directory/"plan.json")})
    return plan


def load_protocol(directory):
    directory = Path(directory)
    expected = json.loads((directory/"plan-digest.json").read_text())["sha256"]
    if digest(directory/"plan.json") != expected:
        raise ValueError("frozen protocol changed")
    plan = json.loads((directory/"plan.json").read_text())
    if plan["implementation_sha256"] != digest(__file__):
        raise ValueError("protocol implementation changed; use a new version")
    if validate_spec(plan["spec"]) != plan["power_check"]:
        raise ValueError("power evidence changed")
    return plan


def _state(directory):
    """Verify every permanent receipt; pending attempts refuse further slots."""
    directory = Path(directory)
    plan = load_protocol(directory)
    previous = digest(directory/"plan.json")
    receipts = sorted(directory.glob("slot-*-receipt.json"))
    intents = sorted(directory.glob("slot-*-intent.json"))
    seals = sorted(directory.glob("slot-*-digest.json"))
    if len(intents) != len(receipts) or len(seals) != len(receipts):
        raise ValueError("an interrupted slot is unresolved; new run identity required")
    for index, path in enumerate(receipts):
        expected_path = directory/f"slot-{index:06d}-receipt.json"
        if path != expected_path:
            raise ValueError("receipt sequence has a gap")
        seal = json.loads((directory/f"slot-{index:06d}-digest.json").read_text())
        if digest(path) != seal["sha256"]:
            raise ValueError("slot receipt digest changed")
        receipt = json.loads(path.read_text())
        if receipt["slot_index"] != index:
            raise ValueError("receipt slot identity changed")
        intent_path = directory/f"slot-{index:06d}-intent.json"
        intent = json.loads(intent_path.read_text())
        if (receipt["previous_sha256"] != previous or receipt["intent_sha256"] != digest(intent_path)
                or intent["previous_sha256"] != previous or intent["slot_index"] != index
                or intent["plan_sha256"] != digest(directory/"plan.json")):
            raise ValueError("slot receipt history changed")
        if receipt.get("capture_sha256") is not None:
            if digest(directory/f"slot-{index:06d}-capture.cfile") != receipt["capture_sha256"]:
                raise ValueError("saved capture changed")
        previous = digest(path)
        if receipt["status"] != "accepted_simulation":
            raise ValueError("slot failed; run halted and cannot be retried or extended")
    return plan, len(receipts), previous


def next_simulation_request(directory):
    """Return exactly one slot, never a hardware/transmit request."""
    plan, index, previous = _state(directory)
    spec = plan["spec"]
    if index >= len(spec["slots"]):
        raise ValueError("planned simulation is complete")
    return {"scope": "offline_simulation_only", "slot": deepcopy(spec["slots"][index]),
            "capture": deepcopy(spec["capture"]), "radio": deepcopy(spec["radio"]),
            "waveform": deepcopy(spec["waveform"]), "noise_model": spec["noise_model"],
            "plan_sha256": digest(Path(directory)/"plan.json"), "previous_sha256": previous}


def simulate_next_slot(directory, worker: Callable):
    """Exercise an injected fake worker, saving its capture before checking guards.

    The callback receives one offline request and returns measurement-only
    complex64 samples and health/alignment fields. No implementation supplied
    here opens a device. A later hardware integration requires separate review.
    """
    directory = Path(directory)
    request = next_simulation_request(directory)
    spec = load_protocol(directory)["spec"]
    index = request["slot"]["slot_index"]
    intent_path = directory/f"slot-{index:06d}-intent.json"
    _write_new(intent_path, {"started_utc": _now(), "slot_index": index,
               "plan_sha256": request["plan_sha256"], "previous_sha256": request["previous_sha256"],
               "request": request})
    receipt = {"schema": "pilotproxy-offline-slot-receipt-v1", "slot_index": index,
               "previous_sha256": request["previous_sha256"], "intent_sha256": digest(intent_path),
               "scope": "offline_simulation_only", "capture_sha256": None, "status": "halted"}
    try:
        response = worker(deepcopy(request))
        values = np.asarray(response["samples"])
        if values.dtype != np.dtype("complex64") or values.ndim != 1:
            raise ValueError("worker samples must be a complex64 measurement vector")
        capture_path = directory/f"slot-{index:06d}-capture.cfile"
        with capture_path.open("xb") as stream:
            stream.write(values.tobytes())
        receipt["capture_sha256"] = digest(capture_path)
        receipt["worker_report"] = deepcopy({k: v for k, v in response.items() if k != "samples"})
        # Validate JSON report before accepting it or writing the durable receipt.
        identity(receipt["worker_report"])
        if values.size != spec["capture"]["capture_samples"] or not np.isfinite(values).all():
            raise ValueError("capture count or finiteness failed")
        for key in ("rx_underrun", "rx_overrun", "rx_dropped_packets", "tx_underrun", "tx_overrun", "tx_dropped_packets", "timestamp_discontinuities"):
            if type(response.get(key)) is not int or response[key] != 0:
                raise ValueError(f"stream health failed: {key}")
        if response.get("clock_locked") is not True or response.get("lo_locked") is not True:
            raise ValueError("clock/LO state is not confirmed")
        for key in ("frequency_hz", "sample_rate_hz", "tx_gain_setting_db", "rx_gain_setting_db"):
            if response.get(key) != spec["radio"][key]:
                raise ValueError(f"radio readback differs: {key}")
        start = _integer(response["measurement_start_sample"], "measurement start")
        if start < spec["capture"]["marker_guard_samples"] + spec["capture"]["settle_samples"]:
            raise ValueError("measurement overlaps marker guard or settling")
        if _number(response["marker_correlation"], "marker correlation") < spec["guards"]["marker_correlation_min"]:
            raise ValueError("alignment guard failed")
        components = np.maximum(np.abs(values.real), np.abs(values.imag))
        rms = float(np.sqrt(np.mean(np.abs(values.astype(np.complex128))**2)))
        clip = float(np.mean(components >= spec["guards"]["clip_level"]))
        receipt["levels"] = {"rms": rms, "clip_fraction": clip, "peak_component": float(components.max())}
        if rms > spec["guards"]["rms_max"] or clip > spec["guards"]["clip_fraction_max"]:
            raise ValueError("capture overload guard failed")
        if request["slot"]["kind"] != "tx_zero" and rms < spec["guards"]["rms_min"]:
            raise ValueError("capture lower level guard failed")
        receipt["status"] = "accepted_simulation"
    except BaseException as error:
        # Preserve the failure, not its arbitrary callback object/nonfinite report.
        receipt.pop("worker_report", None)
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
        receipt["completed_utc"] = _now()
        _seal_receipt(directory, index, receipt)
        raise
    receipt["completed_utc"] = _now()
    _seal_receipt(directory, index, receipt)
    return receipt


def _seal_receipt(directory, index, receipt):
    path = directory/f"slot-{index:06d}-receipt.json"
    _write_new(path, receipt)
    _write_new(directory/f"slot-{index:06d}-digest.json", {"sha256": digest(path)})


def draw_noise(signal, *, seed, model):
    """Declared digital models, without RF I/O or signal-dependent AWGN rescaling."""
    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if signal.size < 2 or not np.isfinite(signal).all() or not np.any(signal):
        raise ValueError("finite nonzero signal required")
    rng = np.random.default_rng(_integer(seed, "noise seed"))
    noise = (rng.standard_normal(signal.size) + 1j*rng.standard_normal(signal.size))/math.sqrt(2)
    if model == "orthogonal_unit_power":
        noise -= np.vdot(signal, noise)/np.vdot(signal, signal)*signal
        noise /= math.sqrt(float(np.mean(np.abs(noise)**2)))
    elif model != "independent_complex_gaussian":
        raise ValueError("unknown noise model")
    return noise.astype(np.complex64)
