#!/usr/bin/env python3
"""Frozen full-array digital coarse controls; no RF hardware or physical claim."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
SPEC = importlib.util.spec_from_file_location(
    "_channel29_frontier", HERE / "mask_residual_frontier.py"
)
mrf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mrf
SPEC.loader.exec_module(mrf)
cgs = mrf.cgs
from pilot_proxy.detector_reference import (
    matched_filter_row_projections_cpu_reference_packed,
)
from pilot_proxy.detector_contract import weight_term_norms_sq

SCHEMA = "channel29-residual-controls-v1"
STAGES = ("calibration", "evaluation", "stress")
CHANNEL, STREAMS, K, WINDOWS = 29, 2048, 128, 128
SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def digest_payload(value, field):
    return hashlib.sha256(
        canonical({k: v for k, v in value.items() if k != field})
    ).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def utc_time(value):
    if not isinstance(value, str):
        raise ValueError("frozen timestamp must be an aware ISO UTC string")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("frozen timestamp must be aware UTC")
    return stamp


def write_new(path, value):
    content = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def check_identity(value):
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError("input requires an absolute path and SHA256")
    path = Path(value["path"])
    wanted = value.get("sha256")
    if (
        not path.is_absolute()
        or not path.is_file()
        or not isinstance(wanted, str)
        or not SHA.fullmatch(wanted)
    ):
        raise ValueError("invalid input identity")
    if sha(path) != wanted:
        raise ValueError(f"input hash mismatch: {path}")
    return path


def frame_seed(
    protocol_sha256,
    stage,
    block_index,
    population,
    index,
    num_streams=STREAMS,
    channel=CHANNEL,
):
    fields = [
        SCHEMA,
        protocol_sha256,
        stage,
        int(block_index),
        str(population),
        int(index),
        int(num_streams),
        int(channel),
    ]
    return int.from_bytes(hashlib.sha256(canonical(fields)).digest()[:8], "big")


def effective_seed(seed):
    return int(seed) % (1 << 63)


def source_closure():
    files = [
        Path(__file__),
        HERE / "mask_residual_frontier.py",
        HERE / "current_geometry_sensitivity.py",
    ]
    files.extend(
        sorted(
            p
            for p in (REPO / "src/pilot_proxy").rglob("*")
            if p.is_file()
            and p.suffix
            in (".py", ".json", ".yaml", ".yml", ".toml", ".csv", ".dat", ".npz")
            and "__pycache__" not in p.parts
        )
    )
    return {str(p.resolve()): {"sha256": sha(p)} for p in files}


def runtime_fingerprint(gpu):
    cp = gpu.cp
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    values = {
        "python_executable": str(Path(sys.executable).absolute()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "platform": platform.platform(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "device_name": name,
        "compute_capability": str(cp.cuda.Device().compute_capability),
    }
    return {"values": values, "sha256": hashlib.sha256(canonical(values)).hexdigest()}


def args_for(config, output):
    args = mrf.build_parser().parse_args(
        [
            "--stage",
            "generate",
            "--gpu",
            "--output-dir",
            str(output),
            "--input-iq",
            str(config["input_iq"]),
            "--waveform-audit",
            str(config["waveform_audit"]),
            "--weights-path",
            str(config["weights"]),
            "--lib-path",
            str(config["lib"]),
            "--physical-channel",
            str(CHANNEL),
            "--num-streams",
            str(STREAMS),
            "--spectral-sense",
            "normal",
            "--seed",
            "0",
            "--input-scale",
            str(config.get("input_scale", cgs.DEFAULT_INPUT_SCALE)),
        ]
    )
    for field in (
        "pilot_below_data_db",
        "dtv_bandwidth_hz",
        "bin_enbw_hz",
        "pilot_capture_efficiency",
        "iq_sample_rate_hz",
    ):
        if field in config:
            setattr(args, field, float(config[field]))
    return args


def prepare(config, output):
    """Build a new deterministic PFB cache and authenticate runtime, without controls."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("prepare output must be new")
    args = args_for(config, output)
    profile = mrf._prepare(args)
    gpu = cgs._initialize_gpu(args)
    study_config = mrf._study_config(args, profile)
    _atsc, _tone, cache_meta = cgs._load_signal_cache(args, study_config, profile)
    inputs = {
        name: identity(path)
        for name, path in {
            "input_iq": args.input_iq,
            "waveform_audit": args.waveform_audit,
            "weights": args.weights_path,
            "lib": args.lib_path,
            "weight_manifest": Path(args.weights_path).with_suffix(
                ".bin.manifest.json"
            ),
            "signal_cache": cgs._cache_path(output, profile),
            "cache_config": output / "study_config.json",
        }.items()
    }
    norm = weight_term_norms_sq(profile.packed_weights, bits_per_component=4)
    if (int(norm[0]), int(norm[1] + norm[2])) != (6372, 12713):
        raise ValueError(
            "channel29 packed coefficient norms differ from the frozen candidate"
        )
    result = {
        "schema": "channel29-residual-prepare-v1",
        "created_utc": utc_now(),
        "science_frames_generated": 0,
        "geometry": {
            "channel": CHANNEL,
            "num_streams": STREAMS,
            "K": K,
            "windows_per_stream": WINDOWS,
            "spectral_sense": "normal",
            "input_scale": float(args.input_scale),
        },
        "inputs": inputs,
        "runtime": runtime_fingerprint(gpu),
        "source_artifacts": source_closure(),
        "target_norm_sq": int(norm[0]),
        "reference_norm_sum_sq": int(norm[1] + norm[2]),
        "cache_metadata": cache_meta,
        "packed_profile_sha256": hashlib.sha256(
            profile.packed_weights.tobytes()
        ).hexdigest(),
        "preparation_scope": "Deterministic reference PFB/noise-gain cache only; fixed preparatory gain seed, no science control draw",
    }
    write_new(output / "prepared.json", result)
    return result


def validate_plan(plan, *, verify_files=True):
    if plan.get("schema") != SCHEMA or plan.get("status") != "plan_frozen":
        raise ValueError("a frozen channel29 residual-control plan is required")
    if not isinstance(plan.get("protocol_sha256"), str) or not SHA.fullmatch(
        plan["protocol_sha256"]
    ):
        raise ValueError("protocol digest is invalid")
    if plan.get("plan_sha256") != digest_payload(plan, "plan_sha256"):
        raise ValueError("plan digest mismatch")
    if utc_time(plan["frozen_at"]) > datetime.now(timezone.utc):
        raise ValueError("plan freeze lies in the future")
    expected = {
        "channel": CHANNEL,
        "num_streams": STREAMS,
        "K": K,
        "windows_per_stream": WINDOWS,
        "spectral_sense": "normal",
    }
    geometry = plan["geometry"]
    if any(geometry.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "full2048stream channel29 normal-coordinate geometry is required"
        )
    if (
        not isinstance(geometry.get("input_scale"), (int, float))
        or not math.isfinite(geometry["input_scale"])
        or geometry["input_scale"] <= 0
    ):
        raise ValueError("input scale must be positive")
    populations = plan["populations"]
    if not isinstance(populations, list) or not populations:
        raise ValueError("populations must be a nonempty fixed list")
    names = set()
    for pop in populations:
        if (
            not isinstance(pop.get("name"), str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", pop["name"])
            or pop["name"] in names
        ):
            raise ValueError("population names must be unique safe path components")
        names.add(pop["name"])
        shelf = pop.get("shelf_db")
        if shelf is not None and (
            type(shelf) not in (int, float) or not math.isfinite(shelf)
        ):
            raise ValueError("shelf must be finite or null")
        gamma = pop.get("reference_projection_gamma")
        if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma < 0:
            raise ValueError("reference gamma must be finite and nonnegative")
        if type(pop.get("count_per_block")) is not int or pop["count_per_block"] <= 0:
            raise ValueError("population frame count must be a positive integer")
    if set(plan["stages"]) != set(STAGES):
        raise ValueError("all three fixed stages are required")
    for stage, spec in plan["stages"].items():
        if type(spec.get("blocks")) is not int or spec["blocks"] <= 0:
            raise ValueError("block counts must be fixed positive integers")
        pops = spec.get("populations")
        if (
            not isinstance(pops, list)
            or not pops
            or len(set(pops)) != len(pops)
            or not set(pops) <= names
        ):
            raise ValueError("stage populations must be unique known names")
        if stage in ("calibration", "evaluation") and any(
            p["reference_projection_gamma"] != 0
            for p in populations
            if p["name"] in pops
        ):
            raise ValueError("reference contamination is a separate stress stage")
    audit = plan["audit"]
    if (
        set(audit["populations"]) != names
        or len(audit["populations"]) != len(names)
        or audit.get("frames_per_population") != 1
    ):
        raise ValueError("audit must cover every population once")
    if (
        not isinstance(plan["evaluation_prerequisite"].get("receipt_path"), str)
        or not Path(plan["evaluation_prerequisite"]["receipt_path"]).is_absolute()
    ):
        raise ValueError("evaluation requires an absolute frozen-receipt path")
    if not plan.get("source_artifacts"):
        raise ValueError("source closure must be bound")
    if (
        plan["runtime"].get("sha256")
        != hashlib.sha256(canonical(plan["runtime"]["values"])).hexdigest()
    ):
        raise ValueError("runtime fingerprint digest mismatch")
    required_inputs = {
        "input_iq",
        "waveform_audit",
        "weights",
        "weight_manifest",
        "lib",
        "signal_cache",
        "cache_config",
        "planned_seeds",
        "seed_exclusions",
    }
    if not required_inputs <= set(plan["inputs"]):
        raise ValueError("missing prepared inputs")
    if verify_files:
        for value in plan["inputs"].values():
            check_identity(value)
        for path, item in plan["source_artifacts"].items():
            check_identity({"path": path, **item})
        for path, value in source_closure().items():
            if plan["source_artifacts"].get(path, {}).get("sha256") != value["sha256"]:
                raise ValueError(f"current generator source is not bound: {path}")
    return True


def planned_seed_rows(plan):
    populations = {p["name"]: p for p in plan["populations"]}
    for stage in (*STAGES, "audit"):
        blocks = 1 if stage == "audit" else plan["stages"][stage]["blocks"]
        names = (
            plan["audit"]["populations"]
            if stage == "audit"
            else plan["stages"][stage]["populations"]
        )
        for block in range(blocks):
            for name in names:
                count = 1 if stage == "audit" else populations[name]["count_per_block"]
                for index in range(count):
                    seed = frame_seed(
                        plan["protocol_sha256"], stage, block, name, index
                    )
                    yield stage, block, name, index, seed, effective_seed(seed)


def validate_seed_uniqueness(plan):
    raw, effective = set(), set()
    for _stage, _block, _name, _index, seed, mapped in planned_seed_rows(plan):
        if seed in raw or mapped in effective:
            raise ValueError("raw or effective generator seed collision")
        raw.add(seed)
        effective.add(mapped)
    with np.load(
        plan["inputs"]["seed_exclusions"]["path"], allow_pickle=False
    ) as archive:
        if raw.intersection(map(int, archive["raw_seed"])) or effective.intersection(
            map(int, archive["effective_seed63"])
        ):
            raise ValueError("planned seed reuses historical randomness")
    with np.load(
        plan["inputs"]["planned_seeds"]["path"], allow_pickle=False
    ) as archive:
        recorded = list(
            zip(
                archive["stage"].tolist(),
                archive["block"].tolist(),
                archive["population"].tolist(),
                archive["frame"].tolist(),
                archive["raw_seed"].tolist(),
                archive["effective_seed63"].tolist(),
            )
        )
    expected = list(planned_seed_rows(plan))
    if len(recorded) != len(expected) or set(recorded) != set(expected):
        raise ValueError("planned seed ledger differs from exact generator coordinates")
    return len(raw)


def reference_rows(profile, gamma):
    """Two continuous ideal-reference tones; gamma=|projection mean|²/noise variance."""
    weights = np.asarray(profile.ideal_weights, dtype=np.complex128)
    norms = np.sum(abs(weights) ** 2, axis=1)
    gram = weights @ weights.conj().T
    cross = gram / np.sqrt(norms[:, None] * norms[None, :])
    if np.max(np.abs(cross - np.eye(3))) > 1e-10:
        raise ValueError("ideal target/reference weights are not orthogonal")
    rows = np.zeros((WINDOWS, K), dtype=np.complex128)
    row_index = np.arange(WINDOWS)
    for term, key in [
        (1, "lower_reference_normalized_frequency"),
        (2, "upper_reference_normalized_frequency"),
    ]:
        phase = np.exp(-2j * np.pi * float(profile.layout[key]) * K * row_index)
        rows += (
            math.sqrt(float(gamma) / norms[term])
            * phase[:, None]
            * weights[term][None, :]
        )
    ideal_gamma = abs(rows @ weights.conj().T) ** 2 / norms
    if gamma and not np.allclose(ideal_gamma[:, 1:], gamma, rtol=1e-10, atol=1e-12):
        raise ValueError("reference injection gamma normalization failed")
    return np.asarray(rows, dtype=np.complex64)


def projection_diagnostics(atsc_rows, tones, amplitude, profile):
    data = np.asarray(atsc_rows, np.complex128) * np.float32(amplitude)
    reference = np.asarray(tones, np.complex128)
    out = {}
    for kind, weights in [
        ("ideal", profile.ideal_weights),
        ("unpacked_int4_weights", profile.float_packed_weights),
    ]:
        weights = np.asarray(weights, np.complex128)
        norm = np.sum(abs(weights) ** 2, axis=1)

        def power(rows):
            return np.mean(abs(rows @ weights.conj().T) ** 2 / norm, axis=0)

        a, b, total = power(data), power(reference), power(data + reference)
        out[kind] = {
            "atsc_only_gamma_by_term": a.tolist(),
            "reference_tones_only_gamma_by_term": b.tolist(),
            "combined_gamma_by_term": total.tolist(),
            "coherent_cross_term_by_term": (total - a - b).tolist(),
        }
    out["reference_tones_sample_power"] = float(np.mean(abs(reference) ** 2))
    out["scope"] = (
        "Deterministic prequantization projection/noise ratios; gamma=lambda/2 per proper-complex projection; includes ATSC reference leakage and coherent cross terms"
    )
    return out


def load_prepared(plan):
    inp = plan["inputs"]
    config = json.loads(Path(inp["cache_config"]["path"]).read_text())
    config_copy = dict(config)
    stored = config_copy.pop("config_sha256")
    if (
        stored
        != hashlib.sha256(
            json.dumps(config_copy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("cache configuration digest mismatch")
    files = {
        key: inp[key]["path"]
        for key in ("input_iq", "waveform_audit", "weights", "lib")
    }
    files["input_scale"] = plan["geometry"]["input_scale"]
    files.update(
        {
            k: v
            for k, v in config["calibration"].items()
            if k
            in (
                "pilot_below_data_db",
                "dtv_bandwidth_hz",
                "bin_enbw_hz",
                "pilot_capture_efficiency",
            )
        }
    )
    files["iq_sample_rate_hz"] = config["inputs"]["iq_sample_rate_hz"]
    args = args_for(files, Path(inp["cache_config"]["path"]).parent)
    profile = mrf._prepare(args)
    norms = weight_term_norms_sq(profile.packed_weights, bits_per_component=4)
    if (int(norms[0]), int(norms[1] + norms[2])) != (6372, 12713):
        raise ValueError("channel29 packed coefficient norms differ")
    if (
        cgs._cache_path(args.output_dir, profile).resolve()
        != Path(inp["signal_cache"]["path"]).resolve()
    ):
        raise ValueError("prepared cache location differs")
    if (
        config["geometry"]["spectral_sense"] != "normal"
        or config["geometry"]["input_scale"] != args.input_scale
    ):
        raise ValueError("cache coordinate or input scale mismatch")
    if (
        config["inputs"]["input_iq_sha256"] != inp["input_iq"]["sha256"]
        or config["inputs"]["weights_sha256"] != inp["weights"]["sha256"]
    ):
        raise ValueError("prepared waveform/weight identities differ")
    atsc, _tone, meta = cgs._load_signal_cache(args, config, profile)
    gpu = cgs._initialize_gpu(args)
    if runtime_fingerprint(gpu) != plan["runtime"]:
        raise ValueError("runtime differs from frozen plan")
    return args, profile, atsc, meta, gpu


def population_source(pop, args, profile, atsc, meta, gpu):
    amplitude = (
        0.0
        if pop["shelf_db"] is None
        else cgs._signal_amplitude_for_snr(
            pop["shelf_db"], clean_iq_power=meta["clean_iq_power"], args=args
        )
    )
    tones = reference_rows(profile, pop["reference_projection_gamma"])
    combined = np.asarray(np.float32(amplitude) * atsc + tones, dtype=np.complex64)
    source = mrf.DeviceFrameSource(args, profile, combined, gpu=gpu)
    return source, amplitude, projection_diagnostics(atsc, tones, amplitude, profile)


def coarse_from_packed(source, packed):
    cp, kernel = source.cp, source.gpu.kernel
    diagnostic = cp.zeros(1, dtype=cp.float32)
    handle = kernel.create_raw(
        int(packed.shape[0]), packed.data.ptr, diagnostic.data.ptr
    )
    powers = cp.zeros(3, dtype=cp.uint64)
    try:
        kernel.compute_powers_u64(
            handle, source.profile.packed_weights.ctypes.data, powers.data.ptr
        )
        cp.cuda.Device().synchronize()
        return cp.asnumpy(powers).astype(np.uint64, copy=False)
    finally:
        kernel.destroy(handle)


def exact_cpu_powers(packed, weights, chunk_rows=4096):
    total = np.zeros(3, dtype=np.uint64)
    for start in range(0, len(packed), chunk_rows):
        projection = matched_filter_row_projections_cpu_reference_packed(
            packed[start : start + chunk_rows], weights, 4
        ).astype(np.int64)
        # K128 int4 rows have |component|<=12544, so squaring fits signed64.
        total += np.sum(
            projection[:, :, 0] ** 2 + projection[:, :, 1] ** 2, axis=1, dtype=np.uint64
        )
    return total


def validate_receipt(plan, output):
    path = Path(plan["evaluation_prerequisite"]["receipt_path"])
    if not path.is_file():
        raise ValueError("evaluation requires a frozen calibration receipt")
    receipt = json.loads(path.read_text())
    if (
        receipt.get("schema") != "channel29-residual-calibration-receipt-v1"
        or receipt.get("status") != "calibration_frozen"
    ):
        raise ValueError("calibration receipt status/schema invalid")
    if receipt.get("plan_sha256") != plan["plan_sha256"] or receipt.get(
        "receipt_sha256"
    ) != digest_payload(receipt, "receipt_sha256"):
        raise ValueError("calibration receipt digest/plan binding mismatch")
    frozen = utc_time(receipt["frozen_at"])
    if not utc_time(plan["frozen_at"]) <= frozen <= datetime.now(timezone.utc):
        raise ValueError("receipt timestamp does not follow frozen plan")
    check_identity(receipt["calibration"])
    calibration = json.loads(Path(receipt["calibration"]["path"]).read_text())
    if (
        calibration.get("schema") != "channel29-digital-retained-set-bound-v1"
        or calibration.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise ValueError("calibration payload belongs to another experiment")
    if calibration.get("source_shards") != receipt["calibration_outputs"]:
        raise ValueError("calibration source identities differ from receipt")
    if not utc_time(plan["frozen_at"]) < utc_time(calibration["frozen_at"]) <= frozen:
        raise ValueError("calibration payload timestamp is out of order")
    status = calibration.get("calibration", {}).get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(
            "frozen calibration must report its outcome, including refusal"
        )
    expected = {
        str(Path("calibration") / f"block-{b:04d}" / (name + ".npz"))
        for b in range(plan["stages"]["calibration"]["blocks"])
        for name in plan["stages"]["calibration"]["populations"]
    }
    if set(receipt["calibration_outputs"]) != expected:
        raise ValueError("receipt must bind every planned calibration shard exactly")
    for relative, wanted in receipt["calibration_outputs"].items():
        if sha(Path(output) / relative) != wanted:
            raise ValueError("calibration shard changed after fit")
    return {
        "path": str(path),
        "sha256": sha(path),
        "receipt_sha256": receipt["receipt_sha256"],
        "frozen_at": receipt["frozen_at"],
        "calibration": receipt["calibration"],
        "calibration_status": status,
    }


def validate_shard(path, plan, stage, block, pop, *, receipt=None):
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != sha(path):
        raise ValueError("existing shard is not content authenticated")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"].item()))
        for key, value in {
            "schema": SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "protocol_sha256": plan["protocol_sha256"],
            "stage": stage,
            "block_index": block,
            "population": pop["name"],
            "frames": pop["count_per_block"],
            "geometry": plan["geometry"],
            "target_norm_sq": 6372,
            "reference_norm_sum_sq": 12713,
            "nominal_injected_data_shelf_db": pop["shelf_db"],
            "reference_projection_gamma": pop["reference_projection_gamma"],
            "nominal_injected_data_shelf_linear": 0.0
            if pop["shelf_db"] is None
            else 10.0 ** (pop["shelf_db"] / 10.0),
        }.items():
            if meta.get(key) != value:
                raise ValueError("existing shard metadata differs")
        boundary = utc_time(plan["frozen_at"])
        if stage in ("evaluation", "stress"):
            if receipt is None or meta.get("calibration_receipt") != receipt:
                raise ValueError(
                    "existing shard is not bound to the frozen calibration receipt"
                )
            boundary = utc_time(receipt["frozen_at"])
        if (
            not boundary
            < utc_time(meta["started_utc"])
            <= utc_time(meta["completed_utc"])
            <= datetime.now(timezone.utc)
        ):
            raise ValueError(
                "existing shard timestamps do not follow frozen prerequisites"
            )
        n = pop["count_per_block"]
        if (
            data["coarse_marginals_u64"].shape != (n, 3)
            or data["coarse_marginals_u64"].dtype != np.uint64
        ):
            raise ValueError("bad exact-power shard shape/type")
        for key in ("frame_seed", "frame_seed_effective63"):
            if data[key].shape != (n,) or data[key].dtype != np.uint64:
                raise ValueError("bad seed shape/type")
        expected = np.array(
            [
                frame_seed(plan["protocol_sha256"], stage, block, pop["name"], i)
                for i in range(n)
            ],
            dtype=np.uint64,
        )
        if not np.array_equal(data["frame_seed"], expected) or not np.array_equal(
            data["frame_seed_effective63"], expected % np.uint64(1 << 63)
        ):
            raise ValueError("existing shard seeds differ")
        if (
            data["clip_fraction"].shape != (n,)
            or not np.all(np.isfinite(data["clip_fraction"]))
            or np.any((data["clip_fraction"] < 0) | (data["clip_fraction"] > 1))
        ):
            raise ValueError("bad clipping fractions")
    return meta


def audit_device(plan, output, prepared):
    args, profile, atsc, meta, gpu = prepared
    directory = Path(output) / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "device_audit.json"
    if target.exists():
        raise FileExistsError("audit output already exists")
    populations = {p["name"]: p for p in plan["populations"]}
    rows = []
    for name in plan["audit"]["populations"]:
        started = utc_now()
        source, amplitude, diagnostics = population_source(
            populations[name], args, profile, atsc, meta, gpu
        )
        seed = frame_seed(plan["protocol_sha256"], "audit", 0, name, 0)
        packed, clip = source.packed(seed=seed, amplitude=1.0)
        host = source.host_packed(seed=seed, amplitude=1.0)
        device = source.cp.asnumpy(packed)
        cpu = exact_cpu_powers(host, profile.packed_weights)
        gpu_powers = coarse_from_packed(source, packed)
        raw = directory / "block-0000" / (name + ".npz")
        raw.parent.mkdir(parents=True, exist_ok=True)
        audit_meta = {
            "schema": SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "protocol_sha256": plan["protocol_sha256"],
            "stage": "audit",
            "block_index": 0,
            "population": name,
            "frames": 1,
            "started_utc": started,
            "completed_utc": utc_now(),
            "geometry": plan["geometry"],
            "target_norm_sq": 6372,
            "reference_norm_sum_sq": 12713,
            "nominal_injected_data_shelf_db": populations[name]["shelf_db"],
            "nominal_injected_data_shelf_linear": 0.0
            if populations[name]["shelf_db"] is None
            else 10.0 ** (populations[name]["shelf_db"] / 10.0),
            "reference_projection_gamma": populations[name][
                "reference_projection_gamma"
            ],
            "signal_amplitude": amplitude,
            "projection_diagnostics": diagnostics,
            "truth_scope": "Fresh CPU/GPU audit control; nominal prequantization data-shelf truth; not a science sample",
        }
        with raw.open("xb") as stream:
            np.savez_compressed(
                stream,
                coarse_marginals_u64=gpu_powers[None, :],
                clip_fraction=np.array([clip]),
                frame_seed=np.array([seed], dtype=np.uint64),
                frame_seed_effective63=np.array(
                    [effective_seed(seed)], dtype=np.uint64
                ),
                meta_json=np.asarray(
                    json.dumps(audit_meta, sort_keys=True, allow_nan=False)
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
        with raw.with_suffix(".npz.sha256").open("x") as stream:
            stream.write(sha(raw) + "\n")
        rows.append(
            {
                "population": name,
                "raw_path": str(raw.resolve()),
                "raw_sha256": sha(raw),
                "frame_seed": seed,
                "effective63": effective_seed(seed),
                "packing_identical": bool(np.array_equal(device, host)),
                "coarse_powers_identical": bool(np.array_equal(cpu, gpu_powers)),
                "cpu_powers": [int(x) for x in cpu],
                "gpu_powers": [int(x) for x in gpu_powers],
                "clip_fraction": clip,
                "signal_amplitude": amplitude,
                "projection_diagnostics": diagnostics,
            }
        )
        print(
            f"Audit {name}: packing={rows[-1]['packing_identical']} exact_powers={rows[-1]['coarse_powers_identical']}",
            flush=True,
        )
    passed = all(r["packing_identical"] and r["coarse_powers_identical"] for r in rows)
    validate_plan(plan)
    result = {
        "schema": "channel29-residual-device-audit-v1",
        "plan_sha256": plan["plan_sha256"],
        "created_utc": utc_now(),
        "passed": passed,
        "rows": rows,
        "runtime": runtime_fingerprint(gpu),
        "source_checks_passed": True,
        "source_artifacts_sha256": hashlib.sha256(
            canonical(plan["source_artifacts"])
        ).hexdigest(),
        "inputs_sha256": hashlib.sha256(canonical(plan["inputs"])).hexdigest(),
    }
    write_new(target, result)
    target.with_suffix(".json.sha256").write_text(sha(target) + "\n")
    if not passed:
        raise ValueError("CPU/GPU exact audit failed")
    return result


def require_audit(plan, output):
    path = Path(output) / "audit/device_audit.json"
    if (
        not path.is_file()
        or not path.with_suffix(".json.sha256").is_file()
        or sha(path) != path.with_suffix(".json.sha256").read_text().strip()
    ):
        raise ValueError("science generation requires the authenticated CPU/GPU audit")
    audit = json.loads(path.read_text())
    if (
        audit.get("plan_sha256") != plan["plan_sha256"]
        or audit.get("passed") is not True
        or audit.get("runtime") != plan["runtime"]
    ):
        raise ValueError("CPU/GPU audit does not match this plan/runtime")
    rows = audit.get("rows", [])
    if {r.get("population") for r in rows} != set(plan["audit"]["populations"]) or len(
        rows
    ) != len(plan["audit"]["populations"]):
        raise ValueError("CPU/GPU audit population coverage differs")
    if (
        audit.get("source_checks_passed") is not True
        or audit.get("source_artifacts_sha256")
        != hashlib.sha256(canonical(plan["source_artifacts"])).hexdigest()
        or audit.get("inputs_sha256")
        != hashlib.sha256(canonical(plan["inputs"])).hexdigest()
    ):
        raise ValueError("CPU/GPU audit source/input identity differs")
    for row in rows:
        seed = frame_seed(plan["protocol_sha256"], "audit", 0, row["population"], 0)
        if (
            row.get("frame_seed") != seed
            or row.get("effective63") != effective_seed(seed)
            or row.get("packing_identical") is not True
            or row.get("coarse_powers_identical") is not True
        ):
            raise ValueError("CPU/GPU audit seed or exact-arithmetic proof failed")
        raw = check_identity({"path": row["raw_path"], "sha256": row["raw_sha256"]})
        pop = next(p for p in plan["populations"] if p["name"] == row["population"])
        validate_shard(raw, plan, "audit", 0, {**pop, "count_per_block": 1})


def generate(plan, stage, output, prepared):
    args, profile, atsc, meta, gpu = prepared
    require_audit(plan, output)
    receipt = (
        validate_receipt(plan, output) if stage in ("evaluation", "stress") else None
    )
    populations = {p["name"]: p for p in plan["populations"]}
    norms = weight_term_norms_sq(profile.packed_weights, bits_per_component=4)
    expected_paths = {
        Path(output) / stage / f"block-{block:04d}" / (name + ".npz")
        for block in range(plan["stages"][stage]["blocks"])
        for name in plan["stages"][stage]["populations"]
    }
    if set((Path(output) / stage).glob("block-*/*.npz")) - expected_paths:
        raise ValueError("unexpected raw shards exist in prescribed stage")
    completed = []
    for block in range(plan["stages"][stage]["blocks"]):
        for name in plan["stages"][stage]["populations"]:
            pop = populations[name]
            path = Path(output) / stage / f"block-{block:04d}" / (name + ".npz")
            if path.exists():
                validate_shard(path, plan, stage, block, pop, receipt=receipt)
                completed.append(
                    {"path": str(path.resolve()), "sha256": sha(path), "resumed": True}
                )
                continue
            source, amplitude, diagnostics = population_source(
                pop, args, profile, atsc, meta, gpu
            )
            n = pop["count_per_block"]
            powers = np.empty((n, 3), dtype=np.uint64)
            clip = np.empty(n, dtype=np.float64)
            seeds = np.array(
                [
                    frame_seed(plan["protocol_sha256"], stage, block, name, i)
                    for i in range(n)
                ],
                dtype=np.uint64,
            )
            started, tick = utc_now(), time.monotonic()
            for index, seed in enumerate(seeds):
                packed, clip[index] = source.packed(seed=int(seed), amplitude=1.0)
                powers[index] = coarse_from_packed(source, packed)
            metadata = {
                "schema": SCHEMA,
                "plan_sha256": plan["plan_sha256"],
                "protocol_sha256": plan["protocol_sha256"],
                "stage": stage,
                "block_index": block,
                "population": name,
                "frames": n,
                "started_utc": started,
                "completed_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - tick,
                "geometry": plan["geometry"],
                "target_norm_sq": int(norms[0]),
                "reference_norm_sum_sq": int(norms[1] + norms[2]),
                "nominal_injected_data_shelf_db": pop["shelf_db"],
                "nominal_injected_data_shelf_linear": 0.0
                if pop["shelf_db"] is None
                else 10.0 ** (pop["shelf_db"] / 10.0),
                "reference_projection_gamma": pop["reference_projection_gamma"],
                "signal_amplitude": amplitude,
                "projection_diagnostics": diagnostics,
                "calibration_receipt": receipt,
                "truth_scope": "Nominal injected ATSC data shelf before quantization under the audited waveform/PFB normalization; not physical post-filter residual; reference stress does not redefine target-data truth",
                "coordinate": "post_spectral_sense_normalized normal; no extra CHIME raw-sense reversal",
                "seed_mapping": "raw64 first8 SHA256 bytes big-endian; CuPy uses raw64%(1<<63)",
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    coarse_marginals_u64=powers,
                    clip_fraction=clip,
                    frame_seed=seeds,
                    frame_seed_effective63=seeds % np.uint64(1 << 63),
                    meta_json=np.asarray(
                        json.dumps(metadata, sort_keys=True, allow_nan=False)
                    ),
                )
                stream.flush()
                os.fsync(stream.fileno())
            with path.with_suffix(".npz.sha256").open("x") as stream:
                stream.write(sha(path) + "\n")
            validate_shard(path, plan, stage, block, pop, receipt=receipt)
            completed.append(
                {"path": str(path.resolve()), "sha256": sha(path), "resumed": False}
            )
            print(
                f"{stage} block{block:04d} {name}: {n} frames in {metadata['elapsed_seconds']:.2f}s",
                flush=True,
            )
    validate_plan(plan)
    if stage in ("evaluation", "stress") and validate_receipt(plan, output) != receipt:
        raise ValueError("calibration receipt changed during evaluation")
    report = {
        "schema": "channel29-residual-generation-v1",
        "plan_sha256": plan["plan_sha256"],
        "stage": stage,
        "completed_utc": utc_now(),
        "shards": completed,
        "calibration_receipt": receipt,
        "runtime": runtime_fingerprint(gpu),
        "science_certification": False,
    }
    target = Path(output) / stage / "generation.json"
    if not target.exists():
        write_new(target, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-config", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare_config:
        if args.plan or args.stage or args.audit_only:
            parser.error("prepare is separate from control generation")
        result = prepare(json.loads(args.prepare_config.read_text()), args.output)
        print(
            json.dumps(
                {
                    "prepared": str(args.output),
                    "runtime": result["runtime"],
                    "science_frames_generated": 0,
                },
                indent=2,
            )
        )
        return
    if not args.plan or not args.stage:
        parser.error("generation/audit require --plan and --stage")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    count = validate_seed_uniqueness(plan)
    print(f"Validated {count} disjoint planned raw/effective seeds", flush=True)
    prepared = load_prepared(plan)
    if args.audit_only:
        audit_device(plan, args.output, prepared)
    else:
        generate(plan, args.stage, args.output, prepared)


if __name__ == "__main__":
    main()
