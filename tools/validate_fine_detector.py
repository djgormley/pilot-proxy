#!/usr/bin/env python3
"""Freeze and execute a resumable literal full-array fine-detector campaign.

This driver preserves powers, not only thresholded counts. Calibration, grid
discovery, evaluation and spatial-scaling draws have separate frozen identities.
An interrupted shard is regenerated from exactly the same identities; completed
shards are verified before reuse. A failed scientific gate is a result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral
import re
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAIL = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.testbench.evaluate_snr import _ideal_float_weights_from_layout
from pilot_proxy.testbench.fine_validation_gpu import (
    ALL_STAGES,
    FineValidationEngine,
    prepare_null_input,
)
from pilot_proxy.testbench.fine_validation_scores import construct_geometry
from pilot_proxy.testbench.sensitivity_study import (
    STAGE_FIXED_FLOAT_DECISION,
    STAGE_FIXED_Q16_CPU,
)

FLOAT_STAGES = tuple(x for x in ALL_STAGES if x != STAGE_FIXED_FLOAT_DECISION)
M_VALUES = [2**i for i in range(12)]
SCHEMA = "literal-fine-validation-campaign-v1"
SCIENCE_COUNTS = {
    "null_calibration": 4000,
    "null_validation": 10000,
    "discovery": 24,
    "evaluation": 512,
    "stress": 128,
    "spatial_null": 256,
    "spatial_signal": 256,
}
CASE_NAMES = [
    "nominal",
    "aligned",
    "half",
    "cfo_m1000",
    "cfo_p1000",
    "cfo_m1400",
    "cfo_p1400",
]
COORDINATE_SYSTEM = "native-inverted-adapted"
RAW_ARRAY_NAMES = {
    "float_fine",
    "float_coarse",
    "fixed_fine",
    "fixed_coarse",
    "clip_count",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def seed_for(phase, index, streams=2048):
    identity = f"{SCHEMA}:2026-09-09:{phase}:{streams}:{index}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")


def payload_for(phase, index):
    if phase == "discovery":
        return 0
    if phase == "stress":
        return 1
    # A predeclared independently randomized 50:50 mixture of two fixed,
    # qualified waveform fixtures. Not two draws from all possible ATSC signals.
    identity = f"{SCHEMA}:fixture-choice:{phase}:{index}".encode()
    return 2 + (hashlib.sha256(identity).digest()[0] & 1)


def gpu_context(lib):
    import cupy as cp

    return SimpleNamespace(cp=cp, kernel=FStatKernel(Path(lib)))


def runtime(gpu):
    prop = gpu.cp.cuda.runtime.getDeviceProperties(0)
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "cupy": gpu.cp.__version__,
        "cuda_runtime": gpu.cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": gpu.cp.cuda.runtime.driverGetVersion(),
        "device": prop["name"].decode(),
        "kernel_version": gpu.kernel.version.as_string(),
        "kernel_specs": gpu.kernel.specs.as_descriptive_dict(),
    }


def load_cache(cache, index):
    paths = list((Path(cache) / "cases").glob(f"{index:04d}_*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one cache receipt for index {index}")
    receipt = json.loads(paths[0].read_text())
    path = paths[0].with_suffix(".npz")
    if sha(path) != receipt["array_sha256"]:
        raise ValueError(f"Cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        embedded = json.loads(str(z["meta_json"].item()))
        if any(receipt.get(k) != v for k, v in embedded.items()):
            raise ValueError("Cache receipt differs from embedded metadata")
        if receipt["array_file"] != path.name:
            raise ValueError("Cache receipt array filename differs")
        for name in ("atsc_rows", "ideal_tone_rows"):
            if (
                z[name].shape != (128, 128)
                or z[name].dtype != np.complex64
                or not np.isfinite(z[name]).all()
            ):
                raise ValueError("Cache array geometry/type/finite-value check failed")
        return receipt, z["atsc_rows"], z["ideal_tone_rows"]


def _int(value, name, minimum=0, maximum=(1 << 64) - 1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its declared range")
    return value


def _aware(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp must be an explicit timezone-aware ISO string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _relative(out, relative):
    if not isinstance(relative, str):
        raise ValueError("Artifact binding must use a safe relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("Artifact binding must use a safe relative path")
    resolved = (Path(out) / path).resolve()
    if not resolved.is_relative_to(Path(out).resolve()):
        raise ValueError("Artifact binding escapes its experiment")
    return resolved


def seal_marker(path):
    path = Path(path)
    with path.with_suffix(".sha256").open("x") as stream:
        stream.write(sha(path) + "\n")


def verify_marker(path):
    path = Path(path)
    expected = path.with_suffix(".sha256").read_text().strip()
    if not re.fullmatch("[0-9a-f]{64}", expected) or sha(path) != expected:
        raise ValueError(f"Frozen artifact digest differs: {path}")
    return expected


def _snapshot_relative(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        return str(Path("source_snapshots") / path.relative_to(ROOT))
    return str(Path("source_snapshots/external") / (sha(path)[:16] + "_" + path.name))


def validate_cache_identity(cache_plan, index, receipt):
    profiles = cache_plan["profiles"]
    if [p["physical_channel"] for p in profiles] != list(range(14, 37)):
        raise ValueError("Cache must contain all23 ordered physical profiles")
    if any([case["name"] for case in p["cases"]] != CASE_NAMES for p in profiles) or [
        p["index"] for p in cache_plan["payloads"]
    ] != list(range(4)):
        raise ValueError("Cache case/payload coverage differs")
    index = _int(index, "cache index", 0, 643)
    profile = profiles[index // 28]
    case, payload = (index % 28) // 4, index % 4
    if (
        receipt["case_index"] != index
        or receipt["profile_channel"] != profile["physical_channel"]
        or receipt["case"] != profile["cases"][case]
        or receipt["case"]["name"] != CASE_NAMES[case]
        or receipt["payload"] != cache_plan["payloads"][payload]
        or receipt["packed_profile_sha256"] != profile["packed_profile_sha256"]
    ):
        raise ValueError(f"Cache Cartesian identity differs at {index}")
    if receipt.get("passed") is not True:
        raise ValueError(f"Cache {index} failed its engineering gates")


def validate_plan_contract(plan, identities):
    engineering = plan["scope"] == "engineering_only"
    if plan["scope"] not in ("engineering_only", "digital_literal_M2048_validation"):
        raise ValueError("Unsupported study scope")
    counts = {k: 2 for k in SCIENCE_COUNTS} if engineering else SCIENCE_COUNTS
    if plan["counts"] != counts or any(
        type(v) is not int for v in plan["counts"].values()
    ):
        raise ValueError("Frozen counts differ from the declared campaign")
    if (
        plan["channels"] != list(range(14, 37))
        or plan["streams"] != 2048
        or plan["windows"] != 128
        or plan["window_samples"] != 128
        or plan["fine_bins"] != 256
        or plan["float_stages"] != list(FLOAT_STAGES)
        or plan["fixed_stage"] != STAGE_FIXED_FLOAT_DECISION
        or plan["coordinate_system"] != COORDINATE_SYSTEM
    ):
        raise ValueError("Frozen geometry or stage order differs")
    _aware(plan["frozen_utc"])
    if plan.get("physical_certification") is not False:
        raise ValueError("Digital campaign cannot certify physical performance")
    if plan["discovery_snr_db"] != list(range(-66, -20, 3)):
        raise ValueError("Frozen discovery grid differs")
    expected = []
    for phase, count in counts.items():
        for streams in M_VALUES if phase.startswith("spatial_") else [2048]:
            for trial in range(count):
                seed = seed_for(
                    ("engineering_" if engineering else "") + phase, trial, streams
                )
                expected.append(
                    (
                        phase,
                        streams,
                        trial,
                        seed,
                        seed % (1 << 63),
                        payload_for(phase, trial),
                    )
                )
    if identities.dtype.names != (
        "phase",
        "streams",
        "trial",
        "raw_seed",
        "effective_seed63",
        "payload",
    ):
        raise ValueError("Identity columns differ")
    expected_dtypes = {
        "streams": np.uint32,
        "trial": np.uint32,
        "raw_seed": np.uint64,
        "effective_seed63": np.uint64,
        "payload": np.uint8,
    }
    if any(
        identities.dtype.fields[name][0] != np.dtype(dtype)
        for name, dtype in expected_dtypes.items()
    ) or identities.dtype.fields["phase"][0] != np.dtype("U24"):
        raise ValueError("Identity dtypes differ from exact frozen encoding")
    if identities.ndim != 1 or len(identities) != len(expected):
        raise ValueError("Identity coverage differs")
    actual = [
        (
            str(v["phase"]),
            int(v["streams"]),
            int(v["trial"]),
            int(v["raw_seed"]),
            int(v["effective_seed63"]),
            int(v["payload"]),
        )
        for v in identities
    ]
    if actual != expected or len({r[4] for r in actual}) != len(actual):
        raise ValueError("Frozen noise/fixture identities differ or collide")


def validate_raw_arrays(data, metadata):
    if set(data) not in (RAW_ARRAY_NAMES, RAW_ARRAY_NAMES | {"meta_json"}):
        raise ValueError("Raw shard array inventory differs")
    if metadata["float_stages"] != list(FLOAT_STAGES):
        raise ValueError("Raw shard stage order differs")
    start, stop = _int(metadata["start"], "start"), _int(metadata["stop"], "stop", 1)
    n = stop - start
    phase = metadata["phase"]
    if (
        phase not in SCIENCE_COUNTS
        or n < 1
        or n > (32 if phase.startswith("null_") else 16)
    ):
        raise ValueError("Raw shard phase or trial interval differs")
    streams = _int(metadata["streams"], "streams", 1, 2048)
    if streams not in M_VALUES or (
        not phase.startswith("spatial_") and streams != 2048
    ):
        raise ValueError("Raw shard stream geometry differs")
    seeds = metadata["raw_seeds"]
    if (
        len(seeds) != n
        or len({_int(seed, "raw seed") % (1 << 63) for seed in seeds}) != n
    ):
        raise ValueError("Raw shard seed coverage differs")
    if phase.startswith("null_"):
        if metadata["channels"] != list(range(14, 37)):
            raise ValueError("Null shard must preserve all23 channel profiles")
        shape = (n, 23)
    else:
        _int(metadata["channel"], "channel", 14, 36)
        _int(metadata["case_index"], "case", 0, 6)
        snrs = metadata["snr_db"]
        if not isinstance(snrs, list) or not 1 <= len(snrs) <= 61:
            raise ValueError("Raw shard SNR coverage differs")
        if phase == "spatial_null":
            if snrs != [None]:
                raise ValueError("Spatial null must have no injected signal")
        elif any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in snrs
        ):
            raise ValueError("Raw shard SNRs must be finite")
        if len(metadata["payloads"]) != n or any(
            _int(v, "payload", 0, 3) != payload_for(phase, start + i)
            for i, v in enumerate(metadata["payloads"])
        ):
            raise ValueError("Raw shard fixture assignment differs")
        shape = (n, len(snrs))
    expected = {
        "float_fine": (shape + (len(FLOAT_STAGES), 3, 256), np.float64),
        "float_coarse": (shape + (len(FLOAT_STAGES), 3), np.float64),
        "fixed_fine": (shape + (3, 256), np.uint64),
        "fixed_coarse": (shape + (3,), np.uint64),
        "clip_count": (shape, np.uint32),
    }
    for name, (target_shape, dtype) in expected.items():
        values = np.asarray(data[name])
        if values.shape != target_shape or values.dtype != np.dtype(dtype):
            raise ValueError(f"Raw array shape/dtype differs: {name}")
        if dtype is np.float64 and (
            not np.isfinite(values).all() or np.any(values < 0)
        ):
            raise ValueError(f"Raw floating power is not finite nonnegative: {name}")
        if name == "clip_count" and np.any(values > streams * 128 * 128):
            raise ValueError("Raw clipping count exceeds sample count")
        if name.startswith("fixed_") and np.any(values >= (1 << 63)):
            raise ValueError(
                "Raw fixed power exceeds proven signed accumulation capacity"
            )


def _bindings(out, bindings, *, grid=False):
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Frozen dependent artifact has no input bindings")
    allowed = (
        "raw/null_calibration/",
        "scores/null_calibration/",
        "source_snapshots/",
        "reduction/source_snapshots/",
    )
    singletons = {
        "plan.json",
        "trial-identities.npy",
        "reduction/source-binding.json",
        "reduction/cache-plan.json",
    }
    if grid:
        allowed += ("raw/discovery/", "scores/discovery/")
        singletons |= {"calibration.json", "calibration.sha256"}
    for relative, expected in bindings.items():
        if relative not in singletons and not relative.startswith(allowed):
            raise ValueError(
                "Dependent artifact includes forbidden or unrecognized inputs"
            )
        if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
            raise ValueError("Malformed source digest")
        if sha(_relative(out, relative)) != expected:
            raise ValueError(f"Dependent artifact input changed: {relative}")


def _null_coverage(out, plan, bindings):
    identities = np.load(out / "trial-identities.npy", allow_pickle=False)
    ids = identities[identities["phase"] == "null_calibration"]
    expected_paths = set()
    for start in range(0, plan["counts"]["null_calibration"], 32):
        selected = ids[start : start + 32]
        metadata = {
            "plan_sha256": sha(out / "plan.json"),
            "phase": "null_calibration",
            "start": start,
            "stop": start + len(selected),
            "channels": plan["channels"],
            "streams": 2048,
            "float_stages": list(FLOAT_STAGES),
            "raw_seeds": [int(v) for v in selected["raw_seed"]],
            "axes": ["trial", "channel", "stage_if_float", "term", "bin_if_fine"],
        }
        relative = f"raw/null_calibration/{start:06d}.npz"
        if (
            relative not in bindings
            or str(Path(relative).with_suffix(".json")) not in bindings
        ):
            raise ValueError(
                "Frozen calibration lacks full raw null-calibration coverage"
            )
        if not completed(out / relative, metadata):
            raise ValueError("Null-calibration shard is missing")
        expected_paths.add(relative)
    actual = {
        str(p.relative_to(out)) for p in (out / "raw/null_calibration").glob("*.npz")
    }
    if actual != expected_paths:
        raise ValueError("Unexpected or incomplete null-calibration raw inventory")


def _producer_contract(artifact, plan, status):
    if (
        artifact.get("status") != status
        or artifact.get("refusals") != []
        or artifact.get("scope") != plan["scope"]
        or artifact.get("coordinate_system") != plan["coordinate_system"]
        or artifact.get("physical_certification") is not False
    ):
        raise ValueError("Dependent artifact status/scope/coordinate differs")
    sources = artifact.get("source_sha256")
    reducer = str(ROOT / "tools/reduce_fine_validation.py")
    if not isinstance(sources, dict) or reducer not in sources:
        raise ValueError("Dependent artifact omits its frozen producer")
    for path, digest in sources.items():
        if plan["source_sha256"].get(path) != digest or sha(path) != digest:
            raise ValueError("Dependent artifact producer differs from frozen source")


def _calibration_thresholds(artifact, plan):
    n = plan["counts"]["null_calibration"]
    if (
        artifact.get("null_trials") != n
        or artifact.get("independent_null_validation") is not False
    ):
        raise ValueError("Calibration sample count or independence scope differs")
    cache = json.loads((Path(plan["cache_path"]) / "plan.json").read_text())
    profiles = {p["physical_channel"]: p for p in cache["profiles"]}
    channels = {f"ch{c}" for c in plan["channels"]}
    if set(artifact["fine"]) != channels or set(artifact["coarse"]) != channels:
        raise ValueError("Calibration omits channel thresholds")
    pfas = plan["diagnostic_pfa"] + plan["primary_pfa"]
    stages = (*FLOAT_STAGES, STAGE_FIXED_FLOAT_DECISION, STAGE_FIXED_Q16_CPU)

    def check(records, exact=False):
        if set(records) != {str(pfa) for pfa in pfas}:
            raise ValueError("Calibration Pfa coverage differs")
        for pfa in pfas:
            r = records[str(pfa)]
            if (
                r.get("status") != "available"
                or r.get("trials") != n
                or r.get("nominal_pfa") != pfa
                or r.get("independent_validation") is not False
            ):
                raise ValueError("Calibration threshold status/count differs")
            value = r["threshold"]
            if exact:
                _int(value, "Q16 threshold", 1)
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Calibration threshold must be finite positive")

    for channel in plan["channels"]:
        cases = profiles[channel]["cases"]
        anchors = {int(c["calibrated_anchor_bin_inverted"]) for c in cases}
        anchors.add(int(cases[0]["nominal_anchor_bin_inverted"]))
        table = artifact["fine"][f"ch{channel}"]
        if set(table) != {str(a) for a in anchors}:
            raise ValueError("Calibration anchor coverage differs")
        for anchor in anchors:
            ranks = {str(int(v)) for v in construct_geometry(anchor).ranks}
            if set(table[str(anchor)]) != set(stages):
                raise ValueError("Calibration stage coverage differs")
            for stage in stages:
                records = table[str(anchor)][stage]
                if set(records) != ranks:
                    raise ValueError("Calibration rank coverage differs")
                for record in records.values():
                    check(record, exact=stage == STAGE_FIXED_Q16_CPU)
        coarse = artifact["coarse"][f"ch{channel}"]
        if set(coarse) != set(stages[:-1]):
            raise ValueError("Calibration coarse-stage coverage differs")
        for records in coarse.values():
            check(records)


def verify_calibration(out, plan):
    path = out / "calibration.json"
    verify_marker(path)
    artifact = json.loads(path.read_text())
    if artifact["schema"] != "literal-fine-calibration-v1" or artifact[
        "plan_sha256"
    ] != sha(out / "plan.json"):
        raise ValueError("Calibration schema/plan binding differs")
    _producer_contract(artifact, plan, "calibrated")
    _calibration_thresholds(artifact, plan)
    frozen = _aware(artifact["frozen_utc"])
    if not _aware(plan["frozen_utc"]) <= frozen <= datetime.now(timezone.utc):
        raise ValueError("Calibration freeze time is invalid")
    bindings = artifact["calibration_inputs_sha256"]
    _bindings(out, bindings)
    _null_coverage(out, plan, bindings)
    for relative in bindings:
        if relative.startswith("raw/") and relative.endswith(".json"):
            if (
                _aware(
                    json.loads(_relative(out, relative).read_text())["completed_utc"]
                )
                > frozen
            ):
                raise ValueError("Calibration was frozen before its inputs completed")
    return artifact


def verify_evaluation_grid(out, plan):
    calibration = verify_calibration(out, plan)
    path = out / "evaluation-grid.json"
    verify_marker(path)
    artifact = json.loads(path.read_text())
    if artifact["schema"] != "literal-fine-evaluation-grid-v1" or artifact[
        "plan_sha256"
    ] != sha(out / "plan.json"):
        raise ValueError("Evaluation grid schema/plan binding differs")
    _producer_contract(artifact, plan, "frozen")
    if (
        artifact.get("expected_cells") != 161
        or artifact.get("curves_per_cell") != 26
        or artifact.get("evaluation_outcomes_used") is not False
    ):
        raise ValueError("Grid scope or evaluation separation differs")
    frozen = _aware(artifact["frozen_utc"])
    if not _aware(calibration["frozen_utc"]) <= frozen <= datetime.now(timezone.utc):
        raise ValueError("Evaluation grid freeze time is invalid")
    bindings = artifact["calibration_inputs_sha256"]
    _bindings(out, bindings, grid=True)
    if bindings.get("calibration.json") != sha(
        out / "calibration.json"
    ) or bindings.get("calibration.sha256") != sha(out / "calibration.sha256"):
        raise ValueError("Grid does not bind the frozen calibration")
    if not set(calibration["calibration_inputs_sha256"]) <= set(bindings):
        raise ValueError("Grid omits calibration dependencies")
    keys = {
        f"ch{channel}/case{case}" for channel in plan["channels"] for case in range(7)
    }
    if set(artifact["grids"]) != keys:
        raise ValueError("Evaluation grid must cover all161 declared cells")
    for values in artifact["grids"].values():
        if not isinstance(values, list) or not 1 <= len(values) <= 61:
            raise ValueError("Evaluation grid point count differs")
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < -66
                or value > -21
                or abs((value + 66) / 0.75 - round((value + 66) / 0.75)) > 1e-10
            ):
                raise ValueError(
                    "Evaluation grid violates the frozen SNR range/lattice"
                )
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError(
                "Evaluation grid must be strictly ascending without duplicates"
            )
    discovery_paths = [
        p for p in bindings if p.startswith("raw/discovery/") and p.endswith(".npz")
    ]
    expected = {
        f"raw/discovery/ch{c}_case{k}_M2048_{start:06d}.npz"
        for c in plan["channels"]
        for k in range(7)
        for start in range(0, plan["counts"]["discovery"], 16)
    }
    actual = {str(p.relative_to(out)) for p in (out / "raw/discovery").glob("*.npz")}
    if set(discovery_paths) != expected or actual != expected:
        raise ValueError("Grid lacks complete discovery coverage")
    ids = np.load(out / "trial-identities.npy", allow_pickle=False)
    ids = ids[ids["phase"] == "discovery"]
    for relative in expected:
        path = out / relative
        receipt = path.with_suffix(".json")
        if str(receipt.relative_to(out)) not in bindings:
            raise ValueError("Grid omits a discovery receipt")
        match = re.fullmatch(r"ch(\d+)_case(\d+)_M2048_(\d{6})\.npz", path.name)
        channel, case, start = map(int, match.groups())
        selected = ids[start : start + 16]
        metadata = {
            "plan_sha256": sha(out / "plan.json"),
            "phase": "discovery",
            "channel": channel,
            "case_index": case,
            "streams": 2048,
            "start": start,
            "stop": start + len(selected),
            "snr_db": plan["discovery_snr_db"],
            "float_stages": list(FLOAT_STAGES),
            "raw_seeds": [int(v) for v in selected["raw_seed"]],
            "payloads": [int(v) for v in selected["payload"]],
            "axes": ["trial", "snr", "stage_if_float", "term", "bin_if_fine"],
        }
        if not completed(path, metadata):
            raise ValueError("Grid discovery input identity differs")
        if _aware(json.loads(receipt.read_text())["completed_utc"]) > frozen:
            raise ValueError("Grid was frozen before discovery inputs completed")
    return artifact


@contextmanager
def generation_lock(out):
    with (Path(out) / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another writer is using this experiment") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def seed_file_values(path):
    path = Path(path)
    loaded = np.load(path, allow_pickle=False)
    try:
        fields = loaded.files if hasattr(loaded, "files") else loaded.dtype.names
        if fields is None or not ({"raw_seed", "effective_seed63"} & set(fields)):
            raise ValueError(f"Seed ledger has no recognized exact columns: {path}")
        excluded = set()
        for name in ("raw_seed", "effective_seed63", "conservative_seed32"):
            if name not in fields:
                continue
            values = loaded[name]
            if values.dtype.kind not in "iu":
                raise ValueError("Seed ledger columns must be exact integers")
            bound = (
                1
                << (
                    32
                    if name == "conservative_seed32"
                    else (63 if name == "effective_seed63" else 64)
                )
            ) - 1
            excluded.update(
                _int(int(v), name, 0, bound) % (1 << 63) for v in values.flat
            )
        return excluded
    finally:
        if hasattr(loaded, "close"):
            loaded.close()


def engineering_exclusions():
    base = RAIL / "output/full-fine-validation-2026-09-09"
    values, files = set(range(20260909002, 20260909008)), {}
    for plan_path in base.glob("engineering-preflight-v*/plan.json"):
        plan = json.loads(plan_path.read_text())
        files[str(plan_path)] = sha(plan_path)
        root = int(plan["base_seed"])
        for streams, channels in (
            (8, plan["profiles_small_M8"]),
            (2048, plan["profiles_full_M2048"]),
        ):
            for channel in channels:
                seed = root + streams * 100 + channel
                values.add(seed % (1 << 63))
                if streams == 2048:
                    values.add((seed + 1000000) % (1 << 63))
                    values.update(
                        (seed + 2000000 + i) % (1 << 63)
                        for i in range(plan["timing_repeats"])
                    )
        values.update(
            (root + 9000000 + i) % (1 << 63) for i in range(plan["timing_repeats"])
        )
        values.add((root + 9990000) % (1 << 63))
    for path in base.glob("runner-engineering-*/trial-identities.npy"):
        values.update(seed_file_values(path))
        files[str(path)] = sha(path)
    return values, files


def freeze(args):
    out, cache = args.output.resolve(), args.cache.resolve()
    if out.exists():
        raise ValueError("Freeze requires a new experiment directory")
    cp = json.loads((cache / "plan.json").read_text())
    if cp["cache_count"] != 644 or cp["runtime"]["backend"] != "gpu":
        raise ValueError("Unexpected cache plan geometry/backend")
    cache_files = {"plan.json": sha(cache / "plan.json")}
    if (
        sha(cp["weights_path"]) != cp["weights_sha256"]
        or sha(Path(cp["weights_path"]).with_suffix(".bin.manifest.json"))
        != cp["weight_manifest_sha256"]
    ):
        raise ValueError("Cache and selected weight artifacts differ")
    cache_cases = []
    for i in range(644):
        receipt, _, _ = load_cache(cache, i)
        validate_cache_identity(cp, i, receipt)
        if receipt["plan_sha256"] != cache_files["plan.json"]:
            raise ValueError("Cache plan binding differs")
        if receipt["passed"] is not True:
            raise ValueError(f"Cache {i} failed its frozen engineering gates")
        path = next((cache / "cases").glob(f"{i:04d}_*.json"))
        cache_files[str(path.relative_to(cache))] = sha(path)
        cache_files[str(path.with_suffix(".npz").relative_to(cache))] = receipt[
            "array_sha256"
        ]
        cache_cases.append(
            {
                "index": i,
                "channel": receipt["profile_channel"],
                "payload": receipt["payload"]["index"],
                "case": receipt["case"],
            }
        )
    gpu = gpu_context(args.lib)
    counts = dict(SCIENCE_COUNTS)
    if args.engineering:
        counts = {k: 2 for k in counts}
    identities = []
    for phase, count in counts.items():
        for streams in M_VALUES if phase.startswith("spatial_") else [2048]:
            for index in range(count):
                seed = seed_for(
                    ("engineering_" if args.engineering else "") + phase, index, streams
                )
                identities.append(
                    (
                        phase,
                        streams,
                        index,
                        seed,
                        seed % (1 << 63),
                        payload_for(phase, index),
                    )
                )
    effective = [v[4] for v in identities]
    if len(set(effective)) != len(effective):
        raise ValueError("Planned effective seed collision")
    exclusions = set()
    exclusion_files = {}
    prior = RAIL / "results/channel29_residual_controls_2026-09-09/preflight"
    for path in [
        prior / "seed-exclusions.npz",
        prior / "planned-frame-seeds.npz",
        RAIL
        / "results/fine_detector_validation_2026-09-09/checks/engineering-seed-exclusions.npz",
        *getattr(args, "seed_exclusions", []),
    ]:
        path = Path(path).resolve()
        exclusions.update(seed_file_values(path))
        exclusion_files[str(path)] = sha(path)
    engineering_seeds, engineering_files = engineering_exclusions()
    # Repeated engineering preflights intentionally replay their same declared
    # engineering namespace. Scientific seeds must avoid all of them.
    if not args.engineering:
        exclusions.update(engineering_seeds)
    exclusion_files.update(engineering_files)
    prep = json.loads((Path(cp["preparation"]) / "plan.json").read_text())
    exclusions.update(int(v) % (1 << 63) for v in prep["payload_seeds"])
    exclusions.update(int(v) % (1 << 63) for v in cp["gain_seeds"].values())
    if set(effective) & exclusions:
        raise ValueError("Planned seeds overlap previous development observations")
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if (
            source
            and Path(source).suffix == ".py"
            and Path(source).is_relative_to(ROOT)
        ):
            sources[str(Path(source).resolve())] = sha(source)
    for path in [
        ROOT / "src/pilot_proxy/testbench/fine_validation_stats.py",
        ROOT / "src/pilot_proxy/testbench/fine_validation_scores.py",
        ROOT / "tools/reduce_fine_validation.py",
    ]:
        if not path.is_file():
            raise ValueError(f"Required frozen analysis source is not ready: {path}")
        sources[str(path)] = sha(path)
    # CUDA source, transform tables, wrapper and binary are separate identities.
    for path in (ROOT / "cuda").rglob("*"):
        if path.is_file() and path.suffix in (".cu", ".cuh", ".h", ".cpp"):
            sources[str(path)] = sha(path)
    for path in [
        args.lib.resolve(),
        Path(cp["weights_path"]),
        Path(cp["weights_path"]).with_suffix(".bin.manifest.json"),
    ]:
        sources[str(path)] = sha(path)
    out.mkdir(parents=True)
    source_snapshot_paths = {path: _snapshot_relative(path) for path in sources}
    for path in sources:
        target = out / source_snapshot_paths[path]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    dtype = [
        ("phase", "U24"),
        ("streams", "u4"),
        ("trial", "u4"),
        ("raw_seed", "u8"),
        ("effective_seed63", "u8"),
        ("payload", "u1"),
    ]
    np.save(
        out / "trial-identities.npy",
        np.asarray(identities, dtype=dtype),
        allow_pickle=False,
    )
    input_powers = {}
    for payload in cp["payloads"]:
        if sha(payload["path"]) != payload["sha256"]:
            raise ValueError("Payload waveform changed before amplitude normalization")
        source_sha = payload["sha256"]
        exclusion_files[str(Path(payload["path"]).resolve())] = source_sha
        with Path(payload["path"]).open("rb") as stream:
            stream.seek(payload["window_start"] * 8)
            iq = np.fromfile(
                stream,
                dtype=np.complex64,
                count=payload["window_stop_exclusive"] - payload["window_start"],
            )
        if (
            iq.size != payload["window_stop_exclusive"] - payload["window_start"]
            or not np.isfinite(iq).all()
        ):
            raise ValueError("Payload normalization window is incomplete or nonfinite")
        power = float(np.mean(np.abs(iq.astype(np.complex128)) ** 2))
        if not math.isfinite(power) or power <= 0:
            raise ValueError("Payload normalization power must be finite and positive")
        input_powers[str(payload["index"])] = power
    plan = {
        "schema": SCHEMA,
        "frozen_utc": utc(),
        "scope": "engineering_only"
        if args.engineering
        else "digital_literal_M2048_validation",
        "counts": counts,
        "channels": list(range(14, 37)),
        "streams": 2048,
        "windows": 128,
        "window_samples": 128,
        "fine_bins": 256,
        "coordinate_system": COORDINATE_SYSTEM,
        "coordinate_rule": "Immutable normalized caches are conjugated then reversed within each K128 row, preserving slow windows and streams. No additional phase/gain conversion. Use inverted anchors. Independently drawn circular Gaussian noise is added after this map; this is distributionally equivalent but not sample-identical to transforming a pre-adapter draw. Synthetic symmetric [-7,7] quantization commutes with this map; real -8-rail effects remain outside this conditional model.",
        "float_stages": list(FLOAT_STAGES),
        "fixed_stage": STAGE_FIXED_FLOAT_DECISION,
        "cache_path": str(cache),
        "cache_sha256": cache_files,
        "cache_cases": cache_cases,
        "source_sha256": sources,
        "source_snapshot_paths": source_snapshot_paths,
        "lib": str(args.lib.resolve()),
        "weights_path": cp["weights_path"],
        "runtime": runtime(gpu),
        "identities_sha256": sha(out / "trial-identities.npy"),
        "prior_seed_files_sha256": exclusion_files,
        "engineering_effective_seed_exclusions": sorted(engineering_seeds),
        "seed_exclusion_count": len(exclusions),
        "seed_rule": "First eight SHA256 bytes big endian; see frozen seed_for. CuPy receives raw modulo2^63.",
        "pairing": "One independent noise identity per phase/trial/M, reused across all profiles, offsets, stages and SNRs. Profiles and SNRs are paired observations, never extra independent trials.",
        "waveform_rule": "Calibration/discovery payload00; evaluation independently randomized equal mixture of fixed qualified payload02 and03, same choice across paired SNR/profile/case. Payload01 is a separate failed-stationarity stress family.",
        "input_iq_power_by_payload": input_powers,
        "shelf_coordinate": {
            "bandwidth_hz": 6e6,
            "iq_sample_rate_hz": cp["pfb"]["source_iq_sample_rate_hz"],
            "pilot_below_data_db": 11.3,
            "definition": "Requested model shelf SNR. amplitude^2=10^(SNR/10)*(6MHz/Fs)*(1+10^(-11.3/10))/input_IQ_mean_power. PFB signal normalized by frozen unit-IQ-noise response; independent unit noise is then added at coarse-PFB output. This is conditional digital input truth, not a measured physical shelf transfer.",
        },
        "primary_pfa": [0.01, 0.05],
        "diagnostic_pfa": [0.001],
        "null_rule": "Separate empirical higher-quantile calibration for every stage/anchor/rank; strict comparison. Never report calibration trials as independent null validation. Empirical max over D includes the searched-bin effect; no second trials-factor correction.",
        "geometry": {
            "designated_half_width": 2,
            "guard_native_bins": 1,
            "guard_rule": "Guard the anchor only, then exclude the five designated padded bins; even-bin bulk has125 or126 entries.",
            "rank_fractions": [0.25, 0.5, 0.75, 1.0],
            "rank_rule": "ceil(fraction*n_bulk)-1, zero-based",
            "primary_rank_fraction": 0.5,
            "anchors": "Declared calibrated static anchor for each known CFO; separately replay fixed nominal anchor. No evaluation peak fitting and no unknown-offset acquisition claim.",
        },
        "false_alarm_gate": {
            "confidence": 0.95,
            "family_tests": 1334,
            "family": "23profiles*7calibrated anchors*4ranks*2primaryPfa +23coarse*2primaryPfa; duplicates retained in conservative family count.",
            "cap_multiple": 2.0,
            "pointwise_width_max_multiple_nominal": 1.0,
            "interpretation": "Simultaneous upper cap at twice nominal is an engineering tolerance, not a claim of exact nominal Pfa. Float stages and fixed-nominal stress have pointwise diagnostics outside this primary family.",
        },
        "discovery_snr_db": list(range(-66, -20, 3)),
        "evaluation_grid_rule": {
            "source": "Only payload00 discovery and null-calibration outcomes; freeze grids before evaluation.",
            "target_probabilities": [0.5, 0.9],
            "pfa": [0.01, 0.05],
            "ranks": "primary median rank; four ranks subsequently reported on this same grid",
            "curves": "all six stages and exact Q16 fine, and all six coarse paths",
            "algorithm": "Union every raw adjacent up/down/plateau discovery crossing bracket of target .5 or .9; add1.5dB on each side; sample at0.75dB, clipped to[-66,-21]. If any curve lacks a crossing, retain the entire discovery grid as additional evaluation points. No evaluation-dependent extension, smoothing or extrapolation.",
            "max_points": 61,
            "inference": "Linear crossing interpolation is a descriptive model; report raw brackets and grid discretization bounds separately. An unbracketed or ambiguous curve remains unresolved.",
        },
        "loss_rule": {
            "targets": [0.5, 0.9],
            "bootstrap_replicates": 2000,
            "resampling": "Paired null recalibration and paired H1 IDs across stages/SNR; retain every censored draw, withhold interval if <95percent uniquely bracketed.",
            "equivalence_margin_db": 0.10,
            "pointwise_and_simultaneous": "Pointwise paired intervals do not establish all-profile equivalence. All-cell claim additionally requires a simultaneous bound and discretization allowance; otherwise not established.",
            "failure_rule": "No precision, margin, sample-count, seed or grid retuning after evaluation.",
        },
        "spatial": {
            "streams": M_VALUES,
            "snr_db": -45.0,
            "case": "aligned",
            "scope": "All23profiles; independent-noise scaling. Replicated-stream benchmark is derived from M1 powers and explicitly marked correlated; never counted as M independent observations.",
            "theory_exception": "Ideal single-bin F(2,4) at M1 has infinite variance; theoretical SD omitted there. Raw sample SD descriptive; robust quantile width remains finite.",
        },
        "stress": {
            "payload": 1,
            "snr_grid": "Same grid frozen from payload00, no adaptation to stress outcomes.",
        },
        "raw_storage": "Full float64 five-stage powers and exact uint64 fixed powers/coarse sums; stage order and identities embedded in every shard, file hashes in receipts.",
        "physical_certification": False,
        "open_physical_scope": [
            "Telescope feed covariance and measured PFB thermal correlations",
            "Independent transmitter states and visibility transfer",
            "Radio-path and live Pathfinder acceptance",
        ],
    }
    validate_plan_contract(
        plan, np.load(out / "trial-identities.npy", allow_pickle=False)
    )
    write_new(out / "plan.json", plan)
    seal_marker(out / "plan.json")
    print(
        json.dumps(
            {
                "status": "frozen",
                "plan_sha256": sha(out / "plan.json"),
                "counts": counts,
                "effective_seed_collisions": 0,
            }
        ),
        flush=True,
    )


def verify_plan(out, *, verify_cache=False):
    verify_marker(out / "plan.json")
    plan = json.loads((out / "plan.json").read_text())
    if plan["schema"] != SCHEMA:
        raise ValueError("Unsupported study schema")
    for path, expected in plan["source_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen generating source changed: {path}")
    if set(plan["source_snapshot_paths"]) != set(plan["source_sha256"]):
        raise ValueError("Frozen source snapshot inventory differs")
    for path, relative in plan["source_snapshot_paths"].items():
        if (
            not relative.startswith("source_snapshots/")
            or sha(_relative(out, relative)) != plan["source_sha256"][path]
        ):
            raise ValueError("Frozen source snapshot changed or is missing")
    if sha(out / "trial-identities.npy") != plan["identities_sha256"]:
        raise ValueError("Trial identities changed")
    validate_plan_contract(
        plan, np.load(out / "trial-identities.npy", allow_pickle=False)
    )
    for path, expected in plan["prior_seed_files_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Historical seed ledger changed")
    cache = Path(plan["cache_path"])
    selected = (
        plan["cache_sha256"]
        if verify_cache
        else {"plan.json": plan["cache_sha256"]["plan.json"]}
    )
    for path, expected in selected.items():
        if sha(cache / path) != expected:
            raise ValueError(f"Frozen cache changed: {path}")
    return plan


def engine_for(plan, gpu, bank, channel, case, payload, streams=2048):
    cache_index = (channel - 14) * 28 + case * 4 + payload
    _, atsc, tone = load_cache(plan["cache_path"], cache_index)
    if plan["coordinate_system"] != COORDINATE_SYSTEM:
        raise ValueError("Generating coordinate system differs")
    atsc = np.ascontiguousarray(np.conjugate(atsc[:, ::-1]))
    tone = np.ascontiguousarray(np.conjugate(tone[:, ::-1]))
    packed, valid = bank.get_weights_for_physical_channel(channel)
    if not valid:
        raise ValueError("Invalid weight profile")
    profile = SimpleNamespace(
        packed_weights=packed,
        ideal_weights=_ideal_float_weights_from_layout(
            bank.layout_for_physical_channel(channel), detector_window_samples=128
        ),
    )
    return FineValidationEngine(profile, atsc, tone, gpu=gpu, num_streams=streams)


def amplitude(plan, snr, payload):
    spec = plan["shelf_coordinate"]
    return math.sqrt(
        10 ** (float(snr) / 10)
        * spec["bandwidth_hz"]
        / spec["iq_sample_rate_hz"]
        * (1 + 10 ** (-spec["pilot_below_data_db"] / 10))
        / plan["input_iq_power_by_payload"][str(payload)]
    )


def arrays(shape):
    return {
        "float_fine": np.empty((*shape, len(FLOAT_STAGES), 3, 256), dtype=np.float64),
        "float_coarse": np.empty((*shape, len(FLOAT_STAGES), 3), dtype=np.float64),
        "fixed_fine": np.empty((*shape, 3, 256), dtype=np.uint64),
        "fixed_coarse": np.empty((*shape, 3), dtype=np.uint64),
        "clip_count": np.empty(shape, dtype=np.uint32),
    }


def record(target, index, result):
    for prefix, source in [("fine", "powers_by_stage"), ("coarse", "coarse_by_stage")]:
        if set(result[source]) != set(ALL_STAGES):
            raise ValueError("Engine stage inventory differs")
        for stage in ALL_STAGES:
            values = np.asarray(result[source][stage])
            expected_dtype = (
                np.uint64 if stage == STAGE_FIXED_FLOAT_DECISION else np.float64
            )
            if values.dtype != expected_dtype or values.shape != (
                (3, 256) if prefix == "fine" else (3,)
            ):
                raise ValueError("Engine output shape/dtype differs before storage")
        target["float_" + prefix][index] = np.stack(
            [result[source][s] for s in FLOAT_STAGES]
        )
        target["fixed_" + prefix][index] = result[source][STAGE_FIXED_FLOAT_DECISION]
    target["clip_count"][index] = _int(
        result["clip_count"], "clip count", 0, 2048 * 128 * 128
    )


def completed(path, metadata):
    receipt = path.with_suffix(".json")
    if not path.exists() and not receipt.exists():
        return False
    if not path.exists() or not receipt.exists():
        raise ValueError(
            f"Incomplete committed shard; investigate before resuming: {path}"
        )
    saved = json.loads(receipt.read_text())
    if saved["metadata"] != metadata or sha(path) != saved["sha256"]:
        raise ValueError(f"Shard identity/hash differs: {path}")
    with np.load(path, allow_pickle=False) as z:
        if json.loads(str(z["meta_json"].item())) != metadata:
            raise ValueError("Embedded shard identity differs")
        validate_raw_arrays(z, metadata)
    return True


def save_shard(path, data, metadata, elapsed):
    validate_raw_arrays(data, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing.npz")
    # Only this transaction's uncommitted temporary file may be replaced.
    with temporary.open("wb") as stream:
        np.savez(
            stream, **data, meta_json=np.asarray(json.dumps(metadata, sort_keys=True))
        )
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("Refusing to overwrite a committed shard")
    os.rename(temporary, path)
    write_new(
        path.with_suffix(".json"),
        {
            "metadata": metadata,
            "sha256": sha(path),
            "elapsed_seconds": elapsed,
            "completed_utc": utc(),
        },
    )


def execute(args):
    with generation_lock(args.output.resolve()):
        return _execute_unlocked(args)


def _execute_unlocked(args):
    out = args.output.resolve()
    plan = verify_plan(out, verify_cache=True)
    gpu = gpu_context(plan["lib"])
    if runtime(gpu) != plan["runtime"]:
        raise ValueError("Generating runtime differs from frozen plan")
    bank = DetectorWeightBank(explicit_path=plan["weights_path"])
    identities = np.load(out / "trial-identities.npy", allow_pickle=False)
    phase = args.stage
    if phase == "null_validation":
        verify_calibration(out, plan)
    ids = identities[identities["phase"] == phase]
    plan_hash = sha(out / "plan.json")
    if phase.startswith("null_"):
        engines = {c: engine_for(plan, gpu, bank, c, 0, 0) for c in plan["channels"]}
        for start in range(0, len(ids), 32):
            selected = ids[start : start + 32]
            meta = {
                "plan_sha256": plan_hash,
                "phase": phase,
                "start": start,
                "stop": start + len(selected),
                "channels": plan["channels"],
                "streams": 2048,
                "float_stages": list(FLOAT_STAGES),
                "raw_seeds": [int(v) for v in selected["raw_seed"]],
                "axes": ["trial", "channel", "stage_if_float", "term", "bin_if_fine"],
            }
            path = out / "raw" / phase / f"{start:06d}.npz"
            if completed(path, meta):
                continue
            data = arrays((len(selected), len(engines)))
            begin = time.monotonic()
            for i, identity in enumerate(selected):
                seed = int(identity["raw_seed"])
                prepared = prepare_null_input(seed, num_streams=2048, cp=gpu.cp)
                for j, channel in enumerate(plan["channels"]):
                    result = engines[channel].evaluate(seed, 0, null_input=prepared)
                    record(data, (i, j), result)
                del prepared
            save_shard(path, data, meta, time.monotonic() - begin)
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "completed_trials": start + len(selected),
                        "planned_trials": len(ids),
                        "elapsed_shard_seconds": time.monotonic() - begin,
                    }
                ),
                flush=True,
            )
    else:
        spatial = phase.startswith("spatial_")
        grids = None
        if phase in ("evaluation", "stress"):
            grids = verify_evaluation_grid(out, plan)
        channels = plan["channels"] if args.channel is None else args.channel
        if not set(channels) <= set(plan["channels"]):
            raise ValueError("Unplanned channel")
        for channel in channels:
            for case in [1] if spatial else range(7):
                if phase == "discovery":
                    snrs = plan["discovery_snr_db"]
                elif spatial:
                    snrs = [
                        plan["spatial"]["snr_db"] if phase == "spatial_signal" else None
                    ]
                else:
                    snrs = grids["grids"][f"ch{channel}/case{case}"]
                for streams in M_VALUES if spatial else [2048]:
                    selected_ids = ids[ids["streams"] == streams]
                    engines = {}
                    for start in range(0, len(selected_ids), 16):
                        selected = selected_ids[start : start + 16]
                        meta = {
                            "plan_sha256": plan_hash,
                            "phase": phase,
                            "channel": channel,
                            "case_index": case,
                            "streams": streams,
                            "start": start,
                            "stop": start + len(selected),
                            "snr_db": snrs,
                            "float_stages": list(FLOAT_STAGES),
                            "raw_seeds": [int(v) for v in selected["raw_seed"]],
                            "payloads": [int(v) for v in selected["payload"]],
                            "axes": [
                                "trial",
                                "snr",
                                "stage_if_float",
                                "term",
                                "bin_if_fine",
                            ],
                        }
                        if grids is not None:
                            meta["evaluation_grid_sha256"] = sha(
                                out / "evaluation-grid.json"
                            )
                        path = (
                            out
                            / "raw"
                            / phase
                            / f"ch{channel}_case{case}_M{streams}_{start:06d}.npz"
                        )
                        if completed(path, meta):
                            continue
                        data = arrays((len(selected), len(snrs)))
                        begin = time.monotonic()
                        for i, identity in enumerate(selected):
                            payload, seed = (
                                int(identity["payload"]),
                                int(identity["raw_seed"]),
                            )
                            if payload not in engines:
                                engines[payload] = engine_for(
                                    plan, gpu, bank, channel, case, payload, streams
                                )
                            engine = engines[payload]
                            for j, snr in enumerate(snrs):
                                value = (
                                    0 if snr is None else amplitude(plan, snr, payload)
                                )
                                result = engine.evaluate(seed, value, reuse_noise=True)
                                record(data, (i, j), result)
                            engine.clear_noise_cache()
                        save_shard(path, data, meta, time.monotonic() - begin)
                        print(
                            json.dumps(
                                {
                                    "phase": phase,
                                    "channel": channel,
                                    "case": case,
                                    "streams": streams,
                                    "completed_trials": start + len(selected),
                                    "planned_trials": len(selected_ids),
                                    "snr_count": len(snrs),
                                    "elapsed_shard_seconds": time.monotonic() - begin,
                                }
                            ),
                            flush=True,
                        )
                    del engines
                    gpu.cp.get_default_memory_pool().free_all_blocks()
    verify_plan(out)
    write_new(
        out / "stage_receipts" / f"{phase}" / f"{utc().replace(':', '-')}.json",
        {
            "phase": phase,
            "plan_sha256": plan_hash,
            "completed_utc": utc(),
            "selected_channels": args.channel,
            "note": "Execution receipt; final coverage requires enumerating every planned shard.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "freeze",
            "null_calibration",
            "null_validation",
            "discovery",
            "evaluation",
            "stress",
            "spatial_null",
            "spatial_signal",
        ],
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=RAIL / "results/fine_detector_validation_2026-09-09/cache",
    )
    parser.add_argument("--lib", type=Path, default=ROOT / "cuda/libfstatistic.so")
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--channel", type=int, nargs="+")
    parser.add_argument(
        "--seed-exclusions",
        type=Path,
        action="append",
        default=[],
        help="Additional immutable exact raw_seed/effective_seed63 NPY/NPZ ledger",
    )
    args = parser.parse_args()
    freeze(args) if args.stage == "freeze" else execute(args)


if __name__ == "__main__":
    main()
