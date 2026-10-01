#!/usr/bin/env python3
"""Reduce complete frozen literal powers, calibrate nulls, freeze discovery grids.

These four stages never inspect null-validation or evaluation powers. Sources
are snapshotted before reduction, every declared raw shard is authenticated,
and a missing/invalid curve refuses calibration rather than losing trials.
All outputs are exclusive and preserve their input/source identities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pilot_proxy.testbench.fine_validation_scores import (
    construct_geometry, decode_requirements, fixed_scores, float_scores,
)
from pilot_proxy.testbench.fine_validation_stats import (
    empirical_null_threshold, exact_q16_null_threshold, raw_crossing_brackets,
)
from pilot_proxy.testbench.sensitivity_study import FLOAT_RESPONSE_STAGES, STAGE_FIXED_Q16_CPU

PLAN_SCHEMA = "literal-fine-validation-campaign-v1"
SCORE_SCHEMA = "literal-fine-scores-v1"
CAL_SCHEMA = "literal-fine-calibration-v1"
GRID_SCHEMA = "literal-fine-evaluation-grid-v1"
FLOAT_STAGES = tuple(FLOAT_RESPONSE_STAGES)
Q16_STAGE = STAGE_FIXED_Q16_CPU
GRID = np.arange(-66.0, -20.999, 0.75)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rehash(out, mapping):
    if not mapping:
        raise ValueError("input hash map must be nonempty")
    for name, expected in mapping.items():
        path = Path(name)
        path = path if path.is_absolute() else out / path
        if sha(path) != expected:
            raise ValueError(f"Bound input changed: {path}")


def bind_sources(out):
    path = out / "reduction/source-binding.json"
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    runtime = {"python": platform.python_version(), "python_executable": sys.executable,
               "numpy": np.__version__}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).suffix == ".py" and Path(name).resolve().is_relative_to(ROOT):
            sources[str(Path(name).resolve())] = sha(name)
    if path.exists():
        saved = read_json(path)
        if saved["source_sha256"] != sources:
            raise ValueError("Reducer sources differ from their first frozen reduction")
        if saved["runtime"] != runtime:
            raise ValueError("Reducer runtime differs from its first frozen reduction")
        rehash(out, saved["snapshots_sha256"])
    else:
        snapshots = {}
        for name, digest in sources.items():
            relative = Path("reduction/source_snapshots") / Path(name).relative_to(ROOT)
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("Refusing to overwrite a reduction source snapshot")
            shutil.copyfile(name, target)
            if sha(target) != digest:
                raise ValueError("Source changed while snapshotting")
            snapshots[str(relative)] = digest
        saved = {"schema": "literal-fine-reducer-source-v1", "frozen_utc": utc(),
                 "source_sha256": sources, "snapshots_sha256": snapshots, "runtime": runtime}
        write_json(path, saved)
    return saved, {str(path.relative_to(out)): sha(path), **saved["snapshots_sha256"]}


def load_contract(out):
    plan = read_json(out / "plan.json")
    if plan["schema"] != PLAN_SCHEMA:
        raise ValueError("Unexpected campaign schema")
    if plan["scope"] not in ("engineering_only", "digital_literal_M2048_validation"):
        raise ValueError("Unknown study scope")
    if plan["coordinate_system"] != "native-inverted-adapted":
        raise ValueError("This reducer requires the native-inverted-adapted campaign")
    if plan["scope"] != "engineering_only" and plan["channels"] != list(range(14, 37)):
        raise ValueError("Scientific campaign must include all 23 channels")
    if len(set(plan["channels"])) != len(plan["channels"]) or not plan["channels"]:
        raise ValueError("Duplicate or absent channel")
    if any(c not in range(14, 37) for c in plan["channels"]):
        raise ValueError("Unsupported channel")
    if (plan["streams"], plan["windows"], plan["window_samples"], plan["fine_bins"]) != (2048, 128, 128, 256):
        raise ValueError("Campaign power geometry differs")
    if tuple(plan["float_stages"]) != FLOAT_STAGES[:-1] or plan["fixed_stage"] != FLOAT_STAGES[-1]:
        raise ValueError("Power stage order differs")
    expected = 2 if plan["scope"] == "engineering_only" else 4000
    discovery = 2 if plan["scope"] == "engineering_only" else 24
    if plan["counts"]["null_calibration"] != expected or plan["counts"]["discovery"] != discovery:
        raise ValueError("Unplanned calibration/discovery counts")
    if plan["discovery_snr_db"] != list(range(-66, -20, 3)):
        raise ValueError("Discovery grid differs")
    if plan["primary_pfa"] != [0.01, 0.05] or plan["diagnostic_pfa"] != [0.001]:
        raise ValueError("False-alarm levels differ")
    if (plan["geometry"]["designated_half_width"] != 2
            or plan["geometry"]["guard_native_bins"] != 1
            or plan["geometry"]["rank_fractions"] != [0.25, 0.5, 0.75, 1.0]
            or plan["evaluation_grid_rule"]["target_probabilities"] != [0.5, 0.9]
            or plan["evaluation_grid_rule"]["pfa"] != [0.01, 0.05]
            or plan["evaluation_grid_rule"]["max_points"] != 61):
        raise ValueError("Frozen score/grid geometry differs")
    rehash(out, plan["source_sha256"])
    identity_path = out / "trial-identities.npy"
    if sha(identity_path) != plan["identities_sha256"]:
        raise ValueError("Trial identity digest differs")
    ids = np.load(identity_path, allow_pickle=False)
    if not {"phase", "streams", "trial", "raw_seed", "effective_seed63", "payload"} <= set(ids.dtype.names or ()):
        raise ValueError("Identity schema differs")
    if len(np.unique(ids["effective_seed63"])) != len(ids):
        raise ValueError("Independent identities have effective seed collisions")
    if any(int(a) % (1 << 63) != int(b) for a, b in zip(ids["raw_seed"], ids["effective_seed63"])):
        raise ValueError("Effective seed mapping differs")
    cache_path = Path(plan["cache_path"]) / "plan.json"
    if sha(cache_path) != plan["cache_sha256"]["plan.json"]:
        raise ValueError("Cache plan changed")
    cache = read_json(cache_path)
    profiles = {p["physical_channel"]: p for p in cache["profiles"]}
    geometry = {}
    for channel in plan["channels"]:
        profile = profiles[channel]
        cases = profile["cases"]
        if len(cases) != 7:
            raise ValueError("Each channel needs all seven frozen cases")
        anchors = [int(c["calibrated_anchor_bin_inverted"]) for c in cases]
        nominal = int(cases[0]["nominal_anchor_bin_inverted"])
        if any(int(c["nominal_anchor_bin_inverted"]) != nominal for c in cases):
            raise ValueError("Nominal anchor differs between cases")
        layout = profile["layout"]
        target = int(layout["target_norm_sq"])
        reference = int(layout["reference_norm_sum_sq"])
        if target <= 0 or reference <= 0:
            raise ValueError("Nonpositive profile norm")
        geometry[channel] = {"anchors": anchors + [nominal], "mu0": 2 * target / reference,
                             "norm_target": target, "norm_reference_sum": reference}
    cache_snapshot = out / "reduction/cache-plan.json"
    cache_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not cache_snapshot.exists():
        with cache_snapshot.open("xb") as stream:
            stream.write(cache_path.read_bytes())
    if sha(cache_snapshot) != sha(cache_path):
        raise ValueError("Frozen reducer cache snapshot differs")
    binding = {"plan.json": sha(out / "plan.json"), "trial-identities.npy": sha(identity_path),
               str(cache_snapshot.relative_to(out)): sha(cache_path)}
    return plan, ids, geometry, binding


def phase_ids(plan, ids, phase):
    selected = ids[(ids["phase"] == phase) & (ids["streams"] == 2048)]
    expected = plan["counts"][phase]
    if (len(selected) != expected or np.count_nonzero(ids["phase"] == phase) != expected
            or selected["trial"].tolist() != list(range(expected))):
        raise ValueError(f"Incomplete or unordered identities for {phase}")
    if phase == "discovery" and np.any(selected["payload"] != 0):
        raise ValueError("Discovery must use only calibration payload00")
    return selected


def raw_inventory(out, plan, ids, phase):
    """Authenticate full planned coverage before loading any scientific arrays."""
    selected = phase_ids(plan, ids, phase)
    null = phase == "null_calibration"
    if not null and phase != "discovery":
        raise ValueError("This reducer only consumes calibration and discovery")
    step = 32 if null else 16
    cells = [(None, None)] if null else [(c, k) for c in plan["channels"] for k in range(7)]
    inventory, bindings = [], {}
    for channel, case in cells:
        for start in range(0, len(selected), step):
            chunk = selected[start:start + step]
            name = f"{start:06d}.npz" if null else f"ch{channel}_case{case}_M2048_{start:06d}.npz"
            path = out / "raw" / phase / name
            receipt = path.with_suffix(".json")
            saved = read_json(receipt)
            metadata = saved["metadata"]
            expected = {"plan_sha256": sha(out / "plan.json"), "phase": phase,
                        "start": start, "stop": start + len(chunk), "streams": 2048,
                        "float_stages": plan["float_stages"],
                        "raw_seeds": [int(v) for v in chunk["raw_seed"]]}
            if null:
                expected["channels"] = plan["channels"]
                expected["axes"] = ["trial", "channel", "stage_if_float", "term", "bin_if_fine"]
            else:
                expected.update(channel=channel, case_index=case, snr_db=plan["discovery_snr_db"],
                                payloads=[int(v) for v in chunk["payload"]],
                                axes=["trial", "snr", "stage_if_float", "term", "bin_if_fine"])
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Raw identity differs: {path}")
            digest = sha(path)
            if digest != saved["sha256"]:
                raise ValueError(f"Raw shard hash differs: {path}")
            completed = datetime.fromisoformat(saved["completed_utc"])
            frozen = datetime.fromisoformat(plan["frozen_utc"])
            if completed.tzinfo is None or frozen.tzinfo is None or not frozen <= completed <= datetime.now(timezone.utc):
                raise ValueError("Raw receipt chronology differs")
            inventory.append((path, metadata))
            bindings[str(path.relative_to(out))] = digest
            bindings[str(receipt.relative_to(out))] = sha(receipt)
    actual = {p for p in (out / "raw" / phase).glob("*") if p.suffix in (".npz", ".json")}
    if actual != {p for path, _ in inventory for p in (path, path.with_suffix(".json"))}:
        raise ValueError("Raw stage has missing, extra or undeclared shards")
    return inventory, bindings


def read_raw(path, metadata, second):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"float_fine", "float_coarse", "fixed_fine", "fixed_coarse", "clip_count", "meta_json"}:
            raise ValueError("Unexpected raw array fields")
        if read_json_string(archive["meta_json"]) != metadata:
            raise ValueError("Embedded raw metadata differs")
        n = metadata["stop"] - metadata["start"]
        shapes = {"float_fine": (n, second, 5, 3, 256), "float_coarse": (n, second, 5, 3),
                  "fixed_fine": (n, second, 3, 256), "fixed_coarse": (n, second, 3),
                  "clip_count": (n, second)}
        data = {}
        for name, shape in shapes.items():
            value = archive[name]
            dtype = np.float64 if name.startswith("float") else np.uint32 if name == "clip_count" else np.uint64
            if value.shape != shape or value.dtype != np.dtype(dtype):
                raise ValueError(f"Raw shape/dtype differs for {name}")
            if not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError("Raw powers must be finite nonnegative")
            data[name] = value
        if np.any(data["clip_count"] > 2048 * 128 * 128):
            raise ValueError("Clip count exceeds literal samples")
        return data


def read_json_string(value):
    return json.loads(str(value.item()))


def reduce_powers(float_fine, fixed_fine, float_coarse, fixed_coarse, anchors, mu0):
    """Frame-flat array reducer; six floating ablations plus exact Q16."""
    n = len(fixed_fine)
    fine = np.empty((n, len(anchors), 6, 4))
    valid = np.empty(fine.shape, dtype=bool)
    required = np.zeros((n, len(anchors), 4), dtype=np.uint64)
    always = np.zeros(required.shape, dtype=bool)
    fixed_valid = np.zeros(required.shape, dtype=bool)
    bulk_count = np.empty((n, len(anchors)), dtype=np.uint16)
    cache = {}
    for slot, anchor in enumerate(anchors):
        if anchor not in cache:
            geometry = construct_geometry(anchor)
            floats = [float_scores(float_fine[:, k], geometry) for k in range(5)]
            fixed = fixed_scores(fixed_fine, geometry)
            cache[anchor] = floats, fixed
        floats, fixed = cache[anchor]
        for stage, result in enumerate(floats):
            fine[:, slot, stage] = result["response"]
            valid[:, slot, stage] = result["valid"]
        fine[:, slot, 5] = fixed["fixed_float_response"]
        valid[:, slot, 5] = fixed["valid"]
        required[:, slot] = fixed["required_q16"]
        always[:, slot] = fixed["always_masked"]
        fixed_valid[:, slot] = fixed["valid"]
        bulk_count[:, slot] = fixed["n_bulk"]
    coarse, coarse_valid = np.empty((n, 6)), np.empty((n, 6), dtype=bool)
    geometry = construct_geometry(anchors[0])
    for stage in range(6):
        # Input-only quantization with ideal unit-modulus weights retains mu0=1.
        normalization = 1.0 if stage in (0, 1, 2) else mu0
        power = float_coarse[:, stage] if stage < 5 else fixed_coarse
        # Only the small coarse array is required; tiny placeholder fine avoids
        # inventing another normalization implementation.
        result = float_scores(np.zeros((n, 3, 256)), geometry,
                              coarse_powers=power, mu0=normalization)
        coarse[:, stage], coarse_valid[:, stage] = result["coarse_ratio"], result["coarse_valid"]
    return {"fine_float": fine, "fine_valid": valid, "required_q16": required,
            "always_masked": always, "fixed_valid": fixed_valid, "fixed_n_bulk": bulk_count,
            "coarse_float": coarse, "coarse_valid": coarse_valid}


def save_scores(path, data, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **data, meta_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    receipt = {"schema": SCORE_SCHEMA, "metadata": metadata, "sha256": sha(path), "completed_utc": utc()}
    write_json(path.with_suffix(".json"), receipt)
    return {str(path): sha(path), str(path.with_suffix(".json")): sha(path.with_suffix(".json"))}


def score_stage(out, phase):
    plan, ids, geometries, bindings = load_contract(out)
    source, source_bindings = bind_sources(out)
    bindings.update(source_bindings)
    inventory, raw_bindings = raw_inventory(out, plan, ids, phase)
    bindings.update(raw_bindings)
    target = out / "scores" / phase
    if target.exists():
        raise ValueError("Refusing to overwrite a derived score stage")
    target.mkdir(parents=True)
    outputs = {}
    if phase == "null_calibration":
        pieces = {c: [] for c in plan["channels"]}
        for path, metadata in inventory:
            raw = read_raw(path, metadata, len(plan["channels"]))
            for column, channel in enumerate(plan["channels"]):
                g = geometries[channel]
                pieces[channel].append(reduce_powers(raw["float_fine"][:, column], raw["fixed_fine"][:, column],
                    raw["float_coarse"][:, column], raw["fixed_coarse"][:, column], g["anchors"], g["mu0"]))
        cells = [(c, None, pieces[c]) for c in plan["channels"]]
    else:
        cells = []
        for channel in plan["channels"]:
            g = geometries[channel]
            for case in range(7):
                pieces = []
                for path, metadata in inventory:
                    if (metadata["channel"], metadata["case_index"]) != (channel, case):
                        continue
                    raw = read_raw(path, metadata, len(plan["discovery_snr_db"]))
                    shape = raw["fixed_fine"].shape[:2]
                    flat = lambda a: a.reshape((-1, *a.shape[2:]))
                    reduced = reduce_powers(flat(raw["float_fine"]), flat(raw["fixed_fine"]),
                        flat(raw["float_coarse"]), flat(raw["fixed_coarse"]),
                        [g["anchors"][case], g["anchors"][7]], g["mu0"])
                    pieces.append({k: v.reshape((*shape, *v.shape[1:])) for k, v in reduced.items()})
                cells.append((channel, case, pieces))
    for channel, case, pieces in cells:
        merged = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        expected = plan["counts"][phase]
        if any(len(v) != expected for v in merged.values()):
            raise ValueError("Derived coverage count differs")
        anchors = geometries[channel]["anchors"] if case is None else [geometries[channel]["anchors"][case], geometries[channel]["anchors"][7]]
        metadata = {"schema": SCORE_SCHEMA, "plan_sha256": bindings["plan.json"], "phase": phase,
                    "channel": channel, "case_index": case, "anchors": anchors,
                    "ranks": [construct_geometry(a).ranks.tolist() for a in anchors],
                    "float_stages": list(FLOAT_STAGES), "trials": expected,
                    "snr_db": None if case is None else plan["discovery_snr_db"],
                    "scope": plan["scope"], "mu0_packed_weights": geometries[channel]["mu0"],
                    "coordinate_system": plan["coordinate_system"],
                    "coarse_scope": "Floating normalized coarse responses; fixed raw powers retained separately, not an exact coarse decision replay."}
        path = target / (f"ch{channel}.npz" if case is None else f"ch{channel}_case{case}.npz")
        outputs.update({str(Path(p).relative_to(out)): d for p, d in save_scores(path, merged, metadata).items()})
    rehash(out, bindings)
    rehash(out, source["source_sha256"])
    manifest = {"schema": SCORE_SCHEMA, "plan_sha256": bindings["plan.json"], "phase": phase,
                "scope": plan["scope"], "completed_utc": utc(), "coverage_complete": True,
                "channels": plan["channels"], "trials": plan["counts"][phase],
                "cells": len(cells), "inputs_sha256": bindings, "outputs_sha256": outputs,
                "source_sha256": source["source_sha256"], "physical_certification": False}
    manifest["coordinate_system"] = plan["coordinate_system"]
    write_json(target / "manifest.json", manifest)
    return manifest


def load_score_manifest(out, plan, phase):
    path = out / "scores" / phase / "manifest.json"
    manifest = read_json(path)
    expected_cells = len(plan["channels"]) * (7 if phase == "discovery" else 1)
    if (manifest["schema"] != SCORE_SCHEMA or manifest["plan_sha256"] != sha(out / "plan.json")
            or manifest["phase"] != phase or manifest["coverage_complete"] is not True
            or manifest["trials"] != plan["counts"][phase] or manifest["cells"] != expected_cells
            or manifest["channels"] != plan["channels"]):
        raise ValueError("Incomplete score coverage manifest")
    if manifest["coordinate_system"] != plan["coordinate_system"]:
        raise ValueError("Score coordinate system differs")
    expected = set()
    for channel in plan["channels"]:
        for case in range(7) if phase == "discovery" else [None]:
            stem = f"ch{channel}_case{case}" if case is not None else f"ch{channel}"
            expected.update(f"scores/{phase}/{stem}{ext}" for ext in (".npz", ".json"))
    if set(manifest["outputs_sha256"]) != expected:
        raise ValueError("Score manifest omits or adds a cell")
    ids = np.load(out / "trial-identities.npy", allow_pickle=False)
    _, expected_raw = raw_inventory(out, plan, ids, phase)
    if any(manifest["inputs_sha256"].get(name) != digest for name, digest in expected_raw.items()):
        raise ValueError("Score manifest omits or changes a declared raw dependency")
    for key in ("inputs_sha256", "outputs_sha256", "source_sha256"):
        rehash(out, manifest[key])
    return manifest, {str(path.relative_to(out)): sha(path), **manifest["inputs_sha256"], **manifest["outputs_sha256"]}


def load_scores(path):
    receipt = read_json(path.with_suffix(".json"))
    if sha(path) != receipt["sha256"]:
        raise ValueError("Score file digest differs")
    with np.load(path, allow_pickle=False) as data:
        if read_json_string(data["meta_json"]) != receipt["metadata"]:
            raise ValueError("Score embedded metadata differs")
        metadata = receipt["metadata"]
        if metadata["schema"] != SCORE_SCHEMA or metadata["float_stages"] != list(FLOAT_STAGES):
            raise ValueError("Score schema/stages differ")
        if metadata["coordinate_system"] != "native-inverted-adapted":
            raise ValueError("Score coordinate system differs")
        anchors = metadata["anchors"]
        if metadata["ranks"] != [construct_geometry(a).ranks.tolist() for a in anchors]:
            raise ValueError("Score rank geometry differs")
        lead = (metadata["trials"],) if metadata["snr_db"] is None else (metadata["trials"], len(metadata["snr_db"]))
        layout = {
            "fine_float": ((*lead, len(anchors), 6, 4), np.float64),
            "fine_valid": ((*lead, len(anchors), 6, 4), bool),
            "required_q16": ((*lead, len(anchors), 4), np.uint64),
            "always_masked": ((*lead, len(anchors), 4), bool),
            "fixed_valid": ((*lead, len(anchors), 4), bool),
            "fixed_n_bulk": ((*lead, len(anchors)), np.uint16),
            "coarse_float": ((*lead, 6), np.float64),
            "coarse_valid": ((*lead, 6), bool),
        }
        if set(data.files) != set(layout) | {"meta_json"}:
            raise ValueError("Score fields omitted or added")
        result = {}
        for name, (shape, dtype) in layout.items():
            value = data[name]
            if value.shape != shape or value.dtype != np.dtype(dtype):
                raise ValueError(f"Score shape/dtype differs: {name}")
            result[name] = value
        if np.any(result["always_masked"] & ~result["fixed_valid"]):
            raise ValueError("Invalid rank cannot carry a valid always-masked sentinel")
        return result, metadata


def calibrate(out):
    plan, _, geometries, bindings = load_contract(out)
    source, source_bindings = bind_sources(out)
    bindings.update(source_bindings)
    _, score_bindings = load_score_manifest(out, plan, "null_calibration")
    bindings.update(score_bindings)
    fine, coarse, refusals = {}, {}, []
    pfas = plan["diagnostic_pfa"] + plan["primary_pfa"]
    for channel in plan["channels"]:
        values, metadata = load_scores(out / "scores/null_calibration" / f"ch{channel}.npz")
        if (metadata["channel"] != channel or metadata["case_index"] is not None
                or metadata["phase"] != "null_calibration" or metadata["plan_sha256"] != bindings["plan.json"]
                or metadata["anchors"] != geometries[channel]["anchors"]):
            raise ValueError("Null score cell identity differs")
        n = plan["counts"]["null_calibration"]
        if values["fine_float"].shape != (n, 8, 6, 4) or values["required_q16"].shape != (n, 8, 4):
            raise ValueError("Null score shapes differ")
        fine[f"ch{channel}"], coarse[f"ch{channel}"] = {}, {}
        for slot, anchor in enumerate(metadata["anchors"]):
            if str(anchor) in fine[f"ch{channel}"]:
                continue
            dest = fine[f"ch{channel}"].setdefault(str(anchor), {})
            for stage_index, stage in enumerate((*FLOAT_STAGES, Q16_STAGE)):
                dest[stage] = {}
                for rank_column, rank in enumerate(metadata["ranks"][slot]):
                    dest[stage][str(rank)] = {}
                    try:
                        if stage == Q16_STAGE:
                            sample = decode_requirements(values["required_q16"][:, slot, rank_column],
                                values["always_masked"][:, slot, rank_column], values["fixed_valid"][:, slot, rank_column])
                            calibrator = exact_q16_null_threshold
                        else:
                            if not values["fine_valid"][:, slot, stage_index, rank_column].all():
                                raise ValueError("Invalid fine rank")
                            sample = values["fine_float"][:, slot, stage_index, rank_column]
                            calibrator = empirical_null_threshold
                        for pfa in pfas:
                            record = calibrator(sample, pfa=pfa)
                            dest[stage][str(rank)][str(pfa)] = record
                            if record["status"] != "available":
                                raise ValueError("Null threshold is not deployable")
                    except (ValueError, TypeError) as error:
                        refusals.append({"channel": channel, "anchor": anchor, "stage": stage,
                                         "rank": rank, "reason": str(error)})
        for stage_index, stage in enumerate(FLOAT_STAGES):
            coarse[f"ch{channel}"][stage] = {}
            try:
                if not values["coarse_valid"][:, stage_index].all():
                    raise ValueError("Invalid coarse reference")
                for pfa in pfas:
                    coarse[f"ch{channel}"][stage][str(pfa)] = empirical_null_threshold(values["coarse_float"][:, stage_index], pfa=pfa)
            except (ValueError, TypeError) as error:
                refusals.append({"channel": channel, "stage": stage, "coarse": True, "reason": str(error)})
    report = {"schema": CAL_SCHEMA, "plan_sha256": bindings["plan.json"], "frozen_utc": utc(),
              "scope": plan["scope"], "status": "refused" if refusals else "calibrated",
              "coordinate_system": plan["coordinate_system"],
              "null_trials": plan["counts"]["null_calibration"], "fine": fine, "coarse": coarse,
              "refusals": refusals, "calibration_inputs_sha256": bindings,
              "source_sha256": source["source_sha256"], "physical_certification": False,
              "independent_null_validation": False, "higher_rule": "ceil((1-pfa)*(n-1)); strict response greater than threshold"}
    rehash(out, bindings)
    rehash(out, source["source_sha256"])
    write_json(out / ("calibration-refused.json" if refusals else "calibration.json"), report)
    if refusals:
        raise ValueError(f"Calibration refused for {len(refusals)} curves; see calibration-refused.json")
    with (out / "calibration.sha256").open("x") as stream:
        stream.write(sha(out / "calibration.json") + "\n")
    return report


def curve_grid(snr_db, curves):
    """Exact declared grid union, preserving downward/plateau brackets."""
    snr = np.asarray(snr_db, dtype=float)
    if snr.tolist() != list(range(-66, -20, 3)) or not curves:
        raise ValueError("Full declared discovery grid and nonempty curves are required")
    selected, details, fallback = set(), [], False
    for name, rates in curves.items():
        for target in (0.5, 0.9):
            result = raw_crossing_brackets(snr, rates, target=target)
            segments = [v["indices"] for v in result["upward_brackets"] + result["downward_brackets"]] + result["plateau_segments"]
            if not segments:
                fallback = True
            for lo, hi in segments:
                selected.update(float(v) for v in GRID if snr[lo] - 1.5 <= v <= snr[hi] + 1.5)
            details.append({"curve": name, **result})
    if fallback:
        selected.update(float(v) for v in snr)
    if not selected or len(selected) > 61:
        raise ValueError("Frozen grid has invalid size")
    return sorted(selected), {"fallback_discovery_grid": fallback, "curves": details}


def freeze_grid(out):
    plan, _, geometries, bindings = load_contract(out)
    source, source_bindings = bind_sources(out)
    bindings.update(source_bindings)
    for phase in ("evaluation", "stress"):
        if next((out / "raw" / phase).glob("*.npz"), None) is not None:
            raise ValueError("Grid must be frozen before any evaluation/stress shard exists")
    calibration_path = out / "calibration.json"
    calibration = read_json(calibration_path)
    if (out / "calibration.sha256").read_text().strip() != sha(calibration_path):
        raise ValueError("Calibration completion marker differs")
    if (calibration["schema"] != CAL_SCHEMA or calibration["plan_sha256"] != bindings["plan.json"]
            or calibration["status"] != "calibrated" or calibration["refusals"]
            or calibration["coordinate_system"] != plan["coordinate_system"]
            or calibration["null_trials"] != plan["counts"]["null_calibration"]):
        raise ValueError("Calibration contract differs or refused")
    rehash(out, calibration["calibration_inputs_sha256"])
    rehash(out, calibration["source_sha256"])
    bindings.update(calibration["calibration_inputs_sha256"])
    bindings["calibration.json"] = sha(calibration_path)
    bindings["calibration.sha256"] = sha(out / "calibration.sha256")
    _, score_bindings = load_score_manifest(out, plan, "discovery")
    bindings.update(score_bindings)
    grids, details, refusals = {}, {}, []
    for channel in plan["channels"]:
        for case in range(7):
            key = f"ch{channel}/case{case}"
            values, metadata = load_scores(out / "scores/discovery" / f"ch{channel}_case{case}.npz")
            if (metadata["channel"] != channel or metadata["case_index"] != case
                    or metadata["phase"] != "discovery" or metadata["plan_sha256"] != bindings["plan.json"]):
                raise ValueError("Discovery score cell identity differs")
            n, ns = plan["counts"]["discovery"], len(plan["discovery_snr_db"])
            if (values["fine_float"].shape != (n, ns, 2, 6, 4)
                    or values["required_q16"].shape != (n, ns, 2, 4)
                    or metadata["anchors"] != [geometries[channel]["anchors"][case], geometries[channel]["anchors"][7]]):
                raise ValueError("Discovery score shape/geometry differs")
            curves = {}
            anchor = str(metadata["anchors"][0])
            rank = str(metadata["ranks"][0][1])
            try:
                for stage_index, stage in enumerate((*FLOAT_STAGES, Q16_STAGE)):
                    thresholds = calibration["fine"][f"ch{channel}"][anchor][stage][rank]
                    if stage == Q16_STAGE:
                        sample = decode_requirements(values["required_q16"][:, :, 0, 1],
                            values["always_masked"][:, :, 0, 1], values["fixed_valid"][:, :, 0, 1])
                    else:
                        if not values["fine_valid"][:, :, 0, stage_index, 1].all():
                            raise ValueError("Invalid discovery rank")
                        sample = values["fine_float"][:, :, 0, stage_index, 1]
                        if not np.isfinite(sample).all():
                            raise ValueError("Nonfinite discovery response")
                    for pfa in plan["primary_pfa"]:
                        threshold = thresholds[str(pfa)]["threshold"]
                        curves[f"fine/{stage}/pfa{pfa}"] = np.mean(sample > threshold, axis=0)
                for stage_index, stage in enumerate(FLOAT_STAGES):
                    if not values["coarse_valid"][:, :, stage_index].all() or not np.isfinite(values["coarse_float"][:, :, stage_index]).all():
                        raise ValueError("Invalid discovery coarse response")
                    for pfa in plan["primary_pfa"]:
                        threshold = calibration["coarse"][f"ch{channel}"][stage][str(pfa)]["threshold"]
                        curves[f"coarse/{stage}/pfa{pfa}"] = np.mean(values["coarse_float"][:, :, stage_index] > threshold, axis=0)
                if len(curves) != 26:
                    raise ValueError("Missing discovery curve")
                grids[key], details[key] = curve_grid(plan["discovery_snr_db"], curves)
            except (ValueError, KeyError, TypeError) as error:
                refusals.append({"cell": key, "reason": str(error)})
    report = {"schema": GRID_SCHEMA, "plan_sha256": bindings["plan.json"], "frozen_utc": utc(),
              "scope": plan["scope"], "status": "refused" if refusals else "frozen",
              "coordinate_system": plan["coordinate_system"],
              "grids": grids, "details": details, "refusals": refusals,
              "calibration_inputs_sha256": bindings, "source_sha256": source["source_sha256"],
              "expected_cells": len(plan["channels"]) * 7, "curves_per_cell": 26,
              "evaluation_outcomes_used": False, "physical_certification": False}
    rehash(out, bindings)
    rehash(out, source["source_sha256"])
    write_json(out / ("evaluation-grid-refused.json" if refusals else "evaluation-grid.json"), report)
    if refusals:
        raise ValueError(f"Grid refused for {len(refusals)} cells; see evaluation-grid-refused.json")
    with (out / "evaluation-grid.sha256").open("x") as stream:
        stream.write(sha(out / "evaluation-grid.json") + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=["score-null", "calibrate", "score-discovery", "freeze-grid"])
    args = parser.parse_args()
    out = args.output.resolve()
    actions = {"score-null": lambda: score_stage(out, "null_calibration"),
               "calibrate": lambda: calibrate(out), "score-discovery": lambda: score_stage(out, "discovery"),
               "freeze-grid": lambda: freeze_grid(out)}
    result = actions[args.stage]()
    print(json.dumps({"stage": args.stage, "status": result.get("status", "complete"), "scope": result["scope"]}), flush=True)


if __name__ == "__main__":
    main()
