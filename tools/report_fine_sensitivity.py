#!/usr/bin/env python3
"""Report complete frozen H1 cohorts without retuning policies or SNR grids.

Ordinary paired bootstrap draws are represented by integer multiplicities.
Each original null sample is sorted once; cumulative resampled multiplicities
select the exact higher order statistic. Common resampling arrays are reused
across every channel/case, both Pfa levels, all stages and both Pd targets.
Pointwise intervals and raw-grid bounds do not establish all-band equivalence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from pilot_proxy.testbench.fine_validation_scores import construct_geometry, decode_requirements
from pilot_proxy.testbench.fine_validation_stats import binomial_interval, raw_crossing_brackets
from reduce_fine_validation import (
    CAL_SCHEMA, GRID_SCHEMA, FLOAT_STAGES, Q16_STAGE, load_contract, load_score_manifest,
    load_scores, phase_ids, read_json, read_raw, reduce_powers, rehash, sha, utc, write_json,
)

SCHEMA = "literal-fine-sensitivity-report-v1"
STAGES = (*FLOAT_STAGES, Q16_STAGE)
SENTINEL = 1 << 64
PAIRS = ((0, 1, "tone_to_atsc_model"), (1, 2, "input_quantization"),
         (1, 3, "weight_quantization"), (2, 4, "weights_after_input"),
         (3, 4, "input_after_weights"), (4, 5, "fixed_transform"),
         (5, 6, "q16_decision"), (1, 6, "total_atsc_to_q16"))
STATUS = {0: "unique", 1: "unbracketed", 2: "ambiguous", 3: "null_threshold_unavailable", 4: "invalid_response"}


def aware(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Timestamp must include an explicit timezone")
    return result.astimezone(timezone.utc)


def artifact(out, name, schema, plan):
    path = out / name
    if path.with_suffix(".sha256").read_text().strip() != sha(path):
        raise ValueError(f"Completion marker differs: {name}")
    result = read_json(path)
    if (result["schema"] != schema or result["plan_sha256"] != sha(out / "plan.json")
            or result["coordinate_system"] != plan["coordinate_system"]
            or result["scope"] != plan["scope"] or result["refusals"]
            or result["physical_certification"] is not False
            or result["status"] != ("calibrated" if schema == CAL_SCHEMA else "frozen")):
        raise ValueError(f"Frozen artifact contract differs: {name}")
    if not aware(plan["frozen_utc"]) <= aware(result["frozen_utc"]) <= datetime.now(timezone.utc):
        raise ValueError(f"Frozen artifact time differs: {name}")
    rehash(out, result["calibration_inputs_sha256"])
    rehash(out, result["source_sha256"])
    return result


def raw_headers(path, metadata):
    """Validate every NPY header/byte length without examining power values."""
    n, ns = metadata["stop"] - metadata["start"], len(metadata["snr_db"])
    expected = {"float_fine": ((n, ns, 5, 3, 256), np.dtype(np.float64)),
                "float_coarse": ((n, ns, 5, 3), np.dtype(np.float64)),
                "fixed_fine": ((n, ns, 3, 256), np.dtype(np.uint64)),
                "fixed_coarse": ((n, ns, 3), np.dtype(np.uint64)),
                "clip_count": ((n, ns), np.dtype(np.uint32))}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 6 or set(names) != {name + ".npy" for name in (*expected, "meta_json")}:
            raise ValueError("Raw archive fields differ or repeat")
        for name, (wanted_shape, wanted_dtype) in expected.items():
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported raw NPY header version")
                if shape != wanted_shape or dtype != wanted_dtype or fortran:
                    raise ValueError("Raw NPY header shape/dtype/order differs")
                expected_size = stream.tell() + int(np.prod(shape)) * dtype.itemsize
                if archive.getinfo(name + ".npy").file_size != expected_size:
                    raise ValueError("Raw NPY member byte length differs")
        with archive.open("meta_json.npy") as stream:
            embedded = np.lib.format.read_array(stream, allow_pickle=False)
            if json.loads(str(embedded.item())) != metadata:
                raise ValueError("Raw embedded metadata differs")


def inventory(out, plan, ids, phase, grid):
    if phase not in ("evaluation", "stress"):
        raise ValueError("Only evaluation or the separately labelled stress phase is supported")
    selected = phase_ids(plan, ids, phase)
    expected_count = 2 if plan["scope"] == "engineering_only" else 512 if phase == "evaluation" else 128
    if len(selected) != expected_count:
        raise ValueError("Held-out phase count differs")
    if (phase == "evaluation" and not set(selected["payload"].tolist()) <= {2, 3}) or (phase == "stress" and np.any(selected["payload"] != 1)):
        raise ValueError("Held-out waveform roles differ")
    if phase == "evaluation":
        for item in selected:
            identity = f"literal-fine-validation-campaign-v1:fixture-choice:{phase}:{int(item['trial'])}".encode()
            if int(item["payload"]) != 2 + (hashlib.sha256(identity).digest()[0] & 1):
                raise ValueError("Frozen randomized fixture rule differs")
    expected_keys = {f"ch{c}/case{k}" for c in plan["channels"] for k in range(7)}
    if set(grid["grids"]) != expected_keys or grid["status"] != "frozen":
        raise ValueError("Grid does not cover every planned channel/case")
    files, result, bindings = set(), {}, {}
    grid_sha = sha(out / "evaluation-grid.json")
    for channel in plan["channels"]:
        for case in range(7):
            snrs = grid["grids"][f"ch{channel}/case{case}"]
            x = np.asarray(snrs, dtype=float)
            if (x.ndim != 1 or not 2 <= len(x) <= 61 or not np.isfinite(x).all()
                    or np.any(np.diff(x) <= 0) or np.any(x < -66) or np.any(x > -21)
                    or not np.all((x + 66) / .75 == np.rint((x + 66) / .75))):
                raise ValueError("Evaluation grid violates the frozen range/lattice")
            shards = []
            for start in range(0, len(selected), 16):
                chunk = selected[start:start + 16]
                metadata = {"plan_sha256": sha(out / "plan.json"), "phase": phase, "channel": channel,
                    "case_index": case, "streams": 2048, "start": start, "stop": start + len(chunk),
                    "snr_db": snrs, "float_stages": plan["float_stages"],
                    "raw_seeds": [int(v) for v in chunk["raw_seed"]],
                    "payloads": [int(v) for v in chunk["payload"]],
                    "axes": ["trial", "snr", "stage_if_float", "term", "bin_if_fine"],
                    "evaluation_grid_sha256": grid_sha}
                path = out / "raw" / phase / f"ch{channel}_case{case}_M2048_{start:06d}.npz"
                receipt = path.with_suffix(".json")
                saved = read_json(receipt)
                if saved["metadata"] != metadata or saved["sha256"] != sha(path):
                    raise ValueError(f"Held-out raw identity/hash differs: {path}")
                if not aware(grid["frozen_utc"]) <= aware(saved["completed_utc"]) <= datetime.now(timezone.utc):
                    raise ValueError("Held-out shard does not follow the frozen grid")
                raw_headers(path, metadata)
                files.update((path, receipt))
                bindings[str(path.relative_to(out))] = sha(path)
                bindings[str(receipt.relative_to(out))] = sha(receipt)
                shards.append((path, metadata))
            result[channel, case] = shards
    actual = {p for p in (out / "raw" / phase).glob("*") if p.suffix in (".json", ".npz")}
    if actual != files:
        raise ValueError("Held-out raw inventory has missing or extra shards")
    return selected, result, bindings


def resampling(null_ids, h1_ids, *, replicates, seed):
    """Ordinary independent bootstrap weights, preserving original paired IDs."""
    null_ids, h1_ids = np.asarray(null_ids), np.asarray(h1_ids)
    if (null_ids.ndim != 1 or h1_ids.ndim != 1 or not len(null_ids) or not len(h1_ids)
            or len(np.unique(null_ids)) != len(null_ids) or len(np.unique(h1_ids)) != len(h1_ids)
            or set(null_ids.tolist()) & set(h1_ids.tolist())):
        raise ValueError("Unique disjoint null/H1 trial IDs are required")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("Positive integer bootstrap count is required")
    rng = np.random.Generator(np.random.PCG64(seed))
    n0, n1 = len(null_ids), len(h1_ids)
    null_counts, h1_counts = np.empty((replicates, n0), np.uint16), np.empty((replicates, n1), np.uint16)
    for b in range(replicates):
        left = np.bincount(rng.integers(n0, size=n0), minlength=n0)
        right = np.bincount(rng.integers(n1, size=n1), minlength=n1)
        if max(left.max(), right.max()) > np.iinfo(np.uint16).max:
            raise ValueError("Bootstrap multiplicity exceeds declared storage")
        null_counts[b], h1_counts[b] = left, right
    return {"null_counts": null_counts, "h1_counts": h1_counts,
            "null_ids": null_ids, "h1_ids": h1_ids, "seed": np.asarray(seed, dtype=np.uint64)}


def weighted_thresholds(samples, counts, *, pfa, exact=False):
    """Exact higher threshold on each resampled multiset; no float Q16 cast."""
    from fractions import Fraction
    values = np.asarray(samples, dtype=object if exact else float)
    n = len(values)
    if values.shape != (counts.shape[1],) or not np.all(np.sum(counts, axis=1) == n):
        raise ValueError("Bootstrap null multiplicity denominator differs")
    if exact:
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 1 <= int(v) <= SENTINEL for v in values):
            raise ValueError("Exact requirements need decoded logical integer boundaries")
    elif not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Null scalar responses must be finite and nonnegative")
    probability = Fraction(str(pfa))
    if not 0 < probability < 1:
        raise ValueError("False-alarm probability must lie in (0,1)")
    q = 1 - probability
    index = (q.numerator * (n - 1) + q.denominator - 1) // q.denominator
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(counts[:, order], axis=1, dtype=np.uint32)
    positions = np.argmax(cumulative > index, axis=1)
    thresholds = values[order[positions]]
    available = thresholds < SENTINEL if exact else np.ones(len(thresholds), dtype=bool)
    return thresholds, np.asarray(available, dtype=bool)


def bootstrap_rates(samples, thresholds, available, counts, *, exact=False, batch=32):
    """Paired H1 proportions [replicate,SNR], preserving ties and sentinels."""
    samples = np.asarray(samples, dtype=object if exact else float)
    if samples.ndim != 2 or samples.shape[1] != counts.shape[1]:
        raise ValueError("Paired H1 sample shape differs")
    n = samples.shape[1]
    if not np.all(counts.sum(axis=1) == n) or len(thresholds) != len(counts):
        raise ValueError("Bootstrap H1 multiplicity denominator differs")
    if exact:
        if any(not isinstance(v, (int, np.integer)) or not 1 <= int(v) <= SENTINEL for v in samples.flat):
            raise ValueError("Invalid decoded H1 requirements")
        always = samples == SENTINEL
        encoded = np.asarray(np.where(always, 0, samples), dtype=np.uint64)
        threshold_values = np.asarray([int(v) if ok else 0 for v, ok in zip(thresholds, available)], dtype=np.uint64)
    else:
        if not np.isfinite(samples).all() or np.any(samples < 0):
            raise ValueError("Invalid H1 floating response")
        encoded, threshold_values = samples, np.asarray(thresholds, dtype=float)
    rates = np.full((len(counts), samples.shape[0]), np.nan)
    for start in range(0, len(counts), batch):
        stop = min(len(counts), start + batch)
        detection = encoded[None] > threshold_values[start:stop, None, None]
        if exact:
            detection |= always[None]
        rates[start:stop] = np.einsum("bsn,bn->bs", detection, counts[start:stop], dtype=np.float64, optimize=True) / n
    rates[~available] = np.nan
    return rates


def crossing_arrays(snrs, rates, target):
    estimates, codes = np.full(len(rates), np.nan), np.full(len(rates), 3, dtype=np.uint8)
    for i, values in enumerate(rates):
        if not np.isfinite(values).all():
            continue
        result = raw_crossing_brackets(snrs, values, target=target)
        codes[i] = {"unique": 0, "unbracketed": 1, "ambiguous": 2}[result["status"]]
        if result["status"] == "unique":
            estimates[i] = result["estimate_db"]
    return estimates, codes


def loss_bounds(left, right):
    """Outer difference of the two raw adjacent SNR brackets, second-first."""
    if left["status"] != "unique" or right["status"] != "unique":
        return {"status": "unresolved", "lower_db": None, "upper_db": None, "point_loss_db": None}
    a, b = left["upward_brackets"][0], right["upward_brackets"][0]
    return {"status": "raw_bracket_outer_difference", "lower_db": b["snr_lo_db"] - a["snr_hi_db"],
            "upper_db": b["snr_hi_db"] - a["snr_lo_db"], "point_loss_db": right["estimate_db"] - left["estimate_db"],
            "scope": "Grid discretization outer bound only; not a statistical confidence interval"}


def curve_summary(samples, valid, threshold, snrs, *, exact=False, always=None):
    """All trials stay in the denominator; invalid frozen decisions force keep."""
    if samples.shape != valid.shape or samples.ndim != 2:
        raise ValueError("Curve samples/validity must be [trial,SNR]")
    invalid = np.count_nonzero(~valid, axis=0)
    nonfinite = np.zeros(len(snrs), dtype=int) if exact else np.count_nonzero(~np.isfinite(samples), axis=0)
    detected = valid & (samples > (np.uint64(threshold) if exact else threshold))
    if exact:
        detected |= valid & always
    counts = np.count_nonzero(detected, axis=0)
    n = len(samples)
    intervals = [binomial_interval(int(k), n) for k in counts]
    rates = counts / n
    crossings = {str(t): raw_crossing_brackets(snrs, rates, target=t) for t in (.5, .9)}
    return {"trials": n, "counts": counts.tolist(), "pd": rates.tolist(),
            "cp95_lower": [v["lower"] for v in intervals], "cp95_upper": [v["upper"] for v in intervals],
            "invalid_trials": invalid.tolist(), "nonfinite_response_trials": nonfinite.tolist(),
            "response_qualified": not bool(np.any(invalid) or np.any(nonfinite)),
            "crossings": crossings, "threshold": threshold,
            "invalid_rule": "No row removal; frozen invalid decisions force keep. Invalid/nonfinite responses forbid inferential qualification."}


def paired_ladder(null_by_stage, h1_by_stage, qualified, point_crossings, snrs, weights, pfas):
    """All eight paired stage contrasts from one globally paired resampling plan."""
    b = len(weights["null_counts"])
    estimates = np.full((len(pfas), 2, len(STAGES), b), np.nan)
    codes = np.full(estimates.shape, 4, dtype=np.uint8)
    for p, pfa in enumerate(pfas):
        for stage, name in enumerate(STAGES):
            if not qualified[name]:
                continue
            exact = name == Q16_STAGE
            thresholds, available = weighted_thresholds(null_by_stage[name], weights["null_counts"], pfa=pfa, exact=exact)
            rates = bootstrap_rates(h1_by_stage[name], thresholds, available, weights["h1_counts"], exact=exact)
            for t, target in enumerate((.5, .9)):
                estimates[p, t, stage], codes[p, t, stage] = crossing_arrays(snrs, rates, target)
    loss = np.full((len(pfas), 2, len(PAIRS), b), np.nan)
    valid = np.zeros(loss.shape, dtype=bool)
    summaries = []
    for p, pfa in enumerate(pfas):
        for t, target in enumerate((.5, .9)):
            for pair, (left, right, label) in enumerate(PAIRS):
                ok = (codes[p, t, left] == 0) & (codes[p, t, right] == 0)
                valid[p, t, pair] = ok
                loss[p, t, pair, ok] = estimates[p, t, right, ok] - estimates[p, t, left, ok]
                original_left = point_crossings[str(pfa)][STAGES[left]][str(target)]
                original_right = point_crossings[str(pfa)][STAGES[right]][str(target)]
                original_unique = original_left["status"] == original_right["status"] == "unique"
                fraction = float(np.mean(ok))
                interval = np.quantile(loss[p, t, pair, ok], [.025, .975]).tolist() if fraction >= .95 and original_unique else None
                bound = loss_bounds(original_left, original_right)
                summaries.append({"pfa": pfa, "target_pd": target, "pair": label,
                    "left_stage": STAGES[left], "right_stage": STAGES[right],
                    "valid_replicates": int(ok.sum()), "replicates": b, "valid_fraction": fraction,
                    "pointwise_percentile95_db": interval, "interval_available": interval is not None,
                    "censor_counts": {f"{STATUS[int(a)]}/{STATUS[int(c)]}": int(n) for (a, c), n in Counter(zip(codes[p, t, left], codes[p, t, right])).items() if a or c},
                    "raw_discretization_bound": bound,
                    "pointwise_ci_inside_0p10": interval is not None and interval[0] >= -.1 and interval[1] <= .1,
                    "pointwise_upper_loss_at_most_0p10": interval is not None and interval[1] <= .1,
                    "equivalence_demonstrated": False,
                    "scope": "Conditional percentile interval on the explicitly counted uniquely bracketed subset; no simultaneous or combined discretization guarantee"})
    return {"crossing_estimate_db": estimates, "crossing_status": codes, "loss_db": loss, "loss_valid": valid}, summaries


def freeze_report_sources(out, target, phase, plan_hash):
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix == ".py" and Path(path).resolve().is_relative_to(ROOT):
            sources[str(Path(path).resolve())] = sha(path)
    snapshots = {}
    for source, digest in sources.items():
        path = target / "source_snapshots" / Path(source).relative_to(ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        if sha(path) != digest:
            raise ValueError("Reporting source changed during snapshot")
        snapshots[str(path.relative_to(out))] = digest
    record = {"schema": SCHEMA, "phase": phase, "plan_sha256": plan_hash, "frozen_utc": utc(),
              "source_sha256": sources, "snapshots_sha256": snapshots,
              "runtime": {"python": platform.python_version(), "executable": sys.executable,
                          "numpy": np.__version__, "scipy": package_version("scipy"), "bootstrap_generator": "numpy.PCG64"},
              "bootstrap_scope": "Median rank, calibrated anchor, both primary Pfa, both Pd targets, all8 paired stage contrasts; fixed nominal and other ranks descriptive only",
              "all_profile_equivalence": "not established; no simultaneous/discretization confidence bound"}
    write_json(target / "analysis-contract.json", record)
    return sources, snapshots


def run(out, phase):
    plan, ids, geometry, bindings = load_contract(out)
    calibration = artifact(out, "calibration.json", CAL_SCHEMA, plan)
    grid = artifact(out, "evaluation-grid.json", GRID_SCHEMA, plan)
    if (calibration["status"] != "calibrated" or calibration["null_trials"] != plan["counts"]["null_calibration"]
            or aware(grid["frozen_utc"]) < aware(calibration["frozen_utc"])):
        raise ValueError("Calibration/grid chronology differs")
    if grid["calibration_inputs_sha256"].get("calibration.json") != sha(out / "calibration.json"):
        raise ValueError("Grid does not bind this calibration")
    if any(grid["calibration_inputs_sha256"].get(k) != v for k, v in calibration["calibration_inputs_sha256"].items()):
        raise ValueError("Grid omits or changes a frozen calibration dependency")
    bindings.update(grid["calibration_inputs_sha256"])
    for name in ("calibration.json", "calibration.sha256", "evaluation-grid.json", "evaluation-grid.sha256"):
        bindings[name] = sha(out / name)
    _, null_bindings = load_score_manifest(out, plan, "null_calibration")
    bindings.update(null_bindings)
    h1_ids, files, raw_bindings = inventory(out, plan, ids, phase, grid)
    bindings.update(raw_bindings)
    target = out / "reports" / phase
    if target.exists():
        raise ValueError("Refusing to overwrite a sensitivity report")
    target.mkdir(parents=True)
    sources, snapshots = freeze_report_sources(out, target, phase, sha(out / "plan.json"))
    null_ids = phase_ids(plan, ids, "null_calibration")
    seed = int.from_bytes(hashlib.sha256(f"{SCHEMA}:{sha(out / 'plan.json')}:{phase}:global-paired-bootstrap".encode()).digest()[:8], "big")
    replicates = plan["loss_rule"]["bootstrap_replicates"]
    if plan["scope"] != "engineering_only" and replicates != 2000:
        raise ValueError("Frozen bootstrap count differs")
    weights = resampling(null_ids["raw_seed"], h1_ids["raw_seed"], replicates=replicates, seed=seed)
    weights_path = target / "bootstrap-resampling.npz"
    with weights_path.open("xb") as stream:
        np.savez(stream, **weights)
    outputs = {str(weights_path.relative_to(out)): sha(weights_path),
               str((target / "analysis-contract.json").relative_to(out)): sha(target / "analysis-contract.json")}
    cells = []
    fixture_counts = {str(p): int(np.count_nonzero(h1_ids["payload"] == p)) for p in sorted(set(h1_ids["payload"].tolist()))}
    for channel in plan["channels"]:
        null, null_meta = load_scores(out / "scores/null_calibration" / f"ch{channel}.npz")
        if (null_meta["channel"] != channel or null_meta["plan_sha256"] != sha(out / "plan.json")
                or null_meta["anchors"] != geometry[channel]["anchors"]
                or null_meta["trials"] != plan["counts"]["null_calibration"]):
            raise ValueError("Calibration score identity/geometry differs")
        for case in range(7):
            snrs = grid["grids"][f"ch{channel}/case{case}"]
            anchors = [geometry[channel]["anchors"][case], geometry[channel]["anchors"][7]]
            pieces = []
            for path, metadata in files[channel, case]:
                raw = read_raw(path, metadata, len(snrs))
                shape = raw["fixed_fine"].shape[:2]
                flat = lambda a: a.reshape((-1, *a.shape[2:]))
                result = reduce_powers(flat(raw["float_fine"]), flat(raw["fixed_fine"]),
                    flat(raw["float_coarse"]), flat(raw["fixed_coarse"]), anchors, geometry[channel]["mu0"])
                result = {k: v.reshape((*shape, *v.shape[1:])) for k, v in result.items()}
                result["clip_count"] = raw["clip_count"]
                pieces.append(result)
            data = {k: np.concatenate([v[k] for v in pieces]) for k in pieces[0]}
            if any(len(v) != len(h1_ids) for v in data.values()):
                raise ValueError("Sensitivity trial denominator differs")
            curve_records, primary_crossings, qualified = [], {str(p): {} for p in plan["primary_pfa"]}, {}
            primary_null, primary_h1 = {}, {}
            for slot, anchor in enumerate(anchors):
                ranks = construct_geometry(anchor).ranks
                for stage_index, stage in enumerate(STAGES):
                    for column, rank in enumerate(ranks):
                        exact = stage == Q16_STAGE
                        sample = data["required_q16"][:, :, slot, column] if exact else data["fine_float"][:, :, slot, stage_index, column]
                        valid = data["fixed_valid"][:, :, slot, column] if exact else data["fine_valid"][:, :, slot, stage_index, column]
                        always = data["always_masked"][:, :, slot, column] if exact else None
                        for pfa in plan["diagnostic_pfa"] + plan["primary_pfa"]:
                            threshold = calibration["fine"][f"ch{channel}"][str(anchor)][stage][str(int(rank))][str(pfa)]["threshold"]
                            summary = curve_summary(sample, valid, threshold, snrs, exact=exact, always=always)
                            summary.update(kind="fine", stage=stage, geometry="calibrated" if slot == 0 else "fixed_nominal",
                                           anchor=anchor, rank=int(rank), rank_index=column, pfa=pfa)
                            decisions = valid & (sample > (np.uint64(threshold) if exact else threshold))
                            if exact:
                                decisions |= valid & always
                            summary["fixture_descriptive"] = {
                                str(payload): {"trials": int(mask.sum()), "counts": np.count_nonzero(decisions[mask], axis=0).tolist()}
                                for payload in sorted(set(h1_ids["payload"].tolist())) for mask in [h1_ids["payload"] == payload]}
                            curve_records.append(summary)
                            if slot == 0 and column == 1 and pfa in plan["primary_pfa"]:
                                primary_crossings[str(pfa)][stage] = summary["crossings"]
                                qualified[stage] = summary["response_qualified"]
                        if slot == 0 and column == 1:
                            if exact:
                                primary_null[stage] = decode_requirements(null["required_q16"][:, case, column], null["always_masked"][:, case, column], null["fixed_valid"][:, case, column])
                                primary_h1[stage] = decode_requirements(sample, always, valid).T if valid.all() else None
                            else:
                                primary_null[stage] = null["fine_float"][:, case, stage_index, column]
                                primary_h1[stage] = sample.T
            coarse_records = []
            for stage_index, stage in enumerate(FLOAT_STAGES):
                for pfa in plan["diagnostic_pfa"] + plan["primary_pfa"]:
                    threshold = calibration["coarse"][f"ch{channel}"][stage][str(pfa)]["threshold"]
                    summary = curve_summary(data["coarse_float"][:, :, stage_index], data["coarse_valid"][:, :, stage_index], threshold, snrs)
                    summary.update(kind="coarse", stage=stage, pfa=pfa)
                    coarse_records.append(summary)
            bootstrap, contrasts = paired_ladder(primary_null, primary_h1, qualified, primary_crossings, snrs, weights, plan["primary_pfa"])
            benchmarks = []
            if case == 1:  # Frozen aligned case, never selected by measured response.
                for coarse in coarse_records:
                    if coarse["pfa"] not in plan["primary_pfa"]:
                        continue
                    for pd in (.5, .9):
                        fine = primary_crossings[str(coarse["pfa"])][coarse["stage"]][str(pd)]
                        # Positive is a fine sensitivity advantage: coarse crossing minus fine.
                        benchmarks.append({"stage": coarse["stage"], "pfa": coarse["pfa"], "target_pd": pd,
                            "coarse_minus_fine_db": loss_bounds(fine, coarse["crossings"][str(pd)]),
                            "scope": "Aligned-case raw-grid benchmark, no paired confidence interval for this coarse/fine contrast"})
            prefix = target / "cells" / f"ch{channel}_case{case}"
            prefix.parent.mkdir(parents=True, exist_ok=True)
            score_path, bootstrap_path = prefix.with_suffix(".npz"), prefix.with_name(prefix.name + "-bootstrap.npz")
            with score_path.open("xb") as stream:
                np.savez(stream, **data, raw_seed=h1_ids["raw_seed"], payload=h1_ids["payload"], snr_db=np.asarray(snrs), anchors=np.asarray(anchors))
            with bootstrap_path.open("xb") as stream:
                np.savez(stream, **bootstrap)
            report = {"schema": SCHEMA, "phase": phase, "channel": channel, "case_index": case,
                "scope": plan["scope"], "coordinate_system": plan["coordinate_system"], "snr_db": snrs,
                "trials": len(h1_ids), "fixture_counts": fixture_counts, "fine_curves": curve_records,
                "coarse_curves": coarse_records, "paired_contrasts": contrasts, "aligned_benchmarks": benchmarks,
                "bootstrap_array_axes": {"crossing_estimate_db": ["pfa", "pd_target", "stage", "replicate"],
                                         "loss_db": ["pfa", "pd_target", "pair", "replicate"]},
                "bootstrap_pfas": plan["primary_pfa"], "bootstrap_targets": [.5, .9], "bootstrap_stages": list(STAGES),
                "bootstrap_pairs": [p[2] for p in PAIRS], "bootstrap_status_codes": STATUS,
                "bootstrap_seed": seed, "bootstrap_resampling_sha256": sha(weights_path),
                "bootstrap_replicates": replicates, "all_profile_equivalence": "not established",
                "inference_scope": "Conditional independently randomized mixture of two fixed qualified fixtures" if phase == "evaluation" else "Separate failed-stationarity payload01 stress cohort; no primary acceptance inference",
                "physical_certification": False}
            report_path = prefix.with_suffix(".json")
            write_json(report_path, report)
            for path in (score_path, bootstrap_path, report_path):
                outputs[str(path.relative_to(out))] = sha(path)
            cells.append({"channel": channel, "case_index": case, "report": str(report_path.relative_to(out)),
                          "fine_curves": len(curve_records), "coarse_curves": len(coarse_records),
                          "paired_contrasts": len(contrasts), "paired_intervals_available": sum(v["interval_available"] for v in contrasts)})
            print(json.dumps({"phase": phase, "channel": channel, "case": case, "completed_cells": len(cells),
                              "total_cells": len(plan["channels"]) * 7}), flush=True)
    rehash(out, bindings)
    rehash(out, sources)
    rehash(out, snapshots)
    rehash(out, outputs)
    report = {"schema": SCHEMA, "phase": phase, "scope": plan["scope"], "completed_utc": utc(),
              "plan_sha256": sha(out / "plan.json"), "coordinate_system": plan["coordinate_system"],
              "coverage_complete": len(cells) == len(plan["channels"]) * 7, "cells": cells,
              "trials_per_cell_snr": len(h1_ids), "fixture_counts": fixture_counts,
              "inputs_sha256": bindings, "outputs_sha256": outputs, "source_sha256": sources,
              "snapshots_sha256": snapshots, "bootstrap_seed": seed, "bootstrap_replicates": replicates,
              "bootstrap_scope": "All paired stage contrasts, both Pd targets and both primary Pfa, median rank/calibrated geometry only",
              "pending_inference": ["Bootstrap for nonmedian ranks and fixed nominal anchor", "Simultaneous all-cell loss bound including grid discretization", "Measured physical receiver/feed covariance and transfer"],
              "all_profile_equivalence": "not established", "physical_certification": False,
              "chronology_scope": "Receipts authenticate completion after the frozen grid. The bound generating runner enforces calibration/grid before generation; completion timestamps alone do not independently prove start time.",
              "thresholds_changed": False, "evaluation_grid_changed": False}
    write_json(target / "report.json", report)
    with (target / "report.sha256").open("x") as stream:
        stream.write(sha(target / "report.json") + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["evaluation", "stress"], required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.phase)


if __name__ == "__main__":
    main()
