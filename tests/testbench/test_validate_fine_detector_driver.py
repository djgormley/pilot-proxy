"""Fail-closed campaign plumbing tests; no GPU and no scientific draws."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "tools/validate_fine_detector.py"
spec = importlib.util.spec_from_file_location("fine_raw_driver_tests", SOURCE)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def fixture_plan():
    counts = {k: 2 for k in d.SCIENCE_COUNTS}
    plan = dict(
        schema=d.SCHEMA,
        scope="engineering_only",
        counts=counts,
        channels=list(range(14, 37)),
        streams=2048,
        windows=128,
        window_samples=128,
        fine_bins=256,
        float_stages=list(d.FLOAT_STAGES),
        fixed_stage=d.STAGE_FIXED_FLOAT_DECISION,
        coordinate_system=d.COORDINATE_SYSTEM,
        frozen_utc="2020-01-01T00:00:00+00:00",
        physical_certification=False,
        discovery_snr_db=list(range(-66, -20, 3)),
        primary_pfa=[0.01, 0.05],
        diagnostic_pfa=[0.001],
    )
    rows = []
    for phase, count in counts.items():
        for m in d.M_VALUES if phase.startswith("spatial_") else [2048]:
            for i in range(count):
                seed = d.seed_for("engineering_" + phase, i, m)
                rows.append(
                    (phase, m, i, seed, seed % (1 << 63), d.payload_for(phase, i))
                )
    ids = np.array(
        rows,
        dtype=[
            ("phase", "U24"),
            ("streams", "u4"),
            ("trial", "u4"),
            ("raw_seed", "u8"),
            ("effective_seed63", "u8"),
            ("payload", "u1"),
        ],
    )
    return plan, ids


def null_metadata(plan, ids):
    chosen = ids[ids["phase"] == "null_calibration"]
    return dict(
        plan_sha256="a" * 64,
        phase="null_calibration",
        start=0,
        stop=2,
        channels=plan["channels"],
        streams=2048,
        float_stages=list(d.FLOAT_STAGES),
        raw_seeds=[int(v) for v in chosen["raw_seed"]],
        axes=["trial", "channel", "stage_if_float", "term", "bin_if_fine"],
    )


def zero_arrays(shape):
    arrays = d.arrays(shape)
    for a in arrays.values():
        a.fill(0)
    return arrays


def save_json(path, value):
    d.write_new(path, value)
    d.seal_marker(path)


def test_exact_seed_recipe_and_distinct_phases():
    expected = int.from_bytes(
        hashlib.sha256(
            (d.SCHEMA + ":2026-09-09:null_calibration:2048:17").encode()
        ).digest()[:8],
        "big",
    )
    assert d.seed_for("null_calibration", 17) == expected
    plan, ids = fixture_plan()
    d.validate_plan_contract(plan, ids)
    assert len(set(ids["effective_seed63"])) == len(ids)
    assert set(ids[ids["phase"] == "discovery"]["payload"]) == {0}
    assert set(ids[ids["phase"] == "stress"]["payload"]) == {1}
    assert {d.payload_for("evaluation", i) for i in range(100)} == {2, 3}


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "science"),
        ("coordinate_system", "normalized"),
        ("streams", 1024),
        ("physical_certification", True),
        ("frozen_utc", "2020-01-01"),
        ("float_stages", list(reversed(d.FLOAT_STAGES))),
        ("channels", list(range(14, 36))),
        ("discovery_snr_db", [-66, -63]),
    ],
)
def test_plan_contract_rejects_changed_scope_geometry(field, value):
    plan, ids = fixture_plan()
    plan[field] = value
    with pytest.raises((ValueError, TypeError)):
        d.validate_plan_contract(plan, ids)


@pytest.mark.parametrize(
    "change", ["missing", "order", "seed", "fixture", "dtype", "count", "bool_count"]
)
def test_identity_and_count_tampering(change):
    plan, ids = fixture_plan()
    if change == "missing":
        ids = ids[:-1]
    elif change == "order":
        ids = ids[::-1]
    elif change == "seed":
        ids[0]["raw_seed"] += np.uint64(1)
    elif change == "fixture":
        ids[0]["payload"] = 0
    elif change == "dtype":
        ids = ids.astype(
            [(n, "f8" if n == "raw_seed" else ids.dtype[n]) for n in ids.dtype.names]
        )
    elif change == "count":
        plan["counts"]["null_calibration"] = 3
    else:
        plan["counts"]["null_calibration"] = True
    with pytest.raises((ValueError, TypeError)):
        d.validate_plan_contract(plan, ids)


def test_cache_cartesian_identity_all644():
    cases = [
        dict(name=n, calibrated_anchor_bin_inverted=i, nominal_anchor_bin_inverted=0)
        for i, n in enumerate(d.CASE_NAMES)
    ]
    cache = dict(
        case_order="channel ascending, each profile case list order, payload index ascending",
        profiles=[
            dict(physical_channel=c, cases=cases, packed_profile_sha256=str(c))
            for c in range(14, 37)
        ],
        payloads=[dict(index=i) for i in range(4)],
    )
    for i in range(644):
        profile = cache["profiles"][i // 28]
        receipt = dict(
            case_index=i,
            profile_channel=profile["physical_channel"],
            case=cases[i % 28 // 4],
            payload=cache["payloads"][i % 4],
            packed_profile_sha256=profile["packed_profile_sha256"],
            passed=True,
        )
        d.validate_cache_identity(cache, i, receipt)
        bad = deepcopy(receipt)
        bad["case_index"] = (i + 1) % 644
        with pytest.raises(ValueError):
            d.validate_cache_identity(cache, i, bad)
    cache["profiles"][0]["cases"] = list(reversed(cases))
    with pytest.raises(ValueError):
        d.validate_cache_identity(cache, 643, receipt)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "added",
        "float32",
        "fixed_float",
        "shape",
        "nan",
        "negative",
        "clip",
        "fixed_overflow",
        "stage",
        "seeds",
        "streams",
    ],
)
def test_raw_shards_fail_closed(damage):
    plan, ids = fixture_plan()
    meta = null_metadata(plan, ids)
    a = zero_arrays((2, 23))
    if damage == "missing":
        a.pop("fixed_fine")
    elif damage == "added":
        a["extra"] = np.zeros(1)
    elif damage == "float32":
        a["float_fine"] = a["float_fine"].astype(np.float32)
    elif damage == "fixed_float":
        a["fixed_fine"] = a["fixed_fine"].astype(float)
    elif damage == "shape":
        a["fixed_coarse"] = a["fixed_coarse"][:1]
    elif damage == "nan":
        a["float_fine"].flat[0] = np.nan
    elif damage == "negative":
        a["float_coarse"].flat[0] = -1
    elif damage == "clip":
        a["clip_count"].flat[0] = 2048 * 128 * 128 + 1
    elif damage == "fixed_overflow":
        a["fixed_coarse"].flat[0] = 1 << 63
    elif damage == "stage":
        meta["float_stages"] = list(reversed(d.FLOAT_STAGES))
    elif damage == "seeds":
        meta["raw_seeds"][1] = (
            meta["raw_seeds"][0] + (1 << 63)
            if meta["raw_seeds"][0] < (1 << 63)
            else meta["raw_seeds"][0] - (1 << 63)
        )
    else:
        meta["streams"] = 1024
    with pytest.raises((ValueError, TypeError)):
        d.validate_raw_arrays(a, meta)


def test_committed_resume_preserves_exact_uint64(tmp_path):
    plan, ids = fixture_plan()
    meta = null_metadata(plan, ids)
    a = zero_arrays((2, 23))
    a["fixed_fine"].flat[0] = (1 << 53) + 1
    path = tmp_path / "raw.npz"
    assert not d.completed(path, meta)
    d.save_shard(path, a, meta, 0.1)
    assert d.completed(path, meta)
    with np.load(path, allow_pickle=False) as z:
        assert int(z["fixed_fine"].flat[0]) == (1 << 53) + 1
    with pytest.raises(ValueError):
        d.save_shard(path, a, meta, 0.2)
    bad = deepcopy(meta)
    bad["start"] = 1
    with pytest.raises(ValueError):
        d.completed(path, bad)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        d.completed(path, meta)


def test_partial_commit_refused(tmp_path):
    plan, ids = fixture_plan()
    meta = null_metadata(plan, ids)
    path = tmp_path / "raw.npz"
    path.write_bytes(b"partial")
    with pytest.raises(ValueError):
        d.completed(path, meta)


def test_record_refuses_silent_integer_cast():
    target = zero_arrays((1,))
    result = {"clip_count": 0}
    for key, shape in [("powers_by_stage", (3, 256)), ("coarse_by_stage", (3,))]:
        result[key] = {
            s: np.zeros(
                shape,
                dtype=np.uint64 if s == d.STAGE_FIXED_FLOAT_DECISION else np.float64,
            )
            for s in d.ALL_STAGES
        }
    result["powers_by_stage"][d.STAGE_FIXED_FLOAT_DECISION].flat[0] = (1 << 53) + 1
    d.record(target, 0, result)
    assert int(target["fixed_fine"].flat[0]) == (1 << 53) + 1
    result["coarse_by_stage"][d.STAGE_FIXED_FLOAT_DECISION] = np.zeros(3, float)
    with pytest.raises(ValueError):
        d.record(target, 0, result)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/escape",
        "../escape",
        "source_snapshots/../../escape",
        "source_snapshots\\escape",
        None,
    ],
)
def test_relative_bindings_refuse_escape(tmp_path, path):
    with pytest.raises(ValueError):
        d._relative(tmp_path, path)


def test_lock_refuses_second_writer_and_releases(tmp_path):
    with d.generation_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with d.generation_lock(tmp_path):
                pass
    with pytest.raises(LookupError):
        with d.generation_lock(tmp_path):
            raise LookupError()
    with d.generation_lock(tmp_path):
        pass


def test_marker_and_snapshot_authentication(tmp_path):
    plan, ids = fixture_plan()
    src = tmp_path / "source.py"
    src.write_text("source")
    snapshot = tmp_path / "source_snapshots/source.py"
    snapshot.parent.mkdir()
    snapshot.write_bytes(src.read_bytes())
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "plan.json").write_text("{}")
    np.save(tmp_path / "trial-identities.npy", ids)
    plan.update(
        source_sha256={str(src): d.sha(src)},
        source_snapshot_paths={str(src): "source_snapshots/source.py"},
        identities_sha256=d.sha(tmp_path / "trial-identities.npy"),
        prior_seed_files_sha256={},
        cache_path=str(cache),
        cache_sha256={"plan.json": d.sha(cache / "plan.json")},
    )
    save_json(tmp_path / "plan.json", plan)
    assert d.verify_plan(tmp_path) == plan
    snapshot.write_text("changed")
    with pytest.raises(ValueError):
        d.verify_plan(tmp_path)
    snapshot.write_bytes(src.read_bytes())
    src.write_text("changed")
    with pytest.raises(ValueError):
        d.verify_plan(tmp_path)
    (tmp_path / "plan.json").write_text("{}")
    with pytest.raises(ValueError):
        d.verify_plan(tmp_path)


def test_native_adapter_reverses_only_K_and_preserves_inputs(monkeypatch):
    real = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    a = (real + 1j * (real + 1)).astype(np.complex64)
    t = (2 * a).copy()
    saved = a.copy()
    monkeypatch.setattr(d, "load_cache", lambda *args: ({}, a, t))
    monkeypatch.setattr(
        d,
        "_ideal_float_weights_from_layout",
        lambda *args, **kwargs: np.zeros((3, 128), complex),
    )
    monkeypatch.setattr(
        d,
        "FineValidationEngine",
        lambda profile, atsc, tone, **kwargs: (atsc, tone, kwargs),
    )
    bank = SimpleNamespace(
        get_weights_for_physical_channel=lambda c: (np.zeros((3, 128), np.int8), True),
        layout_for_physical_channel=lambda c: {},
    )
    atsc, tone, _ = d.engine_for(
        {"cache_path": "unused", "coordinate_system": d.COORDINATE_SYSTEM},
        None,
        bank,
        29,
        0,
        0,
    )
    assert np.array_equal(atsc, np.conjugate(saved[:, ::-1]))
    assert np.array_equal(tone, 2 * atsc) and atsc.flags.c_contiguous
    assert np.array_equal(a, saved) and not np.shares_memory(a, atsc)


def test_seed_ledger_exact_dtypes_and_effective_mapping(tmp_path):
    path = tmp_path / "seeds.npz"
    np.savez(
        path,
        raw_seed=np.array([(1 << 63) + 3], dtype=np.uint64),
        effective_seed63=np.array([4], dtype=np.uint64),
        conservative_seed32=np.array([5], dtype=np.uint32),
    )
    assert d.seed_file_values(path) == {3, 4, 5}
    np.savez(path, raw_seed=np.array([float(1 << 63)]))
    with pytest.raises(ValueError):
        d.seed_file_values(path)


@pytest.mark.parametrize(
    "bindings",
    [
        {},
        {"raw/null_validation/a.npz": "a" * 64},
        {"raw/evaluation/a.npz": "a" * 64},
        {"/absolute": "a" * 64},
    ],
)
def test_forbidden_binding_phases(tmp_path, bindings):
    with pytest.raises(ValueError):
        d._bindings(tmp_path, bindings, grid=True)


def minimal_calibration(tmp_path, monkeypatch):
    plan, ids = fixture_plan()
    np.save(tmp_path / "trial-identities.npy", ids)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    meta = null_metadata(plan, ids)
    meta["plan_sha256"] = d.sha(tmp_path / "plan.json")
    raw = tmp_path / "raw/null_calibration/000000.npz"
    d.save_shard(raw, zero_arrays((2, 23)), meta, 0.1)
    bindings = {
        str(p.relative_to(tmp_path)): d.sha(p) for p in [raw, raw.with_suffix(".json")]
    }
    artifact = dict(
        schema="literal-fine-calibration-v1",
        plan_sha256=d.sha(tmp_path / "plan.json"),
        frozen_utc=d.utc(),
        calibration_inputs_sha256=bindings,
    )
    monkeypatch.setattr(d, "_producer_contract", lambda *a: None)
    monkeypatch.setattr(d, "_calibration_thresholds", lambda *a: None)
    save_json(tmp_path / "calibration.json", artifact)
    return plan, artifact


def test_calibration_requires_full_completed_raw_coverage(tmp_path, monkeypatch):
    plan, artifact = minimal_calibration(tmp_path, monkeypatch)
    assert d.verify_calibration(tmp_path, plan) == artifact
    (tmp_path / "raw/null_calibration/000000.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        d.verify_calibration(tmp_path, plan)


def test_calibration_refuses_missing_marker(tmp_path, monkeypatch):
    plan, _ = minimal_calibration(tmp_path, monkeypatch)
    (tmp_path / "calibration.sha256").unlink()
    with pytest.raises(FileNotFoundError):
        d.verify_calibration(tmp_path, plan)


@pytest.mark.parametrize(
    "damage",
    [
        "missingcell",
        "duplicate",
        "offlattice",
        "range",
        "nan",
        "empty",
        "wrong_count",
        "evaluation_used",
    ],
)
def test_grid_invalid_before_discovery_reads(tmp_path, monkeypatch, damage):
    plan, ids = fixture_plan()
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    cal = dict(
        frozen_utc=plan["frozen_utc"],
        calibration_inputs_sha256={"plan.json": d.sha(tmp_path / "plan.json")},
    )
    save_json(tmp_path / "calibration.json", cal)
    monkeypatch.setattr(d, "verify_calibration", lambda *a: cal)
    monkeypatch.setattr(d, "_producer_contract", lambda *a: None)
    grids = {
        f"ch{c}/case{k}": [-66.0, -63.0] for c in plan["channels"] for k in range(7)
    }
    artifact = dict(
        schema="literal-fine-evaluation-grid-v1",
        plan_sha256=d.sha(tmp_path / "plan.json"),
        frozen_utc=d.utc(),
        grids=grids,
        expected_cells=161,
        curves_per_cell=26,
        evaluation_outcomes_used=False,
        calibration_inputs_sha256={
            p: d.sha(tmp_path / p)
            for p in ["plan.json", "calibration.json", "calibration.sha256"]
        },
    )
    if damage == "missingcell":
        grids.pop("ch14/case0")
    elif damage == "duplicate":
        grids["ch14/case0"] = [-66, -66]
    elif damage == "offlattice":
        grids["ch14/case0"] = [-65.9]
    elif damage == "range":
        grids["ch14/case0"] = [-67]
    elif damage == "nan":
        grids["ch14/case0"] = [float("nan")]
    elif damage == "empty":
        grids["ch14/case0"] = []
    elif damage == "wrong_count":
        artifact["expected_cells"] = 160
    else:
        artifact["evaluation_outcomes_used"] = True
    path = tmp_path / "evaluation-grid.json"
    path.write_text(json.dumps(artifact))
    d.seal_marker(path)
    with pytest.raises(ValueError):
        d.verify_evaluation_grid(tmp_path, plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "refused"),
        ("refusals", ["bad"]),
        ("coordinate_system", "normalized"),
        ("scope", "science"),
        ("physical_certification", True),
    ],
)
def test_producer_contract_refuses_false_scope(tmp_path, field, value):
    plan, _ = fixture_plan()
    source = str(d.ROOT / "tools/reduce_fine_validation.py")
    plan["source_sha256"] = {source: d.sha(source)}
    artifact = dict(
        status="calibrated",
        refusals=[],
        scope=plan["scope"],
        coordinate_system=d.COORDINATE_SYSTEM,
        physical_certification=False,
        source_sha256=plan["source_sha256"],
    )
    d._producer_contract(artifact, plan, "calibrated")
    artifact[field] = value
    with pytest.raises(ValueError):
        d._producer_contract(artifact, plan, "calibrated")


@pytest.mark.parametrize(
    "damage",
    [None, "missing_rank", "bad_q16", "bad_pfa", "zero_float", "count", "wrong_anchor"],
)
def test_calibration_full_threshold_geometry(tmp_path, damage):
    plan, _ = fixture_plan()
    plan["cache_path"] = str(tmp_path)
    profiles = [
        dict(
            physical_channel=c,
            cases=[
                dict(calibrated_anchor_bin_inverted=0, nominal_anchor_bin_inverted=0)
            ]
            * 7,
        )
        for c in plan["channels"]
    ]
    (tmp_path / "plan.json").write_text(json.dumps({"profiles": profiles}))
    stages = (*d.FLOAT_STAGES, d.STAGE_FIXED_FLOAT_DECISION, d.STAGE_FIXED_Q16_CPU)

    def records(exact):
        return {
            str(p): dict(
                status="available",
                trials=2,
                nominal_pfa=p,
                independent_validation=False,
                threshold=(1 << 64) - 1 if exact else 1.0,
            )
            for p in [0.001, 0.01, 0.05]
        }

    ranks = [str(int(r)) for r in d.construct_geometry(0).ranks]
    table = {s: {r: records(s == d.STAGE_FIXED_Q16_CPU) for r in ranks} for s in stages}
    artifact = dict(
        null_trials=2,
        independent_null_validation=False,
        fine={f"ch{c}": {"0": deepcopy(table)} for c in plan["channels"]},
        coarse={
            f"ch{c}": {s: records(False) for s in stages[:-1]} for c in plan["channels"]
        },
    )
    target = artifact["fine"]["ch14"]["0"]
    if damage == "missing_rank":
        target[stages[0]].pop(ranks[0])
    elif damage == "bad_q16":
        target[stages[-1]][ranks[0]]["0.01"]["threshold"] = 1 << 64
    elif damage == "bad_pfa":
        target[stages[0]][ranks[0]].pop("0.01")
    elif damage == "zero_float":
        target[stages[0]][ranks[0]]["0.01"]["threshold"] = 0.0
    elif damage == "count":
        artifact["null_trials"] = 1
    elif damage == "wrong_anchor":
        artifact["fine"]["ch14"]["1"] = artifact["fine"]["ch14"].pop("0")
    if damage is None:
        d._calibration_thresholds(artifact, plan)
    else:
        with pytest.raises((ValueError, TypeError)):
            d._calibration_thresholds(artifact, plan)
