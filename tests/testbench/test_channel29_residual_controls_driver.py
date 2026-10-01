"""Small CPU-only proofs for frozen digital generation and bookkeeping."""

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "channel29_generator",
    Path(__file__).parents[2] / "tools/generate_channel29_residual_controls.py",
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def sealed(plan):
    result = copy.deepcopy(plan)
    result["plan_sha256"] = m.digest_payload(result, "plan_sha256")
    return result


@pytest.fixture
def plan(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    runtime = {"python": "test"}
    result = {
        "schema": m.SCHEMA,
        "status": "plan_frozen",
        "frozen_at": old,
        "protocol_sha256": "a" * 64,
        "geometry": {
            "channel": 29,
            "num_streams": 2048,
            "K": 128,
            "windows_per_stream": 128,
            "spectral_sense": "normal",
            "input_scale": 3.299831645537222,
        },
        "populations": [
            {
                "name": "off",
                "shelf_db": None,
                "reference_projection_gamma": 0.0,
                "count_per_block": 1,
            },
            {
                "name": "stress",
                "shelf_db": -35.0,
                "reference_projection_gamma": 0.1,
                "count_per_block": 1,
            },
        ],
        "stages": {
            "calibration": {"blocks": 1, "populations": ["off"]},
            "evaluation": {"blocks": 1, "populations": ["off"]},
            "stress": {"blocks": 1, "populations": ["stress"]},
        },
        "audit": {"populations": ["off", "stress"], "frames_per_population": 1},
        "runtime": {
            "values": runtime,
            "sha256": hashlib.sha256(m.canonical(runtime)).hexdigest(),
        },
        "source_artifacts": {
            str(Path(__file__).resolve()): {"sha256": m.sha(Path(__file__))}
        },
        "evaluation_prerequisite": {
            "receipt_path": str(tmp_path / "calibration-receipt.json")
        },
        "inputs": {
            name: {"path": str(tmp_path / (name + ".npz")), "sha256": "b" * 64}
            for name in [
                "input_iq",
                "waveform_audit",
                "weights",
                "weight_manifest",
                "lib",
                "signal_cache",
                "cache_config",
                "planned_seeds",
                "seed_exclusions",
            ]
        },
    }
    return sealed(result)


def test_frozen_plan_digest_and_no_mutation(plan):
    before = copy.deepcopy(plan)
    assert m.validate_plan(plan, verify_files=False)
    assert plan == before
    plan["stages"]["evaluation"]["blocks"] = 2
    with pytest.raises(ValueError, match="digest"):
        m.validate_plan(plan, verify_files=False)


@pytest.mark.parametrize(
    "field,value",
    [("num_streams", 1), ("channel", 14), ("spectral_sense", "reverse"), ("K", 64)],
)
def test_resealed_reduced_or_reversed_geometry_refused(plan, field, value):
    plan["geometry"][field] = value
    with pytest.raises(ValueError, match="geometry"):
        m.validate_plan(sealed(plan), verify_files=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shelf_db", float("inf")),
        ("reference_projection_gamma", -0.1),
        ("count_per_block", 0),
        ("count_per_block", True),
    ],
)
def test_population_contract_refuses_invalid_inputs(plan, field, value):
    plan["populations"][0][field] = value
    with pytest.raises(ValueError):
        m.validate_plan(sealed(plan), verify_files=False)


def test_reference_stress_cannot_enter_primary_stage(plan):
    plan["stages"]["calibration"]["populations"] = ["stress"]
    with pytest.raises(ValueError, match="separate stress"):
        m.validate_plan(sealed(plan), verify_files=False)


def test_seed_recipe_big_endian_and_effective_mapping():
    values = [m.SCHEMA, "a" * 64, "calibration", 3, "off", 4, 2048, 29]
    raw = int.from_bytes(
        hashlib.sha256(
            json.dumps(
                values,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode()
        ).digest()[:8],
        "big",
    )
    assert m.frame_seed("a" * 64, "calibration", 3, "off", 4) == raw
    assert m.effective_seed(raw) == raw & ((1 << 63) - 1)
    assert (
        len(
            {
                m.frame_seed("a" * 64, stage, 3, "off", 4)
                for stage in [*m.STAGES, "audit"]
            }
        )
        == 4
    )


def save_ledger(plan):
    rows = list(m.planned_seed_rows(plan))
    np.savez(
        plan["inputs"]["planned_seeds"]["path"],
        stage=np.array([r[0] for r in rows]),
        block=np.array([r[1] for r in rows]),
        population=np.array([r[2] for r in rows]),
        frame=np.array([r[3] for r in rows]),
        raw_seed=np.array([r[4] for r in rows], dtype=np.uint64),
        effective_seed63=np.array([r[5] for r in rows], dtype=np.uint64),
    )
    np.savez(
        plan["inputs"]["seed_exclusions"]["path"],
        raw_seed=np.array([], dtype=np.uint64),
        effective_seed63=np.array([], dtype=np.uint64),
    )
    return rows


def test_seed_partition_includes_audit_and_historical_effective_collisions(plan):
    rows = save_ledger(plan)
    assert m.validate_seed_uniqueness(plan) == 5
    raw = rows[0][4] ^ (1 << 63)
    np.savez(
        plan["inputs"]["seed_exclusions"]["path"],
        raw_seed=np.array([raw], dtype=np.uint64),
        effective_seed63=np.array([m.effective_seed(raw)], dtype=np.uint64),
    )
    with pytest.raises(ValueError, match="historical"):
        m.validate_seed_uniqueness(plan)


def profile():
    frequencies = [0.123, 0.123 - 2 / 128, 0.123 + 2 / 128]
    weights = np.exp(-2j * np.pi * np.array(frequencies)[:, None] * np.arange(128))
    return SimpleNamespace(
        ideal_weights=weights,
        float_packed_weights=weights,
        layout={
            key: value
            for key, value in zip(
                [
                    "target_normalized_frequency",
                    "lower_reference_normalized_frequency",
                    "upper_reference_normalized_frequency",
                ],
                frequencies,
            )
        },
    )


def test_reference_tones_have_declared_gamma_and_no_ideal_target_response():
    p = profile()
    rows = m.reference_rows(p, 0.1)
    gamma = abs(rows.astype(np.complex128) @ p.ideal_weights.conj().T) ** 2 / 128
    np.testing.assert_allclose(gamma[:, 1:], 0.1, rtol=2e-7, atol=1e-9)
    assert gamma[:, 0].max() < 1e-14
    assert rows.shape == (128, 128)
    assert rows.dtype == np.complex64
    np.testing.assert_array_equal(m.reference_rows(p, 0), np.zeros_like(rows))


def test_reference_leakage_is_not_redefined_as_target_data_truth():
    p = profile()
    tones = m.reference_rows(p, 0.1)
    atsc = np.asarray(0.01 * np.tile(p.ideal_weights[1], (128, 1)), np.complex64)
    info = m.projection_diagnostics(atsc, tones, 2, p)["ideal"]
    assert info["atsc_only_gamma_by_term"][1] > 0
    assert info["reference_tones_only_gamma_by_term"][1] == pytest.approx(0.1, rel=2e-7)
    np.testing.assert_allclose(
        np.array(info["combined_gamma_by_term"]),
        np.array(info["atsc_only_gamma_by_term"])
        + info["reference_tones_only_gamma_by_term"]
        + np.array(info["coherent_cross_term_by_term"]),
        atol=1e-15,
    )


def test_exact_cpu_coarse_sums_no_float_or_fine_transform():
    samples = np.full((2, 128), 16, dtype=np.int8)
    weights = np.array(
        [np.full(128, 16), np.tile([16, -16], 64), np.ones(128)], dtype=np.int8
    )
    np.testing.assert_array_equal(
        m.exact_cpu_powers(samples, weights, chunk_rows=1), [2 * 128**2, 0, 2 * 128**2]
    )


def test_evaluation_requires_receipt(plan, tmp_path):
    with pytest.raises(ValueError, match="requires a frozen"):
        m.validate_receipt(plan, tmp_path)


def make_receipt(plan, tmp_path):
    source = tmp_path / "calibration/block-0000/off.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"small calibration artifact identity")
    identities = {"calibration/block-0000/off.npz": m.sha(source)}
    freeze = datetime.now(timezone.utc) - timedelta(hours=1)
    calibration = {
        "schema": "channel29-digital-retained-set-bound-v1",
        "plan_sha256": plan["plan_sha256"],
        "frozen_at": freeze.isoformat(),
        "calibration": {"status": "unsupported_calibration"},
        "source_shards": identities,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    receipt = {
        "schema": "channel29-residual-calibration-receipt-v1",
        "status": "calibration_frozen",
        "plan_sha256": plan["plan_sha256"],
        "frozen_at": (freeze + timedelta(minutes=1)).isoformat(),
        "calibration": m.identity(path),
        "calibration_outputs": identities,
    }
    receipt["receipt_sha256"] = m.digest_payload(receipt, "receipt_sha256")
    Path(plan["evaluation_prerequisite"]["receipt_path"]).write_text(
        json.dumps(receipt)
    )
    return receipt, calibration, path


def test_refused_calibration_receipt_can_authorize_failure_audit(plan, tmp_path):
    make_receipt(plan, tmp_path)
    assert (
        m.validate_receipt(plan, tmp_path)["calibration_status"]
        == "unsupported_calibration"
    )


@pytest.mark.parametrize("change", ["plan", "source_shards", "time", "schema"])
def test_receipt_hash_alone_cannot_wrap_wrong_calibration(plan, tmp_path, change):
    receipt, cal, path = make_receipt(plan, tmp_path)
    if change == "plan":
        cal["plan_sha256"] = "f" * 64
    elif change == "source_shards":
        cal["source_shards"] = {}
    elif change == "time":
        cal["frozen_at"] = plan["frozen_at"]
    else:
        cal["schema"] = "other"
    path.write_text(json.dumps(cal))
    receipt["calibration"] = m.identity(path)
    receipt["receipt_sha256"] = m.digest_payload(receipt, "receipt_sha256")
    Path(plan["evaluation_prerequisite"]["receipt_path"]).write_text(
        json.dumps(receipt)
    )
    with pytest.raises(ValueError):
        m.validate_receipt(plan, tmp_path)


def write_shard(path, plan, **updates):
    stamp = datetime.now(timezone.utc) - timedelta(hours=3)
    meta = {
        "schema": m.SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "protocol_sha256": plan["protocol_sha256"],
        "stage": "calibration",
        "block_index": 0,
        "population": "off",
        "frames": 1,
        "geometry": plan["geometry"],
        "target_norm_sq": 6372,
        "reference_norm_sum_sq": 12713,
        "nominal_injected_data_shelf_db": None,
        "nominal_injected_data_shelf_linear": 0.0,
        "reference_projection_gamma": 0.0,
        "started_utc": stamp.isoformat(),
        "completed_utc": (stamp + timedelta(seconds=1)).isoformat(),
    }
    meta.update(updates)
    seed = m.frame_seed(plan["protocol_sha256"], "calibration", 0, "off", 0)
    np.savez(
        path,
        meta_json=np.array(json.dumps(meta)),
        coarse_marginals_u64=np.ones((1, 3), dtype=np.uint64),
        clip_fraction=np.zeros(1),
        frame_seed=np.array([seed], dtype=np.uint64),
        frame_seed_effective63=np.array([m.effective_seed(seed)], dtype=np.uint64),
    )
    path.with_suffix(".npz.sha256").write_text(m.sha(path) + "\n")


def test_resume_authenticates_powers_metadata_and_seeds(plan, tmp_path):
    path = tmp_path / "off.npz"
    write_shard(path, plan)
    m.validate_shard(path, plan, "calibration", 0, plan["populations"][0])
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="authenticated"):
        m.validate_shard(path, plan, "calibration", 0, plan["populations"][0])


@pytest.mark.parametrize(
    "updates",
    [
        {"nominal_injected_data_shelf_linear": 0.1},
        {"protocol_sha256": "f" * 64},
        {"target_norm_sq": 1},
        {"started_utc": "2020-01-01T00:00:00Z"},
    ],
)
def test_rehashed_wrong_resume_metadata_is_refused(plan, tmp_path, updates):
    path = tmp_path / "off.npz"
    write_shard(path, plan, **updates)
    with pytest.raises(ValueError):
        m.validate_shard(path, plan, "calibration", 0, plan["populations"][0])
