"""Hardware-free schedule, source-binding and fail-stop orchestration checks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sdr_antenna_controls", ROOT / "tools/run_sdr_antenna_controls.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeCapture:
    def __init__(self, fail_at=None, failure="raise"):
        self.fail_at = fail_at
        self.failure = failure
        self.calls = []

    def prepare(self, output, *, mode, **settings):
        output = Path(output)
        output.mkdir()
        plan = {"mode": mode, "record_samples": 4_000_000, "record_seconds": 2.,
                "frequency_hz": 500_000_000, "sample_rate_hz": 2_000_000, "native_rx_gain_db": 30,
                "requested_native_tx_gain_db": 0 if mode == "noise" else 50,
                "expected_native_tx_gain_readback_db": None if mode == "noise" else 50,
                "tone_peak_component": .005 if mode == "tone" else 0.}
        runner.write_new(output / "plan.json", plan)
        runner.write_new(output / "plan-digest.json", {"sha256": runner.sha(output / "plan.json")})
        (output / "tx.cfile").write_bytes(b"fake payload; never sent")
        assert settings["record_seconds"] == 2.
        assert settings["tone_component"] == .005
        assert settings["native_tx_gain_db"] == settings["expected_tx_gain_readback_db"] == 50
        return plan

    def load(self, output):
        return json.loads((Path(output) / "plan.json").read_text())

    def capture(self, output, **authorization):
        self.calls.append(Path(output))
        assert authorization == {"hardware_authorized": True, "transmit": True, "rf_confined_authorized": True}
        fail = len(self.calls) - 1 == self.fail_at
        if fail and self.failure == "raise":
            raise RuntimeError("fake transport failure; hardware is not available")
        output = Path(output)
        plan = self.load(output)
        receipt = {"success": True, "cleanup_confirmed": True, "accepted_interval_confirmed": True,
                   "hardware_attempted": True, "plan_sha256": runner.sha(output / "plan.json"), "mode": plan["mode"]}
        if fail and self.failure == "cleanup":
            receipt["cleanup_confirmed"] = False
        if fail and self.failure == "identity":
            receipt["plan_sha256"] = "0" * 64
        if fail and self.failure == "interval":
            receipt["accepted_interval_confirmed"] = False
        # Sparse fake file: byte-count check only; never a measured IQ record.
        with (output / "accepted.cfile").open("wb") as stream:
            stream.truncate(16 if fail and self.failure == "short" else 32_000_000)
        if not (fail and self.failure == "receipt"):
            runner.write_new(output / "receipt.json", receipt)
        return receipt


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    analyzer = tmp_path / "analyzer.py"
    analyzer.write_text("# Frozen fake analyzer; no hardware API\n")
    worker = tmp_path / "worker"
    worker.write_text("not executable; fake test fixture\n")
    library = tmp_path / "library"
    library.write_text("not a library; fake test fixture\n")
    build = tmp_path / "build.json"
    runner.write_new(build, {"inputs": {str(worker): runner.sha(worker), str(library): runner.sha(library)},
                             "worker": str(worker), "worker_sha256": runner.sha(worker), "library": str(library)})
    scale_reference = tmp_path / "adapter_metadata.json"
    runner.write_new(scale_reference, {"sample_quantization": {"scale": 500.0}})
    scale_summary = tmp_path / "summary.json"
    runner.write_new(scale_summary, {"fixture": "independent scale reference"})
    monkeypatch.setattr(runner, "SCALE_REFERENCE", scale_reference)
    monkeypatch.setattr(runner, "SCALE_SUMMARY", scale_summary)
    helper = FakeCapture()
    monkeypatch.setattr(runner, "capture_helper", lambda: helper)
    output = tmp_path / "study"
    plan = runner.prepare(output, build_manifest=build, analysis_sources=[analyzer])
    return output, plan, helper, analyzer, build


def launch(output):
    return runner.run(output, expected_plan_sha256=runner.sha(output / "plan.json"),
                      hardware_authorized=True, transmit=True, rf_confined_authorized=True)


def test_prepare_is_hardware_free_and_binds_exact_five_randomized_triplets(prepared):
    output, plan, helper, analyzer, _ = prepared
    assert helper.calls == []
    assert plan["randomization"]["seed"] == 20260909
    assert plan["analysis_config"] == {"input_rate_hz": 2000000, "mixer_hz": 100000., "target_bin": 0, "sample_scale": 500.}
    assert plan["source_sha256"][str(analyzer)] == runner.sha(analyzer)
    assert plan["numerical_runtime"] == runner.numerical_runtime()
    assert plan["analysis_scale_reference"]["summary_sha256"] == runner.sha(runner.SCALE_SUMMARY)
    for source, item in plan["source_snapshots"].items():
        assert runner.sha(output / item["relative"]) == item["sha256"] == plan["source_sha256"][source]
    assert plan["hardware_attempted_during_preparation"] is False
    assert plan["attempt_count"] == 15
    assert plan["sample_targets"]["accepted_if_all_complete"] == 60_000_000
    assert plan["sample_targets"]["commanded_tx_payload_seconds_if_all_complete"] == 23
    assert [{key: row[key] for key in runner.mode_order()[0]} for row in plan["attempts"]] == runner.mode_order()
    for triplet in range(5):
        assert {row["mode"] for row in plan["attempts"] if row["triplet_index"] == triplet} == {"noise", "txzero", "tone"}
    assert not (output / "launch.json").exists()


def test_success_runs_each_attempt_once_and_refuses_second_launch(prepared):
    output, plan, helper, _, _ = prepared
    result = launch(output)
    assert result["status"] == "complete"
    assert result["complete_attempts"] == result["attempted"] == len(helper.calls) == 15
    assert result["remaining_unattempted"] == 0
    assert [p.name for p in helper.calls] == [row["directory"] for row in plan["attempts"]]
    with pytest.raises(ValueError, match="already launched"):
        launch(output)
    assert len(helper.calls) == 15


@pytest.mark.parametrize("position", [0, 4, 14])
def test_first_failed_capture_stops_and_leaves_rest_unattempted(prepared, position):
    output, plan, helper, _, _ = prepared
    helper.fail_at = position
    with pytest.raises(RuntimeError, match="study stopped"):
        launch(output)
    receipt = json.loads((output / "run-receipt.json").read_text())
    assert len(helper.calls) == position + 1
    assert receipt["status"] == "stopped_failed"
    assert receipt["complete_attempts"] == position
    assert receipt["attempted"] == position + 1
    assert receipt["remaining_unattempted"] == 14 - position
    assert receipt["attempts"][position]["status"] == "failed"
    for item in plan["attempts"][position + 1:]:
        assert {p.name for p in (output / item["directory"]).iterdir()} == {"plan.json", "plan-digest.json", "tx.cfile"}


@pytest.mark.parametrize("failure", ["cleanup", "identity", "interval", "short", "receipt"])
def test_success_flag_alone_cannot_continue_after_failed_contract(prepared, failure):
    output, _, helper, _, _ = prepared
    helper.fail_at = 1
    helper.failure = failure
    with pytest.raises(RuntimeError, match="study stopped"):
        launch(output)
    assert len(helper.calls) == 2
    receipt = json.loads((output / "run-receipt.json").read_text())
    assert receipt["complete_attempts"] == 1 and receipt["remaining_unattempted"] == 13


@pytest.mark.parametrize("target", ["analyzer", "plan", "payload", "attempt_plan", "build", "snapshot"])
def test_source_or_plan_tamper_refuses_before_any_hardware(prepared, target):
    output, plan, helper, analyzer, build = prepared
    expected = runner.sha(output / "plan.json")
    path = {"analyzer": analyzer, "plan": output / "plan.json", "build": build,
            "payload": output / plan["attempts"][3]["directory"] / "tx.cfile",
            "attempt_plan": output / plan["attempts"][3]["directory"] / "plan.json",
            "snapshot": output / plan["source_snapshots"][str(analyzer)]["relative"]}[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        runner.run(output, expected_plan_sha256=expected, hardware_authorized=True, transmit=True, rf_confined_authorized=True)
    assert helper.calls == [] and not (output / "launch.json").exists()


def test_nonempty_preparation_is_not_reused(prepared):
    output, _, helper, analyzer, build = prepared
    with pytest.raises(FileExistsError, match="new empty"):
        runner.prepare(output, build_manifest=build, analysis_sources=[analyzer])
    assert helper.calls == []


def test_all_explicit_flags_required_before_launch(prepared):
    output, _, helper, _, _ = prepared
    with pytest.raises(ValueError, match="Run requires"):
        runner.run(output, expected_plan_sha256=runner.sha(output / "plan.json"), hardware_authorized=True)
    assert helper.calls == [] and not (output / "launch.json").exists()


def test_existing_attempt_artifact_refuses_whole_study(prepared):
    output, plan, helper, _, _ = prepared
    (output / plan["attempts"][7]["directory"] / "launch.json").write_text("{}")
    with pytest.raises(ValueError, match="already launched"):
        launch(output)
    assert helper.calls == []


def test_source_change_during_capture_stops_before_next_attempt(prepared, monkeypatch):
    output, _, helper, analyzer, _ = prepared
    original = helper.capture
    def changed_source_capture(*args, **kwargs):
        result = original(*args, **kwargs)
        analyzer.write_text("# changed after acquisition started\n")
        return result
    monkeypatch.setattr(helper, "capture", changed_source_capture)
    with pytest.raises(RuntimeError, match="Frozen source changed during capture"):
        launch(output)
    receipt = json.loads((output / "run-receipt.json").read_text())
    assert len(helper.calls) == 1 and receipt["remaining_unattempted"] == 14
    assert receipt["attempts"][0]["status"] == "failed"


def test_reviewed_plan_hash_is_required_even_if_local_digest_is_rewritten(prepared):
    output, _, helper, _, _ = prepared
    reviewed = runner.sha(output / "plan.json")
    value = json.loads((output / "plan.json").read_text())
    value["settings"]["tone_peak_component"] = .009
    (output / "plan.json").write_text(json.dumps(value))
    (output / "plan-digest.json").write_text(json.dumps({"sha256": runner.sha(output / "plan.json")}))
    with pytest.raises(ValueError, match="Reviewed plan identity differs"):
        runner.run(output, expected_plan_sha256=reviewed, hardware_authorized=True, transmit=True, rf_confined_authorized=True)
    assert helper.calls == [] and not (output / "launch.json").exists()
