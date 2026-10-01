"""Hardware-free checks of publication ownership and finite study orchestration."""
import importlib.util
import io
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "tools/run_sdr_live_handoff.py"
SPEC = importlib.util.spec_from_file_location("live_handoff", SOURCE)
live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live)


def row():
    return dict(accepted_samples_from_chunk=4, phase="qualified", received_count=8,
                timestamp=98, accepted_file_sample_offset=0, expected_timestamp_known=True,
                expected_timestamp=98, timestamp_gap=False, status_return=0, stream_active=True,
                finite=True, underrun_delta=0, overrun_delta=0, dropped_delta=0,
                peak_component=.02, rms=.005)


def test_partial_json_is_not_published():
    stream = io.BytesIO(b'{"a": 1}')
    assert live.committed_line(stream) is None
    assert stream.tell() == 0
    stream.seek(0, 2)
    stream.write(b"\n")
    stream.seek(0)
    assert live.committed_line(stream) == {"a": 1}
    assert live.committed_line(stream) is None


def test_line_bound():
    with pytest.raises(ValueError, match="line bound"):
        live.committed_line(io.BytesIO(b" " * 65537))


def test_exact_partial_start_span():
    schedule = {"accepted_first_timestamp": 100, "accepted_stop_timestamp_exclusive": 104}
    assert live.accepted_span(row(), schedule, 0) == (100, 4)


@pytest.mark.parametrize("key,value", [("timestamp_gap", True), ("finite", False),
                                     ("stream_active", False), ("underrun_delta", 1),
                                     ("overrun_delta", 1), ("dropped_delta", 1),
                                     ("peak_component", .98), ("rms", .35),
                                     ("accepted_file_sample_offset", 1),
                                     ("expected_timestamp", 97), ("status_return", -1)])
def test_failed_native_row_refused(key, value):
    r = row()
    r[key] = value
    with pytest.raises(ValueError, match="qualification/continuity"):
        live.accepted_span(r, {"accepted_first_timestamp": 100, "accepted_stop_timestamp_exclusive": 104}, 0)


def test_duplicate_span_refused():
    with pytest.raises(ValueError, match="qualification/continuity"):
        live.accepted_span(row(), {"accepted_first_timestamp": 100, "accepted_stop_timestamp_exclusive": 104}, 4)


def test_partial_data_waits_without_fabrication(tmp_path):
    path = tmp_path / "accepted.cfile"
    path.write_bytes(bytes(range(16)))
    assert live.published_bytes(path, 0, 3) is None
    with path.open("ab") as f:
        f.write(bytes(range(16, 24)))
    assert live.published_bytes(path, 1, 2) == bytes(range(8, 24))


def complete_receipt():
    disabled = {"TXEN_A": 0, "EN_TXTSP": 0, "EN_G_TRF": 0, "PD_TXPAD_TRF": 1}
    return {"success": True, "cleanup_confirmed": True, "accepted_interval_confirmed": True,
            "worker_report": {"accepted_samples": 4000000, "mode": "noise", "tx_attempted": False,
                              "payload_submission_attempted": False, "tx_samples_sent": 0,
                              "tx_disabled_register_readbacks": {"before": disabled, "after": disabled.copy()}}}


def test_cleanup_and_tx_evidence_required():
    receipt = complete_receipt()
    assert live.receipt_accepted(receipt, 4000000)
    receipt["cleanup_confirmed"] = False
    assert not live.receipt_accepted(receipt, 4000000)
    receipt = complete_receipt()
    receipt["worker_report"]["tx_disabled_register_readbacks"]["after"]["TXEN_A"] = 1
    assert not live.receipt_accepted(receipt, 4000000)


def test_failure_stops_remaining_attempts_without_retry(tmp_path, monkeypatch):
    plan = {"kind": "primary", "scope": "test", "attempts": [{"directory": f"attempt-{i}"} for i in range(3)]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(live, "verify_plan", lambda _out: plan)
    calls = []
    monkeypatch.setattr(live, "live_attempt", lambda folder, _plan: calls.append(folder.name) or {"success": False})
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    result = live.run(tmp_path, True, True)
    assert calls == ["attempt-0"]
    assert [x["state"] for x in result["states"]] == ["failed", "unattempted", "unattempted"]
    with pytest.raises(FileExistsError):
        live.run(tmp_path, True, True)


def test_plan_tamper_refused_before_any_launch(tmp_path):
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "plan-digest.json").write_text('{"sha256": "incorrect"}')
    with pytest.raises(ValueError, match="study plan changed"):
        live.verify_plan(tmp_path)


def test_no_launch_without_flags(tmp_path):
    with pytest.raises(ValueError, match="authorization"):
        live.run(tmp_path, False, False)
    assert not list(tmp_path.iterdir())


class FakeProcess:
    pid = 90000000

    def __init__(self):
        self.returncode = None
        self.signals = []
        self.waits = 0

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)
        self.returncode = -int(signum)

    def wait(self, timeout=None):
        self.waits += 1
        return self.returncode


def test_supervisor_setup_fault_stops_and_reaps_helper(tmp_path, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(live.subprocess, "Popen", lambda *a, **kw: process)
    with pytest.raises(RuntimeError, match="injected setup"):
        with live.CaptureSupervisor(tmp_path, "/not-a-worker"):
            raise RuntimeError("injected setup")
    assert process.signals == [live.signal.SIGTERM]
    assert process.waits == 1
    receipt = json.loads((tmp_path / "supervisor.json").read_text())
    assert receipt["abort_requested"]
    assert not receipt["forced_kill_cleanup_unconfirmed"]


def test_supervisor_interrupt_forwarded_and_retained(tmp_path, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(live.subprocess, "Popen", lambda *a, **kw: process)
    with live.CaptureSupervisor(tmp_path, "/not-a-worker") as supervisor:
        supervisor.interrupt(live.signal.SIGTERM, None)
        assert supervisor.stop.is_set()
    assert process.signals == [live.signal.SIGTERM]
    assert process.waits == 1
    assert json.loads((tmp_path / "supervisor.json").read_text())["interrupted"]


def test_supervisor_forced_timeout_retains_unqualified_cleanup(tmp_path, monkeypatch):
    class HungProcess(FakeProcess):
        def wait(self, timeout=None):
            if self.returncode is None:
                raise live.subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = HungProcess()
    monkeypatch.setattr(live.subprocess, "Popen", lambda *a, **kw: process)
    with live.CaptureSupervisor(tmp_path, "/not-a-worker"):
        pass
    assert process.returncode == -9
    assert json.loads((tmp_path / "supervisor.json").read_text())["forced_kill_cleanup_unconfirmed"]
