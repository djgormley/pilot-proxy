"""Reference worker tests link a fake API only; no radio access is possible."""
import json
import os
import runpy
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
HEADERS = REPO / "tests/support/limesuite/include"
DRIVER = runpy.run_path(str(REPO / "tools/lime_reference_capture.py"))
FAKE = '\n#include <lime/LimeSuite.h>\n#include <atomic>\n#include <chrono>\n#include <complex>\n#include <cstring>\n#include <cstdlib>\n#include <fstream>\n#include <string>\n#include <thread>\n#include <sys/syscall.h>\n#include <unistd.h>\nstatic double rate=2e6, freq[2]={}, bw[2]={};\nstatic unsigned gain[2]={}; static int antenna[2]={};\nstatic std::atomic<unsigned long long> samples{0};\nstatic std::atomic<bool> tx_activated{false}, tone_sent{false};\nstatic bool rx_started=false, tx_started=false, tx_enabled=false;\nstatic unsigned status_calls=0;\nstatic bool mode(const char* s) { const char* p=getenv("FAKE_MODE"); return p && std::string(p)==s; }\nextern "C" {\nint fsync(int fd){\n const char* delay=getenv("FAKE_SCHEDULE_FSYNC_DELAY_MS");\n if(delay)std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(delay)));\n return static_cast<int>(syscall(SYS_fsync,fd));\n}\nconst char* LMS_GetLibraryVersion(){return "23.11.0-FAKE-OFFLINE-TEST";}\nconst char* LMS_GetLastErrorMessage(){return "fake injected failure";}\nint LMS_GetDeviceList(lms_info_str_t* d){if(d)std::strcpy(d[0],"LimeSDR-Mini, serial=00001d423d9108f273");return 1;}\nint LMS_Open(lms_device_t** d,const lms_info_str_t,void*){*d=reinterpret_cast<void*>(1);return 0;}\nint LMS_Init(lms_device_t*){return 0;} int LMS_Close(lms_device_t*){return 0;}\nint LMS_EnableChannel(lms_device_t*,bool d,size_t,bool value){if(d){if(mode("noise_guard") && value)return -1;tx_enabled=value;}return 0;}\nint LMS_ReadParam(lms_device_t*, LMS7Parameter param,uint16_t* value){\n if(std::string(param.name)=="PD_TXPAD_TRF")*value=tx_enabled?0:1;else *value=tx_enabled?1:0;\n if(mode("disabled_readback_fail"))*value=9;return 0;\n}\nint LMS_SetSampleRate(lms_device_t*,double v,size_t){rate=v;return 0;}\nint LMS_GetSampleRate(lms_device_t*,bool,size_t,double* h,double* r){*h=rate;*r=rate;return 0;}\nint LMS_SetLOFrequency(lms_device_t*,bool d,size_t,double v){freq[d]=v;return 0;}\nint LMS_GetLOFrequency(lms_device_t*,bool d,size_t,double* v){*v=freq[d];return 0;}\nint LMS_SetLPFBW(lms_device_t*,bool d,size_t,double v){bw[d]=v;return 0;}\nint LMS_GetLPFBW(lms_device_t*,bool d,size_t,double* v){*v=bw[d];return 0;}\nint LMS_SetGaindB(lms_device_t*,bool d,size_t,unsigned v){\n if(d){if(v>50)return -1;const char* path=getenv("FAKE_TX_GAIN_LOG");if(path)std::ofstream(path,std::ios::app)<<v<<"\\n";}\n gain[d]=v;return 0;\n}\nint LMS_GetGaindB(lms_device_t*,bool d,size_t,unsigned* v){*v=d?(mode("gain_mismatch")?7:(gain[d]==0?6:gain[d])):gain[d];return 0;}\nint LMS_SetAntenna(lms_device_t*,bool d,size_t,size_t v){\n if(d && v==LMS_PATH_TX2){\n  if(mode("noise_guard") || mode("antenna_fail") || samples>0)return -1;\n  tx_activated.store(true);\n  const char* marker=getenv("FAKE_TX_MARKER");if(marker)std::ofstream(marker)<<"TX after RX ready";\n }\n antenna[d]=v;return 0;\n}\nint LMS_GetAntenna(lms_device_t*,bool d,size_t){return antenna[d];}\nint LMS_SetupStream(lms_device_t*,lms_stream_t* s){if(mode("noise_guard") && s->isTx)return -1;return 0;}\nint LMS_DestroyStream(lms_device_t*,lms_stream_t*){return 0;}\nint LMS_StartStream(lms_stream_t* s){\n if(s->isTx){if(rx_started || samples>0)return -1;tx_started=true;}\n else{if(tx_enabled && !tx_started)return -1;rx_started=true;}return 0;\n}\nint LMS_StopStream(lms_stream_t*){return 0;}\nint LMS_RecvStream(lms_stream_t*,void* output,size_t n,lms_stream_meta_t* m,unsigned){\n std::this_thread::sleep_for(std::chrono::duration<double>(mode("slow")?.01:static_cast<double>(n)/rate));\n if(mode("no_rx"))return -1;\n m->timestamp=samples.load(); if(mode("gap") && samples>0)m->timestamp+=samples/16384;\n if(mode("late_gap") && samples>1100000)m->timestamp+=samples/16384;\n auto* p=static_cast<std::complex<float>*>(output);\n for(size_t i=0;i<n;++i)p[i]=mode("clip")?std::complex<float>(1,0):std::complex<float>(.001F+float((m->timestamp+i)%251)/100000.F,.002F-float((m->timestamp+i)%127)/100000.F);\n samples+=n;return static_cast<int>(n);\n}\nint LMS_SendStream(lms_stream_t*,const void*,size_t n,const lms_stream_meta_t* m,unsigned){\n if(mode("noise_guard") || mode("tx_fail") || !tx_started || !rx_started || samples<static_cast<unsigned long long>(rate*.35) || !m->waitForTimestamp)return -1;\n tone_sent.store(true);const char* marker=getenv("FAKE_PAYLOAD_MARKER");if(marker)std::ofstream(marker)<<"payload";return static_cast<int>(mode("partial_send")?std::min<size_t>(n,257):n);\n}\nint LMS_GetStreamStatus(lms_stream_t* stream,lms_stream_status_t* s){\n std::memset(s,0,sizeof(*s));s->active=!mode("inactive");\n if(stream->isTx){\n  if(mode("host_empty"))s->underrun=1;\n  if(mode("tx_drop") && tone_sent.load())s->droppedPackets=1;\n  if(mode("tx_overrun") && tone_sent.load())s->overrun=1;\n }\n if(!stream->isTx){\n  ++status_calls;\n  if(mode("startup_drop") && status_calls==1)s->droppedPackets=2;\n  if(mode("candidate_drop") && status_calls==35)s->droppedPackets=3;\n  if(mode("always_drop"))s->droppedPackets=1;\n  if(mode("late_drop") && tone_sent.load())s->droppedPackets=1;\n }\n return 0;\n}\n}\n'


@pytest.fixture(scope="session")
def fake_build(tmp_path_factory):
    base = tmp_path_factory.mktemp("fake-reference-no-hardware")
    source = base / "fake.cpp"
    source.write_text(FAKE)
    library, worker = base / "libFakeReference.so", base / "worker"
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-pthread", "-I", str(HEADERS), str(source), "-o", str(library)], check=True)
    subprocess.run(["g++", "-std=c++17", "-pthread", "-I", str(HEADERS), str(REPO / "tools/lime_reference_worker.cpp"), str(library), "-o", str(worker)], check=True)
    manifest = base / "build.json"
    manifest.write_text(json.dumps({"worker": str(worker), "worker_sha256": DRIVER["sha"](worker), "library": str(library),
        "inputs": {str(source): DRIVER["sha"](source), str(library): DRIVER["sha"](library)}}))
    return manifest


def prepare(tmp_path, fake_build, mode="tone", seconds=.04, **kwargs):
    out = tmp_path / mode
    DRIVER["prepare"](out, build_manifest=fake_build, serial="0x1d423d9108f273", mode=mode, record_seconds=seconds, **kwargs)
    return out


def capture(out):
    try:
        return DRIVER["capture"](out, hardware_authorized=True, transmit=True, rf_confined_authorized=True)
    except RuntimeError as error:
        evidence = []
        for name in ("receipt.json", "worker.log"):
            path = out / name
            if path.exists():
                evidence.append(f"{name}: {path.read_text()}")
        raise RuntimeError(str(error) + "\n" + "\n".join(evidence)) from error


@pytest.mark.parametrize("mode", ["noise", "txzero", "tone"])
def test_three_modes_have_exact_accepted_sample_and_timestamp_mapping(tmp_path, fake_build, monkeypatch, mode):
    if mode == "noise":
        monkeypatch.setenv("FAKE_MODE", "noise_guard")
    out = prepare(tmp_path, fake_build, mode, seconds=2)
    result = capture(out)
    r = result["worker_report"]
    assert result["success"] and result["accepted_interval_confirmed"] and result["cleanup_confirmed"]
    assert r["mode"] == mode and r["accepted_samples"] == 4000000
    assert r["accepted_stop_timestamp_exclusive"] - r["accepted_first_timestamp"] == 4000000
    schedule = json.loads((out / "schedule.json").read_text())
    assert schedule["accepted_first_timestamp"] == r["accepted_first_timestamp"]
    assert schedule["accepted_stop_timestamp_exclusive"] == r["accepted_stop_timestamp_exclusive"]
    assert schedule["schedule_base_timestamp"] >= r["rx_ready_timestamp"]
    assert schedule["accepted_first_timestamp"] == schedule["schedule_base_timestamp"] + (400000 if mode == "noise" else 700000)
    accepted = np.fromfile(out / "accepted.cfile", dtype="<c8")
    raw = np.fromfile(out / "rx.cfile", dtype="<c8")
    offset = r["accepted_first_timestamp"] - r["first_rx_timestamp"]
    assert accepted.tobytes() == raw[offset:offset + 4000000].tobytes()
    assert len(accepted) // 128 == 31250 and len(accepted) % 128 == 0
    rows = [json.loads(line) for line in (out / "rx-chunks.jsonl").read_text().splitlines()]
    mapped = [row for row in rows if row["accepted_samples_from_chunk"]]
    cursor = 0
    for row in mapped:
        assert row["accepted_file_sample_offset"] == cursor
        cursor += row["accepted_samples_from_chunk"]
    assert cursor == 4000000
    assert r["qualified_rx_dropped"] == r["qualified_rx_overrun"] == r["qualified_rx_underrun"] == 0
    tx = np.fromfile(out / "tx.cfile", dtype="<c8")
    if mode == "noise":
        assert len(tx) == r["tx_samples_sent"] == 0
        assert not r["payload_submission_attempted"] and not r["tx_attempted"]
        assert not r["tx_started_before_rx_qualification"] and r["native_tx_gain_db"] is None
        assert r["tx_disabled_register_readbacks"]["before"] == r["tx_disabled_register_readbacks"]["after"] == {
            "TXEN_A": 0, "EN_TXTSP": 0, "EN_G_TRF": 0, "PD_TXPAD_TRF": 1}
    else:
        assert len(tx) == r["tx_samples_sent"] == 4600000
        assert r["accepted_first_timestamp"] == r["tx_start_timestamp"] + 300000
        assert r["native_tx_gain_db"] == 50
        if mode == "txzero":
            assert np.all(tx == 0)
        else:
            steady = tx[300000:4300000]
            np.testing.assert_allclose(np.abs(steady), .005, rtol=2e-7)
            assert np.all(tx[:100000] == 0) and np.all(tx[-100000:] == 0)


@pytest.mark.parametrize("mode", ["noise", "txzero", "tone"])
def test_launch_authorization_and_immutable_attempts(tmp_path, fake_build, mode):
    out = prepare(tmp_path, fake_build, mode)
    with pytest.raises(ValueError, match="explicit"):
        DRIVER["capture"](out)
    assert not (out / "launch.json").exists()
    if mode != "noise":
        with pytest.raises(ValueError, match="transmit"):
            DRIVER["capture"](out, hardware_authorized=True, rf_confined_authorized=True)
        assert not (out / "launch.json").exists()
    capture(out)
    before = {p.name: DRIVER["sha"](p) for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        capture(out)
    assert before == {p.name: DRIVER["sha"](p) for p in out.iterdir()}


@pytest.mark.parametrize("bad", [0, -.1, 4.01, float("nan"), .02000001])
def test_duration_must_be_bounded_and_resolve_to_integer_samples(tmp_path, fake_build, bad):
    with pytest.raises(ValueError):
        prepare(tmp_path, fake_build, seconds=bad)
    assert not (tmp_path / "tone").exists()


@pytest.mark.parametrize("mode,failure", [("noise", "disabled_readback_fail"), ("noise", "late_gap"),
    ("tone", "gain_mismatch"), ("tone", "tx_fail"), ("txzero", "tx_drop"),
    ("tone", "clip"), ("tone", "no_rx"), ("tone", "inactive")])
def test_failure_records_partial_evidence_and_cleanup(tmp_path, fake_build, monkeypatch, mode, failure):
    out = prepare(tmp_path, fake_build, mode)
    monkeypatch.setenv("FAKE_MODE", failure)
    with pytest.raises(RuntimeError, match="reference capture failed"):
        capture(out)
    receipt = json.loads((out / "receipt.json").read_text())
    assert not receipt["success"] and receipt["cleanup_confirmed"]
    assert receipt["worker_report"]["error"]
    assert (out / "accepted.cfile").exists() and (out / "startup.cfile").exists()
    assert receipt["worker_report"]["cleanup"]["disable_tx_return"] == 0


def test_startup_counter_diagnostics_and_partial_send_progress(tmp_path, fake_build, monkeypatch):
    for mode in ("startup_drop", "partial_send", "host_empty"):
        folder = tmp_path / mode
        folder.mkdir()
        out = prepare(folder, fake_build)
        monkeypatch.setenv("FAKE_MODE", mode)
        result = capture(out)
        assert result["success"]
        if mode == "startup_drop":
            assert result["worker_report"]["startup_rx_dropped"] == 2
        if mode == "host_empty":
            assert result["worker_report"]["tx_underrun"] > 0


def test_changed_frozen_waveform_refuses_before_any_launch(tmp_path, fake_build):
    out = prepare(tmp_path, fake_build)
    (out / "tx.cfile").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        capture(out)
    assert not (out / "launch.json").exists()


def test_signal_during_payload_records_failure_and_cleanup(tmp_path, fake_build):
    out = prepare(tmp_path, fake_build, seconds=2)
    plan = DRIVER["load"](out)
    marker = tmp_path / "payload-start"
    env = os.environ.copy()
    env.update(LD_PRELOAD=plan["library"], FAKE_MODE="slow", FAKE_PAYLOAD_MARKER=str(marker))
    process = subprocess.Popen(plan["command"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    until = time.monotonic() + 5
    while not marker.exists() and process.poll() is None and time.monotonic() < until:
        time.sleep(.01)
    assert marker.exists()
    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=5)
    report = json.loads((out / "worker-status.json").read_text())
    assert not report["success"] and report["payload_submission_attempted"]
    assert report["cleanup"]["disable_tx_return"] == report["cleanup"]["antenna_off_readback"] == 0


def test_parent_exception_stops_worker_before_receipt(tmp_path, fake_build, monkeypatch):
    out = prepare(tmp_path, fake_build, "noise")
    monkeypatch.setenv("FAKE_MODE", "slow")
    real_popen, real_sleep = subprocess.Popen, time.sleep
    children = []
    def launch(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        real_sleep(.03)
        return child
    def fail(_seconds):
        raise RuntimeError("injected parent error")
    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setitem(DRIVER["capture"].__globals__, "time", SimpleNamespace(monotonic=time.monotonic, sleep=fail))
    with pytest.raises(RuntimeError, match="injected parent error"):
        capture(out)
    assert len(children) == 1 and children[0].poll() is not None
    receipt = json.loads((out / "receipt.json").read_text())
    assert not receipt["success"] and receipt["cleanup_confirmed"]


def test_schedule_fsync_delay_preserves_exact_interval(tmp_path, fake_build, monkeypatch):
    monkeypatch.setenv("FAKE_SCHEDULE_FSYNC_DELAY_MS", "150")
    out = prepare(tmp_path, fake_build, "tone")
    result = capture(out)
    report = result["worker_report"]
    assert report["accepted_samples"] == 80000
    assert report["accepted_first_timestamp"] == report["schedule_base_timestamp"] + 700000
    assert report["accepted_stop_timestamp_exclusive"] - report["accepted_first_timestamp"] == 80000
