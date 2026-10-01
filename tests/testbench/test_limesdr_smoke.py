"""Compile the real worker against a fake LimeSuite; never link/open hardware."""
from pathlib import Path
import json
import os
import runpy
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
HEADERS = REPO / "tests/support/limesuite/include"
DRIVER = runpy.run_path(str(REPO/"tools/limesdr_smoke_capture.py"))
FAKE = r"""
#include <lime/LimeSuite.h>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
static double rate=2e6, freq[2]={}, bw[2]={};
static unsigned gain[2]={}; static int antenna[2]={};
static std::atomic<unsigned long long> samples{0};
static std::atomic<bool> tx_activated{false}, tone_sent{false};
static bool rx_started=false, tx_started=false;
static unsigned status_calls=0;
static bool mode(const char* s) { const char* p=getenv("FAKE_MODE"); return p && std::string(p)==s; }
extern "C" {
const char* LMS_GetLibraryVersion(){return "23.11.0-FAKE-OFFLINE-TEST";}
const char* LMS_GetLastErrorMessage(){return "fake injected failure";}
int LMS_GetDeviceList(lms_info_str_t* d){if(d)std::strcpy(d[0],"LimeSDR-Mini, serial=00001d423d9108f273");return 1;}
int LMS_Open(lms_device_t** d,const lms_info_str_t,void*){*d=reinterpret_cast<void*>(1);return 0;}
int LMS_Init(lms_device_t*){return 0;} int LMS_Close(lms_device_t*){return 0;}
int LMS_EnableChannel(lms_device_t*,bool,size_t,bool){return 0;}
int LMS_SetSampleRate(lms_device_t*,double v,size_t){rate=v;return 0;}
int LMS_GetSampleRate(lms_device_t*,bool,size_t,double* h,double* r){*h=rate;*r=rate;return 0;}
int LMS_SetLOFrequency(lms_device_t*,bool d,size_t,double v){freq[d]=v;return 0;}
int LMS_GetLOFrequency(lms_device_t*,bool d,size_t,double* v){*v=freq[d];return 0;}
int LMS_SetLPFBW(lms_device_t*,bool d,size_t,double v){bw[d]=v;return 0;}
int LMS_GetLPFBW(lms_device_t*,bool d,size_t,double* v){*v=bw[d];return 0;}
int LMS_SetGaindB(lms_device_t*,bool d,size_t,unsigned v){
 if(d){if(v>50)return -1;const char* path=getenv("FAKE_TX_GAIN_LOG");if(path)std::ofstream(path,std::ios::app)<<v<<"\n";}
 gain[d]=v;return 0;
}
int LMS_GetGaindB(lms_device_t*,bool d,size_t,unsigned* v){*v=d?(mode("gain_mismatch")?7:(gain[d]==0?6:gain[d])):gain[d];return 0;}
int LMS_SetAntenna(lms_device_t*,bool d,size_t,size_t v){
 if(d && v==LMS_PATH_TX2){
  if(mode("antenna_fail") || samples>0)return -1;
  tx_activated.store(true);
  const char* marker=getenv("FAKE_TX_MARKER");if(marker)std::ofstream(marker)<<"TX after RX ready";
 }
 antenna[d]=v;return 0;
}
int LMS_GetAntenna(lms_device_t*,bool d,size_t){return antenna[d];}
int LMS_SetupStream(lms_device_t*,lms_stream_t*){return 0;}
int LMS_DestroyStream(lms_device_t*,lms_stream_t*){return 0;}
int LMS_StartStream(lms_stream_t* s){
 if(s->isTx){if(rx_started || samples>0)return -1;tx_started=true;}
 else{if(!tx_started)return -1;rx_started=true;}return 0;
}
int LMS_StopStream(lms_stream_t*){return 0;}
int LMS_RecvStream(lms_stream_t*,void* output,size_t n,lms_stream_meta_t* m,unsigned){
 std::this_thread::sleep_for(std::chrono::milliseconds(mode("slow")?10:1));
 if(mode("no_rx"))return -1;
 m->timestamp=samples.load(); if(mode("gap") && samples>0)m->timestamp+=samples/16384;
 if(mode("late_gap") && tone_sent.load())m->timestamp+=samples/16384;
 auto* p=static_cast<std::complex<float>*>(output);
 for(size_t i=0;i<n;++i)p[i]=mode("clip")?std::complex<float>(1,0):std::complex<float>(.001F,.002F);
 samples+=n;return static_cast<int>(n);
}
int LMS_SendStream(lms_stream_t*,const void*,size_t n,const lms_stream_meta_t* m,unsigned){
 if(mode("tx_fail") || !tx_started || !rx_started || samples<static_cast<unsigned long long>(rate*.35) || !m->waitForTimestamp)return -1;
 tone_sent.store(true);return static_cast<int>(n);
}
int LMS_GetStreamStatus(lms_stream_t* stream,lms_stream_status_t* s){
 std::memset(s,0,sizeof(*s));s->active=true;
 if(stream->isTx){
  if(mode("host_empty"))s->underrun=1;
  if(mode("tx_drop") && tone_sent.load())s->droppedPackets=1;
  if(mode("tx_overrun") && tone_sent.load())s->overrun=1;
 }
 if(!stream->isTx){
  ++status_calls;
  if(mode("startup_drop") && status_calls==1)s->droppedPackets=2;
  if(mode("candidate_drop") && status_calls==35)s->droppedPackets=3;
  if(mode("always_drop"))s->droppedPackets=1;
  if(mode("late_drop") && tone_sent.load())s->droppedPackets=1;
 }
 return 0;
}
}
"""


@pytest.fixture(scope="session")
def fake_build(tmp_path_factory):
    if not (HEADERS/"lime/LimeSuite.h").exists():
        pytest.skip("preserved LimeSuite 23.11 headers are required for C++ offline tests")
    base = tmp_path_factory.mktemp("fake-lime-smoke-only")
    source = base/"fake.cpp"
    source.write_text(FAKE)
    library = base/"libFakeLimeSmoke.so"
    worker = base/"worker"
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-pthread", "-I", str(HEADERS), str(source), "-o", str(library)], check=True)
    subprocess.run(["g++", "-std=c++17", "-pthread", "-I", str(HEADERS), str(REPO/"tools/limesdr_smoke_worker.cpp"), str(library), "-o", str(worker)], check=True)
    manifest = base/"build.json"
    manifest.write_text(json.dumps({"worker": str(worker), "worker_sha256": DRIVER["sha"](worker),
                                   "library": str(library), "inputs": {str(source): DRIVER["sha"](source), str(library): DRIVER["sha"](library)}}))
    return manifest


def prepared(tmp_path, fake_build):
    output = tmp_path/"attempt"
    DRIVER["prepare"](output, build_manifest=fake_build, serial="0x1d423d9108f273", frequency_hz=500e6, native_tx_gain_db=0, expected_tx_gain_readback_db=6)
    return output


def test_preparation_never_launches_and_requires_new_directory(tmp_path, fake_build):
    out = prepared(tmp_path, fake_build)
    tx = np.fromfile(out/"tx.cfile", dtype=np.complex64)
    assert len(tx) == 300000
    assert np.all(tx[:100000] == 0) and np.all(tx[-100000:] == 0)
    assert np.abs(tx).max() <= .005000001
    assert not (out/"launch.json").exists()
    with pytest.raises(ValueError, match="explicit"):
        DRIVER["capture"](out)
    assert not (out/"launch.json").exists()
    with pytest.raises(FileExistsError):
        DRIVER["prepare"](out, build_manifest=fake_build, serial="0x1d423d9108f273", frequency_hz=500e6, native_tx_gain_db=0, expected_tx_gain_readback_db=6)


def test_both_starts_precede_qualification_and_tone_waits_for_clean_rx(tmp_path, fake_build):
    out = prepared(tmp_path, fake_build)
    result = DRIVER["capture"](out, transmit=True, rf_confined_authorized=True)
    report = result["worker_report"]
    assert report["library_version"].endswith("FAKE-OFFLINE-TEST")
    assert result["success"] and result["cleanup_confirmed"]
    assert report["rx_ready_timestamp"] < report["tx_start_timestamp"]
    assert report["tx_samples_sent"] == 300000
    assert report["tone_submission_attempted"] and report["tx_started_before_rx_qualification"]
    assert report["native_tx_gain_db"] == 6
    assert report["requested_native_tx_gain_db"] == 0
    assert report["expected_native_tx_gain_readback_db"] == 6
    assert report["lpf_configuration_includes_internal_filter_tuning"]
    assert report["rx_lpf_hz"] == 1500000 and report["tx_lpf_hz"] == 5000000
    assert (out/"rx.cfile").stat().st_size == report["captured_samples"]*8
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        DRIVER["capture"](out, transmit=True, rf_confined_authorized=True)
    assert all((out/name).read_bytes() == data for name, data in before.items())


@pytest.mark.parametrize("mode,tx_attempted", [("no_rx", True), ("gap", True), ("clip", True), ("tx_fail", True), ("antenna_fail", True), ("gain_mismatch", False)])
def test_worker_failure_halts_and_cleans_up(tmp_path, fake_build, monkeypatch, mode, tx_attempted):
    out = prepared(tmp_path, fake_build)
    monkeypatch.setenv("FAKE_MODE", mode)
    with pytest.raises(RuntimeError, match="smoke failed"):
        DRIVER["capture"](out, transmit=True, rf_confined_authorized=True)
    result = json.loads((out/"receipt.json").read_text())
    assert not result["success"] and result["cleanup_confirmed"]
    assert result["worker_report"]["tx_attempted"] is tx_attempted
    assert result["worker_report"]["cleanup"]["disable_tx_return"] == 0
    assert (out/"rx.cfile").exists()
    if mode == "gain_mismatch":
        assert result["worker_report"]["captured_samples"] == 0
        assert result["worker_report"]["tx_samples_sent"] == 0


def test_changed_frozen_waveform_refuses_before_launch(tmp_path, fake_build):
    out = prepared(tmp_path, fake_build)
    (out/"tx.cfile").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        DRIVER["capture"](out, transmit=True, rf_confined_authorized=True)
    assert not (out/"launch.json").exists()


def test_signal_after_tx_activation_records_cleanup(tmp_path, fake_build):
    out = prepared(tmp_path, fake_build)
    plan = DRIVER["load"](out)
    marker = tmp_path/"fake-tx-started"
    env = os.environ.copy()
    env.update(LD_PRELOAD=plan["library"], FAKE_MODE="slow", FAKE_TX_MARKER=str(marker))
    process = subprocess.Popen(plan["command"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    until = time.monotonic()+5
    while not marker.exists() and process.poll() is None and time.monotonic() < until:
        time.sleep(.01)
    assert marker.exists()
    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=5)
    report = json.loads((out/"worker-status.json").read_text())
    assert not report["success"] and report["tx_attempted"]
    assert report["cleanup"]["disable_tx_return"] == 0
    assert report["cleanup"]["antenna_off_readback"] == 0


def test_parent_exception_stops_launched_worker_before_receipt(tmp_path, fake_build, monkeypatch):
    out = prepared(tmp_path, fake_build)
    monkeypatch.setenv("FAKE_MODE", "slow")
    real_popen = subprocess.Popen
    children = []

    def injected_parent_failure(_seconds):
        raise RuntimeError("injected parent failure")

    real_sleep = time.sleep
    def tracked_with_real_wait(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        real_sleep(.03)
        return child

    monkeypatch.setattr(subprocess, "Popen", tracked_with_real_wait)
    monkeypatch.setitem(DRIVER["capture"].__globals__, "time",
                        SimpleNamespace(monotonic=time.monotonic, sleep=injected_parent_failure))
    with pytest.raises(RuntimeError, match="injected parent failure"):
        DRIVER["capture"](out, transmit=True, rf_confined_authorized=True)
    assert len(children) == 1 and children[0].poll() is not None
    receipt = json.loads((out/"receipt.json").read_text())
    assert "injected parent failure" in receipt["launch_error"]
    assert not receipt["success"] and not receipt["forced_kill"]
    assert receipt["cleanup_confirmed"]


@pytest.mark.parametrize("expected", [None, -1, 51, True, 6.0])
def test_expected_tx_readback_requires_explicit_bounded_integer(tmp_path, fake_build, expected):
    with pytest.raises(ValueError, match="expected TX readback"):
        DRIVER["prepare"](tmp_path/"invalid", build_manifest=fake_build, serial="0x1d423d9108f273",
                          frequency_hz=500e6, native_tx_gain_db=0, expected_tx_gain_readback_db=expected)
    assert not (tmp_path/"invalid").exists()


def test_expected_readback_does_not_modify_minimum_tx_request(tmp_path, fake_build):
    out = prepared(tmp_path, fake_build)
    plan = DRIVER["load"](out)
    assert plan["requested_native_tx_gain_db"] == 0
    assert plan["expected_native_tx_gain_readback_db"] == 6
    assert plan["measured_power_dbm"] is None and not plan["power_calibrated"]
    assert "--expected-tx-gain-readback-db" in plan["command"]
    assert plan["lpf_configuration_includes_internal_filter_tuning"]


@pytest.mark.parametrize("mode,expected_drops", [("startup_drop",2),("candidate_drop",3)])
def test_bounded_startup_deltas_and_raw_overlap_preserved(tmp_path, fake_build, monkeypatch, mode, expected_drops):
    out = prepared(tmp_path, fake_build)
    monkeypatch.setenv("FAKE_MODE",mode)
    result = DRIVER["capture"](out,transmit=True,rf_confined_authorized=True)
    report = result["worker_report"]
    assert result["success"] and report["rx_dropped"] == expected_drops
    assert report["qualified_rx_dropped"] == 0
    assert report["qualification_samples"] >= 200000
    assert report["startup_samples"] >= 700000
    rows=[json.loads(line) for line in (out/"rx-chunks.jsonl").read_text().splitlines()]
    assert sum(row["dropped_delta"] for row in rows) == expected_drops
    assert any(row["action"] == "startup_reset" for row in rows)
    startup=np.fromfile(out/"startup.cfile",dtype=np.complex64)
    qualified=np.fromfile(out/"rx.cfile",dtype=np.complex64)
    offset=report["qualification_overlap_startup_sample_offset"]
    count=report["qualification_samples"]
    assert len(startup)==report["startup_samples"]
    assert np.array_equal(startup[offset:offset+count],qualified[:count])
    assert len(qualified)==report["captured_samples"]
    assert report["next_rx_timestamp"]-report["first_rx_timestamp"] == len(qualified)
    assert report["rx_ready_timestamp"]-report["first_rx_timestamp"] == count
    assert report["tx_start_timestamp"]-report["first_rx_timestamp"] >= 500000


@pytest.mark.parametrize("mode",["always_drop","late_drop","late_gap"])
def test_startup_budget_or_post_ready_errors_are_fatal(tmp_path, fake_build, monkeypatch, mode):
    out=prepared(tmp_path,fake_build)
    monkeypatch.setenv("FAKE_MODE",mode)
    with pytest.raises(RuntimeError,match="smoke failed"):
        DRIVER["capture"](out,transmit=True,rf_confined_authorized=True)
    receipt=json.loads((out/"receipt.json").read_text())
    report=receipt["worker_report"]
    assert receipt["cleanup_confirmed"] and not receipt["success"]
    if mode=="always_drop":
        assert not report["tone_submission_attempted"] and report["captured_samples"] == 0
        assert "startup qualification budget" in report["error"]
        assert report["startup_samples"] <= 4000000+16384
    else:
        assert report["tx_attempted"]
        assert "Qualified RX" in report["error"]
    assert report["error"] != "Interrupted or smoke deadline reached"


def test_host_fifo_empty_waits_are_preserved_as_phase_diagnostics(tmp_path,fake_build,monkeypatch):
    out=prepared(tmp_path,fake_build)
    monkeypatch.setenv("FAKE_MODE","host_empty")
    result=DRIVER["capture"](out,transmit=True,rf_confined_authorized=True)
    report=result["worker_report"]
    assert result["success"] and report["tx_samples_sent"]==300000
    rows=[json.loads(line) for line in (out/"tx-status.jsonl").read_text().splitlines()]
    assert rows[0]["phase"] == "before_first_submission"
    assert not rows[0]["tone_submission_attempted"]
    assert rows[-1]["phase"] == "after_capture"
    assert report["tx_underrun"] == sum(row["host_fifo_empty_wait_delta"] for row in rows)
    assert report["tx_host_empty_waits_before_submission"] == 1
    assert report["tx_host_empty_waits_after_submission_attempt"] == len(rows)-1
    assert report["tx_dropped"] == 0 and report["tx_overrun"] == 0


@pytest.mark.parametrize("mode",["tx_drop","tx_overrun"])
def test_actual_tx_drop_or_overrun_still_fails(tmp_path,fake_build,monkeypatch,mode):
    out=prepared(tmp_path,fake_build)
    monkeypatch.setenv("FAKE_MODE",mode)
    with pytest.raises(RuntimeError,match="smoke failed"):
        DRIVER["capture"](out,transmit=True,rf_confined_authorized=True)
    result=json.loads((out/"receipt.json").read_text())
    assert not result["success"] and result["cleanup_confirmed"]
    assert result["worker_report"]["tone_submission_attempted"]
    assert "TX stream overrun/drop" in result["worker_report"]["error"]
    rows=[json.loads(line) for line in (out/"tx-status.jsonl").read_text().splitlines()]
    assert rows[-1]["phase"] == "failure_cleanup_after_stop"
    assert any(row["dropped_delta"] or row["overrun_delta"] for row in rows)


@pytest.mark.parametrize("requested",[None,-1,51,True,30.0,"30"])
def test_requested_tx_gain_is_explicit_and_bounded(tmp_path,fake_build,requested):
    with pytest.raises(ValueError,match="requested TX gain"):
        DRIVER["prepare"](tmp_path/"invalid",build_manifest=fake_build,serial="0x1d423d9108f273",
                          frequency_hz=500e6,native_tx_gain_db=requested,expected_tx_gain_readback_db=30)
    assert not (tmp_path/"invalid").exists()


@pytest.mark.parametrize("gain",[30,50])
@pytest.mark.parametrize("failure",[None,"tx_fail"])
def test_explicit_configuration_gain_and_zero_cleanup(tmp_path,fake_build,monkeypatch,gain,failure):
    out=tmp_path/"gain-attempt"
    DRIVER["prepare"](out,build_manifest=fake_build,serial="0x1d423d9108f273",
                      frequency_hz=500e6,native_tx_gain_db=gain,expected_tx_gain_readback_db=gain)
    journal=tmp_path/"fake-gain-writes.txt"
    monkeypatch.setenv("FAKE_TX_GAIN_LOG",str(journal))
    if failure:
        monkeypatch.setenv("FAKE_MODE",failure)
        with pytest.raises(RuntimeError,match="smoke failed"):
            DRIVER["capture"](out,transmit=True,rf_confined_authorized=True)
        receipt=json.loads((out/"receipt.json").read_text())
    else:
        receipt=DRIVER["capture"](out,transmit=True,rf_confined_authorized=True)
        assert receipt["success"]
    assert receipt["cleanup_confirmed"]
    report=receipt["worker_report"]
    assert report["requested_native_tx_gain_db"] == gain
    assert report["expected_native_tx_gain_readback_db"] == gain
    assert report["native_tx_gain_db"] == gain
    writes=[int(line) for line in journal.read_text().splitlines()]
    assert writes[:3]==[0,0,gain]
    assert len(writes)>3 and all(value==0 for value in writes[3:])
    assert DRIVER["load"](out)["measured_power_dbm"] is None


def test_expected_readback_is_independent_of_requested_gain(tmp_path,fake_build):
    out=tmp_path/"mismatched-contract"
    DRIVER["prepare"](out,build_manifest=fake_build,serial="0x1d423d9108f273",
                      frequency_hz=500e6,native_tx_gain_db=30,expected_tx_gain_readback_db=50)
    with pytest.raises(RuntimeError,match="smoke failed"):
        DRIVER["capture"](out,transmit=True,rf_confined_authorized=True)
    report=json.loads((out/"receipt.json").read_text())["worker_report"]
    assert report["native_tx_gain_db"]==30 and report["expected_native_tx_gain_readback_db"]==50
    assert not report["tone_submission_attempted"] and report["captured_samples"]==0
    assert report["error"]=="Gain readback differs"
