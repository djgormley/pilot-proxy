"""Real HDF5 -> CUDA -> archived products, including an OS-level interrupt."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest

from pilot_proxy.chime.baseband_format import make_synth_file
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.fxfft import fine_power_fx
from gpu import cuda_available
from pilot_proxy.kernel import FStatKernel
from pilot_proxy.paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH

pytestmark = pytest.mark.cuda
ROOT = Path(__file__).resolve().parents[2]
CHILD = r"""
import json, sys, time
from pathlib import Path
from pilot_proxy.archive.scan import run_chime_scan
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader

input_dir, output_dir, library, pause = sys.argv[1:]
if pause == "pause":
    original = ChimeBasebandPackedReader.iter_arrays
    def interrupted_reader(self, path, ctx):
        for index, block in enumerate(original(self, path, ctx)):
            yield block
            if "baseband_1001_" in path and index == 0:
                Path(output_dir, "ready").touch()
                while True:
                    time.sleep(0.1)
    ChimeBasebandPackedReader.iter_arrays = interrupted_reader
try:
    run_chime_scan(
        input_dir=input_dir, output_dir=output_dir, source="local", select="844",
        download_workers=2, max_staged_files=3, checkpoint_every=1,
        staging_dir=str(Path(output_dir, "staging")),
        analyzer_options={"lib_path": library, "fine_products": "on"}, verbose=False,
    )
except KeyboardInterrupt:
    raise SystemExit(130)
"""


def read_product(output):
    with np.load(output / "_per_pilot/844.npz", allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def test_archive_gpu_scan_interrupt_resume_and_cpu_reference(tmp_path):
    pytest.importorskip("cupy")
    ok, reason = cuda_available()
    if not ok:
        pytest.skip(reason)
    if not DEFAULT_LIB_PATH.is_file():
        pytest.skip("build the default K=128 CUDA library first")
    kernel = FStatKernel(DEFAULT_LIB_PATH)
    assert kernel.specs.detector_window_samples == 128
    source = tmp_path / "input"
    source.mkdir()
    for event in range(1000, 1003):
        make_synth_file(
            source / f"baseband_{event}_844.h5",
            n_time=32768,
            n_feeds=4,
            f_center_mhz=470.3125,
            f_tone_bb=3059.0,
            seed=event,
        )
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))

    def command(output, mode="run"):
        return [
            sys.executable,
            "-c",
            CHILD,
            str(source),
            str(output),
            str(DEFAULT_LIB_PATH),
            mode,
        ]

    clean, resumed = tmp_path / "clean", tmp_path / "resumed"
    result = subprocess.run(
        command(clean), env=env, capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with (tmp_path / "interrupted.log").open("w") as log:
        child = subprocess.Popen(
            command(resumed, "pause"), env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 30
            while (
                not (resumed / "ready").exists()
                and child.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert (resumed / "ready").exists(), (
                tmp_path / "interrupted.log"
            ).read_text()
            os.kill(child.pid, signal.SIGINT)
            assert child.wait(timeout=35) == 130
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    partial = read_product(resumed)
    assert partial["frame_index"].size == 2
    assert partial["unit_keys"].size == 1
    assert not list((resumed / "staging").rglob("*.h5"))
    for _ in range(2):
        result = subprocess.run(
            command(resumed), env=env, capture_output=True, text=True, timeout=45
        )
        assert result.returncode == 0, result.stdout + result.stderr
    reference, actual = read_product(clean), read_product(resumed)
    for field in (
        "p_target_u64",
        "p_ref_lower_u64",
        "p_ref_upper_u64",
        "p_ref_sum_u64",
        "fine_power_u64",
        "reject_mask",
        "valid",
        "frame_index",
        "unit_keys",
        "unit_order",
        "frame_unit_index",
        "frame_in_unit",
        "unit_input_map_sha256",
        "baseband_power_linear",
        "integrated_spectrum_before_mask",
        "integrated_spectrum_after_mask",
    ):
        np.testing.assert_array_equal(actual[field], reference[field], err_msg=field)
    assert actual["frame_index"].size == 6
    assert len(set(actual["unit_keys"])) == 3
    assert json.loads((resumed / "scan_scope.json").read_text())["complete"]

    # Independently unpack the stored offset-binary bytes and reverse each window.
    weights, valid = DetectorWeightBank(
        explicit_path=DEFAULT_WEIGHTS_PATH
    ).get_weights_for_physical_channel(14)
    assert valid
    w = weights.astype(np.int64) & 255
    wr, wi = ((w >> 4) ^ 8) - 8, ((w & 15) ^ 8) - 8
    expected_power, expected_fine = [], []
    for path in sorted(source.glob("*.h5")):
        with h5py.File(path) as handle:
            data = handle["baseband"][:]
        for start in (0, 16384):
            x = data[start : start + 16384].T.reshape(-1, 128)[:, ::-1].astype(np.int64)
            xr, xi = (x >> 4) - 8, (x & 15) - 8
            rows = np.stack(
                ((xr @ wr.T + xi @ wi.T).T, (xi @ wr.T - xr @ wi.T).T), axis=-1
            )
            expected_power.append((rows**2).sum(axis=(1, 2)))
            expected_fine.append(fine_power_fx(rows, num_streams=4))
    for term, name in enumerate(("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")):
        np.testing.assert_array_equal(
            actual[name], np.asarray(expected_power)[:, term : term + 1]
        )
    np.testing.assert_array_equal(actual["fine_power_u64"], expected_fine)
