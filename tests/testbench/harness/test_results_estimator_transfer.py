"""The results layer's estimator-transfer figure: reader, axis conversion, render."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy import figure_style as style
from pilot_proxy.testbench.harness.estimator_transfer import Calibration, figure_estimator_transfer, load_release

HAVE_TEX = all(shutil.which(c) for c in ("latex", "dvipng", "kpsewhich"))
CAL = Calibration(pilot_below_data_db=11.918446870168612, bin_enbw_hz=3051.7578125, dtv_bandwidth_hz=6.0e6)


def _write_release(root: Path, *, ota: bool) -> Path:
    (root / "data").mkdir(parents=True)
    xcol = "received_input_data_shelf_snr_db" if ota else "requested_data_shelf_snr_db"
    exp = "control_conditioned_expected_db" if ota else "waveform_conditioned_expected_db"
    xs = np.arange(-30.0, 3.0, 3.0)
    fields = [xcol, "ideal_local_reference_db", exp, "gpu_fixed_db", "gpu_ci95_low_db", "gpu_ci95_high_db"]
    if not ota:
        fields += ["cpu_float_db", "cpu_float_ci95_low_db", "cpu_float_ci95_high_db", "cpu_packed_db"]
    with (root / "data" / "plot_points.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for i, x in enumerate(xs):
            y = x - 10 * np.log10(1 + 10 ** (x / 10))
            row = {xcol: x, "ideal_local_reference_db": y, exp: y + 0.3,
                   "gpu_fixed_db": "" if i == 0 else y + 0.1, "gpu_ci95_low_db": "" if i == 0 else y - 0.4,
                   "gpu_ci95_high_db": y + 0.6}
            if not ota:
                row.update({"cpu_float_db": y + 0.1, "cpu_float_ci95_low_db": y - 0.3,
                            "cpu_float_ci95_high_db": y + 0.5, "cpu_packed_db": "" if i == 0 else y + 0.1})
            w.writerow(row)
    (root / "data" / "analysis.json").write_text(json.dumps({"plot": {"interval": "95% trial bootstrap"}}))
    if ota:
        (root / "run").mkdir()
        (root / "run" / "run_state.json").write_text(json.dumps({"detector_output_calibration": {
            "pilot_below_data_db": CAL.pilot_below_data_db, "bin_enbw_hz": CAL.bin_enbw_hz,
            "dtv_bandwidth_hz": CAL.dtv_bandwidth_hz, "pilot_capture_efficiency": 1.0}}))
    return root


def test_pilot_axis_conversion_round_trips_and_matches_the_recorded_offset():
    # 10 log10(6 MHz / 3051.76 Hz) - 11.918 dB = +21.0 dB of pilot excess per dB of shelf SNR
    assert CAL.offset_db == pytest.approx(21.016, abs=0.01)
    x = np.array([-40.0, 0.0, 12.5])
    assert np.allclose(CAL.pilot_to_shelf(CAL.shelf_to_pilot(x)), x)


def test_ota_release_reads_its_own_calibration(tmp_path):
    rel = load_release(_write_release(tmp_path / "ota", ota=True))
    assert rel.kind == "ota" and rel.calibration == CAL
    assert rel.x_column == "received_input_data_shelf_snr_db"


def test_digital_release_requires_an_explicit_calibration(tmp_path):
    root = _write_release(tmp_path / "digital", ota=False)
    with pytest.raises(ValueError, match="pass calibration="):
        load_release(root)
    rel = load_release(root, calibration=CAL)
    assert rel.kind == "digital" and len(rel.rows) == 11
    assert np.isnan(rel.column("gpu_fixed_db")[0]) and np.isfinite(rel.column("gpu_ci95_high_db")[0])


@pytest.mark.parametrize("ota", [False, True])
def test_render_is_byte_stable(tmp_path, ota):
    style.configure(require_tex=HAVE_TEX)
    rel = load_release(_write_release(tmp_path / "rel", ota=ota), calibration=None if ota else CAL)
    a = figure_estimator_transfer(rel, out=tmp_path / "a.pdf")
    b = figure_estimator_transfer(rel, out=tmp_path / "b.pdf")
    assert a.stat().st_size > 5000
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()
