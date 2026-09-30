"""The marker map the numpy-only entry points read equals the capture scripts' maps of record."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("yaml")

from pilot_proxy.capture.marker_map import SCHEMA, main, marker_map  # noqa: E402
from pilot_proxy.config.project import default_project  # noqa: E402

# the frozen frame_analysis pilot_to_inband.py (8ec3cf01) L15-19: NFFT, PILOT and SCI of record
FA_NFFT = 16384
FA_PILOT = {14:844,15:829,16:813,17:798,18:783,19:767,20:752,21:736,22:721,23:706,24:690,25:675,
            26:660,27:644,28:629,29:614,30:598,31:583,32:568,33:552,34:537,35:521,36:506,37:491}
FA_SCI = ["20260916162300","20260917040230","20260917090230","20260917140230"]


def test_the_map_equals_the_record(tmp_path):
    out = tmp_path / "m.json"
    assert main(["--out", str(out)]) == 0
    spec = json.loads(out.read_text())
    assert spec == marker_map(default_project())
    assert spec["schema"] == SCHEMA
    assert {int(k): v for k, v in spec["markers"].items()} == FA_PILOT
    assert list(spec["markers"]) == [str(k) for k in FA_PILOT]
    assert spec["nfft"] == FA_NFFT and spec["science_events"] == FA_SCI and spec["control_bands"] == ["37"]
