"""The report modules take their external inputs from the environment and have
no default location. These tests ran against fixed locations on the author's
machine; the fixture restores those locations when the variables are unset, so
each test sees what it saw before. Tests that need other inputs set the
variables themselves, and the real-data tests skip when the inputs are absent."""
import os

import pytest

from pilot_proxy.characterization.report import accounting, detection

REAL_INPUTS = {
    accounting.FREEZE_ENV: "/home/djg/rail/archive_inputs/chime-pilots-v5",
    detection.SWEEP_ENV: "/home/djg/rail/products/estimator_transfer_2026-08-25",
    detection.OTA_ENV: "/home/djg/rail/products/ota_transfer_2026-08-24_ch35",
}


@pytest.fixture(autouse=True)
def _real_inputs(monkeypatch):
    for name, path in REAL_INPUTS.items():
        if not os.environ.get(name):
            monkeypatch.setenv(name, path)
