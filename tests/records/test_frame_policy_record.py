"""The capture frame-policy record is byte-frozen and runs through ``pilot-proxy records frame-policy``."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("h5py")
pytest.importorskip("yaml")

RECORD = Path(__file__).resolve().parents[2] / "src" / "pilot_proxy" / "records" / "chime_atsc_2026" / "frame_policy.py"
RECORD_SHA256 = "c49ae37f9541a12e9471b88e6da449fb6248af4ee6244cfca78e66b008a9a513"


def test_the_record_is_byte_frozen():
    assert hashlib.sha256(RECORD.read_bytes()).hexdigest() == RECORD_SHA256


def test_the_record_runs_as_main_through_the_command(capsys):
    from pilot_proxy.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["records", "frame-policy", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--root" in out and "--output" in out and "--channels" in out
