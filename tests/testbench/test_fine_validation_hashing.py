"""Validation source hashes support the oldest declared Python version."""
import hashlib
import runpy
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[2] / "tools"


@pytest.mark.parametrize("name", [
    "audit_fine_native_coordinates.py",
    "prepare_fine_validation_cache.py",
    "prepare_fine_validation_waveforms.py",
    "reduce_fine_validation.py",
    "run_fine_validation_campaign.py",
    "validate_fine_detector.py",
])
@pytest.mark.parametrize("payload", [b"", bytes(range(256)) * 8193])
def test_sha_matches_standard_digest_without_file_digest(tmp_path, monkeypatch, name, payload):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    module = runpy.run_path(str(TOOLS / name))
    assert module["sha"](source) == hashlib.sha256(payload).hexdigest()
