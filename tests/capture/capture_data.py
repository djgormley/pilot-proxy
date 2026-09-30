"""Where the capture data tests find the reduced dumps of the 2026 capture.

``PILOT_REDUCE_ROOT`` names the root holding ``pilot_reduce_<event>/``
(default ``/home/djg/rail/datasets``, the workspace of record). The data are
CHIME collaboration products and are not published with the repository, so
the tests that read them skip where the root is absent (continuous
integration runs the synthetic tests only).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("PILOT_REDUCE_ROOT", "/home/djg/rail/datasets"))


def dump_dir(event: str) -> Path:
    d = ROOT / f"pilot_reduce_{event}"
    if not d.is_dir():
        pytest.skip(f"capture data not present: {d} (set PILOT_REDUCE_ROOT)")
    return d
