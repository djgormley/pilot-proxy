"""Entry points that run where the package's profile cannot be loaded import numpy and the standard library only."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "pilot_proxy"
ALLOWED = {
    SRC / "detectors" / "narrowband_marker" / "marker_to_inband.py": {"numpy"},
}


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            yield ("." if node.level else (node.module or "").split(".")[0]), node.lineno


@pytest.mark.parametrize("path", sorted(ALLOWED), ids=lambda p: str(p.relative_to(SRC)))
def test_imports_are_numpy_and_the_standard_library(path):
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    bad = [(name, line) for name, line in _imports(path) if name not in stdlib and name not in ALLOWED[path]]
    assert not bad, bad
