"""Review M5: the characterization's direct imports of the narrowband-marker adapter are pinned; the list can
only shrink (the declared interface, pilot_proxy.detectors.interface, is where they go). No general layer
imports a record package by name."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "pilot_proxy"
ADAPTER = "pilot_proxy.detectors.narrowband_marker"
RECORDS = "pilot_proxy.records"

# module (relative to src/pilot_proxy) -> the adapter modules it imports directly, as of M3
ALLOWED = {
    "characterization/flaggers.py": {"pilot_proxy.detectors.narrowband_marker.scores"},
    "characterization/histograms.py": {"pilot_proxy.detectors.narrowband_marker.reference_models"},
    "characterization/masked_spectra.py": {"pilot_proxy.detectors.narrowband_marker",
                                           "pilot_proxy.detectors.narrowband_marker.scores"},
    "characterization/nulls.py": {"pilot_proxy.detectors.narrowband_marker.exchangeability"},
    "characterization/residual_chain.py": {"pilot_proxy.detectors.narrowband_marker.shelf"},
    "characterization/run.py": {"pilot_proxy.detectors.narrowband_marker",
                                "pilot_proxy.detectors.narrowband_marker.scores"},
    "characterization/stability.py": {"pilot_proxy.detectors.narrowband_marker.scores"},
    "characterization/surface.py": {"pilot_proxy.detectors.narrowband_marker.scores"},
}


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
    return out


def _scan(prefix: str) -> dict[str, set[str]]:
    found = {}
    for package in ("characterization", "products"):
        for path in sorted((ROOT / package).rglob("*.py")):
            hits = {m for m in _imports(path) if m == prefix or m.startswith(prefix + ".")}
            if hits:
                found[str(path.relative_to(ROOT))] = hits
    return found


def test_the_direct_adapter_imports_are_the_pinned_ones():
    found = _scan(ADAPTER)
    grown = {m: sorted(h - ALLOWED.get(m, set())) for m, h in found.items() if h - ALLOWED.get(m, set())}
    assert not grown, f"new direct adapter imports (go through pilot_proxy.detectors.interface): {grown}"
    shrunk = {m: sorted(ALLOWED[m] - found.get(m, set())) for m in ALLOWED if ALLOWED[m] - found.get(m, set())}
    assert not shrunk, f"imports removed: shrink ALLOWED to match {shrunk}"


def test_no_general_layer_imports_a_record_package():
    assert _scan(RECORDS) == {}
