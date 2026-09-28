"""The general layers (characterization, products, the detector interface) name no emitter, instrument or band."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "pilot_proxy"
GENERAL = sorted([*SRC.joinpath("characterization").rglob("*.py"), *SRC.joinpath("products").rglob("*.py"),
                  SRC / "detectors" / "interface.py"])
DOMAIN = re.compile(r"pilot|atsc|dtv|chime|bao|radiofisher", re.IGNORECASE)
# Names fixed on disk by the product contract (v5 product keys), a frozen table column, or the archive's
# directory layout: they are read and written as they are, whatever the layer.
FROZEN_NAMES = {
    "pilot_hz": "v5 product key",
    "chime_frequency_hz": "v5 product key",
    "pilot_suppression_db": "frozen column of held_out_spectra.csv",
    "per_pilot": "the archive's _per_pilot product directory",
}
BAND_COUNT = 23


def _defined_names(tree: ast.Module):
    """Function, class and module-level names, parameters and class attributes: the layer's vocabulary."""
    def visit(node, top):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node.lineno
            args = node.args
            for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
                if a is not None:
                    yield a.arg, node.lineno
        elif isinstance(node, ast.ClassDef):
            yield node.name, node.lineno
            for child in node.body:
                yield from visit(child, True)
        elif top and isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        yield name.id, node.lineno
    for node in tree.body:
        yield from visit(node, True)


@pytest.mark.parametrize("path", GENERAL, ids=lambda p: str(p.relative_to(SRC)))
def test_no_domain_word_in_the_layers_vocabulary(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = sorted({f"{name} (line {line})" for name, line in _defined_names(tree)
                    if DOMAIN.search(name) and name not in FROZEN_NAMES})
    assert not found, found


def _code_constants(tree):
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and id(n) not in docstrings]


@pytest.mark.parametrize("path", GENERAL + sorted(SRC.joinpath("detectors").rglob("*.py")),
                         ids=lambda p: str(p.relative_to(SRC)))
def test_no_band_range_or_band_count(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "range":
            values = [a.value for a in node.args if isinstance(a, ast.Constant)]
            assert values[:2] != [14, 37], f"line {node.lineno}: the screened band list comes from the plan"
    counts = [n.lineno for n in _code_constants(tree) if type(n.value) is int and n.value == BAND_COUNT]
    assert not counts, f"a band-count literal at lines {counts}"


def test_no_home_directory_under_src():
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if "/home/djg" in p.read_text(encoding="utf-8")]
    assert not offenders, offenders
