"""Nothing in this repository imports the science package (the dependency points the other way)."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("src", "tests", "tools", "scripts")
SCIENCE = ("rfisher", "rfisher_results", "baonoise")


def _imports(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, node.lineno
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) == "import_module":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                yield node.args[0].value, node.lineno


def test_no_science_package_is_imported():
    offenders = []
    for scope in SCOPES:
        for path in sorted((ROOT / scope).rglob("*.py")) if (ROOT / scope).is_dir() else ():
            for module, line in _imports(path):
                if module.split(".")[0] in SCIENCE:
                    offenders.append(f"{path.relative_to(ROOT)}:{line}: {module}")
    assert not offenders, offenders
