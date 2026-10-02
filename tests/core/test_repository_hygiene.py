"""Repository hygiene: no workstation path in the code, no version suffix in a file name but a format's, no reference
to a removed path except at the commit that holds it, and the byte-pinned record modules unchanged."""
from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REMOVED = ROOT / "tests" / "data" / "removed_paths.txt"
# records: frozen evidence and provenance keep the paths they were written with
RECORDS = ("docs/evidence/", "data/provenance/", "CHANGELOG.md", "tests/data/removed_paths.txt", "tests/golden/",
           "tests/config/data/", "projects/chime_atsc/eras/")
# version suffixes that name a format, not a revision of a tool
KEPT_VERSIONED = re.compile(r"^(docs/schemas/[\w]+_v\d+\.schema\.json"
                            r"|src/pilot_proxy/(capture|characterization)/schemas/[\w]+_v\d+\.schema\.json"
                            r"|src/pilot_proxy/records/chime_atsc_2026/dissertation_export_v1\.py"
                            r"|tests/data/fxfft256_golden_v1\.npz"
                            r"|tests/products/(data/pilotproxy_v5_fixture\.json|test_pilotproxy_v5\.py|test_pre_v5_refused\.py))$")
# modules whose bytes a record pins: the lag study snapshots lag_moments.py, the capture's frame policy is a record
PINNED = {"src/pilot_proxy/capture/lag_moments.py": "6254a771d304a7766be6a138f6581e4d74359835091ee3a3d5880cab2ebc4010",
          "src/pilot_proxy/records/chime_atsc_2026/frame_policy.py":
              "c49ae37f9541a12e9471b88e6da449fb6248af4ee6244cfca78e66b008a9a513"}


def _tracked():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout")
    files = [f for f in out.decode().split("\0") if f]
    if not files:
        pytest.skip("needs a git checkout")
    return files


def _text(path):
    try:
        return (ROOT / path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
        return None


def test_no_workstation_path_in_the_code():
    found = [f for f in _tracked() if f.startswith(("src/", "projects/", "scripts/", "tools/"))
             and (_text(f) or "").find("/home/djg") >= 0]
    assert not found, found


def test_no_version_suffix_in_a_file_name():
    found = [f for f in _tracked() if re.search(r"_[vV]\d", Path(f).name)
             and not f.startswith(RECORDS) and not KEPT_VERSIONED.match(f)]
    assert not found, found


def _removed():
    rows = [line.split("\t") for line in REMOVED.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    return {path: commit for path, commit in rows}


def test_the_removed_paths_are_gone():
    tracked = set(_tracked())
    assert not [p for p in _removed() if p in tracked]


def test_no_reference_to_a_removed_path():
    removed = _removed()
    pattern = re.compile(r"(?<![\w/.-])(" + "|".join(re.escape(p) for p in sorted(removed, key=len, reverse=True))
                         + r")(?![\w/.-]|@)")
    found = []
    for f in _tracked():
        if f.startswith(RECORDS):
            continue
        text = _text(f)
        if text is None:
            continue
        for m in pattern.finditer(text):
            found.append(f"{f}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(1)}")
    assert not found, found


def _path_parts(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.split("/")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _path_parts(node.left) + _path_parts(node.right)
    if isinstance(node, ast.Call):
        name = ast.unparse(node.func)
        if name in ("Path", "pathlib.Path", "os.path.join", "path.join", "join"):
            return [part for arg in node.args for part in _path_parts(arg)]
        if isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath":
            return _path_parts(node.func.value) + [
                part for arg in node.args for part in _path_parts(arg)
            ]
    return [None]


def _composed_removed_paths(text, removed, tracked):
    tails = {}
    for path in removed:
        parts = path.split("/")
        for count in range(2, len(parts) + 1):
            tails.setdefault("/".join(parts[-count:]), set()).add(path)
    live_tails = {
        "/".join(path.split("/")[-count:])
        for path in tracked for count in range(2, len(path.split("/")) + 1)
    }
    found = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.BinOp, ast.Call)):
            continue
        parts = _path_parts(node)
        start = max((i + 1 for i, part in enumerate(parts) if part is None), default=0)
        suffix = parts[start:]
        for count in range(len(suffix), 1, -1):
            tail = "/".join(suffix[-count:])
            if tail in tails and (tail in removed or tail not in live_tails):
                found.add((node.lineno, tail))
                break
    return sorted(found)


def test_no_removed_path_composed_in_code():
    tracked = _tracked()
    removed = _removed()
    found = []
    for path in tracked:
        if path.endswith(".py") and not path.startswith(RECORDS):
            found.extend(
                f"{path}:{line}: {tail}"
                for line, tail in _composed_removed_paths(_text(path) or "", removed, tracked)
            )
    assert not found, found


@pytest.mark.parametrize("expression", [
    'ROOT / "retired" / "reader.py"',
    'Path("retired") / "reader.py"',
    'os.path.join(ROOT, "retired", "reader.py")',
    'ROOT.joinpath("retired", "reader.py")',
])
def test_composed_path_check_finds_removed_files(expression):
    assert _composed_removed_paths(expression, {"retired/reader.py"}, [])


def test_composed_path_check_preserves_dynamic_parts_and_live_tails():
    removed = {"old/retired/reader.py"}
    assert not _composed_removed_paths('ROOT / "retired" / name / "reader.py"', removed, [])
    assert not _composed_removed_paths(
        'ROOT / "retired" / "reader.py"', removed, ["new/retired/reader.py"]
    )


def test_a_cited_removed_path_exists_at_its_commit():
    removed = _removed()
    cited = set()
    for f in _tracked():
        if f.startswith(RECORDS):
            continue
        for m in re.finditer(r"([\w./-]+)@([0-9a-f]{7,40})\b", _text(f) or ""):
            if m.group(1) in removed:
                cited.add((m.group(1), m.group(2)))
    for path, commit in sorted(cited):
        probe = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}:{path}"], capture_output=True)
        if probe.returncode != 0 and b"Not a valid object name" in probe.stderr:
            pytest.skip("the history is not in this checkout")
        assert probe.returncode == 0, f"{path}@{commit}: not in that commit"


def test_the_byte_pinned_record_modules_are_unchanged():
    for path, sha in PINNED.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, path
