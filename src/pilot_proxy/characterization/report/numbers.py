"""``numbers.json``: every number a table emits, keyed, so the dissertation's
number markers can be verified against a source.

Each export writes one document beside its table (``numbers/<export>.numbers.json``)
with the producing repository and commit, the inputs it read (paths and
SHA-256), and a list of numbers: a stable key, the value, its kind (float,
int, text, range), the renderings the text may use, the printed precision,
and where it came from (table, row, column). The schema token
``rfisher-dissertation-numbers`` is a frozen on-disk value: the documents the
dissertation vendors carry it.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

SCHEMA = {"name": "rfisher-dissertation-numbers", "version": 1}
KINDS = ("float", "int", "text", "range")
STATUSES = ("measured", "bounded", "refused", "pending", "derived")


@dataclass(frozen=True)
class Number:
    key: str                      # e.g. 'ch08.nulls.coarse_core_width_factor.ch29'
    value: float | int | str | None
    kind: str = "float"
    unit: str = ""
    renderings: tuple[str, ...] = ()
    precision: int | None = None  # decimals the text prints, when known
    status: str = "measured"
    source: dict = field(default_factory=dict)     # {"table": ..., "row": {...}, "column": ...}
    tex: dict = field(default_factory=dict)        # {"label": ..., "row_label": ..., "cell": n}

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown number kind {self.kind!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown number status {self.status!r}")


def git_commit(repo: Path | str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class NumbersDocument:
    export: str
    producer: dict
    inputs: list = field(default_factory=list)
    numbers: list = field(default_factory=list)

    @classmethod
    def new(cls, export: str, *, repository: str, commit: str, script: str, generated: str) -> "NumbersDocument":
        return cls(export=export, producer={"repository": repository, "commit": commit, "script": script, "generated": generated})

    def add_input(self, path: Path | str, **extra) -> None:
        p = Path(path)
        self.inputs.append({"path": str(p), "sha256": sha256_of(p) if p.is_file() else None, **extra})

    def add(self, number: Number) -> None:
        if any(n.key == number.key for n in self.numbers):
            raise ValueError(f"duplicate number key {number.key!r}")
        self.numbers.append(number)

    def to_json(self) -> dict:
        return {"schema": dict(SCHEMA), "export": self.export, "producer": dict(self.producer), "inputs": list(self.inputs),
                "numbers": [_number_json(n) for n in self.numbers]}

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1, sort_keys=False) + "\n", encoding="utf-8")
        return path


def _number_json(n: Number) -> dict:
    d = asdict(n)
    v = d["value"]
    if isinstance(v, float) and not math.isfinite(v):
        d["value"] = None
    d["renderings"] = list(n.renderings)
    return d


def load_numbers(paths: Iterable[Path | str]) -> list[Number]:
    out = []
    for path in paths:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("schema", {}).get("name") != SCHEMA["name"]:
            raise ValueError(f"{path}: not a numbers document")
        for item in doc["numbers"]:
            out.append(Number(key=item["key"], value=item.get("value"), kind=item.get("kind", "float"), unit=item.get("unit", ""),
                              renderings=tuple(item.get("renderings", ())), precision=item.get("precision"),
                              status=item.get("status", "measured"), source=item.get("source", {}), tex=item.get("tex", {})))
    return out


# --------------------------------------------------------------- markers
@dataclass(frozen=True)
class Marker:
    file: str


__all__ = ["KINDS", "Number", "NumbersDocument", "SCHEMA", "STATUSES", "git_commit", "load_numbers", "sha256_of"]
