"""The per-band detector ledger: one record per band carrying every
detector-side quantity the characterization tables are generated from.

Each analysis module returns a frozen result with an ``as_row()`` view; the
ledger merges those views under stable prefixes (``era_``, ``anchor_``,
``containment_``, ``null_``, ``chain_``, ``surface_``, ``occupancy_``), keeps
the run identity (products, their digests, the configuration digests, the
producer) once at the top, and writes one JSON per band plus a flat CSV of all
bands. The science side writes its own ledger sections (tolerance, verdict).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = {"name": "pilot_proxy_characterization_ledger", "version": 1}


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


@dataclass
class ChannelRecord:
    channel: int
    freq_id: int
    product: str
    product_sha256: str
    sections: dict = field(default_factory=dict)      # prefix -> row dict
    notes: list = field(default_factory=list)

    def add(self, prefix: str, row: Mapping[str, Any] | None, *, drop: Sequence[str] = ("channel", "freq_id")) -> None:
        """Attach one module's row under a prefix (None records the section as absent)."""
        if row is None:
            self.sections[prefix] = None
            return
        self.sections[prefix] = {k: _jsonable(v) for k, v in row.items() if k not in drop}

    def flat(self) -> dict:
        out = {"channel": self.channel, "freq_id": self.freq_id, "product": self.product, "product_sha256": self.product_sha256}
        for prefix, row in self.sections.items():
            if row is None:
                out[f"{prefix}_present"] = False
                continue
            for k, v in row.items():
                out[f"{prefix}_{k}"] = v
        out["notes"] = "; ".join(self.notes)
        return out

    def to_json(self) -> dict:
        return {"channel": self.channel, "freq_id": self.freq_id, "product": self.product, "product_sha256": self.product_sha256,
                "sections": self.sections, "notes": list(self.notes)}


@dataclass
class Ledger:
    run: dict                                     # identity: commit, products dir, configs, provisional values, timestamp
    channels: list = field(default_factory=list)

    def add(self, record: ChannelRecord) -> None:
        self.channels.append(record)

    def sorted(self) -> list:
        return sorted(self.channels, key=lambda r: r.channel)

    def write(self, directory: Path | str) -> dict[str, Path]:
        directory = Path(directory)
        (directory / "channels").mkdir(parents=True, exist_ok=True)
        paths = {}
        for rec in self.sorted():
            p = directory / "channels" / f"ch{rec.channel:02d}_fid{rec.freq_id}.json"
            p.write_text(json.dumps(_jsonable(rec.to_json()), indent=1) + "\n", encoding="utf-8")
            paths[f"ch{rec.channel:02d}"] = p
        run_path = directory / "run.json"
        run_path.write_text(json.dumps({"schema": SCHEMA, **_jsonable(self.run),
                                        "channels": [f"channels/ch{r.channel:02d}_fid{r.freq_id}.json" for r in self.sorted()]},
                                       indent=1) + "\n", encoding="utf-8")
        paths["run"] = run_path
        rows = [rec.flat() for rec in self.sorted()]
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        flat = directory / "ledger.csv"
        with flat.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: ("" if row.get(k) is None else repr(row[k]) if isinstance(row[k], float) else row.get(k, ""))
                                 for k in keys})
        paths["ledger"] = flat
        return paths


def load_ledger_rows(path: Path | str) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
