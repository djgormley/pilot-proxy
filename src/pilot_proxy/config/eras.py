"""Era list: author-dated eras per band and dated transmitter-off intervals.

Two files:
- the author-dated era list, kept byte for byte as released: ``version``,
  ``criterion``, ``unchanged_channels`` (bands whose eras the predeclared rule
  places), ``channels`` (band label -> ordered ``[first, last, evidence]``
  spans in YYYY-MM) and ``notes``;
- the transmitter-off intervals: band label -> a list of ``{from, through,
  evidence, independently_verified}``. ``from`` null is the start of the
  record and ``through`` null is open-ended, so an intermittent emitter is
  several intervals. ``external_record`` (optional) is the class of the best
  external record found for the date (``confirmed``, ``consistent``,
  ``contradicted``, ``not found``); an interval is independently verified only
  with a confirmed record.

The era states are the vocabulary the project declares (``project.yaml``).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from ._files import ProfileError, check_keys, choice, month, read_json, text

TRANSMITTER_OFF_SCHEMA = "pilot_proxy_transmitter_off_v1"


@dataclass(frozen=True)
class EraSpan:
    first_month: str
    last_month: str
    evidence: str


EXTERNAL_RECORD_CLASSES = ("confirmed", "consistent", "contradicted", "not found")


@dataclass(frozen=True)
class OffInterval:
    from_month: Optional[str]
    through_month: Optional[str]
    evidence: str
    independently_verified: bool
    external_record: str = ""    # the class of the best external record, when one was sought


@dataclass(frozen=True)
class EraList:
    version: str
    criterion: str
    overrides: Mapping[str, tuple[EraSpan, ...]]
    unchanged_bands: tuple[str, ...]
    notes: Mapping[str, str]
    transmitter_off: Mapping[str, tuple[OffInterval, ...]]
    states: tuple[str, ...]
    source_sha256: str
    transmitter_off_sha256: str

    def overrides_spec(self) -> dict[int, list[tuple[str, str, str]]]:
        """The overrides in the release driver's form: int band -> span tuples."""
        return {int(label): [(s.first_month, s.last_month, s.evidence) for s in spans]
                for label, spans in self.overrides.items()}

    def overrides_sha256(self) -> str:
        """Digest of the overrides as the archive run records it
        (``era_overrides_sha256`` in the run ledger)."""
        spec = self.overrides_spec()
        payload = {str(c): [list(e) for e in v] for c, v in sorted(spec.items())}
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def off_through(self) -> dict[int, str]:
        """Bands off from the start of the record, mapped to the last off month."""
        return {int(label): item.through_month
                for label, items in self.transmitter_off.items() for item in items
                if item.from_month is None and item.through_month is not None}

    def off_from(self) -> dict[int, str]:
        """Bands off to the end of the record, mapped to the first off month."""
        return {int(label): item.from_month
                for label, items in self.transmitter_off.items() for item in items
                if item.from_month is not None and item.through_month is None}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spans(label: str, raw: Any, where: str) -> tuple[EraSpan, ...]:
    at = f"{where}: channels[{label!r}]"
    if not isinstance(raw, list) or not raw:
        raise ProfileError(f"{at} must be a non-empty list of spans")
    spans = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, list) or len(entry) != 3:
            raise ProfileError(f"{at}[{index}] must be [first, last, evidence]")
        first = month(entry[0], field="first month", where=at)
        last = month(entry[1], field="last month", where=at)
        if last < first:
            raise ProfileError(f"{at}[{index}]: last month precedes first month")
        spans.append(EraSpan(first, last, text(entry[2], field="evidence", where=at)))
    for before, after in zip(spans, spans[1:]):
        if after.first_month <= before.last_month:
            raise ProfileError(f"{at}: spans must be ordered and must not overlap")
    return tuple(spans)


def _intervals(label: str, raw: Any, where: str) -> tuple[OffInterval, ...]:
    at = f"{where}: bands[{label!r}]"
    if not isinstance(raw, list) or not raw:
        raise ProfileError(f"{at} must be a non-empty list of intervals")
    items = []
    for index, entry in enumerate(raw):
        here = f"{at}[{index}]"
        if not isinstance(entry, Mapping):
            raise ProfileError(f"{here} must be a mapping")
        check_keys(entry, required=("from", "through", "evidence", "independently_verified"),
                   optional=("external_record",), where=here)
        start = None if entry["from"] is None else month(entry["from"], field="from", where=here)
        end = None if entry["through"] is None else month(
            entry["through"], field="through", where=here)
        if start is None and end is None:
            raise ProfileError(f"{here}: from and through cannot both be null")
        if start is not None and end is not None and end < start:
            raise ProfileError(f"{here}: through precedes from")
        verified = entry["independently_verified"]
        if not isinstance(verified, bool):
            raise ProfileError(f"{here}: independently_verified must be true or false")
        external = ""
        if "external_record" in entry:
            external = choice(entry["external_record"], EXTERNAL_RECORD_CLASSES,
                              field="external_record", where=here)
        if verified and external != "confirmed":
            raise ProfileError(f"{here}: independently_verified needs a confirmed external record")
        items.append(OffInterval(start, end, text(entry["evidence"], field="evidence",
                                                  where=here), verified, external))
    return tuple(items)


def load_era_list(author_eras: Path | str, transmitter_off: Path | str,
                  states: tuple[str, ...]) -> EraList:
    author_eras, transmitter_off = Path(author_eras), Path(transmitter_off)
    where = str(author_eras)
    data = read_json(author_eras)
    check_keys(data, required=("version", "criterion", "channels"),
               optional=("unchanged_channels", "notes"), where=where)
    raw = data["channels"]
    if not isinstance(raw, Mapping):
        raise ProfileError(f"{where}: channels must be a mapping of band label to spans")
    overrides = {str(label): _spans(str(label), spans, where) for label, spans in raw.items()}
    unchanged = tuple(str(label) for label in data.get("unchanged_channels", ()))
    overlap = sorted(set(unchanged) & set(overrides))
    if overlap:
        raise ProfileError(f"{where}: bands both unchanged and dated: {overlap}")
    notes = data.get("notes", {})
    if not isinstance(notes, Mapping):
        raise ProfileError(f"{where}: notes must be a mapping")

    where_off = str(transmitter_off)
    off = read_json(transmitter_off)
    check_keys(off, required=("schema", "bands"),
               optional=("source", "months", "independently_verified"), where=where_off)
    if off["schema"] != TRANSMITTER_OFF_SCHEMA:
        raise ProfileError(f"{where_off}: schema must be {TRANSMITTER_OFF_SCHEMA!r}")
    if not isinstance(off["bands"], Mapping):
        raise ProfileError(f"{where_off}: bands must be a mapping")
    intervals = {str(label): _intervals(str(label), items, where_off)
                 for label, items in off["bands"].items()}

    states = tuple(text(s, field="era state", where="project eras.states") for s in states)
    if not states or len(set(states)) != len(states):
        raise ProfileError("project eras.states must be a non-empty list of distinct names")
    return EraList(
        version=text(data["version"], field="version", where=where),
        criterion=text(data["criterion"], field="criterion", where=where),
        overrides=MappingProxyType(overrides),
        unchanged_bands=unchanged,
        notes=MappingProxyType({str(k): str(v) for k, v in notes.items()}),
        transmitter_off=MappingProxyType(intervals),
        states=states,
        source_sha256=_file_sha256(author_eras),
        transmitter_off_sha256=_file_sha256(transmitter_off),
    )


__all__ = [
    "EraList",
    "EraSpan",
    "EXTERNAL_RECORD_CLASSES",
    "OffInterval",
    "TRANSMITTER_OFF_SCHEMA",
    "load_era_list",
]
