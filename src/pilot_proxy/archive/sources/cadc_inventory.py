"""Strict inventory schema and identity for the CADC Datatrail source.

The source owns archive identity and verified URI fields; readers own only the
relative filenames and product-specific columns. Keeping those rules here gives
survey and enumeration one validation boundary without coupling them to the
survey's network or persistence machinery.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral

from ..names import baseband_filename

RESERVED_FIELDS = frozenset({
    "scope", "event", "name", "size_bytes", "common_path", "obs_date",
    "datasets", "collection_restored",
})
_REQUIRED_FIELDS = frozenset({
    "scope", "event", "name", "size_bytes", "common_path",
})


def join_uri(common_path, name) -> str:
    """Join a verified common path and canonical relative archive name."""
    return f"{str(common_path).rstrip('/')}/{str(name).lstrip('/')}"


def _safe_archive_name(name: str) -> bool:
    """Whether a verified filename is a canonical relative POSIX path."""
    if (not name or name != name.strip() or "\\" in name or any(ord(c) < 32 for c in name)
            or name.startswith("/")):
        return False
    return all(part not in ("", ".", "..") for part in name.split("/"))


def _safe_common_path(value: str) -> bool:
    # Collection-qualified paths and local test/source identifiers share the
    # same component contract. Never normalize away a parent traversal.
    suffix = value.split(":", 1)[-1]
    return _safe_archive_name(suffix)


def _legacy_baseband_row(row):
    """Restore only the documented schema-1 CHIME baseband filename.

    This is a read-time view, never a rewrite of the inventory. The historical
    floating size estimate is preserved and is not promoted to a frame count.
    Partial current rows and unrelated product types still fail validation.
    """
    required = {"scope", "event", "freq_id", "size_bytes", "common_path",
                "obs_date", "datasets", "freq_mhz", "n_frames"}
    if "name" in row or not required <= set(row):
        return row
    scope, event, freq = row["scope"], row["event"], row["freq_id"]
    if (scope not in ("chime.event.baseband.raw", "chime.scheduled.baseband.raw")
            or not isinstance(event, str) or not event.isascii() or not event.isdigit()
            or isinstance(freq, bool) or not isinstance(freq, Integral)
            or not 0 <= freq < 1024):
        return row
    if not isinstance(row["common_path"], str) or not row["common_path"].rstrip("/").endswith("/astro_" + event):
        return row
    estimate = row["n_frames"]
    if (isinstance(estimate, bool) or not isinstance(estimate, (int, float))
            or not math.isfinite(estimate) or estimate < 0):
        return row
    return dict(row, name=baseband_filename(event, freq),
                common_path=row["common_path"].rstrip("/"),
                inventory_compatibility="chime_baseband_schema1_filename",
                n_frames_legacy_estimate=estimate)


def candidate_file(item, reader_name: str) -> tuple[str, dict]:
    """Validate one reader-owned archive candidate before any CADC request."""
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        raise SystemExit(
            f"reader {reader_name!r} survey_files() must yield "
            f"(relative_name, fields) pairs; got {item!r}")
    name, fields = item
    if not isinstance(name, str):
        raise SystemExit(
            f"reader {reader_name!r} yielded non-string archive name "
            f"{name!r}; names must be non-empty paths relative to the event "
            "common path")
    if not _safe_archive_name(name):
        raise SystemExit(
            f"reader {reader_name!r} yielded unsafe archive name {name!r}; "
            "names must be non-empty paths relative to the event common path")
    if fields is None:
        fields = {}
    if not isinstance(fields, Mapping):
        raise SystemExit(
            f"reader {reader_name!r} yielded non-mapping inventory fields for "
            f"{name!r}: {type(fields).__name__}")
    overlap = RESERVED_FIELDS.intersection(fields)
    if overlap:
        raise SystemExit(
            f"reader {reader_name!r} tried to overwrite source-owned inventory "
            f"field(s) {sorted(overlap)} for {name!r}")
    return name, dict(fields)


def annotate_row(shape, row: dict, instrument) -> None:
    """Run reader annotation while protecting source-verified identity."""
    before = {key: deepcopy(row[key]) for key in RESERVED_FIELDS if key in row}
    shape.annotate_row(row, instrument)
    changed = [key for key, value in before.items()
               if key not in row or row[key] != value]
    if changed:
        reader_name = getattr(getattr(shape, "info", None), "name",
                              type(shape).__name__)
        raise SystemExit(
            f"reader {reader_name!r} annotate_row() changed source-owned "
            f"inventory field(s) {sorted(changed)}")


def logical_unit_key(scope, event, name) -> str:
    """Archive-location-independent identity used by resume and quarantine."""
    return "cadc-datatrail:" + json.dumps(
        [str(scope), str(event), str(name)], separators=(",", ":"))


def _inventory_error(path: str, line_number: int, detail: str) -> SystemExit:
    return SystemExit(
        f"invalid inventory row {path}:{line_number}: {detail}. Rebuild this "
        "inventory with `pilot-proxy chime-survey`; legacy or partial rows cannot be "
        "scanned safely.")


def parse_row(text: str, path: str, line_number: int) -> dict:
    """Parse one current-schema row, failing closed with its exact location."""
    try:
        row = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise _inventory_error(path, line_number, f"malformed JSON ({exc})")
    if not isinstance(row, dict):
        raise _inventory_error(
            path, line_number, f"expected a JSON object, got {type(row).__name__}")
    row = _legacy_baseband_row(row)
    missing = sorted(_REQUIRED_FIELDS - set(row))
    if missing:
        raise _inventory_error(path, line_number,
                               f"missing required field(s) {missing}")
    for field in ("scope", "event", "name", "common_path"):
        if (not isinstance(row[field], str) or not row[field].strip()
                or row[field] != row[field].strip()):
            raise _inventory_error(
                path, line_number,
                f"{field!r} must be a non-empty, unpadded string")
    if not _safe_archive_name(row["name"]):
        raise _inventory_error(
            path, line_number,
            "'name' must be a canonical relative path below the common path")
    if not _safe_common_path(row["common_path"]):
        raise _inventory_error(path, line_number,
                               "'common_path' must be canonical below its collection")
    if (isinstance(row["size_bytes"], bool)
            or not isinstance(row["size_bytes"], Integral)
            or row["size_bytes"] <= 0):
        raise _inventory_error(
            path, line_number, "'size_bytes' must be a positive integer")
    if "freq_id" in row and (
            isinstance(row["freq_id"], bool)
            or not isinstance(row["freq_id"], Integral)
            or row["freq_id"] < 0):
        raise _inventory_error(
            path, line_number, "'freq_id' must be a non-negative integer")
    datasets = row.get("datasets")
    if datasets is not None and (
            not isinstance(datasets, list)
            or any(not isinstance(label, str) or not label.strip()
                   for label in datasets)):
        raise _inventory_error(
            path, line_number, "'datasets' must be a list of non-empty strings")
    return row
