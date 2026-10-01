"""Inventory naming and metadata helpers."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

from . import invpaths
from .names import validate_identifier
from .survey_provenance import inventory_identity, runtime_provenance

INVENTORY_META_SCHEMA_KEY = "datatrawl_inventory"
INVENTORY_META_SCHEMA_VERSION = 1
_MAX_SLUG_LENGTH = 40
_SLUG_HASH_LENGTH = 8


def inventory_meta_path(inventory_path: str | Path) -> Path:
    """Return the sidecar path for an inventory."""
    return Path(inventory_path).with_suffix(".meta.json")


def _freq_id_slug(freq_ids: object) -> str:
    if not freq_ids or str(freq_ids).strip().lower() in {"all", "*"}:
        return ""
    slug = str(freq_ids).strip().lower().replace(" ", "").replace(",", "-")
    slug = re.sub(r"[^a-z0-9._-]", "", slug)
    if slug and slug[0].isdigit():
        slug = "fid" + slug
    if len(slug) > _MAX_SLUG_LENGTH:
        import hashlib

        digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()
        slug = "fid" + digest[:_SLUG_HASH_LENGTH]
    return slug


def derive_inventory_name(instrument_name: str, freq_ids: object) -> str:
    """Derive a stable inventory name from an instrument and selection."""
    instrument_name = validate_identifier(instrument_name, label="instrument name")
    slug = _freq_id_slug(freq_ids)
    return f"{instrument_name}-{slug}" if slug else instrument_name


def resolve_inventory(
    *,
    inventory: str | Path | None = None,
    name: str | None = None,
    root: str | Path | None = None,
) -> Path | None:
    """Resolve an explicit path or managed inventory name."""
    if inventory is not None and name is not None:
        raise SystemExit("pass either --inventory or --inventory-name, not both")
    if inventory is not None:
        return Path(inventory).expanduser()
    if name is None:
        return None
    try:
        safe_name = validate_identifier(name, label="inventory name")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if root is not None:
        return Path(root).expanduser() / "data" / safe_name / "inventory.jsonl"
    return invpaths.resolve_inventory(safe_name)


def _scopes_in_inventory(inventory_path: Path) -> list[str]:
    scopes: list[str] = []
    with inventory_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{inventory_path}:{line_number}: invalid inventory JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"{inventory_path}:{line_number}: expected a JSON object"
                )
            scope = row.get("scope")
            if scope is not None and not isinstance(scope, str):
                raise ValueError(
                    f"{inventory_path}:{line_number}: inventory field 'scope' "
                    "must be a string"
                )
            if scope and scope not in scopes:
                scopes.append(scope)
    return scopes


def write_inventory_meta(
    inventory_path: str | Path,
    instrument: object,
    *,
    source: str,
    reader: str,
    freq_ids: object = None,
    name: str | None = None,
    scope_request: str | None = None,
) -> Path:
    """Write the compatible provenance sidecar next to an inventory."""
    inventory_path = Path(inventory_path)
    scopes = _scopes_in_inventory(inventory_path)
    if not scopes:
        if scope_request:
            scopes = [part.strip() for part in scope_request.split(",") if part.strip()]
        else:
            scopes = list(getattr(instrument, "scopes", ()) or ())
    payload = {
        INVENTORY_META_SCHEMA_KEY: INVENTORY_META_SCHEMA_VERSION,
        "name": name,
        "telescope": getattr(instrument, "name"),
        "source": source,
        "reader": reader,
        "scope": ",".join(scopes) if scopes else None,
        "scopes": scopes or None,
        "scope_request": scope_request or None,
        "freq_ids": freq_ids,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **runtime_provenance(),
        **inventory_identity(inventory_path),
    }
    # Writing a sidecar for an older inventory must not invent its producer.
    runs = sorted((inventory_path.parent / "survey_runs").glob("*.json"))
    payload["provenance_basis"] = (
        "survey execution records; top-level software identifies this metadata writer"
        if runs else "metadata writer only; original survey execution unavailable")
    payload["survey_runs"] = [str(p.relative_to(inventory_path.parent)) for p in runs]
    meta_path = inventory_meta_path(inventory_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".inventory-meta-", suffix=".tmp", dir=meta_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, meta_path)
    except BaseException:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
        raise
    return meta_path


def read_inventory_meta(inventory_path: str | Path) -> Mapping[str, object] | None:
    """Read and validate a metadata sidecar when one exists."""
    meta_path = inventory_meta_path(inventory_path)
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid inventory metadata {meta_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid inventory metadata {meta_path}: expected an object")
    version = payload.get(INVENTORY_META_SCHEMA_KEY)
    if type(version) is not int or version != INVENTORY_META_SCHEMA_VERSION:
        raise ValueError(
            f"incompatible inventory metadata {meta_path}: "
            f"{INVENTORY_META_SCHEMA_KEY!r} must be exactly "
            f"{INVENTORY_META_SCHEMA_VERSION}, got {version!r}"
        )
    return payload


__all__ = [
    "INVENTORY_META_SCHEMA_KEY",
    "INVENTORY_META_SCHEMA_VERSION",
    "derive_inventory_name",
    "inventory_meta_path",
    "read_inventory_meta",
    "resolve_inventory",
    "write_inventory_meta",
]
