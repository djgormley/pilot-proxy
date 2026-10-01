# coding=utf-8
"""Inventory location policy: one absolute, CWD-independent home.

Inventories live under a single root so that surveys and scans agree on
where things are no matter which directory a command runs from:

    $PILOT_PROXY_INVENTORY_ROOT    env override
    ~/datatrawl-inventories        default

Reads and writes resolve by name only through the canonical root. Historical
``data/<name>`` fallbacks were removed at the current cleanup boundary because they
made command behavior depend on the current working directory. An explicit
``--root``/``--inventory`` still wins
everywhere -- this module only supplies the defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

from .names import validate_identifier

ENV = "PILOT_PROXY_INVENTORY_ROOT"
DEFAULT_ROOT = "~/datatrawl-inventories"


def inventory_root() -> Path:
    """The canonical inventory root (env override, else the default)."""
    root = Path(os.environ.get(ENV, DEFAULT_ROOT)).expanduser()
    if not root.is_absolute():
        raise ValueError(
            f"{ENV} must be an absolute path, got {str(root)!r}")
    return root


def inventory_dir_for_write(name: str) -> Path:
    """Where a new inventory named ``name`` is written."""
    return inventory_root() / validate_identifier(name, label="inventory name")


def resolve_inventory(name: str) -> Path:
    """Resolve ``<dir>/inventory.jsonl`` for reading, by canonical name."""
    return inventory_dir_for_write(name) / "inventory.jsonl"
