"""Validation for names that become managed filesystem components.

Explicit path options (``--out``, ``--inventory``, ``--root``) remain paths.
Names, plugin identifiers, and instrument identifiers do not: accepting path
separators or ``..`` in those values lets an apparently managed output escape
its configured root and makes two spellings refer to the same object.
"""
from __future__ import annotations

import re

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def baseband_filename(event, freq_id) -> str:
    """Shared archive naming rule, including explicitly identified legacy rows."""
    return f"baseband_{event}_{freq_id}.h5"


def validate_identifier(value: object, *, label: str = "name") -> str:
    """Return an exact filesystem-safe identifier or raise ``ValueError``."""
    text = value.strip() if isinstance(value, str) else ""
    if (value != text or not text or text in {".", ".."}
            or not _IDENTIFIER_RE.fullmatch(text)
            or ".." in text):
        raise ValueError(
            f"invalid {label} {value!r}: use letters, digits, '_', '-', or '.', "
            "starting with a letter or digit; path components are not allowed")
    return text
