"""Safe, eagerly materialised NumPy archive loading.

Survey products are external inputs.  NumPy's pickle-backed object arrays can
execute arbitrary code while loading, so the package never enables them.
Returning ordinary arrays also ensures no ``NpzFile`` handle escapes its
context manager.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np


def load_npz(path: str | Path, *, required: Iterable[str] = ()) \
        -> dict[str, np.ndarray]:
    """Load a ``.npz`` archive without pickle and copy all arrays into memory.

    Parameters
    ----------
    path
        Archive path.
    required
        Field names that must be present. Missing fields are reported together
        before callers begin a scientific calculation.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(required) - set(archive.files))
            if missing:
                raise ValueError(
                    f"{Path(path).name} is missing required NPZ field(s): "
                    + ", ".join(missing))
            return {name: np.array(archive[name], copy=True)
                    for name in archive.files}
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                f"{Path(path).name} contains pickle-backed object arrays; "
                "convert metadata to strings or numeric arrays before using "
                "this untrusted survey product") from exc
        raise
