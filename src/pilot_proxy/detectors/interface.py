"""The interface between a detector and the characterization.

A detector adapter turns one product into a :class:`FrameTable`: per frame,
the time, the band, the detector's statistic, the in-band residual estimate
and whether the frame may be used. It also declares the statistic's ideal
null law (:class:`NullLaw`), how it estimates the in-band residual, and the
threshold families its candidates come in. The characterization reads only
the frame table and these declarations; anything specific to how the
detector works stays in the adapter. An emitter with no marker (a wideband
OFDM or a pulsed radar) is then a new adapter, not a new characterization.

The adapter of this repository is
:class:`pilot_proxy.detectors.narrowband_marker.NarrowbandMarkerAdapter`.
The frame table keeps the adapter's opaque ``handle`` (the product) so that
steps whose arithmetic is release-authoritative (the exact integer rank
scores, the per-frame spectra) can still reach the product through the
adapter's own modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class NullLaw:
    """The ideal law of the detector statistic when no emitter is present."""

    name: str                      # e.g. "F(2P, 4P)"
    dof: tuple[int, ...]           # its degrees of freedom
    description: str
    law: Any = field(default=None, repr=False, compare=False)   # a frozen scipy distribution


@dataclass(frozen=True)
class FrameTable:
    """Per-frame detector output of one product: the characterization's only per-frame input.

    Arrays are aligned on the product's frame axis. ``time`` is UTC seconds of
    each frame's first sample (NaN where the acquisition recorded no sample
    interval); ``unit`` the acquisition index; ``statistic`` the detector
    statistic (``unit`` names it; NaN where invalid); ``residual_db`` the
    in-band residual estimate in dB relative to system noise in the band (NaN
    where unresolved); ``health`` the frames the characterization may use
    (valid and passing the health gate); ``flagged`` the stored survey flag.
    """

    band_id: str
    time: np.ndarray
    unit: np.ndarray
    statistic: np.ndarray
    residual_db: np.ndarray
    health: np.ndarray
    flagged: np.ndarray
    unit_name: str = "Q"
    rank_scores: Any = None        # optional per-frame rank scores (the adapter's own form)
    handle: Any = field(default=None, repr=False, compare=False)   # the adapter's opaque handle

    def __post_init__(self):
        n = np.asarray(self.statistic).shape
        for name in ("time", "unit", "residual_db", "health", "flagged"):
            if np.asarray(getattr(self, name)).shape != n:
                raise ValueError(f"frame table column {name!r} does not match the frame axis")

    @property
    def frames(self) -> int:
        return int(np.asarray(self.statistic).size)


@runtime_checkable
class DetectorAdapter(Protocol):
    """What a detector adapter supplies to the characterization."""

    kind: str

    def frames(self, product, band_id: str) -> FrameTable: ...

    def null_law(self) -> NullLaw: ...

    def residual_estimator(self) -> str: ...

    def candidate_families(self, frames: FrameTable) -> tuple[str, ...]: ...

    def exchangeability(self, frames: FrameTable, *, rank: int, quiet, marker_bins, bulk_mask): ...


__all__ = ["DetectorAdapter", "FrameTable", "NullLaw"]
