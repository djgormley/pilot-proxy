"""Product to frame table for the narrowband-marker detector.

The statistic is the coarse ``Q = F / mu_0``; the in-band residual estimate is
the product's shelf level (the marker's excess converted to the emitter's
in-band power, :mod:`.shelf`); the frames the characterization may use are
the valid frames the health gate admits.
"""
from __future__ import annotations

from pilot_proxy.config.frequency_plan import Band
from pilot_proxy.detectors.interface import FrameTable


def marker_hz(band: Band, template) -> float:
    """Nominal marker frequency of a band: the lower edge plus the template's offset."""
    marker = template.marker
    if marker is None:
        raise ValueError(f"template kind {template.kind!r} has no marker")
    return band.low_hz + marker.offset_hz


def frames(product, band_id: str) -> FrameTable:
    """The frame table of one open :class:`~pilot_proxy.products.reader.Product`."""
    return FrameTable(band_id=str(band_id), time=product.frame_time, unit=product.frame_unit_index,
                      statistic=product.statistic, residual_db=product.shelf_db, health=product.selected,
                      flagged=product.rejected, unit_name="Q", handle=product)


__all__ = ["frames", "marker_hz"]
