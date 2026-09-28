"""The narrowband-marker detector adapter (today the ATSC 8-VSB pilot).

An emitter that carries a narrowband marker at a known offset from its band
edge is detected on the marker: the coarse statistic ``Q = F / mu_0`` over a
``K``-sample detector window, and the fine order-statistic CFAR over the
marker bins against a bulk of independent bins. Everything marker-specific
lives here: the fine anchor and marker bins (:mod:`.anchors`), the era
spectra and containment (:mod:`.psd`), the reference laws
(:mod:`.reference_models`), the shelf (:mod:`.shelf`), the exact rank scores
(:mod:`.scores`) and the exchangeability test (:mod:`.exchangeability`).
"""
from __future__ import annotations

from pilot_proxy.config.project import Project, default_project
from pilot_proxy.detectors.interface import FrameTable, NullLaw


class NarrowbandMarkerAdapter:
    """:class:`~pilot_proxy.detectors.interface.DetectorAdapter` for a narrowband marker."""

    kind = "narrowband_marker"

    def __init__(self, project: Project | None = None):
        self.project = project or default_project()
        if self.project.detector_adapter != self.kind:
            raise ValueError(f"project detector adapter is {self.project.detector_adapter!r}, not {self.kind!r}")

    def frames(self, product, band_id: str) -> FrameTable:
        from .frames import frames
        return frames(product, band_id)

    def null_law(self) -> NullLaw:
        from .reference_models import NULL
        dof = self.project.detector_config.coarse_null_dof(self.project.instrument)
        return NullLaw(name="F(2P, 4P)", dof=tuple(int(v) for v in dof),
                       description=("the coarse statistic Q under ideal independent Gaussian noise with equal, "
                                    "signal-free references; P = inputs x detector window"), law=NULL)

    def residual_estimator(self) -> str:
        return ("shelf: the marker's normalized excess converted to the emitter's in-band power "
                "(marker_to_band_db, capture efficiency, band / bin noise bandwidth); finite only where the "
                "excess is positive")

    def candidate_families(self, frames: FrameTable) -> tuple[str, ...]:
        """Fine families ``rho=<rank>`` for every rank the bulk supports, and the coarse family ``Q``."""
        bundle = frames.rank_scores
        ranks = () if bundle is None else tuple(int(r) for r in bundle.rho)
        return tuple(f"rho={r}" for r in ranks) + ("Q",)

    def exchangeability(self, frames: FrameTable, *, rank: int, quiet, marker_bins, bulk_mask):
        from pilot_proxy.products.reader import fine_power_ratio

        from .exchangeability import exchangeability_rate
        product = frames.handle
        fine_t = fine_power_ratio(product.fine_terms_all())
        return exchangeability_rate(fine_t[quiet], bulk_mask, int(rank), marker_bins)


__all__ = ["NarrowbandMarkerAdapter"]
