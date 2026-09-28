"""The ROC populations of record (``docs/evidence/youden_j_2026-09-07``).

The null is channel 35 (freq_id 521) in its dated transmitter-off era, through
2021-10; the signals are the 2025-onward frames of channels 36, 35 and 34.
The off era is dated from this archive's own pilot series (the external record
is consistent, not confirmed), so the labels are weak truth.
"""
from __future__ import annotations

from pilot_proxy.characterization.roc import Populations

POPULATIONS = Populations(null_band=35, null_off_through="2021-10", signal={36: 506, 35: 521, 34: 537},
                          signal_from="2025-01")

__all__ = ["POPULATIONS"]
