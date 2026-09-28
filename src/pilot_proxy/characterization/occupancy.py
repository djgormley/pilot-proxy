"""The occupancy wall: a band whose emitter is present in nearly every frame.

A band at the wall has no cheap mask: its survey flag (the coarse rule at
``eta = 1``) fires on nearly every frame of the current era. The class is a
detector fact about the era, reported for every band in ``oc_summary``; the
screen that decides a band (the science side's verdict) reads it. The wall's other
clause, a *selected* point masking nearly everything, needs a selected point
and belongs to the science side.

The threshold is a parameter, recorded on every result: a survey-flag rate at
or above 0.90 marks the wall. The reason strings are the release's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

OCCUPANCY_WALL_FLAG_RATE = 0.90
WALL = "occupancy wall"
BELOW_WALL = "below the occupancy wall"
UNDETERMINED = "undetermined: no era frames"


@dataclass(frozen=True)
class Occupancy:
    occupancy_class: str
    reasons: tuple[str, ...]
    thresholds: dict

    def as_row(self) -> dict:
        return {"occupancy_class": self.occupancy_class, "occupancy_reasons": "; ".join(self.reasons)}


def occupancy(survey_flag_rate: float, *, era_state: str = "", era_level_db: float = math.nan,
              wall_flag_rate: float = OCCUPANCY_WALL_FLAG_RATE) -> Occupancy:
    """The occupancy class of one band's current era from its survey-flag rate."""
    thresholds = {"occupancy_wall_flag_rate": wall_flag_rate}
    if not math.isfinite(survey_flag_rate):
        return Occupancy(UNDETERMINED, (), thresholds)
    reasons: list[str] = []
    if survey_flag_rate >= wall_flag_rate:
        reasons.append(f"survey flag rate {survey_flag_rate:.3f} >= {wall_flag_rate}: transmitter present in nearly every frame")
        if era_state and era_state != "proxy-high":
            level = f" (median level {era_level_db:.2f} dB)" if math.isfinite(era_level_db) else ""
            reasons.append(f"era state {era_state}{level}: a weak carrier present in nearly every frame")
        return Occupancy(WALL, tuple(reasons), thresholds)
    return Occupancy(BELOW_WALL, (f"survey flag rate {survey_flag_rate:.3f} < {wall_flag_rate}",), thresholds)


__all__ = ["BELOW_WALL", "OCCUPANCY_WALL_FLAG_RATE", "Occupancy", "UNDETERMINED", "WALL", "occupancy"]
