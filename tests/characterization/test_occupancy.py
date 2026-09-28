"""The occupancy wall: the survey-flag rate of the current era against 0.90, with the release's reasons."""
from __future__ import annotations

import math

from pilot_proxy.characterization import occupancy as oc


def test_the_wall_is_a_flag_rate_at_or_above_the_threshold():
    at = oc.occupancy(0.90)
    assert at.occupancy_class == oc.WALL
    assert at.reasons == ("survey flag rate 0.900 >= 0.9: transmitter present in nearly every frame",)
    assert at.thresholds == {"occupancy_wall_flag_rate": 0.90} == {"occupancy_wall_flag_rate": oc.OCCUPANCY_WALL_FLAG_RATE}
    below = oc.occupancy(0.8999)
    assert below.occupancy_class == oc.BELOW_WALL and below.reasons == ("survey flag rate 0.900 < 0.9",)


def test_a_weak_carrier_at_the_wall_is_named():
    weak = oc.occupancy(0.99, era_state="proxy-low", era_level_db=0.21)
    assert weak.reasons[1] == "era state proxy-low (median level 0.21 dB): a weak carrier present in nearly every frame"
    assert len(oc.occupancy(0.99, era_state="proxy-high").reasons) == 1
    assert oc.occupancy(0.99, era_state="ambiguous").reasons[1] == \
        "era state ambiguous: a weak carrier present in nearly every frame"


def test_no_era_frames_is_undetermined_and_the_row_names_the_class():
    assert oc.occupancy(math.nan).occupancy_class == oc.UNDETERMINED
    row = oc.occupancy(0.95).as_row()
    assert row == {"occupancy_class": oc.WALL, "occupancy_reasons": oc.occupancy(0.95).reasons[0]}
    assert oc.occupancy(0.5, wall_flag_rate=0.4).occupancy_class == oc.WALL
