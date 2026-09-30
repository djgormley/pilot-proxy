"""The frame counts the admissible minimum is read at are the dumps' own (stacks axis 0), and they equal the capture
ruling's table (release r5.2, a18_units.NFRAMES; tests/test_a18.py test_frame_counts)."""
from __future__ import annotations

import numpy as np

from capture_data import dump_dir
from pilot_proxy.capture import units

NFRAMES = {"20260916162300": 11, "20260917040230": 33, "20260917090230": 33, "20260917140230": 33}
for _e in ("20260917160208", "20260917160223", "20260917160238", "20260917160308", "20260917160408", "20260917160608",
           "20260917161008", "20260917161408", "20260917161808", "20260917162208"):
    NFRAMES[_e] = 4


def test_frame_counts():
    for ev, n in NFRAMES.items():
        z = np.load(units.first_product(dump_dir(ev)))
        assert units.frames_per_dump(z) == n, (ev, n)
