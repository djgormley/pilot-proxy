"""The 2026 matched capture: its dumps, their labels and the detector runs made on them.

Four science dumps (the marker-bin dump of 2026-09-16 and three 33-frame
dumps of 2026-09-17) and ten 4-frame cadence dumps (2026-09-17, 0 to 1200 s
after the first). The labels are the ones the capture tables print (the
first science dump's label, ``pilot``, is a frozen token of the tables of
record). Each dump was run through the detector with the archive's nominal
bank (``kernel_<event>_k230``) and, for band 33, with the bank built on its
measured marker (``kernel_<event>_k230_ch33measured``).
"""
from __future__ import annotations

SCIENCE_EVENTS = {"20260916162300": "pilot", "20260917040230": "D1", "20260917090230": "D2", "20260917140230": "D3"}
CADENCE_EVENTS = {"20260917160208": "C0", "20260917160223": "C15", "20260917160238": "C30", "20260917160308": "C60",
                  "20260917160408": "C120", "20260917160608": "C240", "20260917161008": "C480",
                  "20260917161408": "C720", "20260917161808": "C960", "20260917162208": "C1200"}
# bands with no emitter in the capture, whose mean phasor the lag coherence subtracts (cadence_lags.py L29-33)
LAG_REFERENCE_BANDS = (34, 37)
# detector runs per bank: the archive's nominal bank, and the measured-marker bank of band 33
DETECTOR_RUNS = {"nominal": "kernel_{event}_k230", "measured_marker": "kernel_{event}_k230_ch33measured"}
MEASURED_MARKER_BANDS = ("33",)


def dataset_dir(root, event):
    """The reduced products of one dump under a datasets root."""
    import os
    return os.path.join(root, f"pilot_reduce_{event}")


__all__ = ["CADENCE_EVENTS", "DETECTOR_RUNS", "LAG_REFERENCE_BANDS", "MEASURED_MARKER_BANDS", "SCIENCE_EVENTS",
           "dataset_dir"]
